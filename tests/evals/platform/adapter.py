"""
The one experiment loop, and the backend protocol it drives.

A platform adds what the harness deliberately lacks: run history, a diff between
two experiments, and per-call traces you can click into. What it must not do is
own the loop. ``run_case`` and ``score_case`` stay the single definition of what
was measured, and a backend receives finished results.

The judge scores a *continuation*: the session is scored exactly as the recall
ratchet scores it (questions left unanswered), then carried to a compiled prompt
with the scripted respondent so the judge has something to read. The scoring
session is never changed by that.

Import as:

import tests.evals.platform.adapter as adapter
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import subprocess
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal, Protocol

import pydantic

import crux.domain.output as coutput
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.evidence as ievid
import crux.infra.settings as isettn
import crux.ports.llm as pllm
import tests.evals.harness as harness
import tests.evals.judge as judge
import tests.evals.runner as runner
import tests.evals.scorers as scorers

CassetteMode = Literal["record", "replay"]


class RunMetadata(pydantic.BaseModel):
    """
    What a person needs to compare two experiments honestly.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    git_sha: str
    crux_version: str
    model: str
    judge_model: str | None
    cassette_mode: CassetteMode
    corpus_size: int
    started_at: dt.datetime


class CaseResult(pydantic.BaseModel):
    """
    Everything one case produced, ready for a backend to log.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case: harness.EvalCase
    score: harness.CaseScore | None
    """``None`` when the case failed before it could be scored."""
    scores: dict[str, float]
    metrics: dict[str, int]
    rendered: str
    judge_rationale: str = ""
    feedback: str = ""
    """Why it scored what it did, one finding per line. See ``scorers.feedback``."""
    calls: tuple[ievid.CallRecord, ...] = ()
    elapsed_seconds: float = 0.0
    error: str = ""


class EvalBackend(Protocol):
    """
    Where results go. Four calls, in this order, once per experiment.
    """

    name: str

    def upsert_dataset(self, cases: Sequence[harness.EvalCase]) -> dict[str, str]:
        """
        :param cases: The corpus.
        :return: Case id to the platform's dataset item id.
        """
        ...

    def start_experiment(self, meta: RunMetadata, item_ids: dict[str, str]) -> None:
        """
        :param meta: What this run is.
        :param item_ids: What ``upsert_dataset`` returned.
        """
        ...

    def log_result(self, result: CaseResult) -> None:
        """
        :param result: One finished case, with its scores and its calls.
        """
        ...

    def finish(self) -> str:
        """
        :return: A URL or a summary line for the person who ran it.
        """
        ...


class CallLog:
    """
    Collects call records the way ``RunRecorder`` does, without writing a directory.
    """

    def __init__(self) -> None:
        self._calls: list[ievid.CallRecord] = []

    def record_call(
        self,
        *,
        operation: str,
        model: str,
        messages: list[dict[str, Any]],
        forced_tool: str | None,
        response: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
        error: str = "",
        attempt: int = 0,
    ) -> None:
        """
        Keep one exchange. Same signature as ``RunRecorder.record_call``.
        """
        self._calls.append(
            ievid.CallRecord(
                index=len(self._calls) + 1,
                operation=operation,
                model=model,
                messages=tuple(messages),
                forced_tool=forced_tool,
                response=response or {},
                elapsed_seconds=elapsed_seconds,
                error=error,
                attempt=attempt,
            )
        )

    def drain(self) -> tuple[ievid.CallRecord, ...]:
        """
        :return: Everything recorded so far, clearing the log.
        """
        calls = tuple(self._calls)
        self._calls.clear()
        return calls


class TracingLlm:
    """
    Records every completion that passes through it, then re-raises or returns.

    Sits *above* the cassette so replayed calls are traced too. Per-attempt
    fallback detail is unavailable here, and that is fine: measurement runs
    refuse a fallback chain.
    """

    def __init__(self, inner: pllm.LlmClient, log: CallLog, *, default_model: str) -> None:
        """
        :param inner: The client that actually completes.
        :param log: Where records go.
        :param default_model: What to record when the caller named no model.
        """
        self._inner = inner
        self._log = log
        self._default_model = default_model

    async def complete(
        self,
        messages: Sequence[pllm.Message],
        *,
        tools: Sequence[pllm.ToolSchema] = (),
        force_tool: str | None = None,
        model: str | None = None,
        require_tool: bool = True,
    ) -> pllm.LlmTurn:
        """
        Complete through the inner client and record what happened.

        :return: The inner client's turn.
        """
        started = time.monotonic()
        sent = [m.model_dump(mode="json") for m in messages]
        try:
            turn = await self._inner.complete(
                messages, tools=tools, force_tool=force_tool, model=model, require_tool=require_tool
            )
        except Exception as exc:
            self._log.record_call(
                operation=force_tool or "chat",
                model=model or self._default_model,
                messages=sent,
                forced_tool=force_tool,
                elapsed_seconds=time.monotonic() - started,
                error=str(exc),
            )
            raise
        self._log.record_call(
            operation=force_tool or "chat",
            model=model or self._default_model,
            messages=sent,
            forced_tool=force_tool,
            response=ievid.summarise_turn(turn),
            elapsed_seconds=time.monotonic() - started,
        )
        return turn


def collect_metadata(
    settings: isettn.CruxSettings,
    *,
    record: bool,
    judge_model: str | None,
    corpus_size: int,
) -> RunMetadata:
    """
    Gather what identifies this run.

    :param settings: Model configuration.
    :param record: Whether the cassette was allowed to call the real model.
    :param judge_model: Which model judged, when one was named.
    :param corpus_size: How many cases ran.
    :return: The metadata.
    """
    return RunMetadata(
        git_sha=_git_sha(),
        crux_version=_crux_version(),
        model=settings.model,
        judge_model=judge_model,
        cassette_mode="record" if record else "replay",
        corpus_size=corpus_size,
        started_at=dt.datetime.now(tz=dt.UTC),
    )


RunCase = Callable[[harness.EvalCase, pllm.LlmClient], Awaitable[csessn.Session]]
FinishCase = Callable[
    [harness.EvalCase, pllm.LlmClient, csessn.Session], Awaitable[coutput.CompiledPrompt | None]
]


async def run_experiment(
    cases: Sequence[harness.EvalCase],
    backend: EvalBackend,
    client: pllm.LlmClient,
    judge_client: pllm.LlmClient | None,
    meta: RunMetadata,
    *,
    run_case: RunCase | None = None,
    finish_case: FinishCase | None = None,
    judge_model: str | None = None,
) -> tuple[CaseResult, ...]:
    """
    Run the corpus and hand every result to the backend.

    :param cases: What to run.
    :param backend: Where results go.
    :param client: Where crux's completions come from.
    :param judge_client: Where the judge's completions come from. ``None``
        skips the judge entirely.
    :param meta: What this run is.
    :param run_case: How to run one case. Defaults to the shared runner.
    :param finish_case: How to carry a scored session to a compiled prompt for
        the judge. Defaults to the scripted respondent.
    :param judge_model: Which model judges.
    :return: Every result, in corpus order.
    """
    run = run_case or _default_run_case
    finish = finish_case or _default_finish_case
    log = CallLog()
    traced = TracingLlm(client, log, default_model=meta.model)

    item_ids = backend.upsert_dataset(cases)
    backend.start_experiment(meta, item_ids)
    results: list[CaseResult] = []
    for case in cases:
        started = time.monotonic()
        try:
            result = await _run_one(
                case, traced, judge_client, run, finish, judge_model=judge_model
            )
        except cerrors.CruxError as exc:
            # One case must not cost the run. The saturation experiment learnt
            # this the expensive way: a provider declining a single request
            # threw away eight completed cases.
            result = CaseResult(
                case=case,
                score=None,
                scores={},
                metrics={},
                rendered="",
                feedback=f"failed: {exc}",
                error=str(exc),
            )
        result = result.model_copy(
            update={"calls": log.drain(), "elapsed_seconds": time.monotonic() - started}
        )
        backend.log_result(result)
        results.append(result)
    backend.finish()
    return tuple(results)


async def _run_one(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    judge_client: pllm.LlmClient | None,
    run: RunCase,
    finish: FinishCase,
    *,
    judge_model: str | None,
) -> CaseResult:
    """
    Run, score and judge one case.

    :return: The result, without its calls or timing; the loop adds those.
    """
    session = await run(case, client)
    score = harness.score_case(case, session)
    scores = scorers.decompose(score, case)
    rendered = ""
    judgement: judge.Judgement | None = None
    compiled = await finish(case, client, session)
    if compiled is not None:
        rendered = compiled.render()
    if judge_client is not None and case.expected is not None and compiled is not None:
        judgement = await judge.judge(compiled, case.expected, judge_client, model=judge_model)
        scores["judge"] = judgement.score
    return CaseResult(
        case=case,
        score=score,
        scores=scores,
        metrics=scorers.metrics(score),
        rendered=rendered,
        judge_rationale=judgement.rationale if judgement else "",
        feedback=scorers.feedback(score, case, judgement),
    )


class PrintBackend:
    """
    The built-in backend: renders the report and per-case scores to stdout.

    Exists so the command is exercisable without any SDK installed, and so a
    replay run can be eyeballed before anything is uploaded.
    """

    name = "stdout"

    def __init__(self) -> None:
        self.results: list[CaseResult] = []
        self.meta: RunMetadata | None = None

    def upsert_dataset(self, cases: Sequence[harness.EvalCase]) -> dict[str, str]:
        return {case.id: case.id for case in cases}

    def start_experiment(self, meta: RunMetadata, item_ids: dict[str, str]) -> None:
        self.meta = meta

    def log_result(self, result: CaseResult) -> None:
        status = f"FAILED: {result.error[:100]}" if result.error else "ok"
        print(f"{result.case.id} ... {status}", flush=True)
        if result.score is not None and not result.score.clean:
            print("\n".join(f"    {line}" for line in result.feedback.splitlines()), flush=True)
        self.results.append(result)

    def finish(self) -> str:
        report = harness.Report(scores=tuple(r.score for r in self.results if r.score is not None))
        lines = [report.render(), ""]
        names = sorted({name for r in self.results for name in r.scores})
        lines.append(f"{'case':<28} " + " ".join(f"{n:>13}" for n in names))
        for result in self.results:
            cells = [
                f"{result.scores[n]:>13.2f}" if n in result.scores else f"{'-':>13}" for n in names
            ]
            lines.append(f"{result.case.id:<28} " + " ".join(cells))
        failed = [r for r in self.results if r.error]
        if failed:
            lines.append("")
            lines.append(f"{len(failed)} case(s) failed and are excluded from the numbers above:")
            lines.extend(f"  {r.case.id}: {r.error[:200]}" for r in failed)
        if self.meta is not None:
            lines.append("")
            lines.append(
                f"model {self.meta.model} · cassette {self.meta.cassette_mode} · "
                f"git {self.meta.git_sha} · crux {self.meta.crux_version}"
            )
        summary = "\n".join(lines)
        print(summary)
        return summary


async def _default_run_case(case: harness.EvalCase, client: pllm.LlmClient) -> csessn.Session:
    return await runner.run_case(case, client)


async def _default_finish_case(
    case: harness.EvalCase, client: pllm.LlmClient, session: csessn.Session
) -> coutput.CompiledPrompt | None:
    if session.outcome is not None:
        return session.outcome
    if not session.pending:
        return None
    step: csessn.Step = csessn.NeedsInput(session=session, questions=session.pending)
    finished = await runner.finish_case(case, client, step)
    return finished.outcome


def _git_sha() -> str:
    """
    :return: The short commit hash, or ``unknown`` outside a checkout.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _crux_version() -> str:
    """
    :return: The installed package version, or ``unknown``.
    """
    try:
        return importlib.metadata.version("crux-clarify")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
