"""
Does a compiled prompt make a downstream agent do better work?

    uv run python -m tests.evals.downstream --agent scripted
    uv run python -m tests.evals.downstream --agent-command 'claude -p "$(cat {prompt})"'

That is the claim the whole system rests on, and nothing else measures it.
For each corpus case with a fixture repository, the fixture is copied into two
sandboxes and an agent runs twice: once on the raw prompt, once on the prompt
crux compiled headless. The two diffs are judged against a rubric built from
the case: what the task should do, what the change must not claim, and the
files retrieval said to read. The score is the fraction of rubric bullets the
diff meets; the gap between the two runs is the result.

Every agent run is real work and every judgement is a model call. The
scripted agent exists so the harness itself is testable without either.

Import as:

import tests.evals.downstream as downstream
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import pathlib
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from typing import Protocol

import pydantic

import crux.domain.output as coutput
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.llm as pllm
import tests.evals.compare as compare
import tests.evals.harness as harness
import tests.evals.judge as judge
import tests.evals.runner as runner

PROMPT_FILE = "PROMPT.md"


class Agent(Protocol):
    """
    Whatever turns a prompt and a working copy into changes on disk.
    """

    async def run(self, prompt: str, workdir: pathlib.Path) -> None:
        """
        :param prompt: What to do.
        :param workdir: A sandbox copy of the fixture to change in place.
        """
        ...


class ScriptedAgent:
    """
    Writes the prompt it was given into one file. Enough to test the harness.
    """

    def __init__(self, filename: str = "CHANGES.md") -> None:
        """
        :param filename: The file it creates in the sandbox.
        """
        self.filename = filename
        self.prompts: list[str] = []

    async def run(self, prompt: str, workdir: pathlib.Path) -> None:
        self.prompts.append(prompt)
        (workdir / self.filename).write_text(prompt, encoding="utf-8")


class CommandAgent:
    """
    Runs a shell command in the sandbox with the prompt written to a file.

    ``{prompt}`` in the command is replaced by the prompt file's path, so any
    coding agent with a command line fits.
    """

    def __init__(self, command: str, *, timeout: float = 600.0) -> None:
        """
        :param command: The shell command, with ``{prompt}`` as a placeholder.
        :param timeout: Seconds before the run is killed.
        """
        self._command = command
        self._timeout = timeout

    async def run(self, prompt: str, workdir: pathlib.Path) -> None:
        prompt_path = workdir / PROMPT_FILE
        prompt_path.write_text(prompt, encoding="utf-8")
        command = self._command.replace("{prompt}", shlex.quote(str(prompt_path)))
        proc = await asyncio.create_subprocess_shell(command, cwd=workdir)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._timeout)
        except TimeoutError as exc:
            proc.kill()
            raise cerrors.CruxError(f"agent timed out after {self._timeout:.0f}s") from exc
        if proc.returncode != 0:
            raise cerrors.CruxError(f"agent exited with {proc.returncode}")


class PairResult(pydantic.BaseModel):
    """
    One case, both runs.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    stratum: str
    raw_score: float
    compiled_score: float
    assumptions_made_explicit: int
    raw_diff: str
    compiled_diff: str
    error: str = ""


def diff_rubric(case: harness.EvalCase, compiled: coutput.CompiledPrompt | None) -> harness.Rubric:
    """
    What a good change for this case looks like, as far as a diff can show it.

    :param case: The case.
    :param compiled: The compiled prompt, for the files retrieval named.
    :return: The rubric the judge reads against the diff.
    """
    expected = case.expected or harness.Rubric()
    read_first = tuple(
        f"changes or reads {c.locator}" for c in (compiled.context if compiled else ())
    )
    return harness.Rubric(
        task_should=expected.task_should,
        constraints_should=expected.constraints_should + read_first,
        assumptions_should_cite=tuple(
            f"states the choice made about {b}" for b in expected.assumptions_should_cite
        ),
        must_not_claim=expected.must_not_claim,
    )


def snapshot(root: pathlib.Path) -> dict[str, str]:
    """
    :param root: A directory.
    :return: Every text file under it, keyed by relative path.
    """
    files: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        try:
            files[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return files


def unified_diff(before: dict[str, str], after: dict[str, str]) -> str:
    """
    :param before: The fixture as copied.
    :param after: The sandbox after the agent ran.
    :return: One unified diff over every changed, added or removed file.
    """
    parts: list[str] = []
    for name in sorted(set(before) | set(after)):
        if before.get(name) == after.get(name):
            continue
        parts.extend(
            difflib.unified_diff(
                before.get(name, "").splitlines(keepends=True),
                after.get(name, "").splitlines(keepends=True),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
            )
        )
    return "".join(parts)


async def run_pair(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    judge_client: pllm.LlmClient,
    agent: Agent,
    *,
    sandbox_root: pathlib.Path,
    judge_model: str | None = None,
) -> PairResult:
    """
    Run one case both ways and judge both diffs.

    :param case: A case with a fixture repository.
    :param client: Where crux's completions come from.
    :param judge_client: Where the judge's completions come from.
    :param agent: The downstream agent.
    :param sandbox_root: Where the two copies are made.
    :param judge_model: Which model judges.
    :return: Both scores, both diffs, and how many guesses the compiled prompt
        made explicit: every decision closed by anything but a respondent.
    :raises ConfigurationError: When the case has no fixture.
    """
    if case.root is None:
        raise cerrors.ConfigurationError(f"{case.id} has no fixture repo to run an agent in.")
    session = await runner.run_case(
        case, client, budget=csessn.Budget(max_questions_total=0), answer=True
    )
    compiled = session.outcome
    rendered = compiled.render() if compiled else case.prompt
    rubric = diff_rubric(case, compiled)

    scores: list[float] = []
    diffs: list[str] = []
    for label, prompt in (("raw", case.prompt), ("compiled", rendered)):
        workdir = sandbox_root / case.id / label
        if workdir.exists():
            shutil.rmtree(workdir)
        shutil.copytree(case.root, workdir)
        before = snapshot(workdir)
        await agent.run(prompt, workdir)
        diff = unified_diff(before, snapshot(workdir))
        diffs.append(diff)
        judged = await judge.judge_text(
            diff or "(no changes)", rubric, judge_client, model=judge_model
        )
        scores.append(judged.score)
    return PairResult(
        case_id=case.id,
        stratum=case.stratum,
        raw_score=scores[0],
        compiled_score=scores[1],
        assumptions_made_explicit=(
            sum(1 for d in compiled.all_decisions if d.source != "respondent") if compiled else 0
        ),
        raw_diff=diffs[0],
        compiled_diff=diffs[1],
    )


def to_results(pairs: Sequence[PairResult], *, model: str, git_sha: str) -> compare.RunResults:
    """
    :param pairs: Every case's pair.
    :param model: The crux model.
    :param git_sha: The commit.
    :return: Rows the comparison tools read; ``raw`` and ``compiled`` are the scores.
    """
    import datetime as dt

    return compare.RunResults(
        experiment="downstream",
        model=model,
        git_sha=git_sha,
        cassette_mode="record",
        recorded_at=dt.datetime.now(tz=dt.UTC),
        rows=tuple(
            compare.CaseRow(
                case_id=p.case_id,
                stratum=p.stratum,
                scores={"raw": p.raw_score, "compiled": p.compiled_score} if not p.error else {},
                metrics={"assumptions_made_explicit": p.assumptions_made_explicit},
                error=p.error,
            )
            for p in pairs
        ),
    )


def render(pairs: Sequence[PairResult]) -> str:
    """
    :param pairs: Every case's pair.
    :return: The table.
    """
    scored = [p for p in pairs if not p.error]
    lines = [
        f"{'case':<28} {'stratum':<10} {'raw':>6} {'compiled':>9} {'explicit':>9}",
        "-" * 66,
    ]
    lines.extend(
        f"{p.case_id:<28} {p.stratum:<10} {p.raw_score:>6.2f} {p.compiled_score:>9.2f} "
        f"{p.assumptions_made_explicit:>9}"
        for p in scored
    )
    lines.append("-" * 66)
    if scored:
        raw = sum(p.raw_score for p in scored) / len(scored)
        compiled = sum(p.compiled_score for p in scored) / len(scored)
        wins = sum(1 for p in scored if p.compiled_score > p.raw_score)
        losses = sum(1 for p in scored if p.compiled_score < p.raw_score)
        lines.append(
            f"raw {raw:.2f} · compiled {compiled:.2f} · compiled leads {wins}, trails {losses}, "
            f"ties {len(scored) - wins - losses}"
        )
    failed = [p for p in pairs if p.error]
    if failed:
        lines.append(f"{len(failed)} case(s) failed:")
        lines.extend(f"  {p.case_id}: {p.error[:120]}" for p in failed)
    return "\n".join(lines)


async def _main(args: argparse.Namespace) -> int:
    """
    :param args: Parsed command line.
    :return: Process exit code.
    """
    isettn.load_dotenv()
    settings = isettn.CruxSettings(model=args.model) if args.model else isettn.CruxSettings()
    cases = [c for c in harness.load_corpus() if c.fixture_repo and runner.fixture_exists(c)]
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
    agent: Agent = CommandAgent(args.agent_command) if args.agent_command else ScriptedAgent()
    client = runner.build_client("recall", record=args.record, settings=settings)
    judge_client = runner.build_client("downstream-judge", record=args.record, settings=settings)
    sandbox_root = pathlib.Path(args.sandbox or tempfile.mkdtemp(prefix="crux-downstream-"))

    pairs: list[PairResult] = []
    for case in cases:
        print(f"{case.id} ...", flush=True)
        try:
            pairs.append(
                await run_pair(
                    case,
                    client,
                    judge_client,
                    agent,
                    sandbox_root=sandbox_root,
                    judge_model=args.judge_model,
                )
            )
        except (cerrors.CruxError, KeyError, OSError) as exc:
            pairs.append(
                PairResult(
                    case_id=case.id,
                    stratum=case.stratum,
                    raw_score=0.0,
                    compiled_score=0.0,
                    assumptions_made_explicit=0,
                    raw_diff="",
                    compiled_diff="",
                    error=str(exc),
                )
            )
    print()
    print(render(pairs))
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    out = compare.RESULTS / f"downstream-{runner.model_slug(settings.model)}.json"
    compare.save(to_results(pairs, model=settings.model, git_sha=sha or "unknown"), out)
    print(f"wrote {out}; sandboxes under {sandbox_root}")
    return 0


def main() -> int:
    """
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="tests.evals.downstream", description=__doc__)
    parser.add_argument(
        "--agent-command",
        default=None,
        help="shell command run in the sandbox; {prompt} is the prompt file. Default: scripted",
    )
    parser.add_argument("--model", default=None, help="crux model, overriding CRUX_MODEL")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--record", action="store_true", help="call the real model")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--sandbox", default=None, help="where to make the working copies")
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
