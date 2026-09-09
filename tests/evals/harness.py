"""
The eval harness: a corpus, a scorer, and a cassette.

What this measures is **decision recall** — did crux surface the decisions that
matter, and did it stay quiet about the ones it should have resolved itself. It
does not measure whether the wording was nice.

Two deliberate choices worth defending:

- **No LLM judge in the ratchets.** Matching is canonical-id equality, falling
  back to token overlap for freeform decisions. A judge would make the harness
  itself flaky, and a flaky harness is one nobody reruns. A case may still carry
  an ``expected`` rubric, but that is scored by ``tests.evals.judge`` for the
  eval platforms only, replays from its own cassette, and never gates a test.
- **Aggregate assertions, never per-case.** Individual cases move for reasons
  that have nothing to do with a regression. The ratchet is on the mean.

The cassette exists because a re-scoring you have to pay for is a re-scoring you
do not do. Record once, replay for free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pathlib
from collections.abc import Sequence
from typing import Any, Literal

import pydantic
import yaml

import crux.domain.ids as cids
import crux.domain.session as csessn
import crux.ports.llm as pllm

CORPUS = pathlib.Path(__file__).parent / "corpus"
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CASSETTES = pathlib.Path(__file__).parent / "cassettes"

# A freeform decision matches a corpus expectation above this overlap. Much lower
# than the dedup threshold on purpose: dedup must not merge two real decisions,
# while a corpus matcher only has to recognise the same idea worded differently.
#
# Set at 0.5 rather than 0.6 because token overlap cannot see synonyms. "How
# callers are identified" and "how requests are identified" are the same decision
# and score 0.5; unrelated decisions in this domain score 0.0, so the gap is wide
# and the risk of a false match is small. A corpus `match` string should still use
# the vocabulary the domain uses, because nothing here understands paraphrase.
MATCH_THRESHOLD = 0.5

Stratum = Literal["one_liner", "feature", "project"]


class Expectation(pydantic.BaseModel):
    """
    One decision a case says crux should or should not have surfaced.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str | None = None
    match: str | None = None

    def matched_by(self, decision_id: str, undecided: str) -> bool:
        """
        Say whether one decision satisfies this expectation.

        :param decision_id: The decision's canonical or freeform id.
        :param undecided: What it says is undecided.
        :return: Whether it counts as a hit.
        """
        if self.id is not None:
            return decision_id == self.id
        if self.match is not None:
            return cids.similarity(self.match, undecided) >= MATCH_THRESHOLD
        return False

    def label(self) -> str:
        """
        :return: How to name this expectation in a report.
        """
        return self.id or f"~{self.match}"


RubricSection = Literal[
    "task_should", "constraints_should", "assumptions_should_cite", "must_not_claim"
]


class Rubric(pydantic.BaseModel):
    """
    What a good compiled prompt for this case says. Judged, never matched.

    Each bullet is a claim a reader could check against the rendered prompt, in
    the vocabulary the domain uses. ``must_not_claim`` bullets are met when the
    claim is absent.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    task_should: tuple[str, ...] = ()
    constraints_should: tuple[str, ...] = ()
    assumptions_should_cite: tuple[str, ...] = ()
    must_not_claim: tuple[str, ...] = ()

    def bullets(self) -> tuple[tuple[RubricSection, str], ...]:
        """
        :return: Every bullet with the section it belongs to, in a stable order.
        """
        sections: tuple[RubricSection, ...] = (
            "task_should",
            "constraints_should",
            "assumptions_should_cite",
            "must_not_claim",
        )
        return tuple((section, bullet) for section in sections for bullet in getattr(self, section))


class EvalCase(pydantic.BaseModel):
    """
    One prompt, and what crux is expected to make of it.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    prompt: str
    author: str
    """Who wrote the expectations and the rubric. Recorded so a prompt is never
    tuned against cases written by the person tuning it without that being
    visible."""
    stratum: Stratum = "feature"
    fixture_repo: str | None = None
    must_surface: tuple[Expectation, ...] = ()
    must_not_surface: tuple[Expectation, ...] = ()
    must_resolve_by_retrieval: tuple[Expectation, ...] = ()
    max_questions: int = 3
    expected: Rubric | None = None

    @property
    def root(self) -> pathlib.Path | None:
        """
        :return: The fixture repo this case runs against, if any.
        """
        return FIXTURES / self.fixture_repo if self.fixture_repo else None


class CaseScore(pydantic.BaseModel):
    """
    How one case went.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    stratum: Stratum
    recall: float
    surfaced: tuple[str, ...]
    missed: tuple[str, ...]
    should_have_been_quiet: tuple[str, ...]
    retrieval_expected_but_not: tuple[str, ...]
    questions_asked: int
    over_question_budget: bool
    assumptions: int
    decisions_total: int
    llm_calls: int

    @property
    def clean(self) -> bool:
        """
        :return: Whether the case hit every expectation it had.
        """
        return (
            not self.missed
            and not self.should_have_been_quiet
            and not self.retrieval_expected_but_not
            and not self.over_question_budget
        )


class Report(pydantic.BaseModel):
    """
    The aggregate, which is the only thing worth asserting on.
    """

    scores: tuple[CaseScore, ...] = ()

    @property
    def recall(self) -> float:
        """
        :return: Mean decision recall across the corpus.
        """
        if not self.scores:
            return 0.0
        return sum(s.recall for s in self.scores) / len(self.scores)

    @property
    def precision_misses(self) -> int:
        """
        :return: How many times crux surfaced something it should have resolved.
        """
        return sum(len(s.should_have_been_quiet) for s in self.scores)

    @property
    def budget_breaches(self) -> int:
        """
        :return: How many cases asked more questions than the case allowed.
        """
        return sum(1 for s in self.scores if s.over_question_budget)

    def ask_rate(self, stratum: Stratum) -> float:
        """
        Mean questions asked for one stratum.

        Tracked as a first-class product metric: the ask threshold is the entire
        product, and it is ungrounded enough that a prompt edit can move it
        without anyone noticing until users complain.

        :param stratum: Which prompt size to report on.
        :return: The mean, or 0.0 when the stratum is empty.
        """
        matching = [s for s in self.scores if s.stratum == stratum]
        if not matching:
            return 0.0
        return sum(s.questions_asked for s in matching) / len(matching)

    def render(self) -> str:
        """
        :return: A table a person can read.
        """
        lines = [
            f"{'case':<28} {'stratum':<10} {'recall':>7} {'asked':>6} {'assumed':>8}  status",
            "-" * 76,
        ]
        for score in self.scores:
            status = (
                "ok"
                if score.clean
                else ", ".join(
                    filter(
                        None,
                        [
                            f"missed {len(score.missed)}" if score.missed else "",
                            f"noisy {len(score.should_have_been_quiet)}"
                            if score.should_have_been_quiet
                            else "",
                            "over budget" if score.over_question_budget else "",
                        ],
                    )
                )
            )
            lines.append(
                f"{score.case_id:<28} {score.stratum:<10} {score.recall:>7.2f} "
                f"{score.questions_asked:>6} {score.assumptions:>8}  {status}"
            )
        lines.append("-" * 76)
        lines.append(
            f"recall {self.recall:.2f} · precision misses {self.precision_misses} · "
            f"budget breaches {self.budget_breaches}"
        )
        strata: tuple[Stratum, ...] = ("one_liner", "feature", "project")
        for stratum in strata:
            rate = self.ask_rate(stratum)
            if rate:
                lines.append(f"ask rate [{stratum}]: {rate:.1f}")
        return "\n".join(lines)


def load_corpus(directory: pathlib.Path = CORPUS) -> tuple[EvalCase, ...]:
    """
    Read every case in a corpus directory.

    :param directory: Where the YAML lives.
    :return: The cases, sorted by id.
    """
    cases = [
        EvalCase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.yaml"))
    ]
    return tuple(sorted(cases, key=lambda c: c.id))


def score_case(case: EvalCase, session: csessn.Session) -> CaseScore:
    """
    Score one finished session against its case.

    :param case: What was expected.
    :param session: What crux actually did.
    :return: The score.
    """
    decisions = [(n.id, n.undecided, n) for n in session.graph.nodes.values()]
    asked = {exchange.question.decision_id for exchange in session.transcript} | {
        q.decision_id for q in session.pending
    }

    surfaced: list[str] = []
    missed: list[str] = []
    for expectation in case.must_surface:
        if any(expectation.matched_by(i, u) for i, u, _ in decisions):
            surfaced.append(expectation.label())
        else:
            missed.append(expectation.label())

    noisy = [
        expectation.label()
        for expectation in case.must_not_surface
        if any(expectation.matched_by(i, u) and i in asked for i, u, _ in decisions)
    ]

    not_retrieved = [
        expectation.label()
        for expectation in case.must_resolve_by_retrieval
        if not any(
            expectation.matched_by(i, u)
            and n.resolution is not None
            and n.resolution.source == "retrieval"
            for i, u, n in decisions
        )
    ]

    outcome = session.outcome
    return CaseScore(
        case_id=case.id,
        stratum=case.stratum,
        recall=len(surfaced) / len(case.must_surface) if case.must_surface else 1.0,
        surfaced=tuple(surfaced),
        missed=tuple(missed),
        should_have_been_quiet=tuple(noisy),
        retrieval_expected_but_not=tuple(not_retrieved),
        questions_asked=session.budget.spent_questions,
        over_question_budget=session.budget.spent_questions > case.max_questions,
        assumptions=len(outcome.assumptions) if outcome else 0,
        decisions_total=len(session.graph.nodes),
        llm_calls=session.budget.spent_llm_calls,
    )


# #############################################################################
# Cassette
# #############################################################################


class CassetteLlm:
    """
    Records real completions to disk, then replays them for free.

    Forty lines, and it is the difference between rerunning the evals weekly and
    never rerunning them.
    """

    def __init__(
        self,
        inner: pllm.LlmClient | None,
        path: pathlib.Path,
        *,
        record: bool = False,
    ) -> None:
        self.salt = ""
        """Mixed into the cache key. The noise floor resamples an *identical*
        request on purpose, and without a salt all three samples hash the same
        and replay one another — reporting perfect self-agreement while having
        measured nothing at all."""

        """
        :param inner: The real client. Only needed when recording.
        :param path: Where the cassette lives.
        :param record: Whether to call the real client for unseen requests.
        """
        self._inner = inner
        self._path = path
        self._record = record
        self._entries: dict[str, Any] = {}
        if path.is_file():
            self._entries = json.loads(path.read_text(encoding="utf-8"))
        self.hits = 0
        self.misses = 0

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
        Replay a recorded turn, or record a new one.

        :param messages: The conversation.
        :param tools: Tools the model may call.
        :param force_tool: A tool it must call.
        :param model: Which model the caller asked for. Part of the key: without
            it, two identical requests routed to different models collide and
            silently replay each other's answers, which reads as a model quality
            change rather than as a bug.
        :param require_tool: Whether prose counts as a failed attempt. Not part of
            the key; it changes how a turn is judged, not what was asked.
        :return: The turn.
        :raises KeyError: When replaying and the request was never recorded.
        """
        key = _key(messages, force_tool, self.salt, model)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return pllm.LlmTurn.model_validate(cached)
        self.misses += 1
        if not self._record or self._inner is None:
            raise KeyError(
                f"Cassette miss for {force_tool!r} and recording is off. "
                f"Rerun with --record to capture it."
            )
        turn = await self._inner.complete(
            messages,
            tools=tools,
            force_tool=force_tool,
            model=model,
            require_tool=require_tool,
        )
        self._entries[key] = turn.model_dump(mode="json")
        self.save()
        return turn

    def save(self) -> None:
        """
        Write the cassette out.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")


def _key(
    messages: Sequence[pllm.Message],
    force_tool: str | None,
    salt: str = "",
    model: str | None = None,
) -> str:
    """
    Build a stable key for one request.

    The model joins the key only when one was named. Callers that route nothing
    pass ``None``, so every existing cassette keeps its keys and stays valid.

    :param messages: The conversation.
    :param force_tool: The tool that was forced.
    :param salt: Distinguishes deliberately identical requests.
    :param model: The model asked for, when the caller named one.
    :return: A hex digest.
    """
    parts = [[m.role, m.content] for m in messages] + [force_tool or "", salt]
    if model:
        parts.append(model)
    payload = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@contextlib.contextmanager
def cassette(name: str, *, record: bool = False) -> Any:
    """
    Open a cassette by name, saving it on the way out.

    :param name: The cassette's file stem.
    :param record: Whether to allow real calls.
    :return: A context manager yielding a factory that wraps a client.
    """
    path = CASSETTES / f"{name}.json"
    holder: dict[str, CassetteLlm] = {}

    def wrap(inner: pllm.LlmClient | None) -> CassetteLlm:
        holder["client"] = CassetteLlm(inner, path, record=record)
        return holder["client"]

    try:
        yield wrap
    finally:
        if "client" in holder and record:
            holder["client"].save()
