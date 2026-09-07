"""
The serializable state of one clarification, and the two things a step returns.

A session holds **zero port references**. The engine holds the ports; the session
holds only data. That is what lets a host call ``start()`` on one worker, persist
the session anywhere JSON goes, and call ``resume()`` on another an hour later —
which is the whole reason yield/resume is the core and the blocking callback is
only a wrapper.

Import as:

import crux.domain.session as csessn
"""

from __future__ import annotations

import datetime as dt
import pathlib
from typing import Annotated, Literal

import pydantic

import crux.domain.evidence as cevid
import crux.domain.graph as cgraph
import crux.domain.output as coutput
import crux.domain.replies as creply

Phase = Literal[
    "seeding",
    "expanding",
    "retrieving",
    "routing",
    "awaiting_input",
    "compiling",
    "done",
]
"""Where a session is in the loop."""

# A pass yielding less than this fraction of genuinely new decisions has
# saturated. Not zero: a generative expander always produces *something*, so
# "until nothing new" would never terminate.
SATURATION_RATIO = 0.15


# #############################################################################
# Expansion history
# #############################################################################


class PassRecord(pydantic.BaseModel):
    """
    What one expansion pass produced.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    index: int
    lens: str
    proposed: int
    new_after_dedup: int
    saw_evidence: bool = False

    @property
    def ratio(self) -> float:
        """
        :return: The fraction of proposals that were genuinely new.
        """
        if self.proposed == 0:
            return 0.0
        return self.new_after_dedup / self.proposed


class PassLog(pydantic.BaseModel):
    """
    Every expansion pass this session has run.

    A first-class field rather than instrumentation, because the saturation
    experiment that decides whether the graph machinery survives is then a matter
    of reading a session rather than maintaining a forked build.
    """

    records: tuple[PassRecord, ...] = ()
    dirty: bool = False

    @property
    def count(self) -> int:
        """
        :return: How many passes have run.
        """
        return len(self.records)

    @property
    def saturated(self) -> bool:
        """
        Say whether another expansion pass is worth its tokens.

        :return: Whether the last pass fell below the saturation ratio and
            nothing has since dirtied it.
        """
        if self.dirty or not self.records:
            return False
        return self.records[-1].ratio < SATURATION_RATIO

    def with_record(self, record: PassRecord) -> PassLog:
        """
        Append a pass and clear the dirty flag it was run to satisfy.

        :param record: What the pass produced.
        :return: A new log.
        """
        return PassLog(records=self.records + (record,), dirty=False)

    def dirtied(self) -> PassLog:
        """
        Mark that a reply invalidated the decision set.

        :return: A new log owing another pass.
        """
        return self.model_copy(update={"dirty": True})


# #############################################################################
# Budget
# #############################################################################


class Budget(pydantic.BaseModel):
    """
    The caps that make termination provable.

    Every loop in crux is bounded by one of these. Without them an expander that
    keeps finding decisions, or a respondent who keeps asking back, would run for
    ever on someone else's money.
    """

    max_passes: int = 3
    max_rounds: int = 3
    max_questions_total: int = 6
    max_questions_per_round: int = 3
    max_exchanges_per_node: int = 2

    spent_questions: int = 0
    spent_rounds: int = 0
    spent_llm_calls: int = 0

    @property
    def questions_left(self) -> int:
        """
        :return: How many more questions may be asked, never negative.
        """
        return max(0, self.max_questions_total - self.spent_questions)

    @property
    def is_exhausted(self) -> bool:
        """
        Say whether crux must stop asking and compile with what it has.

        :return: Whether either the question or the round cap is spent.
        """
        return self.questions_left == 0 or self.spent_rounds >= self.max_rounds

    def spend(self, *, questions: int = 0, rounds: int = 0, llm_calls: int = 0) -> Budget:
        """
        Record consumption.

        :param questions: Questions put to a respondent.
        :param rounds: Round-trips completed.
        :param llm_calls: Model calls made.
        :return: A new budget.
        """
        return self.model_copy(
            update={
                "spent_questions": self.spent_questions + questions,
                "spent_rounds": self.spent_rounds + rounds,
                "spent_llm_calls": self.spent_llm_calls + llm_calls,
            }
        )


# #############################################################################
# Transcript
# #############################################################################


class Exchange(pydantic.BaseModel):
    """
    One question, and what came back.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    question: creply.Question
    reply: creply.RawReply
    classified: creply.ClassifiedReply
    at: dt.datetime = pydantic.Field(default_factory=lambda: dt.datetime.now(tz=dt.UTC))


# #############################################################################
# Session
# #############################################################################


class SessionContext(pydantic.BaseModel):
    """
    What the host told crux about the world this prompt lives in.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    root: pathlib.Path | None = None
    pack_ids: tuple[str, ...] = ("software",)
    downstream_agent: str = ""
    host_notes: str = ""


class Session(pydantic.BaseModel):
    """
    Everything crux knows about one clarification.
    """

    schema_version: Literal["1"] = "1"
    id: str
    prompt: str
    context: SessionContext = pydantic.Field(default_factory=SessionContext)

    phase: Phase = "seeding"
    graph: cgraph.DecisionGraph = pydantic.Field(default_factory=cgraph.DecisionGraph)
    evidence: cevid.EvidenceGraph = pydantic.Field(default_factory=cevid.EvidenceGraph)
    pending: tuple[creply.Question, ...] = ()
    transcript: tuple[Exchange, ...] = ()
    passes: PassLog = pydantic.Field(default_factory=PassLog)
    budget: Budget = pydantic.Field(default_factory=Budget)
    outcome: coutput.CompiledPrompt | None = None

    created_at: dt.datetime = pydantic.Field(default_factory=lambda: dt.datetime.now(tz=dt.UTC))
    updated_at: dt.datetime = pydantic.Field(default_factory=lambda: dt.datetime.now(tz=dt.UTC))

    def touched(self, **updates: object) -> Session:
        """
        Apply updates and stamp the session as changed.

        :param updates: Fields to replace.
        :return: A new session.
        """
        return self.model_copy(update={**updates, "updated_at": dt.datetime.now(tz=dt.UTC)})

    def question_for(self, question_id: str) -> creply.Question | None:
        """
        Find a pending question by id.

        :param question_id: The id a reply named.
        :return: The question, or ``None`` when it was never asked.
        """
        return next((q for q in self.pending if q.id == question_id), None)

    @property
    def may_expand(self) -> bool:
        """
        :return: Whether another expansion pass is both owed and affordable.
        """
        return not self.passes.saturated and self.passes.count < self.budget.max_passes


# #############################################################################
# What a step returns
# #############################################################################


class NeedsInput(pydantic.BaseModel):
    """
    crux has questions and is waiting.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["needs_input"] = "needs_input"
    session: Session
    questions: tuple[creply.Question, ...]

    @property
    def needs_input(self) -> bool:
        """
        :return: ``True``.
        """
        return True


class Done(pydantic.BaseModel):
    """
    crux has stopped and produced a compiled prompt.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["done"] = "done"
    session: Session
    compiled: coutput.CompiledPrompt

    @property
    def needs_input(self) -> bool:
        """
        :return: ``False``.
        """
        return False


Step = Annotated[NeedsInput | Done, pydantic.Field(discriminator="kind")]
"""What ``start()`` and ``resume()`` return. There is no failure arm: failures
raise a :class:`crux.errors.CruxError`."""
