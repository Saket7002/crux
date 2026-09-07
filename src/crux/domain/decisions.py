"""
The open-decision primitive and everything that closes one.

An open decision is something undecided about the intent. It is deliberately not
a question: asking is one of four ways to close one, and three of those never
reach a human. A model built on questions cannot represent "closed by looking it
up", which is the majority of them.

Import as:

import crux.domain.decisions as cdecis
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

import pydantic

# #############################################################################
# Closed sets
# #############################################################################

# Every closed set here is a Literal rather than an Enum, because a Session must
# round-trip through a host's JSON storage with no crux code on the other side.
# A Literal is already a string on the wire; an Enum needs a custom encoder.

DecisionType = Literal["ambiguity", "underspecification", "vagueness", "missing_context"]
"""How a decision is undecided, which is what picks its resolution route."""

ResolutionSource = Literal["respondent", "retrieval", "default", "delegated"]
"""What closed a decision. Only ``respondent`` involved a human."""

Grade = Literal["confirmed", "inferred", "assumed"]
"""How much weight a resolution carries. Derived from the source, never judged."""

Cost = Literal["low", "medium", "high"]
"""Blast radius of getting this decision wrong."""

Reversibility = Literal["easy", "hard"]
"""How expensive it is to change one's mind later."""

Status = Literal["open", "resolved", "pruned", "invalidated"]
"""Where a decision is in its lifecycle."""

EdgeKind = Literal["requires", "prunes", "constrains", "informs"]
"""What one decision does to another."""

OptionSource = Literal["pack", "expansion", "retrieval", "proposal"]
"""Where a candidate answer came from."""

SeededBy = Literal["pack", "expansion", "proposal", "retype"]
"""What put a decision into the graph."""


# #############################################################################
# Option
# #############################################################################


class Option(pydantic.BaseModel):
    """
    One candidate answer to a decision.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    value: str
    label: str = ""
    source: OptionSource
    evidence: tuple[str, ...] = ()

    def display(self) -> str:
        """
        Render the option for a respondent.

        :return: The label where one was authored, else the raw value.
        """
        return self.label or self.value


# #############################################################################
# Edge
# #############################################################################


class Edge(pydantic.BaseModel):
    """
    A dependency, stored on the *target* decision.

    Storing edges on the target is what keeps the graph free of traversal
    algorithms: every edge kind evaluates as a local check against one source
    decision, so the frontier and the prune pass are each a single loop over the
    node dict with O(1) lookups.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    kind: EdgeKind
    source_id: str
    when_value: str | None = None

    @pydantic.model_validator(mode="after")
    def _conditional_kinds_need_a_value(self) -> Edge:
        """
        Reject an edge whose kind is meaningless without a trigger value.

        :return: The validated edge.
        :raises ValueError: When a conditional kind carries no ``when_value``.
        """
        if self.kind in ("requires", "prunes", "constrains") and self.when_value is None:
            raise ValueError(f"{self.kind!r} edge from {self.source_id!r} needs a when_value")
        return self

    def fires(self, source_value: str | None) -> bool:
        """
        Say whether this edge is triggered by the source's resolved value.

        :param source_value: The value the source resolved to, or ``None`` when
            it is still open.
        :return: Whether the edge's condition is met.
        """
        if source_value is None:
            return False
        if self.when_value is None:
            # An `informs` edge has no condition; it fires whenever the source
            # resolves at all.
            return True
        return source_value == self.when_value


# #############################################################################
# Origin
# #############################################################################


class Origin(pydantic.BaseModel):
    """
    Where a decision came from, kept for the audit trail.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    seeded_by: SeededBy
    pass_index: int = 0
    pack_id: str | None = None
    spec_id: str | None = None


# #############################################################################
# Resolution
# #############################################################################


def derive_grade(source: ResolutionSource, *, dominant: bool = False) -> Grade:
    """
    Work out how much weight a resolution carries.

    The grade is derived rather than judged so it cannot drift from reality: a
    model asked "how confident are you?" will answer, and the answer means
    nothing. Where the value came from is a fact.

    :param source: What closed the decision.
    :param dominant: For a retrieval, whether one candidate clearly beat the
        others. Ignored for every other source.
    :return: The grade.
    """
    if source == "respondent":
        return "confirmed"
    if source == "retrieval" and dominant:
        return "inferred"
    return "assumed"


class Resolution(pydantic.BaseModel):
    """
    How a decision closed, and by what.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    value: str
    source: ResolutionSource
    grade: Grade
    evidence: tuple[str, ...] = ()
    rationale: str = ""
    at: dt.datetime = pydantic.Field(
        default_factory=lambda: dt.datetime.now(tz=dt.UTC),
    )

    @property
    def is_assumption(self) -> bool:
        """
        Say whether this resolution belongs in the output's assumptions block.

        :return: Whether it was closed without a human confirming it.
        """
        return self.source != "respondent"


# #############################################################################
# OpenDecision
# #############################################################################


class OpenDecision(pydantic.BaseModel):
    """
    Something undecided about the intent.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    undecided: str
    type: DecisionType
    cost_if_wrong: Cost
    reversibility: Reversibility
    origin: Origin

    options: tuple[Option, ...] | None = None
    mandatory: bool = False
    resolution: Resolution | None = None
    depends_on: tuple[Edge, ...] = ()
    retrieval_hint: str = ""
    brief_id: str | None = None
    status: Status = "open"
    ask_exchanges: int = 0
    retyped_from: DecisionType | None = None
    invalidation_reason: str = ""

    # ## Predicates

    @property
    def is_open(self) -> bool:
        """
        :return: Whether this decision still needs closing.
        """
        return self.status == "open"

    @property
    def is_resolved(self) -> bool:
        """
        :return: Whether this decision has a resolution.
        """
        return self.status == "resolved" and self.resolution is not None

    @property
    def value(self) -> str | None:
        """
        :return: The resolved value, or ``None`` while still open.
        """
        return self.resolution.value if self.resolution is not None else None

    @property
    def is_expensive_and_irreversible(self) -> bool:
        """
        Say whether a wrong guess here would be both costly and hard to undo.

        This is the only condition that earns a respondent's attention. Cost
        alone is not enough — an expensive mistake that takes an afternoon to
        correct is cheaper than the interruption.

        :return: Whether the decision clears the ask threshold on its own merits.
        """
        return self.cost_if_wrong == "high" and self.reversibility == "hard"

    @property
    def is_ask_worthy(self) -> bool:
        """
        :return: Whether this decision may be put to a respondent at all.
        """
        return self.is_expensive_and_irreversible or self.mandatory

    @property
    def is_askable_type(self) -> bool:
        """
        Say whether this decision's type permits asking a human.

        ``missing_context`` never reaches a respondent — it is a fact about the
        world, and asking someone to read their own codebase aloud is the
        failure mode this whole design exists to avoid. When retrieval cannot
        settle one, it is retyped rather than asked.

        :return: Whether the type is one a human can answer.
        """
        return self.type != "missing_context"

    def edges(self, kind: EdgeKind) -> tuple[Edge, ...]:
        """
        Select this decision's inbound edges of one kind.

        :param kind: The edge kind to select.
        :return: The matching edges, in declaration order.
        """
        return tuple(edge for edge in self.depends_on if edge.kind == kind)

    # ## Transitions
    #
    # Every transition returns a new decision. The graph is mutable by copy, so
    # a caller holding an older one still sees what it saw.

    def resolve(
        self,
        value: str,
        *,
        source: ResolutionSource,
        dominant: bool = False,
        evidence: tuple[str, ...] = (),
        rationale: str = "",
    ) -> OpenDecision:
        """
        Close this decision.

        :param value: What it resolved to.
        :param source: What closed it.
        :param dominant: For a retrieval, whether one candidate clearly beat the
            others; this is what separates ``inferred`` from ``assumed``.
        :param evidence: Ids of the evidence items supporting the value.
        :param rationale: Why, in one line, for the audit trail.
        :return: A resolved copy.
        """
        resolution = Resolution(
            value=value,
            source=source,
            grade=derive_grade(source, dominant=dominant),
            evidence=evidence,
            rationale=rationale,
        )
        return self.model_copy(update={"resolution": resolution, "status": "resolved"})

    def retype(
        self,
        new_type: DecisionType,
        *,
        options: tuple[Option, ...] | None = None,
    ) -> OpenDecision:
        """
        Change how this decision is undecided.

        The one caller that matters is a ``missing_context`` decision retrieval
        could not settle. Without retyping it has no legal terminal state: its
        type forbids asking and its evidence is empty, so it would sit open for
        ever.

        :param new_type: The type to become.
        :param options: Candidate answers to carry over, where retrieval found
            some but could not choose between them.
        :return: A retyped copy, remembering what it was.
        """
        return self.model_copy(
            update={
                "type": new_type,
                "retyped_from": self.retyped_from or self.type,
                "options": options if options is not None else self.options,
            }
        )

    def prune(self) -> OpenDecision:
        """
        Delete this decision because an answer elsewhere made it moot.

        :return: A pruned copy.
        """
        return self.model_copy(update={"status": "pruned"})

    def invalidate(self, reason: str) -> OpenDecision:
        """
        Mark this decision as one the respondent rejected as wrong to ask.

        :param reason: What they said was wrong with it.
        :return: An invalidated copy.
        """
        return self.model_copy(update={"status": "invalidated", "invalidation_reason": reason})

    def make_mandatory(self) -> OpenDecision:
        """
        Promote this decision from optional to one that must be settled.

        :return: A copy that clears the ask threshold regardless of its cost.
        """
        return self.model_copy(update={"mandatory": True})

    def with_options(self, options: tuple[Option, ...]) -> OpenDecision:
        """
        Replace this decision's candidate answers.

        :param options: The new options.
        :return: A copy carrying them.
        """
        return self.model_copy(update={"options": options})

    def with_brief(self, brief_id: str) -> OpenDecision:
        """
        Record that a retrieval brief has been raised for this decision.

        :param brief_id: The brief's id.
        :return: A copy carrying it.
        """
        return self.model_copy(update={"brief_id": brief_id})

    def asked_again(self) -> OpenDecision:
        """
        Count one more round-trip with the respondent on this decision.

        :return: A copy with the exchange counter advanced.
        """
        return self.model_copy(update={"ask_exchanges": self.ask_exchanges + 1})
