"""
Retrieval briefs, what came back, and what it adds up to.

Session-scoped by design: this is a citation structure for one prompt, not an
index of the repo. Staleness and invalidation belong to whatever retrieval stack
the host plugged in.

The distinction that matters here is between an *item* and a *candidate*. A grep
for a common term returns twenty items; that is not twenty possible answers. The
escalation rule counts candidates — distinct possible values, adjudicated out of
the items — because counting raw hits escalated every single decision and turned
the product into an interrogation.

Import as:

import crux.domain.evidence as cevid
"""

from __future__ import annotations

from typing import Literal

import pydantic

import crux.domain.decisions as cdecis

Depth = Literal["shallow", "normal", "deep"]
"""How hard a retriever should dig for one brief."""

ItemKind = Literal["file", "symbol", "snippet", "fact", "doc"]
"""What sort of thing a retriever found."""

# How hard to dig, keyed by what a wrong answer would cost. A cheap decision does
# not deserve a deep crawl, and a deep crawl on every brief is what makes a
# clarification layer cost more than the mistake it prevents.
_MAX_ITEMS: dict[Depth, int] = {"shallow": 5, "normal": 15, "deep": 40}

_DEPTH_FOR_COST: dict[cdecis.Cost, Depth] = {
    "low": "shallow",
    "medium": "normal",
    "high": "deep",
}

# A candidate dominates when it has at least twice the support of its nearest
# rival. Below that the evidence is genuinely split and crux should not pick.
_DOMINANCE_RATIO = 2.0


def depth_for_cost(cost: cdecis.Cost) -> Depth:
    """
    Pick a retrieval depth from what a wrong answer would cost.

    :param cost: The decision's blast radius.
    :return: The depth to brief at.
    """
    return _DEPTH_FOR_COST[cost]


def max_items_for_depth(depth: Depth) -> int:
    """
    :param depth: The brief's depth.
    :return: How many items a retriever should return at most.
    """
    return _MAX_ITEMS[depth]


# #############################################################################
# Brief
# #############################################################################


class Brief(pydantic.BaseModel):
    """
    One thing to go and find out.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    decision_id: str
    query: str
    accept_if: str
    hints: tuple[str, ...] = ()
    depth: Depth = "normal"
    max_items: int = _MAX_ITEMS["normal"]

    @staticmethod
    def for_decision(
        decision: cdecis.OpenDecision,
        *,
        brief_id: str,
        query: str,
        accept_if: str,
        hints: tuple[str, ...] = (),
    ) -> Brief:
        """
        Raise a brief sized to what the decision is worth.

        ``accept_if`` is the field people skip, and skipping it is expensive:
        without it an agentic retriever returns prose and the answer has to be
        re-parsed by another model call.

        :param decision: The decision the brief serves.
        :param brief_id: Id to mint it under.
        :param query: What to look for.
        :param accept_if: What would count as settling this.
        :param hints: Extra terms worth trying.
        :return: The brief.
        """
        depth = depth_for_cost(decision.cost_if_wrong)
        return Brief(
            id=brief_id,
            decision_id=decision.id,
            query=query,
            accept_if=accept_if,
            hints=hints,
            depth=depth,
            max_items=max_items_for_depth(depth),
        )


# #############################################################################
# What came back
# #############################################################################


class EvidenceItem(pydantic.BaseModel):
    """
    One thing a retriever found.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    brief_id: str
    locator: str
    kind: ItemKind
    excerpt: str = ""
    score: float = 1.0
    retriever: str = ""


class BriefResult(pydantic.BaseModel):
    """
    Everything one brief turned up.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    brief_id: str
    items: tuple[EvidenceItem, ...] = ()
    exhausted: bool = True
    error: str = ""

    @property
    def found_nothing(self) -> bool:
        """
        Say whether this brief came back empty.

        Finding nothing is not a failure — it is a legal outcome that routes the
        decision to a default or a retype.

        :return: Whether there are no items.
        """
        return len(self.items) == 0


# #############################################################################
# Candidate
# #############################################################################


class Candidate(pydantic.BaseModel):
    """
    One distinct possible answer, adjudicated out of the evidence items.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    value: str
    support: tuple[str, ...] = ()
    score: float = 1.0


def is_dominant(candidates: tuple[Candidate, ...]) -> bool:
    """
    Say whether one candidate clearly beats the rest.

    A single candidate always dominates. Two or more dominate only when the top
    one carries at least twice the support of its nearest rival — below that the
    evidence is genuinely split, and crux either escalates or records an
    assumption rather than pretending it knows.

    :param candidates: The adjudicated candidates, in any order.
    :return: Whether the field has a clear winner.
    """
    if not candidates:
        return False
    if len(candidates) == 1:
        return True
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    runner_up = ranked[1].score
    if runner_up <= 0.0:
        return True
    return ranked[0].score >= _DOMINANCE_RATIO * runner_up


def best(candidates: tuple[Candidate, ...]) -> Candidate | None:
    """
    :param candidates: The adjudicated candidates.
    :return: The highest-scoring one, or ``None`` when there are none.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.score)


def to_options(
    candidates: tuple[Candidate, ...],
) -> tuple[cdecis.Option, ...]:
    """
    Turn candidates into answerable options, best first.

    :param candidates: The adjudicated candidates.
    :return: Options carrying their supporting evidence ids.
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    return tuple(
        cdecis.Option(value=c.value, source="retrieval", evidence=c.support) for c in ranked
    )


# #############################################################################
# EvidenceGraph
# #############################################################################


class EvidenceGraph(pydantic.BaseModel):
    """
    Everything retrieval surfaced for one session.
    """

    briefs: dict[str, Brief] = pydantic.Field(default_factory=dict)
    items: dict[str, EvidenceItem] = pydantic.Field(default_factory=dict)
    by_decision: dict[str, tuple[str, ...]] = pydantic.Field(default_factory=dict)
    candidates: dict[str, tuple[Candidate, ...]] = pydantic.Field(default_factory=dict)

    def with_briefs(self, briefs: tuple[Brief, ...]) -> EvidenceGraph:
        """
        Record briefs that are about to go out.

        :param briefs: The briefs raised this round.
        :return: A new evidence graph holding them.
        """
        merged = dict(self.briefs)
        merged.update({brief.id: brief for brief in briefs})
        return self.model_copy(update={"briefs": merged})

    def with_results(self, results: tuple[BriefResult, ...]) -> EvidenceGraph:
        """
        Fold a batch of retrieval results in.

        :param results: What the retriever returned, one per brief.
        :return: A new evidence graph holding the items, indexed by decision.
        """
        items = dict(self.items)
        by_decision = {key: value for key, value in self.by_decision.items()}
        for result in results:
            brief = self.briefs.get(result.brief_id)
            if brief is None:
                # A retriever that answers a brief we never raised is confused,
                # not dangerous; its items simply have nowhere to attach.
                continue
            for item in result.items:
                items[item.id] = item
            existing = by_decision.get(brief.decision_id, ())
            by_decision[brief.decision_id] = existing + tuple(i.id for i in result.items)
        return self.model_copy(update={"items": items, "by_decision": by_decision})

    def with_candidates(self, decision_id: str, candidates: tuple[Candidate, ...]) -> EvidenceGraph:
        """
        Record what the adjudicator made of one decision's items.

        :param decision_id: The decision the candidates answer.
        :param candidates: The distinct possible values.
        :return: A new evidence graph holding them.
        """
        merged = dict(self.candidates)
        merged[decision_id] = candidates
        return self.model_copy(update={"candidates": merged})

    def candidates_for(self, decision_id: str) -> tuple[Candidate, ...]:
        """
        :param decision_id: The decision to look up.
        :return: Its adjudicated candidates, empty when none were found.
        """
        return self.candidates.get(decision_id, ())

    def items_for(self, decision_id: str) -> tuple[EvidenceItem, ...]:
        """
        :param decision_id: The decision to look up.
        :return: The evidence items retrieved for it.
        """
        return tuple(
            self.items[item_id]
            for item_id in self.by_decision.get(decision_id, ())
            if item_id in self.items
        )

    def locators(self, item_ids: tuple[str, ...]) -> tuple[str, ...]:
        """
        Turn evidence ids into something a reader can open.

        :param item_ids: The ids to resolve.
        :return: Their locators, skipping any id that is not present.
        """
        return tuple(self.items[i].locator for i in item_ids if i in self.items)
