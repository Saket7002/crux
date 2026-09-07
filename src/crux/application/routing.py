"""
The route table: what closes each decision, decided without a model.

This is the heart of crux and it is deliberately pure. Given a decision, the
candidates adjudicated for it, whatever default its pack authored, and whether
the budget is spent, exactly one route comes out — every time, with no sampling
and no prompt to drift.

Two properties are load-bearing:

- **The table is total.** Every decision has a legal route in every state. A
  ``missing_context`` decision retrieval could not settle is *retyped* rather
  than left open, because its own type forbids asking and it would otherwise sit
  in the graph for ever.
- **Budget exhaustion blocks asking, not resolving.** A spent budget must never
  discard good evidence; it only closes the route to a human.

Import as:

import crux.application.routing as aroute
"""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic

import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid


class Resolve(pydantic.BaseModel):
    """
    Close the decision now, without asking anyone.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    route: Literal["resolve"] = "resolve"
    value: str
    source: cdecis.ResolutionSource
    dominant: bool = False
    evidence: tuple[str, ...] = ()
    rationale: str = ""
    options: tuple[cdecis.Option, ...] = ()


class Retype(pydantic.BaseModel):
    """
    Change what sort of undecided this is, so it has a route at all.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    route: Literal["retype"] = "retype"
    new_type: cdecis.DecisionType
    options: tuple[cdecis.Option, ...] = ()
    rationale: str = ""


class Ask(pydantic.BaseModel):
    """
    Put it to a respondent.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    route: Literal["ask"] = "ask"
    options: tuple[cdecis.Option, ...] = ()
    recommended: str | None = None
    rationale: str = ""


RouteOutcome = Annotated[Resolve | Retype | Ask, pydantic.Field(discriminator="route")]
"""What the table says to do with one decision."""


def route(
    decision: cdecis.OpenDecision,
    *,
    candidates: tuple[cevid.Candidate, ...] = (),
    pack_default: str | None = None,
    budget_spent: bool = False,
) -> RouteOutcome:
    """
    Decide how one open decision gets closed.

    :param decision: The decision to route.
    :param candidates: Distinct answers adjudicated out of its evidence. Note
        that these are candidates, never raw retrieval hits — a grep for a common
        term returns twenty files, which is not twenty answers, and counting hits
        escalated every decision when it was tried.
    :param pack_default: A named default authored by a pack, where one exists. A
        model-invented value is never a default; there is no code path from the
        expander to this argument.
    :param budget_spent: Whether crux may still ask a human anything.
    :return: The route.
    """
    outcome = (
        _route_fact(candidates, pack_default)
        if decision.type == "missing_context"
        else _route_judgement(decision, candidates, pack_default)
    )
    if isinstance(outcome, Ask) and budget_spent:
        return _fallback(candidates, pack_default, reason="budget spent")
    return outcome


def _route_fact(
    candidates: tuple[cevid.Candidate, ...],
    pack_default: str | None,
) -> RouteOutcome:
    """
    Route a ``missing_context`` decision, which may never reach a human.

    Asking someone to read their own codebase aloud is the failure mode the whole
    design exists to avoid, so when retrieval cannot settle one it is retyped
    into something a human *can* answer rather than put to them as a fact.

    :param candidates: What the evidence supports.
    :param pack_default: A named default, where one exists.
    :return: The route.
    """
    winner = cevid.best(candidates)
    if winner is not None and cevid.is_dominant(candidates):
        return Resolve(
            value=winner.value,
            source="retrieval",
            dominant=True,
            evidence=winner.support,
            rationale="one candidate clearly best supported by the evidence",
        )
    if candidates:
        return Retype(
            new_type="underspecification",
            options=cevid.to_options(candidates),
            rationale=f"{len(candidates)} candidates and none dominant",
        )
    if pack_default is not None:
        return Resolve(
            value=pack_default,
            source="default",
            rationale="retrieval found nothing; taking the pack default",
        )
    return Retype(
        new_type="underspecification",
        rationale="retrieval found nothing and no default is authored",
    )


def _route_judgement(
    decision: cdecis.OpenDecision,
    candidates: tuple[cevid.Candidate, ...],
    pack_default: str | None,
) -> RouteOutcome:
    """
    Route a decision a human could answer.

    :param decision: The decision to route.
    :param candidates: What the evidence supports.
    :param pack_default: A named default, where one exists.
    :return: The route.
    """
    options = cevid.to_options(candidates)
    winner = cevid.best(candidates)
    if winner is not None and cevid.is_dominant(candidates):
        if decision.is_ask_worthy:
            # Even unambiguous evidence gets confirmed when a wrong guess is
            # expensive and hard to undo. Confirming is one tap; being wrong
            # about a breaking change is not.
            return Ask(
                options=options,
                recommended=winner.value,
                rationale="evidence is clear, but the cost of being wrong earns a confirmation",
            )
        return Resolve(
            value=winner.value,
            source="retrieval",
            dominant=True,
            evidence=winner.support,
            options=options,
            rationale="one candidate clearly best supported by the evidence",
        )
    if decision.is_ask_worthy:
        return Ask(
            options=options,
            rationale="above the ask threshold and the evidence does not settle it",
        )
    return _fallback(candidates, pack_default, reason="below the ask threshold")


def _fallback(
    candidates: tuple[cevid.Candidate, ...],
    pack_default: str | None,
    *,
    reason: str,
) -> RouteOutcome:
    """
    Close a decision crux will not ask about.

    A named default where a pack authored one, otherwise the judgement goes
    downstream carrying whatever option space crux found, so the agent need not
    rediscover it.

    :param candidates: What the evidence supports.
    :param pack_default: A named default, where one exists.
    :param reason: Why this decision is not being asked, for the audit trail.
    :return: The route.
    """
    options = cevid.to_options(candidates)
    if pack_default is not None:
        return Resolve(
            value=pack_default,
            source="default",
            options=options,
            rationale=f"{reason}; taking the pack default",
        )
    winner = cevid.best(candidates)
    return Resolve(
        value=winner.value if winner is not None else "left to the downstream agent",
        source="delegated",
        options=options,
        rationale=f"{reason}; no default authored, so the judgement goes downstream",
    )
