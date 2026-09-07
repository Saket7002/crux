"""
When to stop expanding, and when to stop asking.

Two rules for two different limits, and conflating them is the mistake this
module exists to prevent. Expansion stops on **saturation** — a measurable
plateau in genuinely new decisions per pass. Asking stops on **worth** — whether
any open decision is expensive enough and irreversible enough to earn an
interruption.

Neither is a self-reported confidence score. A model asked "are you sure you
understand?" always answers, and the answer is uncalibrated. Marginal new
decisions per pass is an observable; cost of being wrong is a fact about the
world rather than about the model's own epistemic state.

Import as:

import crux.application.stopping as astop
"""

from __future__ import annotations

import crux.domain.graph as cgraph
import crux.domain.session as csessn


def should_expand(session: csessn.Session) -> bool:
    """
    Say whether another expansion pass is worth its tokens.

    :param session: The session so far.
    :return: Whether expansion has neither saturated nor run out of passes.
    """
    return session.may_expand


def should_ask(session: csessn.Session) -> bool:
    """
    Say whether crux should put another round to a respondent.

    :param session: The session so far.
    :return: Whether the budget allows it and something open is worth asking.
    """
    if session.budget.is_exhausted:
        return False
    return bool(session.graph.frontier())


def remaining_open(graph: cgraph.DecisionGraph) -> tuple[str, ...]:
    """
    Name what is still open, for the audit trail and for tests.

    :param graph: The graph to inspect.
    :return: The ids of every live, unresolved decision.
    """
    return tuple(node.id for node in graph.open_decisions())
