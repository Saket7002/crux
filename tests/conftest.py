"""
Factories shared across the suite.

A factory takes overrides and merges them over defaults, so a test names only
the fields its claim depends on and the reader can see what matters.
"""

from __future__ import annotations

from typing import Any

import crux.domain.decisions as cdecis


def build_decision(**overrides: Any) -> cdecis.OpenDecision:
    """
    Build an open decision, defaulting to one that is worth asking about.

    The default is high-cost and hard to reverse because that is the interesting
    case: a test about routing or the frontier usually cares that a decision
    *clears* the ask threshold, and the ones that do not say so explicitly.

    :param overrides: Fields to replace.
    :return: The decision.
    """
    defaults: dict[str, Any] = {
        "id": "d1",
        "undecided": "something undecided",
        "type": "underspecification",
        "cost_if_wrong": "high",
        "reversibility": "hard",
        "origin": cdecis.Origin(seeded_by="expansion", pass_index=1),
    }
    return cdecis.OpenDecision.model_validate(defaults | overrides)


def build_edge(**overrides: Any) -> cdecis.Edge:
    """
    Build a dependency edge.

    :param overrides: Fields to replace.
    :return: The edge.
    """
    defaults: dict[str, Any] = {
        "kind": "requires",
        "source_id": "src",
        "when_value": "yes",
    }
    return cdecis.Edge.model_validate(defaults | overrides)
