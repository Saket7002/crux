"""
The decision-type registry.

A pack is **declarative data**, never a Protocol and never callables. Two reasons,
both load-bearing: a pack with code in it could not be serialised alongside the
session, and it would let a host smuggle I/O into the domain layer. A host that
needs a computed default supplies it through their ``Retriever`` instead.

Packs are also the only calibration anchor crux has. A spec's ``cost_if_wrong``
and ``reversibility`` are *authored by a person*, and a freeform decision the
expander invented inherits them from its nearest sibling. Without that, the ask
threshold rests entirely on an uncalibrated model judgement.

Import as:

import crux.packs.base as kspec
"""

from __future__ import annotations

import re

import pydantic

import crux.domain.decisions as cdecis


class DecisionSpec(pydantic.BaseModel):
    """
    One decision a pack knows about in advance.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    canonical_id: str
    undecided: str
    type: cdecis.DecisionType
    cost_if_wrong: cdecis.Cost
    reversibility: cdecis.Reversibility
    options: tuple[str, ...] = ()
    default: str | None = None
    retrieval_hint: str = ""
    accept_if: str = ""
    triggers: tuple[str, ...] = ()
    always: bool = False
    credential_shaped: bool = False

    def matches(self, prompt: str) -> bool:
        """
        Say whether this spec applies to a prompt.

        :param prompt: What the user asked for.
        :return: Whether the spec always applies, or one of its triggers hits.
        """
        if self.always:
            return True
        return any(re.search(pattern, prompt, re.IGNORECASE) for pattern in self.triggers)

    def instantiate(self, *, pack_id: str) -> cdecis.OpenDecision:
        """
        Turn the spec into an open decision in the graph.

        :param pack_id: The pack this came from, for the audit trail.
        :return: The decision, carrying its authored cost and reversibility.
        """
        return cdecis.OpenDecision(
            id=self.canonical_id,
            undecided=self.undecided,
            type=self.type,
            cost_if_wrong=self.cost_if_wrong,
            reversibility=self.reversibility,
            options=tuple(cdecis.Option(value=value, source="pack") for value in self.options)
            or None,
            retrieval_hint=self.retrieval_hint,
            origin=cdecis.Origin(seeded_by="pack", pack_id=pack_id, spec_id=self.canonical_id),
        )


class DecisionPack(pydantic.BaseModel):
    """
    A named set of decisions that recur in one kind of work.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    description: str
    specs: tuple[DecisionSpec, ...] = ()

    def seed(self, prompt: str) -> tuple[cdecis.OpenDecision, ...]:
        """
        Instantiate every spec this prompt triggers.

        :param prompt: What the user asked for.
        :return: The seeded decisions.
        """
        return tuple(
            spec.instantiate(pack_id=self.id) for spec in self.specs if spec.matches(prompt)
        )

    def spec_for(self, canonical_id: str) -> DecisionSpec | None:
        """
        :param canonical_id: The spec to find.
        :return: The spec, or ``None`` when this pack does not have it.
        """
        return next((s for s in self.specs if s.canonical_id == canonical_id), None)


_REGISTRY: dict[str, DecisionPack] = {}


def register(pack: DecisionPack) -> None:
    """
    Add a pack to the registry, replacing any pack with the same id.

    :param pack: The pack to register.
    """
    _REGISTRY[pack.id] = pack


def get(pack_id: str) -> DecisionPack | None:
    """
    :param pack_id: The pack to find.
    :return: The pack, or ``None`` when it was never registered.
    """
    return _REGISTRY.get(pack_id)


def registered() -> tuple[str, ...]:
    """
    :return: The ids of every registered pack, sorted.
    """
    return tuple(sorted(_REGISTRY))


def seed(prompt: str, pack_ids: tuple[str, ...]) -> tuple[cdecis.OpenDecision, ...]:
    """
    Seed a graph from every named pack a prompt triggers.

    :param prompt: What the user asked for.
    :param pack_ids: Which packs to consult.
    :return: The seeded decisions, deduplicated by canonical id.
    """
    seen: set[str] = set()
    seeded: list[cdecis.OpenDecision] = []
    for pack_id in pack_ids:
        pack = _REGISTRY.get(pack_id)
        if pack is None:
            continue
        for decision in pack.seed(prompt):
            if decision.id in seen:
                continue
            seen.add(decision.id)
            seeded.append(decision)
    return tuple(seeded)


def default_for(canonical_id: str, pack_ids: tuple[str, ...]) -> str | None:
    """
    Find the authored default for a decision, if any pack has one.

    This is the only source of a ``default`` resolution. A value a model invented
    is a bug, not a default, and there is deliberately no other way to reach the
    router's ``pack_default`` argument.

    :param canonical_id: The decision to look up.
    :param pack_ids: Which packs to consult.
    :return: The default, or ``None``.
    """
    for pack_id in pack_ids:
        pack = _REGISTRY.get(pack_id)
        if pack is None:
            continue
        spec = pack.spec_for(canonical_id)
        if spec is not None and spec.default is not None:
            return spec.default
    return None


def specs_for(pack_ids: tuple[str, ...]) -> dict[str, DecisionSpec]:
    """
    Collect every spec from the named packs, for sibling matching.

    :param pack_ids: Which packs to consult.
    :return: Canonical ids mapped to their specs.
    """
    collected: dict[str, DecisionSpec] = {}
    for pack_id in pack_ids:
        pack = _REGISTRY.get(pack_id)
        if pack is None:
            continue
        for spec in pack.specs:
            collected.setdefault(spec.canonical_id, spec)
    return collected
