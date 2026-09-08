"""
Turning what a model proposed into decisions the graph will accept.

Two jobs here carry most of the risk in crux.

**Dedup.** A duplicate decision does not merely ask twice; it splits the edges
that should have pointed at one node, so a prune fires against one copy and
misses the other. Exact on canonical ids, token-overlap on freeform.

**Grading.** ``cost_if_wrong`` comes from the proposal's *position* in a ranked
list rather than from a per-item label, because models rank far more reliably
than they score. Where a proposal resembles a pack spec, it inherits that spec's
authored grading instead — an authored value beats a ranked one, and the packs
are the only calibration anchor crux has.

Import as:

import crux.application.expansion as aexpand
"""

from __future__ import annotations

import logging

import crux.domain.decisions as cdecis
import crux.domain.graph as cgraph
import crux.domain.ids as cids
import crux.domain.session as csessn
import crux.packs.base as kspec
import crux.ports.reasoner as preason

_LOG = logging.getLogger(__name__)

FIRST_LENS = "What is undecided in this request?"

# Pass two onwards asks a genuinely different question. Re-asking the first lens
# would resample the same distribution, and the "new decisions" it produced would
# be sampling noise rather than depth -- which would make the saturation number
# measure nothing at all.
LATER_LENS = (
    "Given these decisions and this evidence, what does resolving them force "
    "someone to also decide?"
)


def lens_for(pass_index: int) -> str:
    """
    Pick the question to ask the expander.

    :param pass_index: Which pass this is, counting from one.
    :return: The lens.
    """
    return FIRST_LENS if pass_index <= 1 else LATER_LENS


def sketches(graph: cgraph.DecisionGraph) -> tuple[preason.DecisionSketch, ...]:
    """
    Describe what already exists, so the expander proposes only new things.

    :param graph: The graph so far.
    :return: One sketch per decision, resolved ones included so later passes can
        reason about what the answers imply.
    """
    return tuple(
        preason.DecisionSketch(
            id=node.id,
            undecided=node.undecided,
            type=node.type,
            resolved_to=node.value,
        )
        for node in graph.nodes.values()
        if node.status in ("open", "resolved")
    )


def absorb(
    graph: cgraph.DecisionGraph,
    result: preason.ExpansionResult,
    *,
    pass_index: int,
    pack_ids: tuple[str, ...],
    saw_evidence: bool = False,
) -> tuple[cgraph.DecisionGraph, csessn.PassRecord]:
    """
    Fold an expansion pass into the graph.

    :param graph: The graph so far.
    :param result: What the expander proposed, ordered most costly first.
    :param pass_index: Which pass this is, counting from one.
    :param pack_ids: Packs to match proposals against for authored grading.
    :param saw_evidence: Whether this pass had retrieved evidence to work
        from. Recorded because a pass run blind and a pass run with the
        codebase in hand are not comparable, and the saturation curve is
        uninterpretable without knowing which was which.
    :return: The new graph, and a record of what the pass actually yielded.
    """
    specs = kspec.specs_for(pack_ids)
    sibling_text = {spec_id: spec.undecided for spec_id, spec in specs.items()}
    existing = {node.id: node.undecided for node in graph.nodes.values()}
    aliases = {node_id: specs[node_id].aliases for node_id in existing if node_id in specs}

    total = len(result.proposed)

    # Ids are minted before anything is inserted, so a proposal can depend on a
    # sibling from the same batch. Without this pass, "digest requires
    # channel = email" is unexpressible whenever both are new -- which is the
    # canonical example of the whole design, and why the first live run attached
    # no edges at all.
    survivors: list[tuple[int, preason.ProposedDecision, str]] = []
    refs: dict[str, str] = {}
    for position, proposal in enumerate(result.proposed):
        duplicate = _duplicate_of(proposal.undecided, existing, aliases)
        if duplicate is not None:
            _LOG.info("Dropping proposal %r: duplicates %r", proposal.undecided, duplicate)
            continue
        decision_id = cids.freeform_id(proposal.undecided, pass_index=pass_index)
        survivors.append((position, proposal, decision_id))
        existing[decision_id] = proposal.undecided
        if proposal.ref:
            refs[proposal.ref] = decision_id

    added = 0
    for position, proposal, decision_id in survivors:
        decision = _to_decision(
            proposal,
            decision_id=decision_id,
            position=position,
            total=total,
            pass_index=pass_index,
            specs=specs,
            sibling_text=sibling_text,
            refs=refs,
            known=tuple(existing),
        )
        graph = graph.add(decision)
        added += 1

    record = csessn.PassRecord(
        index=pass_index,
        lens=lens_for(pass_index),
        proposed=total,
        new_after_dedup=added,
        saw_evidence=saw_evidence,
    )
    _LOG.info(
        "Expansion pass %d proposed %d, kept %d (ratio %.2f)",
        pass_index,
        total,
        added,
        record.ratio,
    )
    return graph, record


def _duplicate_of(
    undecided: str,
    existing: dict[str, str],
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> str | None:
    """
    Find an existing decision that says the same thing.

    Token overlap is the general rule. Pack decisions get two more, because a
    later pass restating the pack's own decisions as two-word titles was the
    commonest duplicate in the first recorded corpus run, and "Build scope"
    against the pack's forty-word text scores 0.17 on overlap: a proposal whose
    every token sits in the pack decision's title is that decision, and so is
    one matching an authored alias.

    :param undecided: The proposal's text.
    :param existing: Known decision ids mapped to their text.
    :param aliases: Pack decision ids mapped to their authored aliases.
    :return: The id of the duplicate, or ``None``.
    """
    proposed = cids.tokens(undecided)
    pack_aliases = aliases or {}
    for decision_id, text in existing.items():
        if cids.is_duplicate(undecided, text):
            return decision_id
        if decision_id not in pack_aliases:
            continue
        title = cids.tokens(text.split(":", 1)[0])
        if len(proposed) >= 2 and proposed <= title:
            return decision_id
        if any(cids.is_duplicate(undecided, alias) for alias in pack_aliases[decision_id]):
            return decision_id
    return None


def _to_decision(
    proposal: preason.ProposedDecision,
    *,
    decision_id: str,
    position: int,
    total: int,
    pass_index: int,
    specs: dict[str, kspec.DecisionSpec],
    sibling_text: dict[str, str],
    refs: dict[str, str],
    known: tuple[str, ...],
) -> cdecis.OpenDecision:
    """
    Build an open decision from one proposal.

    :param proposal: What the expander said is undecided.
    :param decision_id: The id already minted for it.
    :param position: Its place in the cost ranking, zero-based.
    :param total: How many proposals were ranked.
    :param pass_index: Which pass produced it.
    :param specs: Pack specs, for sibling grading.
    :param sibling_text: Spec ids mapped to their text, for matching.
    :param refs: Sibling refs mapped to the ids minted for them.
    :param known: Every id currently in play, for repairing a mistyped one.
    :return: The decision.
    """
    sibling_id = cids.closest(proposal.undecided, sibling_text)
    sibling = specs.get(sibling_id) if sibling_id is not None else None
    if sibling is not None:
        cost, reversibility = sibling.cost_if_wrong, sibling.reversibility
    else:
        cost, reversibility = (
            preason.cost_from_rank(position, total),
            proposal.reversibility,
        )
    return cdecis.OpenDecision(
        id=decision_id,
        undecided=proposal.undecided,
        type=proposal.type,
        cost_if_wrong=cost,
        reversibility=reversibility,
        options=tuple(cdecis.Option(value=value, source="expansion") for value in proposal.options)
        or None,
        retrieval_hint=proposal.retrieval_hint,
        depends_on=tuple(
            cdecis.Edge(
                kind=e.kind,
                source_id=_resolve_source(e.source_id, refs, known),
                when_value=e.when_value,
            )
            for e in proposal.edges
            if _edge_is_well_formed(e)
        ),
        origin=cdecis.Origin(seeded_by="expansion", pass_index=pass_index),
    )


def _resolve_source(source_id: str, refs: dict[str, str], known: tuple[str, ...]) -> str:
    """
    Work out which decision an edge actually points at.

    Three attempts, in order: a sibling's ref, an exact id, then the nearest
    known id. The third exists because ids are slugs up to forty-eight
    characters and a model copying one by hand gets it slightly wrong — one
    dropped the final letter of
    ``...logic-in-the-request``, and the edge was silently discarded. Losing a
    prune to a typo means somebody gets asked a dead question.

    :param source_id: What the model wrote.
    :param refs: Sibling refs mapped to the ids minted for them.
    :param known: Every id currently in play.
    :return: The best match, or the original when nothing is close.
    """
    if source_id in refs:
        return refs[source_id]
    if source_id in known:
        return source_id
    for candidate in known:
        if candidate.startswith(source_id) or source_id.startswith(candidate):
            _LOG.info("Repaired edge source %r to %r", source_id, candidate)
            return candidate
    repaired = cids.closest(source_id, {k: k for k in known}, threshold=0.85)
    if repaired is not None:
        _LOG.info("Repaired edge source %r to %r", source_id, repaired)
        return repaired
    return source_id


def _edge_is_well_formed(edge: preason.ProposedEdge) -> bool:
    """
    Reject an edge the domain would refuse to construct.

    The graph drops edges naming unknown sources and edges closing cycles; this
    catches the one case that would raise instead — a conditional kind with no
    trigger value, which a model produces often enough to matter.

    :param edge: The proposed edge.
    :return: Whether it can be built.
    """
    if edge.kind in ("requires", "prunes", "constrains") and edge.when_value is None:
        _LOG.warning("Dropping %s edge from %s: no when_value", edge.kind, edge.source_id)
        return False
    return True
