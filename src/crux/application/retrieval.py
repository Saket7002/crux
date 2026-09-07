"""
Raising briefs, spending them in one batch, and adjudicating what comes back.

The batch is the point. The decision graph knows every retrieval target at once,
before it asks anyone anything, so crux makes one trip with N briefs rather than
N trips with one. Four sequential explore-agent runs before the first question
would cost more than the mistake the whole layer exists to prevent.

Import as:

import crux.application.retrieval as aretr
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import crux.domain.evidence as cevid
import crux.domain.graph as cgraph
import crux.domain.ids as cids
import crux.errors as cerrors
import crux.packs.base as kspec
import crux.ports.reasoner as preason
import crux.ports.retrieval as pretr

_LOG = logging.getLogger(__name__)


def briefs_for(
    graph: cgraph.DecisionGraph,
    *,
    pack_ids: tuple[str, ...],
) -> tuple[cevid.Brief, ...]:
    """
    Raise a brief for every open decision worth looking up.

    A decision earns a brief when it is a fact about the world, or when it
    carries a retrieval hint. Depth comes from what a wrong answer would cost: a
    cheap decision does not deserve a deep crawl, and a deep crawl on every brief
    is what makes a clarification layer slower than the work it precedes.

    :param graph: The graph so far.
    :param pack_ids: Packs to consult for authored acceptance criteria.
    :return: The briefs, one per decision, none of them already briefed.
    """
    specs = kspec.specs_for(pack_ids)
    briefs: list[cevid.Brief] = []
    for node in graph.open_decisions():
        if node.brief_id is not None:
            continue
        if node.type != "missing_context" and not node.retrieval_hint:
            continue
        spec = specs.get(node.id)
        briefs.append(
            cevid.Brief.for_decision(
                node,
                brief_id=cids.brief_id(node.id),
                query=node.undecided,
                accept_if=(spec.accept_if if spec is not None else "")
                or f"evidence that settles: {node.undecided}",
                hints=tuple(h for h in node.retrieval_hint.split(",") if h.strip()),
            )
        )
    return tuple(briefs)


async def run(
    retriever: pretr.Retriever,
    briefs: tuple[cevid.Brief, ...],
) -> tuple[cevid.BriefResult, ...]:
    """
    Spend the whole batch in one call.

    :param retriever: The host's retriever, or the shipped filesystem one.
    :param briefs: What to find out.
    :return: One result per brief; a brief the retriever ignored comes back empty
        rather than missing, so downstream code need not special-case it.
    :raises RetrievalError: When the retriever itself failed.
    """
    if not briefs:
        return ()
    try:
        returned = await retriever.retrieve(briefs)
    except cerrors.CruxError:
        raise
    except Exception as exc:
        # An adapter boundary is the one place a broad catch belongs: a host's
        # explore agent can raise anything at all, and crux must report it as its
        # own error rather than leaking a stranger's exception type to the host.
        raise cerrors.RetrievalError(
            f"Retriever {type(retriever).__name__} failed on a batch of {len(briefs)} briefs: {exc}"
        ) from exc
    return _fill_gaps(briefs, returned)


def _fill_gaps(
    briefs: tuple[cevid.Brief, ...],
    returned: Sequence[cevid.BriefResult],
) -> tuple[cevid.BriefResult, ...]:
    """
    Give every brief a result, empty where the retriever skipped one.

    :param briefs: What was asked.
    :param returned: What came back, in any order.
    :return: One result per brief, in brief order.
    """
    by_id = {result.brief_id: result for result in returned}
    filled: list[cevid.BriefResult] = []
    for brief in briefs:
        result = by_id.get(brief.id)
        if result is None:
            _LOG.info("Retriever returned nothing for brief %s", brief.id)
            result = cevid.BriefResult(brief_id=brief.id)
        filled.append(result)
    return tuple(filled)


def adjudication_request(
    prompt: str,
    graph: cgraph.DecisionGraph,
    evidence: cevid.EvidenceGraph,
) -> preason.AdjudicationRequest:
    """
    Ask, in one call, what distinct answers the evidence actually supports.

    This is the step that makes escalation workable. A grep for a common term
    returns twenty items, and counting *those* escalated every decision when it
    was tried; counting distinct adjudicated answers does not.

    :param prompt: The user's original request, for context.
    :param graph: The graph so far.
    :param evidence: What retrieval turned up.
    :return: The batched request, empty when nothing needs adjudicating.
    """
    items: list[preason.AdjudicationItem] = []
    for node in graph.open_decisions():
        found = evidence.items_for(node.id)
        if not found:
            continue
        brief = evidence.briefs.get(node.brief_id or "")
        items.append(
            preason.AdjudicationItem(
                decision_id=node.id,
                undecided=node.undecided,
                accept_if=brief.accept_if if brief is not None else "",
                items=found,
            )
        )
    return preason.AdjudicationRequest(prompt=prompt, decisions=tuple(items))


def propagate_informs(
    graph: cgraph.DecisionGraph,
    evidence: cevid.EvidenceGraph,
) -> cevid.EvidenceGraph:
    """
    Attach a resolved decision's evidence to whatever it informs.

    An ``informs`` edge does not gate its target's existence and does not decide
    it; it says the source's evidence bears on the target too. Finding Resend in
    the manifest does not answer "which delivery channel", but it does change
    what answering it costs, and the person being asked should see the file.

    The earlier framing of this edge — that it "populates the target's options" —
    was wrong and is worth naming. The source's candidates answer the *source*:
    "resend" is not an option for "delivery channel". Deriving one decision's
    options from another's candidates would be inventing them, which is the one
    thing crux must never do. What actually travels along the edge is evidence.

    :param graph: The graph so far.
    :param evidence: What retrieval turned up.
    :return: The evidence graph, with items shared onto informed decisions.
    """
    updated = evidence
    for node in graph.open_decisions():
        inherited: list[str] = []
        for edge in node.edges("informs"):
            if not edge.fires(graph.value_of(edge.source_id)):
                continue
            already = set(updated.by_decision.get(node.id, ()))
            inherited.extend(
                item_id
                for item_id in updated.by_decision.get(edge.source_id, ())
                if item_id not in already
            )
        if inherited:
            _LOG.info("Sharing %d evidence items onto %s", len(inherited), node.id)
            merged = dict(updated.by_decision)
            merged[node.id] = merged.get(node.id, ()) + tuple(inherited)
            updated = updated.model_copy(update={"by_decision": merged})
    return updated
