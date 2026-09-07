"""
The decision graph and the four things it can do.

Stored as a DAG, operated as a flat dict. Because every edge lives on its target,
each of ``live`` / ``frontier`` / ``cascade`` is one pass over the nodes with
O(1) source lookups — there is no traversal, no topological sort, and no BFS
anywhere in this module. That is deliberate: the shape is kept because
retrofitting node identity later is painful, but the machinery is not built
until a measurement says it is needed.

Import as:

import crux.domain.graph as cgraph
"""

from __future__ import annotations

import logging

import pydantic

import crux.domain.decisions as cdecis
import crux.errors as cerrors

_LOG = logging.getLogger(__name__)

# A `requires` chain deeper than this is not a dependency, it is a bug. The bound
# is what lets cycle rejection stay a local check at insert time instead of a
# traversal at read time.
_MAX_CHAIN_DEPTH = 4


class DecisionGraph(pydantic.BaseModel):
    """
    Every open decision for one session, and what depends on what.

    Mutable by copy: every operation returns a new graph, so a caller holding an
    older one still sees what it saw.
    """

    nodes: dict[str, cdecis.OpenDecision] = pydantic.Field(default_factory=dict)

    # ## Reads

    def __contains__(self, decision_id: str) -> bool:
        return decision_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, decision_id: str) -> cdecis.OpenDecision:
        """
        Look up one decision.

        :param decision_id: The id to find.
        :return: The decision.
        :raises UnknownDecisionError: When no such decision is in the graph.
        """
        try:
            return self.nodes[decision_id]
        except KeyError:
            raise cerrors.UnknownDecisionError(
                f"No decision {decision_id!r} in this graph; known ids are {sorted(self.nodes)}"
            ) from None

    def value_of(self, decision_id: str) -> str | None:
        """
        Read what a decision resolved to, tolerating an absent source.

        An edge may name a source that a prune removed, which is not an error —
        an edge whose source is gone simply never fires.

        :param decision_id: The decision to read.
        :return: Its resolved value, or ``None`` when open, pruned or unknown.
        """
        node = self.nodes.get(decision_id)
        if node is None or not node.is_resolved:
            return None
        return node.value

    def is_live(self, decision: cdecis.OpenDecision) -> bool:
        """
        Say whether a decision exists yet.

        A ``requires`` edge means the target does not exist until the source
        resolves a particular way. An unborn decision is held in the graph but is
        never briefed, never ranked and never asked — which is what stops crux
        putting "immediate or daily digest?" to someone who chose in-app only.

        :param decision: The decision to test.
        :return: Whether every ``requires`` edge on it fires.
        """
        return all(edge.fires(self.value_of(edge.source_id)) for edge in decision.edges("requires"))

    def open_decisions(self) -> tuple[cdecis.OpenDecision, ...]:
        """
        :return: Every live, still-open decision, in insertion order.
        """
        return tuple(node for node in self.nodes.values() if node.is_open and self.is_live(node))

    def resolved(self) -> tuple[cdecis.OpenDecision, ...]:
        """
        :return: Every resolved decision, in insertion order.
        """
        return tuple(node for node in self.nodes.values() if node.is_resolved)

    def fan_out(self, decision_id: str) -> int:
        """
        Count what hangs off a decision.

        Used to rank the frontier: answering the decision that gates or kills the
        most others collapses the most uncertainty per interruption.

        :param decision_id: The decision to measure.
        :return: How many other decisions carry an edge sourced here.
        """
        return sum(
            1
            for node in self.nodes.values()
            if any(edge.source_id == decision_id for edge in node.depends_on)
        )

    def frontier(self) -> tuple[cdecis.OpenDecision, ...]:
        """
        Select the decisions that could be put to a respondent right now.

        Ranked most-worth-asking first: cost, then reversibility, then fan-out,
        then id for a stable order.

        :return: The ranked frontier.
        """
        askable = [
            node for node in self.open_decisions() if node.is_askable_type and node.is_ask_worthy
        ]
        cost_rank = {"high": 0, "medium": 1, "low": 2}
        askable.sort(
            key=lambda n: (
                cost_rank[n.cost_if_wrong],
                0 if n.reversibility == "hard" else 1,
                -self.fan_out(n.id),
                n.id,
            )
        )
        return tuple(askable)

    def edge_stats(self) -> dict[str, tuple[int, int]]:
        """
        Count edges, and how many of them ever did anything.

        This is the measurement the whole graph question turns on. An edge that
        is attached but never fires is decoration: it costs tokens to generate,
        risks hiding a question if it is wrong, and buys nothing. The saturation
        experiment reads this to decide whether the conditional-edge machinery
        survives at all.

        :return: Edge kind mapped to (attached, fired).
        """
        counts: dict[str, tuple[int, int]] = {}
        for node in self.nodes.values():
            for edge in node.depends_on:
                attached, fired = counts.get(edge.kind, (0, 0))
                did_fire = edge.fires(self.value_of(edge.source_id))
                counts[edge.kind] = (attached + 1, fired + (1 if did_fire else 0))
        return counts

    # ## Writes

    def add(self, decision: cdecis.OpenDecision) -> DecisionGraph:
        """
        Insert a decision, dropping any edge that cannot be honoured.

        Validation happens here rather than at read time so that the read path
        stays a flat loop. Three kinds of edge are dropped with a warning: one
        naming a source the graph does not have, one that would close a cycle,
        and one pointing at the decision itself.

        :param decision: The decision to insert.
        :return: A new graph containing it.
        """
        kept: list[cdecis.Edge] = []
        for edge in decision.depends_on:
            if edge.source_id == decision.id:
                _LOG.warning("Dropping self-edge on %s", decision.id)
                continue
            if edge.source_id not in self.nodes:
                _LOG.warning(
                    "Dropping %s edge on %s: unknown source %s",
                    edge.kind,
                    decision.id,
                    edge.source_id,
                )
                continue
            if self._would_cycle(decision.id, edge.source_id):
                _LOG.warning(
                    "Dropping %s edge %s -> %s: would close a cycle",
                    edge.kind,
                    edge.source_id,
                    decision.id,
                )
                continue
            kept.append(edge)
        nodes = dict(self.nodes)
        nodes[decision.id] = decision.model_copy(update={"depends_on": tuple(kept)})
        return DecisionGraph(nodes=nodes)

    def replace(self, decision: cdecis.OpenDecision) -> DecisionGraph:
        """
        Swap a decision for a changed copy of itself.

        Edges are taken as given; a decision already in the graph has had them
        validated once, and a transition never adds one.

        :param decision: The decision to write back.
        :return: A new graph containing it.
        :raises UnknownDecisionError: When the decision is not already present.
        """
        if decision.id not in self.nodes:
            raise cerrors.UnknownDecisionError(
                f"Cannot replace {decision.id!r}: not in this graph. Use add() to insert it."
            )
        nodes = dict(self.nodes)
        nodes[decision.id] = decision
        return DecisionGraph(nodes=nodes)

    def cascade(self) -> DecisionGraph:
        """
        Apply every edge whose source has now resolved.

        One pass, and it is where an answer earns its keep: a ``prunes`` edge
        deletes a decision outright, a ``constrains`` edge promotes one to
        mandatory. ``informs`` is not applied here because it changes no status —
        it shares the source's evidence onto its target, which needs the evidence
        graph and so lives in
        :func:`crux.application.retrieval.propagate_informs`.

        :return: A new graph with the cascade applied.
        """
        nodes = dict(self.nodes)
        for decision_id, node in self.nodes.items():
            if not node.is_open:
                continue
            if any(edge.fires(self.value_of(edge.source_id)) for edge in node.edges("prunes")):
                nodes[decision_id] = node.prune()
                continue
            if not node.mandatory and any(
                edge.fires(self.value_of(edge.source_id)) for edge in node.edges("constrains")
            ):
                nodes[decision_id] = node.make_mandatory()
        return DecisionGraph(nodes=nodes)

    # ## Internals

    def _would_cycle(self, new_id: str, source_id: str) -> bool:
        """
        Say whether adding an edge from a source to a new decision closes a loop.

        Walks up from the source through its own edges, bounded by
        ``_MAX_CHAIN_DEPTH``. Bounded rather than exhaustive because a legitimate
        dependency chain is two or three hops, and an unbounded walk here would
        be the traversal this module exists to avoid.

        :param new_id: The decision the edge points at.
        :param source_id: The decision the edge comes from.
        :return: Whether the walk reaches ``new_id``.
        """
        seen: set[str] = set()
        frontier = [source_id]
        for _ in range(_MAX_CHAIN_DEPTH):
            nxt: list[str] = []
            for node_id in frontier:
                if node_id == new_id:
                    return True
                if node_id in seen:
                    continue
                seen.add(node_id)
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                nxt.extend(edge.source_id for edge in node.depends_on)
            if not nxt:
                return False
            frontier = nxt
        return False
