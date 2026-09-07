"""
Tier-1 tests for the decision graph.

These are the tests that would fail if the graph stopped earning its keep: birth
gating, pruning, and the ranking that decides which question gets asked first.
"""

from __future__ import annotations

import pytest

import crux.domain.decisions as cdecis
import crux.domain.graph as cgraph
import crux.errors as cerrors
import tests.conftest as factories


def _graph(*decisions: cdecis.OpenDecision) -> cgraph.DecisionGraph:
    """
    Build a graph by adding decisions in order.

    :param decisions: The decisions, sources before their dependents.
    :return: The graph.
    """
    graph = cgraph.DecisionGraph()
    for decision in decisions:
        graph = graph.add(decision)
    return graph


class TestLiveness:
    """
    A `requires` edge means the target does not exist yet.
    """

    def test_a_required_decision_is_not_live_until_its_source_resolves(self) -> None:
        """
        Test that an unborn decision stays out of the open set. This is what stops
        crux asking "immediate or daily digest?" of someone who chose in-app only.
        """
        channel = factories.build_decision(id="channel")
        digest = factories.build_decision(
            id="digest",
            depends_on=(factories.build_edge(source_id="channel", when_value="email"),),
        )
        graph = _graph(channel, digest)

        assert [n.id for n in graph.open_decisions()] == ["channel"]

    def test_resolving_the_source_the_right_way_births_the_dependent(self) -> None:
        """
        Test that answering "email" brings the digest decision into existence.
        The question set cannot be computed up front precisely because of this.
        """
        channel = factories.build_decision(id="channel")
        digest = factories.build_decision(
            id="digest",
            depends_on=(factories.build_edge(source_id="channel", when_value="email"),),
        )
        graph = _graph(channel, digest)

        graph = graph.replace(channel.resolve("email", source="respondent"))

        assert [n.id for n in graph.open_decisions()] == ["digest"]

    def test_resolving_the_source_another_way_leaves_it_unborn(self) -> None:
        """
        Test that "in-app" never births the email-only decision, so it is neither
        asked nor carried into the compiled prompt as an assumption.
        """
        channel = factories.build_decision(id="channel")
        digest = factories.build_decision(
            id="digest",
            depends_on=(factories.build_edge(source_id="channel", when_value="email"),),
        )
        graph = _graph(channel, digest)

        graph = graph.replace(channel.resolve("in-app", source="respondent"))

        assert graph.open_decisions() == ()


class TestCascade:
    """
    What one answer does to everything downstream.
    """

    def test_one_answer_prunes_a_whole_branch(self) -> None:
        """
        Test that answering "MVP" deletes retry, dead-letter and rate-limit
        decisions outright. Pruning is the single clearest thing the graph buys:
        three decisions that are never asked and never assumed.
        """
        scope = factories.build_decision(id="scope")
        killed = [
            factories.build_decision(
                id=name,
                depends_on=(cdecis.Edge(kind="prunes", source_id="scope", when_value="mvp"),),
            )
            for name in ("retry", "dlq", "ratelimit")
        ]
        graph = _graph(scope, *killed)

        graph = graph.replace(scope.resolve("mvp", source="respondent")).cascade()

        assert [n.id for n in graph.open_decisions()] == []
        assert all(graph.get(name).status == "pruned" for name in ("retry", "dlq", "ratelimit"))

    def test_a_prune_that_does_not_fire_leaves_the_branch_alone(self) -> None:
        """
        Test that answering "production" keeps retry open, since a prune edge is
        conditional on its value and not on the source merely resolving.
        """
        scope = factories.build_decision(id="scope")
        retry = factories.build_decision(
            id="retry",
            depends_on=(cdecis.Edge(kind="prunes", source_id="scope", when_value="mvp"),),
        )
        graph = _graph(scope, retry)

        graph = graph.replace(scope.resolve("production", source="respondent")).cascade()

        assert graph.get("retry").is_open

    def test_a_constrains_edge_promotes_a_cheap_decision_to_mandatory(self) -> None:
        """
        Test that choosing email forces the opt-out question. Opt-out is cheap to
        change and legally required, so it has to reach a human by a route that
        the cost threshold alone would never take.
        """
        channel = factories.build_decision(id="channel")
        optout = factories.build_decision(
            id="optout",
            cost_if_wrong="low",
            reversibility="easy",
            depends_on=(cdecis.Edge(kind="constrains", source_id="channel", when_value="email"),),
        )
        graph = _graph(channel, optout)
        assert graph.get("optout").is_ask_worthy is False

        graph = graph.replace(channel.resolve("email", source="respondent")).cascade()

        assert graph.get("optout").mandatory is True
        assert graph.get("optout").is_ask_worthy is True

    def test_cascade_leaves_already_resolved_decisions_alone(self) -> None:
        """
        Test that a prune arriving after an answer does not overwrite it, so a
        respondent's answer is never silently discarded.
        """
        scope = factories.build_decision(id="scope")
        retry = factories.build_decision(
            id="retry",
            depends_on=(cdecis.Edge(kind="prunes", source_id="scope", when_value="mvp"),),
        )
        graph = _graph(scope, retry.resolve("3 attempts", source="respondent"))

        graph = graph.replace(scope.resolve("mvp", source="respondent")).cascade()

        assert graph.get("retry").is_resolved
        assert graph.get("retry").value == "3 attempts"


class TestFrontier:
    """
    Which decisions can be asked now, and in what order.
    """

    def test_the_frontier_excludes_cheap_and_reversible_decisions(self) -> None:
        """
        Test that a decision below the ask threshold never reaches a respondent
        even when it is open and live. Those become assumptions instead.
        """
        graph = _graph(
            factories.build_decision(id="expensive"),
            factories.build_decision(id="cheap", cost_if_wrong="low", reversibility="easy"),
        )
        assert [n.id for n in graph.frontier()] == ["expensive"]

    def test_the_frontier_excludes_facts_about_the_world(self) -> None:
        """
        Test that a high-cost `missing_context` decision is still not asked. Cost
        does not override type; retrieval or a retype has to settle it.
        """
        graph = _graph(factories.build_decision(id="infra", type="missing_context"))
        assert graph.frontier() == ()

    def test_an_unborn_decision_never_reaches_the_frontier(self) -> None:
        """
        Test that liveness gates asking as well as the open set.
        """
        graph = _graph(
            factories.build_decision(id="channel"),
            factories.build_decision(
                id="digest",
                depends_on=(factories.build_edge(source_id="channel", when_value="email"),),
            ),
        )
        assert [n.id for n in graph.frontier()] == ["channel"]

    def test_a_pruned_decision_never_reaches_the_frontier(self) -> None:
        """
        Test the invariant named in CLAUDE.md: a decision an answer killed is
        never put to anyone. Regression guard for asking a dead question.
        """
        graph = _graph(factories.build_decision(id="retry"))
        graph = graph.replace(graph.get("retry").prune())
        assert graph.frontier() == ()

    def test_the_frontier_ranks_by_cost_then_reversibility_then_fan_out(self) -> None:
        """
        Test that the decision unlocking the most downstream work is asked first
        among equals. Answering a high-fan-out decision collapses the most
        uncertainty per interruption, which is the only lever on question count
        the ranking has.
        """
        graph = _graph(
            factories.build_decision(id="lonely"),
            factories.build_decision(id="hub"),
            factories.build_decision(id="medium", cost_if_wrong="medium", mandatory=True),
        )
        graph = graph.add(
            factories.build_decision(
                id="child",
                depends_on=(factories.build_edge(source_id="hub", when_value="x"),),
            )
        )

        assert [n.id for n in graph.frontier()] == ["hub", "lonely", "medium"]


class TestEdgeValidation:
    """
    Bad edges are dropped at insert, so the read path stays a flat loop.
    """

    def test_an_edge_naming_an_unknown_source_is_dropped(self) -> None:
        """
        Test that a model hallucinating a dependency cannot strand a decision.
        Keeping the edge would leave the node permanently unborn and invisible.
        """
        graph = _graph(
            factories.build_decision(
                id="digest",
                depends_on=(factories.build_edge(source_id="nonexistent"),),
            )
        )
        assert graph.get("digest").depends_on == ()
        assert [n.id for n in graph.open_decisions()] == ["digest"]

    def test_a_self_edge_is_dropped(self) -> None:
        """
        Test that a decision cannot depend on itself, which would make it
        permanently unborn.
        """
        graph = _graph(
            factories.build_decision(id="d1", depends_on=(factories.build_edge(source_id="d1"),))
        )
        assert graph.get("d1").depends_on == ()

    def test_an_edge_closing_a_cycle_is_dropped(self) -> None:
        """
        Test that A -> B -> A is rejected at insert. Cycle safety lives here
        rather than in a traversal at read time, which is what keeps the frontier
        a single pass.
        """
        first = factories.build_decision(id="a")
        second = factories.build_decision(id="b", depends_on=(factories.build_edge(source_id="a"),))
        graph = _graph(first, second)

        graph = graph.add(
            factories.build_decision(id="a2", depends_on=(factories.build_edge(source_id="b"),))
        )
        cyclic = factories.build_decision(
            id="b", depends_on=(factories.build_edge(source_id="a2"),)
        )
        graph = graph.add(cyclic)

        assert graph.get("b").depends_on == ()

    def test_replacing_an_absent_decision_is_an_error(self) -> None:
        """
        Test that replace() refuses to insert. Silently adding would let a typo
        in a decision id create a second, orphaned copy of the same decision.
        """
        with pytest.raises(cerrors.UnknownDecisionError, match="Use add"):
            cgraph.DecisionGraph().replace(factories.build_decision())

    def test_getting_an_absent_decision_names_what_is_there(self) -> None:
        """
        Test that the error says what to do about it, per the errors rule.
        """
        with pytest.raises(cerrors.UnknownDecisionError, match="known ids"):
            _graph(factories.build_decision(id="a")).get("b")


class TestSerialisation:
    """
    The graph goes through a host's JSON storage and comes back unchanged.
    """

    def test_a_graph_round_trips_through_json(self) -> None:
        """
        Test the serialisation invariant at the graph level: yield/resume across
        a process boundary is the whole reason the session holds no ports, and it
        only works if the graph itself survives the trip.
        """
        graph = _graph(
            factories.build_decision(
                id="channel", options=(cdecis.Option(value="email", source="pack"),)
            ),
            factories.build_decision(
                id="digest",
                depends_on=(factories.build_edge(source_id="channel", when_value="email"),),
            ),
        )
        graph = graph.replace(graph.get("channel").resolve("email", source="respondent"))

        restored = cgraph.DecisionGraph.model_validate_json(graph.model_dump_json())

        assert restored == graph
        assert [n.id for n in restored.open_decisions()] == ["digest"]


class TestInformsAndMeasurement:
    """
    The edge that shares evidence, and the counter that judges all of them.
    """

    def test_edge_stats_separate_attached_edges_from_firing_ones(self) -> None:
        """
        Test the number the graph's existence rests on. An edge that is attached
        but never fires is decoration — it costs tokens to generate and can hide
        a question if wrong — so the experiment needs to tell the two apart.
        """
        scope = factories.build_decision(id="scope")
        graph = _graph(
            scope,
            factories.build_decision(
                id="retry",
                depends_on=(cdecis.Edge(kind="prunes", source_id="scope", when_value="mvp"),),
            ),
            factories.build_decision(
                id="queue",
                depends_on=(
                    cdecis.Edge(kind="prunes", source_id="scope", when_value="production"),
                ),
            ),
        )

        assert graph.edge_stats() == {"prunes": (2, 0)}

        graph = graph.replace(scope.resolve("mvp", source="respondent"))

        assert graph.edge_stats() == {"prunes": (2, 1)}

    def test_an_informs_edge_never_gates_its_target(self) -> None:
        """
        Test that `informs` is unconditional in the one way that matters: unlike
        `requires`, an unresolved source leaves the target perfectly alive. It
        shares evidence; it does not decide anything.
        """
        graph = _graph(
            factories.build_decision(id="infra"),
            factories.build_decision(
                id="channel",
                depends_on=(cdecis.Edge(kind="informs", source_id="infra"),),
            ),
        )

        assert [n.id for n in graph.open_decisions()] == ["infra", "channel"]
        assert [n.id for n in graph.frontier()] == ["infra", "channel"]
