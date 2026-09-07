"""
Tier-1 tests for the compiled prompt.

The output is the product, and two of its properties are load-bearing: the
assumptions block is derived rather than written, and the rendered prompt stays
readable however many decisions the expander found.
"""

from __future__ import annotations

import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.graph as cgraph
import crux.domain.output as coutput
import tests.conftest as factories


def _resolved(
    decision_id: str,
    *,
    source: cdecis.ResolutionSource,
    cost: cdecis.Cost = "medium",
    reversibility: cdecis.Reversibility = "easy",
    dominant: bool = False,
) -> cdecis.OpenDecision:
    """
    Build a resolved decision.

    :param decision_id: Its id, also used as its text.
    :param source: What closed it.
    :param cost: Blast radius.
    :param reversibility: How hard to undo.
    :param dominant: Whether retrieval had a clear winner.
    :return: The decision.
    """
    return factories.build_decision(
        id=decision_id,
        undecided=decision_id,
        cost_if_wrong=cost,
        reversibility=reversibility,
    ).resolve("v", source=source, dominant=dominant)


def _assemble(*decisions: cdecis.OpenDecision) -> coutput.CompiledPrompt:
    """
    Assemble a compiled prompt from decisions.

    :param decisions: What to include.
    :return: The compiled prompt.
    """
    graph = cgraph.DecisionGraph()
    for decision in decisions:
        graph = graph.add(decision)
    return coutput.CompiledPrompt.assemble(
        task="do the thing",
        constraints=(),
        graph=graph,
        evidence=cevid.EvidenceGraph(),
    )


class TestHonesty:
    """
    Where each resolution lands, and who decided that.
    """

    def test_the_sections_are_derived_from_the_grade_not_written(self) -> None:
        """
        Test that a decision cannot appear under a stronger heading than its
        grade allows. The model writes task prose only; if it could place its own
        guesses under "Decided" the whole output would be worthless.
        """
        compiled = _assemble(
            _resolved("answered", source="respondent"),
            _resolved("looked-up", source="retrieval", dominant=True),
            _resolved("guessed", source="default"),
            _resolved("handed-over", source="delegated"),
        )

        assert [r.id for r in compiled.decided] == ["answered", "looked-up"]
        assert [r.id for r in compiled.assumptions] == ["guessed"]
        assert [r.id for r in compiled.delegations] == ["handed-over"]

    def test_weak_retrieval_is_an_assumption_not_a_decision(self) -> None:
        """
        Test that evidence which did not clearly point one way is reported as a
        guess. `[retrieved]` has to mean "found it", never "the adjudicator
        shrugged", or the reader trusts a tag that has not earned it.
        """
        compiled = _assemble(_resolved("split", source="retrieval", dominant=False))

        assert [r.id for r in compiled.assumptions] == ["split"]
        assert compiled.decided == ()

    def test_every_resolved_decision_appears_somewhere(self) -> None:
        """
        Test the invariant the whole design exists to deliver: nothing crux
        settled is dropped silently.
        """
        sources: tuple[cdecis.ResolutionSource, ...] = (
            "respondent",
            "retrieval",
            "default",
            "delegated",
        )
        compiled = _assemble(
            *(_resolved(f"d{i}", source=source) for i, source in enumerate(sources))
        )

        placed = {r.id for r in compiled.decided + compiled.assumptions + compiled.delegations}
        assert placed == {r.id for r in compiled.all_decisions}


class TestRenderedLength:
    """
    The prompt stays readable however much the expander found.
    """

    def test_a_long_delegation_list_is_capped_and_ranked(self) -> None:
        """
        Test that the rendered prompt does not become a decision dump.

        This test exists because the first live run against a real model produced
        twenty-two open judgements in one list, which a downstream agent is no
        better off reading than the original prompt. The high-stakes ones are
        shown; the tail is counted.
        """
        decisions = [_resolved(f"cheap-{i}", source="delegated", cost="low") for i in range(20)] + [
            _resolved("expensive", source="delegated", cost="high", reversibility="hard")
        ]

        compiled = _assemble(*decisions)
        rendered = compiled.render()

        assert len(compiled.delegations) == 21, "nothing may be dropped from the record"
        section = rendered.split("## Left to you\n")[1]
        listed = [line for line in section.splitlines() if line.startswith("- ")]
        assert len(listed) == coutput.MAX_RENDERED_DELEGATIONS + 1
        assert "expensive" in listed[0], "the costliest judgement must be shown first"
        assert "13 lower-stakes judgements" in listed[-1]

    def test_a_short_list_is_rendered_whole_with_no_summary_line(self) -> None:
        """
        Test that the cap does not fire when it is not needed, so the common case
        reads cleanly.
        """
        compiled = _assemble(
            _resolved("one", source="delegated"), _resolved("two", source="delegated")
        )

        rendered = compiled.render()

        assert "lower-stakes judgements" not in rendered
        assert "- one" in rendered
        assert "- two" in rendered

    def test_a_delegation_carries_the_options_that_were_considered(self) -> None:
        """
        Test that the downstream agent is told the option space rather than
        having to rediscover it.
        """
        decision = factories.build_decision(
            id="strategy",
            undecided="which throttling strategy",
            options=(
                cdecis.Option(value="token bucket", source="expansion"),
                cdecis.Option(value="sliding window", source="expansion"),
            ),
        ).resolve("v", source="delegated")

        rendered = _assemble(decision).render()

        assert "considered: token bucket, sliding window" in rendered


class TestNothingVanishes:
    """
    The strongest form of the honesty invariant.
    """

    def test_a_decision_that_was_never_born_is_still_reported(self) -> None:
        """
        Test that a decision suppressed by an unfired `requires` edge appears in
        the output as out of scope, with the condition that did not hold.

        This test exists because those decisions used to disappear completely:
        not asked, not assumed, not delegated, not listed. A live run attached
        121 edges waiting on values their source could never take, each one
        suppressed a decision, and the missing questions read as the graph
        pruning well. It was deleting them. The older honesty test could not see
        it, because it compared resolved decisions against the output and an
        unborn decision is in neither.
        """
        channel = factories.build_decision(
            id="channel",
            undecided="which channel",
            options=(cdecis.Option(value="in-app", source="pack"),),
        ).resolve("in-app", source="respondent")
        digest = factories.build_decision(
            id="digest",
            undecided="digest cadence",
            depends_on=(cdecis.Edge(kind="requires", source_id="channel", when_value="email"),),
        )
        graph = cgraph.DecisionGraph().add(channel).add(digest)

        compiled = coutput.CompiledPrompt.assemble(
            task="t", constraints=(), graph=graph, evidence=cevid.EvidenceGraph()
        )

        scope = " ".join(compiled.out_of_scope)
        assert "digest cadence" in scope
        assert "in-app" in scope, "the reason it never applied must be named"
        assert "digest cadence" in compiled.render()

    def test_an_edge_waiting_on_an_impossible_value_is_visible(self) -> None:
        """
        Test that a nonsense edge reads as a bug report rather than as silence.

        An edge whose trigger is not among its source's options can never fire,
        so its target is suppressed for ever. Naming the condition in the output
        is what turns that from invisible data loss into something a reader
        notices.
        """
        source = factories.build_decision(
            id="src",
            undecided="which channel",
            options=(cdecis.Option(value="email", source="expansion"),),
        ).resolve("email", source="respondent")
        gated = factories.build_decision(
            id="gated",
            undecided="pigeon loft location",
            depends_on=(
                cdecis.Edge(kind="requires", source_id="src", when_value="carrier pigeon"),
            ),
        )
        graph = cgraph.DecisionGraph().add(source).add(gated)

        compiled = coutput.CompiledPrompt.assemble(
            task="t", constraints=(), graph=graph, evidence=cevid.EvidenceGraph()
        )

        scope = " ".join(compiled.out_of_scope)
        assert "pigeon loft location" in scope
        assert "carrier pigeon" in scope
        assert "email" in scope

    def test_a_legitimately_pruned_decision_still_reads_cleanly(self) -> None:
        """
        Test that the fix did not turn ordinary pruning into noise. A decision an
        answer deliberately killed says so plainly, with no condition clause.
        """
        graph = cgraph.DecisionGraph().add(
            factories.build_decision(id="retry", undecided="retry policy").prune()
        )

        compiled = coutput.CompiledPrompt.assemble(
            task="t", constraints=(), graph=graph, evidence=cevid.EvidenceGraph()
        )

        assert compiled.out_of_scope == ("retry policy",)
