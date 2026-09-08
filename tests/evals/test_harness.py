"""
Tests for the measuring instruments themselves.

Not marked `live`: a harness whose arithmetic is wrong produces a confident
number nobody checks, and the experiment's whole value is that its verdict can be
trusted. These run in CI with everything else.
"""

from __future__ import annotations

import pytest

import crux.domain.decisions as cdecis
import crux.domain.graph as cgraph
import crux.domain.session as csessn
import tests.evals.harness as harness
import tests.evals.saturation as saturation


class TestCorpus:
    """
    The corpus loads and says what it means.
    """

    def test_every_case_parses_and_names_a_real_fixture(self) -> None:
        """
        Test that no case silently refers to a fixture repo that is not there. A
        missing fixture makes retrieval find nothing, which looks like a recall
        regression rather than a broken case.
        """
        cases = harness.load_corpus()
        assert cases
        for case in cases:
            assert case.prompt.strip()
            if case.fixture_repo:
                assert case.root is not None and case.root.is_dir(), case.id

    def test_the_corpus_covers_all_three_strata(self) -> None:
        """
        Test that the saturation curve can actually be broken out per stratum.
        The prediction worth testing is that one-liners saturate earlier than
        project briefs, and a corpus missing a stratum cannot test it.
        """
        strata = {case.stratum for case in harness.load_corpus()}
        assert strata == {"one_liner", "feature", "project"}

    def test_the_strata_are_balanced_and_ids_unique(self) -> None:
        """
        Test that the ask-rate bands compare like with like. A stratum with two
        cases and one with twenty would make the per-stratum mean a coin toss,
        and a duplicated id would let one case silently overwrite another in a
        platform dataset keyed on it.
        """
        cases = harness.load_corpus()
        ids = [case.id for case in cases]
        assert len(ids) == len(set(ids))
        per_stratum = {
            s: sum(1 for c in cases if c.stratum == s) for s in {c.stratum for c in cases}
        }
        assert len(set(per_stratum.values())) == 1, per_stratum

    def test_every_case_carries_a_rubric_with_no_blank_bullet(self) -> None:
        """
        Test that the judge has something to judge on every case. A case without
        a rubric skips the judge score, which reads as a perfect result in a
        platform dashboard rather than as a gap in the corpus.
        """
        for case in harness.load_corpus():
            assert case.expected is not None, case.id
            bullets = case.expected.bullets()
            assert bullets, case.id
            assert all(bullet.strip() for _, bullet in bullets), case.id

    def test_some_cases_run_without_a_fixture(self) -> None:
        """
        Test that headless-without-retrieval is measured. Every host does not
        hand crux a repo, and a corpus where every case has one never exercises
        the path where retrieval cannot resolve anything.
        """
        assert any(case.fixture_repo is None for case in harness.load_corpus())


class TestMatching:
    """
    How an expectation recognises a decision.
    """

    def test_a_canonical_id_matches_exactly(self) -> None:
        """
        Test that pack decisions match on id, so the common case is exact and
        never fuzzy.
        """
        expectation = harness.Expectation(id="software.scope.build")
        assert expectation.matched_by("software.scope.build", "anything")
        assert not expectation.matched_by("software.tests.expectations", "anything")

    def test_a_freeform_expectation_matches_on_wording(self) -> None:
        """
        Test that a decision worded differently still counts. The expander
        invents its own phrasing, so an exact-match corpus would measure
        wording rather than recall.
        """
        expectation = harness.Expectation(match="how callers are identified for counting")
        assert expectation.matched_by("open.p1.x", "How requests are identified for counting")
        assert not expectation.matched_by("open.p1.y", "Which store holds the counters")
        assert not expectation.matched_by("open.p1.z", "Whether breaking callers is acceptable")


class TestScoring:
    """
    Turning a session into a number.
    """

    def test_recall_counts_surfaced_expectations(self) -> None:
        """
        Test the core arithmetic: one of two expectations found is 0.5.
        """
        case = harness.EvalCase(
            id="c",
            prompt="p",
            must_surface=(
                harness.Expectation(id="software.scope.build"),
                harness.Expectation(id="nowhere"),
            ),
        )
        graph = csessn.Session(id="s", prompt="p").graph.add(
            cdecis.OpenDecision(
                id="software.scope.build",
                undecided="scope",
                type="underspecification",
                cost_if_wrong="high",
                reversibility="hard",
                origin=cdecis.Origin(seeded_by="pack"),
            )
        )
        session = csessn.Session(id="s", prompt="p", graph=graph)

        score = harness.score_case(case, session)

        assert score.recall == 0.5
        assert score.missed == ("nowhere",)

    def test_a_case_over_its_question_budget_is_flagged(self) -> None:
        """
        Test that asking too much is caught. Ask rate is the product metric, and
        a case quietly exceeding its budget is the regression that matters most.
        """
        case = harness.EvalCase(id="c", prompt="p", max_questions=2)
        session = csessn.Session(id="s", prompt="p", budget=csessn.Budget(spent_questions=3))

        assert harness.score_case(case, session).over_question_budget


class TestExperimentArithmetic:
    """
    The numbers the verdict is computed from.
    """

    @pytest.mark.parametrize(
        ("ratios", "expected"),
        [
            ([0.05], 1),
            ([1.0, 0.05], 2),
            ([1.0, 0.5, 0.09], 3),
            ([1.0, 1.0, 1.0], 3),
            ([], 0),
        ],
    )
    def test_k_star_is_the_first_pass_below_the_threshold(
        self, ratios: list[float], expected: int
    ) -> None:
        """
        Test the saturation point, including the case where it never saturates —
        which must report the pass count rather than crashing or reporting zero.
        """
        assert saturation.k_star(ratios) == expected

    def test_firing_density_ignores_informs(self) -> None:
        """
        Test that evidence sharing is excluded from the number the decision rule
        reads. `informs` neither gates nor deletes anything, so counting it would
        flatter the case for keeping the conditional-edge machinery.
        """
        stats = {"requires": (4, 1), "prunes": (2, 1), "informs": (10, 10)}
        assert saturation.firing_density(stats) == pytest.approx(2 / 6)

    def test_firing_density_of_nothing_is_zero_not_an_error(self) -> None:
        """
        Test the empty case, which is the *expected* result if the graph turns
        out to be ceremony — the experiment must survive its own null hypothesis.
        """
        assert saturation.firing_density({}) == 0.0

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ({"a", "b"}, {"a", "b"}, 1.0),
            ({"a"}, {"b"}, 0.0),
            ({"a", "b"}, {"a"}, 0.5),
            (set(), set(), 1.0),
        ],
    )
    def test_jaccard_measures_agreement(
        self, left: set[str], right: set[str], expected: float
    ) -> None:
        """
        Test the noise-floor comparator, including two empty samples, which agree
        perfectly rather than being undefined.
        """
        assert saturation.jaccard(frozenset(left), frozenset(right)) == expected


class TestVerdict:
    """
    The decision rule, committed before the run.
    """

    def _measurement(
        self, *, ratios: list[float], stats: dict[str, tuple[int, int]]
    ) -> saturation.CaseMeasurement:
        return saturation.CaseMeasurement(
            case_id="c",
            stratum="feature",
            ratios=ratios,
            new_per_pass=[],
            noise_floor=None,
            edge_stats=stats,
            conditional_edges=0,
            reachable_edges=0,
            questions_with_graph=2,
            questions_flat=2,
            assumptions_with_graph=0,
            assumptions_flat=0,
            decisions_total=0,
        )

    def test_early_saturation_and_dead_edges_condemn_the_graph(self) -> None:
        """
        Test that the rule is genuinely capable of saying "delete this". A
        criterion that cannot return the inconvenient answer is not a criterion.
        """
        result = saturation.verdict([self._measurement(ratios=[0.05], stats={"requires": (20, 0)})])
        assert "CEREMONY" in result

    def test_late_saturation_or_live_edges_keep_it(self) -> None:
        """
        Test the other side of the rule fires on either condition, not both.
        """
        result = saturation.verdict(
            [self._measurement(ratios=[1.0, 0.5, 0.05], stats={"prunes": (10, 4)})]
        )
        assert "EARNS ITS KEEP" in result

    def test_the_middle_keeps_prunes_only(self) -> None:
        """
        Test the in-between call, which is the most likely real outcome.
        """
        # Mean k* of 1.5 clears the ceremony floor without reaching the bar that
        # keeps the whole machinery, and firing sits between the two thresholds.
        result = saturation.verdict(
            [
                self._measurement(ratios=[0.05], stats={"requires": (10, 1)}),
                self._measurement(ratios=[1.0, 0.05], stats={"requires": (10, 1)}),
            ]
        )
        assert "PARTIAL" in result

    def test_replay_mode_says_the_verdict_is_not_yet_trustworthy(self) -> None:
        """
        Test that a run without a noise floor says so loudly. Without it a "new"
        decision on pass two may be the model sampling differently rather than
        finding anything deeper, and k* means nothing.
        """
        rendered = saturation.render(
            [self._measurement(ratios=[1.0, 0.05], stats={"requires": (10, 1)})]
        )
        assert "NOISE FLOOR NOT MEASURED" in rendered


class TestEdgeCoherence:
    """
    The edge metric a respondent's answers cannot skew.
    """

    def _graph(self, *, when_value: str, options: tuple[str, ...]) -> cgraph.DecisionGraph:
        """
        Build a two-node graph with one conditional edge.

        :param when_value: What the edge waits for.
        :param options: What the source can actually resolve to.
        :return: The graph.
        """
        source = cdecis.OpenDecision(
            id="src",
            undecided="which channel",
            type="ambiguity",
            cost_if_wrong="high",
            reversibility="hard",
            options=tuple(cdecis.Option(value=v, source="expansion") for v in options),
            origin=cdecis.Origin(seeded_by="expansion"),
        )
        target = cdecis.OpenDecision(
            id="tgt",
            undecided="digest cadence",
            type="underspecification",
            cost_if_wrong="low",
            reversibility="easy",
            depends_on=(cdecis.Edge(kind="requires", source_id="src", when_value=when_value),),
            origin=cdecis.Origin(seeded_by="expansion"),
        )
        return cgraph.DecisionGraph().add(source).add(target)

    def test_an_edge_whose_trigger_is_offered_is_reachable(self) -> None:
        """
        Test the healthy case: the edge waits for a value the source can actually
        take, so some answer would fire it.
        """
        total, reachable = saturation.edge_coherence(
            self._graph(when_value="email", options=("email", "in-app"))
        )

        assert (total, reachable) == (1, 1)

    def test_an_edge_waiting_for_an_impossible_value_is_dead(self) -> None:
        """
        Test the case this metric exists to catch: an edge waiting on a value the
        source can never take is decoration, and no respondent could fire it.

        Firing density alone cannot tell that apart from "nobody happened to
        answer that way", which is why it needs a companion the answers cannot
        move.
        """
        total, reachable = saturation.edge_coherence(
            self._graph(when_value="carrier pigeon", options=("email", "in-app"))
        )

        assert (total, reachable) == (1, 0)

    def test_informs_edges_are_not_counted(self) -> None:
        """
        Test that evidence sharing stays out of the number the verdict reads, the
        same way it stays out of firing density.
        """
        import crux.domain.graph as cgraph

        source = cdecis.OpenDecision(
            id="src",
            undecided="infra",
            type="missing_context",
            cost_if_wrong="low",
            reversibility="easy",
            origin=cdecis.Origin(seeded_by="expansion"),
        )
        target = cdecis.OpenDecision(
            id="tgt",
            undecided="channel",
            type="ambiguity",
            cost_if_wrong="high",
            reversibility="hard",
            depends_on=(cdecis.Edge(kind="informs", source_id="src"),),
            origin=cdecis.Origin(seeded_by="expansion"),
        )
        graph = cgraph.DecisionGraph().add(source).add(target)

        assert saturation.edge_coherence(graph) == (0, 0)
