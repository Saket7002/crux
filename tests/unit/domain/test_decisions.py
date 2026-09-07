"""
Tier-1 tests for the open-decision primitive.
"""

from __future__ import annotations

import pytest

import crux.domain.decisions as cdecis
import tests.conftest as factories


class TestGradeDerivation:
    """
    The grade is derived from the source, never judged by a model.
    """

    @pytest.mark.parametrize(
        ("source", "dominant", "expected"),
        [
            ("respondent", False, "confirmed"),
            ("respondent", True, "confirmed"),
            ("retrieval", True, "inferred"),
            ("retrieval", False, "assumed"),
            ("default", False, "assumed"),
            ("delegated", False, "assumed"),
        ],
    )
    def test_grade_follows_from_source_and_dominance(
        self,
        source: cdecis.ResolutionSource,
        dominant: bool,
        expected: cdecis.Grade,
    ) -> None:
        """
        Test that every source maps to exactly one grade, and that only a
        retrieval's dominance changes the answer. This is what stops the grade
        drifting from reality: a model asked how confident it is will always
        answer, and the answer means nothing.
        """
        assert cdecis.derive_grade(source, dominant=dominant) == expected

    def test_only_a_respondent_answer_is_not_an_assumption(self) -> None:
        """
        Test that everything crux closed by itself reports as an assumption.
        The output's honesty block is a filter on this property, so if it were
        wrong a guess would render as a confirmed fact.
        """
        confirmed = cdecis.Resolution(value="v", source="respondent", grade="confirmed")
        assert confirmed.is_assumption is False
        for source in ("retrieval", "default", "delegated"):
            resolution = cdecis.Resolution(value="v", source=source, grade="assumed")
            assert resolution.is_assumption is True


class TestAskThreshold:
    """
    What earns a respondent's attention.
    """

    @pytest.mark.parametrize(
        ("cost", "reversibility", "expected"),
        [
            ("high", "hard", True),
            ("high", "easy", False),
            ("medium", "hard", False),
            ("low", "easy", False),
        ],
    )
    def test_only_expensive_and_irreversible_clears_the_bar(
        self,
        cost: cdecis.Cost,
        reversibility: cdecis.Reversibility,
        expected: bool,
    ) -> None:
        """
        Test that cost alone does not earn a question. An expensive mistake that
        takes an afternoon to correct is cheaper than the interruption, which is
        the whole reason this is two variables and not one.
        """
        decision = factories.build_decision(cost_if_wrong=cost, reversibility=reversibility)
        assert decision.is_expensive_and_irreversible is expected

    def test_mandatory_overrides_a_cheap_decision(self) -> None:
        """
        Test that a `constrains` edge can force a question crux would otherwise
        default away. Email opt-out is cheap to change and legally required, so
        the threshold must have an override.
        """
        cheap = factories.build_decision(cost_if_wrong="low", reversibility="easy")
        assert cheap.is_ask_worthy is False
        assert cheap.make_mandatory().is_ask_worthy is True

    def test_missing_context_is_never_askable(self) -> None:
        """
        Test that a fact about the world cannot be routed to a human. Asking
        someone to read their own codebase aloud is the failure mode the whole
        design exists to avoid.
        """
        fact = factories.build_decision(type="missing_context")
        assert fact.is_askable_type is False


class TestTransitions:
    """
    Every transition returns a new decision.
    """

    def test_resolving_records_source_and_derived_grade(self) -> None:
        """
        Test that resolve() sets the grade from the source rather than taking one.
        """
        resolved = factories.build_decision().resolve(
            "in-app + email", source="retrieval", dominant=True, evidence=("e1",)
        )
        assert resolved.is_resolved
        assert resolved.value == "in-app + email"
        assert resolved.resolution is not None
        assert resolved.resolution.grade == "inferred"
        assert resolved.resolution.evidence == ("e1",)

    def test_retyping_remembers_the_original_type(self) -> None:
        """
        Test that a retyped decision keeps its provenance. A `missing_context`
        node retrieval could not settle becomes askable, and the audit trail has
        to show that crux changed the question's nature rather than inventing it.
        """
        retyped = factories.build_decision(type="missing_context").retype("underspecification")
        assert retyped.type == "underspecification"
        assert retyped.retyped_from == "missing_context"
        assert retyped.is_askable_type is True

    def test_retyping_twice_keeps_the_first_type(self) -> None:
        """
        Test that provenance survives a second retype, so the trail points at
        where the decision started rather than at its last hop.
        """
        once = factories.build_decision(type="missing_context").retype("underspecification")
        twice = once.retype("ambiguity")
        assert twice.retyped_from == "missing_context"

    def test_a_resolved_decision_is_no_longer_open(self) -> None:
        """
        Test that status and resolution stay in step, since the frontier reads
        one and the compiler reads the other.
        """
        decision = factories.build_decision()
        assert decision.is_open
        assert decision.resolve("v", source="default").is_open is False

    def test_transitions_do_not_mutate_the_original(self) -> None:
        """
        Test that the graph really is mutable-by-copy. A caller holding an older
        decision must still see what it saw, because resume() runs against a
        session a host may have deserialized from anywhere.
        """
        original = factories.build_decision()
        original.resolve("v", source="default")
        assert original.is_open
        assert original.resolution is None


class TestEdges:
    """
    Edges fire, or they do not.
    """

    def test_conditional_edge_without_a_value_is_rejected(self) -> None:
        """
        Test that a `requires` edge with no trigger value fails validation. Such
        an edge would gate a decision on a condition that can never be checked,
        so the node would never be born.
        """
        with pytest.raises(ValueError, match="when_value"):
            cdecis.Edge(kind="requires", source_id="channel")

    def test_informs_needs_no_value_and_fires_on_any_resolution(self) -> None:
        """
        Test that `informs` is unconditional: the source's evidence reaches its
        target whenever the source resolves, whatever it resolved to. Unlike
        `requires`, there is no value to match against.
        """
        edge = cdecis.Edge(kind="informs", source_id="infra")
        assert edge.fires("anything") is True
        assert edge.fires(None) is False

    def test_an_edge_does_not_fire_while_its_source_is_open(self) -> None:
        """
        Test that an unresolved source leaves its dependents alone, which is what
        keeps unborn decisions unborn.
        """
        assert factories.build_edge(when_value="email").fires(None) is False

    def test_an_edge_fires_only_on_its_own_value(self) -> None:
        """
        Test that `requires(channel, "email")` stays quiet when the answer was
        in-app. This is the check that stops crux asking about digests.
        """
        edge = factories.build_edge(when_value="email")
        assert edge.fires("email") is True
        assert edge.fires("in-app") is False
