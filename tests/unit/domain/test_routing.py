"""
Tier-1 tests for the route table.

The table is the heart of crux, so it gets a case per row plus every boundary.
Two properties are checked hardest, because breaking either quietly ruins the
product: the table is **total** (no decision is ever left without a route), and
budget exhaustion blocks **asking** rather than resolving.
"""

from __future__ import annotations

import pytest

import crux.application.routing as aroute
import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import tests.conftest as factories


def _candidates(*pairs: tuple[str, float]) -> tuple[cevid.Candidate, ...]:
    """
    Build candidates from value/score pairs.

    :param pairs: Value and score for each candidate.
    :return: The candidates, each carrying one supporting evidence id.
    """
    return tuple(
        cevid.Candidate(value=value, score=score, support=(f"ev:{value}",))
        for value, score in pairs
    )


CHEAP = {"cost_if_wrong": "low", "reversibility": "easy"}
COSTLY = {"cost_if_wrong": "high", "reversibility": "hard"}


class TestDominance:
    """
    When the evidence has a clear winner.
    """

    @pytest.mark.parametrize(
        ("scores", "expected"),
        [
            ((10.0,), True),
            ((10.0, 5.0), True),
            ((10.0, 5.01), False),
            ((10.0, 9.0), False),
            ((10.0, 0.0), True),
            ((), False),
        ],
        ids=["single", "exactly-2x", "just-under-2x", "close", "zero-runner-up", "empty"],
    )
    def test_dominance_needs_twice_the_support_of_the_runner_up(
        self, scores: tuple[float, ...], expected: bool
    ) -> None:
        """
        Test the exact dominance boundary, including the 2.0 ratio itself. This
        threshold decides whether crux resolves silently or interrupts a human,
        so an off-by-one here is felt directly as an interrogation or a silent
        wrong guess.
        """
        candidates = _candidates(*((f"v{i}", s) for i, s in enumerate(scores)))
        assert cevid.is_dominant(candidates) is expected


class TestMissingContextIsNeverAsked:
    """
    A fact about the world has to be found, defaulted, or retyped.
    """

    def test_one_dominant_candidate_resolves_from_evidence(self) -> None:
        """
        Test the common case: retrieval found the answer, nobody is interrupted.
        """
        outcome = aroute.route(
            factories.build_decision(type="missing_context"),
            candidates=_candidates(("resend", 10.0)),
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "retrieval"
        assert outcome.dominant is True
        assert outcome.value == "resend"

    def test_split_evidence_is_retyped_rather_than_asked(self) -> None:
        """
        Test that two live candidates turn the fact into a question a human can
        answer, carrying the candidates as options. Its own type forbids asking,
        so without the retype it would sit open for ever.
        """
        outcome = aroute.route(
            factories.build_decision(type="missing_context"),
            candidates=_candidates(("User", 10.0), ("OrgMember", 8.0)),
        )
        assert isinstance(outcome, aroute.Retype)
        assert outcome.new_type == "underspecification"
        assert [o.value for o in outcome.options] == ["User", "OrgMember"]

    def test_no_evidence_but_a_pack_default_takes_the_default(self) -> None:
        """
        Test that an authored default beats a retype, so crux does not ask about
        something a pack already answered.
        """
        outcome = aroute.route(
            factories.build_decision(type="missing_context"),
            pack_default="pytest",
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "default"
        assert outcome.value == "pytest"

    def test_no_evidence_and_no_default_is_retyped_not_dropped(self) -> None:
        """
        Test the escape hatch that makes the table total. Without it a
        `missing_context` decision whose retrieval came back empty has no legal
        terminal state at all.
        """
        outcome = aroute.route(factories.build_decision(type="missing_context"))
        assert isinstance(outcome, aroute.Retype)
        assert outcome.new_type == "underspecification"
        assert outcome.options == ()


class TestJudgementRoutes:
    """
    Decisions a human could answer, and whether one is worth interrupting.
    """

    def test_clear_evidence_on_a_cheap_decision_resolves_silently(self) -> None:
        """
        Test that crux does not ask about something it knows and that is cheap to
        get wrong. This is the row that keeps the question count down.
        """
        outcome = aroute.route(
            factories.build_decision(**CHEAP),
            candidates=_candidates(("redis", 10.0)),
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "retrieval"

    def test_clear_evidence_on_a_costly_decision_still_asks_for_confirmation(self) -> None:
        """
        Test that an expensive, irreversible decision is confirmed even when the
        evidence is unambiguous, with the evidence pre-filled as the
        recommendation. Confirming is one tap; being wrong about a breaking
        change is not.
        """
        outcome = aroute.route(
            factories.build_decision(**COSTLY),
            candidates=_candidates(("v2 only", 10.0)),
        )
        assert isinstance(outcome, aroute.Ask)
        assert outcome.recommended == "v2 only"

    def test_split_evidence_on_a_costly_decision_asks_with_the_candidates(self) -> None:
        """
        Test that a genuinely split high-stakes decision reaches a human carrying
        what crux found, and with no recommendation it has not earned.
        """
        outcome = aroute.route(
            factories.build_decision(**COSTLY),
            candidates=_candidates(("per-user", 10.0), ("per-ip", 7.0)),
        )
        assert isinstance(outcome, aroute.Ask)
        assert outcome.recommended is None
        assert [o.value for o in outcome.options] == ["per-user", "per-ip"]

    def test_split_evidence_on_a_cheap_decision_takes_the_default(self) -> None:
        """
        Test that below the threshold an authored default closes the decision
        rather than a question being asked.
        """
        outcome = aroute.route(
            factories.build_decision(**CHEAP),
            candidates=_candidates(("a", 10.0), ("b", 9.0)),
            pack_default="a",
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "default"

    def test_split_evidence_with_no_default_goes_downstream_with_its_options(self) -> None:
        """
        Test that a cheap undecided thing with no authored default is delegated,
        carrying the option space so the downstream agent need not rediscover it.
        """
        outcome = aroute.route(
            factories.build_decision(**CHEAP),
            candidates=_candidates(("a", 10.0), ("b", 9.0)),
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "delegated"
        assert [o.value for o in outcome.options] == ["a", "b"]

    def test_no_evidence_on_a_costly_decision_asks_an_open_question(self) -> None:
        """
        Test that crux will ask with no options at all when the decision matters
        and it found nothing. The free-text field is what makes this answerable.
        """
        outcome = aroute.route(factories.build_decision(**COSTLY))
        assert isinstance(outcome, aroute.Ask)
        assert outcome.options == ()

    def test_no_evidence_on_a_cheap_decision_is_delegated(self) -> None:
        """
        Test the last cell of the table: nothing known, nothing authored, not
        worth asking. It still closes, as a stated delegation.
        """
        outcome = aroute.route(factories.build_decision(**CHEAP))
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "delegated"

    def test_mandatory_overrides_a_cheap_decision_into_a_question(self) -> None:
        """
        Test that a `constrains` edge can force a question the cost threshold
        would have defaulted away. Email opt-out is the real case: cheap to
        change, and legally not optional.
        """
        outcome = aroute.route(
            factories.build_decision(mandatory=True, **CHEAP),
            candidates=_candidates(("global", 10.0), ("per-category", 8.0)),
        )
        assert isinstance(outcome, aroute.Ask)


class TestBudgetExhaustion:
    """
    A spent budget closes the route to a human, and nothing else.
    """

    def test_a_spent_budget_turns_a_question_into_the_default(self) -> None:
        """
        Test that crux stops asking rather than hanging when the budget runs out.
        """
        outcome = aroute.route(
            factories.build_decision(**COSTLY),
            candidates=_candidates(("a", 10.0), ("b", 9.0)),
            pack_default="a",
            budget_spent=True,
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "default"
        assert "budget spent" in outcome.rationale

    def test_a_spent_budget_with_no_default_delegates(self) -> None:
        """
        Test that every remaining decision still closes when the budget is gone,
        which is what makes termination provable.
        """
        outcome = aroute.route(factories.build_decision(**COSTLY), budget_spent=True)
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "delegated"

    def test_a_spent_budget_does_not_discard_good_evidence(self) -> None:
        """
        Test that budget exhaustion blocks asking, not resolving. Written as
        stated in the table this row would have thrown away a dominant retrieval
        result and replaced it with a default, which would make a long session
        produce *worse* output than a short one.
        """
        outcome = aroute.route(
            factories.build_decision(**CHEAP),
            candidates=_candidates(("resend", 10.0)),
            pack_default="something-else",
            budget_spent=True,
        )
        assert isinstance(outcome, aroute.Resolve)
        assert outcome.source == "retrieval"
        assert outcome.value == "resend"


class TestTotality:
    """
    Every decision has a route in every state.
    """

    @pytest.mark.parametrize(
        "decision_type",
        ["ambiguity", "underspecification", "vagueness", "missing_context"],
    )
    @pytest.mark.parametrize("candidate_count", [0, 1, 2, 3])
    @pytest.mark.parametrize("has_default", [True, False])
    @pytest.mark.parametrize("costly", [True, False])
    @pytest.mark.parametrize("budget_spent", [True, False])
    def test_the_table_always_returns_a_route(
        self,
        decision_type: cdecis.DecisionType,
        candidate_count: int,
        has_default: bool,
        costly: bool,
        budget_spent: bool,
    ) -> None:
        """
        Test that no combination of inputs falls off the end of the table. A
        decision with no route sits open for ever, and the loop that drains the
        graph would never terminate.
        """
        outcome = aroute.route(
            factories.build_decision(type=decision_type, **(COSTLY if costly else CHEAP)),
            candidates=_candidates(*((f"v{i}", 10.0 - i) for i in range(candidate_count))),
            pack_default="fallback" if has_default else None,
            budget_spent=budget_spent,
        )
        assert isinstance(outcome, aroute.Resolve | aroute.Retype | aroute.Ask)

    @pytest.mark.parametrize(
        "decision_type",
        ["ambiguity", "underspecification", "vagueness", "missing_context"],
    )
    def test_nothing_is_ever_asked_when_the_budget_is_spent(
        self, decision_type: cdecis.DecisionType
    ) -> None:
        """
        Test the property the round cap depends on. If any input could still
        produce an Ask after exhaustion, resume() would emit questions the budget
        says it cannot afford and the session would never reach Done.
        """
        for count in range(4):
            for has_default in (True, False):
                outcome = aroute.route(
                    factories.build_decision(type=decision_type, mandatory=True, **COSTLY),
                    candidates=_candidates(*((f"v{i}", 10.0 - i) for i in range(count))),
                    pack_default="fallback" if has_default else None,
                    budget_spent=True,
                )
                assert not isinstance(outcome, aroute.Ask)


class TestCostRanking:
    """
    Turning the expander's ordering into cost bands.
    """

    @pytest.mark.parametrize("total", [1, 2, 3, 4, 7, 12, 30])
    def test_every_batch_has_at_least_one_high_cost_decision(self, total: int) -> None:
        """
        Test that the most costly thing the expander found always counts as
        costly.

        This test exists because it did not. The bands were pure shares, so a
        batch of three graded nothing high — the top-ranked decision came out
        medium, cleared no ask threshold, and the whole expansion fell through to
        delegation without a single question being asked.
        """
        import crux.ports.reasoner as preason

        bands = [preason.cost_from_rank(i, total) for i in range(total)]
        assert bands[0] == "high", bands
        assert bands.count("high") >= 1

    def test_the_ranking_never_goes_back_up(self) -> None:
        """
        Test that cost is monotonic in rank. A later proposal grading higher than
        an earlier one would silently invert the ordering the model was asked to
        supply.
        """
        import crux.ports.reasoner as preason

        order = {"high": 0, "medium": 1, "low": 2}
        for total in (1, 5, 9, 20):
            bands = [order[preason.cost_from_rank(i, total)] for i in range(total)]
            assert bands == sorted(bands), (total, bands)

    def test_a_large_batch_still_reserves_most_of_itself_for_the_tail(self) -> None:
        """
        Test that the fix did not turn everything high. Over-grading would make
        crux ask about everything, which is the failure it exists to prevent.
        """
        import crux.ports.reasoner as preason

        bands = [preason.cost_from_rank(i, 20) for i in range(20)]
        assert bands.count("high") == 6
        assert bands.count("low") == 6
