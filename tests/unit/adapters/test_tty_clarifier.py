"""
Tier-3 tests for the console clarifier.

Small surface, but it is where a typed line becomes a reply, and getting that
wrong silently changes what the respondent said.
"""

from __future__ import annotations

import crux.adapters.clarifier.tty as xtty
import crux.domain.decisions as cdecis
import crux.domain.replies as creply


def _question(*, options: tuple[str, ...] = (), recommended: str | None = None) -> creply.Question:
    """
    Build a question.

    :param options: Values to offer.
    :param recommended: A pre-filled suggestion, if the router earned one.
    :return: The question.
    """
    return creply.Question(
        id="q1",
        decision_id="d1",
        text="Which channel?",
        options=tuple(cdecis.Option(value=v, source="pack") for v in options),
        recommended=recommended,
    )


class TestTypedReplies:
    """
    What a typed line is understood to mean.
    """

    def test_a_number_selects_that_option(self) -> None:
        """
        Test the ordinary case, and that it becomes a *selection* rather than
        free text — a selection is classified without a model call, so getting
        this wrong costs an LLM round trip on every answer.
        """
        reply = xtty._to_reply(_question(options=("email", "in-app")), "2")

        assert reply.selected == ("in-app",)
        assert reply.text == ""
        assert reply.is_bare_choice

    def test_a_number_outside_the_range_is_treated_as_free_text(self) -> None:
        """
        Test that "7" against two options is not silently clamped to an option
        the respondent never chose. It goes to the classifier as what they typed.
        """
        reply = xtty._to_reply(_question(options=("email", "in-app")), "7")

        assert reply.selected == ()
        assert reply.text == "7"

    def test_a_number_with_no_options_offered_is_free_text(self) -> None:
        """
        Test that a numeric answer to an open question stays an answer. "100" to
        "how many requests per minute?" is a value, not a menu choice.
        """
        reply = xtty._to_reply(_question(), "100")

        assert reply.text == "100"
        assert reply.selected == ()

    def test_prose_is_kept_verbatim(self) -> None:
        """
        Test that anything else reaches the classifier unaltered — this is the
        path carrying counter-questions, proposals and rejections, which are the
        replies a click-only interface can never deliver.
        """
        reply = xtty._to_reply(_question(options=("email",)), "what do you mean by channel?")

        assert reply.text == "what do you mean by channel?"
        assert reply.selected == ()

    def test_an_empty_line_accepts_the_recommendation(self) -> None:
        """
        Test that pressing enter takes the suggestion when there is one. The
        router only recommends where the evidence was unambiguous, so this is a
        confirmation, not a guess put in the respondent's mouth.
        """
        reply = xtty._to_reply(_question(options=("email",), recommended="email"), "")

        assert reply.selected == ("email",)

    def test_an_empty_line_with_nothing_recommended_stays_empty(self) -> None:
        """
        Test that enter on an unrecommended question answers nothing rather than
        inventing something. An unanswered question becomes a stated delegation,
        which is honest; a fabricated answer is not.
        """
        reply = xtty._to_reply(_question(options=("email",)), "")

        assert reply.selected == ()
        assert reply.text == ""
