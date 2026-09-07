"""
Turning frontier decisions into questions a respondent can actually answer.

The free-text field is added here, by code, and never by the model that worded
the question. That is what makes it an invariant rather than a suggestion: there
is no prompt a model could write that removes it.

Import as:

import crux.application.questions as aquest
"""

from __future__ import annotations

import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.ids as cids
import crux.domain.replies as creply
import crux.ports.reasoner as preason


def build(
    decisions: tuple[cdecis.OpenDecision, ...],
    phrased: preason.PhraseResult,
    evidence: cevid.EvidenceGraph,
    *,
    recommendations: dict[str, str] | None = None,
) -> tuple[creply.Question, ...]:
    """
    Assemble the round crux is about to put to a respondent.

    :param decisions: The frontier, already ranked.
    :param phrased: The model's wording for them.
    :param evidence: The session's evidence, for citations.
    :param recommendations: Pre-filled answers the router earned, by decision id.
        Set where the evidence was unambiguous but the decision was too expensive
        to resolve without a confirmation.
    :return: The questions, one per decision that got worded.
    """
    by_decision = {q.decision_id: q for q in phrased.questions}
    recommended = recommendations or {}
    questions: list[creply.Question] = []
    for decision in decisions:
        wording = by_decision.get(decision.id)
        if wording is None:
            # A model that skipped a decision does not get to silence it: fall
            # back to the raw text rather than dropping the question.
            text, why = decision.undecided, ""
        else:
            text, why = wording.text, wording.why
        questions.append(
            creply.Question(
                id=cids.question_id(decision.id, exchange=decision.ask_exchanges),
                decision_id=decision.id,
                text=text,
                why=why,
                options=decision.options or (),
                citations=evidence.locators(
                    tuple(
                        item_id
                        for option in (decision.options or ())
                        for item_id in option.evidence
                    )
                ),
                exchange=decision.ask_exchanges,
                recommended=recommended.get(decision.id)
                or (wording.recommended if wording is not None else None),
            )
        )
    return tuple(questions)
