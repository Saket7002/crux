"""
Decomposing a case score into the named 0..1 scores an eval platform charts.

``harness.CaseScore`` is one object with counts and lists in it, which is what
the ratchets want. A platform wants one number per named question, each in
0..1 so they can be averaged across an experiment and diffed across two. This
is the only place that translation lives.

Import as:

import tests.evals.scorers as scorers
"""

from __future__ import annotations

import tests.evals.harness as harness
import tests.evals.judge as judge

Scores = dict[str, float]


def decompose(score: harness.CaseScore, case: harness.EvalCase) -> Scores:
    """
    Split one case score into named 0..1 scores.

    A score whose expectation set is empty is 1.0, not undefined: a case that
    expects nothing to be quiet cannot have been noisy.

    :param score: How the case went.
    :param case: What it expected, for the denominators.
    :return: Scores keyed by name.
    """
    return {
        "recall": score.recall,
        "quiet": _fraction_kept(len(score.should_have_been_quiet), len(case.must_not_surface)),
        "retrieval": _fraction_kept(
            len(score.retrieval_expected_but_not), len(case.must_resolve_by_retrieval)
        ),
        "within_budget": 0.0 if score.over_question_budget else 1.0,
        "clean": 1.0 if score.clean else 0.0,
    }


def metrics(score: harness.CaseScore) -> dict[str, int]:
    """
    The counts worth charting but not averaging with the scores.

    :param score: How the case went.
    :return: Integer metrics keyed by name.
    """
    return {
        "questions_asked": score.questions_asked,
        "assumptions": score.assumptions,
        "decisions_total": score.decisions_total,
        "llm_calls": score.llm_calls,
    }


def _fraction_kept(failures: int, total: int) -> float:
    """
    :param failures: How many expectations were missed.
    :param total: How many there were.
    :return: The fraction met, or 1.0 when there were none.
    """
    if total == 0:
        return 1.0
    return 1.0 - failures / total


def feedback(
    score: harness.CaseScore,
    case: harness.EvalCase,
    judgement: judge.Judgement | None = None,
) -> str:
    """
    Say in words why the case scored what it did.

    The number is for the chart; this is for the person, or the reflection
    model, deciding what to change. It costs no model call: everything here is
    already known to the scorer and the judge, it was just never written down.

    :param score: How the case went.
    :param case: What it expected, for the budget line.
    :param judgement: The judge's verdicts, when the case was judged.
    :return: One finding per line, or a single line saying nothing was wrong.
    """
    lines: list[str] = []
    lines.extend(f"missed: {label}" for label in score.missed)
    lines.extend(
        f"asked, though the repo answers it: {label}" for label in score.should_have_been_quiet
    )
    lines.extend(
        f"expected from retrieval, but not resolved that way: {label}"
        for label in score.retrieval_expected_but_not
    )
    if score.over_question_budget:
        lines.append(
            f"over question budget: asked {score.questions_asked}, allowed {case.max_questions}"
        )
    if judgement is not None:
        for verdict in judgement.verdicts:
            if verdict.met:
                continue
            evidence = f" ({verdict.evidence})" if verdict.evidence else ""
            lines.append(f"rubric not met [{verdict.section}]: {verdict.bullet}{evidence}")
    return "\n".join(lines) if lines else "every expectation met"
