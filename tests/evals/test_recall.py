"""
Decision recall against a real model.

Marked ``live`` and excluded from CI. These are measurements, not correctness
tests: they move when a prompt changes, which is exactly what they are for.

Cassettes are per model: ``cassettes/recall-<model>.json``. Run them with a
recorded cassette (free, deterministic):

    uv run pytest tests/evals -m live

Or re-record against the real model:

    CRUX_RECORD=1 uv run pytest tests/evals -m live

The ratchets below are floors to raise over time, not targets that were tuned to
whatever the current numbers happen to be. Raise one only after a change that
should have improved it.
"""

from __future__ import annotations

import os

import pytest

import tests.evals.harness as harness
import tests.evals.runner as runner

pytestmark = pytest.mark.live

RECALL_FLOOR = 0.75
MAX_PRECISION_MISSES = 2

# One-liners should cost one or two questions; a whole-project brief earns more.
# A prompt edit that moves a stratum out of its band is the earliest warning that
# crux is drifting toward the interrogation it exists to replace.
ASK_RATE_BANDS = {
    "one_liner": (0.0, 3.0),
    "feature": (1.0, 4.0),
    "project": (2.0, 6.0),
}


@pytest.fixture(scope="module")
def report() -> harness.Report:
    """
    Run the whole corpus once and score it.

    :return: The aggregate report.
    """
    import asyncio

    record = os.environ.get("CRUX_RECORD") == "1"
    cases = [c for c in harness.load_corpus() if runner.fixture_exists(c)]
    if not cases:
        pytest.skip("no corpus cases with fixtures on disk")

    client = runner.build_client("recall", record=record)

    async def _run() -> harness.Report:
        scores = []
        for case in cases:
            try:
                session = await runner.run_case(case, client)
            except KeyError:
                pytest.skip("cassette is incomplete; rerun with CRUX_RECORD=1 to capture it")
            scores.append(harness.score_case(case, session))
        return harness.Report(scores=tuple(scores))

    result = asyncio.run(_run())
    print("\n" + result.render())
    return result


def test_decision_recall_clears_the_ratchet(report: harness.Report) -> None:
    """
    Test that crux surfaces the decisions a person said matter, on aggregate.

    Asserted on the mean rather than per case: individual cases move for reasons
    that have nothing to do with a regression, and a per-case assertion would be
    reverted rather than investigated.
    """
    assert report.recall >= RECALL_FLOOR, report.render()


def test_crux_stays_quiet_about_what_it_could_resolve(report: harness.Report) -> None:
    """
    Test the precision side, which matters more to a user than recall does.

    Surfacing a decision the repo already answers is the failure that gets the
    layer uninstalled — it is the exact behaviour the nodes-as-open-decisions
    model was chosen to avoid.
    """
    assert report.precision_misses <= MAX_PRECISION_MISSES, report.render()


def test_no_case_exceeds_its_question_budget(report: harness.Report) -> None:
    """
    Test that the cap holds against a real model. Budget arithmetic is unit
    tested; this checks that nothing about real expansion volume defeats it.
    """
    assert report.budget_breaches == 0, report.render()


@pytest.mark.parametrize("stratum", ["one_liner", "feature", "project"])
def test_ask_rate_stays_inside_its_band(report: harness.Report, stratum: str) -> None:
    """
    Test the product metric directly. The ask threshold is ungrounded enough that
    a prompt edit can move it without anyone noticing until users complain, so it
    gets an alarm rather than a review.
    """
    low, high = ASK_RATE_BANDS[stratum]
    rate = report.ask_rate(stratum)  # type: ignore[arg-type]
    if rate == 0.0:
        pytest.skip(f"no {stratum} cases ran")
    assert low <= rate <= high, f"{stratum} ask rate {rate:.1f} outside [{low}, {high}]"
