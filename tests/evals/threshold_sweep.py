"""
Sweep the three calibration thresholds one at a time.

This is an experiment, not a correctness test. It reports decision recall,
duplicate rate, and question count for each candidate threshold.
"""

from __future__ import annotations

import asyncio

import crux.domain.session as csessn
import tests.evals.harness as harness
import tests.evals.runner as runner


def duplicate_rate(session: csessn.Session) -> float:
    """Return the fraction of proposals removed as duplicates."""
    proposed = sum(record.proposed for record in session.passes.records)
    duplicates = sum(record.proposed - record.new_after_dedup for record in session.passes.records)
    if proposed == 0:
        return 0.0
    return duplicates / proposed


async def run_sweep() -> None:
    cases = [case for case in harness.load_corpus() if runner.fixture_exists(case)]
    client = runner.build_client("recall", record=True)

    candidates = {
        "saturation": (0.05, 0.10, 0.15, 0.20, 0.25),
        "duplicate": (0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
        "sibling": (0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
    }
    for name, values in candidates.items():
        print(f"\n{name} threshold")
        print("value   recall   duplicate_rate   questions")

        for value in values:
            scores = []
            sessions = []

            for case in cases:
                session = await runner.run_case(
                    case,
                    client,
                    saturation_ratio=value if name == "saturation" else None,
                    duplicate_threshold=value if name == "duplicate" else None,
                    sibling_threshold=value if name == "sibling" else None,
                )
                sessions.append(session)
                scores.append(harness.score_case(case, session))

            report = harness.Report(scores=tuple(scores))
            proposed = sum(
                record.proposed for session in sessions for record in session.passes.records
            )
            duplicates = sum(
                record.proposed - record.new_after_dedup
                for session in sessions
                for record in session.passes.records
            )
            dup_rate = duplicates / proposed if proposed else 0.0
            questions = sum(score.questions_asked for score in report.scores)

            print(f"{value:0.2f}   {report.recall:0.3f}   {dup_rate:0.3f}           {questions}")


if __name__ == "__main__":
    asyncio.run(run_sweep())
