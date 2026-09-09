"""
One table across every model that has a results file.

    uv run python -m tests.evals.models [results.json ...]

Reads the results files the platform command writes and prints, per model,
the scores and the ask rate per stratum, then the case-level leads of every
model over the weakest. The mean says which model is better; the leads say on
how many cases, which is the number that survives a corpus change.

Import as:

import tests.evals.models as models
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import tests.evals.compare as compare

STRATA = ("one_liner", "feature", "project")


def ask_rate(run: compare.RunResults, stratum: str) -> float:
    """
    :param run: One run.
    :param stratum: Which prompt size.
    :return: Mean questions asked over that stratum's scored cases, or 0.0.
    """
    rows = [r for r in run.rows if r.stratum == stratum and not r.error]
    if not rows:
        return 0.0
    return sum(r.metrics.get("questions_asked", 0) for r in rows) / len(rows)


def weakest(runs: Sequence[compare.RunResults]) -> compare.RunResults:
    """
    :param runs: Every run.
    :return: The one with the lowest mean recall, the baseline the others are
        compared to.
    """
    return min(runs, key=lambda r: r.mean("recall"))


def render(runs: Sequence[compare.RunResults]) -> str:
    """
    :param runs: One run per model.
    :return: The table.
    """
    scores = sorted({name for run in runs for row in run.rows for name in row.scores})
    header = (
        f"{'model':<36} "
        + " ".join(f"{s:>9}" for s in scores)
        + "  "
        + " ".join(f"{'ask:' + st[:4]:>9}" for st in STRATA)
    )
    lines = [header, "-" * len(header)]
    for run in sorted(runs, key=lambda r: r.mean("recall"), reverse=True):
        cells = " ".join(f"{run.mean(s):>9.2f}" for s in scores)
        asks = " ".join(f"{ask_rate(run, st):>9.1f}" for st in STRATA)
        lines.append(f"{run.model[:36]:<36} {cells}  {asks}")
    if len(runs) > 1:
        base = weakest(runs)
        lines.append("")
        lines.append(f"case-level leads over {base.model} (wins/losses/ties per score):")
        for run in runs:
            if run is base:
                continue
            leads = compare.leads(run, base)
            summary = "  ".join(
                f"{s} {leads[s].wins}/{leads[s].losses}/{leads[s].ties}"
                for s in scores
                if s in leads
            )
            lines.append(f"  {run.model[:36]:<36} {summary}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """
    :param argv: Command line, for tests.
    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(prog="tests.evals.models", description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="results files to compare; default is every crux run in results/",
    )
    args = parser.parse_args(argv)
    paths = [pathlib.Path(f) for f in args.files] or sorted(
        compare.RESULTS.glob("stdout-*.json")
    ) + sorted(compare.RESULTS.glob("braintrust-*.json")) + sorted(
        compare.RESULTS.glob("opik-*.json")
    )
    runs = [compare.load(p) for p in paths]
    if not runs:
        print("no results files; run tests.evals.platform first")
        return 1
    print(render(runs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
