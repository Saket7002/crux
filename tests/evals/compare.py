"""
Comparing two eval runs case by case, not only on the mean.

A prompt that fixes every project-sized case and breaks one one-liner is flat
on the mean and invisible. GEPA (arXiv 2507.19457, Table 3) measured greedy
selection on the mean reaching half the gain of selection over per-instance
bests. So the comparison here says, per score, how many cases each run leads,
and names every case that regressed.

Import as:

import tests.evals.compare as compare
"""

from __future__ import annotations

import datetime as dt
import pathlib

import pydantic

RESULTS = pathlib.Path(__file__).parent / "results"


class CaseRow(pydantic.BaseModel):
    """
    One case from one run, as written to disk.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    stratum: str
    scores: dict[str, float]
    metrics: dict[str, int]
    feedback: str = ""
    error: str = ""
    expected: int = 0
    """How many decisions the case said must surface."""
    missed: tuple[str, ...] = ()
    """The expectation labels that did not surface."""
    surfaced: tuple[str, ...] = ()
    """What every decision in the graph said was undecided."""


class RunResults(pydantic.BaseModel):
    """
    One run's rows, plus enough metadata to know what it was.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    experiment: str
    model: str
    git_sha: str
    cassette_mode: str
    recorded_at: dt.datetime
    rows: tuple[CaseRow, ...] = ()

    def by_case(self) -> dict[str, CaseRow]:
        """
        :return: Rows keyed by case id.
        """
        return {row.case_id: row for row in self.rows}

    def mean(self, score: str) -> float:
        """
        :param score: Which score.
        :return: Its mean over the cases that have it, or 0.0.
        """
        values = [row.scores[score] for row in self.rows if score in row.scores]
        return sum(values) / len(values) if values else 0.0


class Leads(pydantic.BaseModel):
    """
    How many cases each side leads on, for one score.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    wins: int
    losses: int
    ties: int


class Regression(pydantic.BaseModel):
    """
    One case that got worse on one score.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    score: str
    before: float
    after: float


def load(path: pathlib.Path) -> RunResults:
    """
    :param path: A results file written by the platform command.
    :return: The run.
    """
    return RunResults.model_validate_json(path.read_text(encoding="utf-8"))


def save(results: RunResults, path: pathlib.Path) -> None:
    """
    :param results: The run.
    :param path: Where to write it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results.model_dump_json(indent=2), encoding="utf-8")


def leads(current: RunResults, baseline: RunResults) -> dict[str, Leads]:
    """
    Count, per score, the cases the current run leads, trails, and ties.

    Only cases present in both runs with that score count, so a case that
    failed in one run neither wins nor loses.

    :param current: The run under test.
    :param baseline: What it is compared against.
    :return: Leads keyed by score name.
    """
    before = baseline.by_case()
    out: dict[str, Leads] = {}
    for score in _score_names(current, baseline):
        wins = losses = ties = 0
        for row in current.rows:
            other = before.get(row.case_id)
            if other is None or score not in row.scores or score not in other.scores:
                continue
            if row.scores[score] > other.scores[score]:
                wins += 1
            elif row.scores[score] < other.scores[score]:
                losses += 1
            else:
                ties += 1
        out[score] = Leads(wins=wins, losses=losses, ties=ties)
    return out


def regressions(current: RunResults, baseline: RunResults) -> tuple[Regression, ...]:
    """
    Name every case that got worse on any score.

    :param current: The run under test.
    :param baseline: What it is compared against.
    :return: Regressions, worst drop first.
    """
    before = baseline.by_case()
    found: list[Regression] = []
    for row in current.rows:
        other = before.get(row.case_id)
        if other is None:
            continue
        for score, value in row.scores.items():
            if score in other.scores and value < other.scores[score]:
                found.append(
                    Regression(
                        case_id=row.case_id, score=score, before=other.scores[score], after=value
                    )
                )
    return tuple(sorted(found, key=lambda r: r.after - r.before))


def render(current: RunResults, baseline: RunResults) -> str:
    """
    :param current: The run under test.
    :param baseline: What it is compared against.
    :return: A table a person can read.
    """
    lines = [
        f"{current.experiment} ({current.model}, {current.git_sha}) vs "
        f"{baseline.experiment} ({baseline.model}, {baseline.git_sha})",
        "",
        f"{'score':<15} {'mean':>7} {'base':>7} {'leads':>6} {'trails':>7} {'ties':>5}",
        "-" * 52,
    ]
    for score, lead in leads(current, baseline).items():
        lines.append(
            f"{score:<15} {current.mean(score):>7.2f} {baseline.mean(score):>7.2f} "
            f"{lead.wins:>6} {lead.losses:>7} {lead.ties:>5}"
        )
    worse = regressions(current, baseline)
    if worse:
        lines.append("")
        lines.append(f"{len(worse)} regression(s):")
        lines.extend(
            f"  {r.case_id:<28} {r.score:<14} {r.before:.2f} -> {r.after:.2f}" for r in worse
        )
    return "\n".join(lines)


def _score_names(*runs: RunResults) -> tuple[str, ...]:
    """
    :param runs: Runs to collect score names from.
    :return: Every score name seen, sorted.
    """
    return tuple(sorted({name for run in runs for row in run.rows for name in row.scores}))
