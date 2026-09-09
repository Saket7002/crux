"""
Bounding the matcher's error by hand.

The recall matcher is token overlap and cannot see paraphrase, so "timeout
duration for graceful shutdown" does not match "how long in-flight jobs get to
finish". How much of the recall gap is that? Nothing automatic can say, so
this prints every miss beside the surfaced decisions nearest to it, a person
marks which misses were really covered, and corrected recall is reported next
to the automatic one. Only that number decides whether the matcher needs
paraphrase support.

Import as:

import tests.evals.rescoring as rescoring
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pydantic
import yaml

import crux.domain.ids as cids
import tests.evals.compare as compare

NEAREST = 3


class Miss(pydantic.BaseModel):
    """
    One expectation that did not surface, with its nearest surfaced decisions.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    expectation: str
    nearest: tuple[tuple[str, float], ...] = ()


class Rescore(pydantic.BaseModel):
    """
    A person's verdict on one miss.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    case_id: str
    expectation: str
    covered_by: str | None = None
    """The surfaced decision that actually covers it, or ``None`` for a true miss."""


def misses(run: compare.RunResults, *, nearest: int = NEAREST) -> tuple[Miss, ...]:
    """
    List every miss with the surfaced decisions closest to it by overlap.

    :param run: A run with ``missed`` and ``surfaced`` on its rows.
    :param nearest: How many candidates to show per miss.
    :return: The misses, in corpus order.
    """
    out: list[Miss] = []
    for row in run.rows:
        for label in row.missed:
            text = label.removeprefix("~")
            ranked = sorted(
                ((s, cids.similarity(text, s)) for s in row.surfaced),
                key=lambda pair: pair[1],
                reverse=True,
            )
            out.append(
                Miss(case_id=row.case_id, expectation=label, nearest=tuple(ranked[:nearest]))
            )
    return tuple(out)


def template(found: Sequence[Miss]) -> str:
    """
    :param found: The misses.
    :return: YAML a person fills in, ``covered_by`` left null.
    """
    rows = [
        {
            "case_id": m.case_id,
            "expectation": m.expectation,
            "nearest": [s for s, _ in m.nearest],
            "covered_by": None,
        }
        for m in found
    ]
    return yaml.safe_dump(rows, sort_keys=False, allow_unicode=True)


def render(found: Sequence[Miss]) -> str:
    """
    :param found: The misses.
    :return: A table a person can read.
    """
    lines = [f"{len(found)} missed expectation(s); nearest surfaced decisions by overlap:"]
    for m in found:
        lines.append(f"{m.case_id}: {m.expectation}")
        lines.extend(f"    {score:.2f}  {text}" for text, score in m.nearest)
    return "\n".join(lines)


def load_rescores(path: pathlib.Path) -> tuple[Rescore, ...]:
    """
    :param path: A filled-in template.
    :return: The verdicts. Rows without ``covered_by`` are true misses.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return tuple(
        Rescore(
            case_id=str(r["case_id"]),
            expectation=str(r["expectation"]),
            covered_by=r.get("covered_by") or None,
        )
        for r in raw
    )


def corrected(
    run: compare.RunResults, rescores: Sequence[Rescore]
) -> tuple[dict[str, tuple[float, float]], float, float]:
    """
    Recall per case before and after the hand marks, and the two means.

    :param run: The run that was rescored.
    :param rescores: The verdicts.
    :return: Per-case (automatic, corrected), then the automatic mean, then
        the corrected mean, over the scored cases.
    """
    covered = {(r.case_id, r.expectation) for r in rescores if r.covered_by}
    per_case: dict[str, tuple[float, float]] = {}
    for row in run.rows:
        if row.error:
            continue
        if row.expected == 0:
            per_case[row.case_id] = (1.0, 1.0)
            continue
        automatic = (row.expected - len(row.missed)) / row.expected
        regained = sum(1 for label in row.missed if (row.case_id, label) in covered)
        per_case[row.case_id] = (automatic, automatic + regained / row.expected)
    if not per_case:
        return per_case, 0.0, 0.0
    auto_mean = sum(a for a, _ in per_case.values()) / len(per_case)
    fixed_mean = sum(c for _, c in per_case.values()) / len(per_case)
    return per_case, auto_mean, fixed_mean


def render_corrected(run: compare.RunResults, rescores: Sequence[Rescore]) -> str:
    """
    :param run: The run that was rescored.
    :param rescores: The verdicts.
    :return: The comparison a person reads.
    """
    per_case, auto_mean, fixed_mean = corrected(run, rescores)
    lines = [
        f"{'case':<28} {'automatic':>10} {'corrected':>10}",
        "-" * 50,
    ]
    lines.extend(
        f"{case_id:<28} {a:>10.2f} {c:>10.2f}" for case_id, (a, c) in per_case.items() if a != c
    )
    marked = sum(1 for r in rescores if r.covered_by)
    lines.append("-" * 50)
    lines.append(
        f"recall {auto_mean:.2f} automatic, {fixed_mean:.2f} after {marked} hand-covered "
        f"miss(es) out of {len(rescores)}; the gap is matcher error"
    )
    return "\n".join(lines)
