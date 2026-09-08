"""
The Braintrust backend.

Uses ``init_dataset`` + ``init`` + ``Experiment.log`` rather than
``braintrust.Eval``: ``Eval`` owns the loop, the task and the scorers, which is
exactly the part every backend shares. Precomputed scores go in as rows, and
each crux call becomes an ``llm`` span under the row.

The SDK is imported inside the constructor so this module loads without it.

Import as:

import tests.evals.platform.braintrust_backend as btback
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import crux.errors as cerrors
import tests.evals.harness as harness
import tests.evals.platform.adapter as adapter


class BraintrustBackend:
    """
    One dataset for the corpus, one experiment per run.
    """

    name = "braintrust"

    def __init__(self, *, project: str, experiment: str | None = None) -> None:
        """
        :param project: The Braintrust project.
        :param experiment: The experiment name. ``None`` lets Braintrust pick.
        :raises ConfigurationError: When the SDK is not installed.
        """
        try:
            import braintrust
        except ImportError as exc:
            raise cerrors.ConfigurationError(
                "The braintrust SDK is not installed. Run `uv sync --extra evals`."
            ) from exc
        # Held as Any: the SDK's own typing is partial and moves between releases.
        self._sdk: Any = braintrust
        self._project = project
        self._experiment_name = experiment
        self._dataset: Any = None
        self._experiment: Any = None

    def upsert_dataset(self, cases: Sequence[harness.EvalCase]) -> dict[str, str]:
        self._dataset = self._sdk.init_dataset(project=self._project, name="crux-corpus")
        item_ids: dict[str, str] = {}
        for case in cases:
            # Inserting with an explicit id upserts, so a rerun updates the row.
            item_ids[case.id] = self._dataset.insert(
                id=case.id,
                input=_case_input(case),
                expected=_case_expected(case),
                metadata={"stratum": case.stratum, "fixture_repo": case.fixture_repo},
            )
        self._dataset.flush()
        return item_ids

    def start_experiment(self, meta: adapter.RunMetadata, item_ids: dict[str, str]) -> None:
        self._experiment = self._sdk.init(
            project=self._project,
            experiment=self._experiment_name,
            dataset=self._dataset,
            metadata=meta.model_dump(mode="json"),
        )
        self._item_ids = item_ids

    def log_result(self, result: adapter.CaseResult) -> None:
        with self._experiment.start_span(
            name=result.case.id,
            type="eval",
            input=_case_input(result.case),
            output=result.rendered,
            error=result.error or None,
            expected=_case_expected(result.case),
            scores=result.scores,
            metrics={**result.metrics, "elapsed_seconds": result.elapsed_seconds},
            metadata={
                "stratum": result.case.stratum,
                "judge_rationale": result.judge_rationale,
                "error": result.error,
                "missed": list(result.score.missed) if result.score else [],
                "should_have_been_quiet": (
                    list(result.score.should_have_been_quiet) if result.score else []
                ),
            },
            dataset_record_id=self._item_ids.get(result.case.id),
        ) as row:
            for call in result.calls:
                with row.start_span(name=call.operation, type="llm") as span:
                    span.log(
                        input=list(call.messages),
                        output=call.response,
                        error=call.error or None,
                        metrics={"elapsed_seconds": call.elapsed_seconds},
                        metadata={"model": call.model, "attempt": call.attempt},
                    )

    def finish(self) -> str:
        summary = self._experiment.summarize()
        return str(getattr(summary, "experiment_url", "") or summary)


def _case_input(case: harness.EvalCase) -> dict[str, Any]:
    return {
        "case_id": case.id,
        "prompt": case.prompt,
        "stratum": case.stratum,
        "fixture_repo": case.fixture_repo,
    }


def _case_expected(case: harness.EvalCase) -> dict[str, Any]:
    return case.model_dump(
        mode="json",
        include={
            "must_surface",
            "must_not_surface",
            "must_resolve_by_retrieval",
            "max_questions",
            "expected",
        },
    )
