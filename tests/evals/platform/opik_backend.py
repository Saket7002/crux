"""
The Comet Opik backend.

Uses the client and trace API rather than ``opik.evaluate``: ``evaluate`` wants
a synchronous task and its own metric classes, which would duplicate the
scoring path. Each case becomes a trace with one ``llm`` span per crux call and
one feedback score per named score, linked to its dataset item through an
experiment.

Opik deduplicates dataset items by content and assigns its own ids, so the
case id is stored inside the item and the mapping is read back after insert.

Import as:

import tests.evals.platform.opik_backend as opback
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import crux.errors as cerrors
import tests.evals.harness as harness
import tests.evals.platform.adapter as adapter

DATASET_NAME = "crux-corpus"


class OpikBackend:
    """
    One dataset for the corpus, one experiment per run.
    """

    name = "opik"

    def __init__(self, *, project: str, experiment: str | None = None) -> None:
        """
        :param project: The Opik project.
        :param experiment: The experiment name. ``None`` lets Opik pick.
        :raises ConfigurationError: When the SDK is not installed.
        """
        try:
            import opik
        except ImportError as exc:
            raise cerrors.ConfigurationError(
                "The opik SDK is not installed. Run `uv sync --extra evals`."
            ) from exc
        # Held as Any: the SDK's own typing is partial and moves between releases.
        self._sdk: Any = opik
        self._client: Any = opik.Opik(project_name=project)
        self._project = project
        self._experiment_name = experiment
        self._experiment: Any = None
        self._item_ids: dict[str, str] = {}

    def upsert_dataset(self, cases: Sequence[harness.EvalCase]) -> dict[str, str]:
        dataset = self._client.get_or_create_dataset(name=DATASET_NAME)
        dataset.insert(
            [
                {
                    "case_id": case.id,
                    "input": _case_input(case),
                    "expected": _case_expected(case),
                }
                for case in cases
            ]
        )
        wanted = {case.id for case in cases}
        for item in dataset.get_items():
            case_id = item.get("case_id")
            if case_id in wanted:
                self._item_ids[case_id] = str(item["id"])
        missing = wanted - set(self._item_ids)
        if missing:
            raise cerrors.ConfigurationError(
                f"Opik returned no dataset item for {sorted(missing)}; the insert did not land."
            )
        return dict(self._item_ids)

    def start_experiment(self, meta: adapter.RunMetadata, item_ids: dict[str, str]) -> None:
        self._item_ids = dict(item_ids)
        self._experiment = self._client.create_experiment(
            dataset_name=DATASET_NAME,
            name=self._experiment_name,
            experiment_config=meta.model_dump(mode="json"),
        )

    def log_result(self, result: adapter.CaseResult) -> None:
        trace = self._client.trace(
            name=result.case.id,
            input=_case_input(result.case),
            output={"compiled": result.rendered},
            metadata={
                "stratum": result.case.stratum,
                "judge_rationale": result.judge_rationale,
                "error": result.error,
                "missed": list(result.score.missed) if result.score else [],
                "should_have_been_quiet": (
                    list(result.score.should_have_been_quiet) if result.score else []
                ),
                **result.metrics,
            },
        )
        for call in result.calls:
            span = trace.span(
                name=call.operation,
                type="llm",
                model=call.model,
                input={"messages": list(call.messages)},
                output=call.response,
                metadata={"attempt": call.attempt, "elapsed_seconds": call.elapsed_seconds},
                error_info={"exception_type": "ReasonerError", "message": call.error}
                if call.error
                else None,
            )
            span.end()
        for name, value in result.scores.items():
            trace.log_feedback_score(
                name=name,
                value=value,
                reason=result.judge_rationale if name == "judge" else None,
            )
        trace.end()
        self._experiment.insert(
            [
                self._sdk.ExperimentItemReferences(
                    dataset_item_id=self._item_ids[result.case.id], trace_id=trace.id
                )
            ]
        )

    def finish(self) -> str:
        self._client.flush()
        return f"opik experiment {self._experiment.name!s} in project {self._project}"


def _case_input(case: harness.EvalCase) -> dict[str, Any]:
    return {
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
