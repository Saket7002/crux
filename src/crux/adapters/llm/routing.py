"""
Which model serves which operation.

Two tiers, grouped by what the work actually demands rather than by one knob per
function. Naming what is undecided in a whole-project brief is the hardest thing
crux asks of a model; labelling a one-line reply as one of six kinds is the
easiest. Paying frontier prices for the second is waste, and paying free-tier
prices for the first is measurable: it is what produced 2% edge coherence in the
saturation run.

Pure policy. No I/O, no provider knowledge, no settings import beyond reading a
flat field bag once in :meth:`ModelRouting.of`.

Import as:

import crux.adapters.llm.routing as xroute
"""

from __future__ import annotations

from collections.abc import Mapping

import pydantic

import crux.infra.settings as isettn
import crux.ports.reasoner as preason

REASONING: tuple[preason.Operation, ...] = ("expand", "adjudicate")
"""Open-ended judgement. Naming what is undecided and ranking it by cost, and
deciding what a pile of retrieved code actually settles."""

MECHANICAL: tuple[preason.Operation, ...] = (
    "phrase",
    "classify",
    "answer_counter",
    "draft",
    "select_pack",
)
"""Bounded transformations over material already supplied.

``draft`` sits here deliberately, and it is the placement worth defending: the
compiled prompt's substance comes from the decision records, and its assumptions
block is derived by code rather than written. The model supplies only the task
sentence and the constraints, which a cheap model does perfectly well.
"""


class ModelRouting(pydantic.BaseModel):
    """
    The resolved answer to "which model for this operation".
    """

    model_config = pydantic.ConfigDict(frozen=True)

    reasoning: str | None = None
    mechanical: str | None = None
    overrides: Mapping[preason.Operation, str] = pydantic.Field(default_factory=dict)

    @staticmethod
    def of(settings: isettn.CruxSettings) -> ModelRouting:
        """
        Read a routing out of settings.

        :param settings: Where the model names live.
        :return: The routing.
        """
        return ModelRouting(
            reasoning=settings.model,
            mechanical=settings.weak_model or settings.model,
            overrides=settings.operation_overrides(),
        )

    def model_for(self, operation: preason.Operation) -> str | None:
        """
        Pick the model for one operation.

        Total and deterministic: every operation resolves, and ``None`` is a
        legal answer meaning "whatever the client's own default is". That is what
        keeps every existing caller working unchanged.

        :param operation: Which of the six is being run.
        :return: The model name, or ``None`` for the client's default.
        """
        override = self.overrides.get(operation)
        if override:
            return override
        if operation in REASONING:
            return self.reasoning
        return self.mechanical

    @property
    def is_uniform(self) -> bool:
        """
        Say whether every operation resolves to the same place.

        :return: Whether no tier split or override is in effect.
        """
        resolved = {self.model_for(op) for op in REASONING + MECHANICAL}
        return len(resolved) == 1


DEFAULT = ModelRouting()
"""Everything on the client's own default, which is today's behaviour exactly."""
