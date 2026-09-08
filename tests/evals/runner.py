"""
Running corpus cases against a real model, or a cassette of one.

Kept separate from the harness so the saturation experiment and the recall evals
share exactly one definition of "run a case", and cannot drift into measuring
subtly different things.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import crux.adapters.llm.litellm as xlitell
import crux.adapters.llm.reasoner as xreason
import crux.adapters.retrieval.fs as xfsretr
import crux.application.engine as aengine
import crux.domain.replies as creply
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.llm as pllm
import tests.evals.harness as harness


def build_client(
    cassette_name: str,
    *,
    record: bool,
    settings: isettn.CruxSettings | None = None,
) -> harness.CassetteLlm:
    """
    Build a cassette-backed client.

    :param cassette_name: Which cassette to use.
    :param record: Whether real calls are allowed for unseen requests.
    :param settings: Model configuration.
    :return: The client.
    """
    resolved = require_single_model(settings or isettn.CruxSettings())
    inner: pllm.LlmClient | None = xlitell.LiteLlmClient(resolved) if record else None
    return harness.CassetteLlm(inner, harness.CASSETTES / f"{cassette_name}.json", record=record)


def require_single_model(settings: isettn.CruxSettings) -> isettn.CruxSettings:
    """
    Refuse a fallback chain for any measurement.

    :param settings: Model configuration.
    :return: The same settings, when they name exactly one model.
    :raises ConfigurationError: When a fallback chain is configured.
    """
    if settings.fallback_models:
        # A fallback mid-run would make the noise floor a model *difference*
        # rather than sampling noise, and the cassette would record the
        # substitute's answer under the primary's key. Both corruptions are
        # silent, so the chain is refused rather than warned about.
        raise cerrors.ConfigurationError(
            "Measurement runs must use a single model: a fallback would make the "
            "noise floor a model difference and record the wrong answer in the "
            "cassette. Unset CRUX_FALLBACK_MODELS."
        )
    return settings


class FirstOptionRespondent:
    """
    Answers every question with its first option, deterministically.

    Exists because a headless run cannot measure edges. An edge fires when its
    source resolves to a *particular value*, and headless resolves everything to
    "delegated" — so edge-firing density would read exactly 0.0 on every case and
    the experiment would return "the graph is ceremony" for a reason having
    nothing to do with the graph.

    First-option rather than random so a re-run is comparable to the last one.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    def answer(self, questions: Sequence[creply.Question]) -> tuple[creply.RawReply, ...]:
        """
        :param questions: The round crux put.
        :return: One reply per question.
        """
        replies: list[creply.RawReply] = []
        for question in questions:
            self.asked.append(question.decision_id)
            if question.options:
                replies.append(
                    creply.RawReply(question_id=question.id, selected=(question.options[0].value,))
                )
            else:
                # No options means free text, and inventing prose here would put
                # a model in the loop that the experiment is trying to hold still.
                replies.append(creply.RawReply(question_id=question.id, text="you decide"))
        return tuple(replies)


def build_engine(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    *,
    budget: csessn.Budget | None = None,
    expand_instruction: str | None = None,
) -> aengine.Crux:
    """
    Wire crux up for one case, the same way every eval path does.

    :param case: The case to run.
    :param client: Where completions come from.
    :param budget: Caps to run under.
    :param expand_instruction: A candidate expansion instruction under test.
    :return: The engine.
    """
    return aengine.Crux(
        reasoner=xreason.LlmReasoner(client, expand_instruction=expand_instruction),
        retriever=xfsretr.FilesystemRetriever(case.root) if case.root else None,
        budget=budget or csessn.Budget(max_questions_total=case.max_questions),
    )


async def run_case(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    *,
    budget: csessn.Budget | None = None,
    answer: bool = False,
    expand_instruction: str | None = None,
) -> csessn.Session:
    """
    Run one case to completion.

    :param case: The case to run.
    :param client: Where completions come from.
    :param budget: Caps to run under.
    :param answer: Whether to answer the questions with a scripted respondent.
        Recall scoring leaves them unanswered — it measures which decisions crux
        *surfaced*, and answering would change what later passes expand. The
        saturation experiment must answer, because unanswered decisions never
        resolve to a value and so no edge can ever fire.
    :param expand_instruction: A candidate expansion instruction under test.
    :return: The finished session.
    """
    crux = build_engine(case, client, budget=budget, expand_instruction=expand_instruction)
    step = await crux.start(
        case.prompt,
        context=csessn.SessionContext(root=case.root) if case.root else None,
        session_id=case.id,
    )
    if not answer:
        return step.session
    return await finish_case(case, client, step, budget=budget)


async def finish_case(
    case: harness.EvalCase,
    client: pllm.LlmClient,
    step: csessn.Step,
    *,
    budget: csessn.Budget | None = None,
) -> csessn.Session:
    """
    Answer every remaining question with the scripted respondent.

    Exists so a session scored the recall way (questions left unanswered) can
    still be carried to a compiled prompt for the judge, without the scoring
    session itself changing.

    :param case: The case being run.
    :param client: Where completions come from.
    :param step: Where the session got to.
    :param budget: Caps to run under.
    :return: The finished session, with an outcome.
    """
    crux = build_engine(case, client, budget=budget)
    respondent = FirstOptionRespondent()
    while isinstance(step, csessn.NeedsInput):
        step = await crux.resume(step.session, respondent.answer(step.questions))
    return step.session


def fixture_exists(case: harness.EvalCase) -> bool:
    """
    :param case: The case to check.
    :return: Whether its fixture repo is present on disk.
    """
    return case.root is None or pathlib.Path(case.root).is_dir()
