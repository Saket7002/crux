"""
The whole pipeline, against the real fixture repo.

Everything here is real except the HTTP call: the engine, the route table, the
filesystem retriever, the prompt builders and the response parsers all run. Only
the model's answers are scripted.

This is the test that would notice if the pieces stopped fitting together, which
none of the tier-1 or tier-2 tests can see.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import pytest

import crux.adapters.llm.reasoner as xreason
import crux.adapters.retrieval.fs as xfsretr
import crux.api as capi
import crux.domain.replies as creply
import crux.domain.session as csessn
import tests.support.llm as scripted

FIXTURE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "tiny_fastapi"

# What the expander "finds" in the prompt. Ordered most costly first, because the
# engine derives cost from position.
_EXPANSION = {
    "decisions": [
        {
            "undecided": "Which endpoints the throttle applies to",
            "type": "underspecification",
            "reversibility": "hard",
            "options": ["all /api/v2 routes", "write endpoints only"],
        },
        {
            "undecided": "How requests are identified for counting",
            "type": "ambiguity",
            "reversibility": "hard",
            "options": ["per authenticated user", "per IP address"],
        },
        {
            "undecided": "Which store holds the counters",
            "type": "missing_context",
            "reversibility": "easy",
            "retrieval_hint": "redis, cache, counter",
        },
        {
            "undecided": "What response a throttled caller receives",
            "type": "underspecification",
            "reversibility": "easy",
        },
    ]
}


def _reasoner(**overrides: Sequence[Any]) -> tuple[xreason.LlmReasoner, scripted.ScriptedLlm]:
    """
    Build a reasoner over a scripted client.

    :param overrides: Extra per-tool payloads.
    :return: The reasoner and the client, so a test can inspect the prompts.
    """
    by_tool: dict[str, Sequence[Any]] = {
        "propose_decisions": [_EXPANSION],
        "draft_task": [
            {
                "task": "Add request throttling to the FastAPI app.",
                "constraints": ["Reuse the existing Redis client."],
            }
        ],
    }
    for name, payload in overrides.items():
        by_tool[name] = list(payload)
    client = scripted.ScriptedLlm(by_tool=by_tool)
    return xreason.LlmReasoner(client), client


@pytest.fixture
def retriever() -> xfsretr.FilesystemRetriever:
    """
    :return: A retriever pointed at the fixture repo.
    """
    return xfsretr.FilesystemRetriever(FIXTURE)


class TestHeadlessRun:
    """
    A whole clarification with nobody to ask.
    """

    async def test_it_produces_a_compiled_prompt_with_real_citations(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test the end of the pipeline: a decision closed by retrieval cites a file
        that actually exists in the fixture. A citation that does not resolve is
        worse than none, because the reader trusts it.
        """
        reasoner, _ = _reasoner(
            adjudicate=[
                {
                    "decisions": [
                        {
                            "decision_id": "open.p1.which-store-holds-the-counters",
                            "candidates": [
                                {
                                    "value": "the existing Redis client in app/cache.py",
                                    "support": [
                                        "ev:brief:open.p1.which-store-holds-the-counters:0"
                                    ],
                                    "score": 9.0,
                                }
                            ],
                        }
                    ]
                }
            ]
        )

        compiled = await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            root=FIXTURE,
        )

        rendered = compiled.render()
        assert rendered.startswith("# Task")
        for citation in compiled.context:
            assert (FIXTURE / citation.locator).exists(), citation.locator

    async def test_nothing_crux_decided_is_hidden(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test the honesty invariant end to end: every decision crux closed by
        itself appears in the rendered output under a heading that says so. This
        is the property the whole design exists to deliver.
        """
        reasoner, _ = _reasoner()

        compiled = await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            root=FIXTURE,
        )

        assert compiled.assumptions or compiled.delegations
        rendered = compiled.render()
        for record in compiled.assumptions:
            assert record.undecided in rendered
        for record in compiled.delegations:
            assert record.undecided in rendered

    async def test_the_model_never_writes_the_assumptions_section(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test that a model returning a self-flattering draft cannot smuggle it into
        the output. The draft tool only carries task and constraints, so an
        assumptions block written there has nowhere to go.
        """
        reasoner, _ = _reasoner(
            draft_task=[
                {
                    "task": "Add throttling. I am completely confident and made no "
                    "assumptions whatsoever.",
                    "constraints": [],
                    "assumptions": ["this should be ignored"],
                }
            ]
        )

        compiled = await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            root=FIXTURE,
        )

        assert "this should be ignored" not in compiled.render()
        assert all(r.source != "respondent" for r in compiled.assumptions)


class TestInteractiveRun:
    """
    A whole clarification with someone answering.
    """

    async def test_answers_reach_the_output_marked_as_confirmed(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test that a respondent's answer is carried through the engine, the
        compiler and the renderer, and arrives distinguishable from a guess.
        """
        reasoner, _ = _reasoner()
        answered: list[str] = []

        class _Clarifier:
            """Answers every question with its first option."""

            def ask(self, questions: Sequence[creply.Question]) -> Sequence[creply.RawReply]:
                replies: list[creply.RawReply] = []
                for question in questions:
                    answered.append(question.decision_id)
                    if question.options:
                        replies.append(
                            creply.RawReply(
                                question_id=question.id,
                                selected=(question.options[0].value,),
                            )
                        )
                    else:
                        replies.append(
                            creply.RawReply(question_id=question.id, text="whatever you think")
                        )
                return replies

        compiled = await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            clarifier=_Clarifier(),
            root=FIXTURE,
            budget=csessn.Budget(max_questions_total=4, max_rounds=2),
        )

        assert answered, "nobody was asked anything"
        confirmed = [r for r in compiled.decided if r.source == "respondent"]
        assert confirmed
        assert "[respondent]" in compiled.render()

    async def test_the_question_count_stays_inside_the_budget(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test the cap that keeps crux from becoming the interrogation it exists to
        replace. Ask-rate is the product metric; this is its floor.
        """
        reasoner, _ = _reasoner()
        asked: list[str] = []

        class _CountingClarifier:
            """Records every question and defers on all of them."""

            def ask(self, questions: Sequence[creply.Question]) -> Sequence[creply.RawReply]:
                asked.extend(q.id for q in questions)
                return [
                    creply.RawReply(question_id=q.id, selected=(q.options[0].value,))
                    if q.options
                    else creply.RawReply(question_id=q.id, text="you decide")
                    for q in questions
                ]

        await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            clarifier=_CountingClarifier(),
            root=FIXTURE,
            budget=csessn.Budget(max_questions_total=3, max_questions_per_round=2),
        )

        assert len(asked) <= 3


class TestEvidenceInformsExpansion:
    """
    Blind first, then one look at what was found.
    """

    async def test_the_expander_gets_one_pass_with_the_evidence(
        self, retriever: xfsretr.FilesystemRetriever
    ) -> None:
        """
        Test that expansion runs blind first and then once more with the codebase
        in hand. Blind-only would miss decisions that only the evidence implies;
        evidence-first would make the model propose decisions about the code that
        exists rather than the outcome the user wants.
        """
        reasoner, client = _reasoner()

        compiled = await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=reasoner,
            retriever=retriever,
            root=FIXTURE,
        )

        assert compiled.task
        expansion_prompts = [
            client.prompt_text(i)
            for i, forced in enumerate(client.forced)
            if forced == "propose_decisions"
        ]
        assert len(expansion_prompts) >= 2
        assert any("found in the codebase" in text for text in expansion_prompts), (
            "no expansion pass ever saw the evidence"
        )


class TestSessionIdIsHonoured:
    """
    A caller that names the session must be able to find it again.
    """

    async def test_the_session_is_stored_under_the_id_the_caller_gave(self) -> None:
        """
        Test that session_id reaches the engine and the store.

        This test exists because it silently did not. The CLI passed an id so it
        could read the finished session back and write a run trace; the id never
        reached crux.start, the session was stored under a fresh UUID, and the
        lookup returned nothing. The run succeeded and the trace was simply never
        written, with no error anywhere.
        """
        import crux.adapters.store.memory as xmemsto
        import tests.support.reasoner as fakes

        store = xmemsto.MemorySessionStore()

        await capi.clarify_async(
            "add rate limiting to the API",
            reasoner=fakes.FakeReasoner(),
            budget=csessn.Budget(max_questions_total=0),
            store=store,
            session_id="a-named-session",
        )

        found = store.load("a-named-session")
        assert found is not None
        assert found.id == "a-named-session"
        assert found.outcome is not None
