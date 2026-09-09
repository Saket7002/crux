"""
Tier-3 tests: the real prompt builders and parsers, over canned tool calls.

This tier exists to catch the failure the other two cannot see — a response
schema changing without its parser, or a model returning something the parser
was not written for.
"""

from __future__ import annotations

import pytest

import crux.adapters.llm.reasoner as xreason
import crux.domain.replies as creply
import crux.errors as cerrors
import crux.ports.reasoner as preason
import tests.support.llm as scripted


class TestExpansion:
    """
    Proposals in, typed decisions out.
    """

    async def test_a_well_formed_response_becomes_proposals_in_order(self) -> None:
        """
        Test that the ranking survives parsing. Cost is derived from position, so
        a parser that reordered the list would silently regrade every decision.
        """
        client = scripted.ScriptedLlm(
            [
                {
                    "decisions": [
                        {"undecided": "first", "type": "ambiguity", "reversibility": "hard"},
                        {
                            "undecided": "second",
                            "type": "underspecification",
                            "reversibility": "easy",
                        },
                    ]
                }
            ]
        )

        result = await xreason.LlmReasoner(client).expand(
            preason.ExpansionRequest(prompt="add caching", lens="what is undecided?")
        )

        assert [p.undecided for p in result.proposed] == ["first", "second"]

    async def test_one_malformed_entry_does_not_lose_the_batch(self) -> None:
        """
        Test that a bad enum value costs one decision rather than the session. A
        model getting one of six entries wrong is a normal Tuesday; failing the
        whole call would turn it into an outage.
        """
        client = scripted.ScriptedLlm(
            [
                {
                    "decisions": [
                        {"undecided": "good", "type": "ambiguity", "reversibility": "hard"},
                        {"undecided": "bad", "type": "not-a-real-type", "reversibility": "hard"},
                        {"undecided": "also good", "type": "vagueness", "reversibility": "easy"},
                    ]
                }
            ]
        )

        result = await xreason.LlmReasoner(client).expand(
            preason.ExpansionRequest(prompt="add caching", lens="what is undecided?")
        )

        assert [p.undecided for p in result.proposed] == ["good", "also good"]

    async def test_the_prompt_names_existing_decisions_by_id(self) -> None:
        """
        Test that the expander is told what already exists, by id. Without the
        ids it cannot attach an edge to anything, and every edge it proposes
        would be dropped as naming an unknown source.
        """
        client = scripted.ScriptedLlm([{"decisions": []}])

        await xreason.LlmReasoner(client).expand(
            preason.ExpansionRequest(
                prompt="add caching",
                lens="what is undecided?",
                existing=(
                    preason.DecisionSketch(
                        id="software.scope.build",
                        undecided="build scope",
                        type="underspecification",
                    ),
                ),
            )
        )

        assert "software.scope.build" in client.prompt_text()

    async def test_a_model_that_ignores_the_tool_is_an_error(self) -> None:
        """
        Test that plain text where a tool call was required fails loudly rather
        than silently producing an empty expansion, which would look like a
        prompt with nothing undecided.
        """
        client = scripted.ScriptedLlm(["I think you should add caching."])

        with pytest.raises(cerrors.ReasonerParseError, match="did not call"):
            await xreason.LlmReasoner(client).expand(
                preason.ExpansionRequest(prompt="add caching", lens="what?")
            )


class TestAdjudication:
    """
    Items in, candidates out.
    """

    async def test_candidates_for_an_unknown_decision_are_dropped(self) -> None:
        """
        Test that a hallucinated decision id cannot inject candidates. They would
        attach to nothing and, worse, could make an unrelated decision look
        settled.
        """
        client = scripted.ScriptedLlm(
            [
                {
                    "decisions": [
                        {"decision_id": "real", "candidates": [{"value": "redis", "score": 9}]},
                        {"decision_id": "invented", "candidates": [{"value": "x", "score": 9}]},
                    ]
                }
            ]
        )

        result = await xreason.LlmReasoner(client).adjudicate(
            preason.AdjudicationRequest(
                prompt="add caching",
                decisions=(
                    preason.AdjudicationItem(
                        decision_id="real", undecided="which store", accept_if="a store is named"
                    ),
                ),
            )
        )

        assert set(result.candidates) == {"real"}


class TestClassification:
    """
    Free text in, one of six reply kinds out.
    """

    @pytest.mark.parametrize(
        ("entry", "expected"),
        [
            ({"kind": "choice", "value": "mvp"}, creply.ChoiceReply),
            ({"kind": "value", "value": "per-user"}, creply.ValueReply),
            ({"kind": "question", "asks": "what do you mean?"}, creply.CounterQuestionReply),
            ({"kind": "proposal", "value": "webhooks"}, creply.ProposalReply),
            ({"kind": "defer"}, creply.DeferReply),
            ({"kind": "reject", "reason": "already settled"}, creply.RejectReply),
        ],
        ids=["choice", "value", "question", "proposal", "defer", "reject"],
    )
    async def test_every_reply_kind_maps_to_its_arm(
        self, entry: dict[str, str], expected: type
    ) -> None:
        """
        Test that the flat wire format maps onto the right union arm for all six
        kinds. The wire shape is deliberately flatter than the union, so this
        mapping is hand-written and worth checking exhaustively.
        """
        client = scripted.ScriptedLlm([{"replies": [{"question_id": "q1", **entry}]}])

        result = await xreason.LlmReasoner(client).classify(preason.ClassifyRequest())

        assert len(result.replies) == 1
        assert isinstance(result.replies[0], expected)

    async def test_an_unknown_kind_is_dropped_not_crashed(self) -> None:
        """
        Test that an invented reply kind loses one reply rather than the round.
        """
        client = scripted.ScriptedLlm([{"replies": [{"question_id": "q1", "kind": "shrug"}]}])

        result = await xreason.LlmReasoner(client).classify(preason.ClassifyRequest())

        assert result.replies == ()

    async def test_a_choice_with_no_value_is_dropped(self) -> None:
        """
        Test that an answer-shaped reply with no answer in it is discarded. It
        would otherwise resolve a decision to the empty string.
        """
        client = scripted.ScriptedLlm([{"replies": [{"question_id": "q1", "kind": "choice"}]}])

        result = await xreason.LlmReasoner(client).classify(preason.ClassifyRequest())

        assert result.replies == ()


class TestDrafting:
    """
    The model writes prose and nothing else.
    """

    async def test_the_draft_prompt_forbids_writing_assumptions(self) -> None:
        """
        Test that the instruction separating prose from provenance is actually
        sent. The honesty of the output depends on the assumptions block being
        derived from the graph, so a model that starts writing its own would be
        producing exactly the flattering list this design exists to avoid.
        """
        client = scripted.ScriptedLlm([{"task": "Add caching.", "constraints": []}])

        await xreason.LlmReasoner(client).draft(preason.DraftRequest(prompt="add caching"))

        text = client.prompt_text().lower()
        assert "assumption" in text
        assert "do not add an assumptions section" in text

    async def test_a_missing_task_falls_back_to_the_original_prompt(self) -> None:
        """
        Test that an empty draft still produces a usable compiled prompt rather
        than one with no task in it at all.
        """
        client = scripted.ScriptedLlm([{"constraints": ["be fast"]}])

        result = await xreason.LlmReasoner(client).draft(preason.DraftRequest(prompt="add caching"))

        assert result.task == "add caching"
        assert result.constraints == ("be fast",)


class TestDraftTolerance:
    """
    The one operation where prose is still the thing we asked for.
    """

    async def test_a_prose_draft_is_used_rather_than_discarded(self) -> None:
        """
        Test that a model which writes the task instead of calling the tool still
        produces a usable result.

        This test exists because a live run died at the last step: every decision
        was settled, retrieval had run, and the session was thrown away because
        the model answered the "write the task" request by writing the task.
        """
        client = scripted.ScriptedLlm(["Add request throttling to the v2 routes."])

        result = await xreason.LlmReasoner(client).draft(
            preason.DraftRequest(prompt="add rate limiting")
        )

        assert result.task == "Add request throttling to the v2 routes."
        assert result.constraints == ()

    async def test_an_empty_prose_draft_falls_back_to_the_prompt(self) -> None:
        """
        Test that a silent model still yields a compiled prompt with a task in it.
        """
        client = scripted.ScriptedLlm([""])

        result = await xreason.LlmReasoner(client).draft(
            preason.DraftRequest(prompt="add rate limiting")
        )

        assert result.task == "add rate limiting"

    async def test_prose_is_still_an_error_for_operations_that_need_structure(self) -> None:
        """
        Test that the tolerance is confined to draft. An expansion returned as
        prose is unusable, and silently treating it as empty would look like a
        prompt with nothing undecided.
        """
        client = scripted.ScriptedLlm(["I think you should add caching."])

        with pytest.raises(cerrors.ReasonerParseError, match="plain text"):
            await xreason.LlmReasoner(client).expand(
                preason.ExpansionRequest(prompt="add caching", lens="what?")
            )


class TestOperationRouting:
    """
    Each operation asking for its own model.
    """

    async def _run(self, routing: object, payload: dict[str, object]) -> scripted.ScriptedLlm:
        """
        Drive every operation once and hand back the client that saw them.

        :param routing: The routing to install.
        :param payload: A tool payload loose enough for any of the six.
        :return: The client, for inspecting which models were asked for.
        """
        client = scripted.ScriptedLlm(
            by_tool={
                "propose_decisions": [payload],
                "adjudicate": [payload],
                "phrase_questions": [payload],
                "classify_replies": [payload],
                "answer_back": [payload],
                "draft_task": [payload],
            }
        )
        reasoner = xreason.LlmReasoner(client, routing)  # type: ignore[arg-type]
        await reasoner.expand(preason.ExpansionRequest(prompt="p", lens="l"))
        await reasoner.adjudicate(preason.AdjudicationRequest(prompt="p"))
        await reasoner.phrase(preason.PhraseRequest(prompt="p"))
        await reasoner.classify(preason.ClassifyRequest())
        await reasoner.answer_counter(preason.CounterRequest(prompt="p"))
        await reasoner.draft(preason.DraftRequest(prompt="p"))
        return client

    async def test_each_tier_reaches_its_own_model(self) -> None:
        """
        Test the split end to end through the reasoner: the two open-ended
        operations ask for the strong model and the four bounded ones ask for
        the cheap one.
        """
        import crux.adapters.llm.routing as xroute

        routing = xroute.ModelRouting(reasoning="big/one", mechanical="small/one")

        client = await self._run(routing, {})

        assert client.models == [
            "big/one",  # expand
            "big/one",  # adjudicate
            "small/one",  # phrase
            "small/one",  # classify
            "small/one",  # answer_counter
            "small/one",  # draft
        ]

    async def test_no_routing_asks_for_nothing_in_particular(self) -> None:
        """
        Test that the default leaves the model choice to the client, which is
        what keeps every existing caller working unchanged.
        """
        client = await self._run(None, {})

        assert client.models == [None] * 6

    async def test_draft_alone_tolerates_prose(self) -> None:
        """
        Test that only the drafting call says prose is acceptable. If the others
        did, a weak model answering in prose would read as success and the
        fallback chain would never fire.
        """
        client = await self._run(None, {})

        assert client.require_tool == [True, True, True, True, True, False]


class TestSelectPack:
    """
    The pack-selection prompt and its parsing.
    """

    async def test_the_prompt_lists_every_pack_and_the_choice_is_parsed(self) -> None:
        """
        Test that the model is shown each pack id with its description and
        that its answer comes back as written, invented or not; the engine
        owns the fallback.
        """
        client = scripted.ScriptedLlm([{"pack_id": "video", "rationale": "it is a script"}])
        reasoner = xreason.LlmReasoner(client)

        result = await reasoner.select_pack(
            preason.PackSelectRequest(
                prompt="write a script",
                packs=(
                    preason.PackSketch(id="software", description="code changes"),
                    preason.PackSketch(id="video", description="short video scripts"),
                ),
            )
        )

        assert result.pack_id == "video" and result.rationale == "it is a script"
        text = client.prompt_text()
        assert "software: code changes" in text and "video: short video scripts" in text
        assert client.forced == ["select_pack"]
