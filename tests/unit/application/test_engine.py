"""
Tier-2 tests: the whole loop, driven by fakes, with no JSON and no network.

These are the named tests the design committed to. Each one guards a property
that would quietly ruin the product if it broke — a loop that never terminates,
a question crux cannot afford, a guess that never reaches the output.
"""

from __future__ import annotations

import pytest

import crux.application.engine as aengine
import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.replies as creply
import crux.domain.session as csessn
import crux.packs.software  # noqa: F401 - registers the pack this suite seeds from
import crux.ports.reasoner as preason
import tests.support.reasoner as fakes

HEADLESS = csessn.Budget(max_questions_total=0)


def _reply(question_id: str, *, text: str = "", selected: tuple[str, ...] = ()) -> creply.RawReply:
    """
    Build a raw reply.

    :param question_id: The question being answered.
    :param text: Free text, if any.
    :param selected: Options clicked, if any.
    :return: The reply.
    """
    return creply.RawReply(question_id=question_id, text=text, selected=selected)


def _find(session: csessn.Session, text: str) -> cdecis.OpenDecision:
    """
    Find a decision by what it says, rather than by a hand-written slug.

    Freeform ids are slugs truncated to a fixed width, so hardcoding one in a
    test makes the test brittle for no benefit.

    :param session: The session to search.
    :param text: A distinctive fragment of the decision's text.
    :return: The decision.
    """
    return next(n for n in session.graph.nodes.values() if text in n.undecided)


class TestHeadless:
    """
    No respondent attached is a supported mode, not a degraded one.
    """

    async def test_a_zero_question_budget_completes_without_asking(self) -> None:
        """
        Test that crux runs to a compiled prompt with nobody to ask. Plenty of
        hosts embed it with no human in the loop at all, and raising there would
        make it unusable in exactly the setting a library is most often used.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS)

        step = await crux.start("add rate limiting to the API")

        assert isinstance(step, csessn.Done)
        assert step.session.phase == "done"
        assert step.compiled.task

    async def test_every_unasked_decision_reaches_the_output(self) -> None:
        """
        Test the honesty invariant: nothing crux decided for itself is dropped
        silently. A decision resolved but absent from the output is a guess the
        user can never find, which is the failure this whole design exists to
        prevent.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS)

        step = await crux.start("add rate limiting to the API")

        assert isinstance(step, csessn.Done)
        resolved = {n.id for n in step.session.graph.resolved()}
        rendered = {r.id for r in step.compiled.all_decisions}
        assert resolved == rendered

        unconfirmed = {
            n.id
            for n in step.session.graph.resolved()
            if n.resolution is not None and n.resolution.source != "respondent"
        }
        surfaced = {r.id for r in step.compiled.assumptions + step.compiled.delegations}
        assert unconfirmed <= surfaced | {r.id for r in step.compiled.decided}


class TestAsking:
    """
    Which decisions reach a respondent, and how many.
    """

    async def test_only_expensive_irreversible_decisions_are_asked(self) -> None:
        """
        Test that the frontier crux emits is the ask-worthy set and nothing else.
        The software pack's build-scope decision is high and hard; its test
        expectations are low and easy and must never be asked.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner())

        step = await crux.start("add rate limiting to the API")

        assert isinstance(step, csessn.NeedsInput)
        asked = {q.decision_id for q in step.questions}
        assert "software.scope.build" in asked
        assert "software.tests.expectations" not in asked

    async def test_a_round_never_exceeds_the_per_round_cap(self) -> None:
        """
        Test that a host is never handed more questions than the budget allows,
        however many decisions clear the threshold.
        """
        crux = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                expansions=[
                    preason.ExpansionResult(
                        proposed=tuple(
                            fakes.proposal(f"decision number {i} about throttling")
                            for i in range(8)
                        )
                    )
                ]
            )
        )

        step = await crux.start("add rate limiting to the API")

        assert isinstance(step, csessn.NeedsInput)
        assert len(step.questions) <= step.session.budget.max_questions_per_round

    async def test_every_question_offers_free_text(self) -> None:
        """
        Test the invariant that makes the other five reply kinds reachable. A
        respondent who can only click cannot say the question is wrong.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner())

        step = await crux.start("add rate limiting to the API")

        assert isinstance(step, csessn.NeedsInput)
        assert step.questions
        assert all(q.free_text for q in step.questions)


class TestReplyKinds:
    """
    All six things a respondent can do.
    """

    async def test_a_bare_click_costs_no_model_call(self) -> None:
        """
        Test that the commonest reply is classified deterministically. A
        classifier that occasionally read a click as a proposal would be
        maddening, and paying a model to read a button press is waste.
        """
        reasoner = fakes.FakeReasoner()
        crux = aengine.Crux(reasoner=reasoner)
        step = await crux.start("add rate limiting to the API")
        assert isinstance(step, csessn.NeedsInput)

        before = reasoner.count("classify")
        await crux.resume(
            step.session,
            tuple(_reply(q.id, selected=("mvp",)) for q in step.questions),
        )

        assert reasoner.count("classify") == before

    async def test_defer_resolves_as_a_delegation_not_an_answer(self) -> None:
        """
        Test that "you decide" is recorded honestly. crux did not decide it
        either, so it must reach the downstream agent as an open judgement rather
        than as a confirmed fact.
        """
        step = await self._ask_once()
        question = step.questions[0]
        crux = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                classifications=[
                    preason.ClassifyResult(replies=(creply.DeferReply(question_id=question.id),))
                ]
            ),
            budget=HEADLESS,
        )

        done = await crux.resume(step.session, (_reply(question.id, text="you decide"),))

        assert isinstance(done, csessn.Done)
        node = done.session.graph.get(question.decision_id)
        assert node.resolution is not None
        assert node.resolution.source == "delegated"

    async def test_proposal_dirties_saturation_and_triggers_one_more_pass(self) -> None:
        """
        Test that an answer nobody anticipated forces another expansion pass. A
        value outside the option set usually implies decisions nothing expanded
        against, so treating it as an ordinary answer would leave those unfound.
        """
        step = await self._ask_once()
        question = step.questions[0]
        reasoner = fakes.FakeReasoner(
            classifications=[
                preason.ClassifyResult(
                    replies=(
                        creply.ProposalReply(
                            question_id=question.id,
                            option=cdecis.Option(value="webhooks", source="proposal"),
                        ),
                    )
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, budget=HEADLESS)

        done = await crux.resume(step.session, (_reply(question.id, text="what about webhooks?"),))

        assert isinstance(done, csessn.Done)
        assert done.session.passes.count >= 2
        node = done.session.graph.get(question.decision_id)
        assert node.value == "webhooks"

    async def test_reject_invalidates_the_decision_and_reexpands(self) -> None:
        """
        Test that "that is the wrong question" is acted on rather than swallowed.
        It is the most informative reply crux can get and the one a click-only
        interface can never deliver.
        """
        step = await self._ask_once()
        question = step.questions[0]
        crux = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                classifications=[
                    preason.ClassifyResult(
                        replies=(
                            creply.RejectReply(
                                question_id=question.id, reason="we already decided this"
                            ),
                        )
                    )
                ]
            ),
            budget=HEADLESS,
        )

        done = await crux.resume(
            step.session, (_reply(question.id, text="wrong question, we settled that"),)
        )

        assert isinstance(done, csessn.Done)
        node = done.session.graph.get(question.decision_id)
        assert node.status == "invalidated"
        assert node.invalidation_reason == "we already decided this"
        assert done.session.passes.count >= 2

    async def test_counter_question_terminates_after_two_exchanges(self) -> None:
        """
        Test the loop guard. A model with no evidence still produces confident
        prose, so the bound is a hard count rather than a quality judgement —
        without it a respondent and crux could haggle for ever.
        """
        step = await self._ask_once()
        question = step.questions[0]
        reasoner = fakes.FakeReasoner(
            classifications=[
                preason.ClassifyResult(
                    replies=(
                        creply.CounterQuestionReply(
                            question_id=question.id, asks="what do you mean by scope?"
                        ),
                    )
                )
            ],
            counters=[
                preason.CounterResult(
                    answers=(
                        preason.CounterAnswer(
                            question_id=question.id,
                            answer="Whether to build retries and dead-lettering.",
                            answered=True,
                        ),
                    )
                )
            ],
        )
        crux = aengine.Crux(reasoner=reasoner)

        second = await crux.resume(step.session, (_reply(question.id, text="what do you mean?"),))

        assert isinstance(second, csessn.NeedsInput)
        node = second.session.graph.get(question.decision_id)
        assert node.ask_exchanges == 1
        assert node.is_open

        # A second counter-question hits the guard and the decision goes
        # downstream rather than being haggled over again.
        reasked = next(q for q in second.questions if q.decision_id == question.decision_id)
        crux2 = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                classifications=[
                    preason.ClassifyResult(
                        replies=(
                            creply.CounterQuestionReply(
                                question_id=reasked.id, asks="still unclear"
                            ),
                        )
                    )
                ],
                counters=[
                    preason.CounterResult(
                        answers=(
                            preason.CounterAnswer(
                                question_id=reasked.id, answer="Still this.", answered=True
                            ),
                        )
                    )
                ],
            ),
            budget=HEADLESS,
        )
        done = await crux2.resume(second.session, (_reply(reasked.id, text="still unclear"),))

        assert isinstance(done, csessn.Done)
        final = done.session.graph.get(question.decision_id)
        assert final.is_resolved
        assert final.resolution is not None
        assert final.resolution.source == "delegated"

    async def test_an_unanswerable_counter_question_delegates_immediately(self) -> None:
        """
        Test that crux admits it does not know rather than inventing an answer
        and re-asking on the strength of it.
        """
        step = await self._ask_once()
        question = step.questions[0]
        crux = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                classifications=[
                    preason.ClassifyResult(
                        replies=(
                            creply.CounterQuestionReply(
                                question_id=question.id, asks="what does the repo use?"
                            ),
                        )
                    )
                ]
            ),
            budget=HEADLESS,
        )

        done = await crux.resume(step.session, (_reply(question.id, text="what does it use?"),))

        assert isinstance(done, csessn.Done)
        node = done.session.graph.get(question.decision_id)
        assert node.resolution is not None
        assert node.resolution.source == "delegated"

    async def test_an_unanswered_question_becomes_a_delegation(self) -> None:
        """
        Test that silence does not block. A respondent who closes the tab still
        gets a compiled prompt, with the unanswered decision stated as open.
        """
        step = await self._ask_once()
        crux = aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS)

        done = await crux.resume(step.session, ())

        assert isinstance(done, csessn.Done)
        assert all(n.is_resolved or n.status != "open" for n in done.session.graph.nodes.values())

    @staticmethod
    async def _ask_once() -> csessn.NeedsInput:
        """
        Drive a session to its first round of questions.

        :return: The step holding them.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner())
        step = await crux.start("add rate limiting to the API")
        assert isinstance(step, csessn.NeedsInput)
        return step


class TestRetrievalIsBatched:
    """
    The economic argument for the whole design.
    """

    async def test_every_brief_goes_out_in_one_trip(self) -> None:
        """
        Test that N decisions cost one retrieval call, not N. Four sequential
        explore-agent runs before the first question would cost more than the
        mistake crux exists to prevent, so batching is not an optimisation here —
        it is the thing that makes the layer viable.
        """
        # Deliberately unalike, because near-identical texts are correctly merged
        # by dedup and would make this test measure the wrong thing.
        facts = [
            "which storage backend already exists for counters",
            "how requests are authenticated today",
            "where middleware is registered",
            "what the current deployment topology looks like",
            "which observability stack is in use",
        ]
        retriever = fakes.FakeRetriever()
        crux = aengine.Crux(
            reasoner=fakes.FakeReasoner(
                expansions=[
                    preason.ExpansionResult(
                        proposed=tuple(
                            fakes.proposal(
                                text,
                                decision_type="missing_context",
                                retrieval_hint="modules",
                            )
                            for text in facts
                        )
                    )
                ]
            ),
            retriever=retriever,
            budget=HEADLESS,
        )

        await crux.start("add rate limiting to the API")

        assert retriever.trips == 1
        assert len(retriever.batches[0]) >= len(facts)

    async def test_adjudication_is_one_call_for_every_decision(self) -> None:
        """
        Test that candidates are worked out in a single batched call. One call
        per decision would multiply the cost of the step that exists to keep
        crux cheap.
        """
        reasoner = fakes.FakeReasoner()
        retriever = fakes.FakeRetriever(
            {
                "software.target.files": (
                    fakes.item("software.target.files", 0, "src/api/routes.py"),
                ),
                "software.deps.policy": (fakes.item("software.deps.policy", 0, "pyproject.toml"),),
            }
        )
        crux = aengine.Crux(reasoner=reasoner, retriever=retriever, budget=HEADLESS)

        await crux.start("add rate limiting to the API")

        assert reasoner.count("adjudicate") == 1
        request = reasoner.last("adjudicate")
        assert isinstance(request, preason.AdjudicationRequest)
        assert len(request.decisions) >= 2


class TestSerialisation:
    """
    The invariant that makes yield/resume work across a process boundary.
    """

    async def test_a_session_round_trips_through_json_at_every_phase(self) -> None:
        """
        Test that a host can persist the session anywhere JSON goes and resume
        from it. Without this, yield/resume is only ever an in-process trick and
        the whole control-flow choice collapses.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner())
        step = await crux.start("add rate limiting to the API")
        assert isinstance(step, csessn.NeedsInput)

        restored = csessn.Session.model_validate_json(step.session.model_dump_json())
        assert restored == step.session

        done = await aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS).resume(
            restored,
            tuple(_reply(q.id, selected=("mvp",)) for q in step.questions),
        )
        assert isinstance(done, csessn.Done)
        assert csessn.Session.model_validate_json(done.session.model_dump_json()) == done.session

    async def test_a_session_holds_no_port_references(self) -> None:
        """
        Test that nothing but data survives into the session. A port smuggled in
        would serialise as null and resume against a broken object, which is the
        kind of failure that shows up in production and never in a test.
        """
        crux = aengine.Crux(reasoner=fakes.FakeReasoner(), retriever=fakes.FakeRetriever())
        step = await crux.start("add rate limiting to the API")

        dumped = step.session.model_dump(mode="json")
        _assert_json_only(dumped)


def _assert_json_only(value: object, path: str = "session") -> None:
    """
    Walk a dumped model and assert every leaf is a JSON primitive.

    :param value: The value to check.
    :param path: Where in the structure it sits, for the failure message.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_only(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_json_only(item, f"{path}[{index}]")
    else:
        assert isinstance(value, str | int | float | bool | type(None)), (
            f"{path} is {type(value).__name__}, which will not survive JSON"
        )


@pytest.mark.parametrize("prompt", ["add caching", "rename the user endpoint", "make it faster"])
async def test_any_prompt_reaches_a_compiled_prompt(prompt: str) -> None:
    """
    Test that crux terminates on prompts of different shapes. Every loop in the
    engine is bounded by the budget, and this is the coarse check that no
    combination of pack triggers leaves one running.
    """
    crux = aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS)

    step = await crux.start(prompt)

    assert isinstance(step, csessn.Done)
    assert step.compiled.render().startswith("# Task")


class TestInformsSharesEvidence:
    """
    What travels along an `informs` edge, and what deliberately does not.
    """

    async def test_evidence_reaches_the_decision_it_informs(self) -> None:
        """
        Test that a resolved decision's evidence is attached to whatever it
        informs, so the person asked about the target can see the file that bears
        on it. Finding Redis does not answer "how do we identify callers", but it
        does change what answering costs.
        """
        retriever = fakes.FakeRetriever(
            {
                "open.p1.which-store-holds-the-counters": (
                    fakes.item(
                        "open.p1.which-store-holds-the-counters", 0, "app/cache.py", "import redis"
                    ),
                )
            }
        )
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal(
                            "which store holds the counters",
                            decision_type="missing_context",
                            retrieval_hint="redis",
                        ),
                        fakes.proposal(
                            "how callers are identified for counting",
                            edges=(
                                preason.ProposedEdge(
                                    kind="informs",
                                    source_id="open.p1.which-store-holds-the-counters",
                                ),
                            ),
                        ),
                    )
                )
            ],
            adjudications=[
                preason.AdjudicationResult(
                    candidates={
                        "open.p1.which-store-holds-the-counters": (
                            cevid.Candidate(
                                value="redis",
                                score=9.0,
                                support=("ev:open.p1.which-store-holds-the-counters:0",),
                            ),
                        )
                    }
                )
            ],
        )
        crux = aengine.Crux(reasoner=reasoner, retriever=retriever, budget=HEADLESS)

        done = await crux.start("add rate limiting to the API")

        assert isinstance(done, csessn.Done)
        informed = done.session.evidence.by_decision.get(
            "open.p1.how-callers-are-identified-for-counting", ()
        )
        assert "ev:open.p1.which-store-holds-the-counters:0" in informed

    async def test_an_informs_edge_does_not_invent_options_for_its_target(self) -> None:
        """
        Test that the source's candidates stay the source's. "redis" is an answer
        to which store holds the counters; it is not an option for how callers
        are identified, and offering it as one would be crux inventing a value —
        the single thing it must never do.
        """
        retriever = fakes.FakeRetriever(
            {
                "open.p1.which-store-holds-the-counters": (
                    fakes.item("open.p1.which-store-holds-the-counters", 0, "app/cache.py"),
                )
            }
        )
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal(
                            "which store holds the counters",
                            decision_type="missing_context",
                            retrieval_hint="redis",
                        ),
                        fakes.proposal(
                            "how callers are identified for counting",
                            edges=(
                                preason.ProposedEdge(
                                    kind="informs",
                                    source_id="open.p1.which-store-holds-the-counters",
                                ),
                            ),
                        ),
                    )
                )
            ],
            adjudications=[
                preason.AdjudicationResult(
                    candidates={
                        "open.p1.which-store-holds-the-counters": (
                            cevid.Candidate(value="redis", score=9.0),
                        )
                    }
                )
            ],
        )
        crux = aengine.Crux(reasoner=reasoner, retriever=retriever, budget=HEADLESS)

        done = await crux.start("add rate limiting to the API")

        assert isinstance(done, csessn.Done)
        target = done.session.graph.get("open.p1.how-callers-are-identified-for-counting")
        assert all(option.value != "redis" for option in (target.options or ()))


class TestSiblingDependencies:
    """
    A decision proposed in one batch depending on another from the same batch.
    """

    async def test_a_proposal_can_depend_on_a_sibling_by_ref(self) -> None:
        """
        Test that "digest requires channel = email" survives when both are new.

        This test exists because it did not. Freeform ids are minted from slugs
        *after* the model answers, and the expansion tool used to say "use ids
        from the existing list only" — so a proposal could never reference a
        sibling, and the canonical example of the entire design was
        unexpressible. The first live run attached zero edges, and it was crux
        never letting the model try rather than the model declining.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal(
                            "which delivery channel notifications use",
                            ref="channel",
                            options=("email", "in-app"),
                        ),
                        fakes.proposal(
                            "whether email arrives immediately or as a daily digest",
                            ref="digest",
                            edges=(
                                preason.ProposedEdge(
                                    kind="requires", source_id="channel", when_value="email"
                                ),
                            ),
                        ),
                    )
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, budget=HEADLESS)

        done = await crux.start("add notifications")

        assert isinstance(done, csessn.Done)
        digest = _find(done.session, "daily digest")
        channel = _find(done.session, "delivery channel")
        assert [e.source_id for e in digest.edges("requires")] == [channel.id]

    async def test_an_unborn_sibling_is_never_asked_about(self) -> None:
        """
        Test the payoff of the edge above: choosing in-app means the digest
        question does not exist, so nobody is asked something their own earlier
        answer made irrelevant. This is the single clearest thing the graph buys,
        and until refs existed it could not happen at all.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal(
                            "which delivery channel notifications use",
                            ref="channel",
                            options=("email", "in-app"),
                        ),
                        fakes.proposal(
                            "whether email arrives immediately or as a daily digest",
                            ref="digest",
                            edges=(
                                preason.ProposedEdge(
                                    kind="requires", source_id="channel", when_value="email"
                                ),
                            ),
                        ),
                    )
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner)
        step = await crux.start("add notifications")
        assert isinstance(step, csessn.NeedsInput)

        channel_id = _find(step.session, "delivery channel").id
        question = next(q for q in step.questions if q.decision_id == channel_id)
        done = await aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS).resume(
            step.session, (_reply(question.id, selected=("in-app",)),)
        )

        assert isinstance(done, csessn.Done)
        digest_id = _find(done.session, "daily digest").id
        assert digest_id not in {r.id for r in done.compiled.all_decisions}

    async def test_a_ref_nobody_declared_is_dropped_not_crashed(self) -> None:
        """
        Test that a hallucinated ref costs one edge rather than the session.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal(
                            "whether email arrives immediately or as a daily digest",
                            ref="digest",
                            edges=(
                                preason.ProposedEdge(
                                    kind="requires", source_id="nonexistent", when_value="email"
                                ),
                            ),
                        ),
                    )
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, budget=HEADLESS)

        done = await crux.start("add notifications")

        assert isinstance(done, csessn.Done)
        digest = _find(done.session, "daily digest")
        assert digest.depends_on == ()
        assert digest.is_resolved


class TestPackDecisionsAreNotReinvented:
    """
    A later pass restating the pack's own decisions adds nothing.
    """

    async def test_a_two_word_restatement_of_a_pack_decision_is_a_duplicate(self) -> None:
        """
        Test that "Build scope" and friends merge into the pack decisions they
        restate, and a genuinely new decision beside them survives.

        Seen live on every project-sized prompt: pass three proposed "Build
        scope", "Codebase location", "Test coverage" and "Third-party
        dependency policy", each became a fresh node, and the build-scope copy
        reached the question queue next to the real one. Overlap with the pack's
        long wording is far below the duplicate threshold, so the title and the
        authored aliases have to carry it.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal("Build scope"),
                        fakes.proposal("Codebase location"),
                        fakes.proposal("Test coverage"),
                        fakes.proposal("Third-party dependency policy"),
                        fakes.proposal("how long in-flight jobs get to finish"),
                    )
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, budget=HEADLESS)

        done = await crux.start("make the worker shut down gracefully")

        assert isinstance(done, csessn.Done)
        freeform = [n.id for n in done.session.graph.nodes.values() if n.id.startswith("open.")]
        assert len(freeform) == 1
        assert _find(done.session, "in-flight").id == freeform[0]

    async def test_sharing_a_word_with_a_pack_decision_is_not_a_duplicate(self) -> None:
        """
        Test the false-merge direction, which is the expensive one: "rate
        limiting" appears inside the build-scope wording, and a proposal about
        it must not vanish into that decision.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(fakes.proposal("rate limiting threshold per caller"),)
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, budget=HEADLESS)

        done = await crux.start("add rate limiting")

        assert isinstance(done, csessn.Done)
        assert _find(done.session, "threshold per caller").id.startswith("open.p1.")


class TestEdgesNeedAnswers:
    """
    What has to be true before an edge can fire at all.
    """

    async def test_no_edge_can_fire_when_nothing_is_answered(self) -> None:
        """
        Test that a headless run leaves every edge unfired, however many exist.

        This test exists because the saturation experiment measured edge firing
        on headless runs. An edge fires when its source resolves to a particular
        value, and headless resolves everything to "delegated" — so the metric
        the experiment's decision rule reads was structurally pinned at zero, and
        it would have concluded "the graph is ceremony" for a reason with nothing
        to do with the graph.
        """
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(
                    proposed=(
                        fakes.proposal("which channel to use", ref="ch", options=("email", "sms")),
                        fakes.proposal(
                            "how often digests are sent",
                            ref="dg",
                            edges=(
                                preason.ProposedEdge(
                                    kind="requires", source_id="ch", when_value="email"
                                ),
                            ),
                        ),
                    )
                )
            ]
        )
        done = await aengine.Crux(reasoner=reasoner, budget=HEADLESS).start("add notifications")

        assert isinstance(done, csessn.Done)
        attached, fired = done.session.graph.edge_stats().get("requires", (0, 0))
        assert attached == 1, "the edge should exist"
        assert fired == 0, "and it cannot fire, because nothing resolved to a value"

    async def test_an_answered_run_lets_the_edge_fire(self) -> None:
        """
        Test the other half: answering the source with a real option value is
        what makes an edge measurable. This is the arrangement the experiment
        must use, and the contrast with the test above is the whole point.
        """
        expansion = preason.ExpansionResult(
            proposed=(
                fakes.proposal("which channel to use", ref="ch", options=("email", "sms")),
                fakes.proposal(
                    "how often digests are sent",
                    ref="dg",
                    edges=(
                        preason.ProposedEdge(kind="requires", source_id="ch", when_value="email"),
                    ),
                ),
            )
        )
        crux = aengine.Crux(reasoner=fakes.FakeReasoner(expansions=[expansion]))
        step = await crux.start("add notifications")
        assert isinstance(step, csessn.NeedsInput)

        channel_id = _find(step.session, "which channel").id
        question = next(q for q in step.questions if q.decision_id == channel_id)
        done = await aengine.Crux(reasoner=fakes.FakeReasoner(), budget=HEADLESS).resume(
            step.session, (_reply(question.id, selected=("email",)),)
        )

        assert isinstance(done, csessn.Done)
        attached, fired = done.session.graph.edge_stats().get("requires", (0, 0))
        assert (attached, fired) == (1, 1)


class TestEvidencePassSeesResolutions:
    """
    What the evidence-informed expansion pass is shown.
    """

    async def test_it_is_told_what_retrieval_already_settled(self) -> None:
        """
        Test that the evidence pass sees resolved values, not just raw evidence.

        This test exists because a live run proposed "whether to use redis for
        rate limiting" and "whether to use pytest for rate limiting tests"
        immediately after retrieval had resolved those very decisions to Redis
        and pytest. The pass ran before routing, so every decision still looked
        open however conclusive the evidence was.
        """
        retriever = fakes.FakeRetriever(
            {
                "software.deps.policy": (
                    fakes.item("software.deps.policy", 0, "pyproject.toml", "redis>=5.0"),
                )
            }
        )
        reasoner = fakes.FakeReasoner(
            adjudications=[
                preason.AdjudicationResult(
                    candidates={
                        "software.deps.policy": (
                            cevid.Candidate(
                                value="redis", score=9.0, support=("ev:software.deps.policy:0",)
                            ),
                        )
                    }
                )
            ]
        )
        crux = aengine.Crux(reasoner=reasoner, retriever=retriever, budget=HEADLESS)

        await crux.start("add rate limiting to the API")

        evidence_passes = [
            r
            for name, r in reasoner.calls
            if name == "expand" and isinstance(r, preason.ExpansionRequest) and r.evidence_digest
        ]
        assert evidence_passes, "no expansion pass ever saw the evidence"
        sketches = {s.id: s.resolved_to for s in evidence_passes[-1].existing}
        assert sketches.get("software.deps.policy") == "redis", (
            "the evidence pass was shown the dependency decision as unresolved, "
            "so it will propose deciding it again"
        )


class TestEvidencePassIsNotStarved:
    """
    The evidence pass gets to run even when blind expansion used the budget.
    """

    async def test_it_runs_after_the_blind_pass_budget_is_spent(self) -> None:
        """
        Test that max_passes bounds blind expansion, not the evidence pass.

        This test exists because a live run spent all three passes blind and
        never looked at the codebase at all: the evidence pass checked the same
        budget the blind ones had just exhausted, so the more productive blind
        expansion was, the more certain it became that the evidence would be
        ignored.
        """
        retriever = fakes.FakeRetriever(
            {"software.target.files": (fakes.item("software.target.files", 0, "app/routes.py"),)}
        )
        reasoner = fakes.FakeReasoner(
            expansions=[
                preason.ExpansionResult(proposed=(fakes.proposal(f"decision about topic {w}"),))
                for w in ("alpha", "bravo", "charlie", "delta")
            ]
        )
        crux = aengine.Crux(
            reasoner=reasoner,
            retriever=retriever,
            budget=csessn.Budget(max_questions_total=0, max_passes=2),
        )

        done = await crux.start("add rate limiting to the API")

        assert isinstance(done, csessn.Done)
        records = done.session.passes.records
        blind = [r for r in records if not r.saw_evidence]
        informed = [r for r in records if r.saw_evidence]
        assert len(blind) == 2, "blind expansion must still respect max_passes"
        assert len(informed) == 1, "the evidence pass must run anyway"
