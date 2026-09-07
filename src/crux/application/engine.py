"""
The phase machine: seed, expand, retrieve, route, ask, compile.

Yield/resume is the core here, and the blocking wrapper in :mod:`crux.api` sits
on top of it. The engine holds the ports; the session it returns holds only data,
so a host can ``start()`` on one worker, persist the session anywhere JSON goes,
and ``resume()`` on another an hour later.

Termination is provable rather than hoped for. ``max_passes`` bounds expansion,
``max_rounds`` and ``max_questions_total`` bound asking,
``max_exchanges_per_node`` bounds counter-questions, and the route table treats
budget exhaustion as a legal way to close every remaining decision.

Import as:

import crux.application.engine as aengine
"""

from __future__ import annotations

import logging
import uuid

import crux.application.compile as acomp
import crux.application.expansion as aexpand
import crux.application.questions as aquest
import crux.application.replies as arepl
import crux.application.retrieval as aretr
import crux.application.routing as aroute
import crux.application.stopping as astop
import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.graph as cgraph
import crux.domain.replies as creply
import crux.domain.session as csessn
import crux.packs.base as kspec
import crux.ports.reasoner as preason
import crux.ports.retrieval as pretr

_LOG = logging.getLogger(__name__)

# A retype turns one decision into another sort of decision, which then needs
# routing again. Two rounds settles every path through the table; the third is
# insurance against a pack that retypes into something that retypes again.
_MAX_ROUTE_PASSES = 3


class Crux:
    """
    The clarification agent. Holds the ports; the sessions it returns do not.
    """

    def __init__(
        self,
        *,
        reasoner: preason.Reasoner,
        retriever: pretr.Retriever | None = None,
        pack_ids: tuple[str, ...] = ("software",),
        budget: csessn.Budget | None = None,
    ) -> None:
        """
        :param reasoner: The model boundary. The only thing crux cannot run without.
        :param retriever: Where to look things up. Without one, every fact about
            the world falls through to a default or a delegation — which is a
            supported mode, not a degraded one.
        :param pack_ids: Which decision packs to seed from.
        :param budget: Caps on passes, rounds and questions. Setting
            ``max_questions_total=0`` runs crux headless: it asks nothing and
            states every open decision as an assumption.
        """
        self._reasoner = reasoner
        self._retriever = retriever
        self._pack_ids = pack_ids
        self._budget = budget or csessn.Budget()

    # ## Public surface

    async def start(
        self,
        prompt: str,
        *,
        context: csessn.SessionContext | None = None,
        session_id: str | None = None,
    ) -> csessn.Step:
        """
        Begin clarifying a prompt.

        :param prompt: What the user asked for.
        :param context: What the host knows about the world this lives in.
        :param session_id: An id to use, where the host wants to choose it.
        :return: Questions to put to a respondent, or a finished compiled prompt.
        """
        ctx = context or csessn.SessionContext(pack_ids=self._pack_ids)
        session = csessn.Session(
            id=session_id or str(uuid.uuid4()),
            prompt=prompt,
            context=ctx,
            budget=self._budget,
        )
        seeded = kspec.seed(prompt, ctx.pack_ids)
        graph = cgraph.DecisionGraph()
        for decision in seeded:
            graph = graph.add(decision)
        _LOG.info("Seeded %d decisions from packs %s", len(seeded), list(ctx.pack_ids))
        return await self._advance(session.touched(graph=graph, phase="expanding"))

    async def resume(
        self,
        session: csessn.Session,
        replies: tuple[creply.RawReply, ...],
    ) -> csessn.Step:
        """
        Apply a round of answers and carry on.

        :param session: The session handed back by a previous step.
        :param replies: What the respondent sent. Fewer replies than questions is
            allowed; the rest become delegations rather than blocking.
        :return: The next step.
        """
        understood, pairs = arepl.split(session, replies)
        if pairs:
            classified = await self._reasoner.classify(preason.ClassifyRequest(pairs=pairs))
            understood = understood + tuple(classified.replies)
            session = self._spend(session, llm_calls=1)

        session = self._record_transcript(session, replies, understood)
        for reply in understood:
            session = arepl.apply(session, reply)

        session, reasked = await self._answer_counters(session, understood)

        answered = frozenset(reply.question_id for reply in understood)
        session = arepl.resolve_unanswered(session, answered)
        session = session.touched(
            pending=(),
            budget=session.budget.spend(rounds=1),
        )
        notes = {q.decision_id: q.why for q in reasked}
        return await self._advance(session, notes=notes)

    # ## The loop

    async def _advance(
        self,
        session: csessn.Session,
        *,
        notes: dict[str, str] | None = None,
    ) -> csessn.Step:
        """
        Run from wherever the session is until it needs input or finishes.

        :param session: The session to advance.
        :param notes: Wording to carry into a re-asked question, by decision id.
        :return: The next step.
        """
        session = await self._expand(session)
        session = await self._gather(session)
        # Routed before the evidence pass, not after. The expander is shown what
        # already exists, and an unrouted decision looks unsettled even when
        # retrieval has just answered it -- which had it proposing "whether to
        # use redis" immediately after retrieval resolved the dependency
        # question to Redis.
        session, recommendations = self._route_all(session)
        before = len(session.graph.nodes)
        session = await self._reexpand_with_evidence(session)
        if len(session.graph.nodes) != before:
            session, more = self._route_all(session)
            recommendations.update(more)

        if not astop.should_ask(session):
            return await self._finish(session)

        return await self._ask(session, recommendations, notes or {})

    async def _expand(self, session: csessn.Session) -> csessn.Session:
        """
        Run expansion passes until they saturate or the budget stops them.

        :param session: The session to expand.
        :return: The session with a bigger graph and a longer pass log.
        """
        session = session.touched(phase="expanding")
        while astop.should_expand(session):
            session = await self._expand_once(session, saw_evidence=False)
        return session

    async def _expand_once(self, session: csessn.Session, *, saw_evidence: bool) -> csessn.Session:
        """
        Run exactly one expansion pass.

        :param session: The session to expand.
        :param saw_evidence: Whether this pass has retrieved evidence to work
            from, recorded so the saturation curve can tell the two apart.
        :return: The session, with the pass folded in.
        """
        index = session.passes.count + 1
        result = await self._reasoner.expand(
            preason.ExpansionRequest(
                prompt=session.prompt,
                lens=aexpand.lens_for(index),
                existing=aexpand.sketches(session.graph),
                evidence_digest=self._digest(session) if saw_evidence else (),
                host_notes=session.context.host_notes,
            )
        )
        graph, record = aexpand.absorb(
            session.graph,
            result,
            pass_index=index,
            pack_ids=session.context.pack_ids,
            saw_evidence=saw_evidence,
        )
        return self._spend(
            session.touched(graph=graph, passes=session.passes.with_record(record)),
            llm_calls=1,
        )

    async def _reexpand_with_evidence(self, session: csessn.Session) -> csessn.Session:
        """
        Give the expander one look at what retrieval found.

        The first passes run blind, deliberately: showing a model the repo before
        asking what is undecided makes it propose decisions about the code that
        exists rather than about the outcome the user wants. But evidence does
        imply decisions of its own — finding two user models raises a question
        that reading the prompt alone never would — so it gets exactly one pass
        once the looking-up is done.

        :param session: The session, after retrieval.
        :return: The session, possibly with more decisions.
        """
        if not session.evidence.items:
            return session
        if any(record.saw_evidence for record in session.passes.records):
            return session
        # Deliberately exempt from max_passes. That cap bounds *blind* expansion,
        # and letting the two compete starved the evidence pass exactly when
        # blind expansion had been most productive — a live run spent all three
        # passes blind and never looked at the codebase at all. One extra call is
        # a better trade than never seeing the evidence.
        return await self._expand_once(session, saw_evidence=True)

    async def _gather(self, session: csessn.Session) -> csessn.Session:
        """
        Raise every outstanding brief, spend them in one batch, adjudicate once.

        :param session: The session to gather evidence for.
        :return: The session with an evidence graph.
        """
        if self._retriever is None:
            return session
        briefs = aretr.briefs_for(session.graph, pack_ids=session.context.pack_ids)
        if not briefs:
            return session

        session = session.touched(phase="retrieving")
        evidence = session.evidence.with_briefs(briefs)
        results = await aretr.run(self._retriever, briefs)
        evidence = evidence.with_results(results)

        graph = session.graph
        for brief in briefs:
            graph = graph.replace(graph.get(brief.decision_id).with_brief(brief.id))
        session = session.touched(graph=graph, evidence=evidence)

        request = aretr.adjudication_request(session.prompt, session.graph, evidence)
        if not request.decisions:
            return session
        adjudicated = await self._reasoner.adjudicate(request)
        for decision_id, candidates in adjudicated.candidates.items():
            if decision_id in session.graph:
                evidence = evidence.with_candidates(decision_id, candidates)
        return self._spend(session.touched(evidence=evidence), llm_calls=1)

    def _route_all(self, session: csessn.Session) -> tuple[csessn.Session, dict[str, str]]:
        """
        Send every open decision down the table until nothing changes.

        A retype turns one sort of decision into another, so routing runs to a
        fixpoint rather than once. Bounded, because an unbounded fixpoint over
        model-supplied types is a hang waiting to happen.

        :param session: The session to route.
        :return: The routed session, and any recommendations the table earned.
        """
        session = session.touched(phase="routing")
        recommendations: dict[str, str] = {}
        for _ in range(_MAX_ROUTE_PASSES):
            # Evidence from decisions resolved on the previous pass has to reach
            # the decisions it informs BEFORE those are routed, or the sharing
            # arrives after the target has already closed.
            session = session.touched(
                evidence=aretr.propagate_informs(session.graph, session.evidence)
            )
            graph, again, round_recs = self._route_once(session)
            recommendations.update(round_recs)
            session = session.touched(graph=graph.cascade())
            if not again:
                break
        return session, recommendations

    def _route_once(
        self, session: csessn.Session
    ) -> tuple[cgraph.DecisionGraph, bool, dict[str, str]]:
        """
        One pass of the route table over every open decision.

        :param session: The session to route.
        :return: The new graph, whether another pass is owed, and any
            recommendations.
        """
        graph = session.graph
        # The wait check reads the graph as it was when the pass began. Reading
        # the mutating one lets a source resolve and its target route in the same
        # pass, which is exactly early enough to miss the evidence sharing.
        at_pass_start = session.graph
        again = False
        recommendations: dict[str, str] = {}
        for node in graph.open_decisions():
            if self._waits_on_evidence(at_pass_start, node):
                # An `informs` edge does not gate existence, but routing a
                # decision before the evidence bearing on it has arrived would
                # decide it on less than crux knows. Deferred to the next pass.
                again = True
                continue
            outcome = aroute.route(
                node,
                candidates=session.evidence.candidates_for(node.id),
                pack_default=kspec.default_for(node.id, session.context.pack_ids),
                budget_spent=session.budget.is_exhausted,
            )
            if isinstance(outcome, aroute.Resolve):
                updated = node
                if outcome.options:
                    updated = updated.with_options(outcome.options)
                graph = graph.replace(
                    updated.resolve(
                        outcome.value,
                        source=outcome.source,
                        dominant=outcome.dominant,
                        evidence=outcome.evidence,
                        rationale=outcome.rationale,
                    )
                )
            elif isinstance(outcome, aroute.Retype):
                graph = graph.replace(
                    node.retype(outcome.new_type, options=outcome.options or None)
                )
                again = True
            else:
                if outcome.options:
                    graph = graph.replace(node.with_options(outcome.options))
                if outcome.recommended is not None:
                    recommendations[node.id] = outcome.recommended
        return graph, again, recommendations

    @staticmethod
    def _waits_on_evidence(graph: cgraph.DecisionGraph, node: cdecis.OpenDecision) -> bool:
        """
        Say whether a decision should wait for evidence that bears on it.

        :param graph: The graph so far.
        :param node: The decision to test.
        :return: Whether any decision informing it is still open.
        """
        return any(
            edge.source_id in graph and graph.get(edge.source_id).is_open
            for edge in node.edges("informs")
        )

    async def _ask(
        self,
        session: csessn.Session,
        recommendations: dict[str, str],
        notes: dict[str, str],
    ) -> csessn.Step:
        """
        Word the frontier and hand it back to the host.

        :param session: The routed session.
        :param recommendations: Pre-filled answers the table earned.
        :param notes: Wording to carry into a re-asked question.
        :return: The questions, and the session to resume with.
        """
        frontier = session.graph.frontier()[: session.budget.max_questions_per_round]
        frontier = frontier[: session.budget.questions_left]
        phrased = await self._reasoner.phrase(
            preason.PhraseRequest(prompt=session.prompt, decisions=frontier)
        )
        questions = aquest.build(
            frontier,
            phrased,
            session.evidence,
            recommendations=recommendations,
        )
        questions = tuple(
            q.model_copy(update={"why": notes[q.decision_id]})
            if q.decision_id in notes and notes[q.decision_id]
            else q
            for q in questions
        )
        session = self._spend(
            session.touched(phase="awaiting_input", pending=questions),
            llm_calls=1,
        )
        session = session.touched(budget=session.budget.spend(questions=len(questions)))
        return csessn.NeedsInput(session=session, questions=questions)

    async def _finish(self, session: csessn.Session) -> csessn.Step:
        """
        Close anything still open, then compile.

        :param session: The session to finish.
        :return: The compiled prompt.
        """
        session = session.touched(phase="compiling")
        graph = session.graph
        for node in graph.open_decisions():
            graph = graph.replace(
                node.resolve(
                    "left to the downstream agent",
                    source="delegated",
                    rationale="not worth interrupting anyone over",
                )
            )
        session = session.touched(graph=graph)
        compiled = await acomp.compile_prompt(session, self._reasoner)
        session = self._spend(
            session.touched(phase="done", outcome=compiled, pending=()), llm_calls=1
        )
        return csessn.Done(session=session, compiled=compiled)

    # ## Internals

    async def _answer_counters(
        self,
        session: csessn.Session,
        understood: tuple[creply.ClassifiedReply, ...],
    ) -> tuple[csessn.Session, tuple[creply.Question, ...]]:
        """
        Answer whatever the respondent asked back, or delegate and move on.

        :param session: The session to change.
        :param understood: Every classified reply this round.
        :return: The new session and the questions to re-ask.
        """
        request = arepl.counter_requests(session, understood)
        if not request.items:
            return session, ()
        result = await self._reasoner.answer_counter(request)
        session = self._spend(session, llm_calls=1)
        return arepl.apply_counter_answers(session, result)

    def _record_transcript(
        self,
        session: csessn.Session,
        raws: tuple[creply.RawReply, ...],
        understood: tuple[creply.ClassifiedReply, ...],
    ) -> csessn.Session:
        """
        Keep what was asked and what came back.

        The graph is the state; the transcript is the history, and it is what
        makes "why did you ask me that?" answerable after the fact.

        :param session: The session to change.
        :param raws: What the respondent sent.
        :param understood: What crux made of it.
        :return: The session with a longer transcript.
        """
        by_id = {reply.question_id: reply for reply in understood}
        exchanges = list(session.transcript)
        for raw in raws:
            question = session.question_for(raw.question_id)
            classified = by_id.get(raw.question_id)
            if question is None or classified is None:
                continue
            exchanges.append(csessn.Exchange(question=question, reply=raw, classified=classified))
        return session.touched(transcript=tuple(exchanges))

    def _digest(self, session: csessn.Session) -> tuple[str, ...]:
        """
        Summarise the evidence for the expander, without pasting the whole graph.

        :param session: The session to summarise.
        :return: One locator-and-excerpt line per item, capped.
        """
        lines = [
            f"{item.locator}: {item.excerpt}"[:200]
            for item in list(session.evidence.items.values())[:40]
        ]
        return tuple(lines)

    def _spend(self, session: csessn.Session, *, llm_calls: int = 0) -> csessn.Session:
        """
        Record model consumption.

        :param session: The session to change.
        :param llm_calls: How many calls were made.
        :return: The session with the budget updated.
        """
        return session.touched(budget=session.budget.spend(llm_calls=llm_calls))


def open_ids(graph: cgraph.DecisionGraph) -> tuple[str, ...]:
    """
    Name what is still open, for tests and for logging.

    :param graph: The graph to inspect.
    :return: The ids.
    """
    return tuple(node.id for node in graph.open_decisions())


def candidates_of(
    evidence: cevid.EvidenceGraph, decision: cdecis.OpenDecision
) -> tuple[cevid.Candidate, ...]:
    """
    Read a decision's adjudicated candidates.

    :param evidence: The session's evidence.
    :param decision: The decision to look up.
    :return: Its candidates.
    """
    return evidence.candidates_for(decision.id)
