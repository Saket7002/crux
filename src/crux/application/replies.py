"""
Understanding what came back, and applying it.

Six things a respondent can do with a question, and only two of them are an
answer. The other four are the reason the free-text field exists: asking crux
something back, proposing what it failed to list, handing the judgement over, and
saying the question itself is wrong.

The counter-question loop is bounded here. A model with no evidence will still
produce confident prose, so the guard is not "until crux answers well" — it is a
hard count, after which the decision goes downstream rather than being haggled
over.

Import as:

import crux.application.replies as arepl
"""

from __future__ import annotations

import logging

import crux.domain.decisions as cdecis
import crux.domain.replies as creply
import crux.domain.session as csessn
import crux.ports.reasoner as preason

_LOG = logging.getLogger(__name__)


def split(
    session: csessn.Session,
    raws: tuple[creply.RawReply, ...],
) -> tuple[tuple[creply.ClassifiedReply, ...], tuple[preason.ClassifyPair, ...]]:
    """
    Separate the replies a model is needed for from the ones it is not.

    A click with no free text is the most common reply by far and needs no model
    call to understand. Routing it deterministically keeps the cheapest path free
    and, more to the point, exact: a classifier that occasionally read a click as
    a proposal would be maddening.

    :param session: The session holding the pending questions.
    :param raws: What the respondent sent.
    :return: The replies already understood, and the pairs needing a model.
    """
    understood: list[creply.ClassifiedReply] = []
    pairs: list[preason.ClassifyPair] = []
    for raw in raws:
        question = session.question_for(raw.question_id)
        if question is None:
            _LOG.warning("Ignoring reply to unknown question %s", raw.question_id)
            continue
        choice = creply.as_choice(raw)
        if choice is not None:
            understood.append(choice)
        else:
            pairs.append(preason.ClassifyPair(question=question, reply=raw))
    return tuple(understood), tuple(pairs)


def apply(
    session: csessn.Session,
    reply: creply.ClassifiedReply,
) -> csessn.Session:
    """
    Apply one understood reply to the session.

    A counter-question is not applied here: it resolves nothing, and is handled
    by :func:`apply_counter_answers` once crux has tried to answer it.

    :param session: The session to change.
    :param reply: What the respondent meant.
    :return: The new session.
    """
    question = session.question_for(reply.question_id)
    if question is None:
        _LOG.warning("Ignoring reply to unknown question %s", reply.question_id)
        return session
    if reply.kind == "question":
        return session

    decision = session.graph.get(question.decision_id)
    graph = session.graph
    passes = session.passes

    if reply.kind in ("choice", "value"):
        graph = graph.replace(
            decision.resolve(
                reply.value,
                source="respondent",
                rationale="answered directly",
            )
        )
    elif reply.kind == "defer":
        graph = graph.replace(
            decision.resolve(
                "left to the downstream agent",
                source="delegated",
                rationale="the respondent handed the judgement over",
            )
        )
    elif reply.kind == "proposal":
        # An answer nobody anticipated usually implies decisions nobody expanded,
        # so this dirties saturation as well as resolving the node.
        widened = decision.with_options((decision.options or ()) + (reply.option,))
        graph = graph.replace(
            widened.resolve(
                reply.option.value,
                source="respondent",
                rationale=reply.rationale or "proposed by the respondent",
            )
        )
        passes = passes.dirtied()
    elif reply.kind == "reject":
        graph = graph.replace(decision.invalidate(reply.reason))
        passes = passes.dirtied()

    return session.touched(graph=graph.cascade(), passes=passes)


def counter_requests(
    session: csessn.Session,
    replies: tuple[creply.ClassifiedReply, ...],
) -> preason.CounterRequest:
    """
    Gather the questions the respondent asked crux.

    :param session: The session holding the pending questions and the evidence.
    :param replies: Every understood reply this round.
    :return: The batched request, empty when nobody asked anything back.
    """
    items: list[preason.CounterItem] = []
    for reply in replies:
        if reply.kind != "question":
            continue
        question = session.question_for(reply.question_id)
        if question is None:
            continue
        items.append(
            preason.CounterItem(
                question_id=reply.question_id,
                decision_id=question.decision_id,
                asks=reply.asks,
                evidence=session.evidence.items_for(question.decision_id),
            )
        )
    return preason.CounterRequest(prompt=session.prompt, items=tuple(items))


def apply_counter_answers(
    session: csessn.Session,
    result: preason.CounterResult,
) -> tuple[csessn.Session, tuple[creply.Question, ...]]:
    """
    Answer the respondent's questions, or stop haggling and delegate.

    The guard is a count, not a quality judgement, because a model with no
    evidence still produces confident prose. Two exchanges per decision, then the
    judgement goes downstream — which is honest, and which is what makes the loop
    provably terminate.

    :param session: The session to change.
    :param result: What crux managed to answer.
    :return: The new session, and the questions to re-ask.
    """
    graph = session.graph
    reasked: list[creply.Question] = []
    limit = session.budget.max_exchanges_per_node

    for answer in result.answers:
        question = session.question_for(answer.question_id)
        if question is None:
            continue
        decision = graph.get(question.decision_id)
        if decision.ask_exchanges + 1 >= limit or not answer.answered:
            reason = (
                "the respondent's question could not be answered from the evidence"
                if not answer.answered
                else "two exchanges reached without an answer"
            )
            graph = graph.replace(
                decision.resolve(
                    answer.recommended or "left to the downstream agent",
                    source="delegated",
                    rationale=reason,
                )
            )
            continue
        advanced = decision.asked_again()
        graph = graph.replace(advanced)
        reasked.append(
            question.model_copy(
                update={
                    "id": f"{question.id}+{advanced.ask_exchanges}",
                    "why": answer.answer,
                    "exchange": advanced.ask_exchanges,
                    "recommended": answer.recommended or question.recommended,
                }
            )
        )

    return session.touched(graph=graph), tuple(reasked)


def resolve_unanswered(
    session: csessn.Session,
    answered_ids: frozenset[str],
) -> csessn.Session:
    """
    Close every question the respondent ignored.

    Silence is not a reason to block. An unanswered question becomes a delegation
    and is stated as such in the compiled prompt.

    :param session: The session to change.
    :param answered_ids: Ids of the questions that did get a reply.
    :return: The new session.
    """
    graph = session.graph
    for question in session.pending:
        if question.id in answered_ids:
            continue
        decision = graph.get(question.decision_id)
        if not decision.is_open:
            continue
        graph = graph.replace(
            decision.resolve(
                "left to the downstream agent",
                source="delegated",
                rationale="the question went unanswered",
            )
        )
    return session.touched(graph=graph)


def option_from_text(text: str) -> cdecis.Option:
    """
    Build an option out of something a respondent proposed.

    :param text: What they wrote.
    :return: The option, marked as coming from a proposal.
    """
    return cdecis.Option(value=text.strip(), source="proposal")
