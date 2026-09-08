"""
The public surface: the engine, and a blocking wrapper over it.

Yield/resume is the real API. ``clarify`` exists so a script with a human at a
terminal is four lines rather than a loop, and it is written in terms of the same
``start``/``resume`` a host would call itself.

Import as:

import crux.api as capi
"""

from __future__ import annotations

import asyncio
import pathlib

import crux.application.engine as aengine
import crux.domain.output as coutput
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.packs.software  # noqa: F401 - importing registers the shipped pack
import crux.ports.clarifier as pclarif
import crux.ports.reasoner as preason
import crux.ports.retrieval as pretr
import crux.ports.store as pstore

Crux = aengine.Crux


async def clarify_async(
    prompt: str,
    *,
    reasoner: preason.Reasoner,
    retriever: pretr.Retriever | None = None,
    clarifier: pclarif.Clarifier | None = None,
    root: pathlib.Path | None = None,
    budget: csessn.Budget | None = None,
    store: pstore.SessionStore | None = None,
    session_id: str | None = None,
) -> coutput.CompiledPrompt:
    """
    Drive a whole clarification to its compiled prompt.

    With no ``clarifier``, crux runs headless: it asks nothing and every open
    decision arrives as a stated assumption or a delegation. That is a supported
    mode, not a degraded one — plenty of hosts embed this with no human attached.

    :param prompt: What the user asked for.
    :param reasoner: The model boundary.
    :param retriever: Where to look things up.
    :param clarifier: Who to ask. ``None`` runs headless.
    :param root: The directory the work concerns.
    :param budget: Caps on passes, rounds and questions.
    :param store: Somewhere to keep the session between rounds.
    :param session_id: An id to run under, where the caller wants to find the
        session in the store afterwards.
    :return: The compiled prompt.
    :raises SessionError: When the engine yields questions there is nobody to
        answer and the budget somehow still allows more rounds.
    """
    effective = budget or csessn.Budget()
    if clarifier is None:
        effective = effective.model_copy(update={"max_questions_total": 0})

    crux = aengine.Crux(reasoner=reasoner, retriever=retriever, budget=effective)
    step = await crux.start(
        prompt,
        context=csessn.SessionContext(root=root) if root else None,
        session_id=session_id,
    )

    while isinstance(step, csessn.NeedsInput):
        if store is not None:
            store.save(step.session)
        if clarifier is None:
            raise cerrors.SessionError(
                "crux asked a question with no clarifier configured. Pass a "
                "clarifier, or set budget.max_questions_total to 0 to run headless."
            )
        replies = tuple(clarifier.ask(step.questions))
        step = await crux.resume(step.session, replies)

    if store is not None:
        store.save(step.session)
    return step.compiled


def clarify(
    prompt: str,
    *,
    reasoner: preason.Reasoner,
    retriever: pretr.Retriever | None = None,
    clarifier: pclarif.Clarifier | None = None,
    root: pathlib.Path | None = None,
    budget: csessn.Budget | None = None,
    store: pstore.SessionStore | None = None,
    session_id: str | None = None,
) -> coutput.CompiledPrompt:
    """
    Run :func:`clarify_async` from synchronous code.

    Only for callers that are not already in an event loop; a host inside one
    should await :func:`clarify_async`, or drive ``start``/``resume`` directly.

    :param prompt: What the user asked for.
    :param reasoner: The model boundary.
    :param retriever: Where to look things up.
    :param clarifier: Who to ask. ``None`` runs headless.
    :param root: The directory the work concerns.
    :param budget: Caps on passes, rounds and questions.
    :param store: Somewhere to keep the session between rounds.
    :param session_id: An id to run under, where the caller wants to find the
        session in the store afterwards.
    :return: The compiled prompt.
    :raises ConfigurationError: When called from inside a running event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise cerrors.ConfigurationError(
            "crux.clarify() cannot run inside an event loop. Await crux.clarify_async() instead."
        )
    return asyncio.run(
        clarify_async(
            prompt,
            reasoner=reasoner,
            retriever=retriever,
            clarifier=clarifier,
            root=root,
            budget=budget,
            store=store,
            session_id=session_id,
        )
    )
