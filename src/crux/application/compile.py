"""
Producing the compiled prompt.

The model is asked for the task prose and the constraints, and for nothing else.
Assumptions, delegations, out-of-scope and citations are derived from the graph
by :meth:`crux.domain.output.CompiledPrompt.assemble`, because a model asked to
list its own assumptions lists the flattering ones.

Import as:

import crux.application.compile as acomp
"""

from __future__ import annotations

import crux.domain.output as coutput
import crux.domain.session as csessn
import crux.ports.reasoner as preason


async def compile_prompt(
    session: csessn.Session,
    reasoner: preason.Reasoner,
) -> coutput.CompiledPrompt:
    """
    Draft the prose, then derive everything else from the graph.

    :param session: The finished session.
    :param reasoner: The model boundary.
    :return: The compiled prompt.
    """
    draft = await reasoner.draft(
        preason.DraftRequest(prompt=session.prompt, decided=session.graph.resolved())
    )
    return coutput.CompiledPrompt.assemble(
        task=draft.task or session.prompt,
        constraints=draft.constraints,
        graph=session.graph,
        evidence=session.evidence,
        session_id=session.id,
    )
