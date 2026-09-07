"""
The retrieval seam.

Batch-shaped on purpose. The decision graph knows every retrieval target at once
before it asks anyone anything, so crux makes one trip with N briefs rather than
N trips with one — and a clarification layer that costs four sequential agent
runs before its first question costs more than the mistake it prevents.

A host is expected to plug their own explore agent in here. The shipped
filesystem retriever exists so the README example runs, not to compete with it.

Import as:

import crux.ports.retrieval as pretr
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import crux.domain.evidence as cevid


@runtime_checkable
class Retriever(Protocol):
    """
    Answers a batch of briefs about the world.
    """

    async def retrieve(self, briefs: Sequence[cevid.Brief]) -> Sequence[cevid.BriefResult]:
        """
        Find what each brief asks for.

        Finding nothing is a normal outcome, not a failure: return a
        :class:`crux.domain.evidence.BriefResult` with no items and the router
        will take a default or retype the decision. Raise only when retrieval
        itself broke.

        Implementations should honour ``brief.max_items`` and are free to answer
        the batch concurrently — the caller awaits the whole thing once.

        :param briefs: What to find out, one per decision.
        :return: One result per brief, in any order; results are matched by
            ``brief_id``, and a brief with no result is treated as empty.
        :raises crux.errors.RetrievalError: When retrieval itself failed.
        """
        ...
