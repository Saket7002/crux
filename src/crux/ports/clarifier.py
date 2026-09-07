"""
The human-in-the-loop seam.

Sync, unlike every other port, and mirroring ``operant.ports.hitl.Clarifier``:
only the blocking convenience wrapper uses it. A host driving crux through
yield/resume never implements this at all — it renders ``NeedsInput.questions``
however it likes and calls ``resume()`` with the replies.

Batched, because crux asks a whole round at once. A clarifier that renders one
question at a time is free to do so; it just gets the round in one call.

Import as:

import crux.ports.clarifier as pclarif
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import crux.domain.replies as creply


@runtime_checkable
class Clarifier(Protocol):
    """
    Puts a round of questions to a respondent and brings the answers back.
    """

    def ask(self, questions: Sequence[creply.Question]) -> Sequence[creply.RawReply]:
        """
        Block until the respondent answers or gives up.

        Every question accepts free text as well as its options, and an
        implementation that hides that field breaks the contract: a respondent
        must be able to ask back, propose something unlisted, or say the question
        is wrong.

        Returning fewer replies than questions is allowed and means the rest went
        unanswered; those decisions fall through to a default or a delegation
        rather than blocking.

        :param questions: The round to put.
        :return: What came back, matched to questions by ``question_id``.
        """
        ...
