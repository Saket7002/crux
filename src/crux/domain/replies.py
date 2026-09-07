"""
Questions, and the six things a respondent can do with one.

Every question carries a free-text field alongside its options, always. That is
an invariant, not a setting: a respondent who can only click is a respondent who
cannot tell you the question is wrong, cannot ask what you meant, and cannot
offer the answer you failed to list. All three happen, and all three are more
valuable than the click.

Import as:

import crux.domain.replies as creply
"""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic

import crux.domain.decisions as cdecis

ReplyKind = Literal["choice", "value", "question", "proposal", "defer", "reject"]
"""What a respondent did with a question."""


# #############################################################################
# Question
# #############################################################################


class Question(pydantic.BaseModel):
    """
    One decision, put to a respondent.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    decision_id: str
    text: str
    why: str = ""
    options: tuple[cdecis.Option, ...] = ()
    citations: tuple[str, ...] = ()
    exchange: int = 0
    recommended: str | None = None

    @property
    def free_text(self) -> bool:
        """
        Say whether free text is accepted alongside the options.

        Always true. Exposed as a property rather than a field so no caller can
        construct a question that forecloses it.

        :return: ``True``.
        """
        return True


# #############################################################################
# What came back
# #############################################################################


class RawReply(pydantic.BaseModel):
    """
    What a respondent sent, before anyone worked out what it means.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    question_id: str
    text: str = ""
    selected: tuple[str, ...] = ()

    @property
    def is_bare_choice(self) -> bool:
        """
        Say whether this reply is a click and nothing else.

        The most common reply by far, and it needs no model call to understand.
        Routing it deterministically keeps the cheapest path free and, more
        importantly, exact — a classifier that occasionally misreads a click as a
        proposal would be maddening.

        :return: Whether options were selected and no free text was written.
        """
        return bool(self.selected) and not self.text.strip()


class ChoiceReply(pydantic.BaseModel):
    """
    The respondent picked from the options offered.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["choice"] = "choice"
    question_id: str
    value: str


class ValueReply(pydantic.BaseModel):
    """
    The respondent answered in their own words.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["value"] = "value"
    question_id: str
    value: str


class CounterQuestionReply(pydantic.BaseModel):
    """
    The respondent asked crux something back.

    Answered from the evidence graph and the question re-asked. Bounded: crux
    gets two attempts, then delegates the decision downstream rather than
    haggling.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["question"] = "question"
    question_id: str
    asks: str


class ProposalReply(pydantic.BaseModel):
    """
    The respondent offered something that was not on the list.

    Two effects, and the second is the interesting one: the value becomes an
    option and resolves the decision, *and* it dirties saturation, because an
    answer nobody anticipated usually implies decisions nobody expanded.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["proposal"] = "proposal"
    question_id: str
    option: cdecis.Option
    rationale: str = ""


class DeferReply(pydantic.BaseModel):
    """
    The respondent handed the judgement back: "you decide".

    Resolves the decision with source ``delegated``, which is honest — crux did
    not decide it either, the downstream agent will.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["defer"] = "defer"
    question_id: str


class RejectReply(pydantic.BaseModel):
    """
    The respondent said the question itself is wrong.

    The most informative reply crux can get, and the one a click-only interface
    can never deliver.
    """

    model_config = pydantic.ConfigDict(frozen=True)
    kind: Literal["reject"] = "reject"
    question_id: str
    reason: str = ""


ClassifiedReply = Annotated[
    ChoiceReply | ValueReply | CounterQuestionReply | ProposalReply | DeferReply | RejectReply,
    pydantic.Field(discriminator="kind"),
]
"""A reply, once crux knows what sort it is."""


def as_choice(raw: RawReply) -> ChoiceReply | None:
    """
    Classify a reply without a model call, where that is possible.

    :param raw: What the respondent sent.
    :return: The classified choice, or ``None`` when the reply needs a model to
        understand it.
    """
    if not raw.is_bare_choice:
        return None
    # A multi-select ("in-app and email") is one answer, not several; the router
    # resolves a decision to a single value.
    return ChoiceReply(question_id=raw.question_id, value=", ".join(raw.selected))


def dirties_expansion(reply: ClassifiedReply) -> bool:
    """
    Say whether a reply invalidates the saturation the expander reached.

    Only two kinds do. A proposal introduces a value nothing was expanded
    against; a rejection says the shape of the question was wrong. Both mean the
    decision set crux settled on was computed against a premise that has changed.

    :param reply: The classified reply.
    :return: Whether another expansion pass is owed.
    """
    return reply.kind in ("proposal", "reject")
