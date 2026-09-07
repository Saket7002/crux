"""
The upper LLM seam: six typed cognitive operations, and nothing else.

This is the only place the application layer meets a model. Every method takes a
pydantic request and returns a pydantic result — no strings, no JSON, no
messages — and **every one is batched across decisions**. crux never makes one
call per node; on a whole-project brief that would be the difference between a
usable product and an unaffordable one.

Import as:

import crux.ports.reasoner as preason
"""

from __future__ import annotations

import math
from typing import Literal, Protocol, runtime_checkable

import pydantic

import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.replies as creply

Operation = Literal["expand", "adjudicate", "phrase", "classify", "answer_counter", "draft"]
"""The six things crux needs a model for, as addressable names.

Declared here rather than in settings because it is this port's own vocabulary.
Routing a different model to a different operation needs a name for each, and the
tool names are not it: `propose_decisions` is not `expand`, and only one of the
six coincides, so a table keyed on tool names would silently re-route an operation
on any prompt edit.
"""

# The expander ranks its proposals by cost rather than scoring each one, because
# models rank far more reliably than they score, and the frontier only ever needs
# the ordering. These are the shares of a ranked list that become each band.
_HIGH_SHARE = 0.3
_MEDIUM_SHARE = 0.7


def cost_from_rank(position: int, total: int) -> cdecis.Cost:
    """
    Turn a place in the expander's cost ranking into a band.

    Asking a model "how costly is this, low/medium/high?" per decision produces
    uncalibrated labels that drift with the prompt; asking it to put a list in
    order produces something usable. This is where the ordering becomes the
    ``cost_if_wrong`` the ask threshold reads.

    :param position: Zero-based place in the ranking, most costly first.
    :param total: How many decisions were ranked.
    :return: The cost band.
    """
    if total <= 0:
        return "medium"
    # Rounded up so a small batch still has a top. Pure shares meant a batch of
    # three graded nothing high, so the most costly thing the expander found was
    # never asked about and every short expansion fell straight through to
    # delegation.
    high_cut = max(1, math.ceil(_HIGH_SHARE * total))
    medium_cut = max(high_cut, math.ceil(_MEDIUM_SHARE * total))
    if position < high_cut:
        return "high"
    if position < medium_cut:
        return "medium"
    return "low"


# #############################################################################
# Expansion
# #############################################################################


class DecisionSketch(pydantic.BaseModel):
    """
    A decision the expander is told already exists, so it proposes only new ones.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    undecided: str
    type: cdecis.DecisionType
    resolved_to: str | None = None


class ProposedEdge(pydantic.BaseModel):
    """
    A dependency the expander thinks holds. Validated before it enters the graph.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    kind: cdecis.EdgeKind
    source_id: str
    """Either an existing decision's id, or the ``ref`` of a sibling in this
    batch. Refs exist because freeform ids are minted from slugs *after* the
    model answers, so without them a proposal could never depend on another
    proposal — which rules out the single most useful edge there is."""

    when_value: str | None = None


class ProposedDecision(pydantic.BaseModel):
    """
    One thing the expander says is undecided.

    Carries no ``cost_if_wrong``: cost comes from this proposal's position in
    :attr:`ExpansionResult.proposed`, which the expander is asked to order most
    costly first.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    ref: str = ""
    """A short label this proposal answers to, so a sibling can depend on it."""

    undecided: str
    type: cdecis.DecisionType
    reversibility: cdecis.Reversibility
    options: tuple[str, ...] = ()
    retrieval_hint: str = ""
    edges: tuple[ProposedEdge, ...] = ()


class ExpansionRequest(pydantic.BaseModel):
    """
    Ask what is undecided.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    prompt: str
    lens: str
    existing: tuple[DecisionSketch, ...] = ()
    evidence_digest: tuple[str, ...] = ()
    host_notes: str = ""


class ExpansionResult(pydantic.BaseModel):
    """
    What the expander proposed, ordered most costly first.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    proposed: tuple[ProposedDecision, ...] = ()


# #############################################################################
# Adjudication
# #############################################################################


class AdjudicationItem(pydantic.BaseModel):
    """
    One decision's raw evidence, handed over to be turned into candidates.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    decision_id: str
    undecided: str
    accept_if: str
    items: tuple[cevid.EvidenceItem, ...] = ()


class AdjudicationRequest(pydantic.BaseModel):
    """
    Ask what distinct answers the evidence actually supports.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    prompt: str
    decisions: tuple[AdjudicationItem, ...] = ()


class AdjudicationResult(pydantic.BaseModel):
    """
    Candidates per decision.

    This is the correction that makes escalation workable: a grep for a common
    term returns twenty items, which is not twenty answers. Escalation counts
    what comes out of here, never what went in.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    candidates: dict[str, tuple[cevid.Candidate, ...]] = pydantic.Field(default_factory=dict)


# #############################################################################
# Phrasing
# #############################################################################


class PhraseRequest(pydantic.BaseModel):
    """
    Ask for the frontier to be worded for a human.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    prompt: str
    decisions: tuple[cdecis.OpenDecision, ...] = ()
    exchange: int = 0


class PhrasedQuestion(pydantic.BaseModel):
    """
    One decision, worded.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    decision_id: str
    text: str
    why: str = ""
    recommended: str | None = None


class PhraseResult(pydantic.BaseModel):
    """
    The frontier, worded.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    questions: tuple[PhrasedQuestion, ...] = ()


# #############################################################################
# Classification
# #############################################################################


class ClassifyPair(pydantic.BaseModel):
    """
    One question and the free text that came back to it.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    question: creply.Question
    reply: creply.RawReply


class ClassifyRequest(pydantic.BaseModel):
    """
    Ask what the respondent meant.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    pairs: tuple[ClassifyPair, ...] = ()


class ClassifyResult(pydantic.BaseModel):
    """
    The replies, understood.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    replies: tuple[creply.ClassifiedReply, ...] = ()


# #############################################################################
# Counter-questions
# #############################################################################


class CounterItem(pydantic.BaseModel):
    """
    One thing the respondent asked crux, and what crux has to answer it with.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    question_id: str
    decision_id: str
    asks: str
    evidence: tuple[cevid.EvidenceItem, ...] = ()


class CounterRequest(pydantic.BaseModel):
    """
    Ask crux's own questions back.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    prompt: str
    items: tuple[CounterItem, ...] = ()


class CounterAnswer(pydantic.BaseModel):
    """
    What crux can say, and whether it actually knew.

    ``answered`` exists so the loop can terminate honestly. A model with no
    evidence will still produce prose; without this flag crux would re-ask
    forever on the strength of an answer it invented.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    question_id: str
    answer: str
    answered: bool
    recommended: str | None = None


class CounterResult(pydantic.BaseModel):
    """
    Answers to the respondent's questions.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    answers: tuple[CounterAnswer, ...] = ()


# #############################################################################
# Drafting
# #############################################################################


class DraftRequest(pydantic.BaseModel):
    """
    Ask for the task prose.

    Nothing else. The assumptions, delegations and citations in the compiled
    prompt are derived from the graph, because a model asked to list its own
    assumptions lists the flattering ones.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    prompt: str
    decided: tuple[cdecis.OpenDecision, ...] = ()


class DraftResult(pydantic.BaseModel):
    """
    The task and its constraints, in prose.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    task: str
    constraints: tuple[str, ...] = ()


# #############################################################################
# Reasoner
# #############################################################################


@runtime_checkable
class Reasoner(Protocol):
    """
    The six things crux needs a model for.
    """

    async def expand(self, request: ExpansionRequest) -> ExpansionResult:
        """
        Propose what is undecided, ordered most costly first.

        :param request: The prompt, the lens, and what already exists.
        :return: The proposals.
        """
        ...

    async def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult:
        """
        Turn evidence items into distinct candidate answers.

        :param request: Every decision's evidence, in one batch.
        :return: Candidates per decision.
        """
        ...

    async def phrase(self, request: PhraseRequest) -> PhraseResult:
        """
        Word the frontier for a human.

        :param request: The decisions to ask about.
        :return: The wording.
        """
        ...

    async def classify(self, request: ClassifyRequest) -> ClassifyResult:
        """
        Work out what each free-text reply meant.

        :param request: Question and reply pairs, in one batch.
        :return: The classified replies.
        """
        ...

    async def answer_counter(self, request: CounterRequest) -> CounterResult:
        """
        Answer the respondent's questions from the evidence, or admit it cannot.

        :param request: What was asked, and the evidence available.
        :return: The answers, each flagged with whether it is grounded.
        """
        ...

    async def draft(self, request: DraftRequest) -> DraftResult:
        """
        Write the task prose and the constraints.

        :param request: The prompt and the resolved decisions.
        :return: The prose.
        """
        ...
