"""
The reasoner: six typed operations, implemented over a chat client.

Everything that knows about messages, tool calls and JSON lives here and in
:mod:`crux.adapters.llm.prompts`. The application layer above sees only pydantic
in and pydantic out.

Parsing is forgiving in one specific way: a single malformed entry is dropped
with a warning rather than failing the whole batch. A model that gets one of six
decisions wrong should cost one decision, not the session.

Import as:

import crux.adapters.llm.reasoner as xreason
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pydantic

import crux.adapters.llm.prompts as xprompt
import crux.adapters.llm.routing as xroute
import crux.domain.decisions as cdecis
import crux.domain.evidence as cevid
import crux.domain.replies as creply
import crux.errors as cerrors
import crux.ports.llm as pllm
import crux.ports.reasoner as preason

_LOG = logging.getLogger(__name__)


class LlmReasoner:
    """
    Implements :class:`crux.ports.reasoner.Reasoner` over an LLM client.
    """

    def __init__(
        self,
        client: pllm.LlmClient,
        routing: xroute.ModelRouting | None = None,
        *,
        expand_instruction: str | None = None,
    ) -> None:
        """
        :param client: Where completions come from.
        :param routing: Which model serves which operation. The default routes
            everything to the client's own model, which is what keeps every
            existing caller working unchanged.
        :param expand_instruction: Replaces the expansion instruction. Exists
            for the eval optimiser, which proposes candidates for that text;
            hosts leave it unset.
        """
        self._client = client
        self._routing = routing or xroute.DEFAULT
        self._expand_instruction = expand_instruction

    # ## Operations

    async def expand(self, request: preason.ExpansionRequest) -> preason.ExpansionResult:
        """
        Propose what is undecided.

        :param request: The prompt, the lens, and what already exists.
        :return: The proposals, in the order the model ranked them.
        """
        payload = await self._call(
            xprompt.expand_messages(
                prompt=request.prompt,
                lens=request.lens,
                existing=_render_sketches(request.existing),
                evidence="\n".join(request.evidence_digest),
                host_notes=request.host_notes,
                instruction=self._expand_instruction,
            ),
            xprompt.EXPAND_TOOL,
            "expand",
        )
        proposed = _each(payload.get("decisions"), preason.ProposedDecision, "decision")
        return preason.ExpansionResult(proposed=proposed)

    async def select_pack(self, request: preason.PackSelectRequest) -> preason.PackSelectResult:
        """
        Pick the decision pack a prompt belongs to.

        :param request: The prompt and the packs to choose from.
        :return: The choice. An id the model invented comes back as written;
            the engine, not this adapter, decides what to do about it.
        """
        payload = await self._call(
            xprompt.select_pack_messages(
                prompt=request.prompt,
                rendered="\n".join(f"{p.id}: {p.description}" for p in request.packs),
            ),
            xprompt.SELECT_PACK_TOOL,
            "select_pack",
        )
        return preason.PackSelectResult(
            pack_id=str(payload.get("pack_id", "")).strip(),
            rationale=str(payload.get("rationale", "")),
        )

    async def adjudicate(self, request: preason.AdjudicationRequest) -> preason.AdjudicationResult:
        """
        Turn evidence items into distinct candidate answers.

        :param request: Every decision's evidence, in one batch.
        :return: Candidates per decision.
        """
        payload = await self._call(
            xprompt.adjudicate_messages(
                prompt=request.prompt, rendered=_render_adjudication(request.decisions)
            ),
            xprompt.ADJUDICATE_TOOL,
            "adjudicate",
        )
        known = {item.decision_id for item in request.decisions}
        candidates: dict[str, tuple[cevid.Candidate, ...]] = {}
        for entry in payload.get("decisions") or []:
            if not isinstance(entry, dict):
                continue
            decision_id = entry.get("decision_id")
            if decision_id not in known:
                _LOG.warning("Dropping candidates for unknown decision %s", decision_id)
                continue
            candidates[str(decision_id)] = _each(
                entry.get("candidates"), cevid.Candidate, "candidate"
            )
        return preason.AdjudicationResult(candidates=candidates)

    async def phrase(self, request: preason.PhraseRequest) -> preason.PhraseResult:
        """
        Word the frontier for a human.

        :param request: The decisions to ask about.
        :return: The wording.
        """
        payload = await self._call(
            xprompt.phrase_messages(
                prompt=request.prompt, rendered=_render_frontier(request.decisions)
            ),
            xprompt.PHRASE_TOOL,
            "phrase",
        )
        return preason.PhraseResult(
            questions=_each(payload.get("questions"), preason.PhrasedQuestion, "question")
        )

    async def classify(self, request: preason.ClassifyRequest) -> preason.ClassifyResult:
        """
        Work out what each free-text reply meant.

        :param request: Question and reply pairs.
        :return: The classified replies.
        """
        payload = await self._call(
            xprompt.classify_messages(rendered=_render_pairs(request.pairs)),
            xprompt.CLASSIFY_TOOL,
            "classify",
        )
        replies: list[creply.ClassifiedReply] = []
        for entry in payload.get("replies") or []:
            reply = _to_reply(entry)
            if reply is not None:
                replies.append(reply)
        return preason.ClassifyResult(replies=tuple(replies))

    async def answer_counter(self, request: preason.CounterRequest) -> preason.CounterResult:
        """
        Answer the respondent's questions, or admit crux cannot.

        :param request: What was asked, and the evidence available.
        :return: The answers.
        """
        payload = await self._call(
            xprompt.counter_messages(
                prompt=request.prompt, rendered=_render_counters(request.items)
            ),
            xprompt.COUNTER_TOOL,
            "answer_counter",
        )
        return preason.CounterResult(
            answers=_each(payload.get("answers"), preason.CounterAnswer, "answer")
        )

    async def draft(self, request: preason.DraftRequest) -> preason.DraftResult:
        """
        Write the task prose and the constraints.

        :param request: The prompt and the resolved decisions.
        :return: The prose.
        """
        # Unlike every other operation, a plain-text answer here is still the
        # thing we asked for: draft() wants prose. Some models decline the tool
        # call and simply write it, and failing the whole session over that would
        # throw away a finished clarification at the last step.
        payload, text = await self._call_tolerantly(
            xprompt.draft_messages(
                prompt=request.prompt, rendered=_render_decided(request.decided)
            ),
            xprompt.DRAFT_TOOL,
            "draft",
            require_tool=False,
        )
        if payload is None:
            _LOG.warning("Draft came back as prose rather than a tool call; using the prose")
            return preason.DraftResult(task=text.strip() or request.prompt)
        constraints = payload.get("constraints") or []
        return preason.DraftResult(
            task=str(payload.get("task") or request.prompt),
            constraints=tuple(str(c) for c in constraints if isinstance(c, str)),
        )

    # ## Internals

    async def _call(
        self,
        messages: tuple[pllm.Message, ...],
        tool: pllm.ToolSchema,
        operation: preason.Operation,
    ) -> dict[str, Any]:
        """
        Complete once through a forced tool call and hand back its arguments.

        :param messages: The conversation.
        :param tool: The tool the model must call.
        :param operation: Which of the six this is, so routing can pick a model.
        :return: The call's arguments.
        :raises ReasonerParseError: When the model called nothing usable.
        """
        payload, text = await self._call_tolerantly(messages, tool, operation)
        if payload is None:
            raise cerrors.ReasonerParseError(
                f"Model did not call {tool.name!r}; it returned "
                f"{'plain text: ' + text[:80] if text else 'nothing usable'}."
            )
        return payload

    async def _call_tolerantly(
        self,
        messages: tuple[pllm.Message, ...],
        tool: pllm.ToolSchema,
        operation: preason.Operation,
        *,
        require_tool: bool = True,
    ) -> tuple[dict[str, Any] | None, str]:
        """
        Complete once, returning whichever of a tool call or prose came back.

        :param messages: The conversation.
        :param tool: The tool the model was asked for.
        :param operation: Which of the six this is, so routing can pick a model.
        :param require_tool: Whether prose should count as a failed attempt and
            send the client to its next model. False only for ``draft``, where
            prose is the thing being asked for.
        :return: The tool arguments if it called one, and whatever prose it
            wrote. Callers that can use prose take the second; the rest raise.
        """
        turn = await self._client.complete(
            messages,
            tools=(tool,),
            force_tool=tool.name,
            model=self._routing.model_for(operation),
            require_tool=require_tool,
        )
        for call in turn.tool_calls:
            if call.name == tool.name:
                return call.arguments, turn.text
        return None, turn.text


_ModelT = Any


def _each(raw: Any, model: type[_ModelT], label: str) -> tuple[_ModelT, ...]:
    """
    Validate a list of entries, dropping the ones that do not fit.

    One bad entry in six should cost one decision, not the session — a model
    that invents an enum value or omits a field is a normal Tuesday, and failing
    the batch would turn it into an outage.

    :param raw: What the model returned.
    :param model: The type each entry should be.
    :param label: What to call an entry in the warning.
    :return: The entries that validated.
    """
    if not isinstance(raw, list):
        return ()
    validated: list[_ModelT] = []
    for entry in raw:
        try:
            validated.append(model.model_validate(entry))
        except pydantic.ValidationError as exc:
            _LOG.warning("Dropping malformed %s: %s", label, exc.errors()[:1])
    return tuple(validated)


def _to_reply(entry: Any) -> creply.ClassifiedReply | None:
    """
    Build a classified reply from one entry.

    The wire format is flatter than the union: the model reports a ``kind`` plus
    a handful of optional fields, and this maps that onto the right arm.

    :param entry: What the model returned for one reply.
    :return: The reply, or ``None`` when it cannot be read.
    """
    if not isinstance(entry, dict):
        return None
    question_id = str(entry.get("question_id") or "")
    kind = entry.get("kind")
    value = str(entry.get("value") or "")
    if not question_id:
        return None
    if kind in ("choice", "value"):
        if not value:
            return None
        arm = creply.ChoiceReply if kind == "choice" else creply.ValueReply
        return arm(question_id=question_id, value=value)
    if kind == "question":
        return creply.CounterQuestionReply(
            question_id=question_id, asks=str(entry.get("asks") or "")
        )
    if kind == "proposal":
        if not value:
            return None
        return creply.ProposalReply(
            question_id=question_id,
            option=cdecis.Option(value=value, source="proposal"),
            rationale=str(entry.get("rationale") or ""),
        )
    if kind == "defer":
        return creply.DeferReply(question_id=question_id)
    if kind == "reject":
        return creply.RejectReply(question_id=question_id, reason=str(entry.get("reason") or ""))
    _LOG.warning("Dropping reply with unknown kind %r", kind)
    return None


# #############################################################################
# Rendering
# #############################################################################


def _render_sketches(sketches: Sequence[preason.DecisionSketch]) -> str:
    """
    :param sketches: What already exists.
    :return: One line per decision, with its id so edges can reference it.
    """
    return "\n".join(
        f"- [{s.id}] ({s.type}) {s.undecided}"
        + (f" — RESOLVED: {s.resolved_to}" if s.resolved_to else "")
        for s in sketches
    )


def _render_adjudication(items: Sequence[preason.AdjudicationItem]) -> str:
    """
    :param items: Each decision with its evidence.
    :return: The rendered batch.
    """
    blocks: list[str] = []
    for item in items:
        evidence = "\n".join(f"  [{e.id}] {e.locator}: {e.excerpt}" for e in item.items)
        blocks.append(
            f"Decision [{item.decision_id}]: {item.undecided}\n"
            f"Settled when: {item.accept_if}\n"
            f"Evidence:\n{evidence}"
        )
    return "\n\n".join(blocks)


def _render_frontier(decisions: Sequence[Any]) -> str:
    """
    :param decisions: The frontier.
    :return: One block per decision, with its options.
    """
    blocks: list[str] = []
    for decision in decisions:
        options = ", ".join(o.value for o in (decision.options or ())) or "(open-ended)"
        blocks.append(
            f"[{decision.id}] {decision.undecided}\n"
            f"  type: {decision.type}, cost if wrong: {decision.cost_if_wrong}, "
            f"reversibility: {decision.reversibility}\n"
            f"  options: {options}"
        )
    return "\n\n".join(blocks)


def _render_pairs(pairs: Sequence[preason.ClassifyPair]) -> str:
    """
    :param pairs: Questions and what came back.
    :return: The rendered batch.
    """
    blocks: list[str] = []
    for pair in pairs:
        options = ", ".join(o.value for o in pair.question.options) or "(none offered)"
        blocks.append(
            f"Question [{pair.question.id}]: {pair.question.text}\n"
            f"  Options offered: {options}\n"
            f"  They selected: {', '.join(pair.reply.selected) or '(nothing)'}\n"
            f"  They wrote: {pair.reply.text or '(nothing)'}"
        )
    return "\n\n".join(blocks)


def _render_counters(items: Sequence[preason.CounterItem]) -> str:
    """
    :param items: What was asked back, and the evidence for it.
    :return: The rendered batch.
    """
    blocks: list[str] = []
    for item in items:
        evidence = (
            "\n".join(f"  {e.locator}: {e.excerpt}" for e in item.evidence)
            or "  (nothing was retrieved for this decision)"
        )
        blocks.append(
            f"Question [{item.question_id}] — they asked: {item.asks}\n"
            f"Evidence available:\n{evidence}"
        )
    return "\n\n".join(blocks)


def _render_decided(decisions: Sequence[Any]) -> str:
    """
    :param decisions: Everything that got settled.
    :return: One line per decision.
    """
    lines: list[str] = []
    for decision in decisions:
        if decision.resolution is None:
            continue
        lines.append(f"- {decision.undecided}: {decision.resolution.value}")
    return "Settled decisions:\n" + "\n".join(lines)
