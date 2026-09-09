"""
The systems the harness can score: crux, and the baselines it is compared to.

Without a baseline scored the same way, nothing shows the decision graph earns
anything. Three are kept here, and each is deliberately simple:

- ``none``: the raw prompt is the compiled prompt. Every expected decision is
  missed, which is the honest floor.
- ``ask3``: one model call that asks three clarifying questions, with no
  retrieval and no graph. Its questions are matched against ``must_surface``
  by the same matcher crux's decisions are.
- ``flat``: crux with every conditional edge stripped, so nothing is pruned
  and the frontier has no order. This is the saturation experiment's B arm.

Import as:

import tests.evals.systems as systems
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import crux.domain.decisions as cdecis
import crux.domain.graph as cgraph
import crux.domain.ids as cids
import crux.domain.output as coutput
import crux.domain.replies as creply
import crux.domain.session as csessn
import crux.errors as cerrors
import crux.ports.llm as pllm
import crux.ports.reasoner as preason
import tests.evals.harness as harness
import tests.evals.runner as runner

System = Literal["crux", "flat", "ask3", "none"]
SYSTEMS: tuple[System, ...] = ("crux", "flat", "ask3", "none")

ASK_TOOL = pllm.ToolSchema(
    name="ask_questions",
    description="The three clarifying questions to ask before starting this request.",
    parameters={
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }
        },
        "required": ["questions"],
    },
)
ASK_COUNT = 3


class EdgeStripper:
    """
    A reasoner that answers normally but removes every conditional edge.

    Comparing crux to itself with the edges taken out is the only way to see
    what the graph actually buys, as opposed to what it plausibly might.
    """

    def __init__(self, inner: preason.Reasoner) -> None:
        """
        :param inner: The reasoner whose edges are dropped.
        """
        self._inner = inner

    async def expand(self, request: preason.ExpansionRequest) -> preason.ExpansionResult:
        """
        :param request: What to expand.
        :return: The inner result with every proposal's edges removed.
        """
        result = await self._inner.expand(request)
        return preason.ExpansionResult(
            proposed=tuple(p.model_copy(update={"edges": ()}) for p in result.proposed)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def run_system(
    system: System,
    case: harness.EvalCase,
    client: pllm.LlmClient,
    *,
    expand_instruction: str | None = None,
) -> csessn.Session:
    """
    Run one case under one system, the recall way: questions left unanswered.

    :param system: Which system.
    :param case: The case to run.
    :param client: Where completions come from.
    :param expand_instruction: A candidate expansion instruction; crux and
        flat only.
    :return: A session the harness can score.
    :raises ConfigurationError: When the system name is unknown.
    """
    if system == "crux":
        return await runner.run_case(case, client, expand_instruction=expand_instruction)
    if system == "flat":
        return await runner.run_case(
            case, client, expand_instruction=expand_instruction, wrap=EdgeStripper
        )
    if system == "ask3":
        return await run_ask3(case, client)
    if system == "none":
        return run_none(case)
    raise cerrors.ConfigurationError(f"Unknown system {system!r}; choose from {SYSTEMS}.")


def run_none(case: harness.EvalCase) -> csessn.Session:
    """
    The raw prompt goes downstream untouched.

    :param case: The case.
    :return: A finished session with an empty graph and the prompt as the task.
    """
    return csessn.Session(
        id=case.id,
        prompt=case.prompt,
        phase="done",
        outcome=coutput.CompiledPrompt(task=case.prompt, session_id=case.id),
    )


async def run_ask3(case: harness.EvalCase, client: pllm.LlmClient) -> csessn.Session:
    """
    One call, three questions, no retrieval, no graph.

    Each question becomes a freeform open decision so the harness matches it
    the way it matches crux's, and a pending question so ask rate counts it.

    :param case: The case.
    :param client: Where the one completion comes from.
    :return: A session awaiting input on three questions.
    :raises ReasonerParseError: When the model answered in prose.
    """
    turn = await client.complete(
        (
            pllm.Message(
                role="system",
                content=(
                    "You are given a software request. Ask the three clarifying questions a "
                    "competent engineer would most need answered before starting. Do not "
                    "answer the request and do not write code."
                ),
            ),
            pllm.Message(role="user", content=case.prompt),
        ),
        tools=(ASK_TOOL,),
        force_tool=ASK_TOOL.name,
    )
    call = next((c for c in turn.tool_calls if c.name == ASK_TOOL.name), None)
    if call is None:
        raise cerrors.ReasonerParseError("Ask-3 answered in prose instead of asking.")
    texts = [
        str(q.get("text", "")).strip()
        for q in call.arguments.get("questions", [])
        if isinstance(q, dict)
    ][:ASK_COUNT]
    return build_asked(case, [t for t in texts if t], llm_calls=1)


def build_asked(
    case: harness.EvalCase, questions: Sequence[str], *, llm_calls: int
) -> csessn.Session:
    """
    Build a session in which the given questions were asked and nothing else.

    :param case: The case.
    :param questions: What was asked, in order.
    :param llm_calls: How many completions it took.
    :return: The session, awaiting input.
    """
    graph = cgraph.DecisionGraph()
    pending: list[creply.Question] = []
    for text in questions:
        decision_id = cids.freeform_id(text, pass_index=1)
        graph = graph.add(
            cdecis.OpenDecision(
                id=decision_id,
                undecided=text,
                type="underspecification",
                cost_if_wrong="high",
                reversibility="hard",
                origin=cdecis.Origin(seeded_by="expansion", pass_index=1),
            )
        )
        pending.append(
            creply.Question(
                id=cids.question_id(decision_id, exchange=0),
                decision_id=decision_id,
                text=text,
            )
        )
    return csessn.Session(
        id=case.id,
        prompt=case.prompt,
        phase="awaiting_input",
        graph=graph,
        pending=tuple(pending),
        budget=csessn.Budget(
            max_questions_total=case.max_questions,
            spent_questions=len(pending),
            spent_rounds=1,
            spent_llm_calls=llm_calls,
        ),
    )
