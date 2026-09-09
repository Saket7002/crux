"""
The rubric judge: does the compiled prompt say what the case says it should.

It routes through crux's own ``LlmClient`` rather than a platform's built-in
judge for one reason: the cassette. A judge call is a completion like any other,
so it records and replays for free, and a rescoring after a rubric edit costs
one recording rather than a platform bill on every run.

The judge reads ``CompiledPrompt.render()``, which omits the session id and the
timestamp, so an unchanged prompt hashes to the same cassette key every time.

Known bias: by default the same model that wrote the prompt judges it. Pass a
different ``model`` when that matters; it joins the cassette key, so the two
judges never replay each other.

Import as:

import tests.evals.judge as judge
"""

from __future__ import annotations

from typing import Any

import pydantic

import crux.domain.output as coutput
import crux.errors as cerrors
import crux.ports.llm as pllm
import tests.evals.harness as harness

JUDGE_TOOL = "judge_compiled_prompt"


class Verdict(pydantic.BaseModel):
    """
    The judge's call on one rubric bullet.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    section: str
    """Which rubric section the bullet came from. Free text on purpose: models
    restate the label ("Task should", "1.", "constraints") and the score never
    reads it, so a strict literal here failed whole judgements for nothing."""
    bullet: str
    met: bool
    evidence: str = ""


class Judgement(pydantic.BaseModel):
    """
    Every verdict, and the score they add up to.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    verdicts: tuple[Verdict, ...] = ()
    rationale: str = ""

    @property
    def score(self) -> float:
        """
        :return: The fraction of bullets met, or 1.0 for an empty rubric.
        """
        if not self.verdicts:
            return 1.0
        return sum(1 for v in self.verdicts if v.met) / len(self.verdicts)


def judge_tool() -> pllm.ToolSchema:
    """
    :return: The tool the judge is forced to call.
    """
    return pllm.ToolSchema(
        name=JUDGE_TOOL,
        description="Report, bullet by bullet, whether the compiled prompt meets the rubric.",
        parameters={
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string"},
                            "bullet": {"type": "string"},
                            "met": {"type": "boolean"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["section", "bullet", "met"],
                    },
                },
                "rationale": {"type": "string"},
            },
            "required": ["verdicts", "rationale"],
        },
    )


def build_messages(rendered: str, rubric: harness.Rubric) -> tuple[pllm.Message, ...]:
    """
    Build the judge's conversation.

    :param rendered: The compiled prompt as a downstream agent would read it.
    :param rubric: What it should say.
    :return: The messages.
    """
    lines = [
        f"{index}. [{section}] {bullet}"
        for index, (section, bullet) in enumerate(rubric.bullets(), start=1)
    ]
    return (
        pllm.Message(
            role="system",
            content=(
                "You grade a compiled prompt against a rubric. Return one verdict per "
                "rubric bullet, quoting the section and bullet text exactly. A "
                "`task_should` or `constraints_should` bullet is met when the prompt "
                "states it or something equivalent. An `assumptions_should_cite` bullet "
                "is met when the prompt names that decision under Decided, Assumptions "
                "or Open judgements. A `must_not_claim` bullet is met when the prompt "
                "does NOT make that claim. Quote the line that decided each verdict as "
                "evidence. Be strict: a vague gesture at the idea does not meet it."
            ),
        ),
        pllm.Message(
            role="user",
            content="## Rubric\n" + "\n".join(lines) + "\n\n## Compiled prompt\n" + rendered,
        ),
    )


async def judge(
    compiled: coutput.CompiledPrompt,
    rubric: harness.Rubric,
    client: pllm.LlmClient,
    *,
    model: str | None = None,
) -> Judgement:
    """
    Judge one compiled prompt.

    :param compiled: What crux produced.
    :param rubric: What the case says it should say.
    :param client: Where completions come from.
    :param model: Which model judges. ``None`` means the client's default.
    :return: The judgement.
    :raises ReasonerParseError: When the model answered in prose or off-schema.
    """
    return await judge_text(compiled.render(), rubric, client, model=model)


async def judge_text(
    rendered: str,
    rubric: harness.Rubric,
    client: pllm.LlmClient,
    *,
    model: str | None = None,
) -> Judgement:
    """
    Judge any text against a rubric: a compiled prompt, or a downstream diff.

    :param rendered: What is being judged.
    :param rubric: What it should say.
    :param client: Where completions come from.
    :param model: Which model judges.
    :return: The judgement.
    :raises ReasonerParseError: When the model answered in prose or off-schema.
    """
    turn = await client.complete(
        build_messages(rendered, rubric),
        tools=(judge_tool(),),
        force_tool=JUDGE_TOOL,
        model=model,
    )
    call = next((c for c in turn.tool_calls if c.name == JUDGE_TOOL), None)
    if call is None:
        raise cerrors.ReasonerParseError(
            f"The judge answered in prose instead of calling {JUDGE_TOOL}."
        )
    return _parse(call.arguments)


def _parse(arguments: dict[str, Any]) -> Judgement:
    """
    :param arguments: The tool call's arguments, straight from the model.
    :return: The judgement.
    :raises ReasonerParseError: When the arguments do not fit.
    """
    try:
        return Judgement.model_validate(arguments)
    except pydantic.ValidationError as exc:
        raise cerrors.ReasonerParseError(f"The judge's verdicts did not fit: {exc}") from exc
