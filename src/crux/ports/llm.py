"""
The lower LLM seam: messages in, a turn out.

Only :mod:`crux.adapters.llm.reasoner` depends on this. The application layer
talks to :mod:`crux.ports.reasoner` instead, which deals in typed requests and
results and knows nothing about messages, tools or JSON.

The split is what keeps the test suite cheap. Engine tests fake the *reasoner*
and never write a tool-call blob; adapter tests fake the *client* and exercise
the real prompt builders and parsers. Collapse the two and a prompt edit breaks
two hundred logic tests.

Import as:

import crux.ports.llm as pllm
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import pydantic

Role = Literal["system", "user", "assistant", "tool"]
"""Who said it."""


class Message(pydantic.BaseModel):
    """
    One turn of conversation.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    role: Role
    content: str
    tool_call_id: str | None = None


class ToolSchema(pydantic.BaseModel):
    """
    A tool the model may call, and the shape of its arguments.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCall(pydantic.BaseModel):
    """
    The model's call to a tool.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class LlmTurn(pydantic.BaseModel):
    """
    What came back from one completion.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""


@runtime_checkable
class LlmClient(Protocol):
    """
    Completes a conversation, optionally through a tool call.
    """

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] = (),
        force_tool: str | None = None,
        model: str | None = None,
        require_tool: bool = True,
    ) -> LlmTurn:
        """
        Run one completion.

        crux always asks for structured output through a forced tool call rather
        than a response format, because tool calling is the more portable of the
        two across the providers litellm fronts.

        :param messages: The conversation so far.
        :param tools: Tools the model may call.
        :param force_tool: Name of a tool the model must call, where the caller
            needs a structured answer and nothing else.
        :param model: Which model to use. ``None`` means the client's own
            default. A bare string rather than a settings object, because this
            port must not know what configuration looks like.
        :param require_tool: Whether a turn that does not call ``force_tool``
            counts as a failed attempt. True by default, because a model that
            answers in prose where a schema was demanded has failed, and an
            implementation with a fallback chain needs to know that in order to
            try the next model. ``draft`` sets it False: prose is what it asked
            for.
        :return: The model's turn.
        :raises crux.errors.ReasonerError: When the provider failed.
        """
        ...
