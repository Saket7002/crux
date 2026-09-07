"""
A scripted LLM client, for exercising the real prompt builders and parsers.

Tier-3 tests use this rather than FakeReasoner because the thing under test IS
the encode/decode layer: prompt construction, forced tool calls, and what happens
when a model returns something malformed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import crux.errors as cerrors
import crux.ports.llm as pllm


class ScriptedLlm:
    """
    Returns canned tool-call payloads and records every conversation it saw.
    """

    def __init__(
        self,
        payloads: Sequence[Any] = (),
        *,
        tool_name: str | None = None,
        by_tool: dict[str, Sequence[Any]] | None = None,
    ) -> None:
        """
        :param payloads: One per call, in order. A dict is returned as a tool
            call's arguments; a string is returned as raw text, which is how a
            model that ignored the forced tool is simulated.
        :param tool_name: Name to answer under. Defaults to whatever was forced.
        :param by_tool: Payloads keyed by forced tool name, consumed in order per
            tool. Prefer this for whole-pipeline tests: a positional queue makes
            them break whenever the engine changes how many calls it makes, which
            is not what those tests are meant to be measuring.
        """
        self._payloads = list(payloads)
        self._by_tool = {name: list(items) for name, items in (by_tool or {}).items()}
        self._tool_name = tool_name
        self.conversations: list[tuple[pllm.Message, ...]] = []
        self.forced: list[str | None] = []
        self.models: list[str | None] = []
        """Which model each call asked for, so a routing test can assert on it."""
        self.require_tool: list[bool] = []
        """Whether each call would accept prose, which only drafting should."""

    def prompt_text(self, index: int = -1) -> str:
        """
        :param index: Which conversation to read.
        :return: Every message in it, concatenated, for asserting on wording.
        """
        return "\n".join(m.content for m in self.conversations[index])

    def _next(self, force_tool: str | None) -> Any:
        """
        Pick the payload for this call.

        :param force_tool: The tool the caller forced.
        :return: The payload.
        :raises ReasonerError: When nothing is left to return.
        """
        queue = self._by_tool.get(force_tool or "")
        if queue is not None:
            # An exhausted per-tool queue answers empty rather than raising, so a
            # pipeline test does not have to predict the exact call count.
            return queue.pop(0) if queue else {}
        if self._by_tool:
            return {}
        if not self._payloads:
            raise cerrors.ReasonerError("ScriptedLlm ran out of payloads")
        return self._payloads.pop(0)

    async def complete(
        self,
        messages: Sequence[pllm.Message],
        *,
        tools: Sequence[pllm.ToolSchema] = (),
        force_tool: str | None = None,
        model: str | None = None,
        require_tool: bool = True,
    ) -> pllm.LlmTurn:
        self.conversations.append(tuple(messages))
        self.models.append(model)
        self.require_tool.append(require_tool)
        self.forced.append(force_tool)
        payload = self._next(force_tool)
        if isinstance(payload, str):
            return pllm.LlmTurn(text=payload)
        return pllm.LlmTurn(
            tool_calls=(
                pllm.ToolCall(
                    id="call_1",
                    name=self._tool_name or force_tool or "unknown",
                    arguments=json.loads(json.dumps(payload)),
                ),
            ),
            finish_reason="tool_calls",
        )
