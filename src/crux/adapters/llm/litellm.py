"""
The litellm-backed client.

litellm rather than a provider SDK because a host should be able to point crux at
whatever model they already pay for. Structured output goes through a forced tool
call, which is the more portable of the two mechanisms across providers.

Two things happen here that are easy to get wrong and were measured rather than
guessed. Retries wait as long as the provider asked for, because a free tier's
per-minute budget refills over a minute and a fast retry is a second failure that
also spends what it was waiting for. And a turn that comes back as prose where a
tool call was demanded counts as a *failed attempt*, not a success, because that
is the commonest way a weak model fails and it would otherwise be invisible to the
fallback chain.

Import as:

import crux.adapters.llm.litellm as xlitell
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Sequence
from typing import Any, Literal

import litellm
import litellm.exceptions

import crux.adapters.llm.prompts as xprompt
import crux.errors as cerrors
import crux.infra.evidence as ievid
import crux.infra.settings as isettn
import crux.ports.llm as pllm

_LOG = logging.getLogger(__name__)

# Providers state how long to wait, in prose, inside the error body.
_RETRY_HINT = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)

# What to wait when the provider does not say. Deliberately generous: a
# tokens-per-minute budget refills over a minute, so a fast retry is not a retry,
# it is a second failure that also spends the budget it was waiting for.
_FALLBACK_BACKOFF = 20.0
_MAX_BACKOFF = 65.0

Verdict = Literal["retry", "fallback", "fatal"]


def _verdict(exc: Exception) -> Verdict:
    """
    Decide what a provider failure means for the chain.

    An ordered ladder, most specific first, and the order is load-bearing. The
    obvious rules are wrong in ways that were checked against the pinned litellm
    rather than assumed:

    - ``ContextWindowExceededError``, ``ContentPolicyViolationError`` and
      ``UnsupportedParamsError`` are all ``BadRequestError`` subclasses, so a
      plain "bad request means fatal" refuses to fall back on three cases where
      another provider genuinely succeeds. They must be caught first.
    - ``ServiceUnavailableError`` and ``BadGatewayError`` are *not*
      ``InternalServerError`` subclasses, so "retry on 5xx" written as
      ``except InternalServerError`` misses 502 and 503, the two commonest.
      Checking the status code catches them without naming them.
    - ``BudgetExceededError`` descends from bare ``Exception`` and is caught by
      no ``APIError`` clause at all.

    :param exc: What the provider raised.
    :return: Whether to wait and retry, move to the next model, or stop.
    """
    E = litellm.exceptions
    if isinstance(exc, E.RateLimitError):
        return "retry"
    if isinstance(exc, E.APIConnectionError):
        return "retry"
    if isinstance(exc, E.InternalServerError):
        return "retry"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return "retry"
    if isinstance(
        exc,
        (
            E.ContextWindowExceededError,
            E.ContentPolicyViolationError,
            getattr(E, "UnsupportedParamsError", E.BadRequestError),
        ),
    ):
        return "fallback"
    if isinstance(
        exc,
        (
            E.AuthenticationError,
            E.PermissionDeniedError,
            E.NotFoundError,
            getattr(E, "BudgetExceededError", E.APIError),
        ),
    ):
        return "fallback"
    return "fatal"


class _Unavailable(Exception):
    """
    One model did not work out. Carries why, so the chain can decide.
    """

    def __init__(self, model: str, reason: str, verdict: Verdict) -> None:
        super().__init__(f"{model}: {reason}")
        self.model = model
        self.reason = reason
        self.verdict = verdict


class LiteLlmClient:
    """
    Completes conversations through litellm, with retries and a fallback chain.
    """

    def __init__(
        self,
        settings: isettn.CruxSettings | None = None,
        recorder: ievid.RunRecorder | None = None,
    ) -> None:
        """
        :param settings: Model, key and limits. Read from the environment when
            not supplied.
        :param recorder: Where to write a trace of every exchange. ``None``
            records nothing, which is the default: recording writes prompts and
            retrieved excerpts to disk, and that should never happen silently.
        """
        self._settings = settings or isettn.CruxSettings()
        self._recorder = recorder
        self._warned_about_global_key = False

    async def complete(
        self,
        messages: Sequence[pllm.Message],
        *,
        tools: Sequence[pllm.ToolSchema] = (),
        force_tool: str | None = None,
        model: str | None = None,
        require_tool: bool = True,
    ) -> pllm.LlmTurn:
        """
        Run one completion, trying each model in the chain in turn.

        :param messages: The conversation.
        :param tools: Tools the model may call.
        :param force_tool: A tool it must call.
        :param model: Which model to start with. ``None`` uses the configured one.
        :param require_tool: Whether a turn with no call to ``force_tool`` counts
            as a failed attempt.
        :return: The model's turn.
        :raises ReasonerError: When every model in the chain failed.
        """
        base: dict[str, Any] = {
            "messages": [_encode(message) for message in messages],
            # Retries are handled here rather than by litellm, whose backoff is
            # tuned for paid tiers and gives up long before a per-minute token
            # budget has refilled.
            "num_retries": 0,
        }
        if tools:
            base["tools"] = [xprompt.tool_of(tool) for tool in tools]
        if force_tool is not None:
            base["tool_choice"] = {"type": "function", "function": {"name": force_tool}}

        chain = self._settings.chain_for(model)
        self._warn_if_global_key_is_ambiguous(chain)
        deadline = time.monotonic() + self._settings.fallback_deadline
        failures: list[_Unavailable] = []

        for position, name in enumerate(chain):
            if position and time.monotonic() >= deadline:
                failures.append(_Unavailable(name, "not tried, deadline reached", "fallback"))
                break
            spec = self._settings.spec_for(name, drop_global_key=len(chain) > 1)
            budget = (
                self._settings.max_retries if position == 0 else self._settings.fallback_retries
            )
            try:
                return await self._one_model(
                    spec, base, force_tool, require_tool, budget, deadline, position
                )
            except _Unavailable as failure:
                if failure.verdict == "fatal":
                    raise cerrors.ReasonerError(
                        f"Model {name} failed and no other model would do better: "
                        f"{failure.reason}. Check CRUX_MODEL and the provider "
                        f"credentials in the environment."
                    ) from failure
                _LOG.warning("Model %s unavailable (%s)", name, failure.reason)
                failures.append(failure)

        tried = "; ".join(f"{f.model}: {f.reason}" for f in failures)
        raise cerrors.ReasonerError(
            f"Every model in the chain failed. Tried {len(failures)}: {tried}. "
            f"On a free tier the per-minute token budget counts the max_tokens "
            f"you RESERVE, not what you use, so lowering CRUX_MAX_TOKENS raises "
            f"the number of calls that fit in a minute."
        )

    async def _one_model(
        self,
        spec: isettn.ModelSpec,
        base: dict[str, Any],
        force_tool: str | None,
        require_tool: bool,
        budget: int,
        deadline: float,
        position: int,
    ) -> pllm.LlmTurn:
        """
        Try one model, waiting out anything it says is temporary.

        :param spec: The model and its resolved call options.
        :param base: Provider-independent completion arguments.
        :param force_tool: The tool that was demanded, if any.
        :param require_tool: Whether prose counts as a failure.
        :param budget: How many retries this model gets.
        :param deadline: Wall clock the whole chain must finish by.
        :param position: Where in the chain this model sits, for the trace.
        :return: The turn.
        :raises _Unavailable: When this model did not work out.
        """
        kwargs = _kwargs_for(spec, base)
        attempts = budget + 1
        for attempt in range(attempts):
            if time.monotonic() >= deadline:
                raise _Unavailable(spec.name, "deadline reached", "fallback")
            started = time.monotonic()
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                verdict = _verdict(exc)
                self._record(spec, kwargs, force_tool, None, started, str(exc), position)
                if verdict == "retry" and attempt < attempts - 1:
                    delay = _wait_for(exc, attempt)
                    _LOG.warning(
                        "Rate limited by %s; waiting %.1fs before attempt %d of %d",
                        spec.name,
                        delay,
                        attempt + 2,
                        attempts,
                    )
                    await asyncio.sleep(delay)
                    continue
                # A retry budget that runs out is promoted to a fallback: this
                # model is busy, another one might not be.
                raise _Unavailable(
                    spec.name,
                    str(exc)[:160],
                    "fatal" if verdict == "fatal" else "fallback",
                ) from exc

            turn = _decode(response)
            # Decoding happens inside the attempt on purpose. A model that
            # answers in prose where a schema was demanded has failed, and if
            # that were only noticed two layers up the fallback chain could never
            # see it -- which is exactly the commonest way a weak model fails.
            if require_tool and force_tool is not None and not _called(turn, force_tool):
                self._record(
                    spec,
                    kwargs,
                    force_tool,
                    turn,
                    started,
                    f"did not call {force_tool!r}",
                    position,
                )
                raise _Unavailable(
                    spec.name, f"answered without calling {force_tool!r}", "fallback"
                )
            self._record(spec, kwargs, force_tool, turn, started, "", position)
            return turn
        raise _Unavailable(spec.name, "retry loop exhausted", "fallback")

    def _warn_if_global_key_is_ambiguous(self, chain: tuple[str, ...]) -> None:
        """
        Say once when a global key cannot mean what it looks like it means.

        ``api_key`` belongs to one provider. With a chain, it is wrong for every
        model but the one it was issued for, and an authentication failure on a
        fallback is a confusing way to discover that.

        :param chain: The models about to be tried.
        """
        if len(chain) < 2 or self._warned_about_global_key:
            return
        if self._settings.api_key is not None or self._settings.api_base is not None:
            self._warned_about_global_key = True
            _LOG.warning(
                "CRUX_API_KEY/CRUX_API_BASE are set alongside a fallback chain, so "
                "they are being ignored: a global key belongs to one provider. "
                "Set each provider's own variable instead."
            )

    def _record(
        self,
        spec: isettn.ModelSpec,
        kwargs: dict[str, Any],
        force_tool: str | None,
        turn: pllm.LlmTurn | None,
        started: float,
        error: str,
        position: int,
    ) -> None:
        """
        Hand one attempt to the recorder, if there is one.

        One record per *attempt*, not per call. A run that quietly degraded to a
        weaker fallback would otherwise produce a trace claiming it ran on the
        primary, in the artefact whose whole purpose is answering "why did it do
        that".

        :param spec: The model that was tried.
        :param kwargs: What was sent.
        :param force_tool: The tool that was forced, which names the operation.
        :param turn: What came back, when anything did.
        :param started: When the attempt began.
        :param error: What went wrong, when something did.
        :param position: Where in the chain this model sat.
        """
        if self._recorder is None:
            return
        self._recorder.record_call(
            operation=force_tool or "chat",
            model=spec.name,
            messages=list(kwargs.get("messages", [])),
            forced_tool=force_tool,
            response=ievid.summarise_turn(turn) if turn is not None else {},
            elapsed_seconds=time.monotonic() - started,
            error=error,
            attempt=position,
        )


def _kwargs_for(spec: isettn.ModelSpec, base: dict[str, Any]) -> dict[str, Any]:
    """
    Apply one model's own credentials and limits to the shared arguments.

    Per attempt rather than once per call, because a chain crosses providers and
    the second model's key, base URL and token ceiling are not the first's.

    :param spec: The model and its resolved options.
    :param base: Provider-independent arguments.
    :return: Complete litellm keyword arguments.
    """
    kwargs = dict(base)
    kwargs["model"] = spec.name
    kwargs["max_tokens"] = spec.max_tokens
    kwargs["timeout"] = spec.request_timeout
    if spec.api_key is not None:
        kwargs["api_key"] = spec.api_key.get_secret_value()
    if spec.api_base is not None:
        kwargs["api_base"] = spec.api_base
    # temperature is rejected outright by the newer Claude models, and the
    # determinism it used to buy has no replacement parameter -- tighten the
    # prompt instead. Only sent when a host explicitly asked for one.
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    if spec.reasoning_effort is not None:
        kwargs["reasoning_effort"] = spec.reasoning_effort
    return kwargs


def _called(turn: pllm.LlmTurn, tool: str) -> bool:
    """
    :param turn: What came back.
    :param tool: The tool that was demanded.
    :return: Whether the turn contains a call to it.
    """
    return any(call.name == tool for call in turn.tool_calls)


def _wait_for(exc: Exception, attempt: int) -> float:
    """
    Work out how long to wait after a rate limit.

    Prefers the provider's own stated wait, padded, because guessing shorter
    than it asked for spends the budget the wait was supposed to refill.

    :param exc: What the provider raised.
    :param attempt: Zero-based attempt number, for backoff.
    :return: Seconds to sleep.
    """
    hinted = _RETRY_HINT.search(str(exc))
    if hinted is not None:
        # Padded: the hint is when the budget frees, and arriving exactly then
        # races every other client doing the same arithmetic.
        return min(_MAX_BACKOFF, float(hinted.group(1)) + 2.0)
    return min(_MAX_BACKOFF, _FALLBACK_BACKOFF * (attempt + 1)) + random.uniform(0, 2)


def _encode(message: pllm.Message) -> dict[str, Any]:
    """
    Render one message the way litellm wants it.

    :param message: The message.
    :return: Its dict form.
    """
    encoded: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        encoded["tool_call_id"] = message.tool_call_id
    return encoded


def _decode(response: Any) -> pllm.LlmTurn:
    """
    Turn a litellm response into a turn.

    :param response: What litellm returned.
    :return: The turn.
    :raises ReasonerParseError: When the response has no usable choice.
    """
    try:
        choice = response.choices[0]
        message = choice.message
    except (AttributeError, IndexError, KeyError) as exc:
        raise cerrors.ReasonerParseError(f"Model returned no usable choice: {response}") from exc

    calls: list[pllm.ToolCall] = []
    for raw in getattr(message, "tool_calls", None) or []:
        arguments = _arguments_of(raw)
        if arguments is None:
            continue
        calls.append(
            pllm.ToolCall(
                id=getattr(raw, "id", "") or "",
                name=raw.function.name,
                arguments=arguments,
            )
        )
    return pllm.LlmTurn(
        text=getattr(message, "content", None) or "",
        tool_calls=tuple(calls),
        finish_reason=getattr(choice, "finish_reason", "") or "",
    )


def _arguments_of(raw: Any) -> dict[str, Any] | None:
    """
    Read a tool call's arguments, whether the provider parsed them or not.

    :param raw: The provider's tool-call object.
    :return: The arguments, or ``None`` when they cannot be read.
    """
    arguments = raw.function.arguments
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        _LOG.warning("Dropping tool call %s: arguments were not valid JSON", raw.function.name)
        return None
    if not isinstance(parsed, dict):
        _LOG.warning("Dropping tool call %s: arguments were not an object", raw.function.name)
        return None
    return parsed
