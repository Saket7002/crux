"""
Tier-3 tests for the litellm adapter's own logic.

Everything here is the code between crux and the provider: how a rate limit is
waited out, and how a response is turned into a turn. None of it needs a network
call, and all of it is what stands between a free tier and a failed run.
"""

from __future__ import annotations

import asyncio
from typing import Any

import litellm
import litellm.exceptions
import pytest

import crux.adapters.llm.litellm as xlitell
import crux.errors as cerrors
import crux.infra.settings as isettn
import crux.ports.llm as pllm


class _Function:
    """A provider's tool-call function object."""

    def __init__(self, name: str, arguments: Any) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    """A provider's tool-call object."""

    def __init__(self, name: str, arguments: Any, call_id: str = "c1") -> None:
        self.id = call_id
        self.function = _Function(name, arguments)


class _Message:
    """A provider's message object."""

    def __init__(self, content: str = "", tool_calls: list[_ToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    """A provider's choice object."""

    def __init__(self, message: _Message, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    """A provider's response object."""

    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


class TestBackoff:
    """
    How long to wait when a provider says no.
    """

    def test_the_providers_own_stated_wait_is_used(self) -> None:
        """
        Test that a "try again in Ns" hint is honoured rather than guessed at.
        Waiting less than the provider asked for spends the budget the wait was
        supposed to refill, which is how three retries can fail a request that
        needed one pause.
        """
        exc = Exception("Rate limit reached ... Please try again in 12.29s. Upgrade")

        assert xlitell._wait_for(exc, 0) == pytest.approx(14.29)

    def test_the_wait_is_padded_past_the_hint(self) -> None:
        """
        Test that crux arrives after the budget frees rather than exactly as it
        does, since every other client is doing the same arithmetic.
        """
        exc = Exception("try again in 2s")

        assert xlitell._wait_for(exc, 0) > 2.0

    def test_an_unhinted_limit_backs_off_and_grows(self) -> None:
        """
        Test that a provider which says nothing still gets a generous, growing
        pause. A token budget refills over a minute, so a fast retry is not a
        retry — it is a second failure that also spends what it was waiting for.
        """
        exc = Exception("429 too many requests")

        first = xlitell._wait_for(exc, 0)
        second = xlitell._wait_for(exc, 1)

        assert first >= xlitell._FALLBACK_BACKOFF
        assert second > first

    def test_the_wait_is_capped(self) -> None:
        """
        Test that an absurd hint cannot hang a run for an hour.
        """
        exc = Exception("try again in 99999s")

        assert xlitell._wait_for(exc, 0) <= xlitell._MAX_BACKOFF


class TestPatientRetry:
    """
    What happens across a rate limit.
    """

    @pytest.fixture(autouse=True)
    def _no_real_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Replace the backoff sleep so the suite does not actually wait a minute.

        :param monkeypatch: pytest's patcher.
        """

        async def _instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", _instant)

    async def test_a_rate_limit_is_waited_out_and_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test the case the whole mechanism exists for: a free tier refuses once,
        crux waits, and the second attempt goes through.
        """
        attempts = {"n": 0}

        async def _flaky(**_kwargs: Any) -> _Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise litellm.exceptions.RateLimitError(
                    "try again in 1s", model="m", llm_provider="p"
                )
            return _Response([_Choice(_Message(content="hello"))])

        monkeypatch.setattr(litellm, "acompletion", _flaky)
        client = xlitell.LiteLlmClient(isettn.CruxSettings(max_retries=3))

        turn = await client.complete([pllm.Message(role="user", content="hi")])

        assert turn.text == "hello"
        assert attempts["n"] == 2

    async def test_staying_rate_limited_fails_with_actionable_advice(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that giving up says the thing that actually fixes it. A free tier
        counts the max_tokens you reserve, so lowering it is what raises the
        number of calls that fit in a minute — and that is not guessable.
        """

        async def _always(**_kwargs: Any) -> _Response:
            raise litellm.exceptions.RateLimitError("try again in 1s", model="m", llm_provider="p")

        monkeypatch.setattr(litellm, "acompletion", _always)
        client = xlitell.LiteLlmClient(isettn.CruxSettings(max_retries=2))

        with pytest.raises(cerrors.ReasonerError, match="CRUX_MAX_TOKENS"):
            await client.complete([pllm.Message(role="user", content="hi")])

    async def test_a_non_rate_limit_failure_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that a bad model name fails immediately. Retrying a permanent error
        wastes a free tier's budget on a request that can never succeed.
        """
        attempts = {"n": 0}

        async def _broken(**_kwargs: Any) -> _Response:
            attempts["n"] += 1
            raise ValueError("model does not exist")

        monkeypatch.setattr(litellm, "acompletion", _broken)
        client = xlitell.LiteLlmClient(isettn.CruxSettings(max_retries=3))

        with pytest.raises(cerrors.ReasonerError, match="Check CRUX_MODEL"):
            await client.complete([pllm.Message(role="user", content="hi")])
        assert attempts["n"] == 1


class TestDecoding:
    """
    Turning a provider's response into a turn.
    """

    def test_tool_arguments_arrive_as_a_dict_or_as_json(self) -> None:
        """
        Test both shapes providers actually send. Some parse the arguments for
        you and some hand back a JSON string; treating either as the other loses
        the whole call.
        """
        as_dict = xlitell._decode(
            _Response([_Choice(_Message(tool_calls=[_ToolCall("t", {"a": 1})]))])
        )
        as_json = xlitell._decode(
            _Response([_Choice(_Message(tool_calls=[_ToolCall("t", '{"a": 1}')]))])
        )

        assert as_dict.tool_calls[0].arguments == {"a": 1}
        assert as_json.tool_calls[0].arguments == {"a": 1}

    def test_malformed_arguments_drop_the_call_rather_than_the_response(self) -> None:
        """
        Test that unparseable arguments lose one tool call, not everything. The
        caller then raises its own error naming the operation, which is more
        useful than a JSONDecodeError from deep in an adapter.
        """
        turn = xlitell._decode(
            _Response([_Choice(_Message(content="hi", tool_calls=[_ToolCall("t", "{not json")]))])
        )

        assert turn.tool_calls == ()
        assert turn.text == "hi"

    def test_arguments_that_are_not_an_object_are_dropped(self) -> None:
        """
        Test that a JSON array where an object was required is rejected, rather
        than reaching pydantic as a shape it cannot validate.
        """
        turn = xlitell._decode(
            _Response([_Choice(_Message(tool_calls=[_ToolCall("t", "[1, 2]")]))])
        )

        assert turn.tool_calls == ()

    def test_a_response_with_no_choices_is_a_parse_error(self) -> None:
        """
        Test that an empty response fails loudly. Returning an empty turn would
        surface later as "the model proposed nothing", which is a very different
        and much harder bug to trace.
        """
        with pytest.raises(cerrors.ReasonerParseError, match="no usable choice"):
            xlitell._decode(_Response([]))

    def test_plain_text_survives_with_no_tool_calls(self) -> None:
        """
        Test the shape draft() relies on: prose and nothing else is still a turn.
        """
        turn = xlitell._decode(_Response([_Choice(_Message(content="just prose"))]))

        assert turn.text == "just prose"
        assert turn.tool_calls == ()


class TestVerdictLadder:
    """
    What a provider failure means for the chain.

    The order of this ladder is load-bearing, and the obvious rules are wrong in
    ways that were checked against the pinned litellm rather than assumed.
    """

    def test_a_rate_limit_is_waited_out_not_abandoned(self) -> None:
        """
        Test that a busy provider gets waited on. The retry loop exists for
        free-tier token budgets where waiting IS the fix, and going straight to
        a fallback would make the fallback chain the primary path on a free tier
        and quietly spend money elsewhere.
        """
        exc = litellm.exceptions.RateLimitError("busy", model="m", llm_provider="p")
        assert xlitell._verdict(exc) == "retry"

    def test_a_502_or_503_is_retried_even_though_it_is_not_an_internal_error(self) -> None:
        """
        Test the trap: ServiceUnavailableError and BadGatewayError are NOT
        InternalServerError subclasses, so "retry on 5xx" written the obvious way
        misses the two commonest 5xx there are. The status code catches them.
        """
        for name in ("ServiceUnavailableError", "BadGatewayError"):
            cls = getattr(litellm.exceptions, name)
            exc = cls("down", model="m", llm_provider="p")
            assert xlitell._verdict(exc) == "retry", name

    @pytest.mark.parametrize("name", ["ContextWindowExceededError", "ContentPolicyViolationError"])
    def test_some_bad_requests_should_move_to_another_provider(self, name: str) -> None:
        """
        Test the trap that would have cost the most: these are BadRequestError
        SUBCLASSES, so a plain "bad request is fatal" refuses to fall back on
        cases where another provider genuinely succeeds. A context window
        overflow is the clearest: a bigger model just works.
        """
        cls = getattr(litellm.exceptions, name)
        exc = cls(message="no", model="m", llm_provider="p")
        assert xlitell._verdict(exc) == "fallback"

    def test_a_missing_key_moves_on_rather_than_stopping(self) -> None:
        """
        Test that an unset key for one provider says nothing about the next, so
        it is worth trying the rest of the chain.
        """
        exc = litellm.exceptions.AuthenticationError("no key", model="m", llm_provider="p")
        assert xlitell._verdict(exc) == "fallback"

    def test_our_own_malformed_schema_is_fatal(self) -> None:
        """
        Test that a request the schema got wrong stops immediately. It will fail
        identically on every provider, so falling back burns the whole chain and
        hides the bug behind three timeouts.
        """
        exc = litellm.exceptions.BadRequestError("bad schema", model="m", llm_provider="p")
        assert xlitell._verdict(exc) == "fatal"

    def test_an_unknown_error_is_fatal(self) -> None:
        """
        Test that anything unrecognised keeps today's behaviour rather than
        silently entering a retry loop nobody designed for it.
        """
        assert xlitell._verdict(ValueError("who knows")) == "fatal"


class TestFallbackChain:
    """
    Trying the next model when one does not work out.
    """

    @pytest.fixture(autouse=True)
    def _no_real_sleeping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Skip the backoff so the suite does not wait a minute.

        :param monkeypatch: pytest's patcher.
        """

        async def _instant(_seconds: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", _instant)

    def _settings(self, **overrides: Any) -> isettn.CruxSettings:
        """
        Build settings with a two-model chain.

        :param overrides: Fields to replace.
        :return: The settings.
        """
        base: dict[str, Any] = {
            "model": "primary/one",
            "fallback_models": ("backup/one",),
            "max_retries": 1,
            "fallback_retries": 0,
        }
        return isettn.CruxSettings(**(base | overrides))

    async def test_a_dead_provider_is_replaced_by_the_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test the whole point: the primary being down does not end the run.
        """
        seen: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            seen.append(kwargs["model"])
            if kwargs["model"] == "primary/one":
                raise litellm.exceptions.AuthenticationError("no key", model="m", llm_provider="p")
            return _Response([_Choice(_Message(content="from the backup"))])

        monkeypatch.setattr(litellm, "acompletion", _fake)

        turn = await xlitell.LiteLlmClient(self._settings()).complete(
            [pllm.Message(role="user", content="hi")]
        )

        assert turn.text == "from the backup"
        assert seen == ["primary/one", "backup/one"]

    async def test_a_schema_error_stops_the_chain_instead_of_burning_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that a fatal verdict does not try the fallback. Our own malformed
        request fails everywhere, so trying two more providers only delays the
        error and buries its cause.
        """
        seen: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            seen.append(kwargs["model"])
            raise litellm.exceptions.BadRequestError("bad", model="m", llm_provider="p")

        monkeypatch.setattr(litellm, "acompletion", _fake)

        with pytest.raises(cerrors.ReasonerError, match="no other model would do better"):
            await xlitell.LiteLlmClient(self._settings()).complete(
                [pllm.Message(role="user", content="hi")]
            )
        assert seen == ["primary/one"]

    async def test_the_primary_is_waited_on_before_being_abandoned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that a rate-limited primary gets its retries before the chain moves
        on. On a free tier the pause is the fix, and switching away immediately
        would make the fallback the default path.
        """
        seen: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            seen.append(kwargs["model"])
            if kwargs["model"] == "primary/one":
                raise litellm.exceptions.RateLimitError(
                    "try again in 1s", model="m", llm_provider="p"
                )
            return _Response([_Choice(_Message(content="ok"))])

        monkeypatch.setattr(litellm, "acompletion", _fake)

        await xlitell.LiteLlmClient(self._settings(max_retries=2)).complete(
            [pllm.Message(role="user", content="hi")]
        )

        assert seen.count("primary/one") == 3, "one attempt plus two retries"
        assert seen[-1] == "backup/one"

    async def test_prose_where_a_tool_was_demanded_counts_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test the failure that routing makes more likely and that the chain would
        otherwise never see.

        Sending the mechanical operations to a weaker model raises the odds it
        answers in prose instead of calling the tool. Decoding used to happen
        outside the retry loop and the semantic check two layers further out, so
        the client returned success and the fallback never fired. We watched
        exactly this happen live on two providers.
        """
        seen: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            seen.append(kwargs["model"])
            if kwargs["model"] == "primary/one":
                return _Response([_Choice(_Message(content="I think you should..."))])
            return _Response([_Choice(_Message(tool_calls=[_ToolCall("wanted", {"a": 1})]))])

        monkeypatch.setattr(litellm, "acompletion", _fake)

        turn = await xlitell.LiteLlmClient(self._settings()).complete(
            [pllm.Message(role="user", content="hi")], force_tool="wanted"
        )

        assert turn.tool_calls[0].name == "wanted"
        assert seen == ["primary/one", "backup/one"]

    async def test_draft_keeps_its_prose_instead_of_falling_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that the one operation which wants prose opts out. Otherwise every
        successful draft would burn the entire fallback chain looking for a tool
        call nobody needed.
        """
        seen: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            seen.append(kwargs["model"])
            return _Response([_Choice(_Message(content="Add throttling."))])

        monkeypatch.setattr(litellm, "acompletion", _fake)

        turn = await xlitell.LiteLlmClient(self._settings()).complete(
            [pllm.Message(role="user", content="hi")],
            force_tool="draft_task",
            require_tool=False,
        )

        assert turn.text == "Add throttling."
        assert seen == ["primary/one"]

    async def test_exhausting_the_chain_names_every_model_and_why(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test that the final error is diagnostic. "Everything failed" is useless;
        which models were tried and what each said is the actual deliverable.
        """

        async def _fake(**kwargs: Any) -> _Response:
            raise litellm.exceptions.AuthenticationError(
                f"no key for {kwargs['model']}", model="m", llm_provider="p"
            )

        monkeypatch.setattr(litellm, "acompletion", _fake)

        with pytest.raises(cerrors.ReasonerError) as caught:
            await xlitell.LiteLlmClient(self._settings()).complete(
                [pllm.Message(role="user", content="hi")]
            )

        message = str(caught.value)
        assert "primary/one" in message
        assert "backup/one" in message
        assert "CRUX_MAX_TOKENS" in message, "the free-tier hint must survive"

    async def test_a_model_carries_its_own_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test that per-model options reach the wire. A free-tier model needs a
        small token ceiling and the frontier one does not, and applying either
        globally breaks the other.
        """
        sent: list[dict[str, Any]] = []

        async def _fake(**kwargs: Any) -> _Response:
            sent.append(kwargs)
            if kwargs["model"] == "primary/one":
                raise litellm.exceptions.AuthenticationError("no", model="m", llm_provider="p")
            return _Response([_Choice(_Message(content="ok"))])

        monkeypatch.setattr(litellm, "acompletion", _fake)
        settings = self._settings(
            models={"backup/one": isettn.ModelOptions(max_tokens=3500, reasoning_effort="low")}
        )

        await xlitell.LiteLlmClient(settings).complete([pllm.Message(role="user", content="hi")])

        assert sent[0]["max_tokens"] == 8192
        assert sent[1]["max_tokens"] == 3500
        assert sent[1]["reasoning_effort"] == "low"

    async def test_a_named_model_overrides_the_configured_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Test the seam routing depends on: whatever model the caller names leads
        the chain, so a tier split actually reaches the provider.
        """
        sent: list[str] = []

        async def _fake(**kwargs: Any) -> _Response:
            sent.append(kwargs["model"])
            return _Response([_Choice(_Message(content="ok"))])

        monkeypatch.setattr(litellm, "acompletion", _fake)

        await xlitell.LiteLlmClient(self._settings()).complete(
            [pllm.Message(role="user", content="hi")], model="chosen/one"
        )

        assert sent == ["chosen/one"]
