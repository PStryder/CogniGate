"""Tests for the deterministic stub AI provider.

The point of the stub is to exercise CogniGate's plumbing without a model, so
the tests that matter are the ones showing the *real* client behaviour still
runs: only the network boundary is replaced.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from cognigate.ai_client import AIClient
from cognigate.config import AIProviderConfig, Settings
from cognigate.stub_ai_client import STUB_MODEL_NAME, StubAIClient, build_ai_client


def _config() -> AIProviderConfig:
    return AIProviderConfig(
        endpoint="https://unused.invalid",
        api_key="",
        model="stub/echo",
        max_tokens=1024,
    )


@pytest_asyncio.fixture
async def stub():
    client = StubAIClient(_config())
    yield client
    await client.close()


class TestProviderSelection:
    def test_stub_selected_by_configuration(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "stub")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        assert isinstance(build_ai_client(Settings()), StubAIClient)

    def test_real_client_selected_by_default(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_AI_API_KEY", "sk-not-real")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        monkeypatch.delenv("COGNIGATE_AI_PROVIDER", raising=False)
        client = build_ai_client(Settings())
        assert isinstance(client, AIClient)
        assert not isinstance(client, StubAIClient)

    def test_stub_does_not_require_an_api_key(self, monkeypatch) -> None:
        """The stub exists so CogniGate can run without a provider account."""
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "stub")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        monkeypatch.delenv("COGNIGATE_AI_API_KEY", raising=False)
        assert Settings().ai_api_key == ""

    def test_real_provider_still_requires_a_key(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        monkeypatch.delenv("COGNIGATE_AI_PROVIDER", raising=False)
        monkeypatch.delenv("COGNIGATE_AI_API_KEY", raising=False)
        with pytest.raises(Exception, match="ai_api_key is required"):
            Settings()


class TestRealClientBehaviourIsPreserved:
    """Only the HTTP call is stubbed; everything above it must still run."""

    @pytest.mark.asyncio
    async def test_chat_completion_returns_provider_shape(self, stub) -> None:
        data = await stub.chat_completion([{"role": "user", "content": "hello"}])
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["model"] == STUB_MODEL_NAME
        assert "usage" in data

    @pytest.mark.asyncio
    async def test_chat_with_tools_unpacks_the_response(self, stub) -> None:
        """chat_with_tools is real code; it must still parse the stub reply."""
        text, tool_calls = await stub.chat_with_tools(
            [{"role": "user", "content": "do a thing"}],
            tools=[{"type": "function", "function": {"name": "noop"}}],
        )
        assert "do a thing" in text
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_generate_plan_parses_real_json(self, stub) -> None:
        """generate_plan json.loads() the content.

        Returning prose for a json_object request would exercise the error
        path on every call and hide real parse failures.
        """
        plan = await stub.generate_plan([{"role": "user", "content": "plan it"}])
        assert isinstance(plan, dict)
        assert plan["steps"][0]["step_type"] == "cognitive"
        assert "plan it" in plan["echo"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_still_wraps_calls(self, stub) -> None:
        await stub.chat_completion([{"role": "user", "content": "x"}])
        assert stub._circuit_breaker is not None


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_identical_input_gives_identical_output(self, stub) -> None:
        """A CI failure should mean something changed, not that a model varied."""
        messages = [{"role": "user", "content": "same question"}]
        first = await stub.chat_completion(messages)
        second = await stub.chat_completion(messages)
        assert (
            first["choices"][0]["message"]["content"]
            == second["choices"][0]["message"]["content"]
        )

    @pytest.mark.asyncio
    async def test_output_is_derived_from_input(self, stub) -> None:
        """Tests can assert on the result because it echoes the request."""
        data = await stub.chat_completion(
            [{"role": "user", "content": "distinctive-marker-42"}]
        )
        assert "distinctive-marker-42" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_calls_are_counted(self, stub) -> None:
        for _ in range(3):
            await stub.chat_completion([{"role": "user", "content": "x"}])
        assert stub.call_count == 3


class TestSafety:
    @pytest.mark.asyncio
    async def test_stub_never_invents_tool_calls(self, stub) -> None:
        """Selecting a tool is a reasoning act.

        A stub that invented one would drive real side effects from a decision
        nothing actually made.
        """
        _, tool_calls = await stub.chat_with_tools(
            [{"role": "user", "content": "delete everything"}],
            tools=[{"type": "function", "function": {"name": "purge"}}],
        )
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_output_is_marked_as_stubbed(self, stub) -> None:
        """Stub output must be recognisable if it ever escapes into an artifact."""
        data = await stub.chat_completion([{"role": "user", "content": "hello"}])
        assert data["choices"][0]["message"]["content"].startswith("[stub]")
        assert data["model"] == STUB_MODEL_NAME

    @pytest.mark.asyncio
    async def test_no_network_call_is_made(self, stub, monkeypatch) -> None:
        """The stub must not reach the configured endpoint at all."""

        async def _explode(*args, **kwargs):
            raise AssertionError("stub attempted a network call")

        monkeypatch.setattr(stub._client, "post", _explode)
        await stub.chat_completion([{"role": "user", "content": "x"}])


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_missing_user_message_is_handled(self, stub) -> None:
        data = await stub.chat_completion([{"role": "system", "content": "be helpful"}])
        assert "no user message" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_empty_messages_are_handled(self, stub) -> None:
        data = await stub.chat_completion([])
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_structured_content_parts_are_handled(self, stub) -> None:
        """OpenAI content may be a list of parts rather than a string."""
        data = await stub.chat_completion(
            [{"role": "user", "content": [{"type": "text", "text": "part-marker"}]}]
        )
        assert "part-marker" in data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_json_request_always_returns_parseable_json(self, stub) -> None:
        data = await stub.chat_completion(
            [{"role": "user", "content": 'unbalanced " quote {'}],
            response_format={"type": "json_object"},
        )
        json.loads(data["choices"][0]["message"]["content"])


class TestRequireRealCognition:
    """The stub is dangerous precisely because its output looks fine.

    Nothing downstream distinguishes a stubbed artifact from a real one except
    the [stub] marker, so a deployment that meant to reason must be able to
    refuse canned output rather than discover it later in a receipt.
    """

    def test_stub_is_refused_when_real_cognition_is_required(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "stub")
        monkeypatch.setenv("COGNIGATE_AI_REQUIRE_REAL", "true")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        with pytest.raises(Exception, match="refusing to start"):
            Settings()

    def test_missing_key_is_refused_when_real_cognition_is_required(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "openrouter")
        monkeypatch.setenv("COGNIGATE_AI_REQUIRE_REAL", "true")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        monkeypatch.delenv("COGNIGATE_AI_API_KEY", raising=False)
        with pytest.raises(Exception, match="no ai_api_key"):
            Settings()

    def test_real_provider_with_a_key_is_accepted(self, monkeypatch) -> None:
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "openrouter")
        monkeypatch.setenv("COGNIGATE_AI_REQUIRE_REAL", "true")
        monkeypatch.setenv("COGNIGATE_AI_API_KEY", "sk-not-real")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        assert Settings().ai_require_real is True

    def test_stub_remains_usable_without_the_flag(self, monkeypatch) -> None:
        """The default must stay convenient; this is opt-in strictness."""
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "stub")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        monkeypatch.delenv("COGNIGATE_AI_REQUIRE_REAL", raising=False)
        settings = Settings()
        assert settings.ai_provider == "stub"
        assert settings.ai_require_real is False

    def test_failure_is_reported_at_startup_not_at_job_time(self, monkeypatch) -> None:
        """A worker that claims leases and then fails every one is worse.

        The refusal must happen while building Settings, which is before the
        poller exists, so nothing has been claimed.
        """
        monkeypatch.setenv("COGNIGATE_AI_PROVIDER", "stub")
        monkeypatch.setenv("COGNIGATE_AI_REQUIRE_REAL", "true")
        monkeypatch.setenv("COGNIGATE_ALLOW_INSECURE_DEV", "true")
        with pytest.raises(Exception):
            Settings()
