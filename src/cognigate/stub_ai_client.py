"""Deterministic AI provider for exercising CogniGate without a model.

CogniGate is the only primitive that talks to a language model, which is why it
sits behind a compose profile and never runs in the stack gate: exercising it
costs money and returns something different every time. That leaves the whole
lease -> execute -> artifact -> receipt path unexercised in CI, even though
none of that plumbing is about cognition.

This stubs the *network boundary* rather than the client. `_do_chat_completion`
is the single method that performs HTTP, so overriding only that keeps every
other behaviour real: the circuit breaker still wraps calls, chat_with_tools
still unpacks choices, generate_plan still parses JSON and still falls back to
_extract_json, and token accounting still runs. A stub that replaced AIClient
wholesale would prove far less.

It is deliberately not a mock framework. Output is derived from the input so
tests can assert on it, and is stable across runs so a CI failure means
something changed rather than the model felt different today.

Enable with COGNIGATE_AI_PROVIDER=stub. Not suitable for anything but testing:
it performs no reasoning whatsoever.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .ai_client import AIClient
from .config import AIProviderConfig

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "stub/echo"


class StubAIClient(AIClient):
    """An AIClient whose provider answers locally and deterministically."""

    def __init__(self, config: AIProviderConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.call_count = 0
        logger.warning(
            "cognigate_stub_ai_provider_active model=%s: responses are canned "
            "and no reasoning is performed",
            STUB_MODEL_NAME,
        )

    @staticmethod
    def _last_user_message(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
                # Content can be a list of parts in the OpenAI schema.
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ).strip()
        return ""

    def _stub_content(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str:
        prompt = self._last_user_message(messages)

        # generate_plan asks for JSON and then json.loads() the content, so a
        # prose answer here would exercise the error path on every call.
        if response_format and response_format.get("type") == "json_object":
            # Shaped for CogniGate's planner, which reads "steps" and builds
            # PlanStep from step_number/step_type/description/instructions. A
            # differently-shaped plan parses fine and then executes nothing,
            # which looks like success while testing almost none of the path.
            return json.dumps(
                {
                    "steps": [
                        {
                            "step_number": 1,
                            "step_type": "cognitive",
                            "description": "Echo the request (stub provider).",
                            "instructions": prompt[:500] or "no user message provided",
                        }
                    ],
                    "summary": "Stub plan: one cognitive step, no reasoning performed.",
                    "echo": prompt[:500],
                }
            )

        return f"[stub] {prompt}" if prompt else "[stub] no user message provided"

    async def _do_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Answer locally in the provider's response shape.

        Returns no tool_calls: selecting a tool is a reasoning act, and a stub
        that invented one would drive real side effects from a fake decision.
        """
        self.call_count += 1
        content = self._stub_content(messages, response_format)

        # Rough but stable, so anything asserting on usage sees plausible
        # numbers without implying real tokenization.
        prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages or []) // 4
        completion_tokens = len(content) // 4

        return {
            "id": f"stub-{self.call_count}",
            "object": "chat.completion",
            "model": STUB_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def build_ai_client(settings: Any, **kwargs: Any) -> AIClient:
    """Return the stub or the real client according to configuration."""
    config = settings.get_ai_config()
    if getattr(settings, "ai_provider", "").lower() == "stub":
        return StubAIClient(config, **kwargs)
    return AIClient(config, **kwargs)
