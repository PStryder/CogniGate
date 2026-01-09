"""AI provider client for CogniGate."""

import json
from typing import Any

import httpx

from .config import AIProviderConfig
from .observability import get_logger
from .metrics import track_ai_request, record_ai_tokens
from .circuit_breaker import CircuitBreaker, CircuitBreakerError


logger = get_logger(__name__)


class AIClient:
    """Client for AI provider (OpenRouter/OpenAI-compatible API)."""

    def __init__(
        self,
        config: AIProviderConfig,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.endpoint = config.endpoint.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.max_tokens = config.max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)
        self._circuit_breaker = CircuitBreaker(
            name="ai_provider",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            # Don't trip on client errors (bad requests)
            excluded_exceptions=(ValueError, json.JSONDecodeError)
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://cognigate.local",
            "X-Title": "CogniGate"
        }

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.7,
        response_format: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of messages in OpenAI format
            tools: Optional list of tool definitions
            tool_choice: How to handle tool selection
            temperature: Sampling temperature
            response_format: Optional response format specification

        Returns:
            The API response data

        Raises:
            CircuitBreakerError: If circuit breaker is open
            httpx.HTTPStatusError: On HTTP errors
        """
        return await self._circuit_breaker.call(
            self._do_chat_completion,
            messages, tools, tool_choice, temperature, response_format
        )

    async def _do_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict,
        temperature: float,
        response_format: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Internal method to perform chat completion (for circuit breaker)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": temperature
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        if response_format:
            payload["response_format"] = response_format

        logger.debug(
            "ai_request",
            model=self.model,
            message_count=len(messages)
        )

        with track_ai_request("chat"):
            response = await self._client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers=self._headers()
            )
            response.raise_for_status()

        data = response.json()

        # Track token usage if available
        usage = data.get("usage", {})
        if usage:
            record_ai_tokens(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )

        logger.debug(
            "ai_response",
            finish_reason=data["choices"][0].get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens")
        )

        return data

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Send a chat request that may involve tool calls.

        Args:
            messages: List of messages
            tools: List of tool definitions
            temperature: Sampling temperature

        Returns:
            Tuple of (text_content, tool_calls)
        """
        data = await self.chat_completion(
            messages=messages,
            tools=tools,
            temperature=temperature
        )

        choice = data["choices"][0]
        message = choice["message"]

        text_content = message.get("content")
        tool_calls = message.get("tool_calls", [])

        return text_content, tool_calls

    async def generate_plan(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.3
    ) -> dict[str, Any]:
        """Generate an execution plan.

        Args:
            messages: Messages including system prompt and task
            temperature: Lower temperature for more consistent plans

        Returns:
            Parsed plan as a dictionary
        """
        # Request JSON response for planning
        data = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        choice = data["choices"][0]
        content = choice["message"].get("content", "{}")

        try:
            plan = json.loads(content)
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Try to extract JSON from the response
            return self._extract_json(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """Try to extract JSON from text that may have extra content."""
        # Look for JSON block
        import re

        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Return empty plan
        return {"steps": [], "summary": "Failed to parse plan"}
