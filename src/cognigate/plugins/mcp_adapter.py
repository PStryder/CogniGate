"""MCP (Model Context Protocol) adapter for CogniGate."""

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ..config import MCPEndpoint


logger = logging.getLogger(__name__)


class MCPRequest(BaseModel):
    """A request to an MCP server."""
    method: str = Field(description="MCP method to call")
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters")


class MCPResponse(BaseModel):
    """Response from an MCP server."""
    success: bool = Field(description="Whether the call succeeded")
    result: Any = Field(default=None, description="Result data if successful")
    error: str | None = Field(default=None, description="Error message if failed")
    error_code: str | None = Field(default=None, description="Error code if failed")


class MCPAdapter:
    """Adapter for communicating with a single MCP server.

    Handles authentication, request formatting, retries, and error normalization.
    """

    # Methods that are always allowed (read-only)
    READ_ONLY_METHODS = frozenset([
        "resources/list",
        "resources/read",
        "tools/list",
        "prompts/list",
        "prompts/get",
    ])

    # Methods that modify state (require write permission)
    WRITE_METHODS = frozenset([
        "tools/call",
        "resources/write",
        "resources/delete",
    ])

    def __init__(
        self,
        endpoint: MCPEndpoint,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 3
    ):
        self.endpoint = endpoint
        self.name = endpoint.name
        self.read_only = endpoint.read_only
        self.max_retries = max_retries
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def close(self) -> None:
        """Close the HTTP client if we own it."""
        if self._owns_client:
            await self._client.aclose()

    def _is_allowed(self, method: str) -> bool:
        """Check if a method is allowed given read_only setting."""
        if not self.read_only:
            return True
        # In read-only mode, only allow read methods
        return method in self.READ_ONLY_METHODS

    async def call(self, request: MCPRequest) -> MCPResponse:
        """Call an MCP method on the server.

        Args:
            request: The MCP request to execute

        Returns:
            MCPResponse with result or error
        """
        if not self._is_allowed(request.method):
            return MCPResponse(
                success=False,
                error=f"Method '{request.method}' not allowed in read-only mode",
                error_code="PERMISSION_DENIED"
            )

        headers = {"Content-Type": "application/json"}
        if self.endpoint.auth_token:
            headers["Authorization"] = f"Bearer {self.endpoint.auth_token}"

        payload = {
            "jsonrpc": "2.0",
            "method": request.method,
            "params": request.params,
            "id": 1
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._client.post(
                    self.endpoint.endpoint,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

                if "error" in data:
                    error = data["error"]
                    return MCPResponse(
                        success=False,
                        error=error.get("message", "Unknown error"),
                        error_code=str(error.get("code", "UNKNOWN"))
                    )

                return MCPResponse(
                    success=True,
                    result=data.get("result")
                )

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                logger.warning(f"MCP call attempt {attempt + 1} failed: {last_error}")
            except httpx.RequestError as e:
                last_error = str(e)
                logger.warning(f"MCP call attempt {attempt + 1} failed: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error in MCP call: {last_error}")
                break

        return MCPResponse(
            success=False,
            error=last_error or "Unknown error after retries",
            error_code="REQUEST_FAILED"
        )


class MCPAdapterRegistry:
    """Registry for MCP adapters. Manages connections to MCP servers."""

    def __init__(self):
        self._adapters: dict[str, MCPAdapter] = {}

    def register(self, endpoint: MCPEndpoint) -> None:
        """Register an MCP endpoint and create an adapter for it."""
        if not endpoint.enabled:
            logger.info(f"Skipping disabled MCP endpoint: {endpoint.name}")
            return

        if endpoint.name in self._adapters:
            raise ValueError(f"MCP adapter '{endpoint.name}' already registered")

        adapter = MCPAdapter(endpoint)
        self._adapters[endpoint.name] = adapter
        logger.info(f"Registered MCP adapter: {endpoint.name} (read_only={endpoint.read_only})")

    def get(self, name: str) -> MCPAdapter | None:
        """Get an MCP adapter by name."""
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())

    async def close_all(self) -> None:
        """Close all adapters."""
        for adapter in self._adapters.values():
            await adapter.close()
