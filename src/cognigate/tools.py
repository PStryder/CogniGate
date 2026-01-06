"""Tool model for CogniGate.

Provides the tools advertised to the AI model:
- mcp.call: Invoke MCP server methods
- artifact.write: Write output artifacts
- receipt.update: Update receipt status (internal)
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

from .models import ToolCall, ToolResult
from .plugins.base import SinkRegistry, ArtifactPointer
from .plugins.mcp_adapter import MCPAdapterRegistry, MCPRequest


logger = logging.getLogger(__name__)


# Tool definitions in OpenAI function calling format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "mcp_call",
            "description": "Call a method on an MCP (Model Context Protocol) server. Use this to interact with external services like GitHub, databases, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "Name of the MCP server to call"
                    },
                    "method": {
                        "type": "string",
                        "description": "MCP method to invoke (e.g., 'resources/read', 'tools/call')"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the MCP method"
                    }
                },
                "required": ["server", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "artifact_write",
            "description": "Write an artifact to the configured output sink. Use this to produce durable outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to write to the artifact"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata for the artifact"
                    }
                },
                "required": ["content"]
            }
        }
    }
]


class ToolContext(BaseModel):
    """Context for tool execution."""
    task_id: str
    lease_id: str
    worker_id: str
    sink_config: dict[str, Any] = Field(default_factory=dict)


class ToolExecutor:
    """Executes tool calls requested by the AI."""

    def __init__(
        self,
        mcp_registry: MCPAdapterRegistry,
        sink_registry: SinkRegistry,
        max_retries: int = 3
    ):
        self.mcp_registry = mcp_registry
        self.sink_registry = sink_registry
        self.max_retries = max_retries
        self._artifacts: list[ArtifactPointer] = []

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for the AI."""
        return TOOL_DEFINITIONS

    def get_artifacts(self) -> list[ArtifactPointer]:
        """Get all artifacts produced during execution."""
        return self._artifacts.copy()

    def clear_artifacts(self) -> None:
        """Clear the artifact list for a new job."""
        self._artifacts.clear()

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute a tool call.

        Args:
            call: The tool call to execute
            context: Execution context

        Returns:
            ToolResult with success/failure and result/error
        """
        logger.info(f"Executing tool: {call.tool_name} (call_id={call.call_id})")

        try:
            if call.tool_name == "mcp_call":
                return await self._execute_mcp_call(call, context)
            elif call.tool_name == "artifact_write":
                return await self._execute_artifact_write(call, context)
            else:
                return ToolResult(
                    call_id=call.call_id,
                    success=False,
                    error=f"Unknown tool: {call.tool_name}"
                )
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=str(e)
            )

    async def _execute_mcp_call(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute an MCP call."""
        args = call.arguments
        server_name = args.get("server")
        method = args.get("method")
        params = args.get("params", {})

        if not server_name or not method:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error="Missing required parameters: server and method"
            )

        adapter = self.mcp_registry.get(server_name)
        if not adapter:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=f"MCP server '{server_name}' not found"
            )

        request = MCPRequest(method=method, params=params)
        response = await adapter.call(request)

        return ToolResult(
            call_id=call.call_id,
            success=response.success,
            result=response.result if response.success else None,
            error=response.error if not response.success else None
        )

    async def _execute_artifact_write(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute an artifact write."""
        args = call.arguments
        content = args.get("content")

        if content is None:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error="Missing required parameter: content"
            )

        # Determine sink from context
        sink_config = context.sink_config
        sink_id = sink_config.get("sink_id", "file")

        sink = self.sink_registry.get(sink_id)
        if not sink:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=f"Sink '{sink_id}' not found"
            )

        # Build metadata
        metadata = {
            "task_id": context.task_id,
            "lease_id": context.lease_id,
            "worker_id": context.worker_id,
            **(args.get("metadata", {}))
        }

        # Deliver to sink
        try:
            pointer = await sink.deliver(content, metadata, sink_config)
            self._artifacts.append(pointer)

            return ToolResult(
                call_id=call.call_id,
                success=True,
                result={"uri": pointer.uri, "sink_id": pointer.sink_id}
            )
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=f"Artifact delivery failed: {e}"
            )


def parse_tool_calls(response_data: dict[str, Any]) -> list[ToolCall]:
    """Parse tool calls from an AI response.

    Args:
        response_data: The AI response containing tool calls

    Returns:
        List of ToolCall objects
    """
    tool_calls = []

    # Handle OpenAI-style tool_calls
    if "tool_calls" in response_data:
        for tc in response_data["tool_calls"]:
            if tc.get("type") == "function":
                func = tc["function"]
                import json
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)

                tool_calls.append(ToolCall(
                    tool_name=func["name"],
                    arguments=args,
                    call_id=tc.get("id", "")
                ))

    return tool_calls
