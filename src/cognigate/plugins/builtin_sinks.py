"""Built-in sink plugins for CogniGate."""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles

from .base import SinkPlugin, ArtifactPointer, SinkRegistry


logger = logging.getLogger(__name__)


class FileSink(SinkPlugin):
    """Sink that writes artifacts to the local filesystem."""

    @property
    def sink_id(self) -> str:
        return "file"

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_path": {
                    "type": "string",
                    "description": "Base directory for artifact storage"
                },
                "filename_template": {
                    "type": "string",
                    "description": "Template for artifact filename",
                    "default": "{task_id}_{timestamp}.txt"
                }
            },
            "required": ["base_path"]
        }

    def _sanitize_path_component(self, value: str) -> str:
        """Remove path separators and dangerous characters from filename component."""
        # Only allow alphanumeric, dash, underscore, and dot
        return "".join(c for c in value if c.isalnum() or c in "-_.")

    async def deliver(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        config: dict[str, Any]
    ) -> ArtifactPointer:
        base_path = Path(config["base_path"])
        base_path.mkdir(parents=True, exist_ok=True)

        template = config.get("filename_template", "{task_id}_{timestamp}.txt")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Sanitize all user-controlled components
        filename = template.format(
            task_id=self._sanitize_path_component(metadata.get("task_id", "unknown")),
            lease_id=self._sanitize_path_component(metadata.get("lease_id", "unknown")),
            timestamp=timestamp,
            uuid=str(uuid4())[:8]
        )

        file_path = base_path / filename
        
        # SECURITY: Verify resolved path is within base_path (prevent traversal)
        try:
            resolved_path = file_path.resolve()
            resolved_base = base_path.resolve()
            if not str(resolved_path).startswith(str(resolved_base)):
                raise ValueError(f"Path traversal attempt detected: {filename}")
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            raise ValueError(f"Invalid file path: {filename}")

        if isinstance(content, str):
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)
        else:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)

        logger.info(f"Artifact written to: {file_path}")

        return ArtifactPointer(
            sink_id=self.sink_id,
            uri=str(file_path.absolute()),
            metadata={"filename": filename, "size": len(content)}
        )


class MCPSink(SinkPlugin):
    """Sink that delivers artifacts via an MCP server.

    Requires an MCP adapter registry to be set before use.
    """

    def __init__(self):
        self._mcp_registry = None

    def set_mcp_registry(self, registry) -> None:
        """Set the MCP adapter registry."""
        self._mcp_registry = registry

    @property
    def sink_id(self) -> str:
        return "mcp"

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mcp_server": {
                    "type": "string",
                    "description": "Name of the MCP server to use"
                },
                "resource_path": {
                    "type": "string",
                    "description": "Path/URI for the resource in MCP"
                }
            },
            "required": ["mcp_server", "resource_path"]
        }

    async def deliver(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        config: dict[str, Any]
    ) -> ArtifactPointer:
        if not self._mcp_registry:
            raise RuntimeError("MCP registry not configured for MCP sink")

        from .mcp_adapter import MCPRequest

        mcp_server = config["mcp_server"]
        resource_path = config["resource_path"]

        adapter = self._mcp_registry.get(mcp_server)
        if not adapter:
            raise ValueError(f"MCP server '{mcp_server}' not found")

        # Format the resource path with metadata
        formatted_path = resource_path.format(
            task_id=metadata.get("task_id", "unknown"),
            lease_id=metadata.get("lease_id", "unknown"),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        # Call MCP to write the resource
        request = MCPRequest(
            method="resources/write",
            params={
                "uri": formatted_path,
                "content": content if isinstance(content, str) else content.decode("utf-8")
            }
        )

        response = await adapter.call(request)

        if not response.success:
            raise RuntimeError(f"MCP delivery failed: {response.error}")

        return ArtifactPointer(
            sink_id=self.sink_id,
            uri=formatted_path,
            metadata={
                "mcp_server": mcp_server,
                "mcp_result": response.result
            }
        )


class StdoutSink(SinkPlugin):
    """Sink that outputs artifacts to stdout (for debugging/testing)."""

    @property
    def sink_id(self) -> str:
        return "stdout"

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Prefix for output",
                    "default": "=== ARTIFACT ==="
                }
            }
        }

    async def deliver(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        config: dict[str, Any]
    ) -> ArtifactPointer:
        prefix = config.get("prefix", "=== ARTIFACT ===")
        task_id = metadata.get("task_id", "unknown")

        print(f"{prefix}")
        print(f"Task: {task_id}")
        print("-" * 40)
        if isinstance(content, bytes):
            print(content.decode("utf-8", errors="replace"))
        else:
            print(content)
        print("-" * 40)

        return ArtifactPointer(
            sink_id=self.sink_id,
            uri=f"stdout://{task_id}",
            metadata={"size": len(content)}
        )


def register_builtin_sinks(registry: SinkRegistry) -> MCPSink:
    """Register all built-in sinks and return the MCP sink for configuration."""
    registry.register(FileSink())
    registry.register(StdoutSink())

    mcp_sink = MCPSink()
    registry.register(mcp_sink)

    return mcp_sink
