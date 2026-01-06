"""Plugin system for CogniGate - sinks and MCP adapters."""

from .base import SinkPlugin, SinkRegistry, ArtifactPointer
from .mcp_adapter import MCPAdapter, MCPAdapterRegistry

__all__ = [
    "SinkPlugin",
    "SinkRegistry",
    "ArtifactPointer",
    "MCPAdapter",
    "MCPAdapterRegistry",
]
