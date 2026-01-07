"""Base plugin interfaces for CogniGate."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import importlib.util
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ArtifactPointer(BaseModel):
    """A durable pointer to a materialized artifact."""
    sink_id: str = Field(description="ID of the sink that stored the artifact")
    uri: str = Field(description="URI, ID, or path to the artifact")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SinkPlugin(ABC):
    """Base class for output sink plugins.

    Sinks receive output from completed jobs and store them durably,
    returning an artifact pointer.
    """

    @property
    @abstractmethod
    def sink_id(self) -> str:
        """Unique identifier for this sink."""
        pass

    @property
    @abstractmethod
    def config_schema(self) -> dict[str, Any]:
        """JSON schema for sink configuration."""
        pass

    @abstractmethod
    async def deliver(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        config: dict[str, Any]
    ) -> ArtifactPointer:
        """Deliver content to this sink.

        Args:
            content: The artifact content to store
            metadata: Metadata about the artifact (task_id, lease_id, etc.)
            config: Sink-specific configuration from the lease

        Returns:
            ArtifactPointer with the location of the stored artifact
        """
        pass

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate configuration against the schema. Override for custom validation."""
        return True


class SinkRegistry:
    """Registry for sink plugins. Discovers and manages sinks."""

    def __init__(self):
        self._sinks: dict[str, SinkPlugin] = {}

    def register(self, sink: SinkPlugin) -> None:
        """Register a sink plugin."""
        if sink.sink_id in self._sinks:
            raise ValueError(f"Sink '{sink.sink_id}' already registered")
        self._sinks[sink.sink_id] = sink
        logger.info(f"Registered sink plugin: {sink.sink_id}")

    def get(self, sink_id: str) -> SinkPlugin | None:
        """Get a sink by ID."""
        return self._sinks.get(sink_id)

    def list_sinks(self) -> list[str]:
        """List all registered sink IDs."""
        return list(self._sinks.keys())

    def _validate_plugin_permissions(self, path: Path) -> bool:
        """Verify plugins directory has safe permissions.
        
        Rejects world-writable or group-writable directories to prevent
        arbitrary code execution via malicious plugin injection.
        
        Args:
            path: Path to plugins directory
            
        Returns:
            True if permissions are safe, False otherwise
        """
        import stat
        import os
        
        try:
            mode = path.stat().st_mode
            
            # Reject if world-writable
            if mode & stat.S_IWOTH:
                logger.critical(
                    f"SECURITY: Plugin directory {path} is world-writable - refusing to load"
                )
                return False
            
            # Reject if group-writable
            if mode & stat.S_IWGRP:
                logger.critical(
                    f"SECURITY: Plugin directory {path} is group-writable - refusing to load"
                )
                return False
            
            # Verify owner is current process user
            current_uid = os.getuid() if hasattr(os, 'getuid') else None
            if current_uid is not None and path.stat().st_uid != current_uid:
                logger.warning(
                    f"SECURITY: Plugin directory {path} not owned by process user "
                    f"(owner={path.stat().st_uid}, process={current_uid})"
                )
                # This is a warning, not a hard failure, for container environments
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate plugin directory permissions: {e}")
            return False

    def discover_plugins(self, plugins_dir: Path) -> None:
        """Discover and load sink plugins from a directory.

        Plugins are Python modules that define a `register_sink` function
        that takes the registry and registers their sink.
        
        SECURITY: Validates directory permissions before loading to prevent
        arbitrary code execution via malicious plugins.
        """
        if not plugins_dir.exists():
            logger.warning(f"Plugins directory does not exist: {plugins_dir}")
            return

        # SECURITY: Validate permissions before loading any plugins
        if not self._validate_plugin_permissions(plugins_dir):
            logger.error(f"Plugin directory {plugins_dir} has unsafe permissions - refusing to load")
            return

        sinks_dir = plugins_dir / "sinks"
        if not sinks_dir.exists():
            return
        
        # Validate sinks subdirectory permissions as well
        if not self._validate_plugin_permissions(sinks_dir):
            logger.error(f"Sinks directory {sinks_dir} has unsafe permissions - refusing to load")
            return

        for plugin_file in sinks_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    f"cognigate_plugin_{plugin_file.stem}",
                    plugin_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "register_sink"):
                        module.register_sink(self)
                        logger.info(f"Loaded sink plugin from: {plugin_file}")
                    else:
                        logger.warning(f"Plugin {plugin_file} has no register_sink function")
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
