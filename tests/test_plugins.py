"""Tests for plugin system."""

import os
from pathlib import Path

import pytest

from cognigate.plugins.base import SinkPlugin, SinkRegistry


class TestSinkRegistry:
    """Tests for SinkRegistry class."""

    def test_register_sink(self):
        """Test basic sink registration."""
        registry = SinkRegistry()
        
        class TestSink(SinkPlugin):
            sink_id = "test-sink"
            
            async def deliver(self, content, metadata, config):
                pass
            
            def validate_config(self, config):
                return True
        
        sink = TestSink()
        registry.register(sink)
        
        assert "test-sink" in registry.list_sinks()
        assert registry.get("test-sink") == sink

    def test_register_duplicate_sink_raises(self):
        """Test duplicate sink IDs are rejected."""
        registry = SinkRegistry()
        
        class TestSink(SinkPlugin):
            sink_id = "test-sink"
            
            async def deliver(self, content, metadata, config):
                pass
            
            def validate_config(self, config):
                return True
        
        sink1 = TestSink()
        sink2 = TestSink()
        
        registry.register(sink1)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register(sink2)

    def test_get_nonexistent_sink_returns_none(self):
        """Test getting non-existent sink returns None."""
        registry = SinkRegistry()
        
        assert registry.get("nonexistent") is None

    def test_list_sinks_empty_registry(self):
        """Test listing sinks in empty registry."""
        registry = SinkRegistry()
        
        assert registry.list_sinks() == []

    def test_discover_plugins_nonexistent_directory(self):
        """Test discovering plugins in non-existent directory."""
        registry = SinkRegistry()
        
        # Should not raise, just log warning
        registry.discover_plugins(Path("/nonexistent/path"))
        
        assert len(registry.list_sinks()) == 0

    def test_discover_plugins_no_sinks_subdirectory(self, temp_dir):
        """Test discovering plugins when sinks subdirectory doesn't exist."""
        registry = SinkRegistry()
        
        plugins_dir = temp_dir / "plugins"
        plugins_dir.mkdir()
        
        # No sinks subdirectory
        registry.discover_plugins(plugins_dir)
        
        assert len(registry.list_sinks()) == 0


class TestPluginPermissionValidation:
    """Tests for plugin directory permission validation."""

    def test_world_writable_directory_rejected(self, unsafe_temp_dir):
        """Test world-writable directory is rejected."""
        registry = SinkRegistry()
        
        plugins_dir = unsafe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Make world-writable
        os.chmod(plugins_dir, 0o777)
        os.chmod(sinks_dir, 0o777)
        
        registry.discover_plugins(plugins_dir)
        
        # Should not load any plugins
        assert len(registry.list_sinks()) == 0

    def test_group_writable_directory_rejected(self, temp_dir):
        """Test group-writable directory is rejected."""
        registry = SinkRegistry()
        
        plugins_dir = temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Make group-writable
        os.chmod(plugins_dir, 0o770)
        os.chmod(sinks_dir, 0o770)
        
        registry.discover_plugins(plugins_dir)
        
        # Should not load any plugins
        assert len(registry.list_sinks()) == 0

    def test_owner_only_writable_accepted(self, safe_temp_dir):
        """Test owner-only writable directory is accepted."""
        registry = SinkRegistry()
        
        plugins_dir = safe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Create a valid plugin file
        plugin_file = sinks_dir / "valid_plugin.py"
        plugin_file.write_text("""
from cognigate.plugins.base import SinkPlugin

class ValidSink(SinkPlugin):
    sink_id = "valid-sink"
    
    async def deliver(self, content, metadata, config):
        return {"pointer": "test"}
    
    def validate_config(self, config):
        return True

def register_sink(registry):
    registry.register(ValidSink())
""")
        
        # Should accept and load
        registry.discover_plugins(plugins_dir)
        
        # Plugin should be loaded
        assert "valid-sink" in registry.list_sinks()

    def test_sinks_subdirectory_permissions_checked(self, temp_dir):
        """Test that sinks subdirectory permissions are also checked."""
        registry = SinkRegistry()
        
        plugins_dir = temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Make parent safe but sinks world-writable
        os.chmod(plugins_dir, 0o700)
        os.chmod(sinks_dir, 0o777)
        
        registry.discover_plugins(plugins_dir)
        
        # Should reject due to sinks directory being unsafe
        assert len(registry.list_sinks()) == 0


class TestPluginLoading:
    """Tests for plugin loading logic."""

    def test_plugin_with_underscore_prefix_skipped(self, safe_temp_dir):
        """Test plugins starting with underscore are skipped."""
        registry = SinkRegistry()
        
        plugins_dir = safe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Create plugin with underscore prefix
        plugin_file = sinks_dir / "_private_plugin.py"
        plugin_file.write_text("""
def register_sink(registry):
    raise Exception("Should not be called")
""")
        
        # Should skip the file
        registry.discover_plugins(plugins_dir)
        
        assert len(registry.list_sinks()) == 0

    def test_plugin_without_register_function_skipped(self, safe_temp_dir):
        """Test plugins without register_sink function are skipped."""
        registry = SinkRegistry()
        
        plugins_dir = safe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Create plugin without register_sink
        plugin_file = sinks_dir / "no_register.py"
        plugin_file.write_text("""
# Plugin without register_sink function
class SomeSink:
    pass
""")
        
        # Should skip but not crash
        registry.discover_plugins(plugins_dir)
        
        assert len(registry.list_sinks()) == 0

    def test_plugin_loading_error_handled_gracefully(self, safe_temp_dir):
        """Test plugin loading errors are handled without crashing."""
        registry = SinkRegistry()
        
        plugins_dir = safe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Create plugin with syntax error
        plugin_file = sinks_dir / "broken_plugin.py"
        plugin_file.write_text("""
def register_sink(registry):
    raise Exception("Intentional error")
""")
        
        # Should handle error gracefully
        registry.discover_plugins(plugins_dir)
        
        # Should not have loaded the broken plugin
        assert len(registry.list_sinks()) == 0

    def test_multiple_plugins_loaded(self, safe_temp_dir):
        """Test multiple valid plugins are loaded."""
        registry = SinkRegistry()
        
        plugins_dir = safe_temp_dir / "plugins"
        sinks_dir = plugins_dir / "sinks"
        sinks_dir.mkdir(parents=True)
        
        # Create first plugin
        plugin1 = sinks_dir / "sink1.py"
        plugin1.write_text("""
from cognigate.plugins.base import SinkPlugin

class Sink1(SinkPlugin):
    sink_id = "sink-1"
    async def deliver(self, content, metadata, config): pass
    def validate_config(self, config): return True

def register_sink(registry):
    registry.register(Sink1())
""")
        
        # Create second plugin
        plugin2 = sinks_dir / "sink2.py"
        plugin2.write_text("""
from cognigate.plugins.base import SinkPlugin

class Sink2(SinkPlugin):
    sink_id = "sink-2"
    async def deliver(self, content, metadata, config): pass
    def validate_config(self, config): return True

def register_sink(registry):
    registry.register(Sink2())
""")
        
        registry.discover_plugins(plugins_dir)
        
        # Both should be loaded
        assert len(registry.list_sinks()) == 2
        assert "sink-1" in registry.list_sinks()
        assert "sink-2" in registry.list_sinks()


class TestFileSink:
    """Tests for FileSink implementation."""

    def test_file_sink_path_sanitization(self, temp_dir):
        """Test FileSink sanitizes path components."""
        from cognigate.plugins.builtin_sinks import FileSink
        
        sink = FileSink()
        
        # Test sanitization method
        sanitized = sink._sanitize_path_component("../../etc/passwd")
        
        # Should remove dots and slashes
        assert ".." not in sanitized
        assert "/" not in sanitized
        assert "\\" not in sanitized

    def test_file_sink_delivers_to_safe_location(self, temp_dir):
        """Test FileSink delivers files to safe locations."""
        from cognigate.plugins.builtin_sinks import FileSink
        
        sink = FileSink()
        
        config = {"base_path": str(temp_dir)}
        metadata = {
            "task_id": "../../dangerous",
            "lease_id": "../../../evil"
        }
        
        pointer = await sink.deliver(
            content=b"test content",
            metadata=metadata,
            config=config
        )
        
        # Verify file is within temp_dir
        file_path = Path(pointer.uri)
        assert temp_dir in file_path.parents


# Mark async tests
pytestmark = pytest.mark.asyncio
