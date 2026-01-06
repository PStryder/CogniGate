"""Example custom sink plugin.

To create a custom sink:
1. Create a class that inherits from SinkPlugin
2. Implement sink_id, config_schema, and deliver()
3. Define a register_sink(registry) function

Drop this file in the plugins/sinks directory and restart CogniGate.
"""

from typing import Any
from datetime import datetime, timezone

# Import from cognigate when installed
# from cognigate.plugins.base import SinkPlugin, ArtifactPointer, SinkRegistry


class ExampleWebhookSink:
    """Example sink that posts artifacts to a webhook URL."""

    @property
    def sink_id(self) -> str:
        return "webhook"

    @property
    def config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "webhook_url": {
                    "type": "string",
                    "description": "URL to POST artifacts to"
                },
                "include_metadata": {
                    "type": "boolean",
                    "default": True
                }
            },
            "required": ["webhook_url"]
        }

    async def deliver(
        self,
        content: str | bytes,
        metadata: dict[str, Any],
        config: dict[str, Any]
    ):
        """Deliver artifact to webhook.

        In a real implementation:
        - POST content to config["webhook_url"]
        - Include metadata in headers or body
        - Handle retries and errors
        """
        import httpx

        webhook_url = config["webhook_url"]
        include_metadata = config.get("include_metadata", True)

        payload = {
            "content": content if isinstance(content, str) else content.decode("utf-8"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if include_metadata:
            payload["metadata"] = metadata

        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()

        # Return artifact pointer
        from cognigate.plugins.base import ArtifactPointer

        return ArtifactPointer(
            sink_id="webhook",
            uri=webhook_url,
            metadata={"status_code": response.status_code}
        )


def register_sink(registry):
    """Called by CogniGate to register this sink."""
    # Uncomment to enable:
    # registry.register(ExampleWebhookSink())
    pass
