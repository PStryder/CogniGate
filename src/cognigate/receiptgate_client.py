"""ReceiptGate MCP client for CogniGate receipt emission."""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .observability import get_logger


logger = get_logger(__name__)


def _normalize_endpoint(endpoint: str) -> str:
    if not endpoint:
        return ""
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/mcp"):
        endpoint = f"{endpoint}/mcp"
    return endpoint


class ReceiptGateClient:
    """Lightweight client for emitting LegiVellum receipts via MCP."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = bool(settings.receiptgate_emit_receipts and settings.receiptgate_endpoint)
        self._endpoint = _normalize_endpoint(settings.receiptgate_endpoint)
        self._token = settings.receiptgate_auth_token
        self._client = httpx.AsyncClient(timeout=10.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def emit_receipt(self, receipt: dict[str, Any]) -> bool:
        if not self._enabled:
            return False

        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "receiptgate.submit_receipt",
                "arguments": {"receipt": receipt},
            },
        }

        try:
            response = await self._client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                logger.warning("receiptgate_receipt_emit_failed", error=str(data["error"]))
                return False
            return True
        except Exception as exc:
            logger.warning("receiptgate_receipt_emit_failed", error=str(exc))
            return False
