#!/usr/bin/env python
"""Golden path script for CogniGate MCP execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from urllib import request, error


def _post_json(url: str, payload: dict, headers: dict[str, str]) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _mcp_call(endpoint: str, api_key: str | None, tool: str, arguments: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": tool,
            "arguments": arguments,
        },
    }
    response = _post_json(endpoint, payload, headers)
    if "error" in response:
        raise RuntimeError(f"MCP error: {response['error']}")
    return response.get("result", {})


def main() -> int:
    parser = argparse.ArgumentParser(description="CogniGate golden path")
    parser.add_argument("--endpoint", default=os.environ.get("COGNIGATE_ENDPOINT", "http://localhost:8000/mcp"))
    parser.add_argument("--api-key", default=os.environ.get("COGNIGATE_API_KEY", ""))
    parser.add_argument("--task-id", default="cognigate-demo-001")
    parser.add_argument("--profile", default="default")
    parser.add_argument("--sink-id", default="stdout")
    args = parser.parse_args()

    payload = {
        "task": "Summarize the text and propose next steps.",
        "context": "LegiVellum coordinates agents using receipts and leases.",
    }

    receipt = _mcp_call(
        args.endpoint,
        args.api_key,
        "cognigate.execute_job",
        {
            "task_id": args.task_id,
            "payload": payload,
            "profile": args.profile,
            "sink_config": {"sink_id": args.sink_id},
        },
    )

    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
