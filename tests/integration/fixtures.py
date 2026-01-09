"""Shared fixtures and mocks for integration tests."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from cognigate.config import Settings, Bootstrap, InstructionProfile, MCPEndpoint
from cognigate.models import Lease, Receipt, JobStatus, ExecutionPlan, PlanStep, PlanStepType


class MockAsyncGateServer:
    """Mock AsyncGate server for integration testing."""

    def __init__(self):
        self.leases: list[Lease] = []
        self.claimed_leases: set[str] = set()
        self.receipts: list[Receipt] = []
        self.extended_leases: list[str] = []
        self.poll_count = 0

    def add_lease(self, lease: Lease) -> None:
        """Add a lease to be claimed."""
        self.leases.append(lease)

    def get_receipts(self, task_id: str | None = None) -> list[Receipt]:
        """Get all receipts, optionally filtered by task_id."""
        if task_id:
            return [r for r in self.receipts if r.task_id == task_id]
        return self.receipts

    async def handle_claim(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle lease claim request."""
        self.poll_count += 1
        available = [l for l in self.leases if l.lease_id not in self.claimed_leases]

        if not available:
            return {"tasks": []}

        lease = available[0]
        self.claimed_leases.add(lease.lease_id)

        return {
            "tasks": [{
                "lease_id": lease.lease_id,
                "task_id": lease.task_id,
                "payload": lease.payload,
                "requirements": lease.constraints
            }]
        }

    async def handle_progress(self, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Handle progress report."""
        receipt = Receipt(
            lease_id=request["lease_id"],
            task_id=task_id,
            worker_id=request.get("worker_id", "test-worker"),
            status=JobStatus.RUNNING,
            timestamp=datetime.now(timezone.utc),
            summary=request.get("progress", {}).get("summary", "")
        )
        self.receipts.append(receipt)
        return {"accepted": True}

    async def handle_complete(self, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Handle task completion."""
        result = request.get("result", {})
        receipt = Receipt(
            lease_id=request["lease_id"],
            task_id=task_id,
            worker_id=request.get("worker_id", "test-worker"),
            status=JobStatus.COMPLETE,
            timestamp=datetime.now(timezone.utc),
            summary=result.get("summary", ""),
            artifact_pointers=result.get("artifact_pointers", [])
        )
        self.receipts.append(receipt)
        return {"accepted": True}

    async def handle_fail(self, task_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Handle task failure."""
        error = request.get("error", {})
        receipt = Receipt(
            lease_id=request["lease_id"],
            task_id=task_id,
            worker_id=request.get("worker_id", "test-worker"),
            status=JobStatus.FAILED,
            timestamp=datetime.now(timezone.utc),
            error_metadata={"code": error.get("code"), "message": error.get("message")}
        )
        self.receipts.append(receipt)
        return {"accepted": True}

    async def handle_extend(self, lease_id: str) -> dict[str, Any]:
        """Handle lease extension."""
        self.extended_leases.append(lease_id)
        return {"extended": True}


class MockAIProvider:
    """Mock AI provider for integration testing."""

    def __init__(self):
        self.requests: list[dict[str, Any]] = []
        self.plan_response: dict[str, Any] = {
            "steps": [
                {"step_number": 1, "step_type": "cognitive", "description": "Analyze the request"},
                {"step_number": 2, "step_type": "output_generation", "description": "Generate output"}
            ],
            "summary": "Test execution plan"
        }
        self.chat_responses: list[str] = ["Test response"]
        self.response_index = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.should_fail = False
        self.fail_count = 0
        self.max_failures = 0

    def set_plan_response(self, plan: dict[str, Any]) -> None:
        """Set the plan response."""
        self.plan_response = plan

    def set_chat_responses(self, responses: list[str]) -> None:
        """Set chat responses in order."""
        self.chat_responses = responses
        self.response_index = 0

    def set_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """Set tool calls to return."""
        self.tool_calls = tool_calls

    def set_failure_mode(self, max_failures: int = 1) -> None:
        """Set failure mode for testing retries."""
        self.should_fail = True
        self.max_failures = max_failures
        self.fail_count = 0

    async def handle_chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle chat completion request."""
        self.requests.append(request)

        # Check for failure mode
        if self.should_fail and self.fail_count < self.max_failures:
            self.fail_count += 1
            raise httpx.HTTPStatusError(
                "Service unavailable",
                request=MagicMock(),
                response=MagicMock(status_code=503)
            )

        # Check if this is a planning request
        if request.get("response_format", {}).get("type") == "json_object":
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps(self.plan_response)
                    },
                    "finish_reason": "stop"
                }]
            }

        # Regular chat response
        response_text = self.chat_responses[min(self.response_index, len(self.chat_responses) - 1)]
        self.response_index += 1

        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "tool_calls": self.tool_calls
                },
                "finish_reason": "stop"
            }]
        }


class MockMCPServer:
    """Mock MCP server for integration testing."""

    def __init__(self, name: str = "test-mcp"):
        self.name = name
        self.requests: list[dict[str, Any]] = []
        self.responses: dict[str, Any] = {}
        self.should_fail = False

    def set_response(self, method: str, result: Any) -> None:
        """Set response for a specific method."""
        self.responses[method] = result

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP request."""
        self.requests.append(request)

        if self.should_fail:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": "Server error"},
                "id": request.get("id", 1)
            }

        method = request.get("method", "")
        result = self.responses.get(method, {"status": "ok"})

        return {
            "jsonrpc": "2.0",
            "result": result,
            "id": request.get("id", 1)
        }


class IntegrationTestHarness:
    """Test harness for full integration tests."""

    def __init__(self):
        self.asyncgate = MockAsyncGateServer()
        self.ai_provider = MockAIProvider()
        self.mcp_servers: dict[str, MockMCPServer] = {}
        self._http_client: httpx.AsyncClient | None = None

    def add_mcp_server(self, name: str) -> MockMCPServer:
        """Add a mock MCP server."""
        server = MockMCPServer(name)
        self.mcp_servers[name] = server
        return server

    def create_mock_http_client(self) -> httpx.AsyncClient:
        """Create a mock HTTP client that routes to mock servers."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            body = json.loads(request.content) if request.content else {}

            # Route to AsyncGate
            if "/v1/leases/claim" in url:
                result = await self.asyncgate.handle_claim(body)
                return httpx.Response(200, json=result)
            elif "/v1/tasks/" in url and "/progress" in url:
                task_id = url.split("/v1/tasks/")[1].split("/progress")[0]
                result = await self.asyncgate.handle_progress(task_id, body)
                return httpx.Response(200, json=result)
            elif "/v1/tasks/" in url and "/complete" in url:
                task_id = url.split("/v1/tasks/")[1].split("/complete")[0]
                result = await self.asyncgate.handle_complete(task_id, body)
                return httpx.Response(200, json=result)
            elif "/v1/tasks/" in url and "/fail" in url:
                task_id = url.split("/v1/tasks/")[1].split("/fail")[0]
                result = await self.asyncgate.handle_fail(task_id, body)
                return httpx.Response(200, json=result)
            elif "/v1/leases/" in url and "/extend" in url:
                lease_id = url.split("/v1/leases/")[1].split("/extend")[0]
                result = await self.asyncgate.handle_extend(lease_id)
                return httpx.Response(200, json=result)

            # Route to AI provider
            elif "/chat/completions" in url:
                try:
                    result = await self.ai_provider.handle_chat_completion(body)
                    return httpx.Response(200, json=result)
                except httpx.HTTPStatusError as e:
                    return httpx.Response(e.response.status_code, json={"error": str(e)})

            # Route to MCP servers
            for name, server in self.mcp_servers.items():
                if name in url or "mcp" in url.lower():
                    result = await server.handle_request(body)
                    return httpx.Response(200, json=result)

            return httpx.Response(404, json={"error": "Not found"})

        transport = httpx.MockTransport(mock_request)
        self._http_client = httpx.AsyncClient(transport=transport)
        return self._http_client

    async def close(self) -> None:
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()


@pytest.fixture
def integration_harness():
    """Create integration test harness."""
    harness = IntegrationTestHarness()
    yield harness
    asyncio.get_event_loop().run_until_complete(harness.close())


@pytest.fixture
def mock_asyncgate():
    """Create mock AsyncGate server."""
    return MockAsyncGateServer()


@pytest.fixture
def mock_ai_provider():
    """Create mock AI provider."""
    return MockAIProvider()


@pytest.fixture
def mock_mcp_server():
    """Create mock MCP server."""
    return MockMCPServer()


@pytest.fixture
def simple_lease():
    """Create a simple lease for testing."""
    return Lease(
        lease_id="integration-lease-001",
        task_id="integration-task-001",
        payload={
            "task": "Process this test request",
            "context": "Integration test context"
        },
        profile="default",
        sink_config={},
        constraints={"max_tokens": 1000}
    )


@pytest.fixture
def complex_lease():
    """Create a complex lease with tool requirements."""
    return Lease(
        lease_id="integration-lease-002",
        task_id="integration-task-002",
        payload={
            "task": "Fetch data from MCP and process it",
            "context": "Requires MCP tool calls",
            "mcp_target": "test-mcp"
        },
        profile="default",
        sink_config={"sink_id": "file", "path": "/tmp/output.txt"},
        constraints={"max_tokens": 2000, "timeout": 120}
    )


@pytest.fixture
def test_instruction_profile():
    """Create test instruction profile."""
    return InstructionProfile(
        name="default",
        system_instructions="You are a helpful test assistant.",
        formatting_constraints="Respond concisely.",
        planning_schema={
            "type": "object",
            "properties": {
                "steps": {"type": "array"},
                "summary": {"type": "string"}
            }
        },
        tool_usage_rules="Use tools when necessary."
    )


@pytest.fixture
def test_mcp_endpoint():
    """Create test MCP endpoint configuration."""
    return MCPEndpoint(
        name="test-mcp",
        endpoint="http://localhost:9000/mcp",
        auth_token="test-token",
        read_only=True,
        enabled=True
    )


@pytest.fixture
def integration_settings(tmp_path):
    """Create settings for integration tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # Create default profile
    default_profile = profiles_dir / "default.yaml"
    default_profile.write_text("""
name: default
system_instructions: You are a helpful test assistant.
formatting_constraints: Respond concisely.
planning_schema:
  type: object
  properties:
    steps:
      type: array
    summary:
      type: string
tool_usage_rules: Use tools when necessary.
""")

    return Settings(
        host="127.0.0.1",
        port=8001,
        config_dir=config_dir,
        plugins_dir=plugins_dir,
        profiles_dir=profiles_dir,
        asyncgate_endpoint="http://localhost:8080",
        asyncgate_auth_token="test-token",
        ai_endpoint="http://localhost:9001/v1",
        ai_api_key="test-ai-key",
        ai_model="test-model",
        api_key="test-api-key",
        allow_insecure_dev=True,
        worker_id="test-worker-1",
        max_concurrent_jobs=2,
        polling_interval=0.1,
        max_retries=3
    )
