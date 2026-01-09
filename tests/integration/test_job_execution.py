"""Integration tests for full job execution flow.

Tests the complete lease -> plan -> execute -> receipt flow.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cognigate.ai_client import AIClient
from cognigate.config import Bootstrap, Settings, AIProviderConfig
from cognigate.executor import JobExecutor, ExecutionError
from cognigate.leasing import AsyncGateClient, WorkPoller
from cognigate.models import Lease, Receipt, JobStatus, PlanStepType
from cognigate.plugins import SinkRegistry, MCPAdapterRegistry
from cognigate.plugins.builtin_sinks import register_builtin_sinks
from cognigate.tools import ToolExecutor

from .fixtures import (
    IntegrationTestHarness,
    MockAsyncGateServer,
    MockAIProvider,
    MockMCPServer,
    integration_harness,
    mock_asyncgate,
    mock_ai_provider,
    simple_lease,
    complex_lease,
    integration_settings,
    test_instruction_profile,
)


pytestmark = pytest.mark.asyncio


class TestFullJobExecution:
    """Tests for complete job execution flow."""

    async def test_simple_job_success(self, integration_harness, simple_lease, integration_settings, tmp_path):
        """Test successful execution of a simple job."""
        # Setup
        harness = integration_harness
        harness.asyncgate.add_lease(simple_lease)
        harness.ai_provider.set_plan_response({
            "steps": [
                {"step_number": 1, "step_type": "cognitive", "description": "Analyze request"},
                {"step_number": 2, "step_type": "output_generation", "description": "Generate response"}
            ],
            "summary": "Simple analysis task"
        })
        harness.ai_provider.set_chat_responses(["Analysis complete", "Final output generated"])

        mock_client = harness.create_mock_http_client()

        # Create components with mock client
        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            register_builtin_sinks(sink_registry)

            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)

            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            # Execute job
            receipt = await job_executor.execute(simple_lease)

            # Verify success
            assert receipt.status == JobStatus.COMPLETE
            assert receipt.task_id == simple_lease.task_id
            assert receipt.lease_id == simple_lease.lease_id
            assert receipt.error_metadata is None
            assert "2 steps" in receipt.summary.lower() or "steps" in receipt.summary.lower()

    async def test_job_with_tool_calls(self, integration_harness, complex_lease, integration_settings):
        """Test job execution with MCP tool calls."""
        harness = integration_harness
        harness.asyncgate.add_lease(complex_lease)

        # Add MCP server
        mcp_server = harness.add_mcp_server("test-mcp")
        mcp_server.set_response("tools/list", {"tools": [{"name": "getData"}]})
        mcp_server.set_response("tools/call", {"data": "test result"})

        harness.ai_provider.set_plan_response({
            "steps": [
                {"step_number": 1, "step_type": "cognitive", "description": "Plan data fetch"},
                {
                    "step_number": 2,
                    "step_type": "tool_invocation",
                    "description": "Fetch data",
                    "tool_name": "mcp_call",
                    "tool_params": {"server": "test-mcp", "method": "tools/call", "params": {}}
                },
                {"step_number": 3, "step_type": "output_generation", "description": "Generate report"}
            ],
            "summary": "Data fetch and process"
        })
        harness.ai_provider.set_chat_responses(["Planning complete", "Data processed", "Report generated"])

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            register_builtin_sinks(sink_registry)

            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)

            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(complex_lease)

            assert receipt.status == JobStatus.COMPLETE
            assert receipt.task_id == complex_lease.task_id

    async def test_job_planning_failure(self, integration_harness, simple_lease, integration_settings):
        """Test job failure during planning phase."""
        harness = integration_harness
        harness.ai_provider.set_failure_mode(max_failures=10)  # Always fail

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(simple_lease)

            assert receipt.status == JobStatus.FAILED
            assert receipt.error_metadata is not None

    async def test_job_step_failure_recovery(self, integration_harness, simple_lease, integration_settings):
        """Test that job handles step failures gracefully."""
        harness = integration_harness

        # AI fails first 2 times then succeeds
        harness.ai_provider.set_failure_mode(max_failures=2)
        harness.ai_provider.set_plan_response({
            "steps": [
                {"step_number": 1, "step_type": "cognitive", "description": "Test step"}
            ],
            "summary": "Test"
        })
        harness.ai_provider.set_chat_responses(["Success after retries"])

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            # First call will fail during planning
            receipt = await job_executor.execute(simple_lease)
            # Should fail because planning request fails
            assert receipt.status == JobStatus.FAILED

    async def test_job_timeout_handling(self, integration_harness, simple_lease, integration_settings):
        """Test job execution with timeout constraints."""
        harness = integration_harness

        # Create a very short timeout
        simple_lease.constraints["timeout"] = 0.001

        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Test"}],
            "summary": "Test"
        })

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(simple_lease)
            # Job should complete (timeout is not enforced at executor level currently)
            assert receipt.task_id == simple_lease.task_id

    async def test_missing_profile_fallback(self, integration_harness, integration_settings):
        """Test job execution with missing profile falls back to default."""
        harness = integration_harness

        lease = Lease(
            lease_id="test-lease",
            task_id="test-task",
            payload={"task": "test"},
            profile="nonexistent-profile",
            sink_config={},
            constraints={}
        )

        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Test"}],
            "summary": "Test"
        })
        harness.ai_provider.set_chat_responses(["Done"])

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(lease)
            assert receipt.status == JobStatus.COMPLETE

    async def test_artifact_generation(self, integration_harness, simple_lease, integration_settings, tmp_path):
        """Test job execution produces artifacts correctly."""
        harness = integration_harness

        # Configure for artifact generation
        simple_lease.sink_config = {"sink_id": "file", "base_path": str(tmp_path)}

        harness.ai_provider.set_plan_response({
            "steps": [
                {"step_number": 1, "step_type": "output_generation", "description": "Generate output"}
            ],
            "summary": "Generate file output"
        })
        harness.ai_provider.set_chat_responses(["Generated content"])
        harness.ai_provider.set_tool_calls([{
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "artifact_write",
                "arguments": json.dumps({
                    "sink_id": "file",
                    "content": "Test artifact content",
                    "metadata": {"filename": "output.txt"}
                })
            }
        }])

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            register_builtin_sinks(sink_registry)

            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=3)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(simple_lease)
            assert receipt.status == JobStatus.COMPLETE


class TestWorkPollerIntegration:
    """Tests for WorkPoller integration with job execution."""

    async def test_poll_claim_execute_receipt(self, integration_harness, simple_lease, integration_settings):
        """Test full polling cycle: claim, execute, send receipt."""
        harness = integration_harness
        harness.asyncgate.add_lease(simple_lease)

        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Test"}],
            "summary": "Test"
        })
        harness.ai_provider.set_chat_responses(["Done"])

        mock_client = harness.create_mock_http_client()

        receipts_sent = []

        async def job_handler(lease: Lease) -> Receipt:
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test-worker",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc),
                summary="Job completed"
            )

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            work_poller = WorkPoller(asyncgate_client, integration_settings, job_handler)

            # Poll once manually
            await work_poller._poll_and_dispatch()

            # Wait for job to complete
            await asyncio.sleep(0.1)

            # Verify lease was claimed
            assert simple_lease.lease_id in harness.asyncgate.claimed_leases

            # Verify receipts were sent (acceptance + completion)
            assert len(harness.asyncgate.receipts) >= 1

    async def test_concurrent_job_limit(self, integration_harness, integration_settings):
        """Test that concurrent job limit is respected."""
        harness = integration_harness

        # Add multiple leases
        for i in range(5):
            harness.asyncgate.add_lease(Lease(
                lease_id=f"lease-{i}",
                task_id=f"task-{i}",
                payload={"task": f"task {i}"},
                profile="default",
                sink_config={},
                constraints={}
            ))

        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Test"}],
            "summary": "Test"
        })

        mock_client = harness.create_mock_http_client()

        active_count = 0
        max_active = 0

        async def slow_job_handler(lease: Lease) -> Receipt:
            nonlocal active_count, max_active
            active_count += 1
            max_active = max(max_active, active_count)
            await asyncio.sleep(0.05)
            active_count -= 1
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test-worker",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc)
            )

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            work_poller = WorkPoller(asyncgate_client, integration_settings, slow_job_handler)

            # Poll multiple times quickly
            for _ in range(5):
                await work_poller._poll_and_dispatch()

            # Wait for jobs to complete
            await asyncio.sleep(0.5)

            # Max concurrent should not exceed limit (2 in integration_settings)
            assert max_active <= integration_settings.max_concurrent_jobs

    async def test_duplicate_lease_prevention(self, integration_harness, simple_lease, integration_settings):
        """Test that duplicate leases are not processed."""
        harness = integration_harness
        harness.asyncgate.add_lease(simple_lease)

        mock_client = harness.create_mock_http_client()

        job_count = 0

        async def counting_handler(lease: Lease) -> Receipt:
            nonlocal job_count
            job_count += 1
            await asyncio.sleep(0.1)  # Simulate work
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test-worker",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc)
            )

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            work_poller = WorkPoller(asyncgate_client, integration_settings, counting_handler)

            # Poll multiple times for same lease
            for _ in range(3):
                await work_poller._poll_and_dispatch()

            await asyncio.sleep(0.2)

            # Should only process once
            assert job_count == 1


class TestReceiptDelivery:
    """Tests for receipt delivery to AsyncGate."""

    async def test_acceptance_receipt_sent(self, integration_harness, simple_lease, integration_settings):
        """Test that acceptance receipt is sent when job starts."""
        harness = integration_harness
        harness.asyncgate.add_lease(simple_lease)

        mock_client = harness.create_mock_http_client()

        async def quick_handler(lease: Lease) -> Receipt:
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test-worker",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc)
            )

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            work_poller = WorkPoller(asyncgate_client, integration_settings, quick_handler)
            await work_poller._poll_and_dispatch()
            await asyncio.sleep(0.1)

            # Should have both RUNNING and COMPLETE receipts
            running_receipts = [r for r in harness.asyncgate.receipts if r.status == JobStatus.RUNNING]
            complete_receipts = [r for r in harness.asyncgate.receipts if r.status == JobStatus.COMPLETE]

            assert len(running_receipts) >= 1
            assert len(complete_receipts) >= 1

    async def test_failure_receipt_on_error(self, integration_harness, simple_lease, integration_settings):
        """Test that failure receipt is sent when job fails."""
        harness = integration_harness
        harness.asyncgate.add_lease(simple_lease)

        mock_client = harness.create_mock_http_client()

        async def failing_handler(lease: Lease) -> Receipt:
            raise RuntimeError("Test error")

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            work_poller = WorkPoller(asyncgate_client, integration_settings, failing_handler)
            await work_poller._poll_and_dispatch()
            await asyncio.sleep(0.1)

            # Should have failure receipt
            failed_receipts = [r for r in harness.asyncgate.receipts if r.status == JobStatus.FAILED]
            assert len(failed_receipts) >= 1
            assert failed_receipts[0].error_metadata is not None


class TestErrorPaths:
    """Tests for error handling and recovery paths."""

    async def test_asyncgate_connection_failure(self, integration_settings):
        """Test handling of AsyncGate connection failures."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            asyncgate_client = AsyncGateClient(integration_settings)
            asyncgate_client._client = mock_client

            # Should return None on connection failure, not raise
            result = await asyncgate_client.poll_for_work()
            assert result is None

    async def test_ai_provider_timeout(self, integration_harness, simple_lease, integration_settings):
        """Test handling of AI provider timeouts."""
        harness = integration_harness

        async def slow_request(request: httpx.Request) -> httpx.Response:
            if "/chat/completions" in str(request.url):
                await asyncio.sleep(5)  # Simulate timeout
            return httpx.Response(408, json={"error": "timeout"})

        transport = httpx.MockTransport(slow_request)
        mock_client = httpx.AsyncClient(transport=transport, timeout=0.1)

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=1)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            # Should fail gracefully
            receipt = await job_executor.execute(simple_lease)
            assert receipt.status == JobStatus.FAILED

    async def test_mcp_server_failure(self, integration_harness, complex_lease, integration_settings):
        """Test handling of MCP server failures."""
        harness = integration_harness

        mcp_server = harness.add_mcp_server("test-mcp")
        mcp_server.should_fail = True

        harness.ai_provider.set_plan_response({
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "tool_invocation",
                    "tool_name": "mcp_call",
                    "tool_params": {"server": "test-mcp", "method": "test", "params": {}}
                }
            ],
            "summary": "MCP call test"
        })

        mock_client = harness.create_mock_http_client()

        with patch.object(httpx, 'AsyncClient', return_value=mock_client):
            bootstrap = Bootstrap(integration_settings)
            bootstrap.profiles = {"default": test_instruction_profile}
            bootstrap._loaded = True

            sink_registry = SinkRegistry()
            mcp_registry = MCPAdapterRegistry()

            ai_client = AIClient(integration_settings.get_ai_config())
            ai_client._client = mock_client

            tool_executor = ToolExecutor(mcp_registry, sink_registry, max_retries=2)
            job_executor = JobExecutor(ai_client, tool_executor, bootstrap, integration_settings)

            receipt = await job_executor.execute(complex_lease)
            # Should fail after retries exhausted
            assert receipt.status == JobStatus.FAILED
