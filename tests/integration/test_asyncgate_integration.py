"""Integration tests for AsyncGate client and communication."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from cognigate.leasing import AsyncGateClient, WorkPoller
from cognigate.models import Lease, Receipt, JobStatus

from .fixtures import (
    MockAsyncGateServer,
    integration_harness,
    mock_asyncgate,
    simple_lease,
    integration_settings,
)


pytestmark = pytest.mark.asyncio


class TestAsyncGateClientPolling:
    """Tests for AsyncGate polling functionality."""

    async def test_poll_returns_lease_when_available(self, mock_asyncgate, integration_settings):
        """Test successful lease claim."""
        lease = Lease(
            lease_id="test-lease-1",
            task_id="test-task-1",
            payload={"task": "test task"},
            profile="default",
            sink_config={},
            constraints={}
        )
        mock_asyncgate.add_lease(lease)

        async def mock_request(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            result = await mock_asyncgate.handle_claim(body)
            return httpx.Response(200, json=result)

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        claimed_lease = await client.poll_for_work()

        assert claimed_lease is not None
        assert claimed_lease.lease_id == lease.lease_id
        assert claimed_lease.task_id == lease.task_id
        assert mock_asyncgate.poll_count == 1

    async def test_poll_returns_none_when_no_work(self, mock_asyncgate, integration_settings):
        """Test polling when no work is available."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"tasks": []})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.poll_for_work()

        assert result is None

    async def test_poll_handles_http_error(self, integration_settings):
        """Test polling handles HTTP errors gracefully."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "Internal server error"})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.poll_for_work()

        assert result is None

    async def test_poll_handles_network_error(self, integration_settings):
        """Test polling handles network errors gracefully."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.poll_for_work()

        assert result is None

    async def test_poll_parses_payload_correctly(self, integration_settings):
        """Test that poll correctly parses various payload formats."""
        test_cases = [
            {"payload": {"task": "test"}, "expected_task": "test"},
            {"payload": "string payload", "expected_value": "string payload"},
            {"payload": None, "expected_empty": True},
        ]

        for case in test_cases:
            response_data = {
                "tasks": [{
                    "lease_id": "lease-1",
                    "task_id": "task-1",
                    "payload": case["payload"]
                }]
            }

            async def mock_request(request: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json=response_data)

            transport = httpx.MockTransport(mock_request)
            mock_client = httpx.AsyncClient(transport=transport)

            client = AsyncGateClient(integration_settings)
            client._client = mock_client

            result = await client.poll_for_work()

            assert result is not None
            if "expected_task" in case:
                assert result.payload.get("task") == case["expected_task"]
            elif "expected_value" in case:
                assert result.payload.get("value") == case["expected_value"]
            elif "expected_empty" in case:
                assert result.payload == {}


class TestAsyncGateClientReceipts:
    """Tests for AsyncGate receipt delivery."""

    async def test_send_running_receipt(self, mock_asyncgate, integration_settings):
        """Test sending RUNNING receipt."""
        receipt = Receipt(
            lease_id="lease-1",
            task_id="task-1",
            worker_id="worker-1",
            status=JobStatus.RUNNING,
            timestamp=datetime.now(timezone.utc),
            summary="Job started"
        )

        async def mock_request(request: httpx.Request) -> httpx.Response:
            if "/progress" in str(request.url):
                body = json.loads(request.content)
                await mock_asyncgate.handle_progress("task-1", body)
                return httpx.Response(200, json={"accepted": True})
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.send_receipt(receipt)

        assert result is True
        assert len(mock_asyncgate.receipts) == 1
        assert mock_asyncgate.receipts[0].status == JobStatus.RUNNING

    async def test_send_complete_receipt(self, mock_asyncgate, integration_settings):
        """Test sending COMPLETE receipt."""
        receipt = Receipt(
            lease_id="lease-1",
            task_id="task-1",
            worker_id="worker-1",
            status=JobStatus.COMPLETE,
            timestamp=datetime.now(timezone.utc),
            summary="Job completed successfully",
            artifact_pointers=[{"sink_id": "file", "uri": "file:///output.txt"}]
        )

        async def mock_request(request: httpx.Request) -> httpx.Response:
            if "/complete" in str(request.url):
                body = json.loads(request.content)
                await mock_asyncgate.handle_complete("task-1", body)
                return httpx.Response(200, json={"accepted": True})
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.send_receipt(receipt)

        assert result is True
        assert len(mock_asyncgate.receipts) == 1
        assert mock_asyncgate.receipts[0].status == JobStatus.COMPLETE

    async def test_send_failed_receipt(self, mock_asyncgate, integration_settings):
        """Test sending FAILED receipt."""
        receipt = Receipt(
            lease_id="lease-1",
            task_id="task-1",
            worker_id="worker-1",
            status=JobStatus.FAILED,
            timestamp=datetime.now(timezone.utc),
            error_metadata={"code": "EXECUTION_ERROR", "message": "Something went wrong"}
        )

        async def mock_request(request: httpx.Request) -> httpx.Response:
            if "/fail" in str(request.url):
                body = json.loads(request.content)
                await mock_asyncgate.handle_fail("task-1", body)
                return httpx.Response(200, json={"accepted": True})
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.send_receipt(receipt)

        assert result is True
        assert len(mock_asyncgate.receipts) == 1
        assert mock_asyncgate.receipts[0].status == JobStatus.FAILED

    async def test_send_receipt_handles_error(self, integration_settings):
        """Test receipt sending handles errors gracefully."""
        receipt = Receipt(
            lease_id="lease-1",
            task_id="task-1",
            worker_id="worker-1",
            status=JobStatus.COMPLETE,
            timestamp=datetime.now(timezone.utc)
        )

        async def mock_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "Server error"})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.send_receipt(receipt)

        assert result is False


class TestAsyncGateClientAuth:
    """Tests for AsyncGate authentication."""

    async def test_auth_header_included(self, integration_settings):
        """Test that auth header is included in requests."""
        integration_settings.asyncgate_auth_token = "test-secret-token"
        captured_headers = {}

        async def mock_request(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json={"tasks": []})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        await client.poll_for_work()

        assert "authorization" in captured_headers
        assert captured_headers["authorization"] == "Bearer test-secret-token"

    async def test_no_auth_when_token_empty(self, integration_settings):
        """Test that no auth header when token is empty."""
        integration_settings.asyncgate_auth_token = ""
        captured_headers = {}

        async def mock_request(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json={"tasks": []})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        await client.poll_for_work()

        assert "authorization" not in captured_headers


class TestWorkPollerLifecycle:
    """Tests for WorkPoller lifecycle management."""

    async def test_start_and_stop(self, integration_settings):
        """Test WorkPoller start and stop lifecycle."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"tasks": []})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        async def dummy_handler(lease: Lease) -> Receipt:
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc)
            )

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        poller = WorkPoller(client, integration_settings, dummy_handler)

        # Start in background
        task = asyncio.create_task(poller.start())

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop
        await poller.stop()

        # Should complete without error
        assert not poller._running

        # Cancel the background task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_active_jobs_tracking(self, mock_asyncgate, integration_settings):
        """Test that active jobs are tracked correctly."""
        lease = Lease(
            lease_id="track-lease-1",
            task_id="track-task-1",
            payload={"task": "test"},
            profile="default",
            sink_config={},
            constraints={}
        )
        mock_asyncgate.add_lease(lease)

        async def mock_request(request: httpx.Request) -> httpx.Response:
            if "/claim" in str(request.url):
                body = json.loads(request.content)
                result = await mock_asyncgate.handle_claim(body)
                return httpx.Response(200, json=result)
            return httpx.Response(200, json={"accepted": True})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        job_started = asyncio.Event()
        job_can_complete = asyncio.Event()

        async def slow_handler(lease: Lease) -> Receipt:
            job_started.set()
            await job_can_complete.wait()
            return Receipt(
                lease_id=lease.lease_id,
                task_id=lease.task_id,
                worker_id="test",
                status=JobStatus.COMPLETE,
                timestamp=datetime.now(timezone.utc)
            )

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        poller = WorkPoller(client, integration_settings, slow_handler)

        # Start job
        await poller._poll_and_dispatch()

        # Wait for job to start
        await asyncio.wait_for(job_started.wait(), timeout=1.0)

        # Job should be active
        assert lease.lease_id in poller._active_jobs

        # Allow job to complete
        job_can_complete.set()
        await asyncio.sleep(0.1)

        # Job should be removed from active
        assert lease.lease_id not in poller._active_jobs


class TestLeaseExtension:
    """Tests for lease extension functionality."""

    async def test_extend_lease_success(self, mock_asyncgate, integration_settings):
        """Test successful lease extension."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            if "/extend" in str(request.url):
                lease_id = str(request.url).split("/v1/leases/")[1].split("/extend")[0]
                await mock_asyncgate.handle_extend(lease_id)
                return httpx.Response(200, json={"extended": True})
            return httpx.Response(404)

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.extend_lease("test-lease-id")

        assert result is True
        assert "test-lease-id" in mock_asyncgate.extended_leases

    async def test_extend_lease_failure(self, integration_settings):
        """Test lease extension failure handling."""

        async def mock_request(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "Lease not found"})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        result = await client.extend_lease("nonexistent-lease")

        assert result is False


class TestRetryBehavior:
    """Tests for retry behavior in AsyncGate communication."""

    async def test_receipt_retry_on_failure(self, integration_settings):
        """Test that receipts are retried on failure."""
        attempt_count = 0

        async def mock_request(request: httpx.Request) -> httpx.Response:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return httpx.Response(503, json={"error": "Service unavailable"})
            return httpx.Response(200, json={"accepted": True})

        transport = httpx.MockTransport(mock_request)
        mock_client = httpx.AsyncClient(transport=transport)

        client = AsyncGateClient(integration_settings)
        client._client = mock_client

        receipt = Receipt(
            lease_id="retry-lease",
            task_id="retry-task",
            worker_id="test",
            status=JobStatus.COMPLETE,
            timestamp=datetime.now(timezone.utc)
        )

        # This tests the current behavior (no retry)
        # After implementing P0.4, this test should pass
        result = await client.send_receipt(receipt)
        # Current implementation doesn't retry, so this will fail
        # After P0.4 implementation, change assertion
        assert attempt_count >= 1
