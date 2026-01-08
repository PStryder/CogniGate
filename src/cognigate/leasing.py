"""Work intake and leasing system for CogniGate."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, Any

import httpx

from .config import Settings
from .models import Lease, Receipt, JobStatus


logger = logging.getLogger(__name__)


class AsyncGateClient:
    """Client for communicating with AsyncGate."""

    def __init__(self, settings: Settings):
        self.endpoint = settings.asyncgate_endpoint
        self.auth_token = settings.asyncgate_auth_token
        self.worker_id = settings.worker_id
        self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        """Get request headers with auth."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def poll_for_work(self) -> Lease | None:
        """Poll AsyncGate for available work.

        Returns:
            A Lease if work is available, None otherwise.
        """
        try:
            response = await self._client.post(
                f"{self.endpoint}/v1/leases/claim",
                json={
                    "worker_id": self.worker_id,
                    "max_tasks": 1,
                },
                headers=self._headers()
            )
            response.raise_for_status()
            data = response.json()

            tasks = data.get("tasks", [])
            if not tasks:
                return None

            task = tasks[0]
            payload = task.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}

            constraints = payload.get("constraints", {})
            if not constraints and isinstance(task.get("requirements"), dict):
                constraints = task["requirements"]

            return Lease(
                lease_id=str(task["lease_id"]),
                task_id=str(task["task_id"]),
                payload=payload,
                profile=payload.get("profile", "default"),
                sink_config=payload.get("sink_config", {}),
                constraints=constraints,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error polling for work: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error polling for work: {e}")
            return None

    async def send_receipt(self, receipt: Receipt) -> bool:
        """Send a receipt to AsyncGate using the task state endpoints."""
        try:
            if receipt.status == JobStatus.RUNNING:
                return await self._report_progress(receipt)
            if receipt.status == JobStatus.COMPLETE:
                return await self._complete_task(receipt)
            if receipt.status == JobStatus.FAILED:
                return await self._fail_task(receipt)

            logger.warning(f"Skipping receipt with unsupported status: {receipt.status}")
            return False

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error sending receipt: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Error sending receipt: {e}")
            return False

    async def _report_progress(self, receipt: Receipt) -> bool:
        payload: dict[str, Any] = {
            "worker_kind": "worker",
            "worker_id": self.worker_id,
            "lease_id": receipt.lease_id,
            "progress": {
                "status": receipt.status.value,
                "summary": receipt.summary,
                "artifact_pointers": receipt.artifact_pointers,
                "timestamp": receipt.timestamp.isoformat(),
            },
        }
        response = await self._client.post(
            f"{self.endpoint}/v1/tasks/{receipt.task_id}/progress",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        logger.info(f"Progress reported: lease={receipt.lease_id}, status={receipt.status}")
        return True

    async def _complete_task(self, receipt: Receipt) -> bool:
        payload: dict[str, Any] = {
            "worker_kind": "worker",
            "worker_id": self.worker_id,
            "lease_id": receipt.lease_id,
            "result": {
                "summary": receipt.summary,
                "artifact_pointers": receipt.artifact_pointers,
            },
        }
        if receipt.artifact_pointers:
            payload["artifacts"] = {"pointers": receipt.artifact_pointers}
        response = await self._client.post(
            f"{self.endpoint}/v1/tasks/{receipt.task_id}/complete",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        logger.info(f"Task completed: lease={receipt.lease_id}, status={receipt.status}")
        return True

    async def _fail_task(self, receipt: Receipt) -> bool:
        error_metadata = receipt.error_metadata or {}
        payload: dict[str, Any] = {
            "worker_kind": "worker",
            "worker_id": self.worker_id,
            "lease_id": receipt.lease_id,
            "error": {
                "code": error_metadata.get("code", "JOB_FAILED"),
                "message": error_metadata.get("message", "Job failed"),
            },
            "retryable": False,
        }
        response = await self._client.post(
            f"{self.endpoint}/v1/tasks/{receipt.task_id}/fail",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        logger.info(f"Task failed: lease={receipt.lease_id}, status={receipt.status}")
        return True


# Type for job handler callback
JobHandler = Callable[[Lease], Awaitable[Receipt]]


class WorkPoller:
    """Polls AsyncGate for work and dispatches to handler."""

    def __init__(
        self,
        client: AsyncGateClient,
        settings: Settings,
        handler: JobHandler
    ):
        self.client = client
        self.settings = settings
        self.handler = handler
        self.polling_interval = settings.polling_interval
        self.max_concurrent = settings.max_concurrent_jobs
        self._running = False
        self._active_jobs: set[str] = set()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def start(self) -> None:
        """Start the polling loop."""
        self._running = True
        logger.info(f"Starting work poller (interval={self.polling_interval}s, max_concurrent={self.max_concurrent})")

        while self._running:
            try:
                # Check if we can accept more work
                if len(self._active_jobs) < self.max_concurrent:
                    await self._poll_and_dispatch()
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")

            await asyncio.sleep(self.polling_interval)

    async def stop(self) -> None:
        """Stop the polling loop."""
        logger.info("Stopping work poller")
        self._running = False

    async def _poll_and_dispatch(self) -> None:
        """Poll for work and dispatch if available."""
        lease = await self.client.poll_for_work()

        if not lease:
            return

        if lease.lease_id in self._active_jobs:
            logger.warning(f"Lease {lease.lease_id} already active, skipping")
            return

        logger.info(f"Received lease: {lease.lease_id} for task {lease.task_id}")
        self._active_jobs.add(lease.lease_id)

        # Dispatch job in background
        asyncio.create_task(self._handle_job(lease))

    async def _handle_job(self, lease: Lease) -> None:
        """Handle a single job."""
        async with self._semaphore:
            try:
                # Send acceptance receipt
                acceptance_receipt = Receipt(
                    lease_id=lease.lease_id,
                    task_id=lease.task_id,
                    worker_id=self.settings.worker_id,
                    status=JobStatus.RUNNING,
                    timestamp=datetime.now(timezone.utc)
                )
                await self.client.send_receipt(acceptance_receipt)

                # Execute the job
                final_receipt = await self.handler(lease)

                # Send completion receipt
                await self.client.send_receipt(final_receipt)

            except Exception as e:
                logger.error(f"Job {lease.lease_id} failed with error: {e}")
                error_receipt = Receipt(
                    lease_id=lease.lease_id,
                    task_id=lease.task_id,
                    worker_id=self.settings.worker_id,
                    status=JobStatus.FAILED,
                    timestamp=datetime.now(timezone.utc),
                    error_metadata={"code": "HANDLER_ERROR", "message": str(e)}
                )
                await self.client.send_receipt(error_receipt)

            finally:
                self._active_jobs.discard(lease.lease_id)
