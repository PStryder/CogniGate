"""Work intake and leasing system for CogniGate."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Awaitable, Any

import httpx

from .config import Settings
from .models import Lease, Receipt, JobStatus
from .observability import get_logger
from .metrics import (
    record_lease_claim,
    record_receipt_sent,
    record_receipt_retry,
    record_lease_extension,
    set_dead_letter_queue_size,
    ACTIVE_JOBS,
)


logger = get_logger(__name__)


class DeadLetterQueue:
    """Persistent dead letter queue for failed receipts.

    Stores failed receipts to disk for later recovery.
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path("/var/lib/cognigate/dlq")
        self._queue: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def store(self, receipt: Receipt) -> None:
        """Store a failed receipt in the dead letter queue.

        Args:
            receipt: The receipt that failed to send
        """
        async with self._lock:
            entry = {
                "receipt": receipt.model_dump(mode="json"),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "attempts": 0
            }
            self._queue.append(entry)
            set_dead_letter_queue_size(len(self._queue))

            logger.warning(
                "receipt_dead_lettered",
                lease_id=receipt.lease_id,
                task_id=receipt.task_id,
                status=receipt.status.value,
                queue_size=len(self._queue)
            )

            # Persist to disk if path is configured
            await self._persist()

    async def _persist(self) -> None:
        """Persist the queue to disk."""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            dlq_file = self.storage_path / "receipts.json"
            with open(dlq_file, "w") as f:
                json.dump(self._queue, f, indent=2, default=str)
        except Exception as e:
            logger.error("dlq_persist_error", error=str(e))

    async def load(self) -> None:
        """Load the queue from disk."""
        try:
            dlq_file = self.storage_path / "receipts.json"
            if dlq_file.exists():
                with open(dlq_file, "r") as f:
                    self._queue = json.load(f)
                set_dead_letter_queue_size(len(self._queue))
                logger.info("dlq_loaded", count=len(self._queue))
        except Exception as e:
            logger.error("dlq_load_error", error=str(e))

    def size(self) -> int:
        """Get the current queue size."""
        return len(self._queue)

    async def get_all(self) -> list[dict[str, Any]]:
        """Get all entries in the queue."""
        async with self._lock:
            return list(self._queue)

    async def remove(self, receipt_lease_id: str) -> bool:
        """Remove an entry from the queue.

        Args:
            receipt_lease_id: The lease ID of the receipt to remove

        Returns:
            True if the entry was removed, False if not found
        """
        async with self._lock:
            original_size = len(self._queue)
            self._queue = [
                e for e in self._queue
                if e.get("receipt", {}).get("lease_id") != receipt_lease_id
            ]
            removed = len(self._queue) < original_size
            if removed:
                set_dead_letter_queue_size(len(self._queue))
                await self._persist()
            return removed


class AsyncGateClient:
    """Client for communicating with AsyncGate."""

    def __init__(
        self,
        settings: Settings,
        dead_letter_queue: DeadLetterQueue | None = None,
        max_receipt_retries: int = 5,
        backoff_base: float = 2.0
    ):
        self.endpoint = settings.asyncgate_endpoint
        self.auth_token = settings.asyncgate_auth_token
        self.worker_id = settings.worker_id
        self._client = httpx.AsyncClient(timeout=30.0)
        self._dead_letter_queue = dead_letter_queue or DeadLetterQueue()
        self._max_receipt_retries = max_receipt_retries
        self._backoff_base = backoff_base

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
        """Send a receipt to AsyncGate with retry logic.

        Implements exponential backoff and dead letter queue for failed receipts.
        """
        return await self.send_receipt_with_retry(receipt)

    async def send_receipt_with_retry(
        self,
        receipt: Receipt,
        max_retries: int | None = None,
        backoff_base: float | None = None
    ) -> bool:
        """Send a receipt with retry and dead letter queue fallback.

        Args:
            receipt: The receipt to send
            max_retries: Maximum retry attempts (defaults to instance setting)
            backoff_base: Backoff base for exponential delay (defaults to instance setting)

        Returns:
            True if receipt was sent successfully, False if it was dead-lettered
        """
        max_retries = max_retries or self._max_receipt_retries
        backoff_base = backoff_base or self._backoff_base

        for attempt in range(max_retries):
            try:
                success = await self._send_receipt_once(receipt)
                if success:
                    return True

                # Request succeeded but returned non-success (e.g., 4xx)
                logger.warning(
                    "receipt_send_failed",
                    lease_id=receipt.lease_id,
                    attempt=attempt + 1,
                    max_retries=max_retries
                )

            except httpx.HTTPStatusError as e:
                # Server returned error status
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    # Client error - don't retry (except 429)
                    if e.response.status_code != 429:
                        logger.error(
                            "receipt_client_error",
                            lease_id=receipt.lease_id,
                            status_code=e.response.status_code
                        )
                        break

                logger.warning(
                    "receipt_http_error",
                    lease_id=receipt.lease_id,
                    status_code=e.response.status_code,
                    attempt=attempt + 1
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(
                    "receipt_connection_error",
                    lease_id=receipt.lease_id,
                    error_type=type(e).__name__,
                    attempt=attempt + 1
                )

            except Exception as e:
                logger.error(
                    "receipt_unexpected_error",
                    lease_id=receipt.lease_id,
                    error=str(e),
                    attempt=attempt + 1
                )

            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                delay = backoff_base ** attempt
                record_receipt_retry()
                logger.debug(
                    "receipt_retry_scheduled",
                    lease_id=receipt.lease_id,
                    delay=delay,
                    next_attempt=attempt + 2
                )
                await asyncio.sleep(delay)

        # All retries failed - dead letter the receipt
        logger.error(
            "receipt_send_exhausted",
            lease_id=receipt.lease_id,
            task_id=receipt.task_id,
            status=receipt.status.value,
            max_retries=max_retries
        )
        await self._dead_letter_queue.store(receipt)
        return False

    async def _send_receipt_once(self, receipt: Receipt) -> bool:
        """Attempt to send a receipt once without retry.

        Args:
            receipt: The receipt to send

        Returns:
            True if successful, False otherwise

        Raises:
            httpx.HTTPStatusError: On HTTP errors
            httpx.ConnectError: On connection errors
            httpx.TimeoutException: On timeout
        """
        if receipt.status == JobStatus.RUNNING:
            return await self._report_progress(receipt)
        if receipt.status == JobStatus.COMPLETE:
            return await self._complete_task(receipt)
        if receipt.status == JobStatus.FAILED:
            return await self._fail_task(receipt)

        logger.warning(
            "receipt_unsupported_status",
            status=receipt.status.value,
            lease_id=receipt.lease_id
        )
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

    async def extend_lease(self, lease_id: str) -> bool:
        """Extend a lease's timeout.

        Args:
            lease_id: The lease ID to extend

        Returns:
            True if extension was successful, False otherwise
        """
        try:
            response = await self._client.post(
                f"{self.endpoint}/v1/leases/{lease_id}/extend",
                headers=self._headers()
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error extending lease {lease_id}: {e}")
            return False


# Type for job handler callback
JobHandler = Callable[[Lease], Awaitable[Receipt]]


class WorkPoller:
    """Polls AsyncGate for work and dispatches to handler."""

    def __init__(
        self,
        client: AsyncGateClient,
        settings: Settings,
        handler: JobHandler,
        heartbeat_interval: float = 60.0
    ):
        self.client = client
        self.settings = settings
        self.handler = handler
        self.polling_interval = settings.polling_interval
        self.max_concurrent = settings.max_concurrent_jobs
        self.heartbeat_interval = heartbeat_interval
        self._running = False
        self._shutting_down = False
        self._active_jobs: set[str] = set()
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}
        self._job_leases: dict[str, Lease] = {}
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the polling loop."""
        self._running = True
        self._shutting_down = False
        logger.info(
            "work_poller_started",
            polling_interval=self.polling_interval,
            max_concurrent=self.max_concurrent
        )

        while self._running:
            try:
                # Check if we can accept more work
                if len(self._active_jobs) < self.max_concurrent:
                    await self._poll_and_dispatch()
            except Exception as e:
                logger.error("polling_loop_error", error=str(e))

            await asyncio.sleep(self.polling_interval)

    async def stop(self) -> None:
        """Stop the polling loop immediately."""
        logger.info("work_poller_stopping")
        self._running = False

    async def stop_gracefully(self, timeout: float = 300.0) -> None:
        """Stop the polling loop and wait for active jobs to complete.

        Args:
            timeout: Maximum time to wait for jobs to complete (default 5 minutes)
        """
        logger.info(
            "graceful_shutdown_initiated",
            active_jobs=len(self._active_jobs),
            timeout=timeout
        )

        # Stop accepting new work
        self._running = False
        self._shutting_down = True

        if not self._active_jobs:
            logger.info("graceful_shutdown_complete", reason="no_active_jobs")
            return

        # Wait for active jobs to complete with timeout
        try:
            await asyncio.wait_for(
                self._wait_for_jobs_completion(),
                timeout=timeout
            )
            logger.info("graceful_shutdown_complete", reason="jobs_completed")
        except asyncio.TimeoutError:
            logger.warning(
                "graceful_shutdown_timeout",
                remaining_jobs=len(self._active_jobs),
                timed_out_leases=list(self._active_jobs)
            )
            # Send failure receipts for timed-out jobs
            await self._send_timeout_receipts()

    async def _wait_for_jobs_completion(self) -> None:
        """Wait for all active jobs to complete."""
        while self._active_jobs:
            logger.debug(
                "waiting_for_jobs",
                active_count=len(self._active_jobs),
                active_leases=list(self._active_jobs)
            )
            await asyncio.sleep(0.5)

    async def _send_timeout_receipts(self) -> None:
        """Send failure receipts for jobs that timed out during shutdown."""
        for lease_id in list(self._active_jobs):
            # Cancel the task if still running
            if lease_id in self._job_tasks:
                task = self._job_tasks[lease_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # We don't have the full lease info here, so we can't send a proper receipt
            # The job handler should handle this case
            logger.error(
                "job_timeout_during_shutdown",
                lease_id=lease_id
            )

        self._active_jobs.clear()
        self._job_tasks.clear()

    def get_active_job_count(self) -> int:
        """Get the number of currently active jobs."""
        return len(self._active_jobs)

    def is_shutting_down(self) -> bool:
        """Check if the poller is in shutdown mode."""
        return self._shutting_down

    async def _start_heartbeat(self, lease: Lease) -> None:
        """Start a heartbeat task for a lease.

        The heartbeat periodically extends the lease to prevent timeout
        for long-running jobs.

        Args:
            lease: The lease to send heartbeats for
        """
        task = asyncio.create_task(self._heartbeat_loop(lease))
        self._heartbeat_tasks[lease.lease_id] = task

    async def _stop_heartbeat(self, lease_id: str) -> None:
        """Stop the heartbeat task for a lease.

        Args:
            lease_id: The lease ID to stop heartbeat for
        """
        if lease_id in self._heartbeat_tasks:
            task = self._heartbeat_tasks.pop(lease_id)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self, lease: Lease) -> None:
        """Heartbeat loop that extends the lease periodically.

        Args:
            lease: The lease to extend
        """
        lease_id = lease.lease_id
        try:
            while lease_id in self._active_jobs and not self._shutting_down:
                await asyncio.sleep(self.heartbeat_interval)

                # Check if still active after sleep
                if lease_id not in self._active_jobs:
                    break

                # Extend the lease
                success = await self.client.extend_lease(lease_id)
                record_lease_extension(success)

                if success:
                    logger.debug(
                        "lease_extended",
                        lease_id=lease_id,
                        task_id=lease.task_id
                    )
                else:
                    logger.warning(
                        "lease_extension_failed",
                        lease_id=lease_id,
                        task_id=lease.task_id
                    )

        except asyncio.CancelledError:
            logger.debug(
                "heartbeat_cancelled",
                lease_id=lease_id
            )
        except Exception as e:
            logger.error(
                "heartbeat_error",
                lease_id=lease_id,
                error=str(e)
            )

    async def _poll_and_dispatch(self) -> None:
        """Poll for work and dispatch if available."""
        if self._shutting_down:
            return

        lease = await self.client.poll_for_work()

        if not lease:
            record_lease_claim("empty")
            return

        if lease.lease_id in self._active_jobs:
            logger.warning(
                "duplicate_lease_skipped",
                lease_id=lease.lease_id
            )
            return

        record_lease_claim("success")
        logger.info(
            "lease_received",
            lease_id=lease.lease_id,
            task_id=lease.task_id,
            profile=lease.profile
        )
        self._active_jobs.add(lease.lease_id)
        self._job_leases[lease.lease_id] = lease
        ACTIVE_JOBS.inc()

        # Start heartbeat for lease extension
        await self._start_heartbeat(lease)

        # Dispatch job in background and track the task
        task = asyncio.create_task(self._handle_job(lease))
        self._job_tasks[lease.lease_id] = task

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
                success = await self.client.send_receipt(acceptance_receipt)
                record_receipt_sent("running", success)

                # Execute the job
                final_receipt = await self.handler(lease)

                # Send completion receipt
                success = await self.client.send_receipt(final_receipt)
                record_receipt_sent(final_receipt.status.value, success)

            except asyncio.CancelledError:
                logger.warning(
                    "job_cancelled",
                    lease_id=lease.lease_id,
                    reason="shutdown"
                )
                # Send failure receipt on cancellation
                error_receipt = Receipt(
                    lease_id=lease.lease_id,
                    task_id=lease.task_id,
                    worker_id=self.settings.worker_id,
                    status=JobStatus.FAILED,
                    timestamp=datetime.now(timezone.utc),
                    error_metadata={"code": "SHUTDOWN_CANCELLED", "message": "Job cancelled during shutdown"}
                )
                await self.client.send_receipt(error_receipt)
                raise

            except Exception as e:
                logger.error(
                    "job_handler_error",
                    lease_id=lease.lease_id,
                    error=str(e)
                )
                error_receipt = Receipt(
                    lease_id=lease.lease_id,
                    task_id=lease.task_id,
                    worker_id=self.settings.worker_id,
                    status=JobStatus.FAILED,
                    timestamp=datetime.now(timezone.utc),
                    error_metadata={"code": "HANDLER_ERROR", "message": str(e)}
                )
                success = await self.client.send_receipt(error_receipt)
                record_receipt_sent("failed", success)

            finally:
                # Stop heartbeat first
                await self._stop_heartbeat(lease.lease_id)

                # Clean up tracking
                self._active_jobs.discard(lease.lease_id)
                self._job_tasks.pop(lease.lease_id, None)
                self._job_leases.pop(lease.lease_id, None)
                ACTIVE_JOBS.dec()
