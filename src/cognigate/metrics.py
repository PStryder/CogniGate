"""Prometheus metrics for CogniGate.

Provides metrics collection and exposure for monitoring.
"""

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Create a custom registry to avoid conflicts
REGISTRY = CollectorRegistry()

# Service info
SERVICE_INFO = Info(
    "cognigate",
    "CogniGate service information",
    registry=REGISTRY
)

# Job metrics
JOBS_TOTAL = Counter(
    "cognigate_jobs_total",
    "Total number of jobs processed",
    ["status"],
    registry=REGISTRY
)

JOB_DURATION_SECONDS = Histogram(
    "cognigate_job_duration_seconds",
    "Duration of job execution in seconds",
    ["profile", "status"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=REGISTRY
)

ACTIVE_JOBS = Gauge(
    "cognigate_active_jobs",
    "Number of currently active jobs",
    registry=REGISTRY
)

# Lease metrics
LEASE_CLAIMS_TOTAL = Counter(
    "cognigate_lease_claims_total",
    "Total number of lease claim attempts",
    ["result"],  # success, empty, error
    registry=REGISTRY
)

LEASE_EXTENSIONS_TOTAL = Counter(
    "cognigate_lease_extensions_total",
    "Total number of lease extensions",
    ["result"],  # success, error
    registry=REGISTRY
)

# Tool call metrics
TOOL_CALLS_TOTAL = Counter(
    "cognigate_tool_calls_total",
    "Total number of tool calls",
    ["tool", "result"],  # result: success, error
    registry=REGISTRY
)

TOOL_CALL_DURATION_SECONDS = Histogram(
    "cognigate_tool_call_duration_seconds",
    "Duration of tool calls in seconds",
    ["tool"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# AI provider metrics
AI_REQUESTS_TOTAL = Counter(
    "cognigate_ai_requests_total",
    "Total number of AI API requests",
    ["type", "result"],  # type: chat, plan; result: success, error
    registry=REGISTRY
)

AI_REQUEST_DURATION_SECONDS = Histogram(
    "cognigate_ai_request_duration_seconds",
    "Duration of AI API requests in seconds",
    ["type"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    registry=REGISTRY
)

AI_TOKENS_USED = Counter(
    "cognigate_ai_tokens_used",
    "Total number of AI tokens used",
    ["type"],  # prompt, completion
    registry=REGISTRY
)

# Receipt metrics
RECEIPTS_TOTAL = Counter(
    "cognigate_receipts_total",
    "Total number of receipts sent",
    ["status", "result"],  # status: running, complete, failed; result: success, error
    registry=REGISTRY
)

RECEIPT_RETRIES_TOTAL = Counter(
    "cognigate_receipt_retries_total",
    "Total number of receipt send retries",
    registry=REGISTRY
)

DEAD_LETTER_QUEUE_SIZE = Gauge(
    "cognigate_dead_letter_queue_size",
    "Number of receipts in dead letter queue",
    registry=REGISTRY
)

# MCP adapter metrics
MCP_CALLS_TOTAL = Counter(
    "cognigate_mcp_calls_total",
    "Total number of MCP calls",
    ["server", "method", "result"],
    registry=REGISTRY
)

MCP_CALL_DURATION_SECONDS = Histogram(
    "cognigate_mcp_call_duration_seconds",
    "Duration of MCP calls in seconds",
    ["server"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "cognigate_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["name"],
    registry=REGISTRY
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "cognigate_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["name"],
    registry=REGISTRY
)

# Health check metrics
HEALTH_CHECK_DURATION_SECONDS = Histogram(
    "cognigate_health_check_duration_seconds",
    "Duration of health checks in seconds",
    ["component"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)


def init_metrics(
    version: str = "0.1.0",
    worker_id: str = "unknown",
    instance_id: str = "unknown"
) -> None:
    """Initialize service metrics.

    Args:
        version: Service version
        worker_id: Worker identifier
        instance_id: Instance identifier
    """
    SERVICE_INFO.info({
        "version": version,
        "worker_id": worker_id,
        "instance_id": instance_id
    })


def get_metrics() -> bytes:
    """Get current metrics in Prometheus format.

    Returns:
        Metrics data as bytes
    """
    return generate_latest(REGISTRY)


def get_metrics_content_type() -> str:
    """Get the content type for metrics response.

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST


@contextmanager
def track_job_duration(
    profile: str,
    status_holder: list[str] | None = None
) -> Generator[None, None, None]:
    """Context manager to track job duration.

    Args:
        profile: The instruction profile name
        status_holder: Optional list to receive the status [status] for labeling

    Yields:
        None
    """
    ACTIVE_JOBS.inc()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        status = status_holder[0] if status_holder else "unknown"
        JOB_DURATION_SECONDS.labels(profile=profile, status=status).observe(duration)
        ACTIVE_JOBS.dec()


@contextmanager
def track_tool_call(tool_name: str) -> Generator[None, None, None]:
    """Context manager to track tool call duration.

    Args:
        tool_name: The tool being called

    Yields:
        None
    """
    start_time = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        TOOL_CALL_DURATION_SECONDS.labels(tool=tool_name).observe(duration)
        TOOL_CALLS_TOTAL.labels(
            tool=tool_name,
            result="success" if success else "error"
        ).inc()


@contextmanager
def track_ai_request(request_type: str) -> Generator[None, None, None]:
    """Context manager to track AI request duration.

    Args:
        request_type: The type of request (chat, plan)

    Yields:
        None
    """
    start_time = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        AI_REQUEST_DURATION_SECONDS.labels(type=request_type).observe(duration)
        AI_REQUESTS_TOTAL.labels(
            type=request_type,
            result="success" if success else "error"
        ).inc()


@contextmanager
def track_mcp_call(server: str, method: str) -> Generator[None, None, None]:
    """Context manager to track MCP call duration.

    Args:
        server: The MCP server name
        method: The MCP method being called

    Yields:
        None
    """
    start_time = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        duration = time.perf_counter() - start_time
        MCP_CALL_DURATION_SECONDS.labels(server=server).observe(duration)
        MCP_CALLS_TOTAL.labels(
            server=server,
            method=method,
            result="success" if success else "error"
        ).inc()


def record_job_complete(status: str) -> None:
    """Record a completed job.

    Args:
        status: The job status (complete, failed)
    """
    JOBS_TOTAL.labels(status=status).inc()


def record_lease_claim(result: str) -> None:
    """Record a lease claim attempt.

    Args:
        result: The claim result (success, empty, error)
    """
    LEASE_CLAIMS_TOTAL.labels(result=result).inc()


def record_lease_extension(success: bool) -> None:
    """Record a lease extension attempt.

    Args:
        success: Whether the extension succeeded
    """
    LEASE_EXTENSIONS_TOTAL.labels(
        result="success" if success else "error"
    ).inc()


def record_receipt_sent(status: str, success: bool) -> None:
    """Record a receipt send attempt.

    Args:
        status: The receipt status (running, complete, failed)
        success: Whether the send succeeded
    """
    RECEIPTS_TOTAL.labels(
        status=status,
        result="success" if success else "error"
    ).inc()


def record_receipt_retry() -> None:
    """Record a receipt retry attempt."""
    RECEIPT_RETRIES_TOTAL.inc()


def set_dead_letter_queue_size(size: int) -> None:
    """Set the dead letter queue size.

    Args:
        size: Current queue size
    """
    DEAD_LETTER_QUEUE_SIZE.set(size)


def record_ai_tokens(prompt_tokens: int, completion_tokens: int) -> None:
    """Record AI token usage.

    Args:
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
    """
    AI_TOKENS_USED.labels(type="prompt").inc(prompt_tokens)
    AI_TOKENS_USED.labels(type="completion").inc(completion_tokens)


def set_circuit_breaker_state(name: str, state: str) -> None:
    """Set circuit breaker state.

    Args:
        name: Circuit breaker name
        state: State (closed, open, half-open)
    """
    state_map = {"closed": 0, "open": 1, "half-open": 2}
    CIRCUIT_BREAKER_STATE.labels(name=name).set(state_map.get(state, 0))


def record_circuit_breaker_failure(name: str) -> None:
    """Record a circuit breaker failure.

    Args:
        name: Circuit breaker name
    """
    CIRCUIT_BREAKER_FAILURES.labels(name=name).inc()
