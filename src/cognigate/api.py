"""REST API for CogniGate."""

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import Settings, Bootstrap
from .models import Lease, Receipt, JobStatus
from .leasing import AsyncGateClient, WorkPoller
from .plugins import SinkRegistry, MCPAdapterRegistry
from .plugins.builtin_sinks import register_builtin_sinks
from .ai_client import AIClient
from .tools import ToolExecutor
from .executor import JobExecutor
from .auth import AuthDependency
from .middleware import get_rate_limiter
from .observability import configure_logging, get_logger, JobContext
from .metrics import (
    init_metrics,
    get_metrics,
    get_metrics_content_type,
    ACTIVE_JOBS,
)


logger = get_logger(__name__)


# Rate limiting dependency
async def rate_limit_dependency(request: Request) -> None:
    """Rate limiting dependency."""
    if state.settings:
        limiter = get_rate_limiter(
            calls_per_minute=state.settings.rate_limit_requests_per_minute,
            enabled=state.settings.rate_limit_enabled
        )
        await limiter.check_request(request)


# Global state (initialized at startup)
class AppState:
    settings: Settings | None = None
    bootstrap: Bootstrap | None = None
    sink_registry: SinkRegistry | None = None
    mcp_registry: MCPAdapterRegistry | None = None
    ai_client: AIClient | None = None
    asyncgate_client: AsyncGateClient | None = None
    tool_executor: ToolExecutor | None = None
    job_executor: JobExecutor | None = None
    work_poller: WorkPoller | None = None
    auth_dependency: AuthDependency | None = None


state = AppState()


async def get_auth(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> bool:
    """Run auth dependency for protected endpoints."""
    if not state.auth_dependency:
        raise HTTPException(status_code=503, detail="Auth not initialized")
    return await state.auth_dependency(authorization=authorization, x_api_key=x_api_key)


async def job_handler(lease: Lease) -> Receipt:
    """Handle a leased job."""
    if not state.job_executor:
        raise RuntimeError("Job executor not initialized")
    return await state.job_executor.execute(lease)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    # Configure structured logging
    json_logs = os.environ.get("COGNIGATE_JSON_LOGS", "true").lower() == "true"
    log_level = os.environ.get("COGNIGATE_LOG_LEVEL", "INFO")
    configure_logging(log_level=log_level, json_logs=json_logs)

    logger.info("cognigate_starting", event="startup_initiated")

    # Load settings
    state.settings = Settings()

    # Initialize metrics
    init_metrics(
        version="0.1.0",
        worker_id=state.settings.worker_id,
        instance_id=state.settings.worker_id
    )

    # Initialize auth dependency
    state.auth_dependency = AuthDependency(state.settings)
    
    # Log auth status
    if state.settings.allow_insecure_dev:
        logger.warning("Running in INSECURE DEV MODE - authentication disabled")
    elif state.settings.api_key:
        logger.info("Authentication enabled: API key configured")
    else:
        logger.warning("No COGNIGATE_API_KEY configured - REST endpoints will reject requests")

    # Bootstrap configuration
    state.bootstrap = Bootstrap(state.settings)
    state.bootstrap.load()
    logger.info(f"Loaded {len(state.bootstrap.profiles)} instruction profiles")

    # Initialize sink registry
    state.sink_registry = SinkRegistry()
    mcp_sink = register_builtin_sinks(state.sink_registry)
    state.sink_registry.discover_plugins(state.settings.plugins_dir)
    logger.info(f"Registered sinks: {state.sink_registry.list_sinks()}")

    # Initialize MCP registry
    state.mcp_registry = MCPAdapterRegistry()
    for endpoint in state.bootstrap.mcp_endpoints:
        state.mcp_registry.register(endpoint)
    logger.info(f"Registered MCP adapters: {state.mcp_registry.list_adapters()}")

    # Wire MCP sink to registry
    mcp_sink.set_mcp_registry(state.mcp_registry)

    # Initialize AI client
    state.ai_client = AIClient(state.settings.get_ai_config())

    # Initialize AsyncGate client
    state.asyncgate_client = AsyncGateClient(state.settings)

    # Initialize tool executor
    state.tool_executor = ToolExecutor(
        state.mcp_registry,
        state.sink_registry,
        max_retries=state.settings.max_retries
    )

    # Initialize job executor
    state.job_executor = JobExecutor(
        state.ai_client,
        state.tool_executor,
        state.bootstrap,
        state.settings
    )

    # Initialize work poller
    state.work_poller = WorkPoller(
        state.asyncgate_client,
        state.settings,
        job_handler
    )

    logger.info("cognigate_started", event="startup_complete")

    yield

    # Shutdown
    logger.info("cognigate_shutdown", event="shutdown_initiated")

    if state.work_poller:
        # Graceful shutdown: wait for active jobs to complete (5 min timeout)
        await state.work_poller.stop_gracefully(timeout=300.0)

    if state.asyncgate_client:
        await state.asyncgate_client.close()

    if state.ai_client:
        await state.ai_client.close()

    if state.mcp_registry:
        await state.mcp_registry.close_all()

    logger.info("cognigate_stopped", event="shutdown_complete")


# Create FastAPI app
app = FastAPI(
    title="CogniGate",
    description="General-purpose cognitive execution worker",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware (avoid instantiating Settings at import time)
def _get_cors_list(env_var: str, default: list[str]) -> list[str]:
    value = os.environ.get(env_var)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_cors_bool(env_var: str, default: bool) -> bool:
    value = os.environ.get(env_var)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_list(
        "COGNIGATE_CORS_ALLOWED_ORIGINS",
        ["http://localhost:3000", "http://localhost:8080"],
    ),
    allow_credentials=_get_cors_bool("COGNIGATE_CORS_ALLOW_CREDENTIALS", True),
    allow_methods=_get_cors_list(
        "COGNIGATE_CORS_ALLOWED_METHODS",
        ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    ),
    allow_headers=_get_cors_list(
        "COGNIGATE_CORS_ALLOWED_HEADERS",
        ["Authorization", "Content-Type", "X-Tenant-ID"],
    ),
)


# API Models
class HealthResponse(BaseModel):
    status: str = Field(description="Service status")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")
    instance_id: str = Field(description="Instance identifier")
    worker_id: str = Field(description="Worker identifier")
    active_jobs: int = Field(description="Number of active jobs")


class SubmitJobRequest(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    payload: dict[str, Any] = Field(description="Task payload")
    profile: str = Field(default="default", description="Instruction profile")
    sink_config: dict[str, Any] = Field(default_factory=dict, description="Sink configuration")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Execution constraints")


class SubmitJobResponse(BaseModel):
    lease_id: str = Field(description="Assigned lease ID")
    task_id: str = Field(description="Task ID")
    status: str = Field(description="Job status")


class JobStatusResponse(BaseModel):
    lease_id: str = Field(description="Lease ID")
    task_id: str = Field(description="Task ID")
    status: str = Field(description="Job status")


# Component health check helpers
async def check_asyncgate_health() -> dict:
    """Check AsyncGate connection health."""
    if not state.asyncgate_client:
        return {"healthy": False, "error": "Client not initialized"}

    try:
        # Try a lightweight operation
        import httpx
        response = await state.asyncgate_client._client.get(
            f"{state.asyncgate_client.endpoint}/health",
            timeout=5.0
        )
        return {
            "healthy": response.status_code == 200,
            "status_code": response.status_code,
            "latency_ms": response.elapsed.total_seconds() * 1000 if hasattr(response, 'elapsed') else None
        }
    except httpx.TimeoutException:
        return {"healthy": False, "error": "Timeout"}
    except httpx.ConnectError:
        return {"healthy": False, "error": "Connection failed"}
    except Exception as e:
        return {"healthy": False, "error": str(e)}


async def check_ai_provider_health() -> dict:
    """Check AI provider connection health."""
    if not state.ai_client:
        return {"healthy": False, "error": "Client not initialized"}

    try:
        # Check circuit breaker state
        cb_state = state.ai_client._circuit_breaker.state.value
        if cb_state == "open":
            return {
                "healthy": False,
                "error": "Circuit breaker open",
                "circuit_state": cb_state
            }

        return {
            "healthy": True,
            "circuit_state": cb_state,
            "model": state.ai_client.model
        }
    except Exception as e:
        return {"healthy": False, "error": str(e)}


async def check_mcp_adapters_health() -> dict:
    """Check MCP adapters health."""
    if not state.mcp_registry:
        return {}

    results = {}
    for name in state.mcp_registry.list_adapters():
        adapter = state.mcp_registry.get(name)
        if adapter:
            try:
                cb_state = adapter._circuit_breaker.state.value
                results[name] = {
                    "healthy": cb_state != "open",
                    "circuit_state": cb_state,
                    "read_only": adapter.read_only
                }
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
    return results


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint (for load balancers)."""
    active_jobs = len(state.work_poller._active_jobs) if state.work_poller else 0
    return HealthResponse(
        status="healthy",
        service="CogniGate",
        version="0.1.0",
        instance_id=state.settings.worker_id if state.settings else "cognigate-1",
        worker_id=state.settings.worker_id if state.settings else "unknown",
        active_jobs=active_jobs
    )


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    import asyncio
    import time

    start_time = time.perf_counter()

    # Run health checks in parallel
    asyncgate_check, ai_check, mcp_checks = await asyncio.gather(
        check_asyncgate_health(),
        check_ai_provider_health(),
        check_mcp_adapters_health(),
        return_exceptions=True
    )

    # Handle exceptions
    if isinstance(asyncgate_check, Exception):
        asyncgate_check = {"healthy": False, "error": str(asyncgate_check)}
    if isinstance(ai_check, Exception):
        ai_check = {"healthy": False, "error": str(ai_check)}
    if isinstance(mcp_checks, Exception):
        mcp_checks = {}

    checks = {
        "asyncgate": asyncgate_check,
        "ai_provider": ai_check,
        "mcp_adapters": mcp_checks
    }

    # Determine overall health
    core_healthy = (
        asyncgate_check.get("healthy", False) and
        ai_check.get("healthy", False)
    )

    # Check if any MCP adapter is unhealthy
    mcp_healthy = all(
        adapter.get("healthy", False)
        for adapter in mcp_checks.values()
    ) if mcp_checks else True

    overall_healthy = core_healthy and mcp_healthy

    # Get additional state info
    active_jobs = len(state.work_poller._active_jobs) if state.work_poller else 0
    shutting_down = state.work_poller.is_shutting_down() if state.work_poller else False

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "service": "CogniGate",
        "version": "0.1.0",
        "instance_id": state.settings.worker_id if state.settings else "unknown",
        "checks": checks,
        "state": {
            "active_jobs": active_jobs,
            "shutting_down": shutting_down,
            "polling": state.work_poller._running if state.work_poller else False
        },
        "check_duration_ms": round(elapsed_ms, 2)
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")

    # Check if shutting down
    if state.work_poller and state.work_poller.is_shutting_down():
        raise HTTPException(status_code=503, detail="Shutting down")

    return {"ready": True}


@app.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    # Simple liveness check - if we can respond, we're alive
    return {"alive": True}


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type()
    )


@app.post("/v1/jobs", response_model=SubmitJobResponse, dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def submit_job(request: SubmitJobRequest, background_tasks: BackgroundTasks):
    """Submit a job directly (for testing/local use).

    In production, jobs come from AsyncGate polling.
    """
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")

    import uuid

    lease = Lease(
        lease_id=str(uuid.uuid4()),
        task_id=request.task_id,
        payload=request.payload,
        profile=request.profile,
        sink_config=request.sink_config,
        constraints=request.constraints
    )

    # Execute in background
    async def run_job():
        receipt = await state.job_executor.execute(lease)
        logger.info(f"Job {lease.task_id} completed with status: {receipt.status}")

    background_tasks.add_task(run_job)

    return SubmitJobResponse(
        lease_id=lease.lease_id,
        task_id=lease.task_id,
        status="accepted"
    )


@app.post("/v1/jobs/{lease_id}/cancel", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def cancel_job(lease_id: str):
    """Cancel a running job.

    The job will be cancelled at the next step boundary.
    Returns 404 if the job is not found or already completed.
    """
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")

    # Check if job is active
    if state.work_poller and lease_id not in state.work_poller._active_jobs:
        raise HTTPException(status_code=404, detail="Job not found or already completed")

    # Request cancellation
    state.job_executor.cancel_job(lease_id)

    logger.info("job_cancel_requested", lease_id=lease_id)

    return {
        "status": "cancellation_requested",
        "lease_id": lease_id,
        "message": "Job will be cancelled at the next step boundary"
    }


@app.post("/v1/polling/start", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def start_polling(background_tasks: BackgroundTasks):
    """Start polling AsyncGate for work."""
    if not state.work_poller:
        raise HTTPException(status_code=503, detail="Not ready")

    background_tasks.add_task(state.work_poller.start)
    return {"status": "polling_started"}


@app.post("/v1/polling/stop", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def stop_polling():
    """Stop polling AsyncGate."""
    if state.work_poller:
        await state.work_poller.stop()
    return {"status": "polling_stopped"}


@app.get("/v1/config/profiles", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def list_profiles():
    """List available instruction profiles."""
    if not state.bootstrap:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "profiles": list(state.bootstrap.profiles.keys())
    }


@app.get("/v1/config/sinks", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def list_sinks():
    """List available output sinks."""
    if not state.sink_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "sinks": state.sink_registry.list_sinks()
    }


@app.get("/v1/config/mcp", dependencies=[Depends(get_auth), Depends(rate_limit_dependency)])
async def list_mcp_adapters():
    """List available MCP adapters."""
    if not state.mcp_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "adapters": state.mcp_registry.list_adapters()
    }
