"""MCP API for CogniGate."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings, Bootstrap
from .models import Lease, Receipt
from .leasing import AsyncGateClient, WorkPoller
from .plugins import SinkRegistry, MCPAdapterRegistry
from .plugins.builtin_sinks import register_builtin_sinks
from .ai_client import AIClient
from .tools import ToolExecutor
from .executor import JobExecutor
from .auth import AuthDependency
from .receipts import ReceiptStore
from .middleware import get_rate_limiter
from .observability import configure_logging, get_logger
from .metrics import init_metrics, get_metrics_snapshot, ACTIVE_JOBS


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
    receipt_store: ReceiptStore | None = None


state = AppState()


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
        logger.warning("No COGNIGATE_API_KEY configured - MCP requests will reject requests")

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

    # Initialize AsyncGate client and work poller (optional in standalone mode)
    if not state.settings.standalone_mode:
        state.asyncgate_client = AsyncGateClient(state.settings)
        state.work_poller = WorkPoller(
            state.asyncgate_client,
            state.settings,
            job_handler
        )
        logger.info("AsyncGate polling initialized")
    else:
        state.asyncgate_client = None
        state.work_poller = None
        logger.info("Running in standalone mode - polling disabled")

    # Receipt storage (standalone mode)
    if state.settings.standalone_mode:
        state.receipt_store = ReceiptStore(state.settings.receipt_storage_dir)
        logger.info(f"Receipt storage enabled: {state.settings.receipt_storage_dir}")

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
    mode: str = Field(description="Operation mode (standalone or asyncgate)")


class SubmitJobRequest(BaseModel):
    task_id: str = Field(description="Unique task identifier")
    payload: dict[str, Any] = Field(description="Task payload")
    profile: str = Field(default="default", description="Instruction profile")
    sink_config: dict[str, Any] = Field(default_factory=dict, description="Sink configuration")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Execution constraints")


# Component health check helpers
async def check_asyncgate_health() -> dict:
    """Check AsyncGate connection health."""
    if not state.asyncgate_client:
        return {"healthy": False, "error": "Client not initialized"}

    try:
        result = await state.asyncgate_client._mcp_call(
            "asyncgate.health",
            {"tenant_id": state.asyncgate_client.tenant_id},
        )
        return {
            "healthy": result.get("status") == "healthy",
            "status": result.get("status"),
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


# Internal handlers used by MCP tools
async def health_check():
    """Basic health check payload."""
    active_jobs = len(state.work_poller._active_jobs) if state.work_poller else 0
    mode = "standalone" if state.settings and state.settings.standalone_mode else "asyncgate"
    return HealthResponse(
        status="healthy",
        service="CogniGate",
        version="0.2.0",
        instance_id=state.settings.worker_id if state.settings else "cognigate-1",
        worker_id=state.settings.worker_id if state.settings else "unknown",
        active_jobs=active_jobs,
        mode=mode
    )

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

    # Determine overall health (AsyncGate not required in standalone mode)
    standalone = state.settings.standalone_mode if state.settings else False
    if standalone:
        core_healthy = ai_check.get("healthy", False)
    else:
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
            "polling": state.work_poller._running if state.work_poller else False,
            "standalone_mode": state.settings.standalone_mode if state.settings else False,
            "receipt_storage_enabled": state.receipt_store is not None
        },
        "check_duration_ms": round(elapsed_ms, 2)
    }

async def readiness_check():
    """Readiness check for orchestrators."""
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")

    # Check if shutting down
    if state.work_poller and state.work_poller.is_shutting_down():
        raise HTTPException(status_code=503, detail="Shutting down")

    return {"ready": True}

async def liveness_check():
    """Liveness check for orchestrators."""
    # Simple liveness check - if we can respond, we're alive
    return {"alive": True}

async def execute_job_sync(request: SubmitJobRequest):
    """Execute a job synchronously and return the receipt."""
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

    # Execute synchronously
    receipt = await state.job_executor.execute(lease)
    
    # Save receipt if storage enabled
    if state.receipt_store:
        state.receipt_store.save(receipt)
    
    return receipt

async def submit_job(request: SubmitJobRequest) -> dict[str, Any]:
    """Submit a job for background execution."""
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

    asyncio.create_task(run_job())

    return {"status": "accepted", "lease_id": lease.lease_id, "task_id": lease.task_id}

async def cancel_job(lease_id: str):
    """Cancel a running job."""
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

async def start_polling() -> dict[str, Any]:
    """Start polling AsyncGate for work."""
    if not state.work_poller:
        raise HTTPException(status_code=503, detail="Not ready")

    asyncio.create_task(state.work_poller.start())
    return {"status": "polling_started"}


async def stop_polling() -> dict[str, Any]:
    """Stop polling AsyncGate."""
    if state.work_poller:
        await state.work_poller.stop()
    return {"status": "polling_stopped"}

async def list_profiles():
    """List available instruction profiles."""
    if not state.bootstrap:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "profiles": list(state.bootstrap.profiles.keys())
    }

async def list_sinks():
    """List available output sinks."""
    if not state.sink_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "sinks": state.sink_registry.list_sinks()
    }

async def list_mcp_adapters():
    """List available MCP adapters."""
    if not state.mcp_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "adapters": state.mcp_registry.list_adapters()
    }

async def get_receipt(lease_id: str):
    """Get receipt for a specific lease."""
    if not state.receipt_store:
        raise HTTPException(
            status_code=503,
            detail="Receipt storage not enabled (set standalone_mode=true)"
        )

    receipt = state.receipt_store.get(lease_id)
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")

    return receipt


async def list_receipts(limit: int = 100):
    """List recent receipts."""
    if not state.receipt_store:
        raise HTTPException(
            status_code=503,
            detail="Receipt storage not enabled (set standalone_mode=true)"
        )

    return {"receipts": state.receipt_store.list(limit)}


# MCP JSON-RPC interface (canonical)


class MCPRequest(BaseModel):
    """JSON-RPC request envelope for MCP."""

    jsonrpc: str = Field(default="2.0")
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    id: Any = None


MCP_TOOLS = [
    {"name": "cognigate.health", "description": "Health check", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.health_detailed", "description": "Detailed health", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.ready", "description": "Readiness check", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.live", "description": "Liveness check", "inputSchema": {"type": "object", "properties": {}}},
    {
        "name": "cognigate.metrics",
        "description": "Metrics snapshot (text or structured)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["text", "structured"], "description": "Response format"},
                "max_bytes": {"type": "integer", "minimum": 1, "description": "Max response size in bytes"},
            },
        },
    },
    {
        "name": "cognigate.execute_job",
        "description": "Execute job synchronously",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "payload": {"type": "object"},
                "profile": {"type": "string"},
                "sink_config": {"type": "object"},
                "constraints": {"type": "object"},
            },
            "required": ["task_id", "payload"],
        },
    },
    {
        "name": "cognigate.submit_job",
        "description": "Submit job for background execution",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "payload": {"type": "object"},
                "profile": {"type": "string"},
                "sink_config": {"type": "object"},
                "constraints": {"type": "object"},
            },
            "required": ["task_id", "payload"],
        },
    },
    {
        "name": "cognigate.cancel_job",
        "description": "Cancel a running job",
        "inputSchema": {"type": "object", "properties": {"lease_id": {"type": "string"}}, "required": ["lease_id"]},
    },
    {"name": "cognigate.polling_start", "description": "Start polling AsyncGate", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.polling_stop", "description": "Stop polling AsyncGate", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.list_profiles", "description": "List instruction profiles", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.list_sinks", "description": "List output sinks", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.list_mcp_adapters", "description": "List MCP adapters", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "cognigate.list_receipts", "description": "List receipts (standalone mode only)", "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "cognigate.get_receipt", "description": "Get receipt by lease id (standalone mode only)", "inputSchema": {"type": "object", "properties": {"lease_id": {"type": "string"}}, "required": ["lease_id"]}},
]


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: Any, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


async def _ensure_auth(http_request: Request) -> None:
    if not state.settings:
        raise HTTPException(status_code=503, detail="Auth not initialized")
    if state.settings.allow_insecure_dev:
        return
    authorization = http_request.headers.get("authorization")
    api_key = http_request.headers.get("x-api-key")
    await state.auth_dependency(authorization=authorization, x_api_key=api_key)


@app.post("/mcp")
async def mcp_entry(request: MCPRequest, http_request: Request):
    """Handle MCP JSON-RPC requests."""
    await rate_limit_dependency(http_request)
    await _ensure_auth(http_request)

    if request.method == "tools/list":
        return _jsonrpc_result(request.id, {"tools": MCP_TOOLS})

    if request.method != "tools/call":
        return _jsonrpc_error(request.id, -32601, f"Method not found: {request.method}")

    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments") or {}
    if not tool_name:
        return _jsonrpc_error(request.id, -32602, "Missing tool name")

    try:
        result = await _handle_tool(tool_name, arguments)
        return _jsonrpc_result(request.id, result)
    except HTTPException as exc:
        return _jsonrpc_error(request.id, exc.status_code, exc.detail)
    except Exception as exc:
        return _jsonrpc_error(request.id, "ERROR", str(exc))


async def _handle_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "cognigate.health":
        return (await health_check()).model_dump()
    if name == "cognigate.health_detailed":
        return await detailed_health_check()
    if name == "cognigate.ready":
        return await readiness_check()
    if name == "cognigate.live":
        return await liveness_check()
    if name == "cognigate.metrics":
        format_value = arguments.get("format") or "text"
        if not isinstance(format_value, str):
            raise HTTPException(status_code=400, detail="format must be a string")
        format_value = format_value.lower()
        if format_value not in ("text", "structured", "prometheus", "prometheus_text", "json"):
            raise HTTPException(status_code=400, detail="format must be 'text' or 'structured'")
        max_bytes = arguments.get("max_bytes")
        if max_bytes is not None:
            try:
                max_bytes = int(max_bytes)
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail="max_bytes must be an integer") from exc
            if max_bytes <= 0:
                raise HTTPException(status_code=400, detail="max_bytes must be > 0")
        return get_metrics_snapshot(format=format_value, max_bytes=max_bytes)

    if name == "cognigate.execute_job":
        request = SubmitJobRequest(**arguments)
        return (await execute_job_sync(request)).model_dump()

    if name == "cognigate.submit_job":
        request = SubmitJobRequest(**arguments)
        return await submit_job(request)

    if name == "cognigate.cancel_job":
        lease_id = arguments.get("lease_id")
        if not lease_id:
            raise HTTPException(status_code=400, detail="lease_id is required")
        if state.work_poller and lease_id not in state.work_poller._active_jobs:
            raise HTTPException(status_code=404, detail="Job not found or already completed")
        if not state.job_executor:
            raise HTTPException(status_code=503, detail="Not ready")
        state.job_executor.cancel_job(lease_id)
        return {"status": "cancellation_requested", "lease_id": lease_id}

    if name == "cognigate.polling_start":
        if not state.work_poller:
            raise HTTPException(status_code=503, detail="Not ready")
        return await start_polling()

    if name == "cognigate.polling_stop":
        if state.work_poller:
            await state.work_poller.stop()
        return {"status": "polling_stopped"}

    if name == "cognigate.list_profiles":
        if not state.bootstrap:
            raise HTTPException(status_code=503, detail="Not ready")
        return {"profiles": list(state.bootstrap.profiles.keys())}

    if name == "cognigate.list_sinks":
        if not state.sink_registry:
            raise HTTPException(status_code=503, detail="Not ready")
        return {"sinks": state.sink_registry.list_sinks()}

    if name == "cognigate.list_mcp_adapters":
        if not state.mcp_registry:
            raise HTTPException(status_code=503, detail="Not ready")
        return {"adapters": state.mcp_registry.list_adapters()}

    if name == "cognigate.get_receipt":
        lease_id = arguments.get("lease_id")
        if not lease_id:
            raise HTTPException(status_code=400, detail="lease_id is required")
        if not state.receipt_store:
            raise HTTPException(status_code=503, detail="Receipt storage not enabled")
        receipt = state.receipt_store.get(lease_id)
        if not receipt:
            raise HTTPException(status_code=404, detail="Receipt not found")
        return receipt.model_dump()

    if name == "cognigate.list_receipts":
        if not state.receipt_store:
            raise HTTPException(status_code=503, detail="Receipt storage not enabled")
        limit = int(arguments.get("limit") or 100)
        return {"receipts": state.receipt_store.list(limit)}

    raise HTTPException(status_code=404, detail=f"Unknown tool: {name}")
