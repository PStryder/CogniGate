"""REST API for CogniGate."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .config import Settings, Bootstrap
from .models import Lease, Receipt, JobStatus
from .leasing import AsyncGateClient, WorkPoller
from .plugins import SinkRegistry, MCPAdapterRegistry
from .plugins.builtin_sinks import register_builtin_sinks
from .ai_client import AIClient
from .tools import ToolExecutor
from .executor import JobExecutor


logger = logging.getLogger(__name__)


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
    logger.info("Starting CogniGate...")

    # Load settings
    state.settings = Settings()

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

    logger.info("CogniGate started successfully")

    yield

    # Shutdown
    logger.info("Shutting down CogniGate...")

    if state.work_poller:
        await state.work_poller.stop()

    if state.asyncgate_client:
        await state.asyncgate_client.close()

    if state.ai_client:
        await state.ai_client.close()

    if state.mcp_registry:
        await state.mcp_registry.close_all()

    logger.info("CogniGate shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="CogniGate",
    description="General-purpose cognitive execution worker",
    version="0.1.0",
    lifespan=lifespan
)


# API Models
class HealthResponse(BaseModel):
    status: str = Field(description="Service status")
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


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    active_jobs = len(state.work_poller._active_jobs) if state.work_poller else 0
    return HealthResponse(
        status="healthy",
        worker_id=state.settings.worker_id if state.settings else "unknown",
        active_jobs=active_jobs
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")
    return {"ready": True}


@app.post("/v1/jobs", response_model=SubmitJobResponse)
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


@app.post("/v1/polling/start")
async def start_polling(background_tasks: BackgroundTasks):
    """Start polling AsyncGate for work."""
    if not state.work_poller:
        raise HTTPException(status_code=503, detail="Not ready")

    background_tasks.add_task(state.work_poller.start)
    return {"status": "polling_started"}


@app.post("/v1/polling/stop")
async def stop_polling():
    """Stop polling AsyncGate."""
    if state.work_poller:
        await state.work_poller.stop()
    return {"status": "polling_stopped"}


@app.get("/v1/config/profiles")
async def list_profiles():
    """List available instruction profiles."""
    if not state.bootstrap:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "profiles": list(state.bootstrap.profiles.keys())
    }


@app.get("/v1/config/sinks")
async def list_sinks():
    """List available output sinks."""
    if not state.sink_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "sinks": state.sink_registry.list_sinks()
    }


@app.get("/v1/config/mcp")
async def list_mcp_adapters():
    """List available MCP adapters."""
    if not state.mcp_registry:
        raise HTTPException(status_code=503, detail="Not ready")

    return {
        "adapters": state.mcp_registry.list_adapters()
    }
