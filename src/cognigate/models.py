"""Data models for CogniGate."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a job/lease."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class PlanStepType(str, Enum):
    """Type of a plan step."""
    COGNITIVE = "cognitive"
    TOOL_INVOCATION = "tool_invocation"
    OUTPUT_GENERATION = "output_generation"


class Lease(BaseModel):
    """A work lease from AsyncGate."""
    lease_id: str = Field(description="Unique identifier for this lease")
    task_id: str = Field(description="ID of the task being leased")
    payload: dict[str, Any] = Field(description="Task payload/parameters")
    profile: str = Field(default="default", description="Instruction profile to use")
    sink_config: dict[str, Any] = Field(default_factory=dict, description="Output sink configuration")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Execution constraints")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Receipt(BaseModel):
    """A receipt documenting job state or completion."""
    lease_id: str = Field(description="ID of the lease this receipt is for")
    task_id: str = Field(description="ID of the task")
    worker_id: str = Field(description="ID of the worker processing the job")
    status: JobStatus = Field(description="Current job status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_pointers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Pointers to produced artifacts"
    )
    summary: str = Field(
        default="",
        max_length=1000,
        description="Bounded summary of results"
    )
    error_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Error information if failed"
    )

    def to_ledger_entry(self) -> dict[str, Any]:
        """Convert to a ledger-safe entry (no large blobs or sensitive data)."""
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "artifact_count": len(self.artifact_pointers),
            "artifact_pointers": self.artifact_pointers,
            "summary": self.summary[:1000] if self.summary else "",
            "has_error": self.error_metadata is not None,
            "error_code": self.error_metadata.get("code") if self.error_metadata else None
        }


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_number: int = Field(description="Order of this step")
    step_type: PlanStepType = Field(description="Type of step")
    description: str = Field(description="Human-readable description")
    tool_name: str | None = Field(default=None, description="Tool to invoke if tool_invocation")
    tool_params: dict[str, Any] | None = Field(default=None, description="Tool parameters")
    instructions: str | None = Field(default=None, description="Instructions for cognitive steps")


class ExecutionPlan(BaseModel):
    """A structured execution plan produced by the planning phase."""
    task_id: str = Field(description="ID of the task this plan is for")
    steps: list[PlanStep] = Field(description="Ordered list of steps")
    estimated_tool_calls: int = Field(default=0, description="Estimated number of tool calls")
    summary: str = Field(default="", description="Brief summary of the plan")


class ToolCall(BaseModel):
    """A tool call request from the AI."""
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(description="Arguments for the tool")
    call_id: str = Field(default="", description="Unique ID for this call")


class ToolResult(BaseModel):
    """Result of a tool call."""
    call_id: str = Field(description="ID of the tool call this is for")
    success: bool = Field(description="Whether the call succeeded")
    result: Any = Field(default=None, description="Result data if successful")
    error: str | None = Field(default=None, description="Error message if failed")
