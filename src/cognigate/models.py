"""Data models for CogniGate."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .plugins.base import ArtifactPointer


class JobStatus(str, Enum):
    """Status of a job/lease."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ReceiptStatus(str, Enum):
    """Status values for receipt lifecycle."""
    ACCEPTED = "accepted"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
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
    caused_by_receipt_id: str | None = Field(
        default=None,
        description="Receipt that caused this lease"
    )
    payload: dict[str, Any] = Field(description="Task payload/parameters")
    payload_pointer: str | None = Field(
        default=None,
        description="Pointer to payload stored in DepotGate or external store"
    )
    principal_ai: str | None = Field(
        default=None,
        description="Principal AI that owns the obligation"
    )
    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    task_type: str | None = Field(default=None, description="Task type/category")
    expected_outcome_kind: str | None = Field(default=None, description="Expected outcome kind")
    expected_artifact_mime: str | None = Field(default=None, description="Expected artifact MIME")
    profile: str = Field(default="default", description="Instruction profile to use")
    sink_config: dict[str, Any] = Field(default_factory=dict, description="Output sink configuration")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Execution constraints")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("constraints", mode="before")
    @classmethod
    def _normalize_constraints(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        raise ValueError("constraints must be a dict")

    @field_validator("lease_id", "task_id")
    @classmethod
    def _non_empty_identifier(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("identifier must be non-empty")
        return v


class Receipt(BaseModel):
    """A receipt documenting job state or completion."""
    receipt_id: str | None = Field(default=None, description="Receipt identifier")
    lease_id: str = Field(description="ID of the lease this receipt is for")
    task_id: str = Field(description="ID of the task")
    worker_id: str | None = Field(default=None, description="ID of the worker processing the job")
    status: ReceiptStatus | JobStatus = Field(description="Current job status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_pointers: list[ArtifactPointer] = Field(
        default_factory=list,
        description="Pointers to produced artifacts",
    )
    summary: str = Field(
        default="",
        max_length=1000,
        description="Bounded summary of results",
    )
    error_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Error information if failed",
    )
    error: str | None = Field(default=None, description="Compatibility error message")

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
            "has_error": self.error_metadata is not None or self.error is not None,
            "error_code": self.error_metadata.get("code") if self.error_metadata else None,
        }

    @property
    def artifacts(self) -> list[ArtifactPointer]:
        """Backwards-compatible alias for artifact_pointers."""
        return self.artifact_pointers

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        data = super().model_dump(*args, **kwargs)
        if "artifacts" not in data:
            data["artifacts"] = data.get("artifact_pointers", [])
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        if "artifacts" in values and "artifact_pointers" not in values:
            values["artifact_pointers"] = values["artifacts"]

        if "error" in values and "error_metadata" not in values:
            err = values["error"]
            values["error_metadata"] = err if isinstance(err, dict) else {"message": err}

        if "error_metadata" in values and "error" not in values:
            if isinstance(values["error_metadata"], dict):
                values["error"] = values["error_metadata"].get("message")

        return values

    model_config = {"populate_by_name": True}


class PlanStep(BaseModel):
    """A single step in an execution plan."""
    step_number: int = Field(description="Order of this step")
    step_type: PlanStepType = Field(description="Type of step")
    description: str = Field(description="Human-readable description")
    tool_name: str | None = Field(default=None, description="Tool to invoke if tool_invocation")
    tool_params: dict[str, Any] | None = Field(default=None, description="Tool parameters")
    instructions: str | None = Field(default=None, description="Instructions for cognitive steps")

    @field_validator("step_number")
    @classmethod
    def _positive_step_number(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step_number must be >= 1")
        return v


class ExecutionSteps(BaseModel):
    """Structured execution steps produced by the planning phase."""
    task_id: str = Field(description="ID of the task this plan is for")
    steps: list[PlanStep] = Field(description="Ordered list of steps")
    estimated_tool_calls: int = Field(default=0, description="Estimated number of tool calls")
    summary: str = Field(default="", description="Brief summary of the plan")


class CognitiveStep(PlanStep):
    """Backward-compatible alias for plan steps."""


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
