"""Tests for data models."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognigate.models import (
    ArtifactPointer,
    CognitiveStep,
    ExecutionPlan,
    Lease,
    Receipt,
    ReceiptStatus,
)


class TestLease:
    """Tests for Lease model."""

    def test_lease_creation_valid(self):
        """Test creating a valid lease."""
        lease = Lease(
            lease_id="test-lease",
            task_id="test-task",
            payload={"task": "Do something"},
            constraints={"timeout": 60}
        )
        
        assert lease.lease_id == "test-lease"
        assert lease.task_id == "test-task"
        assert lease.payload["task"] == "Do something"
        assert lease.constraints["timeout"] == 60

    def test_lease_payload_required(self):
        """Test payload is required."""
        with pytest.raises(ValidationError):
            Lease(
                lease_id="test",
                task_id="test"
                # Missing payload
            )

    def test_lease_constraints_optional(self):
        """Test constraints are optional."""
        lease = Lease(
            lease_id="test",
            task_id="test",
            payload={"task": "test"}
        )
        
        # Constraints should be None or empty
        assert lease.constraints is None or lease.constraints == {}

    def test_lease_with_empty_payload(self):
        """Test lease with empty payload."""
        lease = Lease(
            lease_id="test",
            task_id="test",
            payload={}
        )
        
        assert lease.payload == {}


class TestExecutionPlan:
    """Tests for ExecutionPlan model."""

    def test_execution_plan_creation(self):
        """Test creating an execution plan."""
        plan = ExecutionPlan(
            steps=[
                CognitiveStep(
                    step_number=1,
                    step_type="cognitive",
                    description="Analyze data",
                    instructions="Look at the numbers"
                )
            ],
            summary="Plan to analyze data"
        )
        
        assert len(plan.steps) == 1
        assert plan.steps[0].step_number == 1
        assert plan.summary == "Plan to analyze data"

    def test_execution_plan_steps_required(self):
        """Test steps are required."""
        with pytest.raises(ValidationError):
            ExecutionPlan(
                summary="test"
                # Missing steps
            )

    def test_execution_plan_empty_steps(self):
        """Test plan can have empty steps list."""
        plan = ExecutionPlan(
            steps=[],
            summary="Empty plan"
        )
        
        assert len(plan.steps) == 0


class TestCognitiveStep:
    """Tests for CognitiveStep model."""

    def test_cognitive_step_creation(self):
        """Test creating a cognitive step."""
        step = CognitiveStep(
            step_number=1,
            step_type="cognitive",
            description="Analyze data",
            instructions="Do the analysis"
        )
        
        assert step.step_number == 1
        assert step.step_type == "cognitive"
        assert step.instructions == "Do the analysis"

    def test_tool_invocation_step(self):
        """Test creating a tool invocation step."""
        step = CognitiveStep(
            step_number=2,
            step_type="tool_invocation",
            description="Fetch data",
            tool_name="mcp.call",
            tool_params={"method": "fetch", "args": {}}
        )
        
        assert step.step_type == "tool_invocation"
        assert step.tool_name == "mcp.call"
        assert "method" in step.tool_params

    def test_output_generation_step(self):
        """Test creating an output generation step."""
        step = CognitiveStep(
            step_number=3,
            step_type="output_generation",
            description="Write results",
            instructions="Format the output"
        )
        
        assert step.step_type == "output_generation"

    def test_step_type_validation(self):
        """Test step_type must be valid."""
        valid_types = ["cognitive", "tool_invocation", "output_generation"]
        
        for step_type in valid_types:
            step = CognitiveStep(
                step_number=1,
                step_type=step_type,
                description="test"
            )
            assert step.step_type == step_type


class TestArtifactPointer:
    """Tests for ArtifactPointer model."""

    def test_artifact_pointer_creation(self):
        """Test creating an artifact pointer."""
        pointer = ArtifactPointer(
            sink_id="file-sink",
            uri="file:///path/to/artifact.txt",
            metadata={"size": 1024}
        )
        
        assert pointer.sink_id == "file-sink"
        assert pointer.uri == "file:///path/to/artifact.txt"
        assert pointer.metadata["size"] == 1024

    def test_artifact_pointer_required_fields(self):
        """Test required fields for artifact pointer."""
        # sink_id and uri are required
        with pytest.raises(ValidationError):
            ArtifactPointer(
                sink_id="test"
                # Missing uri
            )

    def test_artifact_pointer_metadata_optional(self):
        """Test metadata is optional."""
        pointer = ArtifactPointer(
            sink_id="test",
            uri="file:///test"
        )
        
        assert pointer.metadata is None or pointer.metadata == {}


class TestReceipt:
    """Tests for Receipt model."""

    def test_receipt_creation(self):
        """Test creating a receipt."""
        receipt = Receipt(
            receipt_id=str(uuid4()),
            task_id="test-task",
            lease_id="test-lease",
            status=ReceiptStatus.COMPLETED,
            summary="Task completed successfully"
        )
        
        assert receipt.status == ReceiptStatus.COMPLETED
        assert receipt.task_id == "test-task"

    def test_receipt_status_enum(self):
        """Test receipt status enum values."""
        statuses = [
            ReceiptStatus.ACCEPTED,
            ReceiptStatus.PLANNING,
            ReceiptStatus.EXECUTING,
            ReceiptStatus.COMPLETED,
            ReceiptStatus.FAILED
        ]
        
        for status in statuses:
            receipt = Receipt(
                receipt_id=str(uuid4()),
                task_id="test",
                lease_id="test",
                status=status
            )
            assert receipt.status == status

    def test_receipt_with_artifacts(self):
        """Test receipt with artifact pointers."""
        receipt = Receipt(
            receipt_id=str(uuid4()),
            task_id="test",
            lease_id="test",
            status=ReceiptStatus.COMPLETED,
            artifacts=[
                ArtifactPointer(
                    sink_id="test-sink",
                    uri="file:///test.txt"
                )
            ]
        )
        
        assert len(receipt.artifacts) == 1
        assert receipt.artifacts[0].uri == "file:///test.txt"

    def test_receipt_with_error(self):
        """Test receipt with error information."""
        receipt = Receipt(
            receipt_id=str(uuid4()),
            task_id="test",
            lease_id="test",
            status=ReceiptStatus.FAILED,
            error="Something went wrong"
        )
        
        assert receipt.status == ReceiptStatus.FAILED
        assert receipt.error == "Something went wrong"


class TestModelSerialization:
    """Tests for model serialization."""

    def test_lease_serialization(self):
        """Test lease can be serialized to dict."""
        lease = Lease(
            lease_id="test",
            task_id="test",
            payload={"task": "test"},
            constraints={"timeout": 60}
        )
        
        data = lease.model_dump()
        
        assert data["lease_id"] == "test"
        assert data["payload"]["task"] == "test"

    def test_lease_json_serialization(self):
        """Test lease can be serialized to JSON."""
        lease = Lease(
            lease_id="test",
            task_id="test",
            payload={"task": "test"}
        )
        
        json_str = lease.model_dump_json()
        
        assert "test-task" not in json_str  # Wrong ID
        assert "test" in json_str

    def test_receipt_serialization(self):
        """Test receipt can be serialized."""
        receipt = Receipt(
            receipt_id=str(uuid4()),
            task_id="test",
            lease_id="test",
            status=ReceiptStatus.COMPLETED
        )
        
        data = receipt.model_dump()
        
        assert "receipt_id" in data
        assert data["status"] == "completed"

    def test_complex_nested_serialization(self):
        """Test serialization of complex nested models."""
        receipt = Receipt(
            receipt_id=str(uuid4()),
            task_id="test",
            lease_id="test",
            status=ReceiptStatus.COMPLETED,
            artifacts=[
                ArtifactPointer(
                    sink_id="sink1",
                    uri="file:///test1.txt",
                    metadata={"size": 100}
                ),
                ArtifactPointer(
                    sink_id="sink2",
                    uri="file:///test2.txt",
                    metadata={"size": 200}
                )
            ]
        )
        
        data = receipt.model_dump()
        
        assert len(data["artifacts"]) == 2
        assert data["artifacts"][0]["sink_id"] == "sink1"
        assert data["artifacts"][1]["metadata"]["size"] == 200


class TestModelValidation:
    """Tests for model validation."""

    def test_invalid_step_type_rejected(self):
        """Test invalid step type is rejected."""
        # This depends on how step_type is validated in the model
        # Assuming it's an enum or has constraints
        with pytest.raises((ValidationError, ValueError)):
            CognitiveStep(
                step_number=1,
                step_type="invalid_type",
                description="test"
            )

    def test_negative_step_number_rejected(self):
        """Test negative step number is rejected."""
        with pytest.raises(ValidationError):
            CognitiveStep(
                step_number=-1,
                step_type="cognitive",
                description="test"
            )

    def test_empty_lease_id_rejected(self):
        """Test empty lease ID is rejected."""
        with pytest.raises(ValidationError):
            Lease(
                lease_id="",
                task_id="test",
                payload={"task": "test"}
            )


# Mark as unit tests
pytestmark = pytest.mark.unit
