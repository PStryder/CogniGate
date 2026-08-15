"""Planning for a caller must produce a plan and stop there.

DeleGate holds the planning authority but has no cognition of its own, so it
asks CogniGate what an intent decomposes into and then mints the obligations
itself. The load-bearing property is the *stop*: if plan_only ran the execution
loop, CogniGate would be performing the work while DeleGate still believed it
was deciding what the work is, breaking DeleGate's stated invariant that it
never executes.

These tests therefore assert as much about what does not happen -- no tool
calls, no artifacts, no sink writes -- as about the plan that comes back.
"""

import httpx
import pytest
from unittest.mock import patch

from cognigate.config import Bootstrap
from cognigate.executor import ExecutionError, JobExecutor
from cognigate.models import Lease, PlanStepType
from cognigate.plugins import MCPAdapterRegistry, SinkRegistry
from cognigate.plugins.builtin_sinks import register_builtin_sinks
from cognigate.ai_client import AIClient
from cognigate.tools import ToolExecutor

from .fixtures import (
    IntegrationTestHarness,
    MockAsyncGateServer,
    MockAIProvider,
    MockMCPServer,
    integration_harness,
    mock_asyncgate,
    mock_ai_provider,
    simple_lease,
    complex_lease,
    integration_settings,
    test_instruction_profile,
    test_mcp_endpoint,
)


pytestmark = pytest.mark.asyncio


def _build_executor(harness, settings, profile):
    """Assemble a JobExecutor over the harness's mock provider."""
    mock_client = harness.create_mock_http_client()
    bootstrap = Bootstrap(settings)
    bootstrap.profiles = {"default": profile}
    bootstrap._loaded = True

    sink_registry = SinkRegistry()
    register_builtin_sinks(sink_registry)

    ai_client = AIClient(settings.get_ai_config())
    ai_client._client = mock_client

    tool_executor = ToolExecutor(MCPAdapterRegistry(), sink_registry, max_retries=3)
    return JobExecutor(ai_client, tool_executor, bootstrap, settings), tool_executor, mock_client


def _plan_lease(profile: str = "default") -> Lease:
    return Lease(
        lease_id="plan-lease-1",
        task_id="plan-request-1",
        payload={"intent": "research competitors and draft a summary", "task_type": "general"},
        profile=profile,
    )


class TestPlanOnly:
    async def test_returns_the_planned_steps(
        self, integration_harness, integration_settings, test_instruction_profile
    ):
        harness = integration_harness
        harness.ai_provider.set_plan_response({
            "steps": [
                {"step_number": 1, "step_type": "cognitive", "description": "Research competitors"},
                {"step_number": 2, "step_type": "output_generation", "description": "Draft summary"},
            ],
            "summary": "Two-step research and drafting plan",
        })

        with patch.object(httpx, "AsyncClient", return_value=harness.create_mock_http_client()):
            executor, _, _ = _build_executor(
                harness, integration_settings, test_instruction_profile
            )
            plan = await executor.plan_only(_plan_lease())

        assert [s.description for s in plan.steps] == [
            "Research competitors",
            "Draft summary",
        ]
        assert plan.steps[0].step_type is PlanStepType.COGNITIVE
        assert plan.summary == "Two-step research and drafting plan"

    async def test_nothing_is_executed(
        self, integration_harness, integration_settings, test_instruction_profile
    ):
        """The whole point: a plan comes back, no work is done."""
        harness = integration_harness
        harness.ai_provider.set_plan_response({
            "steps": [
                {
                    "step_number": 1,
                    "step_type": "tool_invocation",
                    "description": "Fetch competitor data",
                    "tool_name": "getData",
                    "tool_params": {"q": "competitors"},
                },
            ],
            "summary": "Would call a tool if executed",
        })
        # Deliberately empty: any chat call beyond planning would consume one of
        # these and fail, so an accidental execution loop cannot pass silently.
        harness.ai_provider.set_chat_responses([])

        with patch.object(httpx, "AsyncClient", return_value=harness.create_mock_http_client()):
            executor, tool_executor, _ = _build_executor(
                harness, integration_settings, test_instruction_profile
            )
            plan = await executor.plan_only(_plan_lease())

        # The step names a tool, and the tool was still never invoked.
        assert plan.steps[0].tool_name == "getData"
        assert tool_executor.get_artifacts() == []

    async def test_planning_does_not_mark_the_lease_completed(
        self, integration_harness, integration_settings, test_instruction_profile
    ):
        """A planning request is a question, not leased work.

        execute() caches completed leases for idempotency. Planning must not
        write into that cache, or a later real execution of the same lease id
        would return the cached result instead of running.
        """
        harness = integration_harness
        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Think"}],
            "summary": "One step",
        })

        with patch.object(httpx, "AsyncClient", return_value=harness.create_mock_http_client()):
            executor, _, _ = _build_executor(
                harness, integration_settings, test_instruction_profile
            )
            lease = _plan_lease()
            await executor.plan_only(lease)

            assert executor._completed_leases.get(lease.lease_id) is None

    async def test_missing_profile_is_refused(
        self, integration_harness, integration_settings, test_instruction_profile
    ):
        """Planning under no instruction profile would be unbounded cognition."""
        harness = integration_harness
        harness.ai_provider.set_plan_response({
            "steps": [{"step_number": 1, "step_type": "cognitive", "description": "Think"}],
            "summary": "One step",
        })

        with patch.object(httpx, "AsyncClient", return_value=harness.create_mock_http_client()):
            executor, _, _ = _build_executor(
                harness, integration_settings, test_instruction_profile
            )
            executor.bootstrap.profiles = {}

            with pytest.raises(ExecutionError) as excinfo:
                await executor.plan_only(_plan_lease(profile="nonexistent"))

        assert excinfo.value.code == "PROFILE_NOT_FOUND"
