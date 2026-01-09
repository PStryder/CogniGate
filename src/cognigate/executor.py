"""Execution engine for CogniGate.

Handles the planning phase and execution loop for jobs.
"""

from datetime import datetime, timezone
from typing import Any

from .ai_client import AIClient
from .config import Bootstrap, Settings
from .models import (
    Lease,
    Receipt,
    JobStatus,
    ExecutionPlan,
    PlanStep,
    PlanStepType,
    ToolCall,
)
from .prompt import PromptBuilder, build_tool_prompt
from .tools import ToolExecutor, ToolContext, parse_tool_calls
from .observability import get_logger, JobContext, SpanContext
from .metrics import (
    track_job_duration,
    record_job_complete,
    ACTIVE_JOBS,
)


logger = get_logger(__name__)


class ExecutionError(Exception):
    """Error during job execution."""

    def __init__(self, message: str, code: str = "EXECUTION_ERROR", recoverable: bool = False):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class JobCancelledError(ExecutionError):
    """Error raised when a job is cancelled."""

    def __init__(self, message: str = "Job was cancelled"):
        super().__init__(message, code="JOB_CANCELLED", recoverable=False)


class JobExecutor:
    """Executes jobs according to the CogniGate execution model.

    Flow:
    1. Planning phase: AI produces structured plan
    2. Execution loop: Execute each step
       - Cognitive steps: AI reasoning
       - Tool invocations: Execute via ToolExecutor
       - Output generation: Write to sinks
    3. Completion: Produce final receipt
    """

    def __init__(
        self,
        ai_client: AIClient,
        tool_executor: ToolExecutor,
        bootstrap: Bootstrap,
        settings: Settings
    ):
        self.ai_client = ai_client
        self.tool_executor = tool_executor
        self.bootstrap = bootstrap
        self.settings = settings
        self.max_retries = settings.max_retries
        self._cancelled_jobs: set[str] = set()

    def cancel_job(self, lease_id: str) -> bool:
        """Mark a job for cancellation.

        The job will be cancelled at the next step boundary.

        Args:
            lease_id: The lease ID of the job to cancel

        Returns:
            True if the job was marked for cancellation
        """
        self._cancelled_jobs.add(lease_id)
        logger.info("job_cancellation_requested", lease_id=lease_id)
        return True

    def is_cancelled(self, lease_id: str) -> bool:
        """Check if a job has been cancelled.

        Args:
            lease_id: The lease ID to check

        Returns:
            True if the job has been cancelled
        """
        return lease_id in self._cancelled_jobs

    def _check_cancellation(self, lease_id: str) -> None:
        """Check if job is cancelled and raise if so.

        Args:
            lease_id: The lease ID to check

        Raises:
            JobCancelledError: If the job has been cancelled
        """
        if lease_id in self._cancelled_jobs:
            raise JobCancelledError()

    def _clear_cancellation(self, lease_id: str) -> None:
        """Clear cancellation flag for a job.

        Args:
            lease_id: The lease ID to clear
        """
        self._cancelled_jobs.discard(lease_id)

    async def execute(self, lease: Lease) -> Receipt:
        """Execute a leased job.

        Args:
            lease: The work lease to execute

        Returns:
            Receipt documenting completion or failure
        """
        status_holder = ["unknown"]

        with JobContext(
            task_id=lease.task_id,
            lease_id=lease.lease_id,
            worker_id=self.settings.worker_id,
            profile=lease.profile
        ):
            logger.info(
                "job_started",
                task_id=lease.task_id,
                lease_id=lease.lease_id,
                profile=lease.profile
            )

            with track_job_duration(lease.profile, status_holder):
                # Clear artifacts from previous jobs
                self.tool_executor.clear_artifacts()

                try:
                    with SpanContext("get_profile"):
                        # Get instruction profile
                        profile = self.bootstrap.get_profile(lease.profile)
                        if not profile:
                            profile = self.bootstrap.get_default_profile()
                        if not profile:
                            raise ExecutionError(
                                f"No instruction profile found: {lease.profile}",
                                code="PROFILE_NOT_FOUND"
                            )

                    # Build prompt builder
                    prompt_builder = PromptBuilder(profile)

                    # Planning phase
                    with SpanContext("planning_phase"):
                        plan = await self._planning_phase(lease, prompt_builder)
                        logger.info(
                            "plan_created",
                            step_count=len(plan.steps),
                            estimated_tool_calls=plan.estimated_tool_calls
                        )

                    # Execution loop
                    with SpanContext("execution_loop"):
                        context = await self._execution_loop(lease, plan, prompt_builder)

                    # Get artifacts
                    artifacts = self.tool_executor.get_artifacts()
                    artifact_dicts = [
                        {"sink_id": a.sink_id, "uri": a.uri, "metadata": a.metadata}
                        for a in artifacts
                    ]

                    status_holder[0] = "complete"
                    record_job_complete("complete")

                    logger.info(
                        "job_completed",
                        artifact_count=len(artifacts),
                        output_count=len(context.get("outputs", []))
                    )

                    # Create success receipt
                    return Receipt(
                        lease_id=lease.lease_id,
                        task_id=lease.task_id,
                        worker_id=self.settings.worker_id,
                        status=JobStatus.COMPLETE,
                        timestamp=datetime.now(timezone.utc),
                        artifact_pointers=artifact_dicts,
                        summary=self._create_summary(plan, context)
                    )

                except JobCancelledError as e:
                    status_holder[0] = "cancelled"
                    record_job_complete("cancelled")
                    logger.info(
                        "job_cancelled",
                        lease_id=lease.lease_id,
                        task_id=lease.task_id
                    )
                    return Receipt(
                        lease_id=lease.lease_id,
                        task_id=lease.task_id,
                        worker_id=self.settings.worker_id,
                        status=JobStatus.FAILED,
                        timestamp=datetime.now(timezone.utc),
                        error_metadata={"code": e.code, "message": str(e)}
                    )

                except ExecutionError as e:
                    status_holder[0] = "failed"
                    record_job_complete("failed")
                    logger.error(
                        "job_execution_error",
                        error_code=e.code,
                        error_message=str(e),
                        recoverable=e.recoverable
                    )
                    return Receipt(
                        lease_id=lease.lease_id,
                        task_id=lease.task_id,
                        worker_id=self.settings.worker_id,
                        status=JobStatus.FAILED,
                        timestamp=datetime.now(timezone.utc),
                        error_metadata={"code": e.code, "message": str(e)}
                    )

                except Exception as e:
                    status_holder[0] = "failed"
                    record_job_complete("failed")
                    logger.exception(
                        "job_unexpected_error",
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )
                    return Receipt(
                        lease_id=lease.lease_id,
                        task_id=lease.task_id,
                        worker_id=self.settings.worker_id,
                        status=JobStatus.FAILED,
                        timestamp=datetime.now(timezone.utc),
                        error_metadata={"code": "UNEXPECTED_ERROR", "message": str(e)}
                    )

                finally:
                    # Clear cancellation flag after job completes
                    self._clear_cancellation(lease.lease_id)

    async def _planning_phase(
        self,
        lease: Lease,
        prompt_builder: PromptBuilder
    ) -> ExecutionPlan:
        """Execute the planning phase.

        Args:
            lease: The work lease
            prompt_builder: Prompt builder with profile

        Returns:
            ExecutionPlan with ordered steps
        """
        logger.info("Starting planning phase")

        messages = prompt_builder.build_planning_prompt(lease)
        plan_data = await self.ai_client.generate_plan(messages)

        # Parse plan into structured format
        steps = []
        for i, step_data in enumerate(plan_data.get("steps", [])):
            step_type_str = step_data.get("step_type", "cognitive")
            try:
                step_type = PlanStepType(step_type_str)
            except ValueError:
                step_type = PlanStepType.COGNITIVE

            steps.append(PlanStep(
                step_number=step_data.get("step_number", i + 1),
                step_type=step_type,
                description=step_data.get("description", ""),
                tool_name=step_data.get("tool_name"),
                tool_params=step_data.get("tool_params"),
                instructions=step_data.get("instructions")
            ))

        return ExecutionPlan(
            task_id=lease.task_id,
            steps=steps,
            estimated_tool_calls=sum(1 for s in steps if s.step_type == PlanStepType.TOOL_INVOCATION),
            summary=plan_data.get("summary", "")
        )

    async def _execution_loop(
        self,
        lease: Lease,
        plan: ExecutionPlan,
        prompt_builder: PromptBuilder
    ) -> dict[str, Any]:
        """Execute the plan steps.

        Args:
            lease: The work lease
            plan: The execution plan
            prompt_builder: Prompt builder

        Returns:
            Context dictionary with execution state
        """
        logger.info("Starting execution loop")

        context: dict[str, Any] = {
            "results": [],
            "outputs": []
        }

        tool_context = ToolContext(
            task_id=lease.task_id,
            lease_id=lease.lease_id,
            worker_id=self.settings.worker_id,
            sink_config=lease.sink_config
        )

        for step in plan.steps:
            # Check for cancellation at each step boundary
            self._check_cancellation(lease.lease_id)

            logger.info(
                "step_started",
                step_number=step.step_number,
                step_type=step.step_type.value
            )

            try:
                if step.step_type == PlanStepType.COGNITIVE:
                    result = await self._execute_cognitive_step(
                        lease, step, prompt_builder, context
                    )
                    context["results"].append({"step": step.step_number, "result": result})

                elif step.step_type == PlanStepType.TOOL_INVOCATION:
                    result = await self._execute_tool_step(step, tool_context)
                    context["results"].append({"step": step.step_number, "result": result})

                elif step.step_type == PlanStepType.OUTPUT_GENERATION:
                    result = await self._execute_output_step(
                        lease, step, prompt_builder, tool_context, context
                    )
                    context["outputs"].append(result)

                # Check cancellation after step completion
                self._check_cancellation(lease.lease_id)

            except JobCancelledError:
                logger.info(
                    "job_cancelled_during_execution",
                    lease_id=lease.lease_id,
                    step_number=step.step_number
                )
                raise
            except ExecutionError:
                raise
            except Exception as e:
                logger.error(
                    "step_failed",
                    step_number=step.step_number,
                    error=str(e)
                )
                raise ExecutionError(
                    f"Step {step.step_number} failed: {e}",
                    code="STEP_FAILED"
                )

        return context

    async def _execute_cognitive_step(
        self,
        lease: Lease,
        step: PlanStep,
        prompt_builder: PromptBuilder,
        context: dict[str, Any]
    ) -> str:
        """Execute a cognitive (reasoning) step.

        Args:
            lease: The work lease
            step: The plan step
            prompt_builder: Prompt builder
            context: Execution context

        Returns:
            The AI's reasoning output
        """
        # Build context from previous results
        prev_context = None
        if context["results"]:
            prev_parts = [
                f"Step {r['step']}: {r['result'][:500]}..."
                if len(str(r['result'])) > 500 else f"Step {r['step']}: {r['result']}"
                for r in context["results"][-3:]  # Last 3 results
            ]
            prev_context = "\n\n".join(prev_parts)

        instructions = step.instructions or step.description
        messages = prompt_builder.build_execution_prompt(
            lease, instructions, prev_context
        )

        text, _ = await self.ai_client.chat_with_tools(
            messages=messages,
            tools=[],  # No tools for cognitive steps
            temperature=0.7
        )

        return text or ""

    async def _execute_tool_step(
        self,
        step: PlanStep,
        tool_context: ToolContext
    ) -> dict[str, Any]:
        """Execute a tool invocation step.

        Args:
            step: The plan step with tool info
            tool_context: Tool execution context

        Returns:
            Tool result dictionary
        """
        if not step.tool_name:
            raise ExecutionError("Tool step missing tool_name", code="INVALID_STEP")

        tool_call = ToolCall(
            tool_name=step.tool_name,
            arguments=step.tool_params or {},
            call_id=f"step_{step.step_number}"
        )

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries):
            result = await self.tool_executor.execute(tool_call, tool_context)

            if result.success:
                return {"success": True, "result": result.result}

            last_error = result.error
            logger.warning(f"Tool call attempt {attempt + 1} failed: {last_error}")

        # All retries failed
        raise ExecutionError(
            f"Tool {step.tool_name} failed after {self.max_retries} attempts: {last_error}",
            code="TOOL_FAILED"
        )

    async def _execute_output_step(
        self,
        lease: Lease,
        step: PlanStep,
        prompt_builder: PromptBuilder,
        tool_context: ToolContext,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute an output generation step.

        This may involve AI generating content and then writing to a sink.

        Args:
            lease: The work lease
            step: The plan step
            prompt_builder: Prompt builder
            tool_context: Tool context
            context: Execution context

        Returns:
            Output result dictionary
        """
        # Build context from all results
        all_results = "\n\n".join(
            f"Step {r['step']}: {r['result']}"
            for r in context["results"]
        )

        instructions = step.instructions or f"Generate the final output. Context:\n{all_results}"
        messages = prompt_builder.build_execution_prompt(
            lease, instructions, all_results
        )

        # Get tools for potential artifact writing
        tools = self.tool_executor.get_tool_definitions()

        text, tool_calls = await self.ai_client.chat_with_tools(
            messages=messages,
            tools=tools,
            temperature=0.5
        )

        # Handle any tool calls (likely artifact_write)
        for tc in tool_calls:
            if tc.get("type") == "function":
                func = tc["function"]
                import json
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)

                tool_call = ToolCall(
                    tool_name=func["name"],
                    arguments=args,
                    call_id=tc.get("id", "output")
                )
                await self.tool_executor.execute(tool_call, tool_context)

        return {"content": text, "tool_calls": len(tool_calls)}

    def _create_summary(self, plan: ExecutionPlan, context: dict[str, Any]) -> str:
        """Create a bounded summary of execution.

        Args:
            plan: The execution plan
            context: Execution context

        Returns:
            Summary string (max 1000 chars)
        """
        parts = [
            f"Executed {len(plan.steps)} steps.",
            f"Plan: {plan.summary[:200]}" if plan.summary else "",
            f"Outputs: {len(context.get('outputs', []))}"
        ]

        summary = " ".join(p for p in parts if p)
        return summary[:1000]
