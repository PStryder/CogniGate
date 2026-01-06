"""Prompt construction for CogniGate."""

from typing import Any

from .config import InstructionProfile
from .models import Lease


class PromptBuilder:
    """Builds prompts for AI invocation from leases and profiles."""

    def __init__(self, profile: InstructionProfile):
        self.profile = profile

    def build_planning_prompt(self, lease: Lease) -> list[dict[str, Any]]:
        """Build a prompt for the planning phase.

        The prompt is constructed from:
        - Lease payload (the work to be done)
        - Receipt metadata (IDs, constraints)
        - Instruction profile (system instructions, formatting)

        Does NOT include:
        - References to other tasks
        - Prior conversation state
        - System-internal implementation details

        Args:
            lease: The work lease

        Returns:
            List of messages in OpenAI-compatible format
        """
        messages = []

        # System message with instructions
        system_content = self._build_system_prompt(lease)
        messages.append({"role": "system", "content": system_content})

        # User message with the task
        user_content = self._build_user_prompt(lease)
        messages.append({"role": "user", "content": user_content})

        return messages

    def build_execution_prompt(
        self,
        lease: Lease,
        step_instructions: str,
        context: str | None = None
    ) -> list[dict[str, Any]]:
        """Build a prompt for executing a cognitive step.

        Args:
            lease: The work lease
            step_instructions: Specific instructions for this step
            context: Optional context from previous steps

        Returns:
            List of messages in OpenAI-compatible format
        """
        messages = []

        # System message
        system_content = self._build_system_prompt(lease, for_execution=True)
        messages.append({"role": "system", "content": system_content})

        # Context from previous steps if available
        if context:
            messages.append({
                "role": "assistant",
                "content": f"Previous step context:\n{context}"
            })

        # Current step instructions
        messages.append({
            "role": "user",
            "content": step_instructions
        })

        return messages

    def _build_system_prompt(self, lease: Lease, for_execution: bool = False) -> str:
        """Build the system prompt portion."""
        parts = [self.profile.system_instructions]

        if self.profile.formatting_constraints:
            parts.append(f"\n\n## Output Formatting\n{self.profile.formatting_constraints}")

        if self.profile.tool_usage_rules:
            parts.append(f"\n\n## Tool Usage\n{self.profile.tool_usage_rules}")

        if not for_execution and self.profile.planning_schema:
            schema_str = self._format_planning_schema()
            parts.append(f"\n\n## Plan Format\n{schema_str}")

        # Add constraints from lease
        if lease.constraints:
            constraints_str = "\n".join(
                f"- {k}: {v}" for k, v in lease.constraints.items()
            )
            parts.append(f"\n\n## Constraints\n{constraints_str}")

        return "\n".join(parts)

    def _build_user_prompt(self, lease: Lease) -> str:
        """Build the user prompt from the lease payload."""
        parts = [f"Task ID: {lease.task_id}", f"Lease ID: {lease.lease_id}", ""]

        # Format the payload
        payload = lease.payload
        if "task" in payload:
            parts.append(f"## Task\n{payload['task']}")

        if "context" in payload:
            parts.append(f"\n## Context\n{payload['context']}")

        if "inputs" in payload:
            inputs_str = "\n".join(
                f"- {k}: {v}" for k, v in payload["inputs"].items()
            )
            parts.append(f"\n## Inputs\n{inputs_str}")

        # Include any other payload fields
        excluded_keys = {"task", "context", "inputs"}
        other_fields = {k: v for k, v in payload.items() if k not in excluded_keys}
        if other_fields:
            other_str = "\n".join(f"- {k}: {v}" for k, v in other_fields.items())
            parts.append(f"\n## Additional Parameters\n{other_str}")

        parts.append("\n## Instructions")
        parts.append("Analyze this task and produce a structured execution plan.")
        parts.append("Each step should be either:")
        parts.append("1. cognitive - Analysis or synthesis requiring reasoning")
        parts.append("2. tool_invocation - A call to one of the available tools")
        parts.append("3. output_generation - Producing the final output")

        return "\n".join(parts)

    def _format_planning_schema(self) -> str:
        """Format the planning schema as instructions."""
        schema = self.profile.planning_schema
        if not schema:
            return "Return your plan as a JSON object with a 'steps' array."

        # Convert schema to human-readable format
        parts = ["Your plan should be a JSON object with the following structure:"]
        parts.append("```json")
        parts.append("{")
        parts.append('  "steps": [')
        parts.append("    {")
        parts.append('      "step_number": 1,')
        parts.append('      "step_type": "cognitive|tool_invocation|output_generation",')
        parts.append('      "description": "What this step does",')
        parts.append('      "tool_name": "mcp.call or artifact.write (if tool_invocation)",')
        parts.append('      "tool_params": { "...": "..." },')
        parts.append('      "instructions": "Detailed instructions (if cognitive)"')
        parts.append("    }")
        parts.append("  ],")
        parts.append('  "summary": "Brief summary of the plan"')
        parts.append("}")
        parts.append("```")

        return "\n".join(parts)


def build_tool_prompt(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build messages containing tool results to return to the AI.

    Args:
        tool_results: List of tool call results

    Returns:
        Messages in OpenAI-compatible format with tool results
    """
    messages = []

    for result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": result.get("call_id", ""),
            "content": str(result.get("result", result.get("error", "No result")))
        })

    return messages
