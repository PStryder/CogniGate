"""Prompt construction for CogniGate."""

import re
from typing import Any

from .config import InstructionProfile
from .models import Lease


class PromptBuilder:
    """Builds prompts for AI invocation from leases and profiles."""

    def __init__(self, profile: InstructionProfile):
        self.profile = profile

    def _sanitize_user_content(self, content: str, max_length: int = 50000) -> str:
        """Sanitize user-provided content to prevent prompt injection.
        
        Removes common injection patterns and limits length to prevent
        context overflow or jailbreak attempts.
        
        Args:
            content: User-provided content to sanitize
            max_length: Maximum allowed length (default 50000 chars)
            
        Returns:
            Sanitized content with injection attempts redacted
        """
        if not isinstance(content, str):
            content = str(content)
        
        # Remove common prompt injection markers
        dangerous_patterns = [
            (r'SYSTEM\s+OVERRIDE', '[REDACTED_INJECTION]'),
            (r'IGNORE\s+(?:ALL\s+)?PREVIOUS', '[REDACTED_INJECTION]'),
            (r'YOU\s+ARE\s+NOW', '[REDACTED_INJECTION]'),
            (r'</s>', '[REDACTED_TAG]'),  # XML/special token injection
            (r'\[/?INST\]', '[REDACTED_TAG]'),  # Llama-style markers
            (r'<\|.*?\|>', '[REDACTED_TAG]'),  # Special tokens
            (r'###\s*System', '[REDACTED_HEADER]'),  # System header injection
        ]
        
        sanitized = content
        for pattern, replacement in dangerous_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        # Limit length to prevent context overflow
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "...[TRUNCATED_FOR_LENGTH]"
        
        return sanitized

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
        
        SECURITY: Context from previous steps is sanitized to prevent
        injection attacks via accumulated outputs.

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

        # Context from previous steps if available (sanitize it)
        if context:
            sanitized_context = self._sanitize_user_content(context, max_length=10000)
            messages.append({
                "role": "assistant",
                "content": f"Previous step context:\n{sanitized_context}"
            })

        # Current step instructions (sanitize in case they came from plan)
        sanitized_instructions = self._sanitize_user_content(step_instructions, max_length=5000)
        messages.append({
            "role": "user",
            "content": sanitized_instructions
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

        # Add constraints from lease (sanitize user-provided values)
        if lease.constraints:
            sanitized_constraints = []
            for k, v in lease.constraints.items():
                sanitized_key = self._sanitize_user_content(str(k), max_length=200)
                sanitized_val = self._sanitize_user_content(str(v), max_length=500)
                sanitized_constraints.append(f"- {sanitized_key}: {sanitized_val}")
            parts.append(f"\n\n## Constraints\n" + "\n".join(sanitized_constraints))

        return "\n".join(parts)

    def _build_user_prompt(self, lease: Lease) -> str:
        """Build the user prompt from the lease payload.
        
        SECURITY: All user-provided content is sanitized to prevent
        prompt injection attacks.
        """
        # Use XML delimiters to clearly separate user input from instructions
        parts = ["<user_input>"]
        parts.append(f"<task_id>{lease.task_id}</task_id>")
        parts.append(f"<lease_id>{lease.lease_id}</lease_id>")

        # Format the payload (sanitize all user-provided fields)
        payload = lease.payload
        if "task" in payload:
            sanitized_task = self._sanitize_user_content(payload['task'])
            parts.append(f"\n<task>\n{sanitized_task}\n</task>")

        if "context" in payload:
            sanitized_context = self._sanitize_user_content(payload['context'])
            parts.append(f"\n<context>\n{sanitized_context}\n</context>")

        if "inputs" in payload:
            parts.append("\n<inputs>")
            for k, v in payload["inputs"].items():
                sanitized_key = self._sanitize_user_content(str(k), max_length=200)
                sanitized_val = self._sanitize_user_content(str(v), max_length=5000)
                parts.append(f"  <input name=\"{sanitized_key}\">{sanitized_val}</input>")
            parts.append("</inputs>")

        # Include any other payload fields
        excluded_keys = {"task", "context", "inputs"}
        other_fields = {k: v for k, v in payload.items() if k not in excluded_keys}
        if other_fields:
            parts.append("\n<additional_parameters>")
            for k, v in other_fields.items():
                sanitized_key = self._sanitize_user_content(str(k), max_length=200)
                sanitized_val = self._sanitize_user_content(str(v), max_length=5000)
                parts.append(f"  <param name=\"{sanitized_key}\">{sanitized_val}</param>")
            parts.append("</additional_parameters>")

        parts.append("\n</user_input>")
        
        # Add instructions OUTSIDE user_input tags
        parts.append("\n## Instructions")
        parts.append("Analyze the task provided in the <user_input> section above.")
        parts.append("Produce a structured execution plan with steps that are either:")
        parts.append("1. cognitive - Analysis or synthesis requiring reasoning")
        parts.append("2. tool_invocation - A call to one of the available tools")
        parts.append("3. output_generation - Producing the final output")
        parts.append("\nIMPORTANT: Only work with content inside the <user_input> tags.")

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
