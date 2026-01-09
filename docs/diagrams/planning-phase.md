# Planning Phase

This diagram shows how CogniGate handles planning profiles and system prompt construction.

```mermaid
sequenceDiagram
    autonumber
    participant L as Lease
    participant JE as JobExecutor
    participant B as Bootstrap
    participant P as InstructionProfile
    participant AI as AI Provider

    Note over L: Lease contains profile name and payload

    JE->>B: get_profile(lease.profile)

    alt Profile found
        B-->>JE: InstructionProfile
    else Profile not found
        B->>B: get_default_profile()
        B-->>JE: Default profile (or None)
    end

    Note over JE: Build system prompt

    JE->>P: Read system_instructions
    P-->>JE: Base instructions

    JE->>P: Read formatting_constraints
    P-->>JE: Output format rules

    JE->>P: Read tool_usage_rules
    P-->>JE: Tool guidance

    JE->>P: Read planning_schema
    P-->>JE: JSON schema for planning output

    JE->>JE: Combine into system prompt

    Note over JE: System prompt structure:
    Note over JE: 1. System instructions
    Note over JE: 2. Tool definitions
    Note over JE: 3. Formatting constraints
    Note over JE: 4. Tool usage rules
    Note over JE: 5. Planning schema (if present)

    JE->>JE: Extract user message from payload

    JE->>AI: Create chat completion
    Note over AI: messages = [system, user]
    Note over AI: tools = available_tools

    AI-->>JE: Initial response

    alt Has planning_schema
        Note over JE: Response must match schema
        JE->>JE: Validate against schema
    end

    JE->>JE: Continue execution loop
```

## Profile Configuration

Instruction profiles are YAML files in the profiles directory:

```yaml
# /etc/cognigate/profiles/analyst.yaml
name: analyst
system_instructions: |
  You are a data analyst assistant. Your role is to help
  users analyze data and generate insights.

formatting_constraints: |
  - Provide structured responses
  - Include confidence levels for conclusions
  - Cite data sources when applicable

tool_usage_rules: |
  - Prefer read-only operations
  - Validate inputs before tool calls
  - Explain tool results to the user

planning_schema:
  type: object
  properties:
    analysis_plan:
      type: array
      items:
        type: object
        properties:
          step: { type: string }
          tools_needed: { type: array, items: { type: string } }
          expected_output: { type: string }
    confidence: { type: number, minimum: 0, maximum: 1 }
  required: [analysis_plan]
```

## Bootstrap Loading

```mermaid
flowchart TD
    A[Application Start] --> B[Create Bootstrap]
    B --> C{profiles_dir exists?}
    C -->|Yes| D[Scan *.yaml/*.yml files]
    C -->|No| E[Empty profiles dict]
    D --> F[Parse each file]
    F --> G[Create InstructionProfile]
    G --> H[Store in profiles dict]
    H --> I{More files?}
    I -->|Yes| F
    I -->|No| J[Load MCP config]
    J --> K[Bootstrap ready]
    E --> J
```
