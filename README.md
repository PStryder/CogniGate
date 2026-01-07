# CogniGate

CogniGate is a leased cognitive execution worker.

It performs bounded, tool-mediated AI cognition on behalf of other systems, materializes durable artifacts, and reports lifecycle state through receipts.

CogniGate does not think for itself.
It executes cognition under lease, with explicit constraints, explicit tools, and explicit outputs.

## What CogniGate Does

- Accepts leased work from AsyncGate
- Constructs prompts from static instruction profiles and job-scoped payloads
- Produces a machine-readable plan (advisory, not authoritative)
- Executes cognition step-by-step using a minimal, advertised tool surface
- Delivers outputs to explicitly defined sinks
- Reports progress and completion via receipts, not logs

All cognition is:
- Job-scoped
- Stateless
- Externally materialized
- Receipted at every state transition

## What CogniGate Is Not

CogniGate intentionally does not:
- Maintain conversation or memory
- Own goals or intent
- Decide where outputs go
- Expose third-party APIs directly to models
- Store or emit full reasoning chains
- Operate as a chatbot or assistant

These exclusions are design constraints, not omissions.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for deployment)
- AsyncGate instance (for work leasing)
- AI provider credentials (e.g., OpenRouter)

### Installation

```bash
pip install -e ".[dev]"
```

### Configuration

Environment variables (prefix `COGNIGATE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ASYNCGATE_ENDPOINT` | http://localhost:8080 | AsyncGate API endpoint |
| `ASYNCGATE_AUTH_TOKEN` | - | Authentication token for AsyncGate |
| `AI_ENDPOINT` | https://openrouter.ai/api/v1 | AI provider endpoint |
| `AI_API_KEY` | - | API key for AI provider |
| `AI_MODEL` | anthropic/claude-3-opus | AI model to use |
| `AI_MAX_TOKENS` | 4096 | Maximum tokens for AI responses |
| `POLLING_INTERVAL` | 5.0 | Polling interval in seconds |
| `MAX_CONCURRENT_JOBS` | 1 | Maximum concurrent job executions |
| `JOB_TIMEOUT` | 300 | Job timeout in seconds |
| `MAX_RETRIES` | 3 | Maximum retries for failed tool calls |
| `CONFIG_DIR` | /etc/cognigate | Configuration directory |
| `PLUGINS_DIR` | /etc/cognigate/plugins | Plugins directory |
| `PROFILES_DIR` | /etc/cognigate/profiles | Instruction profiles directory |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `WORKER_ID` | cognigate-worker-1 | Worker identifier |

### Running

```bash
# Start the server
uvicorn cognigate.api:app --host 0.0.0.0 --port 8000
```

## API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/ready` | GET | Readiness check for Kubernetes |
| `/v1/jobs` | POST | Submit a job directly (testing/local use) |
| `/v1/polling/start` | POST | Start polling AsyncGate for work |
| `/v1/polling/stop` | POST | Stop polling AsyncGate |
| `/v1/config/profiles` | GET | List available instruction profiles |
| `/v1/config/sinks` | GET | List available output sinks |
| `/v1/config/mcp` | GET | List available MCP adapters |

## Tool Surface

CogniGate advertises a minimal tool surface to the AI model:

### `mcp_call`

Call a method on an MCP (Model Context Protocol) server.

Parameters:
- `server` (required): Name of the MCP server to call
- `method` (required): MCP method to invoke (e.g., 'resources/read', 'tools/call')
- `params` (optional): Parameters for the MCP method

### `artifact_write`

Write an artifact to the configured output sink.

Parameters:
- `content` (required): Content to write to the artifact
- `metadata` (optional): Additional metadata for the artifact

## Bootstrap Configuration

On startup, CogniGate loads configuration from the filesystem:

### Instruction Profiles

YAML files in `PROFILES_DIR` defining:
- `name`: Profile identifier
- `system_instructions`: System prompt instructions
- `formatting_constraints`: Output formatting rules
- `planning_schema`: Planning output schema
- `tool_usage_rules`: Rules for tool usage

### MCP Endpoints

YAML configuration in `CONFIG_DIR/mcp.yaml`:
```yaml
mcp_endpoints:
  - name: github
    endpoint: https://mcp.example.com/github
    auth_token: optional-token
    read_only: true
    enabled: true
```

## Plugin Architecture

### Sink Plugins

Output sinks can be added by:
1. Dropping a Python module into the plugins directory
2. Restarting the service

Sinks self-register with:
- `sink_id`
- `config_schema`
- `deliver()` handler

### MCP Adapters

MCP adapters connect to upstream MCP servers with:
- Configurable endpoints
- Optional authentication
- Read-only mode support

## Design Principles

- Cognition under lease
- Artifacts over messages
- Receipts over logs
- Execution over intent
- Boring in the right places

CogniGate exists to make AI cognition interruptible, auditable, recoverable, and safe to embed in real systems—without pretending it's a mind.

## License

MIT
