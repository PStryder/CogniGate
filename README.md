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
- AI provider credentials (OpenRouter or OpenAI-compatible)
- AsyncGate instance (optional, for leased work)
- ReceiptGate instance (optional, for LegiVellum receipts)
- Docker Desktop (optional, for compose)

### Install

```bash
pip install -e ".[dev]"
```

### Run local (Docker compose)

```bash
./run_local.sh
# or
.\run_local.ps1
```

### Run local (Python)

```bash
# set required env vars, then
python -m cognigate.main
```

### MCP API (canonical HTTP surface)

CogniGate exposes a single JSON-RPC endpoint at `/mcp`.

List tools:
```bash
curl -s http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cg_your-secret-api-key" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Execute a job synchronously:
```bash
curl -s http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cg_your-secret-api-key" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cognigate.execute_job","arguments":{"task_id":"demo-001","payload":{"task":"Summarize the text","context":"LegiVellum uses receipts for coordination."},"profile":"default","sink_config":{"sink_id":"stdout"}}}}'
```

Plan an intent without executing it:
```bash
curl -s http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "X-API-Key: cg_your-secret-api-key" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"cognigate.plan","arguments":{"intent":"research competitors and draft a summary report","task_type":"general","profile":"default"}}}'
```

`cognigate.plan` runs the planning phase and stops: nothing is executed, no
tools are called, no artifacts are written, and no lease is taken. It exists
for DeleGate, which holds the planning authority but no cognition of its own —
DeleGate asks what an intent decomposes into and then mints the obligations
itself, so CogniGate returns a plan document and never mints anything.

The response includes `is_stub` and `model`. When the stub provider answered,
these read `true` and `stub/echo`, because a caller cannot inspect this process
to find out whether any reasoning happened.

For local development you can set `COGNIGATE_ALLOW_INSECURE_DEV=true` to bypass auth.

### Golden path script

```bash
python scripts/golden_path.py --endpoint http://localhost:8000/mcp --api-key cg_your-secret-api-key
```

## Configuration

Environment variables (prefix `COGNIGATE_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `STANDALONE_MODE` | true | Run without AsyncGate polling (local dev) |
| `RECEIPT_STORAGE_DIR` | ./receipts | Receipt storage directory (standalone mode) |
| `ASYNCGATE_ENDPOINT` | http://localhost:8080/mcp | AsyncGate MCP endpoint |
| `ASYNCGATE_AUTH_TOKEN` | - | AsyncGate auth token |
| `ASYNCGATE_TENANT_ID` | default | Tenant identifier for AsyncGate |
| `RECEIPTGATE_ENDPOINT` | - | ReceiptGate MCP endpoint |
| `RECEIPTGATE_AUTH_TOKEN` | - | ReceiptGate auth token |
| `RECEIPTGATE_EMIT_RECEIPTS` | true | Emit LegiVellum receipts |
| `AI_ENDPOINT` | https://openrouter.ai/api/v1 | AI provider endpoint |
| `AI_API_KEY` | - | AI provider key |
| `AI_MODEL` | anthropic/claude-3-opus | AI model |
| `AI_MAX_TOKENS` | 4096 | Max tokens |
| `POLLING_INTERVAL` | 5.0 | Polling interval in seconds |
| `MAX_CONCURRENT_JOBS` | 1 | Max concurrent jobs |
| `JOB_TIMEOUT` | 300 | Job timeout in seconds |
| `MAX_RETRIES` | 3 | Max tool retries |
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `WORKER_ID` | cognigate-worker-1 | Worker identifier |
| `API_KEY` | - | API key for MCP requests |
| `REQUIRE_AUTH` | true | Require API key for MCP |
| `ALLOW_INSECURE_DEV` | false | Disable auth checks (dev only) |

See `.env.example` and `.env.standalone.example` for a complete set.

## Standalone Mode

Standalone mode disables AsyncGate polling and stores receipts locally.
All requests still go through `/mcp`.

To enable:
```bash
COGNIGATE_STANDALONE_MODE=true
```

## Receipts

CogniGate emits LegiVellum receipts to ReceiptGate when configured.
Set `COGNIGATE_RECEIPTGATE_ENDPOINT` and `COGNIGATE_RECEIPTGATE_AUTH_TOKEN` to enable.

## Tool Surface

CogniGate advertises a minimal tool surface to the AI model:

### `mcp_call`

Call a method on an MCP (Model Context Protocol) server.

Parameters:
- `server` (required): Name of the MCP server to call
- `method` (required): MCP method to invoke (e.g., `resources/read`, `tools/call`)
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

YAML configuration in `$COGNIGATE_CONFIG_DIR/mcp.yaml` (the directory named by `COGNIGATE_CONFIG_DIR`; `/etc/cognigate` in the demo stack):
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

CogniGate exists to make AI cognition interruptible, auditable, recoverable, and safe to embed in real systems without pretending it is a mind.

## MetaGate Bootstrap

On startup this gate asks MetaGate for the topology it belongs to and fills in
endpoints the operator did not configure. It resolves: `receiptgate` → `receiptgate_endpoint`, `asyncgate` → `asyncgate_endpoint`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `COGNIGATE_METAGATE_ENDPOINT` | *(unset)* | MetaGate MCP endpoint. Unset disables bootstrap; the gate starts on configured values alone. |
| `COGNIGATE_METAGATE_API_KEY` | *(unset)* | Credential presented to MetaGate |
| `COGNIGATE_METAGATE_COMPONENT_KEY` | `cognigate` | Which component in the manifest this process is |
| `COGNIGATE_METAGATE_BOOTSTRAP_TIMEOUT_SECONDS` | `5.0` | Per-call timeout |

Bootstrap never prevents startup. Every failure — unreachable, timeout, auth
rejected, no binding, malformed packet — degrades to a logged warning and
"carry on with configured values", because a bootstrap authority that can take
the mesh down would be a hidden master. Explicit configuration always wins;
bootstrap fills gaps and logs when the mesh disagrees rather than overriding.

See `LegiVellum/docs/canonical/metagate.bootstrap.md` for the full contract.

## License

MIT
