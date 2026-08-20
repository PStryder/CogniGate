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

Environment variables (prefix `COGNIGATE_`). Generated from the `Settings`
class; MetaGate bootstrap variables are documented in their own section below.

`COGNIGATE_API_KEY` is **required** when `COGNIGATE_REQUIRE_AUTH=true` and `COGNIGATE_ALLOW_INSECURE_DEV=false`; startup fails without it. Separately, `COGNIGATE_ASYNCGATE_AUTH_TOKEN` is required unless `COGNIGATE_STANDALONE_MODE=true`.

See `.env.example` for a working starting point.

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_HOST` | `0.0.0.0` | Bind address |
| `COGNIGATE_PORT` | `8000` | Bind port |
| `COGNIGATE_WORKER_ID` | `cognigate-worker-1` | Worker identifier, used when leasing from AsyncGate |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_ALLOW_INSECURE_DEV` | `false` | Allow unauthenticated access (dev only) |
| `COGNIGATE_API_KEY` | *(empty)* | API key for REST endpoint authentication |
| `COGNIGATE_REQUIRE_AUTH` | `true` | Require authentication for REST endpoints |

### Upstream services

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_ASYNCGATE_AUTH_TOKEN` | *(empty)* | Auth token presented to AsyncGate |
| `COGNIGATE_ASYNCGATE_ENDPOINT` | `http://localhost:8080/mcp` | AsyncGate MCP endpoint |
| `COGNIGATE_ASYNCGATE_TENANT_ID` | `default` | AsyncGate tenant identifier |

### AI and cognition

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_AI_API_KEY` | *(empty)* | AI provider key |
| `COGNIGATE_AI_ENDPOINT` | `https://openrouter.ai/api/v1` | AI provider endpoint (OpenAI-compatible) |
| `COGNIGATE_AI_MAX_TOKENS` | `4096` | Maximum tokens per completion |
| `COGNIGATE_AI_MODEL` | `anthropic/claude-3-opus` | Model used for cognition |
| `COGNIGATE_AI_PROVIDER` | `openrouter` | AI provider: openrouter | stub |
| `COGNIGATE_AI_REQUIRE_REAL` | `false` | Refuse to start unless a real AI provider is configured |

### Storage and sinks

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_CONFIG_DIR` | `Path('/etc/cognigate')` | Directory holding `mcp.yaml` |
| `COGNIGATE_PLUGINS_DIR` | `Path('/etc/cognigate/plugins')` | Directory scanned for sink plugins |
| `COGNIGATE_PROFILES_DIR` | `Path('/etc/cognigate/profiles')` | Directory holding instruction profiles |
| `COGNIGATE_RECEIPT_STORAGE_DIR` | `Path('./receipts')` | Directory for receipt storage (standalone mode) |

### Rate limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `COGNIGATE_RATE_LIMIT_REQUESTS_PER_MINUTE` | `50` | Rate limit per minute |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_CORS_ALLOW_CREDENTIALS` | `true` | Allow credentials in CORS requests |
| `COGNIGATE_CORS_ALLOWED_HEADERS` | `['Authorization', 'Content-Type', 'X-Tenant-ID']` | Allowed request headers |
| `COGNIGATE_CORS_ALLOWED_METHODS` | `['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']` | Allowed HTTP methods |
| `COGNIGATE_CORS_ALLOWED_ORIGINS` | `['http://localhost:3000', 'http://localhost:8080']` | Allowed CORS origins (explicit allowlist for security) |

### Behaviour and limits

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_JOB_TIMEOUT` | `300` | Per-job timeout in seconds |
| `COGNIGATE_MAX_CONCURRENT_JOBS` | `1` | Maximum jobs executed at once |
| `COGNIGATE_MAX_RETRIES` | `3` | Retries for a failed tool call |
| `COGNIGATE_POLLING_INTERVAL` | `5.0` | Seconds between AsyncGate polls |
| `COGNIGATE_STANDALONE_MODE` | `true` | Run in standalone mode without AsyncGate |

## Standalone Mode

Standalone mode disables AsyncGate polling and stores receipts locally.
All requests still go through `/mcp`.

To enable:
```bash
COGNIGATE_STANDALONE_MODE=true
```

## Receipts

CogniGate emits LegiVellum receipts to ReceiptGate when configured.
CogniGate does not write to ReceiptGate. It is a worker, and a worker does
not mint obligations: AsyncGate holds the lease and proposes acceptance and
completion on the worker's behalf.

## MCP Tools

What CogniGate advertises to *callers* on `/mcp`. Distinct from the tool surface
below, which is what CogniGate advertises to the *model*. This is the full set
reported by `tools/list`.

Planning:
- `cognigate.plan` — decompose an intent and return the plan, executing nothing

Job execution:
- `cognigate.execute_job` — execute a job synchronously
- `cognigate.submit_job` — submit a job for background execution
- `cognigate.cancel_job` — cancel a running job

AsyncGate polling:
- `cognigate.polling_start` — begin leasing work from AsyncGate
- `cognigate.polling_stop` — stop leasing

A freshly started CogniGate holds no leases until `polling_start` is called.
That is deliberate: a cognitive worker should not begin consuming obligations
merely because its process exists.

Receipts (standalone mode):
- `cognigate.list_receipts` — list recent receipts
- `cognigate.get_receipt` — fetch a receipt by lease id

Configuration discovery:
- `cognigate.list_profiles` — instruction profiles available
- `cognigate.list_sinks` — output sinks available
- `cognigate.list_mcp_adapters` — MCP adapters available

Health and status:
- `cognigate.health`, `cognigate.health_detailed`, `cognigate.ready`,
  `cognigate.live`, `cognigate.metrics`

## Tool Surface (model-facing)

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
