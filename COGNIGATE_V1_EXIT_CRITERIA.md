Gate v1 Exit Criteria

Component: CogniGate
Repo: https://github.com/PStryder/CogniGate
Owner: Technomancy Labs
Target tag: cognigate-v1.0.0
Date locked: 2026-02-03

Definition of Done

1) Build & Run

- [x] One-command local run exists (`run_local.sh`, `run_local.ps1`).
- [x] Cold start succeeds (import + startup path).
- [x] Health endpoint returns OK (`cognigate.health` MCP tool).
- [x] Config documented (README + `.env.example`, `.env.standalone.example`).
- [ ] Container build verified (`docker build -t cognigate .`).

Artifacts:
- Run instructions: `CogniGate/README.md`
- Example env: `CogniGate/.env.example`, `CogniGate/.env.standalone.example`

2) API & Contract Stability

- [x] MCP tool surface is the v1 contract (`/mcp`, tools/list + tools/call).
- [x] Request/response schemas are stable and in code (`src/cognigate/api.py`).
- [x] Error model is JSON-RPC error envelope.
- [x] REST endpoints removed; MCP-only.

Notes on v1 contract limitations:
- CogniGate is a worker, not an obligation owner; receipts are minted for leases.
- AsyncGate polling is optional (standalone mode supported).

3) Canonical Principals (String IDs)

- [ ] Canonical principal constants not defined (worker uses `worker_id` + `principal_ai` from lease).
- [ ] Define SYSTEM/SERVICE principal IDs if required by stack policy.

4) Receipt Model Invariants

- [x] Emits accepted + complete receipts via ReceiptGate (see `leasing.py`).
- [x] Idempotency enforced per lease in executor (`_completed_leases` cache).
- [x] Escalation path is supported via receipt emission when configured.

5) Persistence & Migration

- [x] No DB dependency required for core worker loop (optional receipts store in standalone mode).

DB notes:
- Storage engine: filesystem (standalone receipts), external ReceiptGate for ledger
- Migration tool: N/A

6) Core Behavioral Guarantees (Standalone)

Golden path:
lease claim → accepted receipt → execute → artifact sink → complete receipt.

- [x] Golden path demo script exists (`scripts/golden_path.py`).
- [x] Artifacts delivered via sinks (file/stdout/MCP sink).

7) Test Requirements

- [x] Unit tests for prompt, plugins, models, security.
- [x] Security regressions for prompt injection + plugin path safety.

Test command:
`pytest`

8) Observability & Debuggability

- [x] Logs include lease_id/task_id/worker_id.
- [x] MCP health surface exists.
- [x] Metrics initialized (`metrics.py`).

9) v1 Lock Rules

Frozen at tag:
- MCP tool surface and schemas
- Receipt emission behavior (accepted/complete)
- Sink contract (ArtifactPointer schema)

10) Open Issues / Deferred Work

- [ ] MetaGate bootstrap integration not implemented.
- [ ] DepotGate wiring via MCP sink must be documented and validated.
- [ ] Container build verification.
- [ ] Tag cognigate-v1.0.0 after sign-off.

Sign-off

- Owner sign-off: pending
- Integration readiness confirmed: pending
- Tag created: pending
