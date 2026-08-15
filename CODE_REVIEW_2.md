<!-- Generated 2026-08-15. Stack-level context: ../LV_STACK_REVIEW.md -->

> **Review 2 — CogniGate**
> Part of a full-stack review of LV_Stack (11 repos, ~97k LOC) conducted 2026-08-15.
> Stack-wide findings that affect this repo but are not fixable inside it are in
> `../LV_STACK_REVIEW.md` and `../_CROSS_REPO_ANALYSIS.md`. Read the stack report first —
> several findings below have a shared root cause.

---

# CogniGate — Code Review

Reviewed: `/home/claude/lv/CogniGate/` @ working tree of 2026-08-15. ~10.1k LOC Python.
Normative sources cross-checked: `LegiVellum/docs/canonical/receipt.schema.v1.json`,
`receipt.rules.md`, `LegiVellum/docs/canonical/CogniGate/alignment.md`,
`ReceiptGate/src/receiptgate/validation_v1.py`, `Gate v1 Exit Criteria Template.txt`.

## Verdict

CogniGate is the best-written repo-shaped thing in this stack — clean layering, real
prompt-injection hardening, an honest stub provider, and a genuinely well-tested
`plan_only` path — sitting on top of a role definition it does not satisfy. The
canonical alignment doc says "executes cognition **without side effects**"; the code
lets any authenticated caller name an absolute `base_path` and have CogniGate `mkdir -p`
and write a model-generated file anywhere the process can reach, including its own
plugin directory (which is `exec_module`'d at next boot). Worse for the protocol: every
successful artifact-producing job emits a `complete` receipt with `outcome_kind:
"artifact_pointer"` and `artifact_mime: "NA"`, which ReceiptGate's own validator rejects
by schema — and the rejection is logged at `debug` and discarded, so the happy path
silently never closes its obligation. Both MCP execution tools (`execute_job`,
`submit_job`) emit no LegiVellum receipts at all. **Not v1-taggable.** Fix the artifact
write boundary, the receipt conformance, and the missing receipts on the MCP path;
everything else is ordinary hardening.

## Exit Criteria Scorecard

| § | Section | Verdict | Justification |
|---|---|---|---|
| 1 | Build & Run | **PARTIAL** | `run_local.sh` + CI `docker build` are real, but `docker-compose.yaml` cannot start (no `COGNIGATE_API_KEY`/`AI_API_KEY` → `Settings()` raises), and `k8s/` crash-loops (`readOnlyRootFilesystem: true` + image default `STANDALONE_MODE=true` → `ReceiptStore.mkdir` on a read-only path). |
| 2 | API & Contract Stability | **PARTIAL** | MCP-only surface with schemas in code (`api.py:645`), consistent JSON-RPC envelope — but the catch-all handler returns raw `str(exc)` to callers (`api.py:766`) and `submit_job` returns `{"status":"accepted"}` for a job that emits no receipt and stores none. |
| 3 | Canonical Principals | **FAIL** | Self-admitted in `COGNIGATE_V1_EXIT_CRITERIA.md:36`. No `SYSTEM_PRINCIPAL_ID`/`SERVICE_PRINCIPAL_ID` constants exist anywhere; `from_principal == for_principal == worker_id` when the lease carries no `principal_ai` (`legivellum_receipts.py:99`), which makes the worker its own requester. |
| 4 | Receipt Model Invariants | **FAIL** | No `TERMINAL_RECEIPT_TYPES` set; cancellation emits `status=failure`, never `canceled` (`executor.py:284`); artifact-bearing `complete` receipts violate the canonical schema (`legivellum_receipts.py:72`); `retryable` is hardcoded `False` (`leasing.py:415`); receipts are emitted only on the AsyncGate poll path. |
| 5 | Persistence & Migration | **PASS** | No DB. Filesystem receipt store + DLQ, both created on demand, no migration surface. (Caveat: `ReceiptStore.delete` exists and contradicts immutability — `receipts.py:150`; DLQ is `emptyDir` in k8s.) |
| 6 | Core Behavioral Guarantees | **FAIL** | The stated golden path is "lease claim → accepted receipt → execute → artifact sink → complete receipt", but `scripts/golden_path.py:61` calls `cognigate.execute_job`, which takes no lease and emits no accepted/complete receipt. `COGNIGATE_JOB_TIMEOUT` is documented in README, `.env`, compose and k8s and is **never read** — no deadlock protection on long-running operations. |
| 7 | Test Requirements | **PARTIAL** | 12 files; prompt sanitization, plugin permissions, and `plan_only` non-execution are properly asserted. Zero tests touch `api.py`, `auth.py`, `legivellum_receipts.py`, `receiptgate_client.py`, `receipts.py`. None of the template's required regressions (terminal-type gating, cancel closes, ack/progress doesn't close, dedupe) exist. |
| 8 | Observability | **PASS** | `JobContext` binds `task_id`/`lease_id`/`worker_id` into every log line (`observability.py:127`), Prometheus metrics with a bounded MCP snapshot, `health`/`health_detailed`/`ready`/`live`. `is_stub` surfaced to callers is a genuinely good touch. |
| 9 | v1 Lock Rules | **PARTIAL** | The frozen set is declared (MCP surface, receipt emission behaviour, `ArtifactPointer`) — but freezing "receipt emission behavior" today freezes a shape ReceiptGate rejects. |
| 10 | Open Issues | **PARTIAL** | The list is stale: it claims MetaGate bootstrap is not implemented (it is — `metagate_client.py`) and container build unverified (CI does it), while omitting the actual blockers found below. |

**Blunt v1 verdict: NOT taggable.** §3, §4 and §6 are hard FAILs, and §4/§6 are the ones
that make the component unsafe to integrate — a caller cannot tell a completed
obligation from a dropped one.

## Side-Effect Freedom Audit

Declared boundary (from `LegiVellum/docs/canonical/CogniGate/alignment.md`): accept
leases from AsyncGate, emit receipts to ReceiptGate, **store outputs in DepotGate and
reference by pointer**, bootstrap from MetaGate, "executes cognition without side
effects or delegation".

| Call site | file:line | In boundary? |
|---|---|---|
| AI provider `POST /chat/completions` | `ai_client.py:118` | ✅ yes — this is the cognition |
| AsyncGate `lease_next` (claim) | `leasing.py:201` | ✅ yes — worker contract |
| AsyncGate `report_progress` | `leasing.py:383` | ✅ yes |
| AsyncGate `complete` (mutates task terminal state) | `leasing.py:401` | ✅ yes — declared in Spec §11 |
| AsyncGate `fail` | `leasing.py:418` | ✅ yes |
| AsyncGate `renew_lease` | `leasing.py:441` | ✅ yes |
| AsyncGate `health` (read) | `api.py:262` | ✅ yes |
| ReceiptGate `submit_receipt` | `receiptgate_client.py:56` | ✅ yes |
| MetaGate bootstrap + `acknowledge_startup` | `metagate_client.py:85,93` | ✅ yes — documented contract |
| Upstream MCP call, method+params chosen **by the model** | `mcp_adapter.py:159` | ⚠️ conditional — safe while `read_only: true` (default); with `read_only: false` the model may issue `tools/call` and `resources/write` to any configured server (`mcp_adapter.py:62-65`). Model-directed external mutation. |
| `MCPSink` → `resources/write` on upstream server | `builtin_sinks.py:161-169` | ⚠️ only in boundary if the target is DepotGate; nothing constrains it, and no DepotGate client exists in this repo |
| `FileSink` `mkdir(parents=True)` on caller-supplied `base_path` | `builtin_sinks.py:59-60` | ❌ **NO — CRITICAL.** Arbitrary directory creation anywhere |
| `FileSink` write of model-generated content | `builtin_sinks.py:86,89` | ❌ **NO — CRITICAL.** Arbitrary file write anywhere |
| `StdoutSink` `print()` | `builtin_sinks.py:213-220` | ✅ yes (process stdout) |
| Standalone `ReceiptStore` mkdir/write | `receipts.py:32,64` | ✅ yes — own state dir |
| `ReceiptStore.delete` (unlink) | `receipts.py:164` | ⚠️ dead code, but deleting receipts contradicts append-only immutability |
| DLQ mkdir + write | `leasing.py:78-81` | ✅ yes — own state dir |
| Plugin `exec_module` of arbitrary `.py` at startup | `plugins/base.py:174` | ⚠️ in boundary by design (operator-owned dir), gated on permissions — but reachable from the FileSink write above |
| Example webhook sink `POST` to caller-supplied URL | `config/plugins/sinks/example_sink.py:71` | ⚠️ registration is commented out, but the shipped example is an SSRF template |

**Bottom line: side-effect freedom is violated.** One CRITICAL (unbounded filesystem
write reachable by any authenticated caller) and two structural conflicts (model-directed
MCP writes; artifact delivery with no DepotGate binding). Nothing in the repo enforces
that a sink target is inside a declared boundary, and no test asserts it.

## Critical & High Findings

### CRITICAL-1 — Any authenticated caller can write files anywhere via `sink_config.base_path` (→ RCE at next restart)
`src/cognigate/plugins/builtin_sinks.py:59`
```python
base_path = Path(config["base_path"])
base_path.mkdir(parents=True, exist_ok=True)
...
file_path = base_path / filename
# SECURITY: Verify resolved path is within base_path (prevent traversal)
if not str(resolved_path).startswith(str(resolved_base)):
```
`config` is `lease.sink_config`, which arrives verbatim from the MCP request
(`api.py:507` → `tools.py:215-216` → `sink.deliver(..., sink_config)`). The traversal
check only proves the *filename* stays under `base_path`; `base_path` itself is
attacker-chosen and unvalidated. The prior review's SEC-001 fix hardened the filename
and left the root open.

**Failure scenario:** `POST /mcp` with a valid API key,
`{"name":"cognigate.execute_job","arguments":{"task_id":"x","payload":{"task":"emit python"},"sink_config":{"sink_id":"file","base_path":"/etc/cognigate/plugins/sinks","filename_template":"evil.py"}}}`.
The model's `artifact_write` content lands at `/etc/cognigate/plugins/sinks/evil.py`
(the Dockerfile `chown`s that tree to the `cognigate` user, so it is writable). On the
next restart `SinkRegistry.discover_plugins` calls `spec.loader.exec_module(module)`
(`plugins/base.py:174`) — arbitrary code execution as the service user. The permission
check at `base.py:85` passes, because the process wrote the file as its own owner with
default `0o755` dirs. Same primitive without the restart: overwrite
`/etc/cognigate/profiles/default.yaml` to rewrite the system prompt of every future job.

**Fix:** resolve `base_path` against an operator-configured allowlist root
(`COGNIGATE_ARTIFACT_ROOT`) and reject anything outside it; use
`resolved.is_relative_to(root)` rather than `str.startswith`; never accept `base_path`
from a request.

---

### HIGH-1 — Every artifact-producing `complete` receipt is rejected by ReceiptGate; the rejection is discarded
`src/cognigate/legivellum_receipts.py:67`
```python
    return {
        "artifact_location": location,
        "artifact_pointer": pointer,
        "artifact_checksum": metadata.get("checksum", "NA"),
        "artifact_size_bytes": metadata.get("size_bytes", 0),
        "artifact_mime": metadata.get("mime", "NA"),
    }
```
No sink populates `metadata["mime"]` — `FileSink` writes `{"filename", "size"}`
(`builtin_sinks.py:97`), `StdoutSink` `{"size"}`, `MCPSink` `{"mcp_server","mcp_result"}`.
So `artifact_mime` is always `"NA"`. Meanwhile `build_receipt` sets
`outcome_kind = "artifact_pointer"` or `"mixed"` whenever artifacts exist
(`legivellum_receipts.py:124-127`). The canonical schema's fourth `allOf` branch
requires, for `phase=complete` with those outcome kinds, that `artifact_mime` is **not**
`"NA"` — and `ReceiptGate/src/receiptgate/validation_v1.py:115` runs exactly that schema
and returns `validation_failed`.

**Failure scenario:** a leased job succeeds and writes one artifact. `_handle_job` builds
the `complete` receipt, `ReceiptGateClient.emit_receipt` gets a JSON-RPC `error`, logs
`receiptgate_receipt_emit_failed` at **warning** and returns `False`
(`receiptgate_client.py:59-62`), and `leasing.py:858` then logs
`receiptgate_receipt_skipped` at **debug**. Nothing raises, nothing retries, nothing
dead-letters. The AsyncGate task is marked complete, but the ledger keeps the obligation
open forever — derived state (`accepted` with no `complete`) says the work is still
running. Also note `artifact_size_bytes` reads `size_bytes` while sinks write `size`, so
it is always 0.

**Fix:** have sinks return a real MIME (`ArtifactPointer.metadata["mime"]`), fall back to
`application/octet-stream` rather than `"NA"`, and treat a ReceiptGate rejection as a
dead-letter event, not a debug log.

---

### HIGH-2 — The MCP execution tools emit no LegiVellum receipts at all
`src/cognigate/api.py:495`
```python
async def execute_job_sync(request: SubmitJobRequest):
    ...
    receipt = await state.job_executor.execute(lease)
    if state.receipt_store:
        state.receipt_store.save(receipt)
    return receipt
```
`_emit_legivellum_receipt` is called only from `leasing.py` (5 sites, all inside
`WorkPoller`). `cognigate.execute_job` and `cognigate.submit_job` — the documented
quick-start path, the golden-path script, and the whole of standalone mode — perform
cognition, write artifacts, and emit **zero** receipts to ReceiptGate. The local
`ReceiptStore` writes CogniGate's private `Receipt` model, not a canonical receipt.

**Failure scenario:** operator runs `scripts/golden_path.py` against a stack with
`COGNIGATE_RECEIPTGATE_ENDPOINT` set. A file is written to disk and a JSON blob is
returned. Nothing in the ledger records that CogniGate accepted or discharged anything —
violating core invariant #2 ("anything accepting responsibility emits an `accepted`
receipt"). Exit-criteria §4's "Emits accepted + complete receipts via ReceiptGate" is
true only for the optional polling path.

---

### HIGH-3 — Concurrent jobs share one artifact list; receipts get the wrong artifacts
`src/cognigate/tools.py:94` / `src/cognigate/executor.py:210`
```python
self._artifacts: list[ArtifactPointer] = []      # tools.py:94, one per process
...
self.tool_executor.clear_artifacts()             # executor.py:210, at each job start
...
artifacts = self.tool_executor.get_artifacts()   # executor.py:241, at each job end
```
One `ToolExecutor` is constructed at startup (`api.py:132`) and shared by every job.
`max_concurrent_jobs` defaults to 1 but `k8s/configmap.yaml:20` sets `2`, and the
`submit_job` path (HIGH-4) has no limit at all.

**Failure scenario:** job A writes artifact `a.txt`; job B starts and calls
`clear_artifacts()`; job B writes `b.txt`; A finishes and reports
`artifact_pointers=[b.txt]` — A's receipt points at another task's output (a provenance
and confidentiality break), while `a.txt` exists on disk referenced by nothing. Reverse
the interleaving and A completes with `artifact_pointers=[]` and
`outcome_kind="response_text"` despite having materialized a file.

**Fix:** make artifact accumulation per-job — carry the list on `ToolContext`, not on the
executor.

---

### HIGH-4 — `submit_job` is fire-and-forget: no reference, no timeout, no concurrency bound, no receipt, exceptions swallowed
`src/cognigate/api.py:537`
```python
    async def run_job():
        receipt = await state.job_executor.execute(lease)
        logger.info(f"Job {lease.task_id} completed with status: {receipt.status}")

    asyncio.create_task(run_job())

    return {"status": "accepted", "lease_id": lease.lease_id, "task_id": lease.task_id}
```
Four defects in five lines: (a) the task object is never stored, so CPython may GC it
mid-flight; (b) no `add_done_callback`, so any exception is only surfaced as an
"exception was never retrieved" warning at GC; (c) `state.settings.max_concurrent_jobs`
is not consulted — 500 `submit_job` calls start 500 concurrent AI jobs; (d) the receipt
is never persisted to `state.receipt_store`, so the `{"status":"accepted", "lease_id":...}`
handshake points at a receipt that `cognigate.get_receipt` will 404 forever.

**Failure scenario:** caller submits 3 background jobs within the rate limit, then polls
`cognigate.get_receipt` with the returned `lease_id`. Every poll returns 404 "Receipt not
found" regardless of whether the job succeeded, failed, or was garbage-collected
half-done. There is no other way to learn the outcome.

---

### HIGH-5 — `COGNIGATE_JOB_TIMEOUT` is documented everywhere and enforced nowhere; plan length is unbounded
`src/cognigate/config.py:92`
```python
    job_timeout: int = Field(default=300)
```
Referenced only by `get_worker_config()` (`config.py:259`), whose `WorkerConfig` is never
consumed. `JobExecutor.execute` has no `asyncio.wait_for`. `_planning_phase`
(`executor.py:356`) iterates `plan_data.get("steps", [])` with no cap, and each cognitive
step is a full provider round-trip (`executor.py:494`).

**Failure scenario:** a model returns a 400-step plan (or a payload nudges it there).
CogniGate makes 400 sequential completions at up to 45 s read timeout each — up to five
hours of wall clock on one lease. The AsyncGate lease heartbeat keeps renewing it
(`leasing.py:634`), so nothing times it out; `max_concurrent_jobs=1` means the worker is
wedged; and the provider bill is unbounded because only *per-call* `max_tokens` is
capped, never per-job token spend. Exit criteria §6 "Timeouts and long-running operations
behave (no deadlocks / no infinite wait)" is not met.

---

### HIGH-6 — Unparseable plan degrades to an empty plan and the job reports `complete`/success
`src/cognigate/ai_client.py:227`
```python
        # Return empty plan
        return {"steps": [], "summary": "Failed to parse plan"}
```
`generate_plan` → `_extract_json` returns this on total parse failure. `_planning_phase`
builds `ExecutionSteps(steps=[])`, `_execution_loop` iterates nothing, and `execute()`
falls through to the success branch at `executor.py:257`, emitting
`JobStatus.COMPLETE` with `summary="Executed 0 steps. Plan: Failed to parse plan Outputs: 0"`
and `status="success"` on the canonical receipt.

**Failure scenario:** provider returns prose (common when `response_format:
json_object` is unsupported by the routed model — OpenRouter silently degrades for some
models). The obligation is closed as **successfully discharged** having done nothing. A
zero-step plan must be an `ExecutionError`, not a completion.

---

### HIGH-7 — `docker-compose.yaml` cannot start, and `k8s/` crash-loops
`docker-compose.yaml:8-16` sets `COGNIGATE_ASYNCGATE_*` and `COGNIGATE_AI_API_KEY=${AI_API_KEY:-}`
but never `COGNIGATE_API_KEY`, and never `COGNIGATE_STANDALONE_MODE=false`.
`Settings()` at `api.py:81` therefore raises twice over: `validate_ai_api_key`
(`config.py:220`, empty key with `ai_provider=openrouter`) and `validate_api_key`
(`config.py:237`, `require_auth=True` + empty `api_key` + `allow_insecure_dev=False`).
No `.env` is baked into the image and compose declares no `env_file`.

**Failure scenario:** `docker compose up` → immediate `ValidationError` traceback,
`restart: unless-stopped` loops forever. And if the key issue were fixed, the file still
would not do what it advertises: the Dockerfile pins `ENV COGNIGATE_STANDALONE_MODE=true`
(`Dockerfile:29`) and compose never overrides it, so the AsyncGate endpoint it configures
is dead config — `WorkPoller` is never constructed (`api.py:147`).

Same class of bug in k8s: `k8s/configmap.yaml` also omits `COGNIGATE_STANDALONE_MODE`, so
the pod runs standalone with `receipt_storage_dir=/var/lib/cognigate/receipts` (Dockerfile
default) while `deployment.yaml:138` sets `readOnlyRootFilesystem: true` and mounts a
volume only at `/var/lib/cognigate/dlq`. `ReceiptStore._ensure_directory`
(`receipts.py:32`) raises `OSError: Read-only file system` inside `lifespan` → CrashLoopBackOff.

---

### HIGH-8 — Cancellation is reported as failure; there is no `canceled` terminal outcome
`src/cognigate/models.py:12` defines `JobStatus` as `pending|running|complete|failed` —
no cancelled state. `executor.py:278`:
```python
                except JobCancelledError as e:
                    ...
                    status=JobStatus.FAILED,
                    error_metadata={"code": e.code, "message": str(e)}
```
and `leasing.py:746` maps anything not `COMPLETE` to `status="failure"` on the canonical
receipt.

**Failure scenario:** operator calls `cognigate.cancel_job`; the job stops at the next
step boundary and the ledger records a *failed* obligation. Downstream retry/alerting
policy cannot distinguish "the operator stopped this" from "cognition broke", and the
exit-criteria template's mandatory `canceled` terminal outcome is absent. Compounding it,
`_fail_task` hardcodes `"retryable": False` (`leasing.py:415`) even for
`ExecutionError(recoverable=True)`, so genuinely transient failures are permanently
terminal — the template's "retryable failure uses a non-terminal type" is also unmet.

## Medium Findings

**MED-1 — Full lease payload is embedded in every receipt, in two places.**
`legivellum_receipts.py:111-116`:
```python
    if lease.payload:
        inputs["payload"] = lease.payload
        task_body = json.dumps(lease.payload)
```
Spec §11 ("Receipts never contain … large blobs, sensitive payloads") is violated by
construction, and `ReceiptGate/src/receiptgate/validation_v1.py:17` enforces `inputs` ≤
64 KB / `task_body` ≤ 100 KB. A 70 KB context payload (well under the prompt builder's
own 50 KB-per-field limit, and trivially reachable with several fields) makes **both**
the `accepted` and the `complete` receipt fail validation → the obligation never appears
in the ledger at all, and the failure is again only a warning. Use `payload_pointer`, or
truncate with a hash.

**MED-2 — No retry or backoff on provider errors; one 429 kills the job.**
`ai_client.py:118` posts once; `raise_for_status()` propagates through the circuit
breaker to `executor.py:311`, producing a FAILED receipt. Provider 429/500/502 are the
single most common transient in this component and there is no `Retry-After` handling
anywhere. The circuit breaker protects the provider from CogniGate, not jobs from the
provider.

**MED-3 — Tool retry loop has no backoff.**
`executor.py:527`:
```python
        for attempt in range(self.max_retries):
            result = await self.tool_executor.execute(tool_call, tool_context)
```
Three immediate hammer-retries against an upstream MCP server that just failed; combined
with `MCPAdapter._do_call`'s own 3 internal retries (`mcp_adapter.py:156`), a single plan
step issues up to 9 back-to-back requests with zero delay.

**MED-4 — `polling_start` can be called repeatedly, spawning parallel poll loops.**
`api.py:570`: `asyncio.create_task(state.work_poller.start())` with no idempotency guard
(`WorkPoller.start` sets `_running = True` unconditionally, `leasing.py:480`). Two calls →
two loops → two `lease_next` calls per interval, and `stop()` sets one flag both loops
read, so shutdown behaviour is undefined. The task handle is also dropped.

**MED-5 — Path containment uses string prefix comparison.**
`builtin_sinks.py:79`: `if not str(resolved_path).startswith(str(resolved_base))`. With
`base_path=/data/artifacts`, a resolved path of `/data/artifacts-evil/x.txt` passes.
Independent of CRITICAL-1 (which makes it moot today); use `Path.is_relative_to`.

**MED-6 — Caller-controlled `filename_template` is passed to `str.format`.**
`builtin_sinks.py:66`: `template.format(task_id=..., lease_id=..., timestamp=..., uuid=...)`.
A template of `{task_id.__class__.__mro__}` renders internal object reprs into the
filename; `{nonexistent}` raises `KeyError` inside the sink, surfacing as a job failure.
Neither is catastrophic, but the template should be an allowlisted enum, not a format
string.

**MED-7 — Raw exception text is returned to unauthenticated-ish callers.**
`api.py:765`:
```python
    except Exception as exc:
        return _jsonrpc_error(request.id, "ERROR", str(exc))
```
Anything raised below (httpx errors carrying full request URLs, `KeyError` from provider
response shapes, filesystem paths) is echoed verbatim to the client. Errors should be
mapped to codes with details logged server-side.

**MED-8 — CORS settings in `Settings` are dead; the middleware re-parses env and can be
made wildcard-with-credentials.** `config.py:115-130` defines four `cors_*` fields that
nothing reads; `api.py:211-226` reads the same env vars through its own parsers.
`COGNIGATE_CORS_ALLOWED_ORIGINS=*` yields `allow_origins=["*"]` alongside
`allow_credentials=True` — Starlette then echoes any Origin with credentials allowed. The
default is safe; the footgun and the duplicated config path are not.

**MED-9 — No bound on request payload size or field count.**
`prompt.py:183-190` iterates every non-reserved payload key into
`<additional_parameters>`; each value is capped at 5 000 chars but the **number** of keys
is unbounded, as is the raw HTTP body. A 50 MB JSON object with 100 000 keys builds a
~500 MB prompt string before the provider call. `models.Lease.payload` has no size
validator.

**MED-10 — Dead-letter queue is written and never drained.**
`leasing.py:49` stores failed receipts and `load()`/`get_all()`/`remove()` exist, but
nothing calls `load()` at startup and no task re-sends entries. `DeadLetterQueue` is a
write-only file whose entries are lost on pod replacement (`k8s/deployment.yaml`
`dlq-storage: emptyDir: {}`). "Persistent dead letter queue for failed receipts"
(`leasing.py:39`) is aspirational.

**MED-11 — Agent-blindness holds by upstream convention, not by construction.**
`leasing.py:216-218` copies AsyncGate's `task["requirements"]` wholesale into
`lease.constraints`, and `prompt.py:142-148` renders every constraint key/value into the
**system** prompt. Today `AsyncGate/src/asyncgate/models/task.py:13` limits
`TaskRequirements` to `capabilities`/`tags`, so no governance state flows — but there is
no allowlist on CogniGate's side. The day anyone adds a score, tier, price, or audit field
to task requirements or to `payload`, it silently becomes model context. SUSPECTED
future-break; the fix is a explicit allowlist of constraint keys admitted to the prompt.

**MED-12 — Per-process idempotency cache under a horizontal autoscaler.**
`executor.py:78` `_completed_leases` is per-process (honestly documented), but
`k8s/hpa.yaml` scales replicas and each pod polls AsyncGate independently. Spec §13
("Worker restart must not duplicate effects") is not met across replicas or restarts —
the cache is in-memory only.

## Low / Nits

- **LOW-1** `receipts.py:150` `ReceiptStore.delete()` unlinks a stored receipt. Unused and
  unexposed, but its existence contradicts append-only immutability; delete it.
- **LOW-2** `dedupe_key` is `lease.lease_id` for both the `accepted` and the `complete`
  receipt of the same lease (`legivellum_receipts.py:163`). ReceiptGate dedupes on
  `receipt_id`, so nothing breaks today, but the key does not identify the event.
- **LOW-3** Version strings disagree four ways: `__init__.py:3` `0.1.0`, `api.py:192`
  `0.1.0` (FastAPI), `api.py:331` `0.2.0` (health), `api.py:406` `0.1.0` (detailed
  health), `pyproject.toml:7` `0.2.0`.
- **LOW-4** `requirements.txt:21-24` ships `pytest`, `pytest-asyncio`, `pytest-cov`,
  `pytest-mock` into the runtime container (`Dockerfile:7`). `pytest-mock` is not used by
  any test.
- **LOW-5** `ai_client.py:55` still hardcodes `"HTTP-Referer": "https://cognigate.local"`
  — open since the January review (HIGH-004).
- **LOW-6** `ai_client.py:137` `data["choices"][0]` and `chat_with_tools`'s
  `data["choices"][0]["message"]` assume a well-formed provider response; a 200 with an
  error body raises `KeyError`/`IndexError` that surfaces as `UNEXPECTED_ERROR`.
- **LOW-7** `executor.py:588` `import json` inside the loop body; `api.py:450,500,525`
  `import uuid` inside functions; `api.py:340` `import asyncio` shadowing the module-level
  import. Flagged as MED-003 in the January review, still open.
- **LOW-8** `tools.py:279` `parse_tool_calls` and `prompt.py:232` `build_tool_prompt` are
  imported by `executor.py:22-23` and never called — the "return tool results to the
  model" half of Spec §8 is not implemented; `_execute_output_step` executes tool calls
  and discards their results (`executor.py:598`).
- **LOW-9** `conftest.py:92` `test_settings` constructs `Settings()` without
  `ai_api_key`, which raises unless the environment happens to define one. No test uses
  it — nor `app_client`, which sets a nonexistent `cognigate.config.settings` global.
- **NIT-1** `models.py:20` `ReceiptStatus` enum (`accepted|planning|executing|completed|failed`)
  is unused; the code paths all use `JobStatus`, and the two enums use different words for
  the same states.
- **NIT-2** `builtin_sinks.py:213` `print()` in a service that otherwise logs structurally.
- **NIT-3** `legivellum_receipts.py:20-28` and `metagate_client.py:41-61` each walk parent
  directories looking for a sibling `LegiVellum/shared` checkout and mutate `sys.path`.
  The shared library is a real runtime dependency of receipt building and is declared in
  neither `pyproject.toml` nor `requirements.txt`; in a container where the walk fails,
  `CanonicalReceipt is None` and receipts are emitted **unvalidated**
  (`legivellum_receipts.py:200`).

## Config & Dependency Drift

- `requirements.txt` vs `pyproject.toml` vs `uv.lock`: `uv.lock` mirrors `pyproject.toml`
  exactly (12 runtime deps, 6 dev). `requirements.txt` adds four test packages to the
  runtime set and omits `black`/`ruff`/`mypy`. CI installs both (`ci.yml:31-32`), so the
  drift is invisible in CI and visible only in the image (test tooling in production) and
  to anyone following the README's `pip install -e ".[dev]"` alone.
- **Two pytest configs.** `pytest.ini` exists *and* `pyproject.toml:57` defines
  `[tool.pytest.ini_options]`. `pytest.ini` wins, so `asyncio_mode = "auto"` is silently
  ignored. Async tests survive only because five files set `pytestmark = pytest.mark.asyncio`
  by hand — a new async test file without that line will be collected, not run, and
  reported as passed-with-warning.
- **Two compose files disagree.** `docker-compose.standalone.yml` is coherent (passes
  `API_KEY`, `AI_API_KEY`, standalone volume) and is what `run_local.sh` uses.
  `docker-compose.yaml` is broken (HIGH-7) and configures an AsyncGate that the image's
  own `STANDALONE_MODE=true` default disables. Two files, one of which has never been run.
- **`config/` defaults for production:** `mcp.yaml` ships `read_only: true` — good. The
  dangerous defaults are in `Settings`: `standalone_mode=True` (a *worker* that by default
  does no leasing), `receiptgate_emit_receipts=True` with `receiptgate_endpoint=""` (so
  the default posture is "emitting receipts" that go nowhere and log at debug), and
  `ai_model="anthropic/claude-3-opus"` (a deprecated, expensive model) in README,
  `.env.example`, `docker-compose.yaml` and `k8s/configmap.yaml`, while
  `docker-compose.standalone.yml:16` uses `claude-sonnet-4`.
- `k8s/configmap.yaml:20` sets `MAX_CONCURRENT_JOBS: "2"`, which activates HIGH-3.
- `.mypy-ci.ini` is a relaxed config used by CI (`ci.yml:44`) while `pyproject.toml:54`
  declares `strict = true` — strict mypy is never actually run.

## Test Coverage Gaps

What is genuinely well tested: prompt sanitization and XML delimiting (`test_prompt.py`,
23 cases), plugin-directory permission gating (`test_plugins.py`), the `plan_only`
non-execution invariant (`test_plan_only.py` — the best file in the repo; it asserts on
what *doesn't* happen and starves the mock provider so an accidental execution loop can't
pass silently), the stub provider's determinism and its refusal to invent tool calls
(`test_stub_ai_client.py`), and the poll→execute→receipt loop against a mock AsyncGate
(`test_job_execution.py`).

Missing regressions, in priority order:

1. **Side-effect freedom is not tested at all.** No test passes a `base_path` outside a
   temp dir (CRITICAL-1); every FileSink test hands it `str(temp_dir)` and then asserts
   the *filename* was sanitized. A regression asserting `deliver()` refuses
   `base_path="/etc/cognigate/plugins/sinks"` would have caught the whole finding.
2. **Zero tests for `legivellum_receipts.build_receipt`.** The single most protocol-critical
   function in the repo — 42 fields, phase-conditional invariants — has no test.
   `jsonschema.validate(build_receipt(...), receipt.schema.v1.json)` for
   `accepted`/`complete-success-with-artifact`/`complete-failure` would have caught HIGH-1
   and MED-1 immediately.
3. **No terminal-semantics tests.** Nothing asserts that cancel produces a `canceled`
   outcome, that `accepted` never closes an obligation, or that a terminal receipt is
   emitted exactly once. `TERMINAL_RECEIPT_TYPES` does not exist to test.
4. **No API-layer tests.** `api.py` (859 lines, the entire public contract) has no test:
   not auth rejection, not rate limiting, not `tools/list` shape, not JSON-RPC error
   codes, not `submit_job`'s promise that `get_receipt` will find the lease.
5. **No concurrency test for artifact isolation** (HIGH-3). `test_concurrent_job_limit`
   asserts the semaphore bound but never that two overlapping jobs get their own artifacts.
6. **No test that an unparseable plan fails the job** (HIGH-6) — `test_job_planning_failure`
   only covers a transport error, not a 200 with prose.
7. **Over-mocking note:** `fixtures.py`'s `MockAsyncGateServer` accepts every receipt shape
   unconditionally (`handle_complete` just appends), so the integration suite's
   "receipt delivered" assertions (`test_job_execution.py:496`) prove only that a POST was
   made — they cannot fail on a receipt ReceiptGate would reject. A schema-validating mock
   would flip those tests red today.

## Cross-repo observations

- **HIGH-1 is a stack-level break, not a CogniGate-local one.** ReceiptGate validates
  strictly and correctly (`validation_v1.py:88` even refuses to fail open when the schema
  file is missing — good), CogniGate emits non-conformant `complete` receipts, and the
  transport between them treats rejection as a debug event. Whoever owns integration
  should assume that *no* CogniGate obligation currently closes in the ledger. Worth
  checking whether the other gates' emitters share the "artifact fields default to NA"
  pattern — it looks like a copied idiom.
- **A conformance test-kit belongs in `LegiVellum/shared`.** Every gate hand-rolls a
  42-field dict. A shared `build_receipt()` + `assert_conformant()` in
  `LegiVellum/shared/legivellum/` would delete this class of bug from all seven repos.
  CogniGate already *tries* to use the shared `CanonicalReceipt` model
  (`legivellum_receipts.py:15`) but silently degrades to unvalidated dicts when the
  sibling checkout is absent — which is the normal case in a container.
- **The canonical role and the repo's own spec disagree, and nobody reconciled them.**
  `docs/canonical/CogniGate/alignment.md` says "executes cognition without side effects"
  and "store outputs in DepotGate"; `CogniGate Spec.txt` §10 says outputs go to whatever
  sink the lease names, and §8 hands the model an `mcp.call` bridge. The implementation
  follows the spec. Either the alignment doc is wrong, or the sink layer needs to be
  narrowed to a DepotGate client. This is an architecture decision, not a bug fix —
  but until it's made, "bounded cognition, no side effects" is not a property of this
  component. No DepotGate client exists in the repo; `COGNIGATE_V1_EXIT_CRITERIA.md:85`
  still lists "DepotGate wiring via MCP sink must be documented and validated" as open.
- **`AsyncGate.TaskRequirements` is the agent-blindness chokepoint** for this pair
  (`AsyncGate/src/asyncgate/models/task.py:13`). It is currently clean
  (`capabilities`/`tags`) and CogniGate pipes it straight into the system prompt. Whoever
  reviews AsyncGate should know that anything added to that model lands in a model
  context unfiltered.
- **Prior-review recurrence.** `.claude/CODE REVIEW1.md` (Jan 6) and
  `.claude/SECURITY_PUNCHLIST.md` (Jan 7) flagged path traversal, prompt injection,
  plugin RCE, missing auth, missing rate limiting and missing background-job timeouts.
  Auth, rate limiting, plugin permission gating and prompt sanitization were genuinely
  fixed. The path-traversal fix hardened the filename and left the root
  (CRITICAL-1); the background-job timeout (HIGH-003) was never fixed; the hardcoded
  `HTTP-Referer` (HIGH-004) and the inline imports (MED-003) are untouched. Two recurrences
  and one half-fix across two prior passes.

## What's solid

- `plan_only` and its test file. The separation between "answer a planning question" and
  "execute an obligation" is real, deliberate, documented in prose that explains *why*,
  and defended by tests that assert on absence. This is the strongest reasoning in the repo.
- The stub provider (`stub_ai_client.py`) stubs the network boundary rather than the
  client, so the circuit breaker, JSON parsing, `_extract_json` fallback and token
  accounting all stay live in CI — and it refuses to invent tool calls because "selecting
  a tool is a reasoning act". `is_stub` is surfaced in `plan` responses and
  `health_detailed` so a caller can tell canned output from cognition without reading
  config. `ai_require_real` fails startup rather than at job time.
- Prompt construction: XML-delimited user content, injection-pattern redaction applied to
  payload, constraints, plan-derived instructions *and* accumulated inter-step context —
  the last of which is the one people usually forget.
- The model never chooses an endpoint: `mcp_call` takes a server *name* resolved against
  operator config (`tools.py:184`), with a hard-deny list, a read-only default, and
  params size/null-byte validation. That closes the SSRF surface the January review
  raised.
- MetaGate bootstrap is non-blocking by construction and says so in a comment that
  explains the failure mode ("a bootstrap authority that can take the mesh down would be
  a hidden master").
- Observability: correlated structured logs, a bounded/truncatable metrics snapshot over
  MCP, graceful shutdown that emits receipts for jobs it had to kill.
