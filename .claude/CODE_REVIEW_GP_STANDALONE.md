# CogniGate Code Review: GP Standalone Mode

**Date**: 2026-01-09  
**Reviewer**: Kee  
**Focus**: Making CogniGate work as a general-purpose standalone cognitive worker

## Executive Summary

CogniGate has solid bones for GP standalone operation but is currently hardwired to AsyncGate leasing. The core execution engine, tool surface, and plugin architecture are clean and well-designed. **Main gap: no direct job submission path that bypasses the leasing model.**

**Estimated effort**: 4-6 hours to add standalone mode with proper config management.

---

## Current Architecture Assessment

### ✅ What Works Well

1. **Clean execution model** - Planning → Execution → Receipt flow is elegant
2. **Profile-based configuration** - YAML profiles for instructions are perfect for GP use
3. **Tool abstraction** - `mcp_call` and `artifact_write` surface is minimal and correct
4. **Plugin architecture** - Sinks and MCP adapters are properly decoupled
5. **Observability** - Structured logging and metrics are production-grade
6. **Security** - Auth middleware, rate limiting, CORS all implemented

### ⚠️ Issues for Standalone GP Operation

1. **AsyncGate Hard Dependency**
   - `WorkPoller` expects to poll AsyncGate for leases
   - No direct "run this task now" execution path
   - `submit_job` endpoint creates fake lease but runs in background
   - Receipt model assumes AsyncGate as destination

2. **Configuration Coupling**
   - `COGNIGATE_ASYNCGATE_ENDPOINT` and `COGNIGATE_ASYNCGATE_AUTH_TOKEN` are required
   - No "standalone mode" flag to bypass AsyncGate entirely
   - Can't run without pointing at an AsyncGate instance (even if fake)

3. **Job Intake Model**
   - `Lease` is the primary job container
   - Direct job submission via `/v1/jobs` is marked "testing/local use"
   - No synchronous execution - always returns immediately

4. **Receipt Destination**
   - Receipts are designed to go back to AsyncGate
   - No local receipt storage/retrieval mechanism
   - No webhook or callback options for standalone operation

---

## Recommended Changes for GP Standalone Mode

### 1. Add Standalone Mode Flag

**File**: `src/cognigate/config.py`

```python
class Settings(BaseSettings):
    # ... existing fields ...
    
    # Standalone mode
    standalone_mode: bool = Field(
        default=False,
        description="Run in standalone mode without AsyncGate"
    )
    
    @field_validator("asyncgate_endpoint", "asyncgate_auth_token")
    @classmethod
    def validate_asyncgate_config(cls, v: str, info) -> str:
        """Allow empty AsyncGate config in standalone mode."""
        standalone = info.data.get("standalone_mode", False)
        if not standalone and not v:
            raise ValueError(
                "asyncgate_endpoint and asyncgate_auth_token required "
                "unless standalone_mode=true"
            )
        return v
```

### 2. Make AsyncGateClient Optional

**File**: `src/cognigate/leasing.py`

Add factory function:

```python
def create_asyncgate_client_or_none(settings: Settings) -> AsyncGateClient | None:
    """Create AsyncGate client only if not in standalone mode."""
    if settings.standalone_mode:
        return None
    return AsyncGateClient(settings)
```

**File**: `src/cognigate/api.py`

Update lifespan to handle None:

```python
# Initialize AsyncGate client (optional in standalone mode)
if not state.settings.standalone_mode:
    state.asyncgate_client = AsyncGateClient(state.settings)
    state.work_poller = WorkPoller(
        state.asyncgate_client,
        state.settings,
        job_handler
    )
else:
    state.asyncgate_client = None
    state.work_poller = None
    logger.info("Running in standalone mode - polling disabled")
```

### 3. Add Synchronous Job Execution Endpoint

**File**: `src/cognigate/api.py`

```python
@app.post(
    "/v1/jobs/execute",
    response_model=Receipt,
    dependencies=[Depends(get_auth), Depends(rate_limit_dependency)]
)
async def execute_job_sync(request: SubmitJobRequest):
    """Execute a job synchronously and return the receipt.
    
    This endpoint is designed for standalone mode where CogniGate
    acts as a direct cognitive worker without AsyncGate leasing.
    """
    if not state.job_executor:
        raise HTTPException(status_code=503, detail="Not ready")

    import uuid

    lease = Lease(
        lease_id=str(uuid.uuid4()),
        task_id=request.task_id,
        payload=request.payload,
        profile=request.profile,
        sink_config=request.sink_config,
        constraints=request.constraints
    )

    # Execute synchronously
    receipt = await state.job_executor.execute(lease)
    
    return receipt
```

### 4. Add Receipt Storage (Optional Enhancement)

**File**: `src/cognigate/receipts.py` (new)

```python
"""Receipt storage for standalone mode."""

from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Optional

from .models import Receipt


class ReceiptStore:
    """Simple file-based receipt storage for standalone mode."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, receipt: Receipt) -> None:
        """Save a receipt to disk."""
        filename = f"{receipt.lease_id}_{receipt.timestamp.isoformat()}.json"
        filepath = self.storage_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(receipt.model_dump(), f, indent=2, default=str)

    def get(self, lease_id: str) -> Optional[Receipt]:
        """Get the most recent receipt for a lease."""
        receipts = list(self.storage_dir.glob(f"{lease_id}_*.json"))
        if not receipts:
            return None
        
        # Get most recent
        latest = max(receipts, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            data = json.load(f)
        
        return Receipt(**data)

    def list(self, limit: int = 100) -> list[Receipt]:
        """List recent receipts."""
        receipt_files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        receipts = []
        for filepath in receipt_files:
            with open(filepath) as f:
                data = json.load(f)
            receipts.append(Receipt(**data))
        
        return receipts
```

Add to Settings:

```python
# Receipt storage (for standalone mode)
receipt_storage_dir: Path = Field(
    default=Path("/var/lib/cognigate/receipts"),
    description="Directory for receipt storage in standalone mode"
)
```

Add endpoints:

```python
@app.get("/v1/receipts/{lease_id}", dependencies=[Depends(get_auth)])
async def get_receipt(lease_id: str):
    """Get receipt for a specific lease."""
    if not state.receipt_store:
        raise HTTPException(status_code=503, detail="Receipt storage not enabled")
    
    receipt = state.receipt_store.get(lease_id)
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    return receipt


@app.get("/v1/receipts", dependencies=[Depends(get_auth)])
async def list_receipts(limit: int = 100):
    """List recent receipts."""
    if not state.receipt_store:
        raise HTTPException(status_code=503, detail="Receipt storage not enabled")
    
    return {"receipts": state.receipt_store.list(limit)}
```

### 5. Environment Variable Configuration Template

**File**: `.env.standalone.example` (new)

```bash
# CogniGate Standalone Mode Configuration

# Mode selection
COGNIGATE_STANDALONE_MODE=true

# AI Provider (OpenRouter or OpenAI-compatible)
COGNIGATE_AI_ENDPOINT=https://openrouter.ai/api/v1
COGNIGATE_AI_API_KEY=your-openrouter-api-key
COGNIGATE_AI_MODEL=anthropic/claude-sonnet-4-20250514
COGNIGATE_AI_MAX_TOKENS=8192

# Worker configuration
COGNIGATE_WORKER_ID=cognigate-gp-1
COGNIGATE_JOB_TIMEOUT=600
COGNIGATE_MAX_RETRIES=3

# Server settings
COGNIGATE_HOST=0.0.0.0
COGNIGATE_PORT=8000

# Authentication
COGNIGATE_API_KEY=cg_your-secret-key
COGNIGATE_REQUIRE_AUTH=true
COGNIGATE_ALLOW_INSECURE_DEV=false

# Rate limiting
COGNIGATE_RATE_LIMIT_ENABLED=true
COGNIGATE_RATE_LIMIT_REQUESTS_PER_MINUTE=60

# Paths
COGNIGATE_CONFIG_DIR=./config
COGNIGATE_PLUGINS_DIR=./config/plugins
COGNIGATE_PROFILES_DIR=./config/profiles
COGNIGATE_RECEIPT_STORAGE_DIR=./receipts

# MCP server tokens (used in mcp.yaml)
GITHUB_MCP_TOKEN=your-github-token

# AsyncGate (not used in standalone mode, but required by validator)
COGNIGATE_ASYNCGATE_ENDPOINT=http://unused
COGNIGATE_ASYNCGATE_AUTH_TOKEN=unused
```

---

## Configuration Management Recommendations

### File-Based vs Environment-Based

**Current split is good:**
- **Environment**: Credentials, endpoints, runtime settings
- **Files**: Instruction profiles, MCP endpoints, plugin configs

**Keep this approach** - it's the right balance for GP usage.

### Profile Loading

**Current system is perfect:**
1. Drop YAML file in `profiles/` directory
2. Restart service
3. Profile auto-discovered

**Recommendation**: Add validation on startup to catch malformed profiles early.

```python
def _load_profiles(self) -> None:
    """Load all instruction profiles from profiles directory."""
    profiles_dir = self.settings.profiles_dir
    if not profiles_dir.exists():
        logger.warning(f"Profiles directory not found: {profiles_dir}")
        return

    for profile_file in profiles_dir.glob("*.yaml"):
        try:
            profile = load_instruction_profile(profile_file)
            self.profiles[profile.name] = profile
            logger.info(f"Loaded profile: {profile.name}")
        except Exception as e:
            logger.error(f"Failed to load profile {profile_file}: {e}")
            # Continue loading other profiles
```

### MCP Endpoint Configuration

**Current system** (`config/mcp.yaml`) works but could be more flexible.

**Enhancement**: Support per-profile MCP configs

```yaml
# config/profiles/code_review.yaml
name: code_review
mcp_servers:
  - github  # Reference to mcp.yaml endpoint
  - filesystem  # Reference to mcp.yaml endpoint
system_instructions: |
  You are performing a code review...
```

This lets profiles declare their tool dependencies.

---

## Testing Standalone Mode

### Quick Start

```bash
# 1. Copy standalone config
cp .env.standalone.example .env

# 2. Set your API key
export COGNIGATE_API_KEY=your-secret-key
export COGNIGATE_AI_API_KEY=your-openrouter-key

# 3. Run in standalone mode
COGNIGATE_STANDALONE_MODE=true uvicorn cognigate.api:app --host 0.0.0.0 --port 8000
```

### Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List profiles
curl -H "X-API-Key: your-secret-key" http://localhost:8000/v1/config/profiles

# Execute job synchronously
curl -X POST http://localhost:8000/v1/jobs/execute \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "payload": {
      "instruction": "Analyze this code snippet",
      "code": "def hello(): print(\"Hello\")"
    },
    "profile": "default"
  }'

# Get receipt (if storage enabled)
curl -H "X-API-Key: your-secret-key" \
  http://localhost:8000/v1/receipts/lease-id-here
```

---

## Security Considerations

### Current State: Good Foundations

1. ✅ API key auth implemented
2. ✅ Rate limiting present
3. ✅ CORS properly configured
4. ✅ Input validation via Pydantic

### Enhancements for Standalone

1. **Add API key rotation** - Currently single static key
2. **Add request signing** - For webhook callbacks
3. **Add tenant isolation** - If supporting multi-tenant

**Recommendation**: Current security is fine for single-tenant GP use. Add tenant headers only if needed.

---

## Performance Considerations

### Concurrent Job Execution

Current: `MAX_CONCURRENT_JOBS=1` (default)

**For GP standalone**, increase this:

```bash
COGNIGATE_MAX_CONCURRENT_JOBS=5
```

**Caveat**: Each job is a full AI execution loop. Monitor memory usage.

### Job Timeout

Current: `JOB_TIMEOUT=300` (5 minutes)

**For GP standalone with complex tasks**:

```bash
COGNIGATE_JOB_TIMEOUT=1800  # 30 minutes
```

### Rate Limiting

Current: `RATE_LIMIT_REQUESTS_PER_MINUTE=50`

**For GP standalone**:
- Burst mode: Set to 200+
- Controlled mode: Keep at 50

---

## Deployment Patterns

### Docker Compose (Standalone)

```yaml
version: '3.8'
services:
  cognigate:
    build: .
    ports:
      - "8000:8000"
    environment:
      - COGNIGATE_STANDALONE_MODE=true
      - COGNIGATE_AI_API_KEY=${AI_API_KEY}
      - COGNIGATE_API_KEY=${API_KEY}
    volumes:
      - ./config:/etc/cognigate
      - ./receipts:/var/lib/cognigate/receipts
```

### Kubernetes (Standalone)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cognigate
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: cognigate
        image: cognigate:latest
        env:
        - name: COGNIGATE_STANDALONE_MODE
          value: "true"
        - name: COGNIGATE_AI_API_KEY
          valueFrom:
            secretKeyRef:
              name: cognigate-secrets
              key: ai-api-key
        volumeMounts:
        - name: config
          mountPath: /etc/cognigate
      volumes:
      - name: config
        configMap:
          name: cognigate-config
```

---

## Migration Path

### Phase 1: Enable Standalone Mode (2 hours)

1. Add `standalone_mode` flag to Settings
2. Make AsyncGateClient optional
3. Test existing `/v1/jobs` endpoint in standalone mode

### Phase 2: Sync Execution (2 hours)

1. Add `/v1/jobs/execute` endpoint
2. Test synchronous execution flow
3. Validate receipt generation

### Phase 3: Receipt Storage (2 hours)

1. Implement ReceiptStore
2. Add receipt endpoints
3. Test persistence across restarts

### Phase 4: Documentation (30 minutes)

1. Update README with standalone mode
2. Add `.env.standalone.example`
3. Create quickstart guide

---

## Open Questions

1. **Receipt delivery in standalone mode** - Store locally? Webhook? Both?
2. **Job queuing** - Should standalone mode have internal queue or require external?
3. **Multi-tenancy** - Support tenant isolation or single-tenant only?
4. **Profile hot-reload** - Should profiles reload without restart?

**Recommendations**:
1. Local storage + optional webhooks
2. No internal queue - keep it simple, use external if needed
3. Single-tenant for v1
4. No hot-reload - require restart for safety

---

## Summary

CogniGate is **90% ready** for GP standalone operation. The core engine is clean and well-designed. Main work:

1. **2 hours**: Standalone mode flag + optional AsyncGate
2. **2 hours**: Sync execution endpoint
3. **2 hours**: Receipt storage (optional but recommended)

After these changes, CogniGate becomes a **pure cognitive execution service** - point it at an AI provider, configure tools/profiles, and execute tasks via REST API.

**Confidence**: High. The architecture is sound, changes are surgical, risk is low.
