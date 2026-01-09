# CogniGate Operations Runbook

This runbook provides operational procedures for deploying, monitoring, and maintaining CogniGate in production environments.

## Table of Contents

1. [Deployment Procedure](#deployment-procedure)
2. [Configuration Reference](#configuration-reference)
3. [Monitoring & Dashboards](#monitoring--dashboards)
4. [Common Failure Modes & Remediation](#common-failure-modes--remediation)
5. [Scaling Guidelines](#scaling-guidelines)
6. [Security Checklist](#security-checklist)
7. [DLQ Recovery Procedures](#dlq-recovery-procedures)
8. [Circuit Breaker Management](#circuit-breaker-management)

---

## Deployment Procedure

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured with cluster access
- Helm 3.x (optional, for Prometheus Operator)
- Container registry access

### Initial Deployment

1. **Create namespace and base resources:**
   ```bash
   kubectl apply -k k8s/
   ```

2. **Configure secrets:**
   ```bash
   # Edit secrets with actual values
   kubectl edit secret cognigate-secrets -n cognigate

   # Or use external secrets operator
   kubectl apply -f k8s/external-secrets.yaml
   ```

3. **Verify deployment:**
   ```bash
   kubectl get pods -n cognigate -w
   kubectl logs -n cognigate -l app.kubernetes.io/name=cognigate -f
   ```

4. **Check health endpoints:**
   ```bash
   kubectl port-forward -n cognigate svc/cognigate 8000:8000
   curl http://localhost:8000/health
   curl http://localhost:8000/health/detailed
   ```

### Rolling Updates

```bash
# Update image
kubectl set image deployment/cognigate cognigate=your-registry/cognigate:v1.2.3 -n cognigate

# Monitor rollout
kubectl rollout status deployment/cognigate -n cognigate

# Rollback if needed
kubectl rollout undo deployment/cognigate -n cognigate
```

### Blue-Green Deployment

For zero-downtime deployments with instant rollback:

```bash
# Deploy new version with different name
kubectl apply -f deployment-v2.yaml

# Test new version
kubectl port-forward svc/cognigate-v2 8001:8000 -n cognigate
curl http://localhost:8001/health

# Switch traffic
kubectl patch svc cognigate -n cognigate -p '{"spec":{"selector":{"version":"v2"}}}'

# Remove old version after verification
kubectl delete deployment cognigate-v1 -n cognigate
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `COGNIGATE_ASYNCGATE_ENDPOINT` | Yes | `http://localhost:8080` | AsyncGate API endpoint |
| `COGNIGATE_ASYNCGATE_AUTH_TOKEN` | Yes | - | AsyncGate authentication token |
| `COGNIGATE_AI_ENDPOINT` | No | `https://openrouter.ai/api/v1` | AI provider API endpoint |
| `COGNIGATE_AI_API_KEY` | Yes | - | AI provider API key |
| `COGNIGATE_AI_MODEL` | No | `anthropic/claude-3-opus` | AI model identifier |
| `COGNIGATE_AI_MAX_TOKENS` | No | `4096` | Maximum tokens per request |
| `COGNIGATE_POLLING_INTERVAL` | No | `5.0` | Seconds between AsyncGate polls |
| `COGNIGATE_MAX_CONCURRENT_JOBS` | No | `1` | Max concurrent job executions |
| `COGNIGATE_JOB_TIMEOUT` | No | `300` | Job timeout in seconds |
| `COGNIGATE_MAX_RETRIES` | No | `3` | Max retries for tool calls |
| `COGNIGATE_WORKER_ID` | No | `cognigate-worker-1` | Unique worker identifier |
| `COGNIGATE_HOST` | No | `0.0.0.0` | Server bind address |
| `COGNIGATE_PORT` | No | `8000` | Server port |
| `COGNIGATE_API_KEY` | Conditional | - | API key for REST auth |
| `COGNIGATE_REQUIRE_AUTH` | No | `true` | Require authentication |
| `COGNIGATE_ALLOW_INSECURE_DEV` | No | `false` | Allow unauthenticated (dev only) |
| `COGNIGATE_RATE_LIMIT_ENABLED` | No | `true` | Enable rate limiting |
| `COGNIGATE_RATE_LIMIT_REQUESTS_PER_MINUTE` | No | `50` | Rate limit per client |
| `COGNIGATE_REDIS_URL` | No | - | Redis URL for distributed rate limiting |
| `COGNIGATE_LOG_LEVEL` | No | `INFO` | Logging level |
| `COGNIGATE_JSON_LOGS` | No | `true` | JSON structured logging |
| `COGNIGATE_CONFIG_DIR` | No | `/etc/cognigate` | Configuration directory |
| `COGNIGATE_PROFILES_DIR` | No | `/etc/cognigate/profiles` | Instruction profiles directory |
| `COGNIGATE_PLUGINS_DIR` | No | `/etc/cognigate/plugins` | Plugins directory |

### CORS Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `COGNIGATE_CORS_ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8080` | Comma-separated origins |
| `COGNIGATE_CORS_ALLOW_CREDENTIALS` | `true` | Allow credentials |
| `COGNIGATE_CORS_ALLOWED_METHODS` | `GET,POST,PUT,DELETE,OPTIONS` | Allowed HTTP methods |
| `COGNIGATE_CORS_ALLOWED_HEADERS` | `Authorization,Content-Type,X-Tenant-ID` | Allowed headers |

---

## Monitoring & Dashboards

### Prometheus Metrics

CogniGate exposes metrics at `/metrics` in Prometheus format.

**Key Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `cognigate_jobs_total` | Counter | Total jobs by status and profile |
| `cognigate_job_duration_seconds` | Histogram | Job execution duration |
| `cognigate_active_jobs` | Gauge | Currently active jobs |
| `cognigate_lease_claims_total` | Counter | Lease claim attempts |
| `cognigate_lease_extensions_total` | Counter | Lease extensions |
| `cognigate_receipts_total` | Counter | Receipts sent by status |
| `cognigate_dead_letter_queue_size` | Gauge | DLQ size |
| `cognigate_tool_calls_total` | Counter | Tool invocations |
| `cognigate_ai_requests_total` | Counter | AI provider requests |
| `cognigate_ai_tokens_used_total` | Counter | Token consumption |
| `cognigate_circuit_breaker_state` | Gauge | Circuit breaker states |
| `cognigate_circuit_breaker_failures_total` | Counter | Circuit breaker failures |

### Grafana Dashboard Setup

Import the following dashboard JSON or create panels for:

1. **Overview Panel:**
   - Jobs/minute rate
   - Success/failure ratio
   - Active jobs gauge
   - P95 job duration

2. **Health Panel:**
   - Circuit breaker states
   - DLQ size trend
   - Error rates by type

3. **Resource Panel:**
   - Token consumption rate
   - Tool call distribution
   - AI request latency

**Sample PromQL Queries:**

```promql
# Job throughput
sum(rate(cognigate_jobs_total[5m])) by (status)

# Success rate
sum(rate(cognigate_jobs_total{status="complete"}[5m]))
/
sum(rate(cognigate_jobs_total[5m]))

# P95 job duration
histogram_quantile(0.95, sum(rate(cognigate_job_duration_seconds_bucket[5m])) by (le))

# Token burn rate
sum(rate(cognigate_ai_tokens_used_total[5m])) by (type)

# Circuit breaker status
cognigate_circuit_breaker_state == 1
```

### Log Queries (Loki/CloudWatch)

```logql
# All errors
{namespace="cognigate"} |= "error"

# Job failures
{namespace="cognigate"} | json | event="job_failed"

# Circuit breaker events
{namespace="cognigate"} | json | event=~"circuit_breaker_.*"

# Slow jobs (>60s)
{namespace="cognigate"} | json | event="job_completed" | duration_seconds > 60
```

---

## Common Failure Modes & Remediation

### 1. AsyncGate Connection Failures

**Symptoms:**
- `lease_claim_error` logs
- No jobs being processed
- `/health/detailed` shows `asyncgate: unhealthy`

**Diagnosis:**
```bash
# Check AsyncGate connectivity
kubectl exec -n cognigate deploy/cognigate -- curl -s http://asyncgate:8080/health

# Check logs
kubectl logs -n cognigate -l app.kubernetes.io/name=cognigate --tail=100 | grep asyncgate
```

**Remediation:**
1. Verify AsyncGate service is running
2. Check network policies allow traffic
3. Verify authentication token is correct
4. Check AsyncGate logs for errors

### 2. AI Provider Errors

**Symptoms:**
- `ai_request_error` logs
- Jobs failing with AI-related errors
- Circuit breaker opening for AI provider

**Diagnosis:**
```bash
# Check circuit breaker state
curl -s http://localhost:8000/health/detailed | jq '.checks.ai_provider'

# Check AI error rate
# PromQL: rate(cognigate_ai_requests_total{status="error"}[5m])
```

**Remediation:**
1. Check AI provider status page
2. Verify API key is valid and has quota
3. Check rate limits on AI provider
4. Wait for circuit breaker to half-open (default 60s)

### 3. High Job Failure Rate

**Symptoms:**
- `cognigate_jobs_total{status="failed"}` increasing
- Jobs completing with errors

**Diagnosis:**
```bash
# Get recent failures
kubectl logs -n cognigate -l app.kubernetes.io/name=cognigate | grep job_failed | tail -20

# Check error distribution
# PromQL: sum(rate(cognigate_jobs_total{status="failed"}[5m])) by (error_code)
```

**Remediation:**
1. Review job payloads for malformed requests
2. Check instruction profiles are valid
3. Verify MCP endpoints are accessible
4. Review tool execution errors

### 4. Memory Pressure / OOMKilled

**Symptoms:**
- Pods restarting with OOMKilled
- High memory usage in metrics

**Diagnosis:**
```bash
kubectl describe pod -n cognigate <pod-name>
kubectl top pods -n cognigate
```

**Remediation:**
1. Increase memory limits in deployment
2. Reduce `MAX_CONCURRENT_JOBS`
3. Review for memory leaks in custom plugins
4. Consider horizontal scaling instead

### 5. Lease Timeouts

**Symptoms:**
- `lease_extension_failed` logs
- Jobs being reclaimed by AsyncGate

**Diagnosis:**
```bash
# Check heartbeat success rate
# PromQL: rate(cognigate_lease_extensions_total{success="true"}[5m])
```

**Remediation:**
1. Reduce job complexity or increase timeout
2. Check network latency to AsyncGate
3. Reduce heartbeat interval
4. Verify jobs aren't blocking

---

## Scaling Guidelines

### Horizontal Scaling

**When to scale out:**
- CPU utilization > 70%
- Job queue growing (check AsyncGate)
- P95 latency increasing

**HPA Configuration:**
```yaml
minReplicas: 2      # Minimum for HA
maxReplicas: 10     # Based on budget/capacity
targetCPU: 70%      # Trigger scale-up
```

**Manual scaling:**
```bash
kubectl scale deployment/cognigate --replicas=5 -n cognigate
```

### Vertical Scaling

**When to scale up:**
- Complex jobs requiring more memory
- High token context windows
- Many concurrent tools

**Recommended resource tiers:**

| Tier | CPU | Memory | Use Case |
|------|-----|--------|----------|
| Small | 100m-500m | 256Mi-512Mi | Simple jobs, low concurrency |
| Medium | 500m-1000m | 512Mi-1Gi | Standard workloads |
| Large | 1000m-2000m | 1Gi-2Gi | Complex jobs, high concurrency |

### Concurrency Tuning

```bash
# Increase concurrent jobs per pod
kubectl set env deployment/cognigate COGNIGATE_MAX_CONCURRENT_JOBS=4 -n cognigate
```

**Trade-offs:**
- Higher concurrency = better throughput, higher memory
- Lower concurrency = lower memory, may need more pods

---

## Security Checklist

### Pre-Production

- [ ] API keys stored in secrets management (Vault, AWS Secrets Manager)
- [ ] `COGNIGATE_ALLOW_INSECURE_DEV=false`
- [ ] `COGNIGATE_REQUIRE_AUTH=true`
- [ ] Network policies restricting ingress/egress
- [ ] TLS termination at ingress
- [ ] Pod security context (non-root, read-only filesystem)
- [ ] Resource limits set
- [ ] CORS origins restricted to known domains

### Runtime

- [ ] Regular secret rotation
- [ ] API key audit logging enabled
- [ ] Rate limiting enabled
- [ ] Circuit breakers configured
- [ ] Monitoring alerts configured

### Network Security

```yaml
# Example NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cognigate-network-policy
  namespace: cognigate
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: cognigate
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: asyncgate
      ports:
        - port: 8080
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443  # AI provider HTTPS
```

---

## DLQ Recovery Procedures

### Understanding the Dead Letter Queue

Failed receipts are stored in `/var/lib/cognigate/dlq/receipts.json` when:
- AsyncGate is unreachable after max retries
- Receipt payload is rejected (4xx errors)
- Network timeouts persist

### Viewing DLQ Contents

```bash
# Access pod
kubectl exec -it -n cognigate deploy/cognigate -- sh

# View DLQ
cat /var/lib/cognigate/dlq/receipts.json | jq .
```

### Manual Recovery

1. **Export DLQ entries:**
   ```bash
   kubectl cp cognigate/<pod>:/var/lib/cognigate/dlq/receipts.json ./dlq-backup.json
   ```

2. **Review and classify:**
   ```bash
   cat dlq-backup.json | jq '.[] | {lease_id: .receipt.lease_id, status: .receipt.status, failed_at}'
   ```

3. **Replay receipts manually:**
   ```bash
   # For each receipt, POST to AsyncGate
   curl -X POST http://asyncgate:8080/v1/tasks/<task_id>/complete \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d @receipt.json
   ```

4. **Clear processed entries:**
   ```bash
   # After successful replay, clear the DLQ
   kubectl exec -n cognigate deploy/cognigate -- rm /var/lib/cognigate/dlq/receipts.json
   ```

### Automated DLQ Processing

For recurring DLQ issues, implement a sidecar or CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cognigate-dlq-processor
  namespace: cognigate
spec:
  schedule: "*/15 * * * *"  # Every 15 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: dlq-processor
              image: cognigate:latest
              command: ["python", "-m", "cognigate.scripts.process_dlq"]
              envFrom:
                - secretRef:
                    name: cognigate-secrets
          restartPolicy: OnFailure
```

---

## Circuit Breaker Management

### Understanding Circuit Breaker States

| State | Description | Behavior |
|-------|-------------|----------|
| `closed` | Normal operation | Requests pass through |
| `open` | Failures exceeded threshold | Requests fail-fast |
| `half_open` | Testing recovery | Limited requests allowed |

### Viewing Circuit Breaker States

```bash
# Via health endpoint
curl -s http://localhost:8000/health/detailed | jq '.checks'

# Via Prometheus
# PromQL: cognigate_circuit_breaker_state
```

### Configuration

Default settings (in code):
- `failure_threshold`: 5 failures to open
- `recovery_timeout`: 60 seconds before half-open
- `half_open_max_calls`: 3 test calls allowed

### Manual Reset

Circuit breakers auto-recover, but for immediate reset:

1. **Restart the pod:**
   ```bash
   kubectl delete pod -n cognigate -l app.kubernetes.io/name=cognigate
   ```

2. **Rolling restart:**
   ```bash
   kubectl rollout restart deployment/cognigate -n cognigate
   ```

### Circuit Breaker Tuning

For flaky external services, adjust thresholds:

```python
# In circuit_breaker.py or configuration
CircuitBreaker(
    name="ai_provider",
    failure_threshold=10,      # More tolerant
    recovery_timeout=30.0,     # Faster recovery
    half_open_max_calls=5      # More test calls
)
```

### Monitoring Circuit Breaker Health

Set up alerts for:
- Circuit breaker opening (`state == "open"`)
- High failure counts before opening
- Prolonged half-open state

```promql
# Alert when circuit breaker opens
ALERT CircuitBreakerOpen
  IF cognigate_circuit_breaker_state{state="open"} == 1
  FOR 1m
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "Circuit breaker {{ $labels.name }} is open",
    description = "External service failures have triggered circuit breaker protection"
  }
```

---

## Emergency Procedures

### Complete Service Outage

1. Check pod status: `kubectl get pods -n cognigate`
2. Check events: `kubectl get events -n cognigate --sort-by=.lastTimestamp`
3. Review logs: `kubectl logs -n cognigate -l app.kubernetes.io/name=cognigate --tail=500`
4. Check dependencies (AsyncGate, AI provider, Redis)
5. If needed, rollback: `kubectl rollout undo deployment/cognigate -n cognigate`

### Data Loss Prevention

1. DLQ is persisted to disk - backup before pod termination
2. In-flight jobs will send failure receipts on graceful shutdown
3. For forced termination, jobs may be reclaimed by AsyncGate after lease timeout

### Escalation Path

1. **L1**: Check dashboards, restart pods if needed
2. **L2**: Review logs, check external dependencies
3. **L3**: Code-level debugging, configuration changes
4. **L4**: Vendor escalation (AI provider, cloud provider)

---

## Appendix

### Useful Commands

```bash
# Get all CogniGate resources
kubectl get all -n cognigate

# Watch pod logs
kubectl logs -f -n cognigate -l app.kubernetes.io/name=cognigate

# Execute into running pod
kubectl exec -it -n cognigate deploy/cognigate -- sh

# Port forward for local access
kubectl port-forward -n cognigate svc/cognigate 8000:8000

# Get current config
kubectl get configmap cognigate-config -n cognigate -o yaml

# Force pod restart
kubectl delete pod -n cognigate -l app.kubernetes.io/name=cognigate

# Check resource usage
kubectl top pods -n cognigate
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-01-XX | Initial release |
