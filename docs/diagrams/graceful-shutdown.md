# Graceful Shutdown Sequence

This diagram shows how CogniGate handles graceful shutdown to ensure job completion.

```mermaid
sequenceDiagram
    autonumber
    participant K8s as Kubernetes
    participant App as FastAPI App
    participant WP as WorkPoller
    participant HB as Heartbeat Tasks
    participant Jobs as Active Jobs
    participant AGC as AsyncGateClient
    participant AG as AsyncGate

    Note over K8s: Pod termination initiated
    K8s->>App: SIGTERM

    Note over App: Lifespan shutdown begins

    App->>WP: stop_gracefully(timeout=300)

    WP->>WP: _running = False
    WP->>WP: _shutting_down = True
    Note over WP: Stop accepting new work

    WP->>WP: Check active_jobs count

    alt No active jobs
        WP-->>App: Shutdown complete (immediate)
    else Has active jobs
        Note over WP: Wait for jobs with timeout

        par Monitor job completion
            loop While active_jobs > 0 and not timed out
                WP->>WP: Sleep 0.5s
                WP->>WP: Check active_jobs
            end
        and Jobs continue executing
            Jobs->>Jobs: Continue current step
            Jobs->>AG: Tool calls (if needed)
            AG-->>Jobs: Results
            Jobs->>Jobs: Complete execution
            Jobs-->>WP: Receipt ready
        and Heartbeats continue
            loop While job active
                HB->>AGC: extend_lease()
                AGC->>AG: POST /leases/{id}/extend
                AG-->>AGC: 200 OK
                HB->>HB: Sleep heartbeat_interval
            end
        end

        alt Jobs completed in time
            WP->>WP: All jobs finished
            WP-->>App: Shutdown complete
        else Timeout reached
            Note over WP: Force cleanup

            loop For each remaining job
                WP->>Jobs: Cancel task
                Jobs->>Jobs: asyncio.CancelledError

                WP->>HB: Stop heartbeat
                HB->>HB: Task cancelled

                WP->>AGC: send_receipt(FAILED)
                AGC->>AG: POST /tasks/{id}/fail
                Note over AG: code: SHUTDOWN_CANCELLED
            end

            WP->>WP: Clear all tracking
            WP-->>App: Shutdown complete (with warnings)
        end
    end

    App->>AGC: close()
    AGC->>AGC: Close HTTP client

    App->>App: Close AI client
    App->>App: Close MCP adapters

    Note over App: Shutdown complete
    App-->>K8s: Exit 0

    Note over K8s: terminationGracePeriodSeconds elapsed
    K8s->>App: SIGKILL (if still running)
```

## Kubernetes Configuration

```yaml
# deployment.yaml
spec:
  template:
    spec:
      # Total time allowed for graceful shutdown
      # Should be > job_timeout + buffer
      terminationGracePeriodSeconds: 330  # 5min + 30s

      containers:
        - name: cognigate
          lifecycle:
            preStop:
              exec:
                # Small delay to allow load balancer to drain
                command: ["/bin/sh", "-c", "sleep 5"]
```

## Shutdown State Diagram

```mermaid
stateDiagram-v2
    [*] --> Running: Application started

    Running --> ShuttingDown: SIGTERM received
    Running --> Running: Processing jobs

    ShuttingDown --> WaitingForJobs: Has active jobs
    ShuttingDown --> CleaningUp: No active jobs

    WaitingForJobs --> WaitingForJobs: Jobs still running\n(within timeout)
    WaitingForJobs --> CleaningUp: All jobs complete
    WaitingForJobs --> ForcedCleanup: Timeout reached

    ForcedCleanup --> ForcedCleanup: Cancel remaining jobs
    ForcedCleanup --> ForcedCleanup: Send failure receipts
    ForcedCleanup --> CleaningUp: All jobs handled

    CleaningUp --> CleaningUp: Close clients
    CleaningUp --> Stopped: Cleanup complete

    Stopped --> [*]: Process exit
```

## Timeline Example

```mermaid
gantt
    title Graceful Shutdown Timeline
    dateFormat ss
    axisFormat %S

    section Trigger
    SIGTERM received           :milestone, m1, 00, 0s

    section WorkPoller
    Stop accepting work        :a1, 00, 1s
    Wait for active jobs       :a2, after a1, 10s

    section Active Job
    Continue execution         :b1, 00, 8s
    Send completion receipt    :b2, after b1, 2s

    section Heartbeat
    Keep extending lease       :c1, 00, 8s
    Stop heartbeat            :c2, after b2, 1s

    section Cleanup
    Close HTTP clients         :d1, after a2, 2s
    Close MCP adapters         :d2, after d1, 1s
    Process exit              :milestone, m2, after d2, 0s
```

## Readiness Probe Behavior

During shutdown, the readiness probe returns 503:

```mermaid
sequenceDiagram
    participant LB as Load Balancer
    participant Pod as CogniGate Pod
    participant K8s as Kubernetes

    Note over Pod: Normal operation
    LB->>Pod: GET /ready
    Pod-->>LB: 200 OK {"ready": true}

    K8s->>Pod: SIGTERM
    Note over Pod: Shutdown initiated

    LB->>Pod: GET /ready
    Pod-->>LB: 503 Service Unavailable
    Note over Pod: {"detail": "Shutting down"}

    Note over LB: Remove pod from rotation
    Note over Pod: Continue processing active jobs
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `graceful_timeout` | 300s | Max wait for jobs in shutdown |
| `terminationGracePeriodSeconds` | 330s | K8s hard limit (timeout + buffer) |
| `preStop.sleep` | 5s | Allow LB to drain connections |
| `heartbeat_interval` | 60s | Lease extension frequency |
