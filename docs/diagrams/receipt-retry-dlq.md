# Receipt Retry and Dead Letter Queue

This diagram shows the receipt delivery mechanism with retry logic and DLQ fallback.

```mermaid
sequenceDiagram
    autonumber
    participant WP as WorkPoller
    participant AGC as AsyncGateClient
    participant AG as AsyncGate
    participant DLQ as DeadLetterQueue
    participant FS as Filesystem

    Note over WP: Job completed, need to send receipt

    WP->>AGC: send_receipt(receipt)
    AGC->>AGC: send_receipt_with_retry()

    loop Attempt 1 to max_retries (default: 5)
        AGC->>AG: POST /v1/tasks/{id}/complete

        alt HTTP 2xx Success
            AG-->>AGC: 200 OK
            AGC-->>WP: True (success)
            Note over WP: Receipt delivered
        else HTTP 4xx Client Error (except 429)
            AG-->>AGC: 4xx Error
            Note over AGC: Don't retry client errors
            AGC->>AGC: Break retry loop
        else HTTP 429 Too Many Requests
            AG-->>AGC: 429 Rate Limited
            Note over AGC: Retry with backoff
            AGC->>AGC: Wait (backoff_base ^ attempt)
        else HTTP 5xx Server Error
            AG-->>AGC: 5xx Error
            Note over AGC: Retry with backoff
            AGC->>AGC: Wait (backoff_base ^ attempt)
        else Connection/Timeout Error
            Note over AGC: Network failure
            AGC->>AGC: Wait (backoff_base ^ attempt)
        end
    end

    Note over AGC: All retries exhausted

    AGC->>DLQ: store(receipt)

    DLQ->>DLQ: Add to in-memory queue
    DLQ->>DLQ: Update metrics (dlq_size)

    DLQ->>FS: Persist to receipts.json
    FS-->>DLQ: Written

    DLQ-->>AGC: Stored
    AGC-->>WP: False (dead-lettered)

    Note over WP: Job tracking cleared
    Note over DLQ: Receipt awaits manual recovery
```

## Backoff Timing

With default `backoff_base=2.0`:

| Attempt | Delay (seconds) |
|---------|-----------------|
| 1 | 0 (immediate) |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |
| **Total** | **~30s worst case** |

## DLQ Storage Format

```json
// /var/lib/cognigate/dlq/receipts.json
[
  {
    "receipt": {
      "lease_id": "abc-123",
      "task_id": "task-456",
      "worker_id": "cognigate-0",
      "status": "complete",
      "timestamp": "2024-01-15T10:30:00Z",
      "summary": "Task completed successfully",
      "artifact_pointers": ["s3://bucket/results.json"]
    },
    "failed_at": "2024-01-15T10:30:45Z",
    "attempts": 5
  }
]
```

## DLQ Recovery Flow

```mermaid
flowchart TD
    A[Operator initiates recovery] --> B[Read receipts.json]
    B --> C{Entries exist?}
    C -->|No| D[Done - queue empty]
    C -->|Yes| E[For each entry]

    E --> F{AsyncGate reachable?}
    F -->|No| G[Wait and retry later]
    F -->|Yes| H[POST receipt to AsyncGate]

    H --> I{Success?}
    I -->|Yes| J[Remove from DLQ]
    I -->|No| K[Log error, skip entry]

    J --> L{More entries?}
    K --> L
    L -->|Yes| E
    L -->|No| M[Persist updated queue]
    M --> D
```

## Metrics

| Metric | Description |
|--------|-------------|
| `cognigate_receipts_total{status, success}` | Receipt send attempts |
| `cognigate_receipt_retries_total` | Number of retry attempts |
| `cognigate_dead_letter_queue_size` | Current DLQ size |
