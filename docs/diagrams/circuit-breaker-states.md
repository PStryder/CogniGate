# Circuit Breaker State Transitions

This diagram shows the circuit breaker pattern implementation for external service protection.

```mermaid
stateDiagram-v2
    [*] --> Closed: Initial state

    Closed --> Closed: Request succeeds\n(reset failure count)
    Closed --> Open: Failures >= threshold

    Open --> Open: Requests fail-fast\n(no external calls)
    Open --> HalfOpen: recovery_timeout elapsed

    HalfOpen --> Closed: Test request succeeds\n(reset counters)
    HalfOpen --> Open: Test request fails\n(reset timeout)
    HalfOpen --> HalfOpen: More test calls allowed\n(calls < half_open_max)

    note right of Closed
        Normal operation
        All requests pass through
        Failures tracked
    end note

    note right of Open
        Protection mode
        Requests rejected immediately
        Timer running for recovery
    end note

    note right of HalfOpen
        Recovery testing
        Limited requests allowed
        Success = recover
        Failure = back to open
    end note
```

## Detailed State Machine

```mermaid
sequenceDiagram
    participant C as Client
    participant CB as CircuitBreaker
    participant S as Service

    Note over CB: State: CLOSED

    rect rgb(200, 255, 200)
        Note over CB: Normal Operation
        C->>CB: Request 1
        CB->>S: Forward request
        S-->>CB: Success
        CB-->>C: Success
        CB->>CB: failures = 0
    end

    rect rgb(255, 255, 200)
        Note over CB: Failures accumulating
        loop Failures 1 to threshold-1
            C->>CB: Request
            CB->>S: Forward request
            S-->>CB: Error
            CB->>CB: failures++
            CB-->>C: Error
        end
    end

    rect rgb(255, 200, 200)
        Note over CB: Threshold reached
        C->>CB: Request N
        CB->>S: Forward request
        S-->>CB: Error
        CB->>CB: failures >= threshold
        CB->>CB: State = OPEN
        CB-->>C: Error
    end

    Note over CB: State: OPEN

    rect rgb(255, 200, 200)
        Note over CB: Fail-fast mode
        C->>CB: Request
        CB-->>C: CircuitBreakerError (no call to service)
        Note over CB: Waiting for recovery_timeout...
    end

    Note over CB: recovery_timeout elapsed
    CB->>CB: State = HALF_OPEN

    Note over CB: State: HALF_OPEN

    rect rgb(200, 200, 255)
        Note over CB: Testing recovery
        C->>CB: Test request 1
        CB->>S: Forward (limited)

        alt Service recovered
            S-->>CB: Success
            CB->>CB: State = CLOSED
            CB-->>C: Success
            Note over CB: Circuit recovered!
        else Still failing
            S-->>CB: Error
            CB->>CB: State = OPEN
            CB-->>C: Error
            Note over CB: Back to open, restart timer
        end
    end
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Failures before opening circuit |
| `recovery_timeout` | 60.0s | Time to wait before half-open |
| `half_open_max_calls` | 3 | Test calls allowed in half-open |
| `excluded_exceptions` | None | Exceptions that don't count as failures |

## Circuit Breaker Registry

```mermaid
flowchart TD
    A[Service Call] --> B{Circuit exists?}
    B -->|No| C[Create new CircuitBreaker]
    C --> D[Register in registry]
    D --> E[Use circuit breaker]
    B -->|Yes| E

    E --> F{State?}
    F -->|Closed| G[Execute call]
    F -->|Open| H[Throw CircuitBreakerError]
    F -->|HalfOpen| I{Calls < max?}
    I -->|Yes| G
    I -->|No| H

    G --> J{Result?}
    J -->|Success| K[Record success]
    J -->|Failure| L[Record failure]

    K --> M[Return result]
    L --> N{Check threshold}
    N -->|Not reached| O[Return error]
    N -->|Reached| P[Open circuit]
    P --> O
```

## Per-Service Circuit Breakers

CogniGate creates separate circuit breakers for:

| Service | Circuit Name | Purpose |
|---------|--------------|---------|
| AI Provider | `ai_provider` | Protect against AI API failures |
| MCP Server 1 | `mcp_<name>` | Isolate MCP endpoint failures |
| MCP Server 2 | `mcp_<name>` | Each MCP has its own circuit |

```mermaid
flowchart LR
    subgraph CogniGate
        CB1[Circuit: ai_provider]
        CB2[Circuit: mcp_filesystem]
        CB3[Circuit: mcp_database]
    end

    CB1 --> AI[AI Provider]
    CB2 --> MCP1[MCP Filesystem]
    CB3 --> MCP2[MCP Database]

    style CB1 fill:#90EE90
    style CB2 fill:#90EE90
    style CB3 fill:#FFB6C1

    Note1[All circuits independent]
```

## Monitoring Metrics

```promql
# Current circuit breaker states
cognigate_circuit_breaker_state{name="ai_provider", state="closed"} 1
cognigate_circuit_breaker_state{name="ai_provider", state="open"} 0
cognigate_circuit_breaker_state{name="ai_provider", state="half_open"} 0

# Failure counts
cognigate_circuit_breaker_failures_total{name="ai_provider"} 12

# Alert query: Any circuit open
cognigate_circuit_breaker_state{state="open"} == 1
```
