# CogniGate Architecture Diagrams

This directory contains Mermaid sequence and state diagrams documenting CogniGate's architecture and operational flows.

## Diagrams

### Core Flows

| Diagram | Description |
|---------|-------------|
| [Lease to Receipt Flow](lease-to-receipt-flow.md) | Complete job lifecycle from work claim to receipt delivery |
| [Planning Phase](planning-phase.md) | Instruction profile loading and system prompt construction |
| [Tool Invocation](tool-invocation.md) | MCP tool routing and execution with circuit breaker |

### Resilience Patterns

| Diagram | Description |
|---------|-------------|
| [Receipt Retry & DLQ](receipt-retry-dlq.md) | Receipt delivery with exponential backoff and dead letter queue |
| [Circuit Breaker States](circuit-breaker-states.md) | Circuit breaker state machine for external service protection |
| [Graceful Shutdown](graceful-shutdown.md) | Shutdown sequence ensuring job completion |

## Viewing Diagrams

These diagrams use [Mermaid](https://mermaid.js.org/) syntax. They can be viewed:

1. **GitHub/GitLab**: Renders Mermaid automatically in markdown
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Mermaid Live Editor**: https://mermaid.live/
4. **Documentation Sites**: Most modern doc tools support Mermaid

## System Overview

```mermaid
flowchart TB
    subgraph External
        AG[AsyncGate]
        AI[AI Provider]
        MCP1[MCP Server 1]
        MCP2[MCP Server 2]
    end

    subgraph CogniGate
        API[REST API]
        WP[WorkPoller]
        JE[JobExecutor]
        TE[ToolExecutor]
        CB[Circuit Breakers]
        DLQ[Dead Letter Queue]
    end

    subgraph Observability
        PROM[Prometheus]
        LOG[Structured Logs]
    end

    AG <-->|Lease/Receipt| WP
    WP --> JE
    JE <-->|Chat| AI
    JE --> TE
    TE --> CB
    CB --> MCP1
    CB --> MCP2
    WP --> DLQ

    API --> PROM
    JE --> LOG
```

## Component Relationships

```mermaid
classDiagram
    class WorkPoller {
        +start()
        +stop()
        +stop_gracefully(timeout)
        -_poll_and_dispatch()
        -_handle_job()
        -_start_heartbeat()
    }

    class JobExecutor {
        +execute(lease)
        +cancel_job(lease_id)
        -_build_system_prompt()
        -_execute_step()
    }

    class AsyncGateClient {
        +poll_for_work()
        +send_receipt()
        +extend_lease()
        -dead_letter_queue
    }

    class ToolExecutor {
        +execute_tool(name, args)
        -_route_to_mcp()
        -_route_to_sink()
    }

    class CircuitBreaker {
        +call(func)
        +state
        -failure_count
        -last_failure_time
    }

    class MCPAdapter {
        +call_tool(name, args)
        +circuit_breaker
        +read_only
    }

    WorkPoller --> AsyncGateClient
    WorkPoller --> JobExecutor
    JobExecutor --> ToolExecutor
    ToolExecutor --> MCPAdapter
    MCPAdapter --> CircuitBreaker
    AsyncGateClient --> DeadLetterQueue
```

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        L[Lease from AsyncGate]
    end

    subgraph Processing
        P[Profile Loading]
        E[AI Execution Loop]
        T[Tool Calls]
    end

    subgraph Output
        R[Receipt]
        A[Artifacts]
    end

    L --> P --> E
    E <--> T
    E --> R
    T --> A
    R --> AG[AsyncGate]
```
