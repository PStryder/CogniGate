# Tool Invocation with MCP

This diagram shows how tool calls are routed through the MCP adapter system.

```mermaid
sequenceDiagram
    autonumber
    participant JE as JobExecutor
    participant TE as ToolExecutor
    participant SR as SinkRegistry
    participant MR as MCPAdapterRegistry
    participant CB as CircuitBreaker
    participant MCP as MCP Server

    Note over JE: AI response contains tool_calls

    loop For each tool_call
        JE->>TE: execute_tool(name, arguments)

        TE->>TE: Parse tool name (format: sink.tool_name)

        alt Is sink tool
            TE->>SR: get_sink(sink_name)
            SR-->>TE: Sink instance
            TE->>TE: Execute sink operation
        else Is MCP tool
            TE->>MR: get_adapter(mcp_name)
            MR-->>TE: MCPAdapter instance

            Note over TE: Check read-only constraint
            alt Write operation on read-only adapter
                TE-->>JE: Error: write not allowed
            else Operation allowed
                TE->>CB: call(mcp_request)

                alt Circuit breaker closed
                    CB->>MCP: HTTP POST /tools/{tool}

                    alt Success
                        MCP-->>CB: 200 OK (result)
                        CB-->>TE: Result
                        CB->>CB: Reset failure count
                    else Failure
                        MCP-->>CB: Error
                        CB->>CB: Increment failures

                        alt Failures >= threshold
                            CB->>CB: State = OPEN
                            Note over CB: Circuit opens
                        end

                        CB-->>TE: Error
                    end

                else Circuit breaker open
                    Note over CB: Fail fast
                    CB-->>TE: CircuitBreakerError

                else Circuit breaker half-open
                    Note over CB: Limited test requests
                    CB->>MCP: Test request
                    alt Success
                        MCP-->>CB: 200 OK
                        CB->>CB: State = CLOSED
                        CB-->>TE: Result
                    else Failure
                        MCP-->>CB: Error
                        CB->>CB: State = OPEN
                        CB-->>TE: Error
                    end
                end
            end
        end

        TE-->>JE: Tool result (success/error)
    end

    JE->>JE: Append tool results to conversation
    JE->>JE: Continue to next AI call
```

## Tool Routing Logic

```mermaid
flowchart TD
    A[Tool Call] --> B{Parse tool name}
    B --> C{Has sink prefix?}
    C -->|Yes| D[Route to SinkRegistry]
    C -->|No| E{Has MCP prefix?}
    E -->|Yes| F[Route to MCPAdapter]
    E -->|No| G{Is builtin tool?}
    G -->|Yes| H[Execute builtin]
    G -->|No| I[Error: Unknown tool]

    D --> J{Sink exists?}
    J -->|Yes| K[Execute sink method]
    J -->|No| I

    F --> L{Adapter exists?}
    L -->|Yes| M{Read-only check}
    L -->|No| I

    M -->|Pass| N[Call via CircuitBreaker]
    M -->|Fail| O[Error: Write not allowed]

    K --> P[Return result]
    N --> P
    H --> P
```

## MCP Adapter Configuration

```yaml
# /etc/cognigate/mcp.yaml
mcp_endpoints:
  - name: filesystem
    endpoint: http://mcp-filesystem:8080
    auth_token: ${MCP_FS_TOKEN}
    read_only: true
    enabled: true

  - name: database
    endpoint: http://mcp-database:8080
    auth_token: ${MCP_DB_TOKEN}
    read_only: false  # Allow writes
    enabled: true

  - name: web
    endpoint: http://mcp-web:8080
    read_only: true
    enabled: true
```

## Circuit Breaker Per Adapter

Each MCP adapter has its own circuit breaker instance:

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open: Failures >= threshold
    Open --> HalfOpen: Recovery timeout elapsed
    HalfOpen --> Closed: Test request succeeds
    HalfOpen --> Open: Test request fails
    Closed --> Closed: Request succeeds
```
