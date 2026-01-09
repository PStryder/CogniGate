# Lease to Receipt Flow

This diagram shows the complete flow from work claim to receipt delivery.

```mermaid
sequenceDiagram
    autonumber
    participant AG as AsyncGate
    participant WP as WorkPoller
    participant JE as JobExecutor
    participant AI as AI Provider
    participant TE as ToolExecutor
    participant MCP as MCP Server

    Note over WP: Polling loop starts

    loop Every polling_interval
        WP->>AG: POST /v1/leases/claim
        alt No work available
            AG-->>WP: 200 OK (empty tasks)
        else Work available
            AG-->>WP: 200 OK (lease)

            Note over WP: Lease received
            WP->>WP: Track active job
            WP->>WP: Start heartbeat task

            WP->>AG: POST /v1/tasks/{id}/progress (RUNNING)
            AG-->>WP: 200 OK

            WP->>JE: execute(lease)

            Note over JE: Job execution begins
            JE->>JE: Load instruction profile
            JE->>JE: Build system prompt

            loop Execution steps (max iterations)
                JE->>JE: Check cancellation
                JE->>AI: Chat completion request
                AI-->>JE: Response with tool_calls

                alt Has tool calls
                    loop For each tool call
                        JE->>TE: execute_tool(call)
                        TE->>MCP: Call MCP tool
                        MCP-->>TE: Tool result
                        TE-->>JE: Result
                    end
                    JE->>JE: Append results to conversation
                else No tool calls (final response)
                    JE->>JE: Extract final response
                    Note over JE: Job complete
                end
            end

            JE-->>WP: Receipt (COMPLETE/FAILED)

            WP->>WP: Stop heartbeat
            WP->>AG: POST /v1/tasks/{id}/complete

            alt Receipt delivery success
                AG-->>WP: 200 OK
                WP->>WP: Clear job tracking
            else Receipt delivery fails
                Note over WP: Retry with backoff
                loop Max retries
                    WP->>AG: POST /v1/tasks/{id}/complete
                    alt Success
                        AG-->>WP: 200 OK
                    else Still failing
                        WP->>WP: Exponential backoff wait
                    end
                end
                Note over WP: Dead letter queue
                WP->>WP: Store in DLQ
            end
        end
    end
```

## Key Points

1. **Polling**: WorkPoller continuously polls AsyncGate for available work
2. **Lease Tracking**: Active jobs are tracked for graceful shutdown support
3. **Heartbeat**: Background task extends lease during long-running jobs
4. **Execution**: JobExecutor handles the AI conversation loop
5. **Tool Calls**: ToolExecutor dispatches to MCP servers
6. **Receipt Delivery**: Retries with exponential backoff, falls back to DLQ
