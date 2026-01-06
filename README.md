CogniGate

CogniGate is a leased cognitive execution worker.

It performs bounded, tool-mediated AI cognition on behalf of other systems, materializes durable artifacts, and reports lifecycle state through receipts.

CogniGate does not think for itself.
It executes cognition under lease, with explicit constraints, explicit tools, and explicit outputs.

What CogniGate Does

Accepts leased work from AsyncGate

Constructs prompts from static instruction profiles and job-scoped payloads

Produces a machine-readable plan (advisory, not authoritative)

Executes cognition step-by-step using a minimal, advertised tool surface

Delivers outputs to explicitly defined sinks

Reports progress and completion via receipts, not logs

All cognition is:

Job-scoped

Stateless

Externally materialized

Receipted at every state transition

What CogniGate Is Not

CogniGate intentionally does not:

maintain conversation or memory

own goals or intent

decide where outputs go

expose third-party APIs directly to models

store or emit full reasoning chains

operate as a chatbot or assistant

These exclusions are design constraints, not omissions.

Architectural Role

CogniGate sits below intent and memory, and above raw execution.

It pairs with:

AsyncGate for work leasing and coordination

MemoryGate (optionally) for persistence handled elsewhere

MCP servers for controlled, least-privilege access to external systems

CogniGate treats AI models as planning and cognition engines, never as autonomous actors.

Design Principles

Cognition under lease

Artifacts over messages

Receipts over logs

Execution over intent

Boring in the right places

CogniGate exists to make AI cognition interruptible, auditable, recoverable, and safe to embed in real systems—without pretending it’s a mind.

CogniGate Spec
