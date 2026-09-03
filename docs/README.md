# MindBridge documentation

Use the shortest path that matches your task. Each page owns one subject so contracts and examples
have one place to stay current.

## Choose a path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Try local memory | [Quick start](quickstart.md) | [Core concepts](concepts.md) |
| Understand the product direction | [Context OS direction](context-os.md) | [Architecture](architecture.md) |
| Build a Python integration | [Configuration](configuration.md) | [Python SDK](api/python-sdk.md) |
| Give an agent a bounded context view | [Context compilation](context-compilation.md) | [Python SDK](api/python-sdk.md) |
| Expose memory to another process | [REST](api/rest.md), [MCP](api/mcp.md), or [CLI](api/cli.md) | [Deployment](deployment.md) |
| Run a durable instance | [Architecture](architecture.md) | [Operations](operations.md) and [troubleshooting](troubleshooting.md) |
| Evaluate retrieval quality | [Benchmarking](benchmarking.md) | [Competitive review](competitive-memory-systems.md) |

## Learn

1. [Context OS direction](context-os.md) — understand the long-term product boundary, fast and slow
   context planes, agentic memory management, and context compilation.
2. [Quick start](quickstart.md) — install MindBridge and run one local memory.
3. [Core concepts](concepts.md) — understand records, content, retrieval, and directory ownership.
4. [Configuration](configuration.md) — select bundled adapters or inject application backends.
5. [Memory types, time, and decay](memory-types-time-and-decay.md) — control cognitive role and
   temporal ranking.
6. [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) — ingest
   completed observations and derive grounded interaction records.
7. [Context compilation](context-compilation.md) — compile a bounded, structured context bundle
   and advertise what one instance can do.

## Integrate

- [Python SDK](api/python-sdk.md) — complete `mindbridge` root-import contract.
- [REST API](api/rest.md) — `/v1` requests, responses, errors, and limits.
- [MCP tools](api/mcp.md) — the seven tool schemas and transport boundary.
- [Command line](api/cli.md) — commands, input forms, JSON output, and exit codes.

## Deploy and operate

- [Architecture](architecture.md) — storage authority, write and retrieval paths, concurrency, and
  model boundaries.
- [Deployment](deployment.md) — embedded, REST, MCP, and edge process shapes.
- [Operations](operations.md) — health, backup, recovery, index maintenance, and telemetry.
- [Troubleshooting](troubleshooting.md) — diagnose startup, retrieval, content, and provider
  failures.
- [Security](../SECURITY.md) — trust boundaries, data exposure, and deployment hardening.

## Evaluate

- [Benchmarking](benchmarking.md) — reproducible behavior evaluation and local-index measurement.
- [Competitive memory-system review](competitive-memory-systems.md) — evidence-backed comparison
  and the design decisions it informed.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md). When a fact
changes, update its owning page and link to it elsewhere instead of copying it.
