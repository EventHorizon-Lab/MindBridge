# MindBridge documentation

Use the shortest path that matches your task. Each page owns one subject so contracts and examples
have one place to stay current.

## Choose a path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Understand what MindBridge can do | [Product capabilities](product-capabilities.md) | [Design principles](design-principles.md) |
| Try MindBridge end to end | [Quick start](quickstart.md) | [Core concepts](concepts.md) |
| Build a Python integration | [Configuration](configuration.md) | [Python SDK](api/python-sdk.md) |
| Give an agent a bounded context view | [Context compilation](context-compilation.md) | [Python SDK](api/python-sdk.md) |
| Expose memory to another process | [REST](api/rest.md), [MCP](api/mcp.md), or [CLI](api/cli.md) | [Deployment](deployment.md) |
| Run a durable instance | [Architecture](architecture.md) | [Operations](operations.md) and [troubleshooting](troubleshooting.md) |
| Evaluate memory quality | [Benchmarking](benchmarking.md) | [Example evaluation configuration](examples/eval.example.yaml) |
| Understand or extend the design | [Design principles](design-principles.md) | [Plugin architecture](plugin-architecture.md) |
| Follow where the product is going | [Context OS direction](context-os.md) | [Design principles](design-principles.md) |

## Learn

1. [Product capabilities](product-capabilities.md) — see the implemented product surface, use
   cases, boundaries, and extension model.
2. [Quick start](quickstart.md) — install MindBridge and exercise its core capabilities.
3. [Core concepts](concepts.md) — understand records, content, retrieval, and directory ownership.
4. [Configuration](configuration.md) — select bundled adapters or inject application backends.
5. [Memory types, time, and decay](memory-types-time-and-decay.md) — control cognitive role and
   temporal ranking.
6. [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) — ingest
   completed observations and derive grounded interaction records.
7. [Context compilation](context-compilation.md) — compile a bounded, structured context bundle for
   one goal instead of a flat hit list.

## Integrate

- [Python SDK](api/python-sdk.md) — complete `mindbridge` root-import contract.
- [REST API](api/rest.md) — `/v1` requests, responses, errors, and limits.
- [MCP tools](api/mcp.md) — the fifteen tool schemas and transport boundary.
- [Command line](api/cli.md) — commands, input forms, JSON output, and exit codes.

## Deploy and operate

- [Architecture](architecture.md) — storage authority, write and retrieval paths, concurrency, and
  model boundaries.
- [Deployment](deployment.md) — embedded, REST, MCP, and edge process shapes.
- [Operations](operations.md) — health, backup, recovery, index maintenance, and telemetry.
- [Troubleshooting](troubleshooting.md) — diagnose startup, retrieval, content, and provider
  failures.
- [Security](../SECURITY.md) — trust boundaries, data exposure, and deployment hardening.

## Understand the direction

- [Context OS direction](context-os.md) — the long-term product boundary, fast and slow context
  planes, agentic memory management, and context compilation.
- [Product goals and design principles](design-principles.md) — the product target, the design
  goals, and the criteria a change must answer. States direction, not implemented status.
- [Plugin architecture](plugin-architecture.md) — the kernel/plugin boundary and the admission rule
  a new public capability must satisfy.
- [Affective memory direction](affective-memory.md) — how affect is stored as sourced, timed
  hypothesis rather than fact, what exists today, and the gates a richer capability must pass.
- [Context OS direction](context-os.md) — the fast and slow context planes, agentic memory
  management, and the evolution gates the current release is working through.

## Evaluate

- [Benchmarking](benchmarking.md) — reproducible behavior evaluation and local-index measurement.
- [Annotated example configuration](examples/eval.example.yaml) — every evaluation slot and the
  order in which a run uses it.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md). When a fact
changes, update its owning page and link to it elsewhere instead of copying it.
