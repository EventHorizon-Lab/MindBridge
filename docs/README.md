# MindBridge documentation

Each page has one job. Start with the learning path, then use the reference or operating page that
owns the fact you need.

## Learn the product

1. [Quick start](quickstart.md) — install MindBridge and run the first local memory.
2. [Core concepts](concepts.md) — understand content, records, retrieval, and isolation.
3. [Configuration](configuration.md) — choose bundled adapters or inject your own.

## API reference

- [Python SDK](api/python-sdk.md) — complete `mindbridge` root-import contract.
- [REST API](api/rest.md) — `/v1` schemas, routes, errors, and limits.
- [MCP tools](api/mcp.md) — fourteen tool schemas and transport boundary.
- [Command line](api/cli.md) — commands, input forms, JSON output, and exit codes.

## Operate MindBridge

- [Architecture](architecture.md) — storage, consistency, concurrency, and model boundaries.
- [Deployment](deployment.md) — embedded, REST, and edge deployment shapes.
- [Operations](operations.md) — backup, recovery, index maintenance, and telemetry.
- [Security](../SECURITY.md) — trust boundaries, data exposure, and deployment hardening.
- [Troubleshooting](troubleshooting.md) — diagnose common startup, retrieval, and model failures.

## Understand the direction

- [Product goals and design principles](design-principles.md) — the product target, the design
  goals, and the criteria a change must answer. States direction, not implemented status.
- [Plugin architecture](plugin-architecture.md) — the kernel/plugin boundary and the admission rule
  a new public capability must satisfy.

## Go deeper

- [Memory types, time, and decay](memory-types-time-and-decay.md) — role and ranking semantics.
- [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) — completed
  observations, speculative recall, and derived records.
- [Benchmarking](benchmarking.md) — reproducible evaluation and local-index measurement.
- [Competitive memory-system review](competitive-memory-systems.md) — source audit of ABot,
  M3-Agent, VoiceMem, eMEM, MIRIX, Graphiti, Mem0, and TeleMem, plus the resulting gap roadmap.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md).
When a fact changes, update its owning page and link to it instead of copying it into another guide.
