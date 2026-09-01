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
- [MCP tools](api/mcp.md) — six tool schemas and transport boundary.
- [Command line](api/cli.md) — commands, input forms, JSON output, and exit codes.

## Operate MindBridge

- [Architecture](architecture.md) — storage, consistency, concurrency, and model boundaries.
- [Deployment](deployment.md) — embedded, REST, and edge deployment shapes.
- [Operations](operations.md) — backup, recovery, index maintenance, and telemetry.
- [Security](../SECURITY.md) — trust boundaries, data exposure, and deployment hardening.
- [Troubleshooting](troubleshooting.md) — diagnose common startup, retrieval, and model failures.

## Go deeper

- [Memory types, time, and decay](memory-types-time-and-decay.md) — role and ranking semantics.
- [Omni streaming and interaction memory](omni-streaming-and-interaction-memory.md) — completed
  observations, speculative recall, and derived records.
- [Benchmarking](benchmarking.md) — reproducible evaluation and local-index measurement.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md).
When a fact changes, update its owning page and link to it instead of copying it into another guide.
