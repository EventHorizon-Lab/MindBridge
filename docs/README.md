# MindBridge documentation

Start with the path that matches what you are doing.

## I want to try it

1. [Quickstart](quickstart.md) — a local stack answering recalls in about fifteen minutes.
2. [Concepts](concepts.md) — what an Observation, Event, Claim, and Memory each are.
3. [Python SDK](api/python-sdk.md) — the client you will actually write against.

## I am building against it

| Guide | Covers |
| --- | --- |
| [REST API](api/rest.md) | Every endpoint, schema, and error code. |
| [Python SDK](api/python-sdk.md) | The async client, job streaming, retry policy. |
| [MCP tools](api/mcp.md) | Seven agent-facing tools and their argument shapes. |
| [Concepts](concepts.md) | The domain model behind those contracts. |

## I am deploying it

| Guide | Covers |
| --- | --- |
| [Deployment](deployment.md) | Process topology, migrations, scaling. |
| [Configuration](configuration.md) | Every environment variable and its failure mode. |
| [Operations](operations.md) | Consolidation, lifecycle, telemetry, drills. |
| [Edge deployment](edge.md) | Capture handoff, on-device identity, outbox sync. |
| [CLI](api/cli.md) | Every command, flag, and exit code. |
| [Troubleshooting](troubleshooting.md) | Symptoms and what they actually indicate. |

## I am contributing

| Guide | Covers |
| --- | --- |
| [Contributing](../CONTRIBUTING.md) | Setup, quality gates, review standards. |
| [Architecture](architecture.md) | Code layout, write and read paths, boundaries. |
| [Technical architecture](technical-architecture.md) | The full design specification (Chinese). |
| [Plugin architecture](plugin-architecture.md) | Writing a generator or embedder adapter. |
| [AGENTS.md](../AGENTS.md) | Repository conventions, for people and agents alike. |

## I am evaluating it

| Guide | Covers |
| --- | --- |
| [Benchmarking](benchmarking.md) | Running the harness, and what its numbers mean. |
| [SOTA baselines](benchmarks-sota.md) | Comparison targets per benchmark. |
| [Edge identity](edge-identity-sota.md) | On-device model selection and validation. |

Read [the stance](benchmarking.md#the-stance) before quoting any number.

## Reference

- [System architecture diagram](architecture-diagram.html)
- [Design specifications](superpowers/specs/) and [implementation plans](superpowers/plans/)
- [Security policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)

## Conventions in these documents

Commands assume the repository root and [uv](https://docs.astral.sh/uv/). Each shows the extras it
needs — `uv run --extra server ...` — because installing only what a process runs is the
recommended deployment shape, not an optimisation.

Where a document explains why something is the way it is, that reasoning is usually load-bearing:
it records a constraint discovered in production or a decision with a stated cost. The
[ADR log](technical-architecture.md#17-关键架构决策记录) names the condition that would justify
revisiting each one.
