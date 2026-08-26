# MindBridge documentation

Start with the path that matches what you are doing.

## I want to try it

1. [Quickstart](quickstart.md) — a local stack answering recalls once its model and object-store
   dependencies are available.
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
| [Technical architecture](technical-architecture.md) | Architecture decisions and implementation roadmap (Chinese). |
| [Plugin architecture](plugin-architecture.md) | Writing a generator or embedder adapter. |
| [AGENTS.md](../AGENTS.md) | Repository conventions, for people and agents alike. |

## I am evaluating it

| Guide | Covers |
| --- | --- |
| [Benchmarking](benchmarking.md) | Running the harness, and what its numbers mean. |
| [SOTA baselines](benchmarks-sota.md) | Comparison targets per benchmark. |
| [Edge identity](edge-identity-sota.md) | Dated implementation status, model selection, and validation gates. |

Read [the stance](benchmarking.md#the-stance) before quoting any number.

## Reference

- [System architecture diagram](architecture-diagram.html)
- [Design specifications](superpowers/specs/) and [implementation plans](superpowers/plans/) —
  historical snapshots; their commands and constraints may no longer describe `master`.
- [Security policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)

## Sources of truth

Use the narrowest current source for the question:

- Public request, response, and error shapes come from `mindbridge.contracts`, the generated
  OpenAPI document, and the MCP contract snapshots.
- Executable commands and defaults come from each command's `--help`.
- Runtime behavior and deployment requirements come from [Architecture](architecture.md),
  [Configuration](configuration.md), and [Deployment](deployment.md).
- [Technical architecture](technical-architecture.md) records decisions, measured rationale, and
  roadmap items. Its phase tables distinguish shipped behavior from target design.

## Conventions in these documents

Commands assume the repository root and [uv](https://docs.astral.sh/uv/). Each shows the extras it
needs — `uv run --extra server ...` — because installing only what a process runs is the
recommended deployment shape, not an optimisation.

`uv sync` makes the environment match exactly the extras it is given, so a later `uv sync` with a
different set uninstalls the earlier one's packages. Per host that is the point: one role, one
environment. On a workstation that plays several roles, name every extra in a single command,
take all of them with `uv sync --all-extras`, or add to what is already installed with
`uv sync --inexact --extra ...`. `uv run --extra ...` only ever adds, so it is safe to mix.

Where a document explains why something is the way it is, that reasoning is usually load-bearing:
it records a constraint discovered in production or a decision with a stated cost. The
[ADR log](technical-architecture.md#17-关键架构决策记录) names the condition that would justify
revisiting each one.
