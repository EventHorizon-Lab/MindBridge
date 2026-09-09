# MindBridge documentation

Use the shortest path that matches your task. Each page owns one subject so contracts and examples
have one place to stay current.

## Choose a path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Understand what MindBridge can do | [Product capabilities](product-capabilities.md) | [Design principles](design-principles.md) |
| Try MindBridge end to end | [Quick start](quickstart.md) | [Core concepts](concepts.md) |
| Build a Python integration | [Configuration](configuration.md) | [Python SDK](api/python-sdk.md) |
| Use a local WeMM embedding service | [Local WeMM deployment](use-local-wemm.md) | [Configuration](configuration.md) |
| Give an agent a bounded context view | [Context compilation](context-compilation.md) | [Python SDK](api/python-sdk.md) |
| Expose memory to another process | [REST](api/rest.md), [MCP](api/mcp.md), or [CLI](api/cli.md) | [Deployment](deployment.md) |
| Run a durable instance | [Architecture](architecture.md) | [Operations](operations.md) and [troubleshooting](troubleshooting.md) |
| Evaluate memory quality | [Benchmarking](benchmarking.md) | [Example evaluation configuration](examples/eval.example.yaml) |
| Review the memory backend evidence | [Memory backend audit](research/memory-backend-audit-2026-09-07.md) | [Benchmarking](benchmarking.md) |
| Compare current memory backends | [Competitor memory backends](research/competitor-memory-backends-2026-09-08.md) | [Hybrid score completion](research/hybrid-score-completion-2026-09-08.md) |
| Review the measured memory changes | [Memory optimization report](research/memory-optimization-2026-09-07.md) | [Benchmarking](benchmarking.md) |
| Review the 2026-09-08 memory backend work | [Chinese conclusions](research/2026-09-09-memory-backend-conclusions-zh.md) | [Research](research/2026-09-08-memory-backend-research.md), [validation](research/2026-09-08-memory-backend-validation.md), and [evaluation](research/2026-09-08-memory-backend-evaluation.md) |
| Review companion evidence obligations | [Companion evidence obligations](research/2026-09-09-companion-evidence-obligations-zh.md) | [Affective memory](affective-memory.md) and [Context compilation](context-compilation.md) |
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
- [Local WeMM deployment](use-local-wemm.md) — configure a local WeMM embedding service with the
  public SDK.

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
  planes, agentic memory management, context compilation, and the evolution gates the current
  release is working through.
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
- [Memory backend audit](research/memory-backend-audit-2026-09-07.md) — source-level capability,
  competitor, benchmark-trust, and research-gate review dated 2026-09-07.
- [Memory backend source ledger](research/memory-backend-sources-2026-09-07.md) — pinned competitor
  commits, inspected paths, evidence limits, and falsification tests for that audit.
- [Competitor memory backends](research/competitor-memory-backends-2026-09-08.md) — current
  architecture, cost, provenance, and transfer evidence for agent and multimodal memory systems.
- [Hybrid score completion](research/hybrid-score-completion-2026-09-08.md) — design rationale and
  acceptance protocol for completing missing dense signals from durable vectors.
- [Historical out-of-scope EgoLife diagnostics](research/historical-egolife-diagnostics-2026-09-08.md)
  — retained negative results and rejected hypotheses from a benchmark removed from acceptance.
- [Timeline neighbor-window risk review](research/timeline-neighbor-window-risk-review-2026-09-08.md)
  — safety constraints and experiment gates for a proposed same-sequence context window.
- [Next raw-video memory experiment](research/video-memory-next-experiment-2026-09-08.md) — a
  retrieval-only plan to separate observation, temporal-neighborhood, and evidence-budget failures.
- [Memory optimization report](research/memory-optimization-2026-09-07.md) — paired benchmark
  results, ablations, and the retained memory changes dated 2026-09-07.
- [Memory backend research](research/2026-09-08-memory-backend-research.md) — evidence-closure
  design, prior-art analysis, and acceptance criteria.
- [Memory backend validation](research/2026-09-08-memory-backend-validation.md) — independent
  review of the implemented compiler and historical-identity constraints.
- [Memory backend evaluation](research/2026-09-08-memory-backend-evaluation.md) — frozen
  evaluation protocol, raw and formed tracks, and paired empirical results.
- [Memory backend conclusions in Chinese](research/2026-09-09-memory-backend-conclusions-zh.md) —
  concise implemented-versus-prior-art boundary, causal results, failures, and next bottleneck.
- [Companion evidence obligations in Chinese](research/2026-09-09-companion-evidence-obligations-zh.md)
  — incremental primary-source synthesis on indexed competition completion, affect source roles,
  multimodal and embodied memory, benchmark coverage, and falsifiable acceptance plans.
- [Evidence excerpt experiment](research/2026-09-09-evidence-excerpt-experiment.md) — isolated
  schema-18 prototype for digest-bound partial raw text under the existing context budget.
- [Compact context prototype](compact-context-prototype.md) — optional request-local ID aliases,
  typed citation resolution, and excerpt coverage boundaries without changing compilation.
- [Annotated example configuration](examples/eval.example.yaml) — every evaluation slot and the
  order in which a run uses it.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md). When a fact
changes, update its owning page and link to it elsewhere instead of copying it.
