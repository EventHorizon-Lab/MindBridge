# MindBridge documentation

Use the shortest path that matches your task. Each page owns one subject so contracts and examples
have one place to stay current.

The guides and API references describe this checkout's `0.2.0` package and Python 3.10–3.14
contract.

## Choose a path

| Goal | Start here | Continue with |
| --- | --- | --- |
| Understand what MindBridge can do | [Product capabilities](product-capabilities.md) | [Design principles](design-principles.md) |
| Try MindBridge end to end | [Quick start](quickstart.md) | [Core concepts](concepts.md) |
| Build a Python integration | [Configuration](configuration.md) | [Python SDK](api/python-sdk.md) |
| Use a local WeMM embedding service | [Local WeMM deployment](use-local-wemm.md) | [Configuration](configuration.md) |
| Use EmbeddingGemma 2 through vLLM | [Model server](deployment.md#embeddinggemma-2-model-server) | [Configuration](configuration.md#embeddinggemma-2-through-vllm) |
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
- [Local WeMM deployment](use-local-wemm.md) — configure a local WeMM embedding service with the
  public SDK.
- [Agent Memory Leaderboard](agent-memory-leaderboard.md) — host the benchmark Add/Search format
  with physically isolated user stores.

## Deploy and operate

- [Architecture](architecture.md) — storage authority, write and retrieval paths, concurrency, and
  model boundaries.
- [Deployment](deployment.md) — embedded, REST, MCP, and edge process shapes.
- [Operations](operations.md) — health, capture backlog, backup, recovery, retention, index
  maintenance, and telemetry.
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

## Evaluate

- [Mem-Gallery evaluation audit](research/2026-10-27-mem-gallery-evaluation-audit.md) — official
  scorer parity, protocol differences, and limits on attributing quality losses.
- [Negation retrieval research](research/2026-10-27-negation-retrieval-research.md) — original
  papers, open implementations, Chinese scope extraction, and evaluation priorities.
- [Controlled semantic recall](research/2026-10-27-semantic-recall.md) — R01 audit, bounded
  opt-in query execution, candidate/grounding loss decomposition and quality limits.
- [Quality diagnosis audit](research/2026-10-27-quality-diagnostics.md) — five observed boundaries,
  source-coverage limits and human evidence review.
- [Predicate stability](research/2026-10-27-predicate-stability.md) — exact aliases, conservative
  claim keys and deterministic correction fixtures.
- [Negation and lexical evidence](research/2026-10-27-chinese-negation.md) — R02 root cause,
  language-neutral IDF, paired retrieval fixtures, and disclosed ranking tradeoffs.
- [Top1 ambiguity](research/2026-10-27-top1-ambiguity.md) — R03 reproduction, policy risk–coverage
  comparison and correct-candidate losses.
- [Accuracy-first retrieval](research/2026-10-26-accuracy-first-retrieval.md) — authoritative
  quantized scoring, candidate retention, quality fixtures, and measured latency tradeoffs.
- [ATM-Bench and Mem-Gallery research strategy](research/2026-10-26-atm-mem-gallery-sota-strategy.md) —
  current competitive results, MindBridge evidence gaps, and experiments for validating improvements.
- [Zvec 0.8 adoption](research/2026-10-26-zvec-08.md) — index migration, concurrency and
  quantization decisions, measured recall, and native API limits.
- [Benchmarking](benchmarking.md) — reproducible behavior evaluation and local-index measurement.
- [Offline semantic support audit](semantic-support-audit.md) — atomic-claim review, evidence
  localization, withdrawal residue and verification costs before changing formation admission.
- [Model quality gate](operations.md#model-quality-gate) — synthetic real-model regression cases,
  explicit snapshots, safe artifacts, and optional scheduled evaluation.
- [Annotated example configuration](examples/eval.example.yaml) — every evaluation slot and the
  order in which a run uses it.

For repository setup and quality gates, see [CONTRIBUTING.md](../CONTRIBUTING.md). When a fact
changes, update its owning page and link to it elsewhere instead of copying it.

## Maintain these documents

Verify Python contracts against `src/mindbridge/__init__.py`, `memory.py`, `types.py`, and
`exceptions.py`; configuration against `configuration.py` and `plugins.py`; transport contracts
against `api/` and `cli.py`; evaluation behavior against `benchmarks/`. Tests are executable
contracts. Update the owning reference when behavior changes.

Keep this entry focused on product guides, API references, and operational instructions.
