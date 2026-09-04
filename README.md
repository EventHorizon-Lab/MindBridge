# MindBridge

[![CI](https://github.com/EventHorizon-Lab/MindBridge/actions/workflows/ci.yml/badge.svg)](https://github.com/EventHorizon-Lab/MindBridge/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/Python-3.10%E2%80%933.14-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-3DA639.svg)](LICENSE)

**Agentic Native Embodied Omni-Modal Memory System.**

MindBridge is an **Agentic Native Embodied Omni-Modal Memory System** built first for emotional
companion robots. The same memory system extends to desktop robots, personal work assistants,
chatbots, voice agents, and other long-lived intelligent products.

Omni-modal means any supported modality can stand alone or combine with the others: text-only,
image-text, video, audio, or an arbitrary ordered mixture. Applications own their input and capture
forms, then normalize them into one content contract. MindBridge preserves that evidence as durable
memory an agent can search, question, and trace—not merely a nearest-neighbor result.

Authoritative records and source media are stored locally. Inference is explicit: use bundled local
adapters, an OpenAI-compatible endpoint, or your own typed backend. A remote backend receives the
inputs required for its configured operation. The same `Memory` kernel powers the Python SDK,
REST, MCP, and command-line interfaces.

> MindBridge is a memory system, not an agent framework, robot runtime, sensor stack, or hosted
> multi-tenant service. The current release is an embedded library with Python, REST, MCP, and CLI
> interfaces; the product itself is not defined by one integration language.

## Why MindBridge

MindBridge pursues three measurable advantages over alternative memory stacks:

| Goal | Product standard |
| --- | --- |
| **Stronger** | State-of-the-art memory quality across text-only, image-text, video, audio, and omni routes—measured per route, never inferred from one text score. |
| **Faster** | Lower end-to-end ingestion, retrieval, and answer latency on the hardware that runs the product. |
| **Leaner** | Lower model, memory, storage, and operational cost without weakening quality, evidence, or durability. |

These are engineering targets, not unsupported leaderboard claims. Comparative results must name
the dataset, model, runtime, hardware, quality metric, latency, and resource cost so they can be
reproduced.

- **One omni-modal memory contract.** Ordered text, image, video, and audio can remain one
  observation instead of being flattened into an untraceable summary or split across unrelated
  stores.
- **User-owned input forms.** Applications define capture objects, sensor events, and interaction
  formats, then adapt them to `ContentInput`, `StreamInput`, or the async stream packets.
- **Memory semantics above vector search.** Semantic, episodic, and procedural roles combine with
  event time, bitemporal validity, spatial scope, optional decay, and explicit reinforcement.
- **Evidence-grounded answers.** `ask()` returns the exact hits used by the answer backend and can
  abstain when evidence is absent or insufficient. `search_with_trace()` explains ranking and
  rejection without copying private content into the trace.
- **Compiled context, not a hit list.** `compile()` returns a budgeted `ContextBundle` — actors,
  episodes, facts, procedures, affect, and traits, with the lineage conflicts it reports and does
  not resolve — so an agent building its own prompt does not re-derive structure from unranked
  hits.
- **Capture now, enrich later.** `capture()` commits a record and its media in one transaction
  before any model call, and `settle()` runs the deferred stages when the host has time. `add()`
  still returns searchable.
- **An auditable memory control plane.** A `ConsolidationBackend` proposes reinforcement,
  consolidation, correction, and forgetting over a bounded evidence set; the kernel validates every
  citation, logs each applied operation, and can reverse it. Forgetting is a reversible policy
  state, distinct from `delete()`.
- **Recoverable embedded storage.** SQLite and the content-addressed media store are authoritative.
  Zvec is a rebuildable dense and lexical projection updated through a durable outbox.
- **Plugin architecture without a second execution plane.** Typed embedding, generation, speech,
  vision, face, formation, and consolidation plugins remain replaceable while the kernel owns memory semantics,
  validation, evidence, and durability.
- **A lifecycle for continuous perception.** Lazy ingestion, speculative omni-modal prefetch, and
  audio/vision reducers distinguish partial updates, final observations, and cancellation.
- **Developer- and agent-friendly APIs.** A concise SDK, REST, fifteen self-described MCP tools,
  and a machine-readable CLI dispatch to the same kernel and preserve the same contracts.

### Where it fits

| Need | MindBridge's choice | Consequence |
| --- | --- | --- |
| Embedded memory | One `Memory` in the host process | No database, cache, queue, or object-store service |
| Local data control | SQLite and media under `data_dir` | The application owns filesystem, backup, and encryption policy |
| Omni-modal evidence | Native ordered text, image, video, and audio, alone or combined | Retrieval and grounding retain source modalities |
| Portable inference | Explicit capabilities and durable model identities | Local and remote backends share memory semantics |
| Explainable retrieval | Stable evidence IDs, abstention, and traces | Callers can inspect why evidence was used or rejected |
| Framework neutrality | SDK, REST, MCP, and CLI | MindBridge can sit below different agents and applications |

### How it differs from common alternatives

| Alternative | MindBridge adds | Prefer the alternative when |
| --- | --- | --- |
| A vector database alone | Canonical multimodal records, cognitive roles, time and space semantics, grounded answers, and a recoverable index projection | Similarity search is the complete requirement |
| A hosted memory service | In-process execution, local authoritative state, and application-selected model boundaries | You want a vendor to operate storage, tenancy, and access control |
| An agent framework | A framework-neutral memory kernel exposed through Python, REST, MCP, and CLI | You need planning, tool loops, or application orchestration rather than memory |
| Custom SQLite plus a vector index | Content identity, media lifecycle, durable index outbox, stale-ID filtering, typed model contracts, and recovery paths | Your schema and retrieval policy are deliberately small and application-specific |

Choose another design if you require a hosted control plane, built-in authentication, autonomous
planning, or multiple tenants inside one physical store. MindBridge deliberately leaves those
concerns to the host.

## Design philosophy

1. **Preserve evidence.** Store source observations before deriving summaries, identities, or typed
   memories; derived state remains linked to evidence.
2. **Commit truth before projections.** SQLite and original media are authoritative; Zvec is
   disposable and rebuildable.
3. **Route by capability, not provider name.** Backends declare supported modalities and fail
   clearly rather than silently dropping unsupported evidence.
4. **Keep automation explicit and observable.** Credentials, model calls, formation, fallback, and
   ranking policy are deployment choices, not hidden global state.
5. **Keep one execution plane.** Every interface shares IDs, defaults, errors, and `Memory`
   operations.
6. **Measure the public path.** Quality and performance claims require reproducible SDK runs with
   dataset, model, runtime, hardware, and validity fields.

See [Design principles](docs/design-principles.md) and
[Architecture](docs/architecture.md) for the complete rationale and implemented invariants.

## Installation

MindBridge supports Python 3.10 through 3.14. Install only the surfaces you use:

| Extra | Choose it when | Adds |
| --- | --- | --- |
| none | You provide an `EmbeddingBackend` | Core storage, retrieval, public types, and Zvec |
| `local` | You want bundled local embedding or speech | Jina, Sentence Transformers, and FunASR |
| `openai` | You use OpenAI or an OpenAI-compatible endpoint | Official SDK adapter and media request handling |
| `face` | You need local face analysis | OpenCV detection and recognition |
| `server` | You expose REST | FastAPI, Starlette, and Uvicorn |
| `mcp` | You expose agent tools | MCP server transport |
| `observability` | You export traces | OpenTelemetry SDK |
| `benchmarks` | You run evaluations | Download, parsing, scoring, YAML/Parquet, and telemetry dependencies |
| `all` | You intentionally need every optional surface | The exact union of all extras |

Common choices:

```bash
uv add "mindbridge[all]"                       # every optional integration
uv add mindbridge                              # bring your own embedder
uv add "mindbridge[local]"                     # first local add/search loop
uv add "mindbridge[openai]"                    # OpenAI-compatible models
uv add "mindbridge[local,server]"              # local models plus REST
uv add "mindbridge[openai,mcp]"                # remote models plus MCP
uv add "mindbridge[benchmarks,local,openai]"   # evaluation environment
```

With `pip`, the same extras apply:

```bash
python -m pip install "mindbridge[all]"
python -m pip install "mindbridge[local]"
```

`mindbridge[all]` installs the complete optional dependency surface. It does not configure model
providers, download benchmark datasets, or supply the YuNet and SFace model files.

Transport extras do not choose a model backend. The base package always requires an application
supplied embedder.

The first `jina-omni` embedding call downloads the pinned
`jinaai/jina-embeddings-v5-omni-small-retrieval` model and executes its pinned remote code with
`trust_remote_code=True`. Its weights are CC BY-NC 4.0. Review that code and license before
sensitive or commercial use; choose another backend when those terms do not fit.

See [Configuration](docs/configuration.md) for provider fields, custom backends, credentials, and
durable embedding-space rules.

## Basic usage

This complete example stores a fact, proves idempotent identity, and retrieves it with no database
service or model API key:

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/mindbridge-demo",
    "embedding": {"provider": "jina-omni"},
    "settings": {
        "minimum_relevance": 0.0,
        "ambiguity_margin": 0.0,
    },
}

with Memory.from_config(config) as memory:
    stored = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    duplicate = memory.add(
        "The spare key is in the blue toolbox.",
        metadata={"source": "workshop-note"},
    )
    hits = memory.search("Where is the spare key?", limit=1)

    assert duplicate.id == stored.id
    assert hits and hits[0].id == stored.id
    print(hits[0].content)
```

```bash
uv run python example.py
```

The first run may pause while the model downloads. Success prints
`The spare key is in the blue toolbox.`. The zero relevance and ambiguity settings make this
one-record checkpoint deterministic; remove them to restore the defaults (`0.10` and `0.01`)
before tuning a real corpus.

Run the script again to reopen the same directory. Repeating the same canonical content, metadata,
time, type, and context returns the same record instead of creating a duplicate. The context manager
closes model, SQLite, and index resources.

The same `add()` call accepts local `Path`, inline `Blob`, existing `AssetRef`, or an ordered
combination. Add a generation backend to call `ask()`; add speech or face backends for local
identity analysis; use `AsyncMemory` and the stream helpers for continuous capture.

Continue with the [Quick start](docs/quickstart.md) and
[Python SDK reference](docs/api/python-sdk.md).

## Capabilities and documentation

| Capability | Implemented surface | Guide |
| --- | --- | --- |
| Product overview | Use cases, capability boundaries, and extension model | [Product capabilities](docs/product-capabilities.md) |
| Omni-modal memory | Text, image, video, audio, arbitrary mixed records, assets, and content identity | [Core concepts](docs/concepts.md) |
| Retrieval and time | Dense/lexical retrieval, trace, memory roles, bitemporal and spatial scope, decay | [Memory types, time, and decay](docs/memory-types-time-and-decay.md) |
| Grounded generation | Optional answer backend, evidence hits, abstention, and evidence budgets | [Configuration](docs/configuration.md) |
| Formation | Optional entity, event, state, relation, affect, trait, and response-policy derivation | [Automatic formation](docs/configuration.md#automatic-memory-formation) |
| Streaming | Durable streams, omni prefetch, audio/vision reducers, interaction memory | [Streaming and interaction memory](docs/omni-streaming-and-interaction-memory.md) |
| Identity | Speech and face analysis, naming, corroborated linking, unlinking, and erasure | [Python SDK](docs/api/python-sdk.md#cross-modal-identity-binding) |
| Context compilation | Budgeted, partitioned context bundles with conflicts and declared capabilities | [Context compilation](docs/context-compilation.md) |
| Memory control plane | Deferred capture settlement, model-proposed consolidation, reversible forgetting, and the operation log | [Memory types, time, and decay](docs/memory-types-time-and-decay.md) |
| Interfaces | Complete SDK; twelve REST operations; fifteen MCP tools; thirty-five CLI operations plus `doctor` | [Python](docs/api/python-sdk.md) · [REST](docs/api/rest.md) · [MCP](docs/api/mcp.md) · [CLI](docs/api/cli.md) |
| Deployment and recovery | Ownership, backup, restore, reindex, telemetry, and security boundaries | [Deployment](docs/deployment.md) · [Operations](docs/operations.md) |
| Extensibility | Narrow typed model protocols with no global plugin registry | [Plugin architecture](docs/plugin-architecture.md) |

The [documentation index](docs/README.md) is the complete task-oriented map.

## Benchmarks

`mindbridge-bench eval` evaluates memory quality and end-to-end behavior through the public SDK.
Its executable catalog covers seventeen benchmark families across long-horizon conversation,
personalization, multimodal video and lifelog QA, embodied scene memory, and grounded retrieval.
Separate utilities produce raw LoCoMo-Refined predictions and measure the local SQLite-to-Zvec
index path.

```bash
mindbridge-bench eval --list-tasks
mindbridge-bench eval \
  --tasks locomo-refined \
  --model-args generation_model=gpt-5-mini \
  --limit 1 \
  --seed 42 \
  --output-path .benchmarks/results/locomo-one-conversation
```

`--limit 1` selects one LoCoMo conversation and evaluates all its questions; it is not one model
request. Results record dataset and implementation pins, digests, confidence intervals, latency,
resources, tokens, abstentions, mandatory controls, and a noise floor.

This README makes no performance or leaderboard claim: a number is meaningful only with its
dataset, scorer, model, runtime, hardware, and validity fields. Read
[Benchmarking](docs/benchmarking.md) before downloading data or comparing runs.

## Operational boundaries

- One physical `data_dir` has one live owner; metadata is not an isolation boundary.
- REST and network MCP add no authentication, authorization, TLS, quotas, or rate limits.
- Backups require SQLite and original assets; Zvec alone is disposable.
- Embedding model, revision, dimension, and recipe form durable identity.
- MindBridge does not fetch remote URLs; the host must download and validate media.
- Stored content remains untrusted model input; inspect `AnswerResult.hits` for high-impact use.

Read [Security](SECURITY.md), [Deployment](docs/deployment.md),
[Operations](docs/operations.md), and [Troubleshooting](docs/troubleshooting.md).

## Citation

If you use MindBridge in academic or technical work, cite the version or commit you evaluated:

```bibtex
@software{mindbridge_2026,
  author  = {{MindBridge contributors}},
  title   = {MindBridge: An Agentic Native Embodied Omni-Modal Memory System},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/EventHorizon-Lab/MindBridge}
}
```

## Acknowledgements

MindBridge builds on SQLite, Zvec, Pydantic, and OpenTelemetry. Optional surfaces integrate
Sentence Transformers and Jina, FunASR, OpenCV, the official OpenAI SDK, FastAPI/Uvicorn, and MCP.
We thank their maintainers and research communities. Benchmark sources, revisions, protocol notes,
and retained licenses are listed in [Benchmarking](docs/benchmarking.md) and the bundled
[scorer notices](src/mindbridge/benchmarks/_official/NOTICE.md).

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the locked `uv` environment,
Python 3.10–3.14 quality gates, documentation checks, architecture constraints, and pull-request
expectations. Participation follows the [Code of Conduct](CODE_OF_CONDUCT.md); report security
issues privately through [SECURITY.md](SECURITY.md).

## License

MindBridge-authored code and documentation are available under the
[Apache License 2.0](LICENSE). Models, datasets, and vendored benchmark material retain their own
terms; review the licenses of every model and dataset selected for your deployment.
