# CLAUDE.md

`AGENTS.md` holds the binding engineering rules for this repository: project structure, quality
gates, storage and isolation invariants, coding style, tests, public contracts, and commit
expectations. Follow it exactly.

This file adds what `AGENTS.md` does not carry: the product judgement in
[docs/design-principles.md](docs/design-principles.md), and a pinned record of what is actually
implemented today.

## Rule zero: direction is not implementation

`docs/design-principles.md` states its own boundary — it "defines product direction and the criteria
for future design decisions. It is not a claim that every target is implemented in the current
release." Its *Product direction* column, and every goal it marks as pending, describe unbuilt work.

Do not write code, tests, documentation, or pull-request prose that assumes a directional target
exists. Verify against the source before referring to any surface.

### Verified surface at version 0.2.0

- **Python.** `Memory` and `AsyncMemory` from `mindbridge`. Implemented operations are `add`,
  `add_many`, `search`, `ask`, `get`, `speech`, `register_speaker`, `reinforce`, `list`, `delete`,
  `reindex`, `optimize`, and `close`. The design document's smaller `add`/`search`/`ask`/`get`/
  `list`/`delete` vocabulary is the intended public core, not the full signature list.
- **REST, prefix `/v1`.** `POST /memories`, `POST /memories/batch`, `GET /memories`,
  `POST /memories/search`, `GET /memories/{memory_id}`, `DELETE /memories/{memory_id}`,
  `POST /answers`, plus an unversioned `GET /healthz` (`src/mindbridge/api/app.py`).
- **MCP.** Exactly six tools: `add_memory`, `search_memories`, `ask_memory`, `get_memory`,
  `list_memories`, `delete_memory` (`src/mindbridge/api/mcp.py`).
- **Console scripts.** Two: `mindbridge` (`src/mindbridge/cli.py`) and `mindbridge-bench`
  (`src/mindbridge/benchmarks/cli.py`). They are one documented CLI surface in two entry points, and
  the split is load-bearing — `tests/test_package.py` scans `ast.Constant` strings, so a single
  dispatcher could not name the benchmark package even to import it lazily. `mindbridge` commands
  are the `Memory` operations kebab-cased plus one non-SDK command, `doctor`. Composition is one of
  `--app MODULE:ATTR`, `--embedder NAME` (`mindbridge.recipes`, a closed three-entry table), or
  `--url URL`; there is no default and no environment fallback. See
  [docs/api/cli.md](docs/api/cli.md).
- **Recipes.** `mindbridge.recipes` names `jina-omni`, `funasr`, and `openai[:model]` over the
  bundled backends and returns the constructed object. It is a closed table, not a registry: a
  backend this package does not bundle is reached through `--app`, never by registering a name.
  `SentenceTransformersEmbedder` deliberately has no recipe, because its revision pin is the
  caller's choice.
- **Extension points.** Five protocols in `src/mindbridge/models/base.py`: `EmbeddingBackend`,
  `GenerationBackend`, `TranscriptionBackend`, `SpeechBackend`, and the optional
  `StreamingGenerationBackend`, which `Memory.ask` selects through a structural `isinstance` check
  when an answerer implements `stream_answer`. Constructor injection is the only plugin mechanism;
  there is no runtime registry.
- **Bundled backends.** `JinaOmniEmbedder`, `SentenceTransformersEmbedder`, `OpenAIModels`, and
  `FunASRTranscriber`. FunASR runs through `funasr.AutoModel` only — no vLLM or llama.cpp adapter
  exists, and upstream availability is not a MindBridge support claim.

When a change moves any line above, update the *Current release and direction* table in
`docs/design-principles.md` in the same patch. A stale row there is how a later contributor talks
themselves into deleting a live constraint.

## Design rules, and where the code enforces them

### Routing and models

- **Route by declared capability, never by provider or model name.** Backends publish
  `embedding_capabilities`, `generation_capabilities`, and `transcription_capabilities`; routing
  reads those declarations. See `_embedding_contract`, `_generation_contract`,
  `_transcription_contract`, `_route_embedding`, and `_route_generation` in
  `src/mindbridge/memory.py`. Branching on a model identifier or class name is the anti-pattern.
- **Fallbacks preserve information, and no valid route is a failure.** Transcribing audio for a
  text-only operation must leave supported image and video evidence on its native route
  (`_fallback_unsupported`). When no route remains, fail before inference. Unsupported media is
  never discarded silently.
- **Durable model identity outlives the runtime.** `embedding_model`, `embedding_space`, and
  `embedding_dimension` are recorded and re-checked on open (`_ensure_store_metadata`,
  `_reembed_memories`). A changed recipe forces a rebuild; it must never produce a silently mixed
  embedding space.
- **The application owns provider clients.** Backends are constructed by the caller and passed in.
  No hidden registry, no environment-driven model construction inside `Memory`. Any future
  automatic selection is a deterministic policy over explicit candidates that reports what it chose
  and why, and never migrates an embedding space or narrows modality coverage on its own.

### Durability, isolation, and privacy

- **One physical `data_dir` is one running instance**, enforced by an operating-system lock in
  `src/mindbridge/infrastructure/local/_lock.py`. Metadata is application data — never an access
  control, scoping, or isolation boundary. The product contract carries no implicit account,
  request, or benchmark identifier.
- **SQLite is authoritative; Zvec is a rebuildable projection.** A durable write commits SQLite
  before the index changes, and an outbox row is acknowledged only after the flush succeeds
  (`pending_index_operations` and `acknowledge_index_operations` in
  `src/mindbridge/infrastructure/local/store.py`, drained by `Memory._drain_outbox`). A faster path
  that can acknowledge unrecoverable data is invalid regardless of its benchmark.
- **Local processing and storage are the baseline.** A remote model call is an explicit deployment
  choice. Source media and identity-derived representations, including voiceprints, must never
  leave the machine implicitly.

### One execution plane

`Memory` is the only execution plane. REST (`api/app.py`) and MCP (`api/mcp.py`) normalize transport
input and serialize output; they must not implement their own modality routing, retrieval,
persistence, provider selection, defaults, or error policy. Reject any change that pushes policy
into a transport, and keep IDs, field meanings, pagination, idempotency, defaults, and error
semantics identical across every surface that shares an operation.

### Change discipline

- **Reuse before building infrastructure**, in this order: the provider's official SDK or runtime; a
  portable maintained ecosystem such as Hugging Face or Sentence Transformers; a target platform's
  native acceleration stack; an already-installed dependency or the standard library; and only then
  the smallest local adapter, once a gap is demonstrated. Innovate in memory semantics, routing,
  retrieval, and consistency — not in another HTTP client, downloader, codec, model loader, or
  queue. Reuse still has to clear the supported Python and hardware matrix, licensing, security,
  performance, and maintenance boundaries, and a new dependency belongs in the narrowest relevant
  optional extra.
- **Measure end to end before optimizing.** Performance is the elapsed time the agent experiences
  across transcription, embedding, media preparation, the durable write, index visibility,
  retrieval, and grounded generation. An optimization counts only when it moves a measured
  bottleneck without weakening retrieval quality, durability, or modality coverage. Concurrency,
  batching, quantization, caching, and native runtimes come after a trace names the limiting stage.
  The trace is available, not hypothetical: `src/mindbridge/_telemetry.py` emits OpenTelemetry
  operation and model spans with token usage and time to first token, documented in
  [docs/observability.md](docs/observability.md). Name the span before proposing the optimization.
- **A quality claim needs full provenance**: dataset and revision, official split and evaluator,
  input route, model and runtime revisions, retrieval settings, hardware, and measured latency and
  resource cost — produced through the public path and replayable. See
  [docs/benchmarking.md](docs/benchmarking.md). Benchmark-specific shortcuts that do not improve a
  real agent scenario are out of scope.
- **Extensions stay narrow and optional.** Omitting one leaves the normal API unchanged. A new
  capability starts from a concrete implementation and an end-to-end use case, and declares its
  accepted modalities, typed output, provenance, configuration, resource lifecycle, concurrency
  behavior, privacy boundary, and failure mapping. It never adds provider branches inside `Memory`.
  See [docs/plugin-architecture.md](docs/plugin-architecture.md).
- **Build the smallest thing that satisfies a measured need.** No registry, factory, service, queue,
  cache, or compatibility layer for a hypothetical deployment. Add one thin boundary only when a
  real second implementation proves the existing boundary insufficient.

Before proposing a model, runtime, extension, or performance change, answer the decision checklist
at the end of `docs/design-principles.md`. If a question cannot be answered with a measurement or a
named source, the change is not ready.

## Working in this repository

- Run the gates in `AGENTS.md` before submitting. Documentation changes additionally need the pinned
  markdownlint and lychee commands in [CONTRIBUTING.md](CONTRIBUTING.md); Ruff formats Python code
  blocks inside Markdown, so documentation examples are part of the formatting gate.
- Changing a signature, response type, exception, endpoint, tool schema, error code, on-disk schema,
  or console entry point is a breaking change, and needs tests and documentation in the same patch.
- Product modules must not import benchmark modules.
- Prefer relative links between repository documents so a moved or renamed file fails the link gate
  instead of rotting quietly.
- Report what was actually run and what it produced. A benchmark number without its provenance, or a
  gate reported as passing without being executed, is worse than no claim at all.
