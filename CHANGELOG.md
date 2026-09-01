# Changelog

All notable changes to MindBridge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). MindBridge is pre-1.0, so minor releases
may contain breaking changes.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.

### Added

- A direct `Memory()` API with `add`, `add_many`, `add_stream`, `search`, `search_with_trace`,
  `ask`, `get`, `speech`, `faces`, `register_speaker`, `register_identity`, `reinforce`, `list`,
  `delete`, `reindex`, and `optimize`.
- An `AsyncMemory` facade with the same operations and return values.
- `AsyncOmniPrefetch`, a per-turn speculative-recall helper that accepts complete text, image,
  video, audio, or combined snapshots, permits only one real search at a time, coalesces queued
  revisions, and confirms the exact final snapshot without persisting partial input.
- An evidence-backed interaction-memory recipe over the existing semantic, episodic, and
  procedural roles, plus a public-SDK benchmark gate that must justify any future graph projection.
- Frozen, slotted public content/result types for text, image, video, audio, and omni memories,
  including per-item `StreamInput` provenance, plus a stable `MindBridgeError` exception hierarchy.
- SQLite as the authoritative local store for records, canonical FP32 embeddings, compatibility
  metadata, and a durable Zvec outbox.
- Content-addressed local media storage with safe `Path`/`Blob` ingestion, MIME validation, and
  reference-counted cleanup.
- Zvec 0.7 dense cosine HNSW, full-text search, and reciprocal-rank hybrid retrieval.
- First-class semantic, episodic, and procedural memory roles across Python, REST, MCP, SQLite,
  Zvec filtering, grounded evidence, and stable return values.
- Event-time interval retrieval with `occurred_at`/`occurred_end`, overlap filters, explicit
  `occurred_from`/`occurred_until` search bounds, reference clocks, deterministic English/Chinese
  calendar expressions, bounded fallback retrieval, and query-time reranking.
- Opt-in `search_with_trace()` diagnostics with candidate/index IDs, reconstructable dense/lexical
  score components, gate confidence, ranks, and terminal rejection reasons, without query/evidence
  payloads, persistence, or OTel cardinality.
- Opt-in non-destructive memory decay with explicit, bounded SQLite reinforcement and no
  background worker or new dependency.
- Aggregate-plus-atomic embeddings for composite memories, bounded overlapping keys for long text,
  max-over-part retrieval, unconditional candidate over-fetch, and soft temporal reranking.
- Configurable weak-evidence and top-two ambiguity gates so retrieval and grounded answers may
  return no evidence instead of replacing model priors with an unrelated high-confidence asset.
- Per-record event-time and metadata sequences on `add_many`, retaining one embedding batch and one
  SQLite transaction.
- Crash-recoverable index replay and rebuild from SQLite without re-embedding stored content.
- An optional resource-oriented REST API under `/v1` and six typed MCP tools over a caller-supplied
  `Memory`; the documented MCP invocation uses stdio.
- One ordered multimodal contract across Python, REST, and MCP; response assets expose stable
  metadata without leaking local paths over wire protocols.
- Independent embedding, generation, and transcription backends with narrow operation-specific
  protocols, explicit capabilities, durable transcription-space identity, and capability-driven
  ASR plus visual-language fallback.
- `MemoryPlugins`, `MemoryConfig`, and `Memory.from_plugins()` / `AsyncMemory.from_plugins()` as an
  explicit grouped composition path over the same typed backends and local policy as the direct
  constructors, with protocol validation before storage opens and without a registry, provider
  factory, or alternate execution plane.
- A narrow `EmbeddingBackend` seam, pinned Jina v5 Omni adapter, and generic Sentence
  Transformers adapter using standard multimodal dict/message inputs for models such as Qwen3-VL.
- A narrow `SpeechBackend` seam and lazy FunASR composition through `funasr.AutoModel`: pinned
  Fun-ASR-Nano,
  FSMN-VAD, CAM++ diarization, timed transcripts, and SQLite-backed anonymous speaker recognition
  across recordings.
- Named local speaker registration without a second inference engine or provider compatibility
  layer inside MindBridge.
- Opt-in add-time speech indexing so transcripts, stable speaker IDs, and known names participate
  in dense and lexical retrieval.
- Local SQLite schema v6 adds event ends on top of memory roles and bounded retrieval access state;
  existing schema versions migrate in place. Older retrieval recipes re-embed from authoritative
  records before rebuilding Zvec, with the recipe marker committed only after success.
- Physical benchmark isolation plus local-index and LoCoMo-Refined runners.
- `mindbridge-bench eval` with pinned adapters for twelve long-memory benchmark families, adaptive
  batching, resumable automatic media acquisition and video preparation, causal manifests,
  deterministic sampling and response caching, cluster-aware confidence intervals, and paired
  regression comparisons.
- OpenTelemetry end-to-end and stage spans with streaming generation TTFT, exact provider-reported
  multimodal token usage, and per-task benchmark duration/token aggregates.
- Enforced POSIX `0700` data directories and `0600` database/lock files, fork-use rejection,
  bounded public input, and REST body limits.
- Typed Jina text inputs so URL- or path-shaped application text cannot trigger the model's remote
  media downloader or bypass MindBridge asset validation.
- `ModelOutputTruncatedError`, the `ModelError` and `model_output_truncated` code raised when
  generation stops at an output token limit, so a deterministic truncation is distinguishable from
  a transient transport failure on every surface.
- `reason`, `stage`, and `subject` on `MindBridgeError`, plus a `retryable` property that is a
  lookup on `reason` and never a judgement. `reason` narrows the stable `code` to a closed
  sub-vocabulary, `stage` names the failing pipeline stage, and `subject` carries the asset ID,
  memory ID, or batch position. All three are optional, so an unclassified raise site is unchanged
  and is never reported as retryable. No new exception classes and no renamed codes.
- `tests/unit/api/test_surface_parity.py`, which compares the Python, REST, and MCP surfaces
  mechanically — defaults, field names, protocol drift, error codes, envelope shape, and serialized
  record fields — deriving everything from the code except one operation map that must list every
  route and tool.
- A CI job that builds the wheel and sdist, asserts `py.typed` and the benchmark `NOTICE.md` ship in
  both, installs the wheel by path into a clean environment, and runs the loader probes against it.
  A `[project.scripts]` entry is written whether or not the build kept the module it names, so a
  source-tree gate could otherwise stay green over a wheel that raises `ModuleNotFoundError`.
- `.github/scripts/installability_probe.py`, which gives each isolated CI leg a loader to run after
  the import probe, and matrix legs for `benchmarks`, `observability`, and `openai`, which had never
  been installed in isolation anywhere.
- `test_imported_distributions_are_declared`, which resolves every third-party root a product module
  imports — including deferred `import_module`/`find_spec` arguments — against the extras table and
  names the module and line that failed.
- `gen_ai.response.finish_reasons` on the generation span, plus
  `mindbridge.grounding.media_elided_hits` and `mindbridge.grounding.dropped_hits` recording the
  retrieved evidence the OpenAI adapter's inline budget removed.
- An explicit OpenAI-compatible minimum-video setting that converts shorter local videos to four
  ordered stills before the first request while preserving the existing media budgets and fallback.
- A `mindbridge` product console script over the shared `Memory` execution plane. Its commands are
  the SDK operations kebab-cased — `add`, `add-many`, `add-stream`, `search`, `search-with-trace`,
  `ask`, `get`, `speech`, `faces`, `register-speaker`, `register-identity`, `reinforce`, `list`,
  `delete`, `reindex`, `optimize` — plus one command with no SDK counterpart, `doctor`. Output is
  one JSON document per invocation on stdout in the REST field vocabulary, diagnostics and the
  shared error envelope go to stderr, and exit statuses are stable, one per error code, so an agent
  branches on `$?` without parsing anything.
- Three explicit CLI composition paths, of which exactly one is required per invocation and none is
  a default: `--app MODULE:ATTR` for any application-composed `Memory`, `--embedder NAME` for the
  bundled backends, and `--url URL` to address a running owner over `/v1`. There is no environment
  variable that selects a backend, and no plugin registry: a backend MindBridge does not bundle is
  reached through `--app`. The resolved model identity is echoed to stderr on every run, and a
  credential's source is reported while its value never is.
- `mindbridge.recipes`, a closed public table naming `jina-omni`, `funasr`, and `openai[:model]`
  over the bundled backends. Each function returns the constructed object so the caller owns it, and
  every entry pins its model identity to a constant already in the source.
- `mindbridge doctor`, which resolves the composition, exercises each configured backend's loader,
  and reports without writing — turning an under-declared dependency into one line before the first
  write instead of a run of silent ingestion failures. It reports how deep each probe reached
  (`weights`, `client`, or `import`) so the result never overstates the check.
- CLI input forms that carry generated content without shell quoting: ordered positional atoms with
  `@PATH` for a local file and `@@TEXT` for a literal `@`, `-` for standard input, `--content-json`
  for the same discriminated parts array REST and MCP accept, and JSONL on `add-many` or
  `add-stream`. The CLI adds
  exactly one part type to that union, `{"type": "input_file", "path": "..."}`, valid in local mode
  only and refused in `--url` mode.
- A published `mindbridge[all]` extra containing the exact union of every optional dependency.

### Changed

- Composite searches now batch the complete aggregate with bounded focused aggregate and atomic
  keys derived from the first text atom and query media. Later answer-format or instruction atoms
  remain in the complete aggregate but cannot become independent dense queries.
- Dense ranking now uses nonnegative cosine separately from rescaled confidence, and exact lexical
  evidence receives a bounded reranking bonus without overriding strong semantic evidence.
- Jina video preprocessing keeps local paths through Transformers' PyAV decoder so source
  fps/duration metadata drives the pinned Jina recipe's floor-spaced sampling of at most 32 unique
  frames, while retaining Qwen's reference per-frame pixel cap. Its recipe advances so existing
  stores re-embed.
- The benchmark runner performs and records one local query-embedding warmup before timed task
  spans, so a cloned store cannot charge lazy Jina loading to several concurrent questions.
- The OpenAI adapter now limits each base64-encoded media item to 20 MiB as well as keeping the
  64 MiB aggregate ceiling. It removes oversized retrieved assets individually, keeps fitting
  siblings from the same hit, and falls back to text when no media from that hit fits.
- The OpenAI adapter's 64 MiB inline media ceiling now counts base64-encoded request bytes instead
  of bytes on disk. Media is sent base64-encoded, so the old accounting admitted about 85 MiB on
  the wire; the documented number is now the number enforced, at the cost of roughly 48 MiB of
  admitted files on disk.
- Concurrent single-memory adds may share one durable Zvec outbox flush after their authoritative
  SQLite commits. Reindexing replays outbox work committed after its SQLite scan.
- Speech identity analysis for audio/video questions overlaps native query retrieval. Grounded
  answers receive timed turns, stable local speaker IDs, registered names, and match confidence;
  transcript-only inference remains limited to embedding fallback.
- Registering or renaming a speaker now atomically refreshes existing add-time speech text and
  vectors, so recordings made before registration are retrievable by the new name.
- Isolation is now one physical `data_dir` per application or benchmark unit. There is no hidden
  default scope or logical partition inside a store.
- The primary developer flow explicitly supplies an embedding backend:
  `Memory(embedder=...)` → `add()` → `search()` or `ask()`.
- The base dependency set is `opentelemetry-api`, `pydantic`, and `zvec`; FastAPI/Uvicorn and MCP
  are optional extras. The OpenTelemetry SDK lives in `observability`, the official OpenAI SDK in
  `openai`, and Sentence Transformers plus local media decoders in `local`.
- Model authentication, HTTP transport, retries, timeouts, and compatible endpoint handling now
  belong to caller-owned official OpenAI SDK clients. Remote REST authentication and TLS belong to
  the deployment gateway or host application.
- Local embedding spaces are derived from adapter recipe, immutable model revision, effective
  native/Matryoshka dimension, normalization, and query/document semantics.
- SQLite commits before Zvec changes. Zvec is disposable, and only successfully flushed outbox
  operations are acknowledged.
- Remote model work may run concurrently; only the short SQLite commit/outbox and Zvec critical
  sections serialize within one `Memory`.
- The first authoritative non-empty name for a CAS digest is reused when identical bytes later
  arrive under a different filename.
- Server deployments use exactly one process worker per directory.
- Composite memories retain an aggregate vector plus de-duplicated text and media vectors. Search
  collapses vector hits to the parent memory by maximum relevance; metadata and memory role remain
  payload/retrieval controls, not authorization.
- Retrieval no longer reinforces every returned hit. Applications call `reinforce()` only after
  observing positive feedback; that explicit confirmation now supplies a bounded ranking boost
  with or without decay and never leaks past a historical query reference.
- The end-to-end benchmark runner enables speech indexing for media tasks, preserves episodic
  source/time metadata, records exact retrieved intervals, reports official MM-Lifelong Ref@300,
  and uses a new cache namespace so pre-change answers cannot mask retrieval changes.
- **Breaking.** `DELETE /v1/memories/{memory_id}` returns `200` with `{"deleted": bool}` instead of
  `204` with no body. Over REST alone an agent could not distinguish "deleted" from "did not exist",
  which the Python SDK and MCP both report.
- **Breaking.** `GET /v1/memories` defaults `limit` to 100, matching `Memory.list`. It was 50.
- **Breaking.** HTTP statuses now follow whether the same call can ever succeed:
  `backend_not_configured` is `501`, `unsupported_modality` is `422`, `schema_unsupported` is `500`,
  and a retryable failure is `503` with `Retry-After`. `ask` without an answerer previously returned
  `502` — the status agents and proxies retry — for a condition that can never succeed.
- **Breaking.** The error envelope gains `reason`, `retryable`, `stage`, and `subject` on both REST
  and MCP. `subject` is withheld over REST for `storage_error`, `index_unavailable`, and
  `internal_error`, which name server state rather than caller input; the CLI reports it for every
  code, because it runs as the invoking user on the machine that owns the directory.
- `POST /v1/memories/batch` carries per-item `occurred_at`, `occurred_end`, and `metadata`. Those
  three values are part of a memory's content-addressed identity, so the same corpus imported over
  REST previously produced different IDs than the SDK, silently defeating idempotency across
  surfaces.
- Model, storage, and index failure messages are forwarded to transports rather than erased. They
  are author-written literals; provider text only ever reaches `__cause__` in the owner process.
- `httpx`, `torch`, and `cairosvg` are now declared. `httpx` is imported at module scope by the
  benchmark downloader and was resolved only because `huggingface-hub` happens to require it;
  `torch` is imported at module scope by FunASR, which declares neither it nor `torchaudio`, and
  arrived only because Sentence Transformers pulls it in; `cairosvg` is how the pinned Jina revision
  converts `image/svg+xml`, a documented input. `cairosvg` is LGPL-3.0, installed separately and
  never linked.

### Fixed

- Remote product CLI requests now default to a finite 30-second timeout, configurable with the
  positive `--timeout SECONDS` option. Timeouts use the existing retryable `storage_error` envelope
  with `reason="timeout"` and `stage="request"` instead of leaving an agent blocked indefinitely.
- Multi-result `search` and `ask` preserve qualified candidates when the top two scores tie;
  ambiguity abstention now applies only to an unresolved `limit=1` choice.
- Local Zvec maintenance periodically optimizes and copy-on-write compacts durable segments, so
  repeated flushes do not exhaust the process file-descriptor limit.
- ATM-Bench raw image and video memories derive event time through the release's filename parser,
  and benchmark cache namespaces advance with the corrected retrieval semantics.
- Transcript derivation is routed by the configured transcriber's declared
  `transcription_capabilities` rather than by the *embedder's* capabilities, so an omni-capable
  embedder no longer suppresses a transcriber that was explicitly configured.
- Video speech is no longer discarded. `_with_audio_transcripts`, `_cache_audio_transcripts`, and
  `_derived_text` selected assets by `modality == "audio"`, while `FunASRTranscriber` declares
  `{audio, video}`, so a video's speech was dropped by string comparison on every path that reads a
  transcript. As a consequence a bare media memory stored empty content, and because the lexical
  document lives on part 0 only, hybrid retrieval silently degraded to dense-only for every media
  memory. This fixes audio and video; an image has no audio track and gains nothing, and no
  benchmark has been re-run, so no score claim is made.
- The derived-transcript marker in a memory's indexed content is `[transcript:<asset_id>]` rather
  than `[audio transcript:<asset_id>]`. That content is also the BM25 document, and one lexical
  match alone reaches the confidence the default weak-evidence floor requires, so naming a modality
  both labelled video speech "audio" and gave every media memory a free full-text match on an
  ordinary English word. Memory identity is unaffected: it is built from the caller's own text and
  each asset's SHA-256, never from derived text.
- The CLI refuses `--content-json` together with positional content instead of silently discarding
  the positional atoms — a write that dropped caller data on `add`, and a different query than the
  one typed on `search` and `ask`. It fails as `validation_error` (exit `3`) during argument
  validation, before any backend is constructed or any request is sent.
- `mindbridge --url ... add-many` validates every JSONL `content` through the same rule single
  `add` uses, so the CLI-only `{"type": "input_file", "path": ...}` part is refused with
  `unsupported_in_remote_mode` instead of sending a local filesystem path to a remote owner.
- The `openai` recipe closes the SDK client it constructs. `OpenAIModels.close()` deliberately
  leaves a caller-supplied client open; a recipe-built client had no other owner, so repeated
  recipe construction retained one HTTP connection pool per call.
- `SpeakerNotFoundError` is mapped on both transports. It served as HTTP `500` and was destroyed
  outright on MCP, where the middleware overwrote any code outside a hand-maintained allowlist.
  `model_output_truncated` had fallen into the same hole, so that set is now derived from the
  exception classes and cannot silently lose a new one.
- MCP's error envelope carries `trace_id` and `issues`, so an MCP failure can be correlated across
  surfaces and an agent is told which argument was rejected.
- The OpenAI adapter raises `from error` instead of `from None` and classifies the failure from the
  official SDK's own exception classes. Authentication, rate limiting, timeouts, connection loss,
  and rejected requests were previously one indistinguishable `model_error`. An unrecognized failure
  stays unclassified rather than being guessed into a retryable reason. Exhausted billing is
  separated from a transient burst: the SDK raises `openai.RateLimitError` for every `429`, so the
  provider's own `APIError.code` selects the new permanent reason `quota_exhausted` instead of the
  retryable `rate_limited` an agent would retry forever.
- `_open_store` separates a busy data directory from an unsupported on-disk schema, and keeps the
  message a literal so the directory travels in `subject` instead of a message every transport
  forwards.
- `add_many` names the failing item in `subject`.
- The REST adapter's `_Memory` protocol declared three defaults the SDK does not have. Mypy does not
  compare defaults across a structural protocol, so nothing caught it.

### Documentation

- The quickstart and README no longer index into a `search` result. `search` returns an empty tuple
  whenever no candidate clears `minimum_relevance` or the top two dense confidences tie within
  `ambiguity_margin`, so the published first example could raise `IndexError` on a correct install.
  Both now iterate the result, and the quickstart explains when and why it is empty.
- The CC BY-NC 4.0 licence of the pinned Jina weights is disclosed at each point of use rather than
  only in a README footer, with a complete working escape hatch: `SentenceTransformersEmbedder.load`
  pinned to `sentence-transformers/all-MiniLM-L6-v2` at its Apache-2.0 commit, plus a recipe for
  resolving a commit hash. Pinning stays required.
- `SpeechBackend.analyze` is documented as returning `tuple[SpeechAnalysis, ...]`, one per asset,
  not a single `SpeechAnalysis`. It is an extension contract, so the wrong signature produced
  third-party backends that fail inside `Memory`.
- `Memory.list` is documented as defaulting to `limit=100`, not 50.
- The three enforced Python input limits are recorded: 128 content parts, 65,536 characters per text
  value, and `limit` between 1 and 100 for `search`, `search_with_trace`, `ask`, and `list`.
- `occurred_end` is in the `MemoryRecord` field table in the concepts guide, `reinforce` is recorded
  as having no REST route and no MCP tool, and a full-text match is documented as scoring 0.6
  confidence and so clearing the default `minimum_relevance` regardless of vector distance.
- `docs/api/cli.md` is now a reference for a shipped command rather than a contract for a pending
  one, and the remaining documentation no longer describes the product CLI as missing. The old
  "one CLI with two command families" design is corrected to two console scripts forming one
  documented surface, with the reason: the packaging guard scans string constants, so a single
  dispatcher could not name the benchmark package even to import it lazily.

### Removed

- The custom OpenAI HTTP client, single-key REST authenticator, and CLI TLS termination.
- The generic product CLI/server, legacy `mindbridge.sdk` re-export, URL downloader, provider
  credential configuration, combined `ModelBackend`, and custom FunASR vLLM compatibility path.
- Tenant, user, run, and implicit-scope fields from Python, REST, MCP, schemas, and storage.
- PostgreSQL, pgvector, numbered SQL migrations, row-level security, and database integration
  setup.
- Celery, Redis, S3, background consolidation, service workers, and telemetry infrastructure.
- Legacy observation, lifecycle, graph, evidence, edge identity, specialized media-pipeline,
  plugin-registry, and service-specific multimodal APIs.
- Benchmark runners coupled to those removed service and specialized media stacks.

### Upgrade notes

- Existing PostgreSQL data is not converted automatically. Export the source text, metadata,
  event time, and source media, then ingest into a new local directory.
- Old Python signatures, REST routes, MCP tools, CLI commands, and environment variables are not
  compatibility-shimmed.
- Construct provider SDK clients and operation adapters explicitly; choose a separate `data_dir`
  for every independent memory domain. Benchmark-only model variables are documented in
  [configuration](docs/configuration.md).
- Do not point `Memory` at an old database directory. Start with an empty path and keep the former
  deployment available until retrieval has been validated.

### Current limits

- No chat-message arrays, large-file wire upload endpoint, update route, metadata filter,
  distributed writer, or runtime plugin registry.
- No automatic role extraction, episode consolidation, procedure execution, long-media
  segmentation, generated semantic keys, or learned reranking stage.
- No in-place re-embedding or retranscription when a persisted embedding/transcription space or
  dimension changes; create a new directory and re-encode source content instead.
- The OpenAI adapter inlines at most 20 MiB per base64-encoded media item and 64 MiB per embedding
  or generation call, roughly 15 MiB per file and 48 MiB in aggregate on disk. Answer requests
  reserve those budgets for question media, keep top-ranked evidence media that fits, and retain
  overflow hits as text when possible. They accept at most 4 MiB of serialized text evidence. Use
  a provider-specific upload adapter for larger media.
- No built-in user authentication, rate limiting, quotas, or secure-erasure guarantee.
- The CLI has no `--format` flag, configuration file, `MINDBRIDGE_*` composition variable, plugin
  registry, backend registration by name, streaming output, interactive prompt, `serve` command, or
  named `SentenceTransformersEmbedder` recipe. `--url` mode covers the seven routed operations plus
  `doctor`; the other nine CLI commands exit 10 and name the surfaces that do support them.
  `add-stream` reads finite JSONL lazily but collects return records until EOF to preserve the
  one-document stdout contract; unbounded sources use the Python SDK.
