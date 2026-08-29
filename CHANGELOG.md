# Changelog

All notable changes to MindBridge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). MindBridge is pre-1.0, so minor releases
may contain breaking changes.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.

### Added

- A direct `Memory()` API with `add`, `add_many`, `search`, `ask`, `get`, `list`, `delete`,
  `reindex`, and `optimize`.
- An `AsyncMemory` facade with the same operations and return values.
- Frozen, slotted public content/result types for text, image, video, audio, and omni memories,
  plus a stable `MindBridgeError` exception hierarchy.
- SQLite as the authoritative local store for records, canonical FP32 embeddings, compatibility
  metadata, and a durable Zvec outbox.
- Content-addressed local media storage with safe `Path`/`Blob` ingestion, MIME validation, and
  reference-counted cleanup.
- Zvec 0.7 dense cosine HNSW, full-text search, and reciprocal-rank hybrid retrieval.
- First-class semantic, episodic, and procedural memory roles across Python, REST, MCP, SQLite,
  Zvec filtering, grounded evidence, and stable return values.
- Event-time temporal retrieval with explicit reference clocks, deterministic English/Chinese
  calendar expressions, bounded fallback retrieval, and query-time reranking.
- Opt-in non-destructive memory decay with bounded SQLite access reinforcement and no background
  worker or new dependency.
- Crash-recoverable index replay and rebuild from SQLite without re-embedding stored content.
- An optional resource-oriented REST API under `/v1` and five typed MCP stdio tools over a
  caller-supplied `Memory`.
- One ordered multimodal contract across Python, REST, and MCP; response assets expose stable
  metadata without leaking local paths over wire protocols.
- Independent embedding, generation, and transcription backends with narrow operation-specific
  protocols, explicit capabilities, durable transcription-space identity, and capability-driven
  ASR plus visual-language fallback.
- A narrow `EmbeddingBackend` seam, pinned Jina v5 Omni adapter, and generic Sentence
  Transformers adapter using standard multimodal dict/message inputs for models such as Qwen3-VL.
- A narrow `SpeechBackend` seam and lazy FunASR composition through `funasr.AutoModel`: pinned
  Fun-ASR-Nano,
  FSMN-VAD, CAM++ diarization, timed transcripts, and SQLite-backed anonymous speaker recognition
  across recordings.
- Named local speaker registration without a second inference engine or provider compatibility
  layer inside MindBridge.
- Local SQLite schema v5 adds memory roles and bounded retrieval access state; existing schema
  versions migrate in place. Older Zvec recipes rebuild automatically from SQLite embeddings.
- Physical benchmark isolation plus local-index and LoCoMo-Refined runners.
- `mindbridge-bench eval` with pinned adapters for twelve long-memory benchmark families, adaptive
  batching, resumable automatic media acquisition and video preparation, causal manifests,
  deterministic sampling and response caching, cluster-aware confidence intervals, and paired
  regression comparisons.
- Enforced POSIX `0700` data directories and `0600` database/lock files, fork-use rejection,
  bounded public input, and REST body limits.
- Typed Jina text inputs so URL- or path-shaped application text cannot trigger the model's remote
  media downloader or bypass MindBridge asset validation.

### Changed

- Isolation is now one physical `data_dir` per application or benchmark unit. There is no hidden
  default scope or logical partition inside a store.
- The primary developer flow explicitly supplies an embedding backend:
  `Memory(embedder=...)` → `add()` → `search()` or `ask()`.
- The base dependency set is `pydantic` and `zvec`; FastAPI/Uvicorn and MCP are optional
  extras. The official OpenAI SDK lives in `openai`; Sentence Transformers and local media decoders
  live in `local`.
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
- The supported product slice is typed aggregate text and media memory. Each memory currently owns
  one embedding; metadata and memory role are payload/retrieval controls, not authorization.

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
- No automatic role extraction, episode consolidation, procedure execution, chunking, multiple
  embeddings per memory, per-asset retrieval, or learned reranking stage.
- No in-place re-embedding or retranscription when a persisted embedding/transcription space or
  dimension changes; create a new directory and re-encode source content instead.
- The OpenAI adapter inlines at most 64 MiB of aggregate raw media per embedding or generation
  call; answer requests emit one binary part per distinct asset and accept at most 4 MiB of
  serialized text evidence. Use a provider-specific upload adapter for larger media.
- No built-in user authentication, rate limiting, quotas, or secure-erasure guarantee.
