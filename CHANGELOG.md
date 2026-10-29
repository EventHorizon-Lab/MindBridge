# Changelog

Notable release changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
MindBridge is pre-1.0; backward compatibility is not maintained.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.
The [documentation index](docs/README.md) describes the current contracts.

### Added

- Optional `OfflineGazetteer` and `generation.offline_gazetteer` add bounded, explicitly untrusted
  external map candidates from a pinned read-only local OSM index to answer evidence. The default
  remains disabled; no coordinates leave the machine for lookup and no memory facts are written.
- Public `Memory` and `AsyncMemory` facades for ordered text, image, video, audio, and composite
  observations. Semantic, episodic, and procedural roles share event-time intervals, metadata,
  spatial context, explicit retrieval scopes, and typed results.
- Deferred ingestion through `capture()`, `settle()`, and durable pending captures; streaming
  observations, grounded answer streaming, and asynchronous speculative prefetch.
- Context compilation with bounded evidence closure, identity and consent obligations, exact raw
  excerpts, affect cues, compact Python presentation, and opt-in selected proof delivery.
  Independent-evidence projection remains opt-in and pinned to the store.
- A memory control plane for consolidation, reinforcement, cognitive forgetting, correction,
  rollback, candidate scheduling, and auditable operations. Derived assertions retain explicit
  provenance and temporal versions.
- Face and speaker recognition, named identities, identity linking and erasure, sourced affect
  hypotheses, visual descriptions, automatic typed formation, and bounded optional history.
- Opt-in `query_vision` adds transient query-image description and OCR retrieval routes through
  the configured vision backend. `vision.video_frames` selects 2–64 description stills, default 4;
  non-default counts participate in the visual-description recipe identity.
- OpenAI embedding, generation, and vision adapters accept opt-in `image_max_pixels` bounds for
  request-local image conversion without changing stored assets. Embedding and answering can
  send oversized videos as bounded ordered stills, with explicit sampling limitations.
- Optional `generation.answer_style: concise` requests only the question's answer without
  unrelated dates, explanations or citations. The default reader prompt and both evidence-policy
  refusal rules are preserved; benchmark baseline arms honor the same configured style.
- Explicit backend composition through `MemoryPlugins`, `MemoryConfig`, declarative
  configuration, and named recipes. Local Sentence Transformers/Jina, FunASR, OpenCV face, and
  caller-supplied OpenAI SDK adapters load their optional dependencies lazily.
- REST under `/v1`, fifteen typed MCP tools, and the `mindbridge` product CLI over the shared SDK.
  Capability documents describe the operations and modalities available on each surface.
- OpenTelemetry stage spans, provider request IDs, token usage, streamed time to first token,
  retrieval explanations, and classified retryable errors without ordinary content logging.
- The `mindbridge-bench` CLI with pinned dataset adapters and official scoring protocols,
  physically isolated stores, response caches, resume support, per-task artifacts, diagnostics,
  rejudging, and local storage measurements. See [benchmarking](docs/benchmarking.md).
- A real-model quality gate over fixed answering fixtures, isolated optional-dependency checks,
  and wheel/sdist validation across the supported Python 3.10–3.14 range.

### Changed

- Answer requests identify original record and attachment modalities, including video sources
  shown as ordered stills or retained only as text after media omission.
- Local speech uses a pinned FunASR main revision with corrected speaker attribution and bounded
  CAM++ clustering work. `ncpu`, `vad_device`, and `speaker_device` expose thread and placement
  controls. The new transcription space requires a fresh directory and replay; see
  [local speech configuration](docs/configuration.md#local-speech-with-funasr).
- SQLite is authoritative for records, FP32 embeddings, metadata, and the durable index outbox.
  Zvec 0.8 provides a derived dense/full-text search projection. Writes commit SQLite before
  flushing Zvec and acknowledging outbox work; missing indexes rebuild without model calls.
- One physical `data_dir` has one live owner. POSIX directory and database permissions, fork-use
  rejection, operation leases, bounded runtime resources, and deterministic close enforce the
  embedded runtime's lifecycle. Separate memory domains require separate directories.
- SQLite schema 20 records predicate-resolution provenance. Startup accepts only the current
  schema and matching model identities. Current quantization changes rebuild from canonical
  vectors; changing an embedding or transcription space requires a fresh directory and replay.
- Public facades validate, wire, and forward to explicitly constructed kernel planes. Local
  storage uses one connection pool with table families; model and transport adapters remain
  separate, and product modules do not import benchmark code.
- Retrieval shares authoritative hydration and ranking across search, answering, compilation,
  and control. Relevance and singleton ambiguity gates can abstain; multimodal keys, lexical
  coverage, event-time filters, decay, and reinforcement contribute under explicit settings.
- Answering exposes `strict` and `best_effort` policies, bounded supplemental evidence, optional
  recall planning, and opt-in semantic query expansion. Planned reads retain their resolved time
  bounds and completeness limits. These options do not establish a general quality improvement.
- Semantic query evidence can replace the supplemental ranked tail while retaining the original
  `limit` window and matched-set limits. With query expansion enabled, strict abstentions can
  retry internally using bounded JSON evidence excerpts; streaming emits only the final round.
  Successful query routes remain eligible across rounds within the same evidence budget, so a
  later query can retain an earlier bridge fact.
- OpenAI completion slots support explicit JSON Schema outputs alongside JSON object mode.
  Known reasoning models use supported sampling controls and completion-token parameters.
  Model output is validated before durable writes; stored evidence remains untrusted input.
- Benchmark artifacts require the current schemas: evaluation results and samples use schema 22.
  PersonaMem-v3 uses compiled context; MEMLENS session labels supply grouped retrieval evidence.
  Resolved configuration, effective controls, answer policies, source questions, and cache
  identities make the actual evaluation protocol inspectable.
- Benchmark `--gen-kwargs` accepts finite temperatures from 0 through 2, default 0, with consistent
  sampling flags and a fixed run seed. Product and baseline requests share these controls, and
  effective artifacts and cache identities record them; conflicting YAML controls remain rejected.
- Wheel packages retain runtime code, typing metadata, and third-party notices while excluding
  contributor instructions.

### Fixed

- Sequence answers order selected evidence by event time and insertion order. ATM numeric
  instructions preserve complete dates, times and ranges.
- Judge requests omit unsupported automatic sampling controls. Judge failures retain local
  answer metrics, and saved `JudgeError` predictions can restore those metrics and be rejudged
  without repeating generation.
- Evaluation shares model capacity dynamically between ingestion and answers. Bounded idle
  SQLite pools and child-worker admission preserve progress across independent directories;
  cancellation waits for in-flight work and store cleanup. See [benchmarking](docs/benchmarking.md).
- Candidate metadata decoding releases its SQLite connection before constructing results.
  Evaluation bootstrap statistics avoid flattened sample copies and repeated paired draws while
  preserving seeded results and cluster weighting.
- Directory-owned SQLite pools share a bounded idle budget, preserving each directory's base
  allocation while reusing spare capacity for bursts. Active admission reclaims idle handles,
  and directory ownership outlasts eviction cleanup.
- Index replay, missing-index recovery, stale search-ID hydration, dead-row maintenance, and
  rejected-open metadata handling preserve SQLite authority and recovery work.
- Concurrent retrieval, deletion, asset leasing, streamed ingestion, and pooled reads preserve
  snapshot and resource-lifecycle invariants.
- Formation and control writes recheck evidence activity inside their SQLite transaction.
  Private history aliases and causal version selection prevent cross-target history confusion;
  unsupported proposals cannot grant consent or host policy authority.
- Evidence independence, proof budgets, exact excerpts, compiled character accounting, and
  consent-aware actor delivery respect their explicit bounds.
- Multilingual lexical retrieval, bounded temporal matching, singleton ambiguity handling, media
  fallback, video speech, source event times, and attachment references retain usable evidence.
- Model refusals, provider context limits, output truncation, billing failures, and unavailable
  providers map to explicit errors. Stream cleanup preserves failure classification and usage.
- Benchmark answer policies, budgets, cache identity, scorer protocols, progress, resume,
  telemetry, and resource cleanup reflect the configured run.

### Removed

- Unused generic token-F1 scoring helpers and the withdrawn raw-video experiment proposal.
  Dataset-specific official scorers and the historical measurement records remain available.
- Unused single-item storage wrappers and the unused media-size query; internal callers use
  the existing batch operations.
- PostgreSQL/pgvector, Celery, Redis, S3, service workers, and the service-specific APIs and
  benchmark runners from the superseded design.
- Tenant, user, run, and other implicit isolation identifiers; metadata remains application data.
- Schema migrations, historical Jina embedding spaces and index recipes, automatic re-embedding,
  pending migration state, and the pre-place content-ID fallback.
- Old result readers, permissive cache formats, task-name and CLI underscore aliases,
  `--memory-config`, `pretrained` aliases, `--model mindbridge`, and `num_fewshot`.
  Evaluation uses hyphenated options, `generation_model` overrides, and judge `model` overrides.
- `MemorySettings`, `mindbridge.sdk`, the combined `ModelBackend`, provider credential
  configuration, the custom OpenAI HTTP client, and older-SDK transport probing.
- Video-MME, EgoLifeQA, EgoMemReason, and MemEye tasks, their unused adapters and submission
  artifacts, duplicate scoring runners, the unused evaluation journal, and unused registries.
  Video-MME-v2 remains supported.
- Completed implementation audits and intermediate development notes duplicated by current
  guides and regression tests. Research protocols, measurements, source pins, and scorer notices
  remain in the [research archive](docs/research/README.md).

### Limits

- The embedded runtime does not supply distributed writers, authentication, quotas, rate limiting,
  secure erasure, or a runtime plugin registry. Hosts own deployment and access policy.
- Automatic semantic support verification, learned reranking, long-media segmentation, and
  general natural-language retraction are not established capabilities.
- Provider input limits and optional experimental policies remain explicit; consult
  [configuration](docs/configuration.md), [operations](docs/operations.md), and the
  [API references](docs/README.md#integrate) before integrating.
