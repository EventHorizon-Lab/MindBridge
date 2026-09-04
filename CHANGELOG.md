# Changelog

All notable changes to MindBridge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). MindBridge is pre-1.0, so minor releases
may contain breaking changes.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.

### Added

- `ask_stream()` on `Memory` and `AsyncMemory`, and the `AnswerChunk` value it yields. `ask()`
  already consumed a provider's token stream, timed the first token into the
  `mindbridge.model.time_to_first_token` span attribute, and then returned only the joined text,
  so a caller waited for the last token to see the first. `ask_stream()` runs the identical path
  — same retrieval, grounding, abstention, and reinforcement — and yields the generated text as
  it arrives. Each `AnswerChunk` carries either `text` or `result`, never both: the deltas join
  to the answer, and the single terminal chunk holds the same `AnswerResult` `ask()` returns.
  `ask()` now drains that generator, so a buffered and a streamed answer cannot drift apart. A
  backend without `stream_answer` yields its whole answer as one delta, so the shape does not
  depend on the provider; `capabilities.streaming_generation` reports whether delivery is
  actually incremental. Arguments are validated at the call rather than at the first pull, and
  the operation the answer holds is released before the terminal chunk, so reading a result and
  stopping there needs no cleanup. REST, MCP, and the CLI have no equivalent: each would have to
  choose a streaming wire format, and all three gaps are documented on their own pages.
- `AffectCue`, the entry type of the compiled `affect` section on `ContextBundle`, exported from
  `mindbridge`: every `SearchHit` field plus `event_ids`, the active events formed from the same
  observations the cue already cites in `context.evidence_ids` -- co-occurrence inside one capture,
  never an attributed cause. `render()` prints `basis`, `confidence`, cue modality, valence, and
  arousal on every affect line, so an agent can tell a model inference from a user statement
  without a second call. The hop is resolved after selection, only for the affect entries the
  budget bought, in one batched store read under the same visibility and scope rules that hydrated
  the hits, and is skipped once `max_latency_ms` has passed (the `stage_skipped` unknown names it
  alongside conflict detection). A cue carries at most eight co-derived events and `render()`
  truncates every ID list to eight with a `+N more` count, so marks that `max_chars` does not
  charge for cannot grow without limit. REST `POST /v1/context`, the MCP `compile_context` tool,
  and `mindbridge compile` publish the same field, and a `ContextBundle` refuses an affect entry
  that is not an `AffectCue`.
- `capture()`, `settle()`, and `pending_captures()` on `Memory` and `AsyncMemory`, plus the
  `capture`, `settle`, and `pending-captures` CLI commands. `capture()` commits a record, its
  media, its observation context, and one durable enrichment queue row in a single SQLite
  transaction and returns before any model call; `settle()` runs the deferred speech identity,
  transcription, embedding, indexing, and formation stages in enqueue order. Every write path
  previously blocked on the complete model chain, so a host with a burst of observations had to
  choose between dropping them and stalling its own loop. `add()` and `add_many()` settle a queued
  record they encounter, so their searchable-on-return contract is unchanged, and `search()`,
  `ask()`, and `compile()` never settle: time to searchable stays under host control.
  `pending_captures()` returns `PendingCapture` values — `memory_id`, `enqueued_at`, `attempts`,
  `last_error`, and `awaiting` — and takes an optional `memory_ids` filter, so a caller can ask
  whether one record is searchable yet and an operator can see why one is not. `awaiting`
  separates a record with no vectors (`"enrichment"`) from one that is already searchable and owes
  only formation (`"formation"`). `settle()` attempts every record it read rather than stopping at
  the first failure, and its `max_attempts` ceiling (default 3) skips a record that has already
  failed that often, so one poisoned capture cannot block the queue; `settle(memory_ids=...)` and
  `mindbridge settle MEMORY_ID...` run named records alone and ignore that ceiling for them, which
  is how a parked capture is retried by hand. One settlement runs at a time per `Memory`, so a
  concurrent `settle()` or an `add()` of the same captured content waits instead of running the
  model stages twice. `capture()` applies the embedder-capability check `add()` applies, so an
  unsettleable record is refused before it becomes durable. With a formation backend configured,
  `add()` holds a queue row from its write transaction until formation returns, so a crash in
  between leaves work the next `settle()` completes without re-embedding.
- `capture=True` on `Memory.add_stream()`, `AsyncMemory.add_stream()`, `AsyncCaptureStream`,
  `AsyncAudioStream`, and `AsyncVisionStream`, plus `mindbridge add-stream --capture`. A streaming
  `FINAL` then commits through `capture()` instead of `add()`, which completes the path from
  continuous observation through speculative working context to low-latency durable
  acknowledgement and deferred enrichment; every `StreamCommit` carries the new
  `pending_settlement` field so the caller knows the record owes a `settle()`. A `StreamInput`
  transcript or description is folded in at capture time, so the deferred commit lands on the same
  content-addressed record the strong path would have written. The default is unchanged: without
  the flag a final still commits through `add()` and is searchable when the commit yields.
- `Memory.compile()` and `AsyncMemory.compile()`, a context compiler that runs the existing
  retrieval kernel once and returns a `ContextBundle`: actors, episodes, facts, procedures, affect
  cues, and traits selected within a `ContextBudget`, plus the lineage conflicts it reports without
  resolving, temporal and spatial bounds, an omitted count, and a deterministic `render()`.
  Grounding selection existed only inside `ask()` and returned a flat hit list, so an agent
  building its own prompt had to re-derive structure and budget from unranked hits. `ask()` is
  unchanged. `mindbridge compile` reaches it locally and over `--url`, `POST /v1/context`
  (`compileContext`) serves it over REST, and `compile_context` is the fifteenth MCP tool.
- `consolidate()`, `forget()`, `rollback()`, and `operations()` on `Memory` and `AsyncMemory`, with
  the `MemoryIntent`, `MemoryTrigger`, `MemoryOperation`, `MemoryOperationRecord`, and
  `ConsolidationReport` vocabulary and the `ConsolidationBackend` protocol injected through
  `MemoryPlugins.consolidator`. A model proposes reinforcement, consolidation, correction, and
  forgetting over a bounded evidence set; the kernel validates every citation, applies each
  accepted operation in its own transaction with an append-only log row, rejects a duplicate
  `operation_key`, and can reverse any of them by `operation_id`. Derived memory could previously
  only be produced one source at a time by formation, and nothing could retire or forget it under
  policy. `MemoryCapabilities.consolidation_model` reports the injected backend. The four
  operations are reachable from the SDK and the `consolidate`, `forget`, `rollback`, and
  `operations` CLI commands only; neither REST nor MCP exposes them.
- `consolidation_candidates()` on `Memory` and `AsyncMemory`, the `ConsolidationCandidate` value,
  and the `consolidation-candidates` CLI command: the durable trigger the loop was missing. It
  answers "what needs deliberation?" from state already committed — a derived record that gained
  independent evidence no standing operation weighed, a lineage whose current visible claims
  disagree, a record confirmed through `reinforce()` since an operation last saw it — with no new
  table, queue, or timer. Every `MemoryTrigger` was previously a label the caller chose, so the
  host was the trigger and nothing recorded why a pass ran. `QUERY_FAILURE`, `PRESSURE`, and `IDLE`
  stay labels; nothing durable records them. Like the rest of the control plane it is SDK and CLI
  only.
- Consolidation forgetting: a `CONSOLIDATE` proposal may name `target_ids` among its own evidence,
  and those sources leave ordinary recall in the same transaction that creates the derived record.
  The evidence links stay, so lineage survives, and the log row carries them as
  `MemoryOperationRecord.forgotten_ids` so one `rollback()` reverses both halves. The three
  forgettings stay distinguishable in the log: `delete()` leaves no row, cognitive forgetting is a
  `FORGET` row, and this is a `CONSOLIDATE` row with `forgotten_ids`. Previously the only way to
  retire a consolidated source was a separate `FORGET` with no lineage relationship to the
  consolidation that motivated it.
- `MemoryRecord.forgotten_at`, cognitive forgetting as a policy state a host can set and clear.
  A forgotten record leaves `search()`, `ask()`, and `compile()` but stays readable through `get()`
  and `list()` with the state visible, so forgetting is auditable and reversible. Physical erasure
  remains `delete()`.
- `build_mcp_server` publishes the composition's capability view as the MCP server instructions, so
  a connecting agent learns the configured modalities, models, and capabilities without a tool
  call. There is no capabilities tool and no capabilities route: `GET /healthz` already reports the
  same view over REST.

- A `consolidation` slot on the declarative configuration surface, plus `recipes.consolidator`
  and a `--consolidator` command-line flag, so a `ConsolidationBackend` is reachable from
  `Memory.from_config()` and from the product CLI. `ConsolidationBackend` was implemented and
  accepted by `MemoryPlugins`, but nothing built one: the memory-management loop existed and no
  declarative deployment or CLI composition could run it. Consolidation stays absent by default:
  it is a paid reasoning call over an evidence set.
- `MemoryOperationRecord.superseded`, the `(memory_id, version)` pairs the kernel's own lineage
  rule retired while applying a `CONSOLIDATE` — the records a new `STATE` or user-stated `TRAIT`
  replaced in its lineage, which the backend never named and may never have been shown. They are
  on the log row, in `mindbridge operations`, and `rollback()` restores exactly them. The
  supersession previously happened outside the evidence window and could not be reversed.

- A `formation` slot on the declarative configuration surface, plus `recipes.former` and a
  `--former` command-line flag, so a `FormationBackend` is reachable from `Memory.from_config()`
  and from the product CLI. `FormationBackend` was implemented, accepted by `MemoryPlugins`, and
  called by `Memory` on every write, but nothing built one: no declarative deployment and no CLI
  composition had ever produced a typed memory, and therefore none had ever revised a belief,
  because the supersession rules fire only on the `STATE` and user-stated `TRAIT` kinds that only
  formation emits. Formation stays absent by default: it adds a model round-trip per write.
- A `vision` slot on the declarative configuration surface, and `describe` on `OpenAIModels`, so a
  `VisionDescriptionBackend` exists and is reachable from `Memory.from_config()`. The protocol was
  declared, accepted by `MemoryPlugins`, and called by `Memory` on every write, but no class
  implemented it and no key selected one, so the derived-text write path returned its input
  unchanged everywhere: on one measured corpus 38.9 % of records were images or video stored with
  a 28-character body, matchable by the dense route alone because a full-text document that is
  empty cannot be matched and scores zero for the lexical re-ranking bonus. The caption is unioned
  into that document, never substituted for the caller's text, and the asset is still embedded
  natively. Video is described from four locally decoded stills rather than by uploading the file.
  A visual sent as several stills is marked with the count, and one caption per *visual* is asked
  for explicitly: over an unmarked four-still clip a measured endpoint returned four separate
  descriptions on every attempt, the one-caption-per-input contract rejected the reply whole, and
  `modalities: [image, video]` therefore paid for every request and stored no caption at all.
  Description stays absent by default: it adds a model call per visual on the write path, reported
  under its own model module so its tokens are separable from answer tokens. A malformed reply is
  retried once -- an endpoint can answer `200 OK` with invalid JSON, which an SDK retry policy
  never sees -- and a describer failure leaves the memory stored without a caption instead of
  failing the write, counted on the vision span as `mindbridge.vision.failed_batches`.
- A store-side caption cache: a `visual_descriptions` table keyed by `(asset content SHA-256,
  vision_space)` is read before any describe call and written with the memory, so a product caller
  who ingests one corpus twice, or re-derives after a crash, pays once and gets the same indexed
  documents. The measured describe endpoint returns a different caption for the same image on every
  request even at temperature 0 with a fixed seed, and the caption is unioned into the memory's
  full-text document, so without this a re-ingest silently rewrote what a memory said and paid for
  every image again. One describe per asset *content*, so two memories over one picture in a single
  write cost one call; the vision model span and its token counters only open when a call is
  actually made. A failed batch is still never cached, so a later ingest retries it. Local schema
  version 12 to 13, with a forward migration that adds the table and rewrites no existing row.
- A description cache in the benchmark harness, keyed by asset SHA-256 and describer model, so two
  ingests of one corpus build identical full-text documents and a repeat run spends no description
  tokens. The measured generation endpoint returns a different caption for the same image on every
  call even at temperature 0 with a fixed seed, which would otherwise make an arm incomparable
  with itself. Opened only when the `vision` slot is configured.
- `explain` on the search tool and the REST query, routing to `search_with_trace` and returning
  the per-candidate trace beside unchanged hits. An empty result over a transport was previously
  indistinguishable between nothing stored, everything below `minimum_relevance`, a `memory_type`
  filter, and an unresolved top-two tie; the trace already named all four and only the SDK and CLI
  could see it.
- `reinforce` on MCP and REST, as `reinforce_memories` and `POST /v1/memories/reinforce`. The
  ranking signals read `access_count`, but no transport could write it, so an agent driving
  MindBridge over MCP or REST held the reinforcement factor at exactly 1.0 for the life of the
  store while age-based decay, when enabled, still applied.
- A `context` parameter on the audio and vision stream adapters, accepting a fixed
  `ObservationContext` or a zero-argument callable sampled at each closed observation. `StreamInput`
  has always carried a context, but neither adapter passed one, so every memory written through the
  microphone and camera paths had a null spatial pose. The callable form exists because a capture
  stream outlives the observations it commits: a moving robot's pose is not a property of the
  stream.

- Face and speaker writes now record `mindbridge.identity.observations` and
  `mindbridge.identity.matched_existing` on their storage span, so a recognizer that cannot tell
  people apart is visible at the write instead of only as a weak answer much later. Both failure
  modes were silent: a detector whose confidence threshold suits posed photographs found no face at
  all in 76 EgoLife frames, and a recognizer whose similarities do not separate the footage created
  4 026 identities from 4 731 observations, 84.1% of them seen exactly once.

- A `formation` slot on the declarative configuration surface, selecting the bundled OpenAI former
  so that entity, event, state, relation, affect, trait, and response-policy memories, their
  validity intervals, spatial pose, and valence/arousal are reachable from `Memory.from_config()`
  and from the benchmark harness. `FormationBackend` was implemented, composed by `MemoryPlugins`,
  and used by `Memory`, but no declarative slot built one, so no `from_config` deployment and no
  benchmark run had ever produced a derived memory. The slot stays absent by default because
  formation adds an LLM round-trip to the write path.
- `embedding.modalities` and `embedding.request_format` on the declarative OpenAI embedding slot,
  so a self-hosted multimodal embedding server can be composed from configuration instead of only
  from constructor injection. Defaults stay text-only and `input`-shaped.
- `evidence_budget_chars`, a retrieval policy that lets `ask()` keep grounding past `limit` while
  the evidence fits one character budget, charging media assets a flat equivalent. `None` keeps the
  previous behaviour of grounding on exactly `limit` memories.
- A direct `Memory()` API with `add`, `add_many`, `add_stream`, `search`, `search_with_trace`,
  `ask`, `get`, `speech`, `faces`, `register_speaker`, `register_identity`, `reinforce`, `list`,
  `delete`, `reindex`, and `optimize`.
- An `AsyncMemory` facade with the same operations and return values.
- `AsyncOmniPrefetch`, a per-turn speculative-recall helper that accepts complete text, image,
  video, audio, or combined snapshots, permits only one real search at a time, coalesces queued
  revisions, and confirms the exact final snapshot without persisting partial input.
- An OpenEQA (EM-EQA) benchmark task pair, `openeqa-hm3d` and `openeqa-scannet`, scored with
  the official LLM-Match protocol. Episode histories are operator-supplied and are prepared by
  encoding each episode's official frame order at one frame per second.
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
- An optional resource-oriented REST API under `/v1` and fifteen typed MCP tools over a
  caller-supplied
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
- `identity(identity_id)`, which resolves an identity ID through any merge alias and returns the
  registered `IdentityProfile`, and `unlink_identity(alias_id)`, which reverses one face-and-voice
  merge and returns the restored ID. Both are on `Memory`, `AsyncMemory`, and the CLI as `identity`
  and `unlink-identity`. Unlinking resets the pair's accumulated evidence; it does not suppress
  the pair, so a voice and face that keep co-occurring are corroborated and merged again.
- An optional `relationship` on `register_speaker` and `register_identity`, stored beside the name
  and readable through `identity()`. Omitting it leaves a recorded relationship intact, so renaming
  a person never discards it.
- `identity_link_min_assets`, the number of distinct assets a voice-and-face pair must co-occur in
  before they merge into one identity.
- `mindbridge.identity.*` span attributes reporting per-asset recognizer yield, including
  observation counts of zero, so a recognizer that runs and detects nothing is distinguishable from
  one that was never configured.

### Changed

- **Breaking:** `VisionDescriptionBackend` now requires a `vision_space` property, mirroring
  `embedding_space`, `transcription_space`, and `formation_space`. A custom describer without it
  stops satisfying `MemoryPlugins`' `isinstance` check, because the protocol is `runtime_checkable`
  and therefore validates on attribute presence. The store caches one caption per asset per space,
  so the space -- not the model name -- is what keeps two describers from sharing captions:
  `OpenAIModels.vision_space` digests the model, its generation controls, **and** the bundled
  caption prompt, so editing the prompt invalidates captions written under the old one. Serving a
  stale caption is the one failure nothing downstream can detect, since it is indistinguishable
  from a fresh one once it is inside a searchable document.
- `search_with_trace` orders its candidate list without going through a `set`, so two runs of one
  query on one library print it in one order whatever the interpreter's string hash seed. The
  ranking never depended on this and does not change: it sorts on `(-final_score, memory_id)`, and
  the record read that hydrates it does not order by its argument. The reachable effect was on a
  stale-index candidate's position in the trace.
- `add_stream` now indexes its committed items in bounded groups (32 items or 250 ms) instead of
  flushing the search index after every observation. Each item still commits to SQLite on its own
  and the committed prefix survives a mid-stream failure; a group the process never reaches leaves
  its outbox rows pending for the next drain, and a `search` on any thread closes the open group
  before it reads. Measured on 300 observations: 25× faster, 10 flushes instead of 300.
- A temporal phrase in a query now only boosts memories that overlap the asked range; memories
  outside it and records without an event time keep the score their relevance earned instead of
  being decayed toward the rank floor. Replayed on 810 paired validation questions the penalty
  recovered no gold memory and only reordered; the boost alone won or tied on every one but one.
- `search` counts the candidates that survive its scope predicates instead of hydrating their
  records to count them, removing one of the two record reads per search (about 15 % of search
  latency at depth 100). Results are identical.
- **Breaking for existing stores:** the bundled OpenAI consolidation recipe is
  `mindbridge-consolidation-v2`. Its system prompt now describes the media parts attached to each
  evidence item, and the recipe is a digest of that prompt. The recipe salts `operation_key` and
  the content address of every derived record, so against a store written before this change the
  duplicate guard no longer fires and an identical proposal mints a new derived record instead of
  being rejected as `"duplicate"`. Rolled-back and re-proposed operations from such a store are
  not recognized either. Re-consolidating a store written with `v1` is the intended migration;
  there is no automatic rewrite, because the `v1` records remain the honest record of what the
  `v1` recipe proposed.

- `Memory.rollback()` and `AsyncMemory.rollback()` return `False` for an operation a later
  standing operation has built on. Operations that touched one lineage reverse newest first;
  reversing an older one out of order would restore a superseded version beside the current one.

- The local schema is version 13. Version 9 directories upgrade in place through four steps:
  version 10 adds `memory_records.place_id` and its index; version 11 adds
  `memory_records.forgotten_at`, the `capture_queue` table that makes deferred enrichment durable
  across a crash, and the append-only `memory_operations` log that makes a control-plane operation
  replayable and reversible; version 12 adds `memory_semantics.identity_id` and rebuilds the
  operation-intent constraint so `identify` is an allowed intent; version 13 adds the
  `visual_descriptions` caption cache.

- **Breaking:** `minimum_relevance` now gates evidence relevance — the cosine the dense route
  reports, or the demoted full-text contribution when only the lexical route matched, times the
  observation's own confidence — and its default moves from `0.55` to `0.10`. It previously gated
  a rescaled `(1 + cosine) / 2`
  confidence, so `0.55` admitted cosine 0.10 and rejected 0.05, and any full-text match was handed
  a flat gate value of `0.6` regardless of its dense similarity or rank. A document at cosine -1.0
  to the query was therefore returned at the default floor and reported as `score 0.825`, and any
  floor above `0.6` silently deleted the entire lexical recall route — which is why the knob was
  never tunable. `0.10` reproduces the previous effective floor exactly, so retrieval behaviour is
  preserved rather than quietly tightened.

  The gate takes the signals the query asked about — retrieval relevance and temporal proximity —
  and leaves out reinforcement and `decay_half_life_days` retention, which the query never
  mentioned. Those still shape `SearchHit.score`, so **an admitted hit can report a score below
  the floor**. Both factors are bounded below by `0.3`, so with retention inside the gate a
  perfectly relevant memory decayed to `0.30`, and to `0.09` once a dated question's window also
  missed it, under the `0.10` default: "prefer recent" silently became "hide old" for the
  deployment that enabled decay and then asked about last year. `search_with_trace` reports the
  gated quantity as `gate_relevance` beside the factors that moved the score off it.

- **Breaking:** `RetrievalCandidateTrace.gate_confidence` is renamed `gate_relevance`. The field
  carries the value the gate compared against `minimum_relevance` — a `final_score` for a ranked
  candidate, a relevance-space estimate for one rejected before ranking — and no longer carries a
  confidence.

- `index_speech` now defaults to `True`. A configured `SpeechBackend` has already produced its
  analysis by the time `add` reaches the index, so reusing that text costs no extra model call and
  no extra token; with the flag off, a video memory's lexical index document could be as short as
  29 characters. The flag remains a no-op unless a `SpeechBackend` is configured, and
  `--no-index-speech` is the CLI opt-out.

- `add` and `add_many` now run a configured `vision_describer` for any visual asset that has no
  description yet, and union its text into the stored and indexed document for every embedder. The
  describer previously ran only from `AsyncVisionStream`, and only when the embedder lacked native
  image support, so an image-only write stored an empty lexical document — including, and
  especially, under the recommended omni composition. A composition with no describer configured
  produces a byte-identical write.

- `speaker_similarity` keeps its `0.78` default, now documented rather than unexplained, with a
  calibration plan. Upstream's own `yesOrno_thr = 0.31` for the pinned CAM++ recipe was evaluated
  and rejected: it is calibrated for a single pair, while `_accepted_identity` accepts on a `max`
  over up to 20 exemplars, where mutually orthogonal random 192-d impostors already reach 0.28. The
  threshold is knowingly wrong in the safe direction, because splitting one speaker costs recall
  while merging two people discloses one person's memories to another.

- `ambiguity_margin` is now compared on the score scale in both branches of the ambiguity check,
  which previously used score differences with a time window and gate-confidence differences
  without one. Differences on the single scale are roughly twice as large, so the same margin
  triggers less often. The default is unchanged and unretuned.

- Grounded answer prompts no longer carry each hit's `memory_id`. The answer system prompt already
  forbids answering with it, and a 64-hex identifier costs about 41 tokens per hit -- more than
  that record's metadata and timestamps combined. Measured with `o200k_base` over four real
  benchmark stores at the default `--recall-limit 20`, the user message drops 808-826 tokens per
  question: 2 943 to 2 117 on locomo-refined, 4 692 to 3 871 on atm-hard, 7 414 to 6 594 on
  m3-bench, 3 497 to 2 689 on mem-gallery. Hit content, occurrence times, memory type, and the
  full application metadata are unchanged: metadata is the source-identity channel the system
  prompt points at, and it costs only 8-12% of the prompt.
- The benchmark harness request timeout now defaults to 300 seconds instead of 3 600. An hour
  bounds nothing a run cares about: a request the server never answers held its task for the full
  hour while the remaining workers idled, and the run reported the stall as elapsed time. The
  slowest mean model call measured across this suite is video grounding at 36.3 seconds.
- A video the embedding model cannot fit in its context is now embedded as four ordered stills
  instead of failing the write, recorded on `mindbridge.embedding.video_sampled_inputs`. Only a
  rejection that declares the length constraint triggers it; any other rejection is unchanged.
  Against a local vLLM serving `tencent/WeMM-Embedding-2B`, every thirty-second EgoLife clip was a
  58 344 token prompt against a 35 768 token model and failed; as stills it is 7 756.
- A declarative `embedding.modalities` must name at least one modality. An empty set built a
  `Memory` that opened cleanly and then failed every write with `does not support: text`.
- Media the embedding model cannot accept inline now degrades the retrieval key instead of the
  whole write. `add` drops the oversized key, stores the memory with its media, keeps it reachable
  through its remaining keys, and records `mindbridge.embedding.elided_parts` on the span. A memory
  left with no key at all still fails. On an ATM-Bench slice this recovers 21 of 7 612 memories
  that previously failed with `payload_too_large`, which is what invalidated the whole task's score.
- Hybrid ranking no longer compares a reciprocal full-text rank with a cosine through `max`. The
  rank contributes a small floor, the IDF-weighted query-term coverage lifts the score toward one
  across the remaining headroom, and a complete term match takes a higher floor. Gold recall at
  eight rose from 0.8239 to 0.8920 on a Mem-Gallery slice and from 0.8194 to 0.9306 on an
  ATM-Bench slice. `SearchHit.score` values change, and a strong candidate is no longer clamped
  to exactly `1.0`.
- A face and a voice now merge into one identity only after the pair co-occurs in
  `identity_link_min_assets` distinct assets, default `2`, instead of on first co-occurrence within
  a single asset. One asset cannot separate "this face spoke" from "this face was listening to
  someone off camera", which is the ordinary case in egocentric capture, where the previous
  behaviour bound the wearer's voice to whoever was visible. The undocumented shortcut that let an
  asset's lone face adopt its lone voice inside the store write has been removed, leaving exactly
  one cross-modal entrance. Set `identity_link_min_assets=1` for the previous behaviour.
  Counting assets bounds that mistake rather than removing it, because a wearer talks to the same
  person across many clips and the wrong pair accumulates as fast as a genuine speaker's, so the
  merge is also contained: only a voice-only and a face-only identity fuse on this path, and an
  identity already holding both modalities absorbs nothing further. On synthetic egocentric
  traffic with one off-camera wearer and three interlocutors, allowing the wider merge collapsed
  all four people into a single identity under every ingestion order tried, while containing it
  held the damage to the first bind and raised correct merges from 0 of 3 to 2 of 3.
- The local schema is version 9. Version 8 directories upgrade in place, adding identity link
  evidence, an identity `relationship`, and the merge record that makes `unlink_identity` possible.
  Merges recorded before the upgrade have no such record and are therefore not reversible.
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

- The benchmark description cache accepts calls from every unit worker thread. It is opened once
  for a run while units ingest on worker threads, and SQLite's per-thread binding made every
  worker-side describe fail; the write path counted each as a failed batch and fell open, so a
  vision arm could build caption-less libraries while every other counter read as healthy.
- The MCP `compile_context` tool's `budget` description told clients the default `max_chars` was
  6,000 when `ContextBudget` has defaulted to 16,000.
- Evidence independence is counted per capture. A derived record cited as evidence inherits the
  evidence groups of its own sources, so several cues formed from one capture (a text and an audio
  `AFFECT` from one turn, say) are one independent source and can no longer corroborate each other
  into a visible model-inferred `TRAIT`. Previously a derived source had no observation row and so
  became its own group, which let one emotional event satisfy the two-source rule. The inherited
  group is re-resolved on every record that cites one whose own evidence changed, so reinforcing,
  rolling back, or deleting a source updates the confidence and visibility of the whole citation
  chain instead of only the record directly touched. Rows written before this change keep their
  stored group until the record they cite next changes; fresh stores are correct.
- Forgetting, retracting, or correcting a naming assertion with a timestamp older than the identity
  row raised `sqlite3.IntegrityError` (`CHECK constraint failed: updated_at >= created_at`) out of
  the store instead of applying. The name projection is rewritten in the same transaction and was
  stamping `identities.updated_at` with the caller's semantic time — a record's `recorded_at`, an
  operation's `applied_at`, a deliberately backdated `forgotten_at` — none of which says when the
  row changed. It now stamps transaction time, like every other write to that row, and the
  projection helpers no longer take a timestamp at all.
- A control-plane operation that lost the idempotency race on the consolidation write path raised
  `StorageError` from a unique-index violation instead of being rejected as `"duplicate"`. The
  formation transaction now makes the same in-transaction key check the other write path already
  made, so one pass's rejection reason is the same whichever path applied it.
- A consolidation stamped its derived record, evidence links, and log row with two clocks taken a
  few microseconds apart. One operation now carries one transaction time.
- Abstention is now detected structurally instead of by exact equality against one English
  sentence, so a refusal is still reported when the model re-punctuates it, wraps it in emphasis,
  appends an explanation, or answers in another language. The grounded prompt requests an opaque
  marker built from `AbstentionReason.INSUFFICIENT_EVIDENCE`, so the prompt and the detector cannot
  drift apart. The marker is an instrument, not a sentence: a refusal still reports
  `UNKNOWN_ANSWER` as `AnswerResult.answer`, so a caller that shows or speaks the answer is
  unaffected, and only the bracketed form counts as a refusal anywhere in an answer — evidence
  that merely quotes the word `insufficient_evidence` is not one. Streaming yields raw deltas, so
  a consumer that streams should render on `abstained`. Note that `benchmarks/prompts.py` supplies
  its own abstention sentinel for memlens, so abstention counts from a memlens run remain
  unusable.

- The formation prompt now states the range of every bounded field — `confidence` 0 to 1,
  `valence` -1 to 1, `arousal` 0 to 1 — and a test asserts it does. One out-of-range value fails
  the whole `add`, and the same omission previously led models to emit a 1-5 scale.

- `--index-speech` derives its default from the SDK instead of hardcoding it, and gains
  `--no-index-speech`. With a literal default, `_reject_embedder_only_options` compared an
  always-`False` argument against the SDK default and rejected every `--app` and `--url`
  invocation with `option_not_applicable` as soon as that default changed.

- `docs/design-principles.md` and `docs/plugin-architecture.md` are restored after being deleted,
  corrected against the code: the extension surface is eight protocols plus one optional rather
  than five, MCP's tools covered the common path but no embodied or identity operation, and a
  dead `architecture.md` anchor is repointed. Caller-asserted validity via
  `ObservationContext(valid_from=..., valid_until=...)` is documented for the first time, including
  that `valid_from` is mandatory.

- The benchmark harness dropped the `former` and `vision_describer` plugins when building each
  isolated store, so a configured formation backend was silently absent from every measured run.
  Both are forwarded now, and a guard test derives the expected keywords from
  `dataclasses.fields(MemoryPlugins)` so the next added slot fails instead of being dropped.
- A cross-modal identity bind no longer cascades. The product link path passed
  `allow_shared_modality=True`, so a fragment could rejoin an identity that already held its
  modality; once a wearer's voice owned one face that identity held both, and every later fragment
  rejoined it. Measured on synthetic egocentric traffic with one off-camera wearer and three
  interlocutors, all four people collapsed into a single identity under every ingestion order
  tried, 0 correct binds and 3 wrong. Refusing the wider merge caps the damage at the unavoidable
  first bind: 2 correct, 1 wrong, 4 identities. `LocalStore` keeps the wider merge for a caller
  that has established the claim another way. The representation, not the rule, remains the
  binding constraint: on real M3-Bench voice exemplars, within-identity cosine 0.7385 against
  nearest-other 0.7301.
- A `scope.valid_at` search no longer discards every memory that carries no declared validity
  interval. The predicate admitted such a record only when neither `valid_at` nor `near` was
  given, so asking what held at any past or present instant returned nothing at all rather than
  nothing relevant, for any corpus written through `add()` without a context. A record with no
  interval is unbounded in both directions and now passes at every instant, matching how the
  semantic path already treats a NULL interval. The spatial `near` filter still excludes records
  with no pose, which is correct: they are not at any location.
- Chinese, Japanese and Korean lexical retrieval. Term extraction matched an entire unsegmented
  run as one token, so a multi-character Chinese query carried a highest-weight term that could
  never match and could never reach full lexical coverage -- the one signal that performs
  cross-route fusion. Runs are now removed before word splitting and re-emitted as characters and
  adjacent bigrams, and 47 Chinese function characters join the noise set that previously held
  only English stopwords. Separately, the index routed Japanese kanji to a Chinese word segmenter,
  which returned nothing for them, and routed Korean to an English stemmer, which cannot match an
  agglutinated eojeol; the full-text field for these scripts is now character bigrams, which
  measured correct for all three languages with no cross-language false positives.
- `ask()` now reinforces the evidence its answer cited, closing the loop the ranking signals were
  written for. Reinforcement failures are suppressed: bookkeeping must not discard an answer that
  has already been paid for. The new `reinforce_on_answer` setting turns it off, and the benchmark
  harness composes every store with it off: reinforcing during a run makes one question's
  retrieval depend on which earlier questions answered, and under concurrency on the order their
  updates committed, so an evaluation would stop being reproducible from its seed.
- Every MCP error now arrives as a bare JSON envelope. Errors raised inside a tool body were
  prefixed by the runtime while errors raised by the middleware were not, so a client parsing the
  text succeeded on argument rejections and failed on `memory_not_found`, `model_error` and
  `storage_error` -- the recoverable ones.

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

- `docs/affective-memory.md` states the affective-memory direction: affect is preserved as a
  sourced, timed, confidence-bearing hypothesis with a perspective rather than recognized as fact,
  with the four affect layers, the behaviour that exists at this release, the phased roadmap
  against the plugin admission rule, the required measurements, and the safety and prohibited-use
  boundary. `docs/README.md`, `docs/context-os.md`, `docs/design-principles.md`, and
  `docs/plugin-architecture.md` point at it where each already named affect cues or a future
  emotion-analysis capability.
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
- `occurred_end` is in the `MemoryRecord` field table in the concepts guide.
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
  named `SentenceTransformersEmbedder` recipe. `--url` mode covers eight of the routed operations plus
  `doctor`; the other nineteen CLI commands exit 10 and name the surfaces that do support them.
  `add-stream` reads finite JSONL lazily but collects return records until EOF to preserve the
  one-document stdout contract; unbounded sources use the Python SDK.
