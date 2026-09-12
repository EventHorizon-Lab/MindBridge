# Changelog

All notable changes to MindBridge are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). MindBridge is pre-1.0, so minor releases
may contain breaking changes.

## Unreleased

This tree targets `0.2.0` and replaces the unreleased service-oriented `0.1.0` design.

### Added

- A recall plan's `match` or `window` step may take `"time": "step:<index>"` and read inside the
  span an earlier step's rows cover. A row's stated dates win over its event time -- ISO days,
  `Month YYYY`, `Month D, YYYY`, `D Month YYYY`, `D-D` ranges, and a year-less day joined to a
  dated one as in `June 23 - July 2, 2022`; never a bare year or a relative phrase -- and a row
  that states none, or whose stated dates spread over more than a year (a booking beside its
  terms revision), contributes its event time instead. This is the read ATM-Bench-Hard's trip
  questions need and a one-shot plan could not express: the dates of a stay are in a booking
  email's text, weeks after the day the email arrived, so a window on the email's own event
  time held none of the trip. A step bound to rows that carry no time reads nothing rather than
  the whole corpus, a stored date the calendar cannot hold is the same non-read rather than an
  exception, a forward or malformed reference is not a plan, several steps bound to one anchor
  derive its span once, and the executed step carries the bounds it resolved so the answer
  prompt states the span it read. `RecallReader` gains `time_span`; `RecallStepResult` gains
  `occurred_from` and `occurred_until`.
- `mindbridge-bench eval` answers `personamem-v3` through `Memory.compile` and the configured
  generator instead of `Memory.ask`. `mindbridge.benchmarks.prompts.task_answer_surface` owns the
  per-task mapping, next to `task_answer_policy`. `ask` answers only from retrieved hits with no
  outside knowledge and abstains on thin evidence; PersonaMem-v3's chatbot, agentic, and proactive
  families ask for an assistant's response and are judged as one. Measured on the 2026-09-11
  baseline, `ask` abstained on 84 % of `proactive_actions`, 46 % of `chatbot_response` and 42 % of
  `agentic` rows, every abstained personalize or agentic row scoring zero. The compiled bundle is
  handed to the generator under a harness-owned assistant prompt, `mindbridge_compile_surface_v1`;
  such rows carry `answer_policy: null`, the arm definition records the `answer_surface` table,
  and the table is part of the response-cache namespace.
- The MEMLENS adapter reads the release's `answer_session_ids` -- present on every row, empty on
  refusal rows -- as `MemLensSession.is_answer_session` and emits them as `evidence_groups`: one
  group of stored turn IDs per answer session, retrieved when any member is ranked. The retrieval
  block scores group labels with that operator and a hypergeometric random-ranker row; a flat
  `evidence_ids` label is the singleton case and keeps its numbers. MEMLENS retrieval quality was
  reported as unmeasurable before this, and the benchmarking guide said the release published
  no label.
- `ask()` can plan how to retrieve before it retrieves, behind `recall_planning` (default
  `False`). Similarity answers "what is most like this"; it has no way to express what a count, a
  list of "all", an adjacency, or "everything about this person" asks for, and measured on the
  round's artifacts those question classes sit at the blind rate while the median gold rank on
  point questions is already 1. With the setting on, the answerer returns a JSON recall plan -- a
  shape and up to six bounded reads over `similar`, `match`, `match` in a time window, `neighbors`
  in corpus order, and `entity` -- which MindBridge validates and executes against SQLite. The
  evidence set is a union with ID dedup and no recomputed score: exhaustive rows in time order up
  to `recall_set_budget_chars` -- never past `evidence_budget_chars` when a caller set one, and
  never more than twice `limit` media rows, since grounding media runs recognition over it -- then
  the ranked window the unplanned path would have grounded, by rank. The plan adds to that window
  and never replaces it: measured on ATM-Hard, grounding a set plan on its exhaustive rows alone
  cost every question whose reads returned few rows the evidence it already had -- 12 grounded
  records down to 1, 2, 4 and 5, and one down to a refusal the unplanned path had answered -- so
  the window is admitted whatever either budget says and the media cap bounds the matched set
  alone. Completeness remains a claim about the exhaustive reads only, and the note now says that
  the question's top-ranked records follow the matched ones and are not part of that set. The
  prompt states what the reads were and whether the set is complete, so a count is licensed by
  completeness instead of guessed, and a new defaulted `exhaustive`
  keyword on `GenerationBackend.answer` and `StreamingGenerationBackend.stream_answer` tells a
  reader that its evidence order is the program's rather than a ranking's. Under
  `answer_policy="best_effort"`, an answer the answerer flagged as thin buys one replan round
  (`recall_rounds`, default 2) that is told how much the first round read, over what dates, and
  why it was not enough; `ask_stream()` still yields one answer, because a round a replan may
  replace is held back and reaches the caller only if it is the round that stands. A backend
  without the new optional `RecallPlanningBackend.plan_recall` capability, a planner error, and
  any plan the kernel will not run all fall back to the single search `ask` has always made, so
  the default path is unchanged. The `mindbridge.recall` stage span carries the plan's shape, its
  op list, the number of exhaustive rows, whether the set was complete, whether the round was a
  replan, and whether the plan was the fallback -- because every failure resolves to that same
  fallback, so nothing else distinguishes a planner that ran from one that was never reached.
  `mindbridge-bench eval` aggregates those into a `recall` block per task under `performance`
  (plan count, plans by shape, fallback count, replan count, incomplete count, non-selective step
  count, exhaustive-row distribution) and stamps each sample's own plan shape into `samples.jsonl`
  as `recall_shape`. Two bounds keep a matched set from being the corpus. A read whose predicate
  selected more than a fifth of the active records -- or more than four times `limit` rows, where
  that fifth is smaller -- contributes nothing, because completeness over most of a corpus carries
  no information about the question: measured on LoCoMo dev, the planner chose `entity` on 229 of
  525 questions, no identity registry existed so the step degraded to matching the name as text,
  and on a corpus whose every turn reads "[date] Caroline said: ..." that name selected about 300
  of 600 records; grounding them cost accuracy 0.721 -> 0.528 on exactly those questions and
  raised abstention from 18 to 57. Such a step is reported on the stage span as
  `mindbridge.recall.non_selective_steps`, and the note tells the reader what the predicate matched
  and that it holds the question's top-ranked records instead of a complete set, so a plan whose
  every step is non-selective grounds precisely what no plan would have. The rows a selective read
  did return are additionally capped by the new `recall_set_max_rows` (default 60), which is the
  character budget's bound on the other axis -- a corpus of short records fits hundreds of matched
  rows inside 30 000 characters -- keeping the earliest rows in the read's own chronological order
  and counting the rest into the same "not shown" shortfall that declares the set incomplete. The
  ranked window is outside both bounds.
- `answer_policy` on `Memory.ask()`, `Memory.ask_stream()`, their `AsyncMemory` twins, REST
  `AnswerRequest`, and the MCP `ask_memory` tool, with the new `AnswerPolicy` alias exported from
  `mindbridge`. Abstaining is a policy the caller owns, not a fixed product behaviour: an
  unanswerable question deserves a refusal, while a multiple-choice caller, or one whose protocol
  gives no credit for "unknown", loses the whole answer to one. The default `"strict"` is
  unchanged in behaviour, keeps its abstention instruction word for word, and its whole prompt is
  byte-identical to the one that predates the policy. An answer-shaping sentence -- answer every
  component asked, prefer a short phrase to a sentence, list only items the hits support -- was
  added to both policies here and then measured harmful and removed: under `strict` it cost
  LoCoMo 0.747 -> 0.545 with abstention rising 9.9 % -> 36.2 %, MemLens 0.300 -> 0.283 with
  abstention 32 % -> 52 %, and ATM-hard-sgm abstention 35 % -> 48 %. Asking for the shortest
  complete answer taught the reader to refuse rather than to answer short, so neither policy
  shapes answers. `"best_effort"` instructs the answerer to commit to the single most likely answer
  the evidence supports -- for a multiple-choice question, always one of the offered options --
  and to flag low confidence with the structured marker on its own line before the answer, which
  MindBridge reads and removes. The result then carries the same `abstained` and
  `abstention_reason` alongside a usable `answer`, so the confidence signal survives. A
  `"best_effort"` question that retrieved nothing at all now reaches the model as a guess instead
  of returning early, and is still reported as abstained with `AbstentionReason.NO_EVIDENCE`.
  `GenerationBackend.answer` and `StreamingGenerationBackend.stream_answer` take the same
  keyword-only argument, defaulted, so a custom backend only needs it once a caller opts in. The
  benchmark harness sets `"best_effort"` for exactly `m3-bench-robot` and the four `mm-lifelong-*`
  splits, whose official evaluations credit no abstention and whose question sets hold no
  unanswerable item; every other task keeps `"strict"`. `--answer-policy` and `benchmark.run.answer_policy` override that table for one run,
  and both the task and the sample rows record the policy the request carried.
- Local storage advances to schema v18. Evidence is stored as clauses: a `CONSOLIDATE` operation's
  cited set is one conjunction, separate operations are alternatives, and withdrawing a source
  retires only the clauses it belonged to, so `(A AND B) OR C` keeps `C` when `A` goes. Derived
  claims enter a compiled context only together with eligible support, lexical-only hybrid
  candidates get their dense score completed from the stored vectors, and `allow_partial_sources`
  can admit digest-bound raw-text excerpts. Rolling back an operation whose output was already
  deleted now succeeds, and deleting one of several alternatives no longer cascades into a claim
  that a still-supported but hidden trait continues to ground. Any other schema version is
  refused on open.
- `mindbridge-bench eval --tasks es-memeval` now evaluates the pinned ES-MemEval EvoEmo QA task:
  18 physically isolated seeker histories, 1,427 questions across the five published capabilities,
  automatic digest-verified GitHub acquisition, session-level evidence recall, the published
  set-overlap F1, and the GPT-4o 0--2 judge retained both raw and as a normalized 0--1 headline.
  Summarization, dialogue generation, and BERTScore remain explicitly outside this adapter.
- `mindbridge-bench eval` support for WorldMemArena checkpoint QA using pinned Hugging Face data
  and upstream protocol revisions. It preserves checkpoint cutoffs and the official
  Correct/Hallucination/Omission, F1, and BLEU-1 metrics. WorldMemArena's separate memory-snapshot
  and evidence-coverage judge metrics remain explicitly unavailable instead of being approximated.
- `mindbridge-bench eval` now persists and can report each task as it finishes, instead of holding
  a whole multi-task run until the last one is done. Every run appends `samples.partial.jsonl` as
  each task stops answering and removes it once the real artifacts land, so an interruption during
  the sixth task no longer discards the first five tasks' answers. The copy holds predictions and
  is written before anything that can fail, and it is guarded like the artifacts it stands in for:
  a rerun into the same output directory refuses without `--overwrite` rather than deleting the
  record of a crashed run. By default, each task is then judged and printed before the next task
  starts; `--no-stream-results` restores the former end-of-run reporting cadence. Client resource
  sampling and optional model-server counter deltas split around each immediate judge pass, so
  judge work is not charged to the product measurement window and the final arithmetic is the same
  under either cadence.
- Completed `eval` output directories now retain `config.yaml` as a resolved comparison manifest.
  It records the effective product, judge, download, server-observation, and run settings after
  file, environment, and command-line precedence, while omitting every API credential and the
  values of unconstrained provider-specific `extra_body` mappings.
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
  actually incremental. `link_identities` means on `ask_stream()` what it means on `ask()`, which
  is this generator drained. Arguments are validated at the call rather than at the first pull, and
  the operation the answer holds is released before the terminal chunk, so reading a result and
  stopping there needs no cleanup. REST exposes the same stream as SSE at
  `POST /v1/answers/stream`; MCP remains a finite buffered tool response and reports completion
  latency instead of claiming TTFT.
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
- A run-scoped description cache in the benchmark harness, keyed by asset SHA-256 and describer
  model, so units sharing an asset build identical full-text documents without silently warming a
  later performance repeat. The measured generation endpoint returns a different caption for the
  same image on every call even at temperature 0 with a fixed seed. Opened only when the `vision`
  slot is configured.
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
- `RetrievalScope.identity_id`, the "about this person" axis, on `search()`, `ask()`, and
  `compile()` and on every transport that carries a scope. A memory is in scope when its semantic
  subject is that identity or the identity was observed in it — a diarised speech segment or a
  face observation on one of its assets — and a merged alias resolves to the surviving identity,
  so either ID of a merged pair answers the same. The edges and their SQLite indexes already
  existed; nothing could ask for them. `place_id` and `identity_id` are also pushed down into the
  search index rather than post-filtered: both are static per document, and a bounded candidate
  window over a library returned an answer that was silently short for a predicate matching two
  records. Index recipe `context-keys-v12` therefore carries the two filter fields, and a
  collection built by an earlier recipe is detected as stale and replayed from SQLite without
  re-embedding stored content. A scope value the index's filter grammar cannot spell — a trailing
  backslash, a control character — drops only its pushdown clause, since SQLite hydration is what
  enforces the scope either way. The CLI's `--scope` decoder accepts both keys, so it now spells
  the whole of `RetrievalScope`.
- `create_app(deliberate_every=)` and `build_mcp_server(deliberate_every=)`, which run
  `deliberate()` on an interval for as long as the surface is serving. `consolidation_candidates()`
  is the only reader of the `QUERY_FAILURE` rows a failed recall writes, and one physical
  `data_dir` has one live owner, so a REST-only or MCP-only deployment collected that signal and
  dropped it. Both surfaces share one loop; it is refused without a consolidation backend or for a
  non-positive interval. Shutdown asks the loop to stop rather than cancelling it, so a round
  already inside its worker thread — which is not cancellable, and which applies its operations
  either way — finishes and reports what it applied.
- `mindbridge --vision NAME`, the visual-description recipe slot, so a captioning backend can be
  composed from the command line the way the embedder, answerer, former, consolidator, and
  transcriber already could. `--url` mode now covers every command that has a `/v1` route, and a
  command with no route says so as a composition error instead of failing as a bad request.
- `mindbridge-bench control-plane`, a behaviour benchmark for the slow loop — `deliberate()`,
  `record_outcome()`, `rollback()` — over a deterministic synthetic long run whose ground truth
  the scenario builder holds, which is what makes consolidation precision, duplicate coverage, and
  contradiction recovery computable at all. `mindbridge-bench eval --deliberate` runs the same loop
  after each cutoff's ingest and before its questions, where a real deployment's loop would have
  run, and the result artifact reports whether it was enabled and how many operations it applied
  (benchmark result schema v14).
- Kernel warnings at the points where a reason string already existed and was discarded:
  re-embedding a store for a new space, an unfinished `settle()`, a refused consolidation or
  formation proposal and why, a vision description that failed so its assets are stored without a
  caption, and a configured model that cannot accept a modality so derived text is used instead.
  Observability rode entirely on OpenTelemetry spans, which are no-ops when the SDK is absent, and
  the kernel held no logger at all: silent degradation is how a capability dies unnoticed.
- A `[facts:<asset>]` document section, split from `[visual description:<asset>]`. A describer's
  reserved `Fact:` lines -- short declarative statements that stay true after the clip ends -- are
  cut out of the visible caption and indexed under their own marker, both reachable through the
  ordinary lexical and dense routes; the visible half never carries a `Fact:` line and the facts
  half never carries the visible prose.
- Automatic `IDENTIFY` from a caption's own stated name. A fact reading `speaker_N is called
  <name>`, where `speaker_N` resolves to a label this same asset's own diarisation produced, binds
  the name through the same `IDENTIFY` assertion `register_identity` writes -- auditable through
  `operations()`, reversible through `rollback()` -- reachable from both `add()` and `settle()`
  with no extra model call, since the name is already in the caption. A standing name a host
  registered is never overwritten by one a model read off a transcript, and a label no recognizer
  in the clip produced binds nobody; both drop into `mindbridge.identity.names_refused` rather
  than only a log line, alongside `mindbridge.identity.names_bound`.
- Vision description retries a throttled or overloaded endpoint before falling open to an
  uncaptioned write. A batch refused as `rate_limited`, on a timeout, a dropped connection, or a
  5xx is described again after three bounded waits (1s, 4s, 16s; configurable only by patching the
  module constant) rather than losing its caption on the first refusal a burst produces. An attempt
  that still has a wait left is counted on `mindbridge.vision.retried_batches`, apart from
  `mindbridge.vision.failed_batches`, which now counts only the attempt that actually lost the
  caption -- summing every attempt into one counter could not tell a provider that throttled an
  ingest from one that ate it. Also retried: a 400 whose provider message says it aborted
  `response_format` JSON generation mid-reply ("Model output became abnormal ... The generation
  was aborted ... Please retry the request"), measured live on an inner-prism gateway where an
  identical retry 5s later succeeds. `request_rejected` otherwise stays a permanent 400 and out of
  the closed `RETRYABLE_REASONS` vocabulary -- this is a narrow message match scoped to describe's
  own retry loop, not a reclassification, and an ordinary rejected request (an unsupported image)
  still fails open on the first attempt.

### Changed

- Identity matching scores one observation against the whole exemplar bank as a matrix instead
  of unpacking and re-normalising every stored exemplar into Python tuples on every call.
  Measured on a 46,360-exemplar store the per-row scan cost 1.3 s per speaker label and grew
  with every clip; the matrix costs about 0.16 s, most of it the read. Results are unchanged:
  the best identity by its highest exemplar, ties broken by identity id, refused below
  `speaker_similarity` or inside `speaker_margin`. `numpy` is now a core dependency; it was
  already installed through `zvec`.
- Speech recognition enrols only the speaker centroids a transcribed turn is labelled with. A
  centroid no turn uses carries no speech, so all it did was mint an identity and an exemplar
  that every later asset was matched against and nothing cited: 27,056 of 44,857 identities on
  one 12,684-clip ingest. The rule lives in the store, where every `SpeechBackend` converges.
- `mindbridge-bench eval` analyses the next ingest chunk's speech while the store embeds and
  writes the current one. The store runs speech, then embedding, then the write for each chunk,
  so the GPU idled during the embedding request and the network idled during speech -- measured
  at about 5 s and 3.7 s of a 10 s batch. The lent speech backend now takes a look-ahead: one
  background thread analyses the next chunk's clips, keyed by content digest (the store's asset
  id), and the store's own call for those clips finds the result. Chunks still commit in order,
  so corpus order and every stored row are unchanged; only the analysis starts early.
- The `mm-lifelong` task group is the evaluation splits: `mm-lifelong-day-test`,
  `mm-lifelong-week-test`, and `mm-lifelong-month-val`. `mm-lifelong-month-train` is the released
  training split, cut from the same 105 h of video as `month_val`, so the group ingested that
  video twice; it stays selectable by name.
- `mindbridge-bench eval` no longer writes a `[source_id: …]` text line into media-only
  memories. The id is already metadata, which is what evidence and retrieval scoring read; as
  content it was a third retrieval key and made the aggregate key differ from the clip's own
  key even when the clip had no speech, so every clip was uploaded to the embedder twice; a clip
  with a transcript still is, since its aggregate key carries the transcript. Text memories keep
  the label.
- `MemoryConfig.evidence_budget_chars` defaults to `24_000` instead of `None`, so `ask()` grounds
  on the `limit` hits and then on as many further ranked records as fit 24,000 characters of
  evidence (media charged at its text equivalent), up to the 100-record rerank pool. A record
  count was the wrong unit for the grounding window: record size is the caller's, so `limit=12`
  was 2.5 k characters on a dialogue corpus and 12 k on a chat-assistant one. Measured paired
  against a same-code control on the 2026-09-11 baseline reader, the default widened
  LoCoMo-Refined's window from 12 to ~100 turns for +0.090 [+0.042, +0.110] on three development
  conversations and +0.076 [+0.054, +0.098] on four held-out ones, cutting refusals from 70 to 24
  of 592; widened LongMemEval-S from 12 to ~26 turns for +0.067 [+0.017, +0.117]; and widened
  MemLens from 12 to ~17 turns for no change. The cost is the answer prompt: 4.8x the tokens and
  2x the latency on the short-turn corpus, unchanged where records were already long.
  `AnswerResult.hits`, `/v1/ask`, and `ask_memory` therefore report more than `limit` hits by
  default; lower the budget to trade evidence for tokens, or set it to `None` to restore the
  exact-`limit` window. The setting's calibration note in `plugins.py` records the measurement,
  including the 8,000-character point (+0.055 / +0.063 on LoCoMo at 2.1x the tokens, half a row
  on LongMemEval).
- The kernel is one module per plane instead of one `memory.py`. `Memory` and `AsyncMemory` stay
  in `memory.py` as facades that validate, wire, and forward; the write, retrieval, answering,
  compilation, identity, control, records, perception, and projection planes live in
  `mindbridge.kernel`, each a class whose constructor names every store, index, backend, setting,
  and sibling plane it uses, so the wiring in `Memory.__init__` is the dependency graph. The
  async observation streams moved to `mindbridge.streams` and are still exported from
  `mindbridge`; `declared_capabilities`, which only the CLI and tests reached through
  `mindbridge.memory`, now lives in `mindbridge.kernel.contracts`.
  `LocalStore` is likewise one connection pool with one attribute per table family --
  `records`, `captures`, `semantics`, `control`, `identities`, `media`, `index`, `recall` -- in
  `mindbridge.infrastructure.local.store`, a package whose top-level imports are unchanged.
  No on-disk schema, public signature, response type, endpoint, tool, or error changed; the
  [architecture guide](docs/architecture.md#code-layout) owns the layout. Kernel warnings now
  log under `mindbridge.kernel.<plane>` rather than `mindbridge.memory`; configure the
  `mindbridge` logger to keep receiving all of them.
- **Breaking:** `GenerationBackend.answer` and `StreamingGenerationBackend.stream_answer` declare
  a keyword-only `answer_policy` argument. Both protocols are `runtime_checkable`, and
  `isinstance` checks the method name rather than its signature, so a custom backend written
  against the two-argument signature keeps answering: MindBridge sends the keyword only when a
  caller asks for something other than the default `"strict"`, and passing `"best_effort"` to a
  backend that does not accept it raises `ModelError` with `reason="model_failed"`. Accept the
  argument to support the policy.
- The benchmark runner records the `answer_policy` each task's product arm requested, next to the
  `arm` and `task` fields of every `results.jsonl` task row and every `samples.jsonl` sample row,
  so a run that asked for a committed answer is distinguishable from every earlier run of the same
  task. The baseline arms do not call `ask`, so their rows carry `null`. Purely additive: the
  evaluation schema version is unchanged, and older result documents still load.

- `AsyncMemory(memory)` now wraps an already-open `Memory` instead of repeating its constructor;
  open one with `AsyncMemory.from_plugins()`, `AsyncMemory.from_config()`, or
  `AsyncMemory(Memory(...))`.
- `mindbridge-bench eval` no longer reports a run-level `resources.energy` block or per-GPU
  `estimated_energy_watt_hours`. Intel RAPL package counters are not read any more; sampled GPU
  power still appears as averages and peaks.
- MCP tool results are validated from the SDK value objects, the way REST already builds its
  responses, instead of through hand-written per-field converters. The published tool schemas and
  their field names are unchanged, and a field added to a value object now reaches both transports
  without a converter to edit.
- The CLI builds its documents from the value objects' declared fields with one timestamp encoder,
  rather than restating each field a third time. Output is unchanged except that `faces` prints
  `observed_at_ms` in the position the value object declares it.
- `Memory.from_plugins` and `AsyncMemory.from_plugins` unpack `MemoryPlugins` and `MemoryConfig`
  instead of naming all twenty-six constructor arguments twice each. The declarative recipe
  factory forwards adapter controls the same way, so a control `OpenAIModels` gains is reachable
  from configuration without a second signature to update.
- Schema upgrades run from one table keyed by version rather than seventeen unrolled branches. A
  step that fails to advance `user_version` now fails as that unsupported version instead of
  falling through to the same error at the end.
- Audited `mindbridge-bench eval` against the pinned Video-MME-v2, BEAM, PersonaMem-v3, and
  OpenEQA evaluators. Video-MME-v2 now reports official 0--100 accuracy and grouped rating; BEAM
  uses its event-equivalence and ordering composite; OpenEQA maps the released 1--5 judge mark to
  the official 0--100 score; PersonaMem-v3 uses its released ranking gains and metric names and
  withholds the micro headline while unsupported task protocols leave coverage incomplete.
- `LocalStore` now pools its SQLite connections instead of opening and closing one per call.
  Nothing held a connection between calls, so every helper opened its own: seven for one `add`,
  five for one `search`, one for a `get` that then spent nine tenths of its latency on it, at
  450 us for the `sqlite3_open` plus the four `PRAGMA` statements that make a connection usable.
  A connection now leaves the pool for exactly one block and is returned by the same `finally`
  that used to close it, so two nested blocks still draw two connections and no two callers ever
  hold the same one; a block that raised closes its connection rather than returning a possibly
  open transaction to the next borrower, and `secure_delete` connections stay unpooled because
  that pragma persists. `get` is 16 times faster, `search` and `ask` about 1.2 times, and a bulk
  ingest 1.3 times, with no change to any result.
- The search-index outbox drains 1024 rows per Zvec flush rather than 256, which is the largest
  write batch Zvec accepts. A flush costs a fixed ~45 ms whatever it carries and leaves one
  durable segment behind, and both of those costs are per flush, so a bulk `add_many` of 8 000
  memories now makes 8 flushes and 8 segments where it made 32 and 32. Ingest is 1.3 times
  faster and searches against the resulting store 1.1 times, for about 32 MiB of transient
  hydration at 1024 dimensions.
- `ZvecIndex.upsert` and `delete` now split a batch that exceeds what Zvec's native writer
  accepts, which is 1024 documents and is not exposed as a value to check against. Refusing one
  was not a failure a caller could retry past: one outbox drain becomes one `upsert`, its rows are
  committed to SQLite before the drain runs, and nothing acknowledges a refused batch, so every
  later `add`, `search` or `ask` that drained re-read the same rows and re-raised. `reindex` had
  the same exposure through `rebuild`'s caller-supplied `batch_size`. Both batch sizes above the
  writer are now free to be chosen for flush cost alone.
- `search` asks for its survivor count only where the answer can still change whether it ranks
  deeper. `count_memories` applies every scope predicate across the whole candidate window and
  its one caller uses the number for nothing else, so a store holding fewer memories than one
  retrieval route returns -- which has already exhausted its routes -- no longer pays for that
  pass on every search it serves. Rankings and results are unchanged.
- **Breaking:** benchmark result schema v11 separates public `search_e2e` from
  `ask_retrieval_core`, measures caller answer latency and TTFT before concurrency admission,
  computes concurrent durations and throughput from interval unions, excludes cached samples from
  product denominators, unions product/judge samples for combined averages, and isolates retrieval
  diagnostics into their own caller latency, SDK-node, and token blocks. Incomplete TTFT
  distributions cannot pass a performance budget; ASR ratios use matching successful-call sets;
  zero-request cache paths are absent from model inference nodes. Model
  usage now reports per-component completeness instead of treating unknown input, output, cached,
  or reasoning tokens as zero; retry attempts and mixed response provenance remain attributable.
  Result artifacts also include fresh-store/warmup/repeat protocol, client hardware and sampled
  power, optional process-global vLLM `/metrics` deltas, and comparable performance budgets.
- **Breaking:** MindBridge-generated benchmark results, media manifests, and LoCoMo sibling
  manifests are now JSONL artifacts with `.jsonl` filenames. The externally specified
  EgoMemReason submission remains a JSON array named `egomemreason_submission.json`.
- **Breaking:** `VisionDescriptionBackend` now requires a `vision_space` property, mirroring
  `embedding_space`, `transcription_space`, and `formation_space`. A custom describer without it
  stops satisfying `MemoryPlugins`' `isinstance` check, because the protocol is `runtime_checkable`
  and therefore validates on attribute presence. The store caches one caption per asset per space,
  so the space -- not the model name -- is what keeps two describers from sharing captions:
  `OpenAIModels.vision_space` digests the model, its generation controls, **and** the bundled
  caption prompt, so editing the prompt invalidates captions written under the old one. Serving a
  stale caption is the one failure nothing downstream can detect, since it is indistinguishable
  from a fresh one once it is inside a searchable document.
- `OpenAIModels.vision_space`'s digested recipe bumps to `mindbridge-vision-v2`, folding in the
  labelled-line prompt and the `Fact:`/`IDENTIFY` contract described above. A caption cached under
  the old recipe is invisible to the new one, not wrong: the next write that cites the asset pays
  one fresh describe call and the store then holds both, keyed by their own space. A store already
  on v1 keeps serving v1 captions to any composition still pointed at v1; there is no in-place
  migration, so a corpus that wants every asset re-described under the new prompt needs a fresh
  `data_dir`.
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
- `ask()` appends the resolved reference time, to the second, as the last line of the generation
  input of every question the answerer reads as text — after routing, so a spoken question carries
  it once transcription has given it text — and the bundled OpenAI answer prompt now tells the
  reader to resolve relative time expressions against that reference and each hit's `occurred_at`
  and to state the resolved date or duration explicitly. Previously the reference was appended only when a caller passed
  `reference_at` or the bounded parser recognized a phrase, so "how long ago did grandpa visit" —
  which narrows no retrieval window and names no date — reached the answerer with the event times
  of its evidence but no time of asking, making the subtraction it asks for unanswerable in
  principle. Nothing told the reader to do that arithmetic either, so a relative phrase in a
  memory could be returned verbatim. No protocol changed: `GenerationBackend.answer` still takes
  `(question, hits)` and the reference travels in `question.text`, as it already did. A question
  routed with no text at all is handed over unchanged.
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

- The local schema is version 16. Version 9 directories upgrade in place through seven steps:
  version 10 adds `memory_records.place_id` and its index; version 11 adds
  `memory_records.forgotten_at`, the `capture_queue` table that makes deferred enrichment durable
  across a crash, and the append-only `memory_operations` log that makes a control-plane operation
  replayable and reversible; version 12 adds `memory_semantics.identity_id` and rebuilds the
  operation-intent constraint so `identify` is an allowed intent; version 13 adds the
  `visual_descriptions` caption cache; version 14 adds the scheduler's durable state
  (`memory_deliberations`, `memory_deliberation_memories`, and `query_failures`), the post-hoc
  `memory_operations.outcome` and `outcome_note` columns, and admits the `merge` intent; version
  15 admits the `consent` intent so a data subject's own statement can be logged; version 16
  backfills a visible naming assertion for identities registered before names became versioned
  claims. A migration must not call a model, so each backfilled assertion is enqueued in the
  capture queue: the projection is correct as soon as the store opens, `pending_captures()`
  names what is still owed, and the next `settle()` embeds and indexes the sentence.

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
- A `RESPONSE_POLICY` proposed by a formation or consolidation backend is refused, whatever basis
  the proposal claims: how the system should behave toward somebody is a grant, not an inference,
  and one observed cue must not become standing guidance. Only an operation the host applies
  itself through `apply()` writes one, and the stored record carries basis `response_feedback`.
  The authorization is stamped on that record rather than on the proposal, which the operation log
  keeps exactly as it was handed over, so a logged row still replays to the same derived record.
- Each network surface narrows the `operations` list in its capability document to what it can
  actually serve, so an agent no longer reads a capability it has no way to invoke. `consolidate`
  is never listed, `speech` and `faces` only with `embodied_operations=True`, and `formation` and
  `describe_vision` — which run on the write path alone — only with `write_operations=True`.
  `transcribe` stays listed either way, because a read tool transcribes the audio a question
  arrives as. `mindbridge doctor` still prints the whole derivation, because the CLI does have the
  commands.
- The consolidation prompt spells the literal operation keys it will be parsed against, and the
  optional `valid_from`, `valid_until`, and `spatial` fields a proposal may carry. It described
  them in prose only, and a measured endpoint answered a correction with `target` for `targets`,
  which the field table rejects as an unknown key — discarding the whole batch, twice, with the
  pass then reporting nothing applied. A rejected reply is now failed loudly, naming which key was
  wrong and quoting the text, rather than leaving an operator with `applied=0` and no account of
  it. The consolidation recipe is `v4`, and the recipe salts each operation key and derived
  record's content address.
- REST operation rows carry `proposal`, the canonical logged payload the CLI already printed. A
  consolidation row read over `--url` reported an intent and no statement, and `apply`, which
  takes a row exactly as it is printed, could not replay it.

### Fixed

- Benchmark speech look-ahead stops accepting work, cancels queued analysis, and waits for
  running analysis before the pool closes its model backends, including when ingest fails.
- A recall step whose anchor has no usable date is reported as incomplete rather than as an
  empty complete set. Time windows derived from several point events include the last point.
- Paired replay hashes the frozen source package recursively, so splitting the kernel and
  storage into modules neither breaks worker startup nor omits moved behavior from its identity.
- `mindbridge-bench eval` gives MM-Lifelong memories an event time. Each prepared clip carried
  its `start_seconds`/`end_seconds` as metadata only, so every memory reached the store with no
  `occurred_at` and the answerer was told to resolve "before"/"after" against `created_at` -- the
  ingest wall clock, identical across a batch and unrelated to the video. The adapter now anchors
  the offsets on one fixed epoch, so the store has a chronology and the questions are anchored at
  the corpus end rather than at the run's wall clock. The `ref_at_300` offsets are unchanged.
- A question that states its own clock as `Today is July, 1 2025` -- a comma between the month
  and the day -- or as `Today is 1 July 2025` now anchors the reference time. Only `July 1, 2025`
  and the ISO form did, so the comma spelling was read against the real wall clock and its "past
  two years" became the wrong two years: on ATM-Bench-Hard that question retrieved none of its
  eight evidence records and abstained. Rolling spans now resolve in every calendar unit, not
  just days: "past two years", "last 3 months", "recent two weeks", "过去两年", "最近三个月" are
  windows ending at the clock, with a month or year shift clamped to the target month's length.
  The count is required except for the bare `past <unit>`, so "in recent years" and "my recent
  day trips" stay unbounded, and every rolling phrase in a question is tried in order so a vague
  "past few days" does not hide a precise "last 3 days" after it. A bare "last year" or "last
  month" remains the calendar phrase it was, and "Today is July 2025" or "Today is 2 decades"
  anchors nothing: a month name ends at a word boundary and a day is not the first digits of a
  year.
- The answer request numbers each attached media asset `attachment 1`, `attachment 2` -- by
  order of first appearance, one number per distinct asset -- under the key `media`, instead of
  listing asset IDs under `assets`, and the grounded prompt's identifier sentence now names
  image, video, and memory IDs explicitly. Those asset IDs are content hashes, and a reader
  asked for "the image ids" returned them as the answer: 2 of 12 ATM-Bench-Hard list questions
  whose retrieval was otherwise complete scored zero that way, and labelled `image 1` the
  reader returned the labels instead. A numbered attachment still says which media belongs to
  which memory and which memories share one, and matches neither the question's words nor the
  shape of an identifier. The same label is written as a text part immediately before the
  asset's own media parts, so the number is defined where the media is: a short video that
  arrives as several stills no longer shifts every later attachment by the extra frames.
- `mindbridge-bench eval` reports the settings it ran with. The effective-config `config.yaml`,
  the `memory_config` block of `results.jsonl`, and the resume checkpoints echoed the parsed
  file's `reinforce_on_answer: true` while every product arm ran with it pinned to `false`; they
  now carry the pinned value, so a run interrupted under the previous code cannot be `--resume`d
  by the new one and re-ingests instead. `arms.definitions.*.retrieval_candidate_limit` and its
  `_basis` also read the product default budget when the run has no config file, instead of
  reporting `min(100, recall_limit * 3)` for a window `ask` actually ranked 100 deep.
- `recall_planning` now actually plans under `mindbridge-bench eval`. The harness lends one
  answerer to every isolated store through a forwarding proxy, and `Memory` probes the optional
  `RecallPlanningBackend` capability with `isinstance` against a `runtime_checkable` protocol --
  which reads attributes with `inspect.getattr_static`, so the proxy's `plan_recall`, reachable
  only through `__getattr__`, was invisible and every question silently answered from the
  fallback point plan: measured, exactly one generation call for each of 31 questions and the
  same twelve hits the unplanned baseline grounded on. The proxy now declares the optional
  capabilities its pooled backend really has, and only those, so a pool that cannot plan is still
  reported as unable to plan. The same declaration fixes the mirror image: `stream_answer` was
  declared unconditionally, so a pooled backend without it failed the call instead of taking the
  buffered path. `Memory` now logs one warning at construction when `recall_planning` is on and
  the answerer has no `plan_recall`, so that wiring failure is no longer silent; every planning
  failure still resolves to the same fallback plan.
- The live `mindbridge-bench eval` progress bar no longer appears frozen while a unit rebuilds,
  ingests, deliberates, or waits for its first answer. It preserves the truthful completed-sample
  count while refreshing elapsed time once a second and summarizing every active unit by phase.
- `mindbridge-bench eval --resume` is no longer refused by the crash copy of the run it continues.
  The guard that keeps a rerun from deleting a leftover `samples.partial.jsonl` covered `--resume`
  too, so the one command written to recover an interrupted run exited on the file that run had
  left behind. `--resume` now passes that file and only that file; `results.jsonl` and
  `samples.jsonl` still need `--overwrite`, because a run that wrote them finished.
- A `--resume` run no longer reuses a store built with the other setting of `--deliberate`. The
  checkpoint recipe named the embedder, the ingest mode and the corpus but not consolidation,
  which applies operations to the store between chunks: a resumed run could inherit memories that
  had been consolidated when it asked for none, or the reverse.
- `generation.min_video_seconds` in a configuration file no longer raises `TypeError` on the first
  `Memory.from_config`. The setting reached the recipe factory, which names every adapter control
  as an explicit keyword and had never been given this one, so the documented field was
  unreachable through declarative composition while it worked when `OpenAIModels` was constructed
  by hand. Every configuration test had replaced that factory with a stub, so a new one now drives
  every generation control through the real factory and asserts the built adapter answers.
- Two `add_stream` calls running at once no longer leave one of their threads permanently deferring
  its index flushes. The deferral that batches a stream's index commits was one shared slot each
  stream saved and restored, so the stream that finished second handed back the id of the thread
  that had finished first; nothing cleared it again, and every later `add`, `delete`,
  `forget_identity`, `reindex`, or `optimize` on that thread returned without flushing. The records
  stayed durable in SQLite but were not searchable until some other caller forced a drain. Deferral
  is now thread-local and scoped to the item write, so it cannot outlive the stream that opened it.
  Scoping it to the write also covers a stream pumped across workers: `add_stream` is a generator,
  so its body runs on whichever thread calls `next`, and a group opened on one thread and left on
  another used to strand the opener the same way. A consumer's own writes between two yields now
  flush as they would outside the stream.
- The `compile_context` MCP tool schema advertised the wrong default evidence ceiling. Its prose
  said 6,000 characters while the field default it publishes beside it, and `ContextBudget`, have
  been 16,000 since the per-modality cost function landed — enough of a gap for an agent that
  budgets against the sentence to leave every video record out on purpose. The sentence now reads
  the numbers off `ContextBudget`, so it cannot drift from them again.
- Withheld or withdrawn consent now omits a person from `actors` however the bundle would have
  named them. The restrained set was derived from the bound `ENTITY` hits retrieval returned, so a
  compilation that reached only somebody's photo or clip -- their naming assertion outside the
  budget's own type bound -- found nothing bound, consulted no consent at all, and then resolved
  that memory's face or speech edge into exactly the `NamedActor` the withheld state promises to
  omit. Consent is read from every identity edge actor enrichment uses, and the `consent_withheld`
  unknown still reports the omission.
- `apply --operation` takes a row as `operations` prints it, and a `CONSOLIDATE` row could not be
  replayed: the printed document omitted `proposal`, which the kernel requires of every
  consolidation, so the advertised pipe failed validation before replay for the intent the slow
  loop produces most. Operation documents now carry the proposal, serialized by the same function
  the log reads back, so the row round-trips.
- `consolidation-candidates --idle` had no effect. The flag was parsed and then dropped by the
  handler, so it asked for exactly what the default asks for and never admitted the never-weighed
  lineages it advertises.
- `ContextBundle.chars` counted the `## Actors` heading that a synthesized actor makes `render()`
  write. Only the sections selection filled were charged a heading, so an episode or photo that
  contributed nothing but an identity edge left the heading uncharged: `chars` came in short of
  the text, and a bundle sized to a tight `max_chars` rendered past it.
- A `max_chars` the context header alone overruns is refused with `ValidationError` naming the
  floor it needs, instead of returning a bundle over the limit it was given and reporting that
  oversized total as `chars`. The header is inside the bound and is written whether or not
  anything else fits, so such a budget has no bundle that satisfies it.
- `compile.media_items` in benchmark telemetry counts grounded media parts, which is the quantity
  `ContextBudget.max_media_items` bounds. It counted hits carrying any asset, so an omni memory
  with a still and a clip reported one part instead of two and multi-asset bundles looked as
  thrifty as single-asset ones.
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
- One badly grounded formation proposal no longer fails the write that produced it. An `AFFECT`
  proposal whose cue modality is absent from its source, or a pose in another coordinate frame,
  raised `ModelError` out of `add`/`add_many`/`settle` although the observation was already
  committed — so the record stayed durable with its formation stuck in the queue and every retry
  re-ran the model and failed identically. Text that merely mentions a photo is enough to make a
  model call it an image cue, so this was the ordinary case: a measured run failed 3 of 7 write
  chunks, then spent 1.7x the successful path's tokens re-forming records one at a time, and
  reported records as unwritten that were in fact stored and searchable. Such a proposal is now
  dropped and counted on the `mindbridge.formation.refused_proposals` span attribute — a total
  over the whole `add`, `add_many`, `settle`, or `consolidate` span, separate from the adapter's
  `mindbridge.formation.dropped_proposals` shape drops — while its
  siblings commit, which is the policy the model adapter already applied to a malformed proposal
  and consolidation already applied to this same rule. Two proposals from one source that
  contradict each other — one record proposed twice with different content, or two overlapping
  states in one lineage — are refused the same way, keeping the first of the pair, instead of
  failing the write. Damage to the batch envelope still raises.
- A formed record no longer drops the symbolic place and the metadata of the observation it was
  formed from, and a `capture()` no longer drops the place it was captured in. `place_id` is a hard
  SQL filter, so a place-scoped `search()`, `ask()`, or `compile()` could previously return only
  raw observations and never the entities, states, or relations formed from them — the household
  question the symbolic axis exists for. A captured record lost its place permanently, because
  `settle()` leaves the committed row alone. A consolidation rests on several sources, so it
  inherits the place and the metadata only when every cited source agrees, and inherits neither
  when they disagree: a hard retrieval filter must not be guessed. A naming assertion inherits
  neither in any case, because who somebody is does not stop being true in another room. Both
  columns follow the live evidence: deleting a source or rolling an operation back recomputes them
  over the sources that remain, so a record the survivors now agree on is scoped again instead of
  staying unreachable from a place- or metadata-filtered read for evidence that no longer exists.
  Derived records still carry no
  media assets of their own: their evidence link points at the observation that holds them.
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
- Lexical retrieval outside English. Term extraction matched an entire unsegmented run as one
  token, so a multi-character Chinese query carried a highest-weight term that could never match
  and could never reach full lexical coverage -- the one signal that performs cross-route fusion.
  Runs are now removed before word splitting and re-emitted as adjacent character bigrams, and 47
  Chinese function characters join the noise set that previously held only English stopwords.
  Separately, the index routed Japanese kanji to a Chinese word segmenter, which returned nothing
  for them, and routed Korean to an English stemmer, which cannot match an agglutinated eojeol; a
  script-agnostic character-bigram full-text field answers all three.
- Full-text search no longer picks one field per query by detecting the query's script, which cost
  it three separate ways. A single CJK character anywhere in a query sent the whole query to the
  bigram field, and only documents containing a listed script were admitted to that field, so
  `Alice 星期二 bakery` could not reach a Latin-only memory at all even though `Alice bakery`
  matched it exactly. And the listed scripts were an open set that never included Thai, Lao,
  Khmer or Myanmar, none of which any tokenizer here can split into words either: a query quoting
  a Thai memory verbatim reached neither field and returned nothing, silently. What goes to the
  bigram field is now decided by what the stemmed field can already answer rather than by which
  script it is: Latin words are stripped from it on both the write and the query side -- bigrams
  over them are close to information-free, "kitchen" and "the garden at noon" sharing "he" and
  "en" -- and everything else is kept. Both fields then answer every query that has something for
  each, fused by reciprocal rank at the same rank constant the route already used, and a purely
  Latin query still runs the single stemmed route it always did. The remaining failure direction
  is noise rather than silence: a script this misjudges keeps its bigrams. Rerank-side term
  extraction gained Thai, Lao, Khmer and Myanmar and now breaks bigrams at combining marks, which
  is where the index's tokenizer breaks them, so a Thai query's terms are the ones the index can
  actually answer. It also stops emitting the single characters of a run alongside the bigrams:
  the index cannot produce a one-character term at `ngram_min` 2, so those were weight in the
  coverage denominator that no document could answer, and they made the denominator
  language-dependent -- eleven terms for `爱丽丝面包店` against three for
  `Alice bakery Tuesday`, which put full coverage out of reach for Chinese at a length it stayed
  reachable for English. The index recipe is `context-keys-v12`; an existing store rebuilds its
  disposable index from SQLite on open and re-embeds nothing.
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
- A compiled bundle no longer overruns the `max_chars` it reports. An affect entry renders as an
  `AffectCue` — its basis, cue modality, valence, arousal, and evidence IDs are on the line — but
  the budget priced the plain hit it was selected as, and a resolved co-derived-event hop
  lengthened a line that had already been paid for. Selection now prices the cues it will render,
  and runs again over any entry a hop lengthened.
- A control-plane operation citing the record it would create is refused as `"target_is_evidence"`
  rather than failing a storage constraint and taking the whole pass down with it. Re-proposing a
  claim that already stands while citing that claim mints the cited record's own ID; a `REINFORCE`
  naming its target among its evidence is the same malformation.
- An identity-scoped read resolves the identity's memory set once instead of re-running the
  three-branch membership UNION for every candidate record — and, on a merge, for every embedding
  in the store.

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

- Every schema migration. A data directory is created at the current schema or refused, and never
  converted in place: the refusal names the version it found and says to re-create the directory
  and re-ingest, or to open it with the MindBridge version that wrote it. This removes seventeen
  upgrade steps, their DDL, and the twenty-five tests that drove them. **A `data_dir` written by
  an earlier build of this unreleased version cannot be opened by this one.** Nothing has been
  released, so no published version is affected.
- `MemorySettings`, the alias for `MemoryConfig`. Declarative `settings` and
  `Memory.from_plugins(config=...)` are unchanged; the class keeps one public name.
- `uvicorn` from the `server` and `all` extras. MindBridge never imports an ASGI server, so the
  deployment installs the one it runs; `docs/deployment.md` shows it beside the command.
- Benchmark code with no caller: the durable evaluation journal, the library half of the paired
  replay helper (the research driver in `benchmarks/paired_replay.py` is unaffected), and the
  second scoring runners in the Video-MME-v2 and EgoTempo adapters that `mindbridge-bench eval`
  replaced. Ten unread `*_ADAPTER_VERSION` constants and the `BENCHMARK_PROMPTS` registry go with
  them; the prompts themselves and every task in the catalog are unchanged.
- Video-MME, EgoLifeQA, EgoMemReason, and MemEye benchmark tasks, adapters, media preparation, and
  the EgoMemReason submission artifact. Video-MME-v2 remains supported.
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
- Records formed before this release keep the empty `place_id` and `metadata` they were written
  with; nothing backfills them. A source is marked formed for its recipe, so re-adding the
  observation forms nothing new: re-form by changing the formation recipe, or add the content
  again as a fresh observation. For the same reason a refused proposal is final for that recipe —
  the source is complete, and the proposal is not retried.

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
