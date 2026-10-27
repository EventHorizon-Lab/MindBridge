# Operations

Operate the process that owns `data_dir`, its model backends, and the local filesystem. MindBridge
has no database service, worker queue, or background owner to manage separately.

## Start and stop

Start one owner with the same model and index recipes that created the store. Opening validates the
SQLite schema and compatibility metadata, opens or creates Zvec, and drains durable index work
before returning.

Stop gracefully by calling `Memory.close()`. It rejects new operations, waits for active work,
releases media leases, closes model and storage resources, and releases the directory lock. The
`.mindbridge.lock` file remains; only its operating-system lock indicates ownership.

`close(timeout_seconds=30.0)` bounds the caller's complete wait, including backend cleanup and
index flush. A timeout raises `StorageError(reason="timeout", stage="close")`. Cleanup continues
on one closing thread; the instance rejects new work and retains directory ownership until its
workers and resources actually stop. Call `close()` again to wait for that same cleanup. Never
delete its lock file to work around a timeout. Synchronous provider calls cannot be forcibly
cancelled safely; configure finite provider timeouts as well.

Before first deployment or after composition changes, use the CLI doctor described in the
[command-line reference](api/cli.md#doctor-and-explain). It checks configuration and loaders
without writing memory. A remote doctor checks the REST liveness endpoint.

## Concurrency and overload

Each physical `data_dir` owns its admission limits and worker leases. Process budgets are shared
across directories. The fixed runtime bounds are:

| Resource | Limit |
| --- | --- |
| Active memory operations across SDK, REST, and MCP | 32 per directory; 128 per process |
| Operations that can invoke a model | 28 per directory and per process |
| Submitted asynchronous SDK calls, including live vision descriptions | 32 per directory; 128 per process |
| Pinned answer-stream workers shared by AsyncMemory and REST | 8 per directory; 32 per process |
| Index-route workers | 4 per directory; 16 per process |
| Speech-overlap workers | 2 per directory; 8 per process |
| SQLite connections, including borrowed and idle handles | 32 per directory; 128 per process |
| REST `/v1` requests and MCP tool calls sharing one directory | 32 per directory; 128 per process |
| Open local stores | 16 per process |
| Materialized input text and media retained by active operations | 512 MiB per operation and per process |
| Inputs in one SDK `add_many()` | 1,024 |
| Captures awaiting enrichment or formation | 10,000 per directory |

Admission is immediate: overload raises `StorageError(reason="capacity_exceeded",
stage="admission")`, which is retryable. REST returns the shared 503 envelope with
`Retry-After: 1`; MCP reports the same reason and retryability. Apply bounded backoff when
retrying. `/healthz` remains outside request admission.

Model work leaves four of a directory's operation slots available to lightweight reads. The
process model budget also leaves room in shared SDK and framework worker pools; it does not
guarantee progress through backend locks or unrelated host work. Local model adapters retain
their inference safety locks. Formation calls over distinct sources may overlap; concurrent
writes of the same source share one formation call, and application remains serialized and
transactionally checked. Naming and identity rollback commit registry names, speech text,
vectors, and repair outbox together, including when embedding fails.

REST reads a request body within 30 seconds total and allows at most 30 seconds for each response
send. Body timeout returns the usual retryable 503 `timeout` envelope. A send timeout ends the
response and returns request capacity. A terminal answer releases its pinned worker before
network delivery. MCP tool admission runs before framework worker dispatch; HTTP connection and
MCP session limits still belong to the serving host.

Index routes use batches of at most four per admitted operation; temporal supplements reuse
the same pool. At most 128 route jobs and 32 speech-overlap jobs can be pending or running.
Parent asynchronous calls use a separate pool so they cannot occupy the workers their child
routes need. Cancelling an asynchronous waiter retains capacity until its synchronous work
actually stops. Stream workers remain pinned through queued cleanup and are reusable before
the terminal result is delivered. Close an abandoned stream with `close()` or `aclose()`.

The process shares 64 asynchronous parent threads, 16 index threads, eight speech threads, and
at most 32 pinned stream threads, created as needed and reused across directories. Closing one
directory waits for its own jobs without shutting down another directory's workers. Each closing
directory can also use one cleanup thread. File-descriptor exhaustion is reported as
`capacity_exceeded`, including during open, rather than as a missing search index.
SQLite connection opens and native index allocations share one descriptor admission lock and
preserve filesystem descriptor headroom before allocating; the numeric connection ceiling is
also subject to the process's actual file-descriptor limit.

These bounds cover MindBridge-owned submissions, workers, stores, and SQLite handles. The byte
budget charges UTF-8 text and media sizes before materialization and releases them on operation
exit. Inputs whose own cumulative size exceeds 512 MiB raise `ValidationError`; contention for
otherwise valid input raises retryable `capacity_exceeded`. This is not an RSS or GPU-memory
ceiling. Caller-owned buffers and threads, framework bodies,
decoded media, native model/Zvec allocations, backend request sizes, and provider quotas still
need host sizing. A full capture backlog refuses new queued records atomically; settle existing
captures before retrying. Duplicate captures remain idempotent at the limit.
The limits are conservative safety bounds, not measured optimal throughput settings.
Use backend timeouts and bound the host's own producers. At the HTTP server boundary,
Uvicorn's [concurrency limit](https://uvicorn.dev/server-behavior/#resource-limits) can
also reject excess requests based on open connections and in-flight tasks. A worker count alone
does not bound queued submissions;
Python's [executor documentation](https://docs.python.org/3/library/concurrent.futures.html)
also describes nested-pool deadlocks and cancellation limits.

## Health

The REST adapter exposes an unauthenticated liveness endpoint outside `/v1`:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

`/healthz` does not call a model, inspect pending outbox rows, or run retrieval. It proves only that
the current process can serve the request; its response also includes the injected `Memory`'s live
embedding, generation, transcription, vision, face, formation, speaker-recognition, and streaming
capabilities. Initial store and index opening happened before the app was constructed. Run any
end-to-end canary against a separate directory so production memory is not polluted.

## Backup

### Capture backlog before a snapshot

A stopped-owner backup preserves queued captures, including failed work. It does not make them
searchable. Before stopping, inspect the queue through the existing owner:

```python
pending = memory.pending_captures(limit=100)
for item in pending:
    print(item.memory_id, item.awaiting, item.attempts, item.last_error)
```

`awaiting="enrichment"` means the record has no searchable vectors; `"formation"` means it is
already searchable and owes typed formation only. The report is bounded: naming IDs asks about
those records specifically. To finish a batch, call `memory.settle(limit=100)` from the host's
approved model-work window. A failed batch may still settle other records; inspect the queue
again after an error. Naming a failed ID in `settle(memory_ids=[...])` bypasses its retry ceiling.
Neither startup nor shutdown drains this queue automatically.

### Stopped-owner copy

Use a stopped-owner snapshot:

1. Stop the owner gracefully and verify that no process still holds the directory lock.
2. Copy the complete directory, including SQLite side files if any remain.
3. Store the snapshot with the MindBridge version and model/index configuration that created it.
4. Restore the snapshot to a new temporary path.
5. Open that copy with the recorded configuration and verify known text and media records.
6. Restart the original owner.

`state.sqlite3` and `assets/` are required. A smaller backup may omit `zvec/`, but restoring it then
requires index rebuild time and enough free space. Never omit media: SQLite stores descriptors and
transcripts, not the original bytes.

Do not treat a copied `.mindbridge.lock` file as evidence that a snapshot is busy. Do not copy only
`state.sqlite3` from a running owner; use a proper SQLite-aware snapshot mechanism if downtime is
not acceptable.

## Restore

Restore into a new, empty path while the target service is stopped:

1. Restore the complete snapshot, or restore `state.sqlite3` plus `assets/` and leave `zvec/`
   absent.
2. Apply the service account's ownership and restrictive permissions.
3. Start exactly one `Memory` with the recorded embedding, transcription, face, and index settings.
4. Verify known records with `get()` or `list()`, then verify representative text and media
   searches.
5. Route traffic to the restored owner only after those checks pass.

When `zvec/` is absent, startup durably queues all stored embeddings, creates the collection, and
replays the queue without embedding historical content again. An unsupported schema or
unrecognized model, vector-space, dimension, analysis-space, or index-recipe mismatch fails at
open instead of mixing state.

Restore with the MindBridge version and configuration that wrote the backup. The current runtime
accepts only its own schema and model spaces. To use another schema or embedding recipe, read the
source data with its original runtime and re-ingest it into a fresh directory.

## Index maintenance and repair

Normal startup, add, delete, and search operations drain the SQLite outbox. Drained rows are
searchable at once; the Zvec flush that acknowledges them runs once 256 applied rows accumulate,
when a drain applied a deletion, inside `optimize()` and `reindex()`, and at `close()`. Zvec also
performs bounded automatic optimization after enough pending vectors or durable segments
accumulate. That bound is a write-side one that a session below roughly 65,000 written rows never
reaches, so a session that flushed at least once merges the durable segments at `close()` as well
whenever two or more of them are persisted. A session that only read the store never does, so
opening someone else's store to search it, back it up, or check a result hands the bytes back
unchanged; its compaction debt is paid by the next session that writes. This is what keeps short
sessions from growing the index: five one-record sessions over a 481 MB store left it at 511 MB in
seven segments without the merge, and at 481 MB in one segment with it, for about 1.4 s of close
each.

From the live owner, rebuild or optimize explicitly when measurement or diagnosis justifies it:

```python
count = memory.reindex()
memory.optimize()
print(f"reindexed {count} memories")
```

`reindex()` replaces the derived collection from SQLite FP32 embeddings, replays writes committed
during its scan, and merges the segments that replay leaves behind, so it ends no more segmented
than `optimize()` does. `optimize()` drains and acknowledges pending work, merges staged vectors,
and flushes the collection. Neither operation repairs missing media or converts an
incompatible embedding space.

Once a collection holds enough vectors for Zvec to build a graph, dense search is approximate and
nothing in the index reports the neighbours it missed. MindBridge therefore searches at the
largest candidate list Zvec accepts. That is chosen per query rather than stored, so it is not
part of the index recipe and changing it neither rebuilds nor invalidates a store. It costs
roughly 6 ms of `search()` p50 on a 24,271-vector 2048-dimension store, and the index-only part
of that grows with the corpus: about 21 ms more per dense route at 100,000 vectors. It costs
nothing on a store small enough that every query is already an exhaustive scan.

If Zvec cannot open or appears corrupt:

1. Stop the owner and retain a tested backup.
2. Move only `zvec/` aside; do not touch `state.sqlite3` or `assets/`.
3. Start one owner with the original configuration and allow startup to rebuild the index.
4. Verify representative searches.
5. Remove the moved index only after the verification succeeds.

Never edit `search_index_queue`, replace `zvec/`, or move authoritative files while a `Memory` is
live.

Zvec 0.8 fixes the deleted-row mapping defect that required copying a collection after an
update or deletion. Ordinary `optimize()` now merges in place, including after reopening a
collection. The old `zvec/.mindbridge-clean` marker is no longer used.

On the first open after upgrading from the recognized Zvec 0.7 index recipes, MindBridge
rebuilds the disposable projection from SQLite's stored FP32 embeddings. This also repairs
old vector-to-ID mapping damage and changes INT8 projections to unrotated cosine quantization.
It does not call the embedding model. Allow time and space for the rebuild before accepting
traffic; keep the original SQLite and media backup if you need to restore the old application.
Unknown recipes still fail rather than guessing how to interpret stored state.

Compaction remains necessary when accumulated idmap files or process file-descriptor pressure
cannot be reclaimed by an in-place merge. It copies live documents into a replacement collection,
so it needs room for a second copy of `zvec/` beside the original. A session killed during this
operation leaves `.zvec.compact-*` debris, which the next open removes, and can leave `zvec/`
absent between renames, which the next open rebuilds from SQLite. Do not remove file-descriptor
headroom or compaction capacity just because ordinary optimize no longer copies the collection.

`capture_queue` is deferred enrichment rather than index work, and no operation drains it
implicitly. A host that uses `capture()` owns the loop that calls `settle()`, and
`pending_captures()` is what to alarm on: it returns up to `limit` queued records oldest first,
each with its `enqueued_at`, `attempts`, `last_error`, and `awaiting`, so a result that stays at
the limit means the loop stopped and those records are unsearchable until it resumes. `awaiting`
separates a row with no vectors (`"enrichment"`) from one that is already searchable and owes only
formation (`"formation"`), so a backlog of the second kind is not a recall outage. Drain the queue
before a planned shutdown, because a queued row survives restart.

A record whose `attempts` reached the `settle(max_attempts=...)` ceiling — three by default — is
skipped rather than retried, so it stops holding up the records behind it while staying queued and
visible with the reason it failed. Fix the cause, then either raise the ceiling for one call or
name the record with `settle(memory_ids=...)`, which ignores the ceiling for the records it names.
With a formation backend configured, `add()` also holds a queue row for the moment between its
commit and its formation, so a short-lived entry under a live writer is expected rather than a
stalled loop.

`memory_operations` is an append-only audit log of every applied control-plane operation.
`operations()` reads it newest first and `rollback(operation_id)` reverses one; neither is
scheduled maintenance, and neither is reachable over REST or MCP.

Applications may set Zvec process-wide resources once, before constructing any `Memory`:

```python
import zvec

zvec.init(query_threads=4, optimize_threads=2, memory_limit_mb=2048)
```

MindBridge does not call `zvec.init()`. Account for several focused dense routes plus one lexical
route, with at most four shared outer search workers per `Memory`. Keep `IndexQuantization.NONE` unless
measured capacity and retrieval results justify a lossy mode. Changing only quantization rebuilds
Zvec from stored vectors; `RABITQ` requires dimensions from 64 through 4095 and native support.
Quantized structures coexist with the FP32 vectors. Historical Zvec 0.7 measurements over ten
stores holding 5,882 records grew `zvec/` by 54 % under `FP16`, 55 % under `INT8`, and 86 % under
`RABITQ`, with no measurable retrieval gain; those percentages are not measurements of 0.8.
Measure resident memory, latency, recall, and disk bytes separately before selecting a lossy mode.

Quantized retrieval keeps a wider parent candidate pool and recalculates final cosine scores from
SQLite's FP32 embeddings before ranking or applying `minimum_relevance`. This improves score
calibration and near-tie ordering without another model call, at the cost of reading and scoring
the surviving candidate vectors. It does not repair ANN candidates that were never discovered.
The [accuracy-first evaluation](research/2026-10-26-accuracy-first-retrieval.md) records the
synthetic quality and latency tradeoffs; these are not real-model semantic accuracy measurements.

Zvec 0.8 INT8 uses `enable_rotate=False`: the rotated cosine path returned unrelated neighbors
in the upgrade evaluation. Grouped retrieval does not support native refinement, so enabling a
refiner is not a repair for that path. Uniform UINT4/7/8 require L2, and unrotated INT4 reduced
grouped recall without a demonstrated speed gain. These modes remain outside the public
configuration. Embedded Jieba is available upstream, but MindBridge retains the stemmed/bigram
fields for multilingual and mixed-script recall. See the
[Zvec 0.8 evaluation](research/2026-10-26-zvec-08.md) for measurements and API limitations.

## Capacity

Growth and deletion are separate policies. `memory_budget_records` creates a control-plane
pressure signal; it does not itself delete records. A configured retention age is also inert until
the host calls `apply_retention()`. Start with a dry run and review the direct and cascading IDs:

```python
report = memory.apply_retention(dry_run=True)
print(report.deleted, report.asset_ids, report.capture_memory_ids)
```

Physical retention and `delete()` cannot be reversed by `rollback()`. Cognitive `forget()` can.
See [retention configuration](configuration.md#retention-policy) and
[data-subject rights](api/python-sdk.md#data-subject-rights) before scheduling deletion.

Monitor:

- Bytes and free inodes for the whole directory, `state.sqlite3`, `assets/`, and `zvec/`.
- Free space for a second index during rebuild or compaction.
- Process file descriptors; Zvec checks headroom before durable work.
- Startup, add, search, ask, reindex, optimize, and model latency.
- Stable error codes and retryable reasons by operation.
- Backup age, restore-test result, and observed recovery duration.

Original media and FP32 embeddings remain authoritative storage costs even when Zvec uses
quantization, which adds its structure beside them. Composite and long-text records create bounded
additional embedding documents.

## Telemetry

MindBridge depends on `opentelemetry-api`; without an SDK, spans are no-ops. Install the existing
observability extra and configure a provider before constructing `Memory`, or inject a tracer:

```bash
uv add "mindbridge[local,observability]"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from mindbridge import SentenceTransformersEmbedder, Memory

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

with Memory(
    "./data",
    embedder=SentenceTransformersEmbedder(
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        revision="e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
    ),
) as memory:
    memory.add("Remember this")
    memory.search("Remember")

provider.shutdown()
```

Every span sets `mindbridge.span.kind` to `operation`, `stage`, `model`, or `transport`.

| Level | Useful span names |
| --- | --- |
| Operation | `mindbridge.add`, `.add_many`, `.search`, `.ask`, `.delete`, `.reindex`, `.optimize` |
| Stage | `mindbridge.content.prepare`, `.retrieve`, `.storage.*`, `.index.*`, `.retrieval.score_completion`, `.retrieval.rank`, `.identity.*` |
| Model | `mindbridge.model.embedding`, `.transcription`, `.face`, `.generation`, `.formation`, `.vision` |
| Transport | `mindbridge.http.request`, `mindbridge.mcp.request` |

Each `add_stream()` item creates an ordinary `mindbridge.add` span. `search_with_trace()` and
speculative prefetch use `mindbridge.search`. `AsyncMemory` preserves tracing context across its
worker thread. Both buffered `AsyncMemory.ask()` and `AsyncMemory.ask_stream()` record their
initial executor queue delay.

Inside `mindbridge.ask`, `mindbridge.retrieve` measures the complete prerequisite-to-ranked-hits
leg: question content preparation, reference-time and temporal parsing, scope normalization,
required query speech work, query embedding, index lookup, and ranking. Its existing
`mindbridge.content.prepare`, model, `mindbridge.index.search`, and `mindbridge.retrieval.rank`
children remain available for attribution.

Hybrid searches emit `mindbridge.retrieval.score_completion` only when a parent admitted by the
lexical route was absent from the bounded ANN result. The stage covers reading that parent's
persisted document vectors from SQLite and exact local cosine scoring; it makes no model request.

`mindbridge.index.sync` retains one parent span and separates its durable stages without changing
the SQLite-before-Zvec-before-ack order:

| Child stage | Work measured |
| --- | --- |
| `mindbridge.index.sync.sqlite.read` | Read and hydrate the durable SQLite outbox batch. |
| `mindbridge.index.sync.zvec.apply` | Delete or upsert the derived Zvec documents. |
| `mindbridge.index.sync.zvec.flush` | Flush the applied Zvec changes; present only in the drain that reaches the flush bound, in `optimize()` and `reindex()`, and at `close()`. |
| `mindbridge.index.sync.zvec.optimize` | Run the conditional Zvec optimization check after a flush. |
| `mindbridge.index.sync.sqlite.ack` | Acknowledge the flushed outbox rows in SQLite. |

Capture acknowledgement, settle duration, and time to settled are three different numbers and
are measured separately. `mindbridge.capture` and `mindbridge.settle` are distinct operation
spans, and the settle span carries the capture-to-settled interval its batch closed:

| Attribute | Meaning |
| --- | --- |
| `mindbridge.capture.records_settled` | Records this call made searchable. |
| `mindbridge.capture.records_failed` | Records that failed and kept their queue row. |
| `mindbridge.capture.max_time_to_searchable_ms` | Longest capture-to-settled interval this call closed, over every row settled including `add()` recoveries. Absent when nothing settled. |

Model spans include request model, module, batch size, modalities, and
`mindbridge.model.request_count`. Provider-reported usage is recorded without estimation:

`mindbridge.model.request_count` counts logical adapter calls issued by MindBridge. Retries hidden
inside a third-party SDK transport remain included in latency, but cannot be counted separately;
token totals and completeness cover observable logical responses, not an earlier hidden attempt for
which the provider returned no usage.

| Attribute | Meaning |
| --- | --- |
| `mindbridge.token_usage.expected_request_count` | Requests expected to report token usage. |
| `mindbridge.token_usage.reported_request_count` | Requests that supplied usage. |
| `mindbridge.token_usage.complete` | Every expected request supplied a usable total. |
| `mindbridge.token_usage.total_tokens` | Exact total when complete; otherwise only an exact reported lower bound. |
| `mindbridge.token_usage.input_tokens.complete` | Every expected request supplied an input-token component. |
| `mindbridge.token_usage.output_tokens.complete` | Every expected request supplied an output-token component. |
| `mindbridge.token_usage.cached_input_tokens.complete` | Every expected request supplied a cached-input component. |
| `mindbridge.token_usage.reasoning_output_tokens.complete` | Every expected request supplied a reasoning-output component. |
| `mindbridge.token_usage.input_tokens.<modality>` | Exact input tokens for text, image, video, audio, or unattributed input. |
| `mindbridge.token_usage.output_tokens.<modality>` | Exact output tokens for the same modality set. |
| `gen_ai.usage.cache_read.input_tokens` | Provider-reported cached input tokens. |
| `gen_ai.usage.reasoning.output_tokens` | Provider-reported reasoning output tokens. |
| `mindbridge.token_usage.audio_seconds` | Provider- or runtime-reported processed audio duration. |

Operation spans roll up descendant model usage. Local backends report request counts and available
audio duration but do not invent token counts. Cached-input and reasoning-output tokens participate
in the same model-to-operation rollup. Embedding model spans set `mindbridge.embedding.task` to
`retrieval.document` or `retrieval.query`, so ingest and retrieval inference can be separated.

OpenAI-compatible model spans record `gen_ai.response.model` and
`gen_ai.openai.response.system_fingerprint` only when the response supplies a non-empty value.
Streaming inspects the final usage chunk too. `mindbridge.model.response_models` and
`mindbridge.model.response_system_fingerprints` retain all distinct values when one span covers
several responses; the scalar field is `mixed` instead of silently assigning the whole span to the
last response. These are response provenance, not a copy of the request or response body.

Provider request IDs are recorded as `gen_ai.request.id` for a single request and collected in
`mindbridge.model.request_ids` when one model span covers multiple requests. They are taken from
the SDK response or provider error, including failure paths; an absent ID is left absent rather
than invented. A local trace ID identifies the MindBridge operation, while the provider request
ID identifies the upstream call for support. Keep both in protected operator telemetry. This
follows OpenAI's [request debugging guidance](https://developers.openai.com/api/reference/overview#debugging-requests).

The first-output clocks deliberately use different scopes and units:

| Attribute | Unit | Boundary |
| --- | --- | --- |
| `mindbridge.operation.time_to_first_token_ms` | Milliseconds | Start of the complete `mindbridge.ask` operation through its first non-empty visible text yield, including retrieval and generation setup. Absent if no visible text is yielded. |
| `mindbridge.model.time_to_first_token` | Seconds | Generation-model request through its first non-empty text delta. This preserves the generation-only model baseline and is absent when no model token is produced. |
| `gen_ai.response.time_to_first_chunk` | Seconds | OpenAI-compatible streaming request through the first provider chunk, even when that chunk carries no visible text. |
| `mindbridge.async.queue_time_ms` | Milliseconds | Initial buffered or streaming `AsyncMemory.ask*` executor queue delay. It is also included in the operation TTFT, while remaining available for decomposition. |

Transport spans measure server-side delivery separately from model and operation timing:

| Transport | Attributes and boundary |
| --- | --- |
| REST | `mindbridge.http.request` records the low-cardinality `http.route` template, `mindbridge.transport.response_mode`, `mindbridge.transport.server_time_to_headers_ms`, `mindbridge.transport.server_time_to_first_body_byte_ms` when a non-empty body is sent, and `mindbridge.transport.server_total_ms`. All three times are milliseconds from server request handling; they are not client-observed network latency. |
| MCP | `mindbridge.mcp.request` records buffered `mindbridge.transport.server_total_ms` in milliseconds. MCP does not stream tool results, so it does not claim TTFT, first-chunk, or first-body timing. |

Providers may also record `gen_ai.response.finish_reasons` on generation model spans.

Watch these degradation and recognition attributes:

- `mindbridge.embedding.elided_parts` counts retrieval keys rejected because media exceeded the
  embedding model's inline limit.
- `mindbridge.embedding.video_sampled_inputs` counts video inputs embedded as four ordered stills
  because the model's prompt exceeded its context.
- `mindbridge.grounding.media_elided_hits` and `mindbridge.grounding.dropped_hits` count evidence
  removed from OpenAI grounding requests.
- `mindbridge.recall.shape`, `mindbridge.recall.ops`, `mindbridge.recall.exhaustive_rows`,
  `mindbridge.recall.complete`, `mindbridge.recall.replan`, and `mindbridge.recall.fallback`
  describe one `mindbridge.recall` stage, present only when `recall_planning` is on. Every
  planning failure resolves to the same fallback plan, so `fallback` is the only signal that the
  planner did not decide that question, and `complete` is what licenses a count over the set.
- Experimental `recall_query_expansion` adds a `mindbridge.recall.ground` stage.
  `mindbridge.recall.query_count`, `query_failures`, and `query_limited` are cumulative per ask;
  use the final round rather than summing them. Other `mindbridge.recall.query_*` counts describe
  that round: `candidates` is the distinct returned query hits, `new_candidates` excludes the
  original ranked pool, `selected` counts novel hits offered after per-query row caps,
  `grounded` counts actual additions, `budget_dropped` counts selected hits rejected by character
  or media caps, and `omitted` counts all novel query hits left out. A zero `new_candidates`
  means no expansion beyond the original pool, not proof of corpus completeness. These counts
  precede provider-side media packing and cannot establish correct reader use.
- `mindbridge.formation.refused_proposals` counts the proposals the kernel refused to ground,
  totalled over the whole operation span and present only when formation ran; the model adapter's
  `mindbridge.formation.dropped_proposals` counts the ones it could not read at all.
- `mindbridge.identity.faces` and `mindbridge.identity.speakers` report observations, distinct
  identities, existing matches, creations, and cache status for each analyzed asset.

Spans never record memory text, media bytes, paths, IDs, metadata, request or response bodies, or
exception details. Apart from the non-content response model, fingerprint, and provider request IDs
above, a failed span
receives only generic error status. Use `search_with_trace()` for one retrieval investigation and
the structured error envelope for failures.

## Model quality gate

Transport tests run without credentials and verify requests, validation, refusals, and error
mapping. A separate small real-model gate exercises the public SDK with synthetic memories in a
fresh directory for every case. It checks a grounded fact, insufficient evidence despite an
unrelated retrieved memory, and instructions embedded in stored content. Every case must pass
every repeat; an upstream error fails the gate rather than disappearing from its denominator.

Grounded fixtures require a single answer word, allowing case, punctuation, and Markdown
decoration. Negated, conflicting, or additional injected text fails. The report identifies this
as an answering-only gate: the output-protocol option configures structural calls, but these
answering fixtures do not establish that an endpoint accepts strict schemas. Verify formation,
planning, vision, and consolidation separately when changing a provider or its protocol.

Set `OPENAI_API_KEY` through a secret manager or the process environment. For a compatible server,
set `OPENAI_BASE_URL` to the SDK base URL ending in `/v1`, without `/chat/completions`. Select an
explicit model snapshot and the output protocol the endpoint supports:

```bash
uv run --frozen python -m mindbridge.benchmarks.model_quality \
  --model "$MODEL_QUALITY_MODEL" \
  --response-format json_schema \
  --repeats 3 \
  --output .benchmarks/model-quality/report.json
```

`MODEL_QUALITY_MODEL` names the model snapshot to evaluate. Use `--response-format json_object`
only for a server that lacks strict schema support. The report retains the fixture version,
configured and returned model identifiers, case pass/fail results, timing, and stable error codes;
it does not retain prompts, answers, credentials, or raw provider exceptions. These synthetic
fixtures do not authorize testing with private production content or establish full application
quality. Use the [dataset harness](benchmarking.md) for broader comparisons, and calibrate quality
criteria against reviewed failures before expanding this suite.

The [model-quality workflow](../.github/workflows/model-quality.yml) runs manually with explicit
model, response-format, and repeat inputs. Scheduled runs are opt-in: configure the repository
secret `MODEL_QUALITY_API_KEY`, set `MODEL_QUALITY_ENABLED=true`, and provide the
`MODEL_QUALITY_MODEL` snapshot variable. `MODEL_QUALITY_BASE_URL` and
`MODEL_QUALITY_RESPONSE_FORMAT` optionally select a compatible endpoint and protocol. Review the
aggregate artifact before approving prompt, model, or retrieval-policy changes. Keep the ordinary
credential-free CI gate enabled even when real-service evaluation is unavailable.

Recorded validation on 2026-10-26 used `gpt-5-mini-2025-08-07` as both the configured and returned
model with fixture `mindbridge-reader-quality-v2`: all four cases passed each of three repeats
(12/12). Separate strict-schema probes passed formation, consolidation, recall planning, vision,
and media selection. The local aggregate reports remain under `.benchmarks/openai-practices/`.
These synthetic checks do not establish benchmark gains or parity with other model deployments;
live EmbeddingGemma 2 through vLLM was not verified.

The formation-time fixture `mindbridge-formation-time-v1` independently tests the snapshot
`gpt-5-mini-2025-08-07` with strict JSON Schema and a 4096-token output budget. Its synthetic
source says “I moved to Paris yesterday” with observation time `2026-03-02T12:00:00Z`, no typed
validity, and no history. The four cases cover observation time despite later creation, creation
time as the fallback, no reference time, and an embedded instruction to use a fabricated date.
Each runs three times. Dated cases must produce an event whose `valid_from` is March 1; the case
without a reference must omit temporal bounds. Every repeat must pass. Run it explicitly with:

```bash
MINDBRIDGE_FORMATION_TIME_EVAL=1 uv run --frozen pytest -W error -rP \
  tests/integration/test_formation_time_quality.py
```

Only fixture IDs, case IDs, repeat numbers, and configured/returned model identifiers are printed.
On 2026-10-27 the credential-free regression reproduced the prior March 2 fallback and verified
timestamp propagation and persistence of the resolved March 1 date after the fix. The real-model
formation fixture and an equivalent real-model baseline comparison were not run because no
service credentials were configured. Transport fixtures establish no model-quality improvement;
run the credentialed checks before making that claim.

## Changing the runtime or stored representations

Before changing MindBridge, model revisions, vector dimensions, transcription or face recipes, or
index settings:

1. Create and restore-test a backup.
2. Stop the owner.
3. Upgrade code and configuration together.
4. Start one owner with matching schema and model identities, or re-ingest into a fresh directory.
5. Verify records, media, retrieval, and telemetry before restoring traffic.

Never bypass identity and schema checks by editing `PRAGMA user_version`, `store_metadata`, or the
outbox.

**Only SQLite schema 20 is accepted.** Startup validates required tables, constraints, and
predicate provenance before admitting the store. It never upgrades an older schema or re-embeds
stored content. A nonempty store must contain all required identity and evidence-policy markers.
The optional face backend can be enabled for the first time when no face analysis or face-space
markers exist; startup records its two space identities together.

`memory_semantics.predicate` retains the original model wording; `predicate_key`,
`predicate_registry`, and `predicate_match` record the stable identifier, registry revision, and
resolution method. Custom lineage IDs remain unchanged. Rebuilding Zvec cannot change semantic
keys, evidence policy, or model spaces.

Current clauses remain the authoritative `OR` of complete `AND` member sets; the flat
`memory_evidence` rows project their active members. Zvec remains derived: rebuilding a missing
index reads the SQLite embeddings and current clause projection without calling the embedding
model again.

Clause versions are append-only transaction history. An inverse written by operation rollback
points to the exact predecessor version it restores. This lets operations reverse newest first
while an unrelated later update, including an equal-valued one, still blocks an older rollback.
Physical `delete()` remains unlogged and irreversible; clause history does not recreate erased
records.

A derived record whose basis is not a host assertion is visible only while it has an active
evidence clause. Such a record stays stored and readable by ID, but drops out of retrieval until
evidence is added or it is deleted.

The schema declares `embedding_text_selectors` and `embedding_text_span_pieces`. A selector belongs
to one pure-text raw embedding part and stores a parent-content digest, reconstructed
embedding-input digest, recipe, and ordered code-point ranges with piece digests. It does not
duplicate parent text. Reopening a current-schema store is idempotent.
A partial or incompatible
selector-table shape is refused after checking primary keys, foreign keys, check constraints, and
the piece index, and nothing is written. Deleting an embedding cascades its selectors. Zvec rebuild
reads existing SQLite vectors and does not create selectors or re-embed content.
