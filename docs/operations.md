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

Before first deployment or after composition changes, use the CLI doctor described in the
[command-line reference](api/cli.md#doctor-and-explain). It checks configuration and loaders
without writing memory. A remote doctor checks the REST liveness endpoint.

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

Restore with the original MindBridge version and configuration first. Perform an upgrade as a
separate backed-up step because a recognized embedding recipe migration may re-embed records.

## Index maintenance and repair

Normal startup, add, delete, and search operations drain the SQLite outbox. Zvec also performs
bounded automatic optimization after enough pending vectors or durable segments accumulate.

From the live owner, rebuild or optimize explicitly when measurement or diagnosis justifies it:

```python
count = memory.reindex()
memory.optimize()
print(f"reindexed {count} memories")
```

`reindex()` replaces the derived collection from SQLite FP32 embeddings and then replays writes
committed during its scan. `optimize()` drains pending work, merges staged vectors, and flushes the
collection. Neither operation repairs missing media or converts an incompatible embedding space.

If Zvec cannot open or appears corrupt:

1. Stop the owner and retain a tested backup.
2. Move only `zvec/` aside; do not touch `state.sqlite3` or `assets/`.
3. Start one owner with the original configuration and allow startup to rebuild the index.
4. Verify representative searches.
5. Remove the moved index only after the verification succeeds.

Never edit `search_index_queue`, replace `zvec/`, or move authoritative files while a `Memory` is
live.

`capture_queue` is deferred enrichment rather than index work, and no operation drains it
implicitly. A host that uses `capture()` owns the loop that calls `settle()`, and
`pending_captures()` is what to alarm on: it returns up to `limit` queued records oldest first,
each with its `enqueued_at`, `attempts`, and `last_error`, so a result that stays at the limit
means the loop stopped and those records are unsearchable until it resumes. Drain it before a
planned shutdown, because a queued row survives restart.

A record whose `attempts` reached the `settle(max_attempts=...)` ceiling — three by default — is
skipped rather than retried, so it stops holding up the records behind it while staying queued and
visible with the reason it failed. Fix the cause and raise the ceiling for one call to retry it.
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
route, with at most four outer search workers per search. Keep `IndexQuantization.NONE` unless
measured capacity and retrieval results justify a lossy mode. Changing only quantization rebuilds
Zvec from stored vectors; `RABITQ` requires dimensions from 64 through 4095 and native support.

## Capacity

Monitor:

- Bytes and free inodes for the whole directory, `state.sqlite3`, `assets/`, and `zvec/`.
- Free space for a second index during rebuild or compaction.
- Process file descriptors; Zvec checks headroom before durable work.
- Startup, add, search, ask, reindex, optimize, and model latency.
- Stable error codes and retryable reasons by operation.
- Backup age, restore-test result, and observed recovery duration.

Original media and FP32 embeddings remain authoritative storage costs even when Zvec uses
quantization. Composite and long-text records create bounded additional embedding documents.

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

from mindbridge import JinaOmniEmbedder, Memory

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

with Memory("./data", embedder=JinaOmniEmbedder()) as memory:
    memory.add("Remember this")
    memory.search("Remember")

provider.shutdown()
```

Every span sets `mindbridge.span.kind` to `operation`, `stage`, or `model`.

| Level | Useful span names |
| --- | --- |
| Operation | `mindbridge.add`, `.add_many`, `.search`, `.ask`, `.delete`, `.reindex`, `.optimize` |
| Stage | `mindbridge.content.prepare`, `.storage.*`, `.index.*`, `.retrieval.rank`, `.identity.*` |
| Model | `mindbridge.model.embedding`, `.transcription`, `.face`, `.generation`, `.formation`, `.vision` |

Each `add_stream()` item creates an ordinary `mindbridge.add` span. `search_with_trace()` and
speculative prefetch use `mindbridge.search`. `AsyncMemory` preserves tracing context across its
worker thread.

Capture acknowledgement, settle duration, and time to searchable are three different numbers and
are measured separately. `mindbridge.capture` and `mindbridge.settle` are distinct operation
spans, and the settle span carries the capture-to-searchable interval its batch closed:

| Attribute | Meaning |
| --- | --- |
| `mindbridge.capture.records_settled` | Records this call made searchable. |
| `mindbridge.capture.records_failed` | Records that failed and kept their queue row. |
| `mindbridge.capture.max_time_to_searchable_ms` | Longest capture-to-searchable interval this call closed. Absent when nothing settled. |

Model spans include request model, module, batch size, modalities, and
`mindbridge.model.request_count`. Provider-reported usage is recorded without estimation:

| Attribute | Meaning |
| --- | --- |
| `mindbridge.token_usage.expected_request_count` | Requests expected to report token usage. |
| `mindbridge.token_usage.reported_request_count` | Requests that supplied usage. |
| `mindbridge.token_usage.complete` | Every expected request supplied a usable total. |
| `mindbridge.token_usage.total_tokens` | Exact total when complete; otherwise only an exact reported lower bound. |
| `mindbridge.token_usage.input_tokens.<modality>` | Exact input tokens for text, image, video, audio, or unattributed input. |
| `mindbridge.token_usage.output_tokens.<modality>` | Exact output tokens for the same modality set. |
| `mindbridge.token_usage.audio_seconds` | Provider- or runtime-reported processed audio duration. |

Operation spans roll up descendant model usage. Local backends report request counts and available
audio duration but do not invent token counts. Streaming generation records
`mindbridge.model.time_to_first_token` after its first non-empty text delta. Providers may also
record `gen_ai.response.time_to_first_chunk` and `gen_ai.response.finish_reasons`.

Watch these degradation and recognition attributes:

- `mindbridge.embedding.elided_parts` counts retrieval keys rejected because media exceeded the
  embedding model's inline limit.
- `mindbridge.embedding.video_sampled_inputs` counts video inputs embedded as four ordered stills
  because the model's prompt exceeded its context.
- `mindbridge.grounding.media_elided_hits` and `mindbridge.grounding.dropped_hits` count evidence
  removed from OpenAI grounding requests.
- `mindbridge.identity.faces` and `mindbridge.identity.speakers` report observations, distinct
  identities, existing matches, creations, and cache status for each analyzed asset.

Spans never record memory text, media bytes, paths, IDs, metadata, model responses, or exception
details. A failed span receives only error status. Use `search_with_trace()` for one retrieval
investigation and the structured error envelope for failures.

## Upgrade

Before changing MindBridge, model revisions, vector dimensions, transcription or face recipes, or
index settings:

1. Create and restore-test a backup.
2. Stop the owner.
3. Upgrade code and configuration together.
4. Start one owner and let only recognized schema or recipe migrations run.
5. Verify records, media, retrieval, and telemetry before restoring traffic.

Never bypass compatibility checks by editing `PRAGMA user_version`, `store_metadata`, or the
outbox.
