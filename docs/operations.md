# Operations

MindBridge has no database service or worker queue. Operate the one process that owns a local
`data_dir`, its model backends, and the filesystem containing that directory.

## Health

The REST adapter exposes an unauthenticated liveness endpoint:

```bash
curl --fail http://127.0.0.1:8000/healthz
```

```json
{"status":"ok"}
```

`/healthz` does not call a model or run retrieval. Opening the application proves only that the
store, durable metadata, and index were openable at startup. For an end-to-end check, use a
separate canary process and directory; do not add canary records to production memory.

## Backup

Use a stopped-process snapshot:

1. Stop the owner gracefully so active calls finish and resources close.
2. Verify that no process still uses the directory.
3. Copy the complete directory, including any SQLite side files.
4. Open and query the copy from a new temporary path with the same backend configuration.
5. Restart the original owner.

The `.mindbridge.lock` file is only the target of an operating-system lock. It normally remains
after shutdown, and its presence does not mean an owner is alive.

SQLite and `assets/` are authoritative. A smaller backup may omit `zvec/`, but it must retain
`state.sqlite3`, any SQLite side files present in the stopped snapshot, and all media under
`assets/`. Test that rebuilding the omitted index fits the restore-time and free-space budget.
SQLite cannot reconstruct missing media bytes.

## Restore

Restore while the target service is stopped:

1. Restore into a new, empty directory and apply the service account's ownership.
2. Restore the complete snapshot, or restore SQLite plus `assets/` and leave `zvec/` absent.
3. Start one `Memory` with the embedding, transcription, face, and index settings that created the
   store.
4. Verify known text and media searches before accepting traffic.

When `zvec/` is absent, startup queues all authoritative embeddings and rebuilds the collection
without embedding historical content again. Pending outbox work is replayed before the instance
opens. An incompatible model, vector space, dimension, analysis space, schema, or unrecognized
index recipe fails at open instead of mixing state.

Restore with the original MindBridge version and configuration first. Perform a version upgrade as
a separate backed-up step because a recognized recipe migration may re-embed records.

## Rebuild and optimize

Use the only live owner of the directory and the same backend configuration as the store:

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory(
    "/var/lib/mindbridge/assistant",
    embedder=JinaOmniEmbedder(),
) as memory:
    count = memory.reindex()
    memory.optimize()
    print(f"reindexed {count} memories")
```

`reindex()` replaces the disposable Zvec collection from FP32 embeddings in SQLite, then replays
the outbox to include writes committed during its scan. `optimize()` drains the outbox, merges
staged Zvec vectors, and flushes the collection. Normal adds, deletes, searches, and startup also
drain the outbox; routine index maintenance runs automatically.

If Zvec cannot open or its files are corrupt, stop the owner, retain a backup, move only `zvec/`
aside, and reopen the store. Keep the moved index until a known search succeeds. Never move SQLite
or `assets/`, edit `search_index_queue`, or replace the index while a `Memory` is live.

Zvec process-wide resource settings may be set once before constructing any `Memory`:

```python
import zvec

zvec.init(query_threads=4, optimize_threads=2, memory_limit_mb=2048)
```

MindBridge does not call `zvec.init`; the application owns that process-global policy. Account for
MindBridge running several focused dense routes and one lexical route concurrently, with no more
than four outer search workers per search.

`IndexQuantization` changes only the derived Zvec collection. Keep `NONE` unless measured capacity
and retrieval results justify a lossy mode. `RABITQ` requires an embedding dimension from 64 through
4095 and native runtime support; changing quantization rebuilds the index without re-embedding
records.

## Capacity signals

Track at least:

- Bytes and free inodes for the whole directory, SQLite, `assets/`, and `zvec/` separately.
- Free space for an index rebuild or compaction alongside the current index.
- Process file-descriptor use; Zvec refuses work before exhausting the configured soft limit.
- Add, search, and answer latency, plus model latency and failures by operation.
- Startup, restore, reindex, and optimize duration.
- REST status and stable error-code counts.

Original media and FP32 embeddings are authoritative storage costs; index quantization does not
remove those vectors from SQLite. Composite and long-text records also create bounded additional
embedding documents.

## Telemetry

MindBridge depends on `opentelemetry-api`. With no configured SDK, instrumentation is a no-op.
Install the optional SDK and configure an exporter before constructing `Memory`, or inject a tracer
with `Memory(..., tracer=...)`:

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

Spans use `mindbridge.span.kind` to distinguish three levels:

| Kind | Names |
| --- | --- |
| Operation | Public operations such as `mindbridge.add`, `mindbridge.search`, `mindbridge.ask`, `mindbridge.delete`, `mindbridge.reindex`, and `mindbridge.optimize` |
| Stage | `mindbridge.content.prepare`, `mindbridge.storage.*`, `mindbridge.index.*`, and `mindbridge.retrieval.rank` |
| Model | `mindbridge.model.embedding`, `.transcription`, `.face`, and `.generation` |

Each `add_stream` item produces an ordinary `mindbridge.add` operation. `search_with_trace` and
speculative prefetch use ordinary `mindbridge.search` operations. `AsyncMemory` preserves tracing
context through its worker thread.

Model spans record `gen_ai.operation.name`, `gen_ai.request.model`,
`mindbridge.model.module`, `mindbridge.model.batch_size`, `mindbridge.input.modalities`, and
`mindbridge.model.request_count`. When a provider reports usage, spans record standard GenAI
input/output totals plus exact MindBridge modality attributes:

```text
mindbridge.token_usage.input_tokens.<text|image|video|audio|unattributed>
mindbridge.token_usage.output_tokens.<text|image|video|audio|unattributed>
```

Operation spans roll up descendant model usage. Interpret these together:

| Attribute | Meaning |
| --- | --- |
| `mindbridge.token_usage.expected_request_count` | Provider requests expected to report token usage. |
| `mindbridge.token_usage.reported_request_count` | Requests that supplied usage. |
| `mindbridge.token_usage.complete` | Every expected request reported a usable total. |
| `mindbridge.token_usage.total_tokens` | Exact total when complete; otherwise present only as an exact reported lower bound. |
| `mindbridge.token_usage.audio_seconds` | Provider- or runtime-reported processed audio duration. |

MindBridge never estimates tokens from text length, media bytes, duration, or a different model's
tokenizer. Local backends report request counts and available audio duration but do not invent
token usage.

Streaming generation additionally reports `mindbridge.model.time_to_first_token` for the first
non-empty text delta. Providers may report `gen_ai.response.time_to_first_chunk` and
`gen_ai.response.finish_reasons`. OpenAI grounding reports
`mindbridge.grounding.media_elided_hits` and `mindbridge.grounding.dropped_hits` when request limits
shrink retrieved evidence.

Spans do not record memory text, media bytes, paths, asset or memory IDs, metadata, model responses,
or exception details. Failed spans receive only error status. For one retrieval investigation, use
the opt-in `search_with_trace()` result or the local `search-with-trace` CLI command; candidate IDs
are intentionally excluded from normal telemetry.

## Failure interpretation

| Signal | Check first |
| --- | --- |
| `validation_error` or REST 422 | Caller input, media type, event bounds, or unsupported modality. |
| `model_error` | Backend configuration, credential, endpoint, timeout, response shape, or declared capability. |
| `index_unavailable` | Zvec open, file descriptors, mutation, flush, search, or on-disk collection. |
| `storage_error` | Directory ownership, filesystem, CAS, SQLite, schema, or compatibility metadata. |
| `memory_not_found` or REST 404 | The record is absent from authoritative SQLite. |

Stable errors expose `code`, optional `reason`, pipeline `stage`, retryability, and an optional
`subject`. CLI, REST, and MCP envelopes also include a `trace_id`. Log those fields with request
metadata, but do not log memory content, credentials, provider bodies, or local paths by default.

## Upgrades

Back up and restore-test the directory before changing MindBridge, model revisions, vector
dimensions, transcription recipes, face recipes, or index settings. Startup applies only known
schema and recipe migrations. Never bypass a compatibility failure by editing `PRAGMA user_version`,
`store_metadata`, or the outbox manually.
