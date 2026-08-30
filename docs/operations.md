# Operations

MindBridge has no separate database or worker fleet to operate. Operational state is one local
directory plus the selected embedding, generation, and speech backends.

## Health

The REST adapter exposes:

```bash
curl http://127.0.0.1:8000/healthz
```

```json
{"status":"ok"}
```

The route is unauthenticated. Successful application startup already proves the store and index
could be opened, but `/healthz` is a process-liveness response; it does not make a model request or
run a retrieval query.

For deeper monitoring, execute a synthetic add/search/delete against a dedicated canary directory
or service. Do not place canary records in production memory.

## Backup

Use a stopped-process backup for a coherent and portable snapshot:

1. Stop the process gracefully.
2. Confirm no process owns the data directory.
3. Copy the complete directory, including SQLite side files if present.
4. Verify the backup can be opened in a temporary location.
5. Restart the original process.

The lock file is not the state; the operating-system lock is. A leftover `.mindbridge.lock` file
after a clean stop is normal and must not be used as evidence that a process is running.

SQLite and `assets/` are authoritative. A smaller recovery backup may retain those two and omit
`zvec/`, but restore testing must confirm that index rebuild fits startup time and disk-space
budgets. SQLite alone cannot reconstruct media bytes.

## Restore

Restore only while the target process is stopped:

1. Restore into a new, empty directory.
2. Apply ownership for the service account; on POSIX, startup enforces top-level mode `0700`.
3. If the Zvec copy is absent or intentionally discarded, leave `zvec/` absent.
4. Start one `Memory` instance with the original embedding settings.
5. Run known text and media retrieval checks.

When `zvec/` is absent, startup queues every authoritative embedding and rebuilds the index without
re-embedding historical content. A configuration metadata mismatch stops startup; use the
embedding model, vector-space ID, dimension, and recipe that created the store.
Restore the original `transcription_space` as well. It identifies the ASR model and
transcript-affecting recipe used by cached transcripts and add-time derived text; a mismatch fails startup.
Configured face embedding and analysis spaces have the same fail-fast guard.

A recognized legacy retrieval-key recipe is different from a missing Zvec directory: startup
re-embeds authoritative records, commits the new recipe marker, and rebuilds Zvec. Schema-only
upgrades from the context-key v6 or grouped-range v7 recipes reuse stored vectors and rebuild only
the disposable Zvec projection.

## Rebuild and optimize

Use the Python API while holding the only live instance:

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory(
    "/var/lib/mindbridge/assistant",
    embedder=JinaOmniEmbedder(),
) as memory:
    indexed = memory.reindex()
    memory.optimize()
    print(f"indexed {indexed} memories")
```

`reindex()` rebuilds the derived collection from SQLite. `optimize()` merges and flushes staged Zvec
data. Routine outbox drains also optimize after 64 durable flushes or 100,000 unindexed vectors and
copy-on-write compact the disposable collection after 256 flushes, bounding Zvec's segment file
descriptors. Compaction briefly needs a second index-sized disk allocation. The explicit operations
remain useful before latency-sensitive workloads; schedule them when traffic is low.

Routine adds, deletes, and searches drain the durable outbox automatically. Do not edit
`search_index_queue` by hand.

Zvec derives process-wide thread and memory defaults from the available CPU and cgroup limits. To
override them, initialize Zvec once at application startup, before constructing any `Memory`:

```python
import zvec

zvec.init(query_threads=4, optimize_threads=2, memory_limit_mb=2048)
```

This is process-global and cannot be changed at runtime. MindBridge deliberately does not call it,
so multiple local directories cannot silently compete to replace one another's resource policy.
MindBridge may fan out up to four independent dense/lexical routes for one composite query; include
that outer concurrency when selecting `query_threads`.

The optional `IndexQuantization` setting changes only the derived vector index. Zvec retains the
original vectors alongside quantized data, so quantization reduces active memory and may improve
query throughput but does not necessarily reduce disk bytes. `RABITQ` additionally requires
x86_64 with AVX2 and dimensions from 64 through 4095. Keep the default `NONE` unless measured
quality and capacity results justify a lossy mode.

## Capacity signals

Track at least:

- Total bytes under `data_dir`.
- SQLite, `assets/`, and Zvec bytes separately.
- Free bytes and inodes on the containing filesystem.
- Add and search latency.
- Embedding, generation, and transcription latency/failures separately.
- Encoded media bytes per OpenAI adapter call; embeddings reject aggregates over 64 MiB after
  base64 expansion, while answers reserve the same limit for question media and fill the remainder
  with ranked evidence.
- REST status counts, especially 502 and 503.
- Startup and rebuild duration.

MindBridge emits these operation and stage timings directly as OpenTelemetry spans, including
generation TTFT and provider-reported multimodal token usage. See
[performance and token observability](observability.md) for the stable attribute contract and
export setup.

The standalone local-index benchmark reports SQLite bytes, Zvec bytes, ingest time, optimize time,
query percentiles, QPS, and recall against exact search. See [benchmarking](benchmarking.md).

## Failure interpretation

| Signal | Likely boundary |
| --- | --- |
| `ModelError` or REST 502 | Capability routing, model endpoint, credential, response, or dimension |
| `IndexUnavailableError` or REST 503 | Zvec open, mutation, flush, or search |
| `StorageError` or REST 503 | Lock, filesystem, CAS, SQLite, schema, or metadata mismatch |
| REST 422 | Invalid caller input |
| REST 404 for a memory ID | Record is absent from authoritative SQLite |

Each REST error includes a `trace_id`. Log it with request metadata, but do not log memory content or
credentials by default.

## Upgrades

Back up the directory before changing MindBridge versions. Schema or index recipe changes are
compatibility events and may require an explicit migration or new directory. Never bypass a startup
compatibility failure by editing `PRAGMA user_version` or `store_metadata` manually.

The schema 6 to 7 migration is automatic: each legacy speaker centroid becomes the first voice
exemplar, existing IDs and names remain stable, and the shared face/voice identity tables are added.

The upgrade from the former service architecture is described in the
[quick start](quickstart.md#upgrading-from-the-service-based-release).
