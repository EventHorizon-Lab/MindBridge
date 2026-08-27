# Technical architecture

This page documents the embedded v0.2 durability, media, model-routing, and retrieval invariants.
It is intended for maintainers and operators diagnosing local state.

## Package boundaries

```text
src/mindbridge/
├── __init__.py                    # stable public exports
├── config.py                      # endpoints, capabilities, and URL policy
├── exceptions.py                 # stable public failures
├── memory.py                      # synchronous core and async facade
├── types.py                       # content and result values
├── api/                           # optional REST and MCP transports
├── models/
│   ├── base.py                    # public embedding and combined-model protocols
│   ├── sentence_transformers.py   # standard multimodal embedding adapter
│   ├── jina.py                    # pinned Jina Omni compatibility adapter
│   └── openai_http.py             # OpenAI-compatible combined backend
└── infrastructure/local/
    ├── _lock.py                   # cross-process directory ownership
    ├── assets.py                  # immutable media CAS and safe URL fetch
    ├── store.py                   # authoritative SQLite state
    └── zvec_index.py              # disposable hybrid-search projection
```

The public orchestration depends directly on these local adapters. A custom embedder or combined
model implements one public protocol; there is no provider registry or dynamic plugin factory.
SQLite and Zvec do not
need speculative repository interfaces because the product currently supports one local storage
path.

## Data directory lifecycle

Construction resolves the path and acquires `.mindbridge.lock` without waiting. Failure to acquire
ownership maps to `StorageError`. After locking, MindBridge:

1. Enforces POSIX mode `0700` on the top-level directory and opens or creates `state.sqlite3`.
2. Migrates supported schemas through SQLite schema v5 when required.
3. Validates required tables and stored embedding/index metadata.
4. Opens the content-addressed asset store.
5. Checkpoints all embeddings if the Zvec directory is missing.
6. Opens or creates `zvec/` and drains durable index work.

`close()` waits for active operations, then closes the model backend, Zvec, SQLite, and filesystem
lock. It is idempotent. Operations after close or from a forked child raise `StorageError`.

## On-disk layout

```text
data_dir/
├── .mindbridge.lock
├── state.sqlite3
├── state.sqlite3-wal             # present during WAL activity
├── assets/
│   └── ab/
│       └── abcdef...             # SHA-256 content address
└── zvec/
```

The top-level directory is forced to owner-only mode `0700` on POSIX, including when it already
exists; SQLite and lock files use `0600`. A source `Path` is validated as a regular file, opened
without following a final symlink where the platform supports that flag, streamed through SHA-256,
and atomically installed. URL downloads are streamed with a byte limit. For each hop, MindBridge
resolves the exact allowlisted hostname, rejects the hop if any result is non-public, and connects
to a verified public IP while retaining the original host for HTTP and TLS SNI. Redirects repeat the
entire check. A concrete expected MIME must match response `Content-Type` exactly; a family range
matches only that image, video, or audio family.

## SQLite authority

Every connection enables:

```text
foreign_keys = ON
journal_mode = WAL
synchronous = FULL
busy_timeout = 30000
```

Writes use `BEGIN IMMEDIATE`. The multimodal schema contains:

| Table | Authoritative content |
| --- | --- |
| `memory_records` | Text, modality, memory role, metadata, event/access times, bounded access count |
| `media_assets` | SHA-256 descriptor, MIME, modality, size, relative CAS path, optional transcript |
| `memory_assets` | Ordered memory-to-asset relationships |
| `embeddings` | FP32 vector, model, space, task, dimension, normalization, object part |
| `speech_analyses` | Completed ASR recipe and transcript per media asset |
| `speech_segments` | Timed transcript turns linked to local speaker identities |
| `speaker_identities` | Optional name, CAM++ centroid, and observation count per local speaker |
| `store_metadata` | Embedding, transcription, and index compatibility identity |
| `search_index_queue` | Ordered `upsert` and `delete` operations awaiting Zvec flush |

Foreign keys cascade record deletion to relationships and embeddings. An asset descriptor/file is
garbage-collected only when it is no longer referenced. SQLite triggers append index operations
when embeddings change.

Memory pages are ordered by `(created_at DESC, memory_id DESC)`. The public cursor encodes the last
pair with a version marker and is not a general query token.

## Content normalization and identity

For each add, MindBridge:

1. Flattens one atom or an ordered atom sequence without treating `str` as a filesystem path.
2. Trims and NFC-normalizes text atoms.
3. Resolves `AssetRef` through SQLite and materializes Path, Blob, and allowed URL atoms into CAS.
4. Infers Path/URL media type only from a deterministic common suffix when no explicit hint exists.
5. Derives modality from the set of media families.
6. Validates the semantic, episodic, or procedural memory role.
7. Converts timezone-aware event time to UTC microsecond precision.
8. Serializes metadata as sorted compact JSON and rejects non-finite or unsupported values.
9. Hashes canonical text, ordered asset digests, event time, metadata, and any non-default role
   into the memory ID; omitting the default semantic marker preserves existing semantic IDs.

The same bytes have one asset ID regardless of source. Memory identity preserves asset order. A
duplicate memory is detected before embedding or transcription; source Path/URL bytes still must be
read to establish their digest unless the caller supplies an existing `AssetRef`. Asset `name` is
stored once per digest, not per relationship. The first authoritative non-empty name is returned
when later inputs supply a different name for the same bytes.

## Model input and fallback

One memory or query becomes one `ModelInput(text, assets)`. The current release stores one
aggregate document embedding at object part zero; the schema field does not imply a public
multi-vector feature.

Embedding uses `EmbedTask.DOCUMENT` for memories and `EmbedTask.QUERY` for queries. The generic
Sentence Transformers adapter maps these to `encode_document` and `encode_query`, builds standard
multimodal dict/message values, and submits one ordered model batch. The Jina adapter alone maps
inputs to the pinned model's legacy tuple form. The OpenAI-compatible shape has no task field, so
it validates the task locally but does not serialize it.

Local `space_id` is a digest of adapter recipe, model ID, immutable revision, effective native or
advertised Matryoshka dimension, normalization, and retrieval-side methods. It is not manually
overridable. A changed model contract must target a new directory and re-encode; `reindex()` only
rebuilds the derived Zvec collection from authoritative stored vectors.

For each operation, routing compares input atomic modalities with `ModelCapabilities`:

- Fully supported input goes directly to the operation.
- Unsupported audio is transcribed when transcription supports it. Transcript text replaces that
  audio for the current call; supported image/video assets remain.
- The reduced input is validated again. Any remaining unsupported modality raises `ModelError`.

Transcripts can be retained with authoritative asset metadata so repeated use need not invoke ASR
again. Add-time embedding fallback persists derived transcript text in `memory_records.content`.
Generation-time fallback caches the asset transcript but does not rewrite an existing record's
content. Neither fallback changes the persisted media modality. SQLite stores the backend's
`transcription_space` beside the embedding/index recipe and rejects a different value at startup.
The identifier must change with any ASR model, preprocessing, language, prompting, or decoding
choice that can change transcript text; changing it requires a new directory and re-ingestion.

The default speech route is FunASR `AutoModel` composed from Fun-ASR-Nano, FSMN-VAD, and CAM++;
the explicit vLLM route batch-decodes the same VAD spans before the same CAM++ clustering.
`Memory.speech` stores timed turns and matches normalized CAM++ centroids within one physical
directory. `SPK0`-style labels are asset-local and are never treated as identities. The first
observation enrolls an opaque `speaker_id`; later clear cosine matches reuse it.
`Memory.register_speaker` attaches a display name without exposing the stored voiceprint.

## Asset transport to models

`MINDBRIDGE_MEDIA_TRANSPORT=data` sends resolved assets as base64 data URLs. `file` sends local
`file:` URLs for a trusted co-located backend. Generation uses standard content parts and the
OpenAI-compatible input-audio shape where applicable. Transcription uses multipart upload to
`/audio/transcriptions`.

For built-in `data` transport, an embedding or generation call is rejected when the aggregate raw
size of its media exceeds 64 MiB, before base64 expansion. Batch embedding counts the media across
the whole call. Answer construction de-duplicates by asset ID, so the binary part for a shared asset
appears once even when the question and several hits refer to it. The `file` transport avoids inline
encoding for a co-located model; provider-native upload or streaming belongs in a custom backend.

The transport choice is separate from ingestion. A remote source is always fetched, validated,
and stored locally before a model sees it; model adapters never receive an untrusted remote URL
directly from the caller.

## Durable index outbox

The outbox is a synchronous recovery boundary, not a background-worker queue. Under the write
lock, each drain:

1. Reads at most 256 ordered operations to bound vector memory.
2. Keeps the last operation per embedding ID within that batch.
3. Reads current SQLite documents for retained IDs.
4. Deletes absent IDs and upserts present documents in Zvec.
5. Flushes Zvec.
6. Deletes the exact original operation rows from SQLite.

Any index failure occurs before acknowledgement. Replaying is safe because desired state comes
from current SQLite truth. Reindex streams SQLite documents in bounded pages rather than loading
the complete corpus.

## Concurrency boundary

The lifecycle counter allows operations on one instance to overlap, and model work runs outside
`Memory`'s write lock. SQLite commit/outbox changes
and Zvec mutation/search are briefly serialized under that lock so a reader cannot observe a
partially applied projection. `reindex()`, `optimize()`, deletion, close, and asset cleanup acquire
the same boundary where required. Sentence Transformers calls use a process-global lock because
the pinned Jina remote module temporarily patches Sentence Transformers during load; MindBridge
binds those methods only to the Jina instance and restores the class before releasing the lock. A
second process still cannot own the directory.

## Retrieval

Vectors are normalized before persistence. When routed query text is non-empty, hybrid search sends
that text and the normalized aggregate vector to Zvec so it can fuse dense and lexical candidates.
A pure-media query uses dense vector search. Both paths are constrained to the stored vector space,
retrieval task, optional memory role, and detected event-time range. SQLite hydration rechecks role
and event time, so a stale or corrupted derived field cannot authorize a hit.

Temporal or decay ranking over-fetches at least 50 candidates or three times the requested limit.
Temporal matches rank before fallback candidates. Optional decay uses the latest access, event, or
update time and a bounded access-strength factor; only returned hits receive durable reinforcement.
Public scores are clamped to `[0, 1]`; stale IDs are removed. The exact parser and formula are in
[memory types, temporal reasoning, and decay](memory-types-time-and-decay.md).

There is currently no chunking, multiple embeddings per memory, metadata-filter pushdown, or
learned reranker. Quality and latency work must measure the one-memory/one-vector contract instead
of claiming an unimplemented pipeline.

## Failure mapping

Internal failures do not leak adapter exceptions through the public API:

| Boundary | Public exception |
| --- | --- |
| Invalid content, metadata, asset source, cursor, time, or limit | `ValidationError` |
| Missing memory ID in `get` | `MemoryNotFoundError` |
| Missing or contradictory opaque asset reference | `ValidationError` |
| Capability routing, embedding, transcription, or generation | `ModelError` |
| Directory, schema, lock, CAS, SQLite, or durable metadata | `StorageError` |
| Zvec open, mutation, flush, rebuild, or query | `IndexUnavailableError` |

All inherit `MindBridgeError`. REST and MCP map them to stable sanitized envelopes.

## Deliberate non-goals

The embedded v0.2 architecture does not provide distributed writers, logical account partitions,
automatic role extraction, background consolidation, executable procedural memories, server-side
metadata filters, automatic media chunking, per-asset vectors, a learned reranker, or a runtime
plugin registry. New layers require a measured public use case.
