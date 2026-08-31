# Technical architecture

## Package layout

```text
src/mindbridge/
├── memory.py                    # public orchestration and consistency
├── plugins.py                   # explicit capability and policy composition
├── types.py                     # stable values
├── exceptions.py                # stable failures
├── models/
│   ├── base.py                  # narrow operation protocols
│   ├── openai_sdk.py            # caller-owned official SDK adapter
│   ├── sentence_transformers.py # generic local embedding adapter
│   ├── jina.py                  # pinned Jina specialization
│   └── funasr.py                # official AutoModel speech mapping
├── infrastructure/local/
│   ├── store.py                 # authoritative SQLite state and outbox
│   ├── assets.py                # local content-addressed media
│   └── zvec_index.py            # rebuildable search projection
├── api/
│   ├── app.py                   # optional REST adapter
│   └── mcp.py                   # optional MCP adapter
└── benchmarks/                  # public-SDK behavior runners
```

Product modules never import benchmark modules.

## Content normalization

Python content atoms are `str`, `Path`, `Blob`, and `AssetRef`. Strings are always text. Paths and
blobs are copied into the CAS before a model sees them. Asset references are resolved against
SQLite in the same directory.

REST and MCP decode base64 data URLs into `Blob` values and map stored IDs to `AssetRef`. HTTP(S)
URLs and server filesystem paths are rejected. This keeps network fetch behavior out of the
product and prevents protocol callers from selecting local files.

Canonical input order, normalized text, content digests, metadata, event time, and memory type
produce a stable memory ID. Identical media bytes share a CAS object.

## Model routing

Each operation validates atomic modality sets from its adapter:

- Embedding routing uses `embedding_capabilities` and stores document vectors in the declared
  embedding space.
- Generation routing uses `generation_capabilities` and never drops unsupported visual evidence.
- Audio unsupported by embedding or generation may be replaced with a transcript when a
  transcriber exists.
- Transcript derivation routes on `transcription_capabilities`, not on the gaps in the embedding
  adapter. A `TranscriptionBackend` therefore transcribes declared audio and video during `add`,
  and the derived text is stored beside the native media vector rather than replacing it.
- `speech()` requires `SpeechBackend` because plain transcription has no timing or speaker data.

Jina application text is wrapped as a non-string text value before its remote model code receives
it. This prevents path- or URL-shaped application text from being reclassified as media. Real
media is always a resolved local CAS path.

## Durable schema

SQLite stores:

- Memory records, memory type, event start/end, storage timestamps, metadata, and access
  reinforcement.
- Asset descriptors, transcript/face caches, timed observations, bounded face/voice exemplars,
  shared identity names, and aliases for IDs retired by a face/voice merge.
- Normalized FP32 aggregate, atomic, and contextual text embeddings with parent memory, object
  part, model, space, task, and dimension.
- Store metadata for embedding identity, transcription and configured face spaces, and index recipe.
- Ordered pending index operations.

Foreign keys and transactions keep records, assets, embeddings, identity merges, aliases, and
outbox changes atomic. Biometric matching scans exact model spaces in SQLite and takes the maximum
cosine score over each identity's bounded exemplar set; Zvec remains a disposable content-search
projection.

## Index protocol

For every durable mutation:

1. Commit SQLite data and an outbox operation.
2. Read a bounded outbox batch.
3. Hydrate current SQLite truth for the affected embedding IDs.
4. Upsert current documents and delete missing documents in Zvec.
5. Flush Zvec.
6. Trigger background index maintenance after 100,000 additional vectors remain unindexed.
7. Acknowledge the exact SQLite outbox batch.

Interrupted work remains durable. Startup drains it. Reindexing builds from SQLite, then replays
the outbox so a record committed after the rebuild scan is not lost from the projection.

Composite records store one aggregate vector and de-duplicated atomic text/media vectors through
the existing embedding `object_part`. Only the aggregate carries full text into BM25, preventing
one record from crowding lexical results. Queries batch their complete aggregate with bounded
focused aggregate and atomic keys derived from the first text atom and query media; later text atoms
never become independent routes. The focused text remains the lexical query. Dense and lexical
routes execute concurrently. Zvec oversamples groups by parent memory so sibling document
embeddings cannot consume the candidate budget; when native best-effort grouping returns too few
parents, a bounded ordinary-query fallback fills them. Dense and lexical routes stay separate until
SQLite drops stale embedding IDs and collapses every route to its authoritative parent memory. Zvec
ranks dense hits by nonnegative cosine while separately rescaled cosine confidence drives
weak-evidence gating. Parent relevance preserves the strongest dense evidence, admits lexical-only
evidence through relative BM25, and applies bounded exact lexical agreement before temporal, decay,
and ambiguity calibration. The standard FTS field lowercases, folds accents, and stems English; a
parallel Jieba field handles Han queries. For ordered multi-text queries, the first text atom and
media provide the focused dense and lexical routes while later text remains only in the complete
aggregate. Event-start and event-end inverted indexes enable Zvec's range optimization.

Relative-time parsing accepts an explicit `reference_at` first. Without one, a natural
`Today is <date>` declaration becomes the reference clock and is removed before parsing phrases
such as `last week`, so the declared day is not mistaken for the requested evidence day.

## Provider boundary

`OpenAIModels` receives already-constructed SDK clients. It supplies model IDs and semantic
payloads to SDK methods, validates returned shapes, and maps failures to `ModelError`. It does not
read keys, create transports, normalize base URLs, implement retries, or close clients.

`SentenceTransformersEmbedder`, `FunASRTranscriber`, and `OpenCVFaceAnalyzer` similarly delegate
inference to their installed runtimes. MindBridge retains only validation and mapping needed by its
public semantics.

## Threading and lifecycle

`Memory` allows model calls, independent SQLite record commits, and ordinary Zvec queries outside
its write lock. Outbox replay remains serialized, so several committed writes may share one flush.
The Zvec adapter admits concurrent queries but exclusively gates collection replacement and close.
Final SQLite hydration and CAS leases share the write boundary so returned media cannot disappear.

Closing starts a lifecycle barrier, rejects new calls, waits for active calls, and closes each
unique adapter or storage resource once. Forked processes must create a new instance and distinct
directory.

## Trust boundaries

- Python callers can intentionally read local paths; the final path must be a regular file and is
  opened without following a final symlink where supported.
- REST and MCP never accept local paths or remote network URLs.
- Model output and provider exceptions are untrusted and validated before persistence or return.
- Public REST/MCP errors omit provider bodies, credentials, and filesystem details.
- Authentication, TLS, rate limits, and request identity remain deployment concerns.
