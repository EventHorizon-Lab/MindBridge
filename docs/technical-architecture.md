# Technical architecture

## Package layout

```text
src/mindbridge/
├── memory.py                    # public orchestration and consistency
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
- `speech()` requires `SpeechBackend` because plain transcription has no timing or speaker data.

Jina application text is wrapped as a non-string text value before its remote model code receives
it. This prevents path- or URL-shaped application text from being reclassified as media. Real
media is always a resolved local CAS path.

## Durable schema

SQLite stores:

- Memory records, memory type, event start/end, storage timestamps, metadata, and access
  reinforcement.
- Asset descriptors, transcript cache, speech turns, speaker centroids, and names.
- Normalized FP32 aggregate, atomic, and contextual text embeddings with parent memory, object
  part, model, space, task, and dimension.
- Store metadata for embedding identity, transcription space, and index recipe.
- Ordered pending index operations.

Foreign keys and transactions keep records, assets, embeddings, and outbox changes atomic.

## Index protocol

For every durable mutation:

1. Commit SQLite data and an outbox operation.
2. Read a bounded outbox batch.
3. Hydrate current SQLite truth for the affected embedding IDs.
4. Upsert current documents and delete missing documents in Zvec.
5. Flush Zvec.
6. Acknowledge the exact SQLite outbox batch.

Interrupted work remains durable. Startup drains it. Reindexing builds from SQLite, then replays
the outbox so a record committed after the rebuild scan is not lost from the projection.

Composite records store one aggregate vector and de-duplicated atomic text/media vectors through
the existing embedding `object_part`. Only the aggregate carries full text into BM25, preventing
one record from crowding lexical results. Dense and hybrid hits hydrate their authoritative parent
IDs from SQLite and collapse by maximum relevance before temporal/decay reranking.

## Provider boundary

`OpenAIModels` receives already-constructed SDK clients. It supplies model IDs and semantic
payloads to SDK methods, validates returned shapes, and maps failures to `ModelError`. It does not
read keys, create transports, normalize base URLs, implement retries, or close clients.

`SentenceTransformersEmbedder` and `FunASRTranscriber` similarly delegate inference to their
installed runtimes. MindBridge retains only validation and mapping needed by its public semantics.

## Threading and lifecycle

`Memory` allows model calls and independent SQLite record commits outside its index lock. Outbox
replay and Zvec access are serialized; several committed writes may therefore share one flush. CAS
leases prevent temporary query media from being removed while another operation is using it.

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
