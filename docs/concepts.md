# Core concepts

MindBridge is an embedded memory library, not a hosted memory service. The application supplies
content and model capabilities; MindBridge owns local persistence, retrieval, and grounded-answer
orchestration.

The durable mental model is: model work produces embeddings and analyses, SQLite commits the
authoritative record, Zvec projects it for search, and every result is hydrated back from SQLite.

## Content becomes a record

Python accepts one content atom or an ordered sequence of atoms:

| Atom | Meaning |
| --- | --- |
| `str` | Application text |
| `pathlib.Path` | A local media file copied into the asset store |
| `Blob` | In-memory media bytes with an explicit MIME type |
| `AssetRef` | Media already stored in the same data directory |

The same `ContentInput` shape is used by `add`, `search`, and `ask`. `add_stream` also accepts
`StreamInput` so each completed observation can carry its own event time, metadata, memory type,
and typed observation context. That optional `MemoryContext` records provenance, world-time
validity, spatial pose, and affect semantics alongside the observation.

MindBridge never downloads remote URLs. In Python, a URL-shaped `str` is still text; fetch remote
media in the application and pass a `Blob` or local `Path`. REST and MCP reject network URLs and
server filesystem paths at their trust boundaries.

Every record has one persisted modality: `text`, `image`, `video`, `audio`, or `omni`. Text plus one
media family keeps that media modality; combining multiple media families produces `omni`.

## Records and assets have stable identities

`Memory.add()` returns a `MemoryRecord` containing its ID, text, modality, memory type, timestamps,
metadata, and resolved assets. Search returns the same application fields in ranked `SearchHit`
values.

Memory IDs are derived from canonical content, ordered asset digests, memory type, event time,
metadata, and any typed observation context. Adding the same logical input again returns the existing record instead of embedding and
storing a duplicate. Media bytes are independently de-duplicated by SHA-256, so several records may
reference one immutable asset.

`delete()` removes the authoritative record and queues removal from the search projection. Shared
media is retained until no record or active lease references it. API deletion is not secure erasure
from SQLite free pages, WAL files, snapshots, or backups; see the
[security model](../SECURITY.md#storage-deletion-and-backups).

Metadata is JSON application payload. It is useful for provenance and display, but it is not an
authorization rule, retrieval filter, tenant ID, or isolation boundary.

## A data directory is one memory domain

```text
data_dir/
├── state.sqlite3
├── assets/
├── .mindbridge.lock
└── zvec/
```

SQLite owns records, embeddings, metadata, and pending index work. `assets/` owns the immutable
media bytes. Zvec is a disposable search projection that can be rebuilt from SQLite without
re-embedding stored content.

Exactly one live `Memory` owns a directory. Use separate directories for applications, accounts,
benchmark cases, or any other domains that must not share data. Filesystem permissions and the
deployment boundary provide access control.

See [architecture](architecture.md) for transaction order, index recovery, and concurrency.

## Models are capabilities

Every memory needs an `EmbeddingBackend`. Grounded answers, transcription, speech identity, face
identity, visual description, and typed formation are optional capabilities supplied by
`GenerationBackend`, `TranscriptionBackend` or `SpeechBackend`, `FaceBackend`,
`VisionDescriptionBackend`, and `FormationBackend`.

Formation is the one capability that runs after a write. An explicitly configured
`FormationBackend` may propose a finer `MemoryKind` once the source observation has committed; it
never rewrites the memory type the caller supplied, and omitting it keeps ordinary add behavior.

`Memory.from_config()` constructs the small set of bundled adapters. `Memory(...)` accepts
already-constructed protocol implementations for application-specific models and SDK clients.
Routing follows each adapter's declared modalities; unsupported media fails explicitly instead of
being silently discarded.

`Memory` closes the backend objects it receives. A directly supplied `OpenAIModels` adapter leaves
its caller-owned OpenAI client open; the caller still owns that client. Declarative composition
creates and closes its own client.

See [configuration](configuration.md) for both composition paths and their ownership rules.

## Retrieval may return no evidence

`search()` returns `tuple[SearchHit, ...]`. The empty tuple is expected when no candidate clears the
configured relevance threshold. With `limit=1`, MindBridge may also withhold an unresolved top-two
tie. Scores rank hits within one request; they are not global probabilities.

`ask()` runs the same evidence retrieval before calling a configured generation backend. Its
`AnswerResult` contains the answer, the canonical hits reported as used, and explicit abstention
state. Calling `ask()` without an answerer is a configuration error.

Memory roles, event time, temporal retrieval, reinforcement, and decay are covered in
[memory types, time, and decay](memory-types-time-and-decay.md). Completed stream ingestion and
speculative query snapshots are covered in
[omni streaming and interaction memory](omni-streaming-and-interaction-memory.md).
