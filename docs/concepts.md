# Core concepts

MindBridge is an embedded memory library. The host application supplies content and model
capabilities; one local `Memory` owns persistence, retrieval, and grounded-answer orchestration.
This page gives the mental model. [Architecture](architecture.md) owns implementation invariants,
and the [Python SDK](api/python-sdk.md) owns exact signatures.

## From input to evidence

```text
content -> model preparation -> SQLite commit -> Zvec projection
query   -> Zvec candidates -> SQLite hydration -> ranked evidence
```

**Contract:** SQLite is authoritative for records, embeddings, metadata, and pending index work.
Zvec is a derived search projection. Search results are hydrated from SQLite, and a missing Zvec
index can be rebuilt from stored embeddings without re-embedding content.

**Guidance:** Treat `Memory` as one embedded runtime boundary. Do not build application state by
reading SQLite, `assets/`, or Zvec directly.

## Content becomes a record

Python accepts one content atom or an ordered sequence of atoms:

| Atom | Use |
| --- | --- |
| `str` | Application text |
| `pathlib.Path` | A local media file copied into the asset store |
| `Blob` | In-memory media bytes with an explicit MIME type |
| `AssetRef` | Media already stored in the same data directory |

`add`, `search`, and `ask` share this `ContentInput` shape. A URL-shaped `str` remains text;
MindBridge never downloads it. Fetch remote media in the application, then pass a `Blob` or local
`Path`. REST and MCP accept neither remote URLs nor server filesystem paths.

Each record has one modality. Text without media is `text`; text plus one media family keeps that
media modality; multiple media families produce `omni`.

`Memory.add()` returns a `MemoryRecord`. Its stable ID covers ordered canonical content, media
digests, metadata, event time, memory type, and optional observation context. Repeating the same
logical add returns the existing record without another model call. Media bytes are independently
de-duplicated by SHA-256, so records can share one immutable asset.

**Contract:** Metadata is JSON application payload. MindBridge stores and returns it but does not
interpret it as a retrieval filter, tenant identifier, authorization rule, or isolation boundary.

## A data directory is one memory domain

```text
data_dir/
├── state.sqlite3
├── assets/
├── .mindbridge.lock
└── zvec/
```

| Path | Role |
| --- | --- |
| `state.sqlite3` | Authoritative records, embeddings, metadata, and durable index outbox |
| `assets/` | Authoritative immutable media bytes |
| `zvec/` | Rebuildable search projection |
| `.mindbridge.lock` | Operating-system lock target; its presence does not imply a live owner |

**Contract:** Exactly one live `Memory` may own a physical directory. Different live owners must
use different directories.

**Guidance:** Allocate separate directories for applications, accounts, benchmark cases, or any
other data domains that must not share records. Use filesystem permissions and the deployment
boundary for access control. See [deployment](deployment.md) and [operations](operations.md) for
process, backup, and recovery procedures.

## Models are capabilities

Every memory requires an `EmbeddingBackend`. Other capabilities are optional:

| Capability | Enables |
| --- | --- |
| `GenerationBackend` | Grounded `ask()` answers |
| `TranscriptionBackend` or `SpeechBackend` | Audio/video text fallback and optional speech analysis |
| `FaceBackend` | Local face observations and identity evidence |
| `VisionDescriptionBackend` | Final-frame text fallback for `AsyncVisionStream` |
| `FormationBackend` | Typed records derived from committed observations |

Routing follows each backend's declared atomic modalities. Unsupported media fails explicitly
instead of being discarded.

Use `Memory.from_config()` for bundled adapters, `Memory.from_plugins()` for an explicit capability
bundle, or `Memory(...)` for direct injection. `Memory` closes backend objects it receives; an
adapter may leave its caller-supplied SDK client open. [Configuration](configuration.md) explains
the three paths and their ownership rules.

## Retrieval can return no evidence

`search()` returns `tuple[SearchHit, ...]`. An empty tuple is normal when no candidate clears
`minimum_relevance`. With `limit=1`, MindBridge may also withhold an unresolved top-two tie. A score
ranks hits against one query and does not move with the `limit` you ask for; it is not a global
probability.

`ask()` uses the same retrieval path before calling a configured generation backend. Its
`AnswerResult` reports the answer, the canonical hits used, and explicit abstention state. Calling
`ask()` without an answerer is a configuration error.

**Guidance:** Handle empty search results and answer abstention as ordinary outcomes. Use
`search_with_trace()` for one local ranking investigation; normal telemetry intentionally omits
candidate identifiers.

## Continue by task

- Choose adapters and policies in [configuration](configuration.md).
- Model roles, event time, typed assertions, spatial scope, and decay in
  [memory types, time, and decay](memory-types-time-and-decay.md).
- Ingest completed observations or continuous capture in
  [omni streaming and interaction memory](omni-streaming-and-interaction-memory.md).
- Use [architecture](architecture.md) for transaction, recovery, and concurrency details.
