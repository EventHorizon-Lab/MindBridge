# Core concepts

MindBridge deliberately has fewer moving parts than a memory service. Six concepts explain its
public behavior.

## Content and modality

Python uses one `ContentInput` contract for `add`, `search`, and `ask`: one `str`, `Path`, `URL`,
`Blob`, or `AssetRef`, or an ordered sequence of those atoms.

| Atom | Boundary |
| --- | --- |
| `str` | Normalized non-blank text |
| `Path` | Local regular media file copied into MindBridge |
| `URL` | HTTPS media fetched only from an allowed host |
| `Blob` | Immutable inline image, video, or audio bytes |
| `AssetRef` | Opaque reference to an asset in the same store |

REST and MCP translate `input_text`, `input_image`, and `input_file` parts into these values. They
do not accept local paths.

Every stored record has an explicit `Modality`: `text`, `image`, `video`, `audio`, or `omni`.
Text does not make a single-media input omni. No media is text; one media family keeps that family;
two or more media families are omni.

## Memory record and assets

`MemoryRecord` contains:

| Field | Meaning |
| --- | --- |
| `id` | Stable SHA-256 memory identity |
| `content` | Normalized text, plus audio transcript text persisted only by add-time embedding fallback |
| `modality` | Persisted modality; response layers do not recompute it |
| `assets` | Ordered, resolved `AssetRef` values |
| `created_at` | Time the record was first stored |
| `occurred_at` | Optional timezone-aware event time |
| `metadata` | Detached JSON application payload |

An asset is immutable media stored by SHA-256. Its public descriptor contains ID, modality, MIME
type, byte size, digest, and optional name. Python also receives the local resolved path so a model
backend can read the file. REST and MCP never expose that path.

A Python caller may pass an opaque `AssetRef(id=...)` to reuse existing bytes. MindBridge resolves
it through authoritative SQLite. The reference is rejected when it does not exist or its modality
hint contradicts the stored descriptor.

## Stable identity and idempotency

Before hashing, MindBridge canonicalizes text, ordered asset digests, event time, and JSON metadata.
Equivalent logical input has the same memory ID. Repeating an add returns the existing record
without storing the same bytes or embedding the record again.

The asset store separately de-duplicates identical media bytes by SHA-256. Multiple memories can
reference one immutable file. Deleting a memory removes a media file only after its final record
reference is gone. The optional name is digest-level CAS metadata, not per-reference metadata: the
first authoritative non-empty name is reused when identical bytes later arrive with another name.

## Data directory

A `data_dir` is the complete durability and isolation boundary for one `Memory`:

```text
data_dir/
├── state.sqlite3
├── assets/
├── .mindbridge.lock
└── zvec/
```

Exactly one live instance may own a directory. Ownership is enforced across threads, instances,
and processes. Independent directories can run concurrently without sharing records, assets,
embeddings, or indexes. On POSIX, opening a directory sets its top-level mode to `0700`, including
for an existing path.

There is no logical scoping layer inside a directory. If two applications or benchmark cases must
not see one another's memories, give them different directories and apply filesystem permissions.

## Authority and search projection

SQLite and the content-addressed asset files are authoritative. SQLite stores records, asset
descriptors and relationships, FP32 embeddings, the embedding/index recipe, and a durable outbox
of index work. It also stores face/voice identity profiles and cached observations. A successful
SQLite commit survives even if Zvec cannot be updated immediately.

Zvec is a rebuildable hybrid-search projection. Search gets ranked candidate IDs from Zvec and
hydrates complete records from SQLite. An index ID that no longer exists in SQLite is discarded.

Deleting `zvec/` while MindBridge is closed is recoverable. The next open checkpoints rebuild work
before creating the collection, then reconstructs it from stored embeddings without calling the
model endpoint for historical content.

## Capability-driven model routing

The default composition has four independent operations:

- Local Jina v5 Omni embedding for stored memories and search queries.
- Generation for grounded answers.
- Local Fun-ASR-Nano transcription, FSMN-VAD diarization, and CAM++ speaker encoding.
- Optional local InsightFace detection and ArcFace encoding for image/video identity.

Each operation declares the atomic modalities it accepts. MindBridge routes from capabilities,
not model names:

1. Natively supported media remains in the model input.
2. If embedding does not support audio, audio is transcribed and removed; its transcript is added
   to the text while supported image and video parts remain.
3. Generation uses the same fallback, allowing a visual-language model to receive ASR text plus
   retained visual evidence.
4. If the remaining input is unsupported, the operation fails with `ModelError`.

An audio-only input therefore becomes transcript-only when fallback is required. A video or image
is not silently discarded merely to reach a text model.

Stored embeddings belong to one adapter recipe, model, immutable revision, dimension, vector
space, and index recipe. Local adapters derive `space_id` from those values. MindBridge refuses to
open a directory with incompatible settings. Use a new directory and re-encode source content when
changing the embedding space. The store also persists `transcription_space`, which identifies the
ASR model and transcript-affecting preprocessing recipe. Reopening with a different value fails
before use so cached transcripts and add-time derived text cannot silently mix recipes.

FunASR's `SPK0` labels and InsightFace frame labels are local to one analysis. `Memory.speech`
matches normalized CAM++ centroids and `Memory.faces` matches normalized ArcFace embeddings to
stable IDs in one local identity registry. A first observation enrolls an identity; only later
matches carry an `identity_score`. `SpeakerSegment.identity_id` exposes the shared ID while the
existing `speaker_id` name remains supported.

Face and voice vectors occupy unrelated spaces and are never compared. After an application has
independent evidence that they represent the same person, `merge_identities` moves both profiles
and cached observations under one canonical ID. `register_identity` assigns its display name. Raw
biometric vectors remain inside the physical data directory.

## Retrieval and grounded answers

`search` uses Zvec dense plus full-text retrieval when the routed query contains text. A pure-media
query uses dense retrieval only. Each hydrated hit has a score from 0 through 1; scores rank results
within a request and are not stable global probabilities.

`ask` retrieves evidence first and routes those hits, including retained media, to generation.
Generation-time ASR can cache an asset transcript, but it does not rewrite an existing record's
`content`. `AnswerResult` contains the answer and source hits for display or grounding audits. The
built-in backend sends one binary content part per distinct asset in an answer request, even when
the question or several hits refer to it.

Operations on one instance may overlap remote model calls. MindBridge serializes the shorter
SQLite commit/outbox and Zvec access sections that must observe one coherent local state.

The current retrieval ceiling is intentionally explicit: one memory has one aggregate embedding.
MindBridge does not yet chunk content, create one vector per asset, or rerank candidates.
With built-in `data` transport, each embedding or generation call also has a 64 MiB aggregate raw
media ceiling before base64 encoding; co-located `file` transport or a streaming custom backend is
the large-video path.

## Metadata is not isolation

Metadata is useful for provenance, display, or application logic. It is not a server-side filter,
permission rule, or ownership boundary. Do not put secrets in metadata unless the complete data
directory is protected appropriately.
