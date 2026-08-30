# Core concepts

MindBridge deliberately has fewer moving parts than a memory service. The concepts below explain
its public behavior.

## Content and modality

Python uses one `ContentInput` contract for `add`, `search`, and `ask`: one `str`, `Path`, `Blob`,
or `AssetRef`, or an ordered sequence of those atoms.

| Atom | Boundary |
| --- | --- |
| `str` | Normalized non-blank text |
| `Path` | Local regular media file copied into MindBridge |
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
| `content` | Normalized text plus any transcript/identity evidence produced by configured add-time indexing |
| `modality` | Persisted modality; response layers do not recompute it |
| `memory_type` | Semantic, episodic, or procedural cognitive role |
| `assets` | Ordered, resolved `AssetRef` values |
| `created_at` | Time the record was first stored |
| `occurred_at` | Optional timezone-aware event time |
| `occurred_end` | Optional timezone-aware end of an interval; requires `occurred_at` and must be later than it |
| `metadata` | Detached JSON application payload |

An asset is immutable media stored by SHA-256. Its public descriptor contains ID, modality, MIME
type, byte size, digest, and optional name. Python also receives the local resolved path so a model
backend can read the file. REST and MCP never expose that path.

A Python caller may pass an opaque `AssetRef(id=...)` to reuse existing bytes. MindBridge resolves
it through authoritative SQLite. The reference is rejected when it does not exist or its modality
hint contradicts the stored descriptor.

## Memory roles

`MemoryType.SEMANTIC` stores facts and stable knowledge and is the default.
`MemoryType.EPISODIC` stores situated experiences; pair it with `occurred_at` when event time is
known. `MemoryType.PROCEDURAL` stores reusable instructions, examples, and routines. Procedural
records are grounding evidence and are never executed by MindBridge.

The role is explicit caller input, persisted in SQLite and Zvec, returned on records and hits, and
available as a search filter. MindBridge does not run another model call to infer or rewrite it.
See [memory types, temporal reasoning, and decay](memory-types-time-and-decay.md).

## Stable identity and idempotency

Before hashing, MindBridge canonicalizes text, ordered asset digests, memory role, event time, and
JSON metadata. Equivalent logical input has the same memory ID. Repeating an add returns the
existing record without storing the same bytes or embedding the record again.

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

SQLite and the content-addressed asset files are authoritative. SQLite stores records, memory
roles, event/access times, asset descriptors and relationships, FP32 embeddings, the
embedding/index recipe, and a durable outbox of index work. A successful SQLite commit survives
even if Zvec cannot be updated immediately.

Zvec is a rebuildable hybrid-search projection. Search gets ranked candidate IDs from Zvec and
hydrates complete records from SQLite. An index ID that no longer exists in SQLite is discarded.

Deleting `zvec/` while MindBridge is closed is recoverable. The next open checkpoints rebuild work
before creating the collection, then reconstructs it from stored embeddings without calling the
model endpoint for historical content.

## Capability-driven model routing

A typical embodied composition has four independent operations:

- Local Jina v5 Omni embedding for stored memories and search queries.
- Generation for grounded answers.
- Local Fun-ASR-Nano transcription, FSMN-VAD diarization, and CAM++ speaker encoding.
- Local YuNet face detection and SFace identity encoding through an optional `FaceBackend`.

Each operation declares the atomic modalities it accepts. MindBridge routes from capabilities,
not model names:

1. Natively supported media remains in the model input.
2. If embedding does not support audio, audio is transcribed and removed; its transcript is added
   to the text while supported image and video parts remain.
3. A configured `TranscriptionBackend` transcribes every asset whose modality it declares during
   `add`, whatever the embedder accepts. The transcript joins the record text, so the memory gains
   lexical and text-vector keys in addition to the native media vector; the media itself is still
   stored, embedded, and returned as evidence.
4. Grounded generation with a `SpeechBackend` resolves timed local speaker identities for supported
   audio/video and includes them as structured evidence. Unsupported audio is removed after that
   analysis, while supported visual evidence remains.
5. With `index_speech=True`, the same timed transcript, stable speaker IDs, and registered names
   are persisted in the record before embedding so exact-name and dialogue queries can retrieve it.
   Registering or renaming an identity refreshes already indexed recordings atomically.
6. `Memory.faces` resolves image/video faces. Grounded generation adds compact face identity
   evidence for retrieved visual memories without rewriting the immutable source record.
7. If the remaining input is unsupported, the operation fails with `ModelError`.

An audio-only input therefore becomes transcript-only when fallback is required. A video or image
is not silently discarded merely to reach a text model.

Stored embeddings belong to one adapter recipe, model, immutable revision, dimension, vector
space, and index recipe. Local adapters derive `embedding_space` from those values. MindBridge refuses to
open a directory with incompatible settings. Use a new directory and re-encode source content when
changing the embedding space. The store also persists `transcription_space`, which identifies the
ASR model and transcript-affecting preprocessing recipe. Reopening with a different value fails
before use so cached transcripts and add-time derived text cannot silently mix recipes.

FunASR's `SPK0` labels and face-detector labels are local to one asset. MindBridge stores neither as
the durable identity. It normalizes their embeddings, keeps at most 20 voice or 10 face exemplars
per identity, and scores a candidate by its maximum cosine similarity to any compatible exemplar.
This preserves different speaking styles, microphones, poses, and lighting that a single running
centroid would blur. When an exemplar set is full, the vector nearest its centroid is removed so
the retained set favors diversity.

Face and voice use independent model spaces, thresholds, and ambiguity margins but share stable
`identity_id` values. A single-face/single-speaker video may link two identities only when their
stored modalities are disjoint and their names do not conflict. Ambiguous scenes remain separate.
A merged source ID remains a durable alias for registration and lookup, while new observations use
the canonical ID. If speech indexing is enabled, changing that canonical ID and refreshing affected
memory text and embeddings commit in the same SQLite transaction.
A first observation enrolls an identity; only later matches carry an `identity_score`. Applications
may name either modality with `register_identity`; `register_speaker` remains the speech-specific
entry point. Biometric vectors and source media remain inside the physical data directory.

## Retrieval and grounded answers

`search` derives bounded aggregate and atomic query keys in one embedding batch; a single text atom
remains one vector. Duplicate query vectors are discarded and independent routes run concurrently.
Zvec returns one dense candidate per parent memory for every key; an ordinary-query fallback fills
groups when native best-effort grouping is incomplete. Relative BM25 uses an English
case/accent/stemming pipeline or Jieba for Han text. SQLite drops stale IDs and collapses all
derived keys to their authoritative parent memory before confidence-preserving fusion. Type and
range-optimized time filters are pushed into Zvec and rechecked after hydration. A temporal query retrieves both
in-range and global candidates, then applies a smooth proximity factor instead of a hard time gate.
Event intervals match by overlap, not only by their start. Optional decay is another soft factor.
Weak dense evidence and unresolved top-two ties are rejected unless qualified lexical or temporal
evidence anchors the winner. Each hydrated hit has a score from 0 through 1; scores rank results
within a request and are not stable global probabilities.

`ask` retrieves evidence first and routes those hits, including retained media, to generation. A
configured `SpeechBackend` caches complete identity analysis and adds each segment's timing, text,
stable ID, optional registered name, and match score to the model-only grounding context; returned
source hits remain unchanged. Generation-time analysis does not rewrite an existing record's
`content`. `AnswerResult` contains the canonical source hits that the generation backend reports it
used, filtering any unknown IDs, for display or grounding audits. The built-in backend sends one
binary content part per distinct asset in an answer request, even when the question or several hits
refer to it. It reserves the 64 MiB encoded-media budget for question assets, then admits ranked hit
media until full; an overflow hit remains as text evidence when it has text.

Operations on one instance may overlap remote model calls and Zvec queries. MindBridge serializes
SQLite commit/outbox sections and final record hydration with asset leasing; collection replacement
remains exclusive while ordinary Zvec queries can overlap.

Long text receives bounded overlapping retrieval keys while remaining one immutable returned
record. MindBridge does not segment long media, generate model-authored semantic keys, or use a
learned reranker. Applications should still ingest natural turns or bounded media clips. The
OpenAI adapter inlines at most 64 MiB of base64-encoded media per model call, which is roughly
48 MiB of files on disk; a provider-specific upload adapter is the large-video path.

## Metadata is not isolation

Metadata is useful for provenance, display, or application logic. It is not a server-side filter,
permission rule, or ownership boundary. Do not put secrets in metadata unless the complete data
directory is protected appropriately.
