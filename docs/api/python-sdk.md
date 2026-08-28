# Python API

The supported Python surface is exported from `mindbridge`:

```python
from mindbridge import (
    AnswerResult,
    AssetRef,
    AsyncMemory,
    Blob,
    Config,
    ContentInput,
    DEFAULT_FUNASR_MODEL_ID,
    DEFAULT_FUNASR_RECIPE,
    EmbeddingBackend,
    EmbedTask,
    FunASRRecipe,
    FunASRTranscriber,
    IndexUnavailableError,
    JinaOmniEmbedder,
    Memory,
    MemoryNotFoundError,
    MemoryRecord,
    MemoryType,
    MindBridgeError,
    Modality,
    ModelBackend,
    ModelCapabilities,
    ModelError,
    ModelInput,
    OpenAIHTTP,
    Page,
    SearchHit,
    SentenceTransformersEmbedder,
    SpeakerEmbedding,
    SpeakerNotFoundError,
    SpeakerSegment,
    SpeechAnalysis,
    SpeechBackend,
    SpeechTurn,
    StorageError,
    URL,
    ValidationError,
)
```

## Content values

`ContentInput` is one content atom or an ordered sequence of atoms:

```python
ContentAtom = str | pathlib.Path | URL | Blob | AssetRef
ContentInput = ContentAtom | Sequence[ContentAtom]
```

Strings are text, never filesystem paths. Use `Path` explicitly for a local file.
An ordered Python input may contain at most 128 atoms. REST and MCP use the tighter wire limit of
16 parts.

`Path` must identify a readable regular file. MindBridge infers image/video/audio MIME only from a
common deterministic suffix; use `Blob` when a suffix is unavailable or ambiguous.

### `URL`

```python
URL(
    value: str,
    media_type: str | None = None,
    name: str | None = None,
)
```

Only HTTPS without credentials or fragments is accepted. The hostname must appear in
`Config.allowed_url_hosts`. A concrete media type or family hint such as `image/*` is checked
against the downloaded `Content-Type`. When omitted, MindBridge only infers common deterministic
filename suffixes; otherwise it raises `ValidationError`. MindBridge resolves the hostname, rejects
any non-public result, pins the connection to a verified public IP while preserving TLS SNI and the
HTTP host, and repeats that validation for every redirect. A concrete MIME hint must match exactly;
a family hint must match the corresponding image, video, or audio family.

### `Blob`

```python
Blob(data: bytes, media_type: str, name: str | None = None)
```

Bytes must be non-empty and `media_type` must be a concrete image, video, or audio MIME type.

### `AssetRef`

```python
AssetRef(id: str, modality: Modality | None = None)
```

This short form reuses an asset already stored in the same data directory. MindBridge resolves
the complete descriptor from SQLite and verifies an optional modality hint. Records and hits
return resolved references with `media_type`, `size_bytes`, `sha256`, `name`, and local `path`.

An asset name belongs to the content digest, not to an individual memory-to-asset relationship.
When the same bytes arrive under different names, the first authoritative non-empty CAS name is
returned for every reference. Put a per-memory label in text or metadata instead.

## `Memory`

```python
Memory(
    data_dir: str | pathlib.Path = ".mindbridge",
    config: Config | None = None,
    *,
    models: ModelBackend | None = None,
    embedder: EmbeddingBackend | None = None,
    transcriber: SpeechBackend | None = None,
    speaker_similarity: float = 0.78,
    speaker_margin: float = 0.05,
)
```

Construction acquires exclusive ownership of `data_dir`, opens SQLite, the asset store, and Zvec,
validates durable compatibility metadata, and replays pending index work. By default, embedding is
pinned Jina v5 Omni, speech is local Fun-ASR-Nano, and generation uses OpenAI-compatible HTTP.
Passing `embedder` or `transcriber` replaces only that operation. Passing `models` without either
narrower backend uses that combined backend for all three operations. The two speaker thresholds
calibrate CAM++ cosine matching; ambiguous matches enroll a new anonymous identity. `Memory` owns
and closes each distinct supplied backend once.

### `add`

```python
memory.add(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryRecord
```

Adds text, media, or ordered combinations. Text atoms are normalized and limited to 65,536
characters. `occurred_at` must be timezone-aware. `memory_type` is semantic, episodic, or
procedural; it is persisted, returned, and included in stable identity. Metadata must be
JSON-compatible and its canonical UTF-8 form may not exceed 262,144 bytes.

Local paths, inline bytes, and allowed HTTPS sources are copied into immutable content-addressed
storage before the record is committed. One aggregate model input produces one embedding. If the
stable ID already exists, `add` returns the stored record without retranscribing or embedding it.
Path and URL inputs still have to be read to establish their content digest; pass an existing
`AssetRef` when the bytes are already stored.

If SQLite commits but Zvec flush fails, `add` raises `IndexUnavailableError` while leaving the
record, assets, embedding, and pending operation durable. Retrying the same input is safe.

### `add_many`

```python
memory.add_many(
    contents: Sequence[ContentInput],
    *,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> tuple[MemoryRecord, ...]
```

Adds a batch with no per-item event time or metadata. A single ordered multimodal item must be
nested inside the outer batch:

```python
from pathlib import Path

records = memory.add_many(
    [
        "plain text",
        ["caption", Path("image.png")],
    ]
)
```

Empty input returns an empty tuple. One `memory_type` applies to the complete batch. Output matches
input order and length; duplicates may return the same ID. New items are routed, embedded in one
model batch, and stored atomically.

### `search`

```python
memory.search(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
) -> tuple[SearchHit, ...]
```

Runs dense plus full-text hybrid retrieval when the routed query contains text, and dense-only
retrieval for a pure-media routed query. `limit` is from 1 through 100. `memory_type` is an optional
hard role filter. `reference_at` must be timezone-aware and resolves ISO/common English or Chinese
relative calendar expressions; current UTC is the default. Multimodal queries follow the same
capability routing as stored memories. Type and event-time constraints are pushed into Zvec and
rechecked while results are hydrated from SQLite; stale or contradictory derived-index IDs are
omitted. Temporal intent and enabled decay rerank a bounded candidate pool.

### `ask`

```python
memory.ask(
    question: ContentInput,
    *,
    limit: int = 5,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
) -> AnswerResult
```

Retrieves up to `limit` memories and asks the configured generation backend to answer only from
those hits. Role filtering and temporal reference semantics match `search`; a resolved reference
time is also sent to generation. Question and evidence assets are preserved when supported. Audio
fallback contributes ASR text while supported image/video evidence remains available to a VLM.
The built-in backend serializes each distinct question/evidence asset once in the outbound answer
request, even if several hits refer to the same digest.

### `get`

```python
memory.get(memory_id: str) -> MemoryRecord
```

Returns one record and resolved assets. A missing ID raises `MemoryNotFoundError`.

### `speech`

```python
memory.speech(memory_id: str) -> tuple[SpeakerSegment, ...]
```

Lazily analyzes every audio/video asset in the memory with the configured `SpeechBackend` and
caches the result. Each segment contains `asset_id`, `start_ms`, `end_ms`, `text`, a stable local
`speaker_id`, optional `speaker_name`, and `identity_score`. The score is `None` when the speaker is
first enrolled and is a cosine score on later recognition. `Memory.speech` never returns raw CAM++
voiceprints. Repeated calls for the same asset and speech space do not run inference again. A
combined backend that only implements plain `transcribe` can provide ASR fallback but raises
`ModelError` here because it has no speaker evidence.

### `register_speaker`

```python
memory.register_speaker(speaker_id: str, name: str) -> None
```

Assigns or replaces the human-readable name for an enrolled local speaker. Names are returned on
both new and cached `SpeakerSegment` values. Unknown IDs raise `SpeakerNotFoundError`; names must be
non-empty, printable, and at most 255 characters.

### `list`

```python
memory.list(*, limit: int = 100, cursor: str | None = None) -> Page
```

Lists newest records first. Pass `Page.next_cursor` unchanged to retrieve the next stable keyset
page. A malformed cursor raises `ValidationError`.

### `delete`

```python
memory.delete(memory_id: str) -> bool
```

Deletes the SQLite record, embedding, asset relationships, and derived index entry. It removes a
CAS file only when no other memory references it. Returns `True` when the record existed and
`False` for a repeated delete.

### `reindex`

```python
memory.reindex() -> int
```

Rebuilds Zvec from authoritative SQLite embeddings and returns the number of indexed memories. It
does not call the embedding model or reread source URLs/files.

### `optimize`

```python
memory.optimize() -> None
```

Optimizes and flushes the Zvec collection. This can be I/O intensive.

### `close`

```python
memory.close() -> None
```

Closes model, Zvec, SQLite, and lock resources. Repeated calls are harmless. Using a closed
or post-fork instance raises `StorageError`.

## Modality routing

Input modality is derived from media families:

| Input | Modality |
| --- | --- |
| No media assets | `Modality.TEXT` |
| Only image-family media, with or without text | `Modality.IMAGE` |
| Only video-family media, with or without text | `Modality.VIDEO` |
| Only audio-family media, with or without text | `Modality.AUDIO` |
| Two or more media families | `Modality.OMNI` |

The configured `ModelCapabilities` drives each operation. Unsupported audio can fall back through
`transcribe`: audio is removed from that model call, transcript text is added, and supported image
or video atoms remain. MindBridge never guesses from model names or silently drops unsupported
visual media.

## `AsyncMemory`

`AsyncMemory` has the same constructor, operations, defaults, and return types. Every operation is
awaitable, and lifecycle uses `async with`:

```python
from pathlib import Path

from mindbridge import AsyncMemory


async def remember() -> None:
    async with AsyncMemory("./data/async") as memory:
        record = await memory.add(["Handoff recording", Path("handoff.wav")])
        hits = await memory.search("When is handoff?")
        await memory.delete(record.id)
        print(hits)
```

It delegates to one synchronous core with `asyncio.to_thread`; it is not a separate storage mode.
Do not open synchronous and asynchronous owners for the same directory. Concurrent calls on one
instance may overlap remote model work. SQLite commit/outbox and Zvec access use short serialized
critical sections so local state and derived search remain consistent.

## Return values

All return values are frozen, slotted dataclasses. Metadata is copied into a detached mapping.

### `MemoryRecord`

```python
MemoryRecord(
    id: str,
    content: str,
    created_at: datetime,
    occurred_at: datetime | None,
    metadata: Mapping[str, object],
    assets: tuple[AssetRef, ...],
    modality: Modality,
    memory_type: MemoryType,
)
```

`content` is the normalized textual component and may be empty for native media-only input. It
includes transcript text only when add-time embedding fallback required ASR. A later
generation-time fallback may cache an asset transcript but does not rewrite the existing record.
`modality` and `memory_type` are persisted and returned directly.

### `SearchHit`

`SearchHit` has the record fields plus `score: float`, constrained to the inclusive range 0
through 1. A score can include temporal and decay factors and is not a stable probability.

### `AnswerResult`

```python
AnswerResult(answer: str, hits: tuple[SearchHit, ...])
```

### `Page`

```python
Page(items: tuple[MemoryRecord, ...], next_cursor: str | None)
```

## Embedding backends

`EmbeddingBackend` is the narrow public embedding seam:

```python
class EmbeddingBackend(Protocol):
    capabilities: frozenset[Modality]
    model_id: str
    space_id: str
    dimension: int

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...
```

Calls may overlap across threads. An implementation must preserve input order, return one finite
vector of the declared dimension per input, and remain thread-safe until `close()`.

`JinaOmniEmbedder()` declares the fixed default model and immutable revision without loading its
weights. The first non-empty embedding call loads them; use `load()` when startup must fail eagerly:

```python
JinaOmniEmbedder.load(
    *,
    dimension: int = 1024,
    device: str | None = None,
    batch_size: int = 32,
    batch_wait_ms: float = 2.0,
) -> JinaOmniEmbedder
```

It declares text, image, video, and audio. The pinned model accepts dimensions `32`, `64`, `128`,
`256`, `512`, and `1024`; another value fails before inference. Jina's legacy tuple conversion and
remote-code isolation exist only in this adapter. Concurrent calls with the same retrieval task are
coalesced up to `batch_size`; `batch_wait_ms` is the maximum collection window and may be set to
zero for latency-first workloads.

`SentenceTransformersEmbedder.load` loads any standard model at an immutable commit:

```python
SentenceTransformersEmbedder.load(
    model_id: str,
    *,
    revision: str,
    dimension: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> SentenceTransformersEmbedder
```

It discovers atomic capabilities with `supports()`, uses a dict for unique multimodal parts and a
standard message for repeated parts, and calls `encode_query` or `encode_document` once per batch.
It never applies Jina model arguments or tuple serialization. For Qwen3-VL-Embedding-2B, use
revision `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`; the model advertises text, image, video, and
message, but not audio.

Both adapters derive `space_id`; it cannot be overridden. The recipe includes adapter type, model,
revision, effective dimension, normalization, retrieval-side methods, and serialization. A change
therefore fails against an existing directory instead of mixing vectors. Re-encode source content
into a new directory. `reindex()` does not re-embed.

## FunASR speech backend

`FunASRTranscriber` defaults to the portable FunASR `AutoModel` engine. Its
`DEFAULT_FUNASR_RECIPE` composes pinned Fun-ASR-Nano, FSMN-VAD, and CAM++ revisions, with a
30-second VAD segment ceiling and `trust_remote_code=False`. Construct a different `FunASRRecipe`
to swap that composition.

For high-throughput CUDA inference, pin the vLLM build matching the machine's NVIDIA driver, then
install `mindbridge[local,vllm]` and select it explicitly:

```python
speech = FunASRTranscriber(
    engine="vllm",
    device="cuda",
    gpu_memory_utilization=0.5,
    tensor_parallel_size=1,
    vllm_dtype="bf16",
)
memory = Memory("./data/vllm", transcriber=speech)
```

The vLLM route batches all FSMN-VAD spans in one decode and then runs CAM++ clustering, so timed
speaker identity is preserved. `max_model_len`, `max_new_tokens`, and `vllm_dtype` tune decoding
and hardware compatibility. Engine and transcript-affecting settings are part of `space_id`; use a
new data directory when they change. `device="auto"` prefers CUDA and otherwise uses CPU, but
`engine="vllm"` requires CUDA.
Use the [FunASR vLLM installation guide](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md)
to select a vLLM/Torch/CUDA combination; the optional extra deliberately does not choose a driver
build for you.

`SpeechBackend` is the narrower custom seam for timed turns and speaker centroids. Its `space_id`
must change whenever ASR, VAD, diarization, speaker encoding, or preprocessing changes.

## Custom model backend

`ModelBackend` is the complete public model seam:

```python
class ModelBackend(Protocol):
    capabilities: ModelCapabilities
    embedding_model: str
    embedding_space: str
    transcription_space: str
    embedding_dimension: int

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> AnswerResult: ...

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...
```

Pass an implementation with `Memory(..., models=backend)`. Compatible vLLM/OpenAI deployments can
use `OpenAIHTTP(Config(...))` as that combined backend. Calls may overlap across threads, so a
custom backend must be thread-safe until `close()`. See [configuration](../configuration.md).

`OpenAIHTTP` also accepts optional `generation_seed` and `generation_temperature` keyword
arguments. They are omitted from normal requests unless set; reproducible evaluation sets both.

## Exceptions

Catch `MindBridgeError` for every supported operational failure, or a specific subclass:

| Exception | Meaning |
| --- | --- |
| `ValidationError` | Content, asset, metadata, cursor, time, or limit is invalid |
| `MemoryNotFoundError` | `get` could not find the requested memory ID |
| `ModelError` | Capability routing, embedding, generation, or transcription failed |
| `StorageError` | Directory, lock, CAS, SQLite, schema, or durable state failed |
| `IndexUnavailableError` | Zvec could not open, mutate, flush, or search |

`IndexUnavailableError` is also a `StorageError`. Internal HTTPX, SQLite, filesystem, Pydantic, and
Zvec exceptions do not cross the public boundary.

## Current limits

There is no public update method, server-side metadata filter, logical account scope, automatic
classification/consolidation, executable procedure runtime, automatic chunking, per-asset
embedding, or learned reranker. One memory produces one aggregate embedding. Allocate a separate
`data_dir` for each independent memory domain. See
[memory types, temporal reasoning, and decay](../memory-types-time-and-decay.md) for the supported
deterministic layer.

For `Config(media_transport="data")`, the built-in backend limits aggregate raw media to 64 MiB
per embedding or generation call, before base64 expansion. Use `file` only with a trusted
co-located backend that can read the local asset paths, or provide a streaming/file-upload custom
backend for large video. Provider limits may be lower.

The built-in backend limits serialized answer text evidence to 4 MiB. The outbound generation
request includes the question plus each hit's content, `memory_type`, `occurred_at`, `created_at`,
metadata, and asset references. Choose model endpoints with an appropriate data-retention policy.
