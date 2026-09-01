# Python API

Supported public imports come from `mindbridge`.

`Memory` is MindBridge's canonical execution plane. `AsyncMemory`, REST, MCP, and the required
product CLI must dispatch to these same domain operations rather than implement parallel routing,
storage, defaults, or errors.

## Content values

```python
from mindbridge import AssetRef, Blob, ContentAtom, ContentInput, StreamInput
```

```python
ContentAtom = str | pathlib.Path | Blob | AssetRef
ContentInput = ContentAtom | Sequence[ContentAtom]
```

`Blob(data, media_type, name=None)` requires non-empty bytes and a concrete image, video, or audio
MIME type. `AssetRef(id, ...)` can be opaque at input boundaries; returned references contain the
authoritative modality, MIME type, size, digest, name, and local path.

An ordered sequence combines text and media into one memory. A plain `str` always means text, even
when it resembles a path or URL. Remote URL fetching is outside the SDK.

### Input limits

These bounds are enforced on the Python surface and raise `ValidationError` before any model call:

| Bound | Value | Message |
| --- | --- | --- |
| Parts in one `ContentInput` sequence | 128 | `content must not exceed 128 parts` |
| Characters in one text value | 65,536 after NFC normalization | `... must not exceed 65536 characters` |
| `limit` for `search`, `search_with_trace`, `ask`, and `list` | 1–100 inclusive | `limit must be between 1 and 100` |

The transports are stricter than the SDK, not equal to it: REST and MCP cap one request at 16 parts.
An agent that composes near the Python bound must therefore expect a smaller ceiling over a
transport; see the [REST](rest.md) and [MCP](mcp.md) references for the per-transport limits.

## Memory

```python
Memory(
    data_dir=".mindbridge",
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    index_speech: bool = False,
    index_quantization: IndexQuantization = IndexQuantization.NONE,
    minimum_relevance: float = 0.55,
    ambiguity_margin: float = 0.01,
    decay_half_life_days: float | None = None,
    speaker_similarity: float = 0.78,
    speaker_margin: float = 0.05,
    face_similarity: float = 0.363,
    face_margin: float = 0.05,
    tracer: opentelemetry.trace.Tracer | None = None,
)
```

`embedder` is required. `Memory` validates adapter capabilities and durable space identity before
opening Zvec. It closes supplied adapters when the memory closes; provider clients owned by an
adapter may remain caller-owned, as documented by that adapter.

Bundled adapters can be selected without constructing runtime objects:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/example",
        "embedding": {"provider": "jina-omni"},
        "speech": {"provider": "funasr"},
        "settings": {"index_speech": True},
    }
) as memory:
    memory.add("Remember this")
```

`Memory.from_config` accepts a `MindBridgeConfig` or mapping. It strictly validates bundled provider
fields, owns the adapters and SDK clients it constructs, and reports invalid fields before opening
storage. `AsyncMemory.from_config` accepts the same input. See
[configuration and composition](../configuration.md) for providers and fields.

`resolve_memory_config(config)` is the public lower-level boundary for constructing adapters
separately from storage. It returns an owned `MemoryComposition`; call `close()` unless its plugins
are transferred to one `Memory`.

Direct `Memory(...)` construction remains the stable plugin API. The compatibility
`MemoryPlugins`/`Memory.from_plugins` bundle is also supported for applications that already group
runtime objects separately from `MemoryConfig` local policy. Every entry point performs the same
capability validation and uses the same storage, routing, lifecycle, and failure behavior.

`tracer` optionally selects a non-global OpenTelemetry provider. With the default `None`,
MindBridge uses the standard global tracer. See
[performance and token observability](../observability.md) for span names, TTFT, usage attributes,
and privacy behavior.

`index_speech=True` requires a speech-capable backend to analyze supported audio/video during
`add`. Its transcript, stable speaker IDs, and names already registered at add time become stored,
retrievable text. The default keeps speech analysis lazy.

A `TranscriptionBackend` needs no flag. `add` transcribes every asset whose modality that backend
declares and stores the transcript in the record, so a media memory has retrievable text next to
its native media vector; the media is never replaced. `SpeechBackend` analysis stays behind
`index_speech` because it also resolves speaker identity.

`index_quantization` controls only Zvec's rebuildable vector index. `NONE` is the default and
preserves maximum retrieval quality. `FP16` and rotated `INT8` reduce active index memory;
`RABITQ` uses HNSW-RaBitQ and requires x86_64 with AVX2 plus an embedding dimension from 64 through
4095. Quantization is lossy, so compare recall and latency before enabling it. Changing this value
rebuilds Zvec from authoritative FP32 embeddings in SQLite without calling the embedder.

`minimum_relevance` rejects weak dense evidence. `ambiguity_margin` rejects an unresolved top-two
tie only when `search()` or `ask()` is called with `limit=1`; a lexical or temporal anchor can clear
the tie. With a larger limit, `search` returns the qualified candidates and `ask` passes them to the
answerer. Both settings are calibrated `[0, 1]` values and may be set to `0` to disable that gate. A
candidate that matches the full-text index is scored at `0.6` confidence regardless of its vector
distance, so it clears the default `minimum_relevance` on the strength of the lexical match alone.

Use `Memory` as a context manager:

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/example", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add("Remember this")
```

### Add

```python
record = memory.add(
    content,
    occurred_at=None,
    occurred_end=None,
    metadata=None,
    memory_type=MemoryType.SEMANTIC,
)
records = memory.add_many(
    contents,
    occurred_at=(first_time, second_time),
    occurred_end=(first_end, second_end),
    metadata=({"source_id": "first"}, {"source_id": "second"}),
    memory_type=MemoryType.SEMANTIC,
)
```

Event times must be timezone-aware. `occurred_end`, when present, requires `occurred_at` and must be
later than it. Metadata must be a JSON-compatible mapping with non-empty string keys. For
`add_many`, the optional event-time and metadata sequences must contain one value per content; this
preserves per-record provenance without losing batched model/storage work. Duplicate inputs return
the same stable record in their original positions.

### Stream input

`add_stream` consumes an iterable lazily and commits each completed item before requesting the
next. A plain `ContentInput` uses the same defaults as `add`; wrap an item in `StreamInput` when a
clip needs its own event time, metadata, or memory role:

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindbridge import MemoryType, StreamInput

started = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)


def camera_clips():
    for sequence, path in enumerate(sorted(Path("./capture").glob("*.mp4"))):
        occurred_at = started + timedelta(seconds=30 * sequence)
        yield StreamInput(
            path,
            occurred_at=occurred_at,
            occurred_end=occurred_at + timedelta(seconds=30),
            metadata={"sequence": sequence},
            memory_type=MemoryType.EPISODIC,
        )


for record in memory.add_stream(camera_clips()):
    print(record.id)
```

Each yielded record is already durable and searchable. The stream is not one transaction: if a
later item fails, earlier records remain committed and the error's `subject` identifies its
`contents[N]` position. `AsyncMemory.add_stream` accepts an `AsyncIterable` and returns an async
iterator with the same item semantics. MindBridge consumes completed chunks; the application owns
camera or microphone capture and chooses chunk boundaries. Composite observations may combine
text, image, video, and audio atoms. See
[omni streaming and interaction memory](../omni-streaming-and-interaction-memory.md).

### Retrieve and answer

```python
hits = memory.search(
    query,
    limit=10,
    memory_type=None,
    reference_at=None,
    occurred_from=None,
    occurred_until=None,
)
result = memory.ask(
    question,
    limit=5,
    memory_type=None,
    reference_at=None,
)
```

`search` returns `tuple[SearchHit, ...]`. `ask` retrieves first and passes only those hits to the
configured answerer. It raises `ModelError` when no answerer is configured.

`reference_at` controls relative-date interpretation and decay reranking. It must be
timezone-aware. When omitted, the current UTC time is used unless the query declares a valid
English reference date such as `Today is May 2, 2024`; an explicit `reference_at` always wins.

`occurred_from` and `occurred_until` are optional timezone-aware hard filters on event time. A
memory matches when its `[occurred_at, occurred_end)` interval overlaps the half-open query
interval; an instant event has a one-microsecond extent. Either bound may be omitted. Supplying any
bound excludes memories without `occurred_at`, and two bounds require
`occurred_until > occurred_from`. These filters are independent of the soft temporal preference
inferred from query text.

Composite records are indexed with an aggregate vector and de-duplicated vectors for each text or
media atom. Text longer than 2,048 characters also receives overlapping contextual retrieval keys;
the complete record remains the returned evidence. Queries batch their complete aggregate with
bounded focused aggregate and atomic keys derived from the first text atom and query media; later
text atoms do not become independent dense routes. The focused text also supplies the lexical
query. Dense and lexical candidates hydrate and collapse aggregate or atomic document keys to their
authoritative parent before reranking. English BM25 uses
case folding, accent folding, and stemming; queries containing Han characters use Jieba. Weak or
missing evidence can therefore return `()`. With `limit=1`, an unresolved top-two tie can also
empty `search` or leave `ask` with no hits; larger limits preserve those qualified candidates.

### Trace one search

```python
result = memory.search_with_trace(
    query,
    limit=10,
    memory_type=None,
    reference_at=None,
    occurred_from=None,
    occurred_until=None,
)
```

`result.hits` is exactly the value the same `search` call returns. `result.trace.candidates`
explains the bounded candidate set actually considered: parent `memory_id`, contributing
`index_ids`, dense relevance and confidence, effective lexical relevance, lexical rerank bonus,
gate confidence, reinforcement, temporal and retention factors, final score, rank, and
`rejected_by`. For a ranked candidate,
`base_relevance = min(1, max(dense_relevance, lexical_relevance) + lexical_rerank_bonus)`;
`gate_confidence` is the value compared with `minimum_relevance`. Rejection values are
`stale_index`, `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`,
`ambiguity`, and `limit`. A stale index candidate has `memory_id=None`.

The trace never contains the query, memory content, metadata, media, vectors, paths, or model
output. It is returned only to the caller and is not persisted or emitted through OpenTelemetry.
`candidate_limit` is the final bounded retrieval width; `exhaustive` means every route returned
fewer candidates than that width, not that the complete corpus was scanned. This diagnostic is
available from Python and the local `search-with-trace` CLI command; REST and MCP do not expose it
in this release.

### Feedback

```python
updated = memory.reinforce((hit.id,))
```

Call `reinforce` only after explicit positive feedback. Retrieval itself never changes access
strength. Confirmations provide a small bounded ranking boost even without decay and also slow
decay when it is enabled. The return value is the number of existing, distinct memories updated.

`reinforce` has no REST route and no MCP tool in this release, so the feedback loop is reachable
only from Python. An application whose retrieval runs over a transport must call back into the
owning process to record a confirmation.

### Read, list, and delete

```python
record = memory.get(memory_id)
page = memory.list(limit=100, cursor=None)
deleted = memory.delete(memory_id)
```

`get` raises `MemoryNotFoundError` for an unknown ID. `delete` is idempotent and returns whether a
record existed. Listing uses an opaque keyset cursor.

### Speech

```python
turns = memory.speech(memory_id)
memory.register_speaker(speaker_id, "Ada")
```

`speech` lazily analyzes stored audio or video through a configured `SpeechBackend`. Returned
`SpeakerSegment` values contain time bounds, transcript text, opaque local speaker ID, optional
registered name, and optional identity score. Grounded `ask` calls reuse this cache and pass the
complete timed identity evidence to the answerer without changing the returned source hits.
With `index_speech=True`, registering or renaming a speaker also re-embeds every existing memory
that contains that identity, so name queries work for recordings captured before registration.

### Face and multimodal identity

```python
observations = memory.faces(memory_id)
memory.register_identity(observations[0].identity_id, "Ada")
```

`faces` lazily analyzes stored images or videos through the configured `FaceBackend`. Each
`FaceObservation` contains a normalized bounding box, optional video timestamp, stable local
identity ID, optional registered name, and optional match score. Face and voice observations share
the same identity namespace. For a video with exactly one resolved face identity and one resolved
speaker identity, MindBridge links them only when their stored modality sets do not conflict; a
multi-face, multi-speaker, or conflicting-name scene remains unlinked.
The retired ID remains a durable alias accepted by `register_identity` and `register_speaker`.
When `index_speech=True`, a merge that changes the speaker's canonical ID atomically refreshes the
affected record text, FP32 embeddings, and durable index outbox.

Both modalities use bounded exemplar sets and max-over-exemplar cosine matching. Voice retains at
most 20 exemplars per identity and face retains at most 10; when full, the exemplar nearest the set
centroid is removed to preserve variation. The first observation enrolls an identity and therefore
has no match score. Threshold and top-two margin gates are configured independently per modality.

### Index maintenance

```python
count = memory.reindex()
memory.optimize()
```

`reindex` rebuilds the disposable Zvec projection from every stored embedding and returns the
number of memories rebuilt. It never calls the embedder. `optimize` compacts the current index.

## AsyncMemory

`AsyncMemory` takes the same constructor arguments and exposes the same methods. Finite operations
use `await`; `add_stream` consumes an `AsyncIterable` with `async for`.

```python
async with AsyncMemory(
    "./data/async",
    embedder=embedder,
    answerer=answerer,
    transcriber=transcriber,
) as memory:
    await memory.add("Remember this")
    hits = await memory.search("Remember")
    async for record in memory.add_stream(observations()):
        print(record.id)
```

It runs the embedded synchronous consistency core through `asyncio.to_thread`. It is not a
provider compatibility layer.

### Async omni prefetch

`AsyncOmniPrefetch` is a per-turn Python orchestration helper over `AsyncMemory.search`. Submit the
complete current multimodal snapshot whenever useful evidence changes, then confirm the final
snapshot:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(memory, limit=8)
prefetch.submit((partial_text, frame_blob, audio_blob))
prefetch.submit((newer_text, frame_blob, audio_blob))
result = await prefetch.finalize((final_text, frame_blob, audio_blob))
```

One search runs at a time and only the newest queued revision survives. `latest` returns the newest
completed `PrefetchResult` without waiting. `finalize` returns a result for the exact final value
and closes the helper; a failed matching speculation is retried as a new revision. Prefetch
snapshots accept immutable text, `Blob`, and `AssetRef` values, but reject mutable `Path` inputs.
`close()` abandons queued work and drains the already-running search without attempting ineffective
thread cancellation. The helper never persists input or reinforces hits.

## Return values

- `MemoryRecord`: stable ID, derived text, modality, memory type, assets, timestamps, metadata.
- `SearchHit`: the same visible memory fields plus a normalized score.
- `TracedSearchResult`: search hits plus an immutable `RetrievalTrace` of
  `RetrievalCandidateTrace` values and stable `RetrievalRejection` reasons.
- `AnswerResult`: grounded answer text, accepted hits, `abstained`, and a machine-readable
  `abstention_reason` (`no_evidence` or `insufficient_evidence`).
- `Page`: records and an optional next cursor.
- `PrefetchResult`: a positive submission revision and immutable search hits.
- `SpeakerSegment`: one timed speech segment and local identity fields.
- `StreamInput`: one completed `ContentInput` with per-item time, metadata, and memory type.

All are frozen, slotted dataclasses. Mappings are detached from caller input.

## Model protocols

The operation-specific contracts are runtime-checkable protocols.

```python
class EmbeddingBackend(Protocol):
    embedding_capabilities: frozenset[Modality]
    embedding_model: str
    embedding_space: str
    embedding_dimension: int

    def embed(
        self,
        inputs: Sequence[ModelInput],
        task: EmbedTask = EmbedTask.DOCUMENT,
    ) -> tuple[tuple[float, ...], ...]: ...

    def close(self) -> None: ...
```

```python
class GenerationBackend(Protocol):
    generation_capabilities: frozenset[Modality]

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
    ) -> AnswerResult: ...

    def close(self) -> None: ...
```

An answerer can additionally implement
`StreamingGenerationBackend.stream_answer() -> Iterator[str]`. `Memory.ask()` consumes the stream
into the same `AnswerResult` and records TTFT on the generation span; the stable
`GenerationBackend.answer()` contract remains available for non-streaming callers.

```python
class TranscriptionBackend(Protocol):
    transcription_capabilities: frozenset[Modality]
    transcription_model: str
    transcription_space: str

    def transcribe(self, assets: Sequence[AssetRef]) -> tuple[str, ...]: ...

    def close(self) -> None: ...
```

`SpeechBackend` has the same transcription identity properties as `TranscriptionBackend` and adds
`analyze()`:

```python
class SpeechBackend(Protocol):
    transcription_capabilities: frozenset[Modality]
    transcription_model: str
    transcription_space: str

    def analyze(self, assets: Sequence[AssetRef]) -> tuple[SpeechAnalysis, ...]: ...

    def close(self) -> None: ...
```

`analyze()` returns one `SpeechAnalysis` per asset, in the order the assets were supplied — not a
single value. Each `SpeechAnalysis` carries `turns` and `speakers`. An implementation that returns
one object per call, rather than a tuple, fails inside `Memory` at runtime.

Backends may implement more than one protocol. Calls can overlap, so adapters must be thread-safe
until `close()`.

## Built-in adapters

### Sentence Transformers

```python
embedder = SentenceTransformersEmbedder.load(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
)
```

`revision` is keyword-only and required. It must be a 40-character immutable commit hash; a branch
or tag name raises `ValidationError: revision must be an immutable 40-character commit hash`. The
immutable revision, dimension, normalization, query/document semantics, and input recipe form the
durable embedding space, which `Memory` records in `data_dir` and re-checks on open.

`dimension`, `device`, and `batch_size` are optional. `device=None` lets Sentence Transformers
choose; pass `"cuda"` or `"cpu"` to pin it:

```python
embedder = SentenceTransformersEmbedder.load(
    "sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    device="cuda",
    batch_size=8,
)
```

Capabilities come from the loaded model, so a text-only model yields a text-only store. See
[choosing an embedding backend](../quickstart.md#choose-an-embedding-backend) for how to resolve a
model's current commit hash.

`JinaOmniEmbedder` is a lazy pinned specialization:

```python
embedder = JinaOmniEmbedder(dimension=1024, device="cuda", batch_size=8)
```

Its pinned weights are licensed CC BY-NC 4.0 — non-commercial use only. That licence covers the
model, not MindBridge.

It calls the model's official Sentence Transformers retrieval methods and wraps application text
so URL- or path-shaped text cannot activate media autodetection.

### FunASR

```python
speech = FunASRTranscriber(device="auto")
```

The adapter delegates model, VAD, and optional speaker execution to `funasr.AutoModel`, then
validates and maps the result into MindBridge speech values. A `FunASRRecipe` with
`speaker_model=None` returns timed transcript turns without speaker embeddings.

### OpenCV face analysis

Install `mindbridge[face]`, obtain the ONNX weights from OpenCV Zoo's
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and
[SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) pages, and pass
their local paths explicitly:

```python
from mindbridge import OpenCVFaceAnalyzer

faces = OpenCVFaceAnalyzer(
    detector_model="./models/face_detection_yunet.onnx",
    recognizer_model="./models/face_recognition_sface.onnx",
)
```

The adapter samples videos with a bounded, configurable interval, runs locally through OpenCV, and
never downloads weights or exports source media. Model-file digests and analysis settings form its
durable spaces. The default face similarity `0.363` is SFace's published LFW cosine threshold;
applications should calibrate thresholds on representative, held-out data before relying on names.

### OpenAI SDK

```python
models = OpenAIModels(
    client=None,
    embedding_client=None,
    generation_client=None,
    transcription_client=None,
    embedding_model="text-embedding-3-small",
    embedding_space=None,
    embedding_dimension=1536,
    embedding_request_format="input",
    generation_model="gpt-5-mini",
    transcription_model="whisper-1",
    transcription_space=None,
    transcription_prompt=None,
    transcription_keywords=None,
    transcription_languages=None,
    embedding_capabilities=frozenset({Modality.TEXT}),
    generation_capabilities=frozenset({Modality.TEXT}),
    transcription_capabilities=frozenset({Modality.AUDIO}),
    generation_seed=None,
    generation_temperature=None,
    generation_max_tokens=None,
    generation_video_limit=8,
    generation_extra_body=None,
)
```

The common `client` is used for operations without a more specific client. Missing clients fail
only when their operation is invoked. Media is bounded and sent inline; use a provider-specific
upload adapter for larger assets. `generation_max_tokens` maps to Chat Completions `max_tokens`;
`generation_extra_body` passes caller-owned provider extensions through the official SDK. Both are
built into the shared grounded request, so `answer()` and `stream_answer()` always send identical
generation controls. `generation_video_limit` bounds distinct retrieved videos while preserving
overflow hit text; set it to `None` to disable that count limit. The byte limits still apply. SDK
clients remain caller-owned.

Set `transcription_model="gpt-transcribe"` to use OpenAI Transcribe for completed audio files.
Optional `transcription_prompt`, `transcription_keywords`, and `transcription_languages` are sent
through the official SDK. Their normalized values are hashed into the default
`transcription_space`; supply an explicit space only when the application maintains an equivalent
stable recipe identity itself. OpenAI Transcribe remains a plain `TranscriptionBackend`; realtime
audio, timestamps, diarization, and speaker embeddings require their corresponding specialized
backends.

`embedding_request_format="input"` uses the standard OpenAI Embeddings request field. Set it to
`"messages"` for OpenAI-compatible servers that implement chat-style embeddings, such as vLLM
multimodal pooling models. That mode sends ordered text and inline media as top-level chat messages
through the same `/embeddings` endpoint. The request format is included in the default durable
`embedding_space`; an explicit space must likewise distinguish recipes that use different formats.
In `messages` mode, `embedding_dimension` validates returned vectors but is not sent as the
provider's optional server-side dimension-reduction parameter.

`OpenAIModels.stream_answer()` requests streamed chat completions with final usage enabled.
`Memory.ask()` selects it automatically to measure first chunk, first token, total generation
latency, and provider-reported token usage. `answer()` remains the synchronous non-streaming method
required by `GenerationBackend`.

## Exceptions

All stable exceptions derive from `MindBridgeError`:

- `ValidationError`
- `MemoryNotFoundError`
- `IdentityNotFoundError`
- `SpeakerNotFoundError`
- `ModelError`
- `ModelOutputTruncatedError`
- `StorageError`
- `IndexUnavailableError`

`ModelOutputTruncatedError` is the `ModelError` raised when generation stopped at an output token
limit rather than finishing. It is deterministic: retrying the same request produces the same
failure. Raise `generation_max_tokens`, or lower the `ask` limit so less evidence competes with the
answer for the model's output budget. Every other `ModelError` may be transient.

Provider bodies, credentials, and local paths are not included in public model/storage error
messages.
