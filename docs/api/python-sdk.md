# Python API

Supported public imports come from `mindbridge`.

`Memory` is MindBridge's canonical execution plane. `AsyncMemory`, REST, MCP, and the required
product CLI must dispatch to these same domain operations rather than implement parallel routing,
storage, defaults, or errors.

## Content values

```python
from mindbridge import AssetRef, Blob, ContentAtom, ContentInput
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

## Memory

```python
Memory(
    data_dir=".mindbridge",
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    index_speech: bool = False,
    minimum_relevance: float = 0.55,
    ambiguity_margin: float = 0.01,
    decay_half_life_days: float | None = None,
    speaker_similarity: float = 0.78,
    speaker_margin: float = 0.05,
    tracer: opentelemetry.trace.Tracer | None = None,
)
```

`embedder` is required. `Memory` validates adapter capabilities and durable space identity before
opening Zvec. It closes supplied adapters when the memory closes; provider clients owned by an
adapter may remain caller-owned, as documented by that adapter.

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

`minimum_relevance` rejects weak dense evidence, while `ambiguity_margin` returns no hits when the
top two dense confidences are effectively tied and the winner has neither a lexical nor temporal
anchor. Both are calibrated `[0, 1]` values and may be set to `0` to disable that gate.

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

### Retrieve and answer

```python
hits = memory.search(
    query,
    limit=10,
    memory_type=None,
    reference_at=None,
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
timezone-aware; the current UTC time is used when omitted.

Composite records are indexed with an aggregate vector and de-duplicated vectors for each text or
media atom. Text longer than 2,048 characters also receives overlapping contextual retrieval keys;
the complete record remains the returned evidence. Search over-fetches candidates and keeps the
maximum part score for each record. Weak or unresolved tied evidence can therefore return `()`.

### Feedback

```python
updated = memory.reinforce((hit.id,))
```

Call `reinforce` only after explicit positive feedback. Retrieval itself never changes access
strength. Confirmations provide a small bounded ranking boost even without decay and also slow
decay when it is enabled. The return value is the number of existing, distinct memories updated.

### Read, list, and delete

```python
record = memory.get(memory_id)
page = memory.list(limit=50, cursor=None)
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

### Index maintenance

```python
count = memory.reindex()
memory.optimize()
```

`reindex` rebuilds the disposable Zvec projection from every stored embedding and returns the
number of memories rebuilt. It never calls the embedder. `optimize` compacts the current index.

## AsyncMemory

`AsyncMemory` takes the same constructor arguments and exposes the same methods with `await`.

```python
async with AsyncMemory(
    "./data/async",
    embedder=embedder,
    answerer=answerer,
    transcriber=transcriber,
) as memory:
    await memory.add("Remember this")
    hits = await memory.search("Remember")
```

It runs the embedded synchronous consistency core through `asyncio.to_thread`. It is not a
provider compatibility layer.

## Return values

- `MemoryRecord`: stable ID, derived text, modality, memory type, assets, timestamps, metadata.
- `SearchHit`: the same visible memory fields plus a normalized score.
- `AnswerResult`: grounded answer text and the accepted hits.
- `Page`: records and an optional next cursor.
- `SpeakerSegment`: one timed speech segment and local identity fields.

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

`SpeechBackend` has the same transcription identity properties and returns `SpeechAnalysis` from
`analyze()`. Backends may implement more than one protocol. Calls can overlap, so adapters must be
thread-safe until `close()`.

## Built-in adapters

### Sentence Transformers

```python
embedder = SentenceTransformersEmbedder.load(
    "organization/model",
    revision="40-character-immutable-commit",
    device="cuda",
    batch_size=8,
)
```

The immutable revision, dimension, normalization, query/document semantics, and input recipe form
the durable embedding space.

`JinaOmniEmbedder` is a lazy pinned specialization:

```python
embedder = JinaOmniEmbedder(dimension=1024, device="cuda", batch_size=8)
```

It calls the model's official Sentence Transformers retrieval methods and wraps application text
so URL- or path-shaped text cannot activate media autodetection.

### FunASR

```python
speech = FunASRTranscriber(device="auto")
```

The adapter delegates model, VAD, and speaker execution to `funasr.AutoModel`, then validates and
maps the result into MindBridge speech values.

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
    generation_model="gpt-5-mini",
    transcription_model="whisper-1",
    transcription_space=None,
    embedding_capabilities=frozenset({Modality.TEXT}),
    generation_capabilities=frozenset({Modality.TEXT}),
    transcription_capabilities=frozenset({Modality.AUDIO}),
    generation_seed=None,
    generation_temperature=None,
    generation_max_tokens=None,
    generation_extra_body=None,
)
```

The common `client` is used for operations without a more specific client. Missing clients fail
only when their operation is invoked. Media is bounded and sent inline; use a provider-specific
upload adapter for larger assets. `generation_max_tokens` maps to Chat Completions `max_tokens`;
`generation_extra_body` passes caller-owned provider extensions through the official SDK. Both are
built into the shared grounded request, so `answer()` and `stream_answer()` always send identical
generation controls. SDK clients remain caller-owned.

`OpenAIModels.stream_answer()` requests streamed chat completions with final usage enabled.
`Memory.ask()` selects it automatically to measure first chunk, first token, total generation
latency, and provider-reported token usage. `answer()` remains the synchronous non-streaming method
required by `GenerationBackend`.

## Exceptions

All stable exceptions derive from `MindBridgeError`:

- `ValidationError`
- `MemoryNotFoundError`
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
