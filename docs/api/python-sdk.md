# Python SDK

## Purpose

The supported root-import SDK is re-exported from `mindbridge`. `Memory` is the synchronous
execution boundary, and `AsyncMemory` is its async facade. REST, MCP, and the product CLI expose
subsets of these operations through their own adapters.

One physical `data_dir` belongs to one live `Memory` instance. There are no tenant, user, request,
or benchmark scope parameters. See [architecture](../architecture.md) for ownership and storage,
[configuration](../configuration.md) for backend composition, and
[operations](../operations.md) for lifecycle and observability.

## Invocation

Pass already-constructed backends directly:

```python
from mindbridge import JinaOmniEmbedder, Memory

with Memory("./data/assistant", embedder=JinaOmniEmbedder()) as memory:
    record = memory.add("The spare key is in the blue toolbox.")
    hits = memory.search("Where is the spare key?")
```

Or let the bundled declarative configuration construct its adapters:

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
    }
) as memory:
    memory.add("Remember this")
```

`Memory` owns and closes every backend object passed to it. A backend may still leave its
caller-supplied provider client open; that ownership is adapter-specific. Prefer a context manager
or call `close()` explicitly.

The Jina adapter used in these examples executes upstream model code at an immutable pinned
revision. Review its trust and license constraints in [configuration](../configuration.md#embedding-choices).

## Contract

### Construction

```text
Memory(
    data_dir: str | Path = ".mindbridge",
    *,
    embedder: EmbeddingBackend,
    answerer: GenerationBackend | None = None,
    transcriber: SpeechBackend | TranscriptionBackend | None = None,
    vision_describer: VisionDescriptionBackend | None = None,
    face_analyzer: FaceBackend | None = None,
    former: FormationBackend | None = None,
    index_speech: bool = False,
    index_quantization: IndexQuantization = IndexQuantization.NONE,
    minimum_relevance: float = 0.55,
    ambiguity_margin: float = 0.01,
    evidence_budget_chars: int | None = None,
    decay_half_life_days: float | None = None,
    speaker_similarity: float = 0.78,
    speaker_margin: float = 0.05,
    face_similarity: float = 0.363,
    face_margin: float = 0.05,
    tracer: opentelemetry.trace.Tracer | None = None,
) -> None
```

`embedder` is required. `index_speech=True` requires a speech-capable `SpeechBackend` and stores
transcripts and resolved speaker names with new memories. `index_quantization` changes only the
rebuildable vector index; supported values are `none`, `fp16`, `int8`, and `rabitq`. The similarity,
margin, relevance, and decay settings are validated when the instance opens. Their behavior and
the supported provider configuration fields live in [configuration](../configuration.md).

A plain `TranscriptionBackend` transcribes supported audio/video during `add` regardless of
`index_speech`; `SpeechBackend` analysis and identity resolution stay behind the explicit flag.

`vision_describer` has no declarative provider and is reachable only through direct construction
or `MemoryPlugins`. `former` proposes typed memories after a source observation commits; omitting
it keeps ordinary add behavior and makes no formation model call. The bundled OpenAI former is
selected by the declarative `formation` slot, which stays off unless it is configured; see
[configuration](../configuration.md#automatic-memory-formation).

Turn `index_speech` on for any corpus whose memories carry speech. Without it a video memory's
indexed document is whatever text the caller supplied, which for clip-shaped ingestion is often a
source identifier and nothing else, leaving the full-text route unable to match anything. On a
15-unit, 187-question M3-Bench robot slice it moved the mean indexed document from 29 to 536
characters and accuracy from 0.2086 to 0.3583 (paired mean +0.1497, 95% CI [+0.0857, +0.2123]),
at unchanged generation token usage because the inlined media dominates the prompt either way.

`evidence_budget_chars` decides how much evidence `ask()` grounds on. Retrieval scores separate the
right memory from the rest only weakly, so the answering model performs the final selection and
needs to see enough candidates. `None` grounds on exactly `limit`. An integer keeps those `limit`
hits and then admits further ranked memories while the grounded evidence fits the budget, ranking
the full rerank pool rather than `3 * limit`. Media is charged its modality's text equivalent —
2000 characters for an image, 4000 for audio, 12 000 for video — because a media part costs a model
far more than the few bytes of text on its record.

The two other construction boundaries are:

```text
Memory.from_plugins(
    data_dir: str | Path = ".mindbridge",
    *,
    plugins: MemoryPlugins,
    config: MemoryConfig | None = None,
    tracer: Tracer | None = None,
) -> Memory

Memory.from_config(
    config: MindBridgeConfig | Mapping[str, object],
    *,
    tracer: Tracer | None = None,
) -> Memory

resolve_memory_config(
    value: MindBridgeConfig | Mapping[str, object],
) -> MemoryComposition
```

`MemoryPlugins` contains `embedder`, optional `answerer`, optional `transcriber`, optional
`vision_describer`, optional `face_analyzer`, and optional `former`. `MemoryConfig` contains the value-only settings from the constructor;
`MemorySettings` is its public alias. `MemoryComposition` contains `data_dir`, `plugins`, and
`settings`; call `close()` unless its plugins have been transferred to a `Memory`.

### Content contract

```python
ContentAtom = str | pathlib.Path | Blob | AssetRef
ContentInput = ContentAtom | Sequence[ContentAtom]
```

- `str` is always text, even when it resembles a path or URL.
- `Path` is a local regular media file. Its suffix supplies the media type.
- `Blob(data: bytes, media_type: str, name: str | None = None)` is non-empty inline image, video,
  or audio data.
- `AssetRef(id, ...)` refers to media already stored in the same `data_dir`; `id` is its 64-character
  lowercase SHA-256 identifier. A returned reference includes `modality`, `media_type`,
  `size_bytes`, `sha256`, `name`, and its local `path`.
- An ordered sequence combines text and media into one record. MindBridge does not fetch URLs.

Event timestamps must be timezone-aware. `occurred_end` requires `occurred_at` and must be later
than it. Metadata is a JSON-compatible mapping with non-empty string keys. `MemoryType` is
`semantic`, `episodic`, or `procedural`; see
[memory types, time, and decay](../memory-types-time-and-decay.md).

The stable memory ID covers the ordered canonical content, metadata, event start/end, and memory
type. Repeating the same canonical input returns the same record without another model call.

`StreamInput` adds per-item values to a stream:

```text
StreamInput(
    content: ContentInput,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
)
```

### Memory operations

```text
add(
    content: ContentInput,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: ObservationContext | None = None,
) -> MemoryRecord

add_many(
    contents: Sequence[ContentInput],
    *,
    occurred_at: Sequence[datetime | None] | None = None,
    occurred_end: Sequence[datetime | None] | None = None,
    metadata: Sequence[Mapping[str, object] | None] | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    context: Sequence[ObservationContext | None] | None = None,
) -> tuple[MemoryRecord, ...]

add_stream(
    contents: Iterable[ContentInput | StreamInput],
) -> Iterator[MemoryRecord]
```

`add` is content-addressed and idempotent. `add_many` uses one model batch and one SQLite
transaction; each optional per-item sequence must have the same length as `contents`. An empty
batch returns `()`. `add_stream` requests one item at a time and makes each yielded record durable
and searchable before requesting the next. If a later item fails, earlier records remain and the
error `subject` identifies `contents[N]`.

```text
search(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> tuple[SearchHit, ...]

search_with_trace(
    query: ContentInput,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> TracedSearchResult

ask(
    question: ContentInput,
    *,
    limit: int = 5,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    scope: RetrievalScope | None = None,
) -> AnswerResult
```

`reference_at` is the timezone-aware clock for relative time and decay. The current UTC time is
used when it is omitted. `occurred_from` and `occurred_until` are optional timezone-aware,
half-open event-overlap filters; either may be omitted, and two bounds require
`occurred_until > occurred_from`. Records without `occurred_at` do not match a bounded search.

`scope` adds the bitemporal and spatial filters. `valid_at` selects world validity, `known_at`
selects the transaction version known then, and `near` with a non-negative `radius_m` restricts
results to one matching coordinate frame and anchor. SQLite reapplies every scope filter after
candidate retrieval; see
[valid time and transaction time](../memory-types-time-and-decay.md#valid-time-and-transaction-time).

`search_with_trace(...).hits` equals the corresponding `search(...)` result. Its bounded trace
contains identifiers, score components, ranks, and rejection reasons, but no query, content,
metadata, media, vectors, paths, or model output. `ask` requires an answerer and returns only the
retrieved hits the answerer actually used.

```text
get(memory_id: str) -> MemoryRecord
speech(memory_id: str) -> tuple[SpeakerSegment, ...]
faces(memory_id: str) -> tuple[FaceObservation, ...]
register_speaker(speaker_id: str, name: str, *, relationship: str | None = None) -> None
register_identity(identity_id: str, name: str, *, relationship: str | None = None) -> None
identity(identity_id: str) -> IdentityProfile | None
unlink_identity(alias_id: str) -> str | None
reinforce(memory_ids: Sequence[str]) -> int
list(*, limit: int = 100, cursor: str | None = None) -> Page
delete(memory_id: str) -> bool
reindex() -> int
optimize() -> None
close() -> None
```

`get` raises `MemoryNotFoundError` for an unknown ID. `speech` and `faces` return `()` when the
record has no relevant media and require their respective configured backend otherwise.
Registration assigns or replaces a printable name, and optionally a relationship, for an
existing local identity; both raise `IdentityNotFoundError` for an unknown ID. There is no
enrollment call that creates a person from a photograph or a voice sample. Register an identity
the recognizers have already observed instead: add the media, read `faces()` or `speech()` for its
`identity_id`, then name it. Omitting `relationship` on a later call leaves any recorded
relationship intact, so renaming never silently discards it, and there is deliberately no way to
clear one. `identity` resolves an ID through any merge alias and returns what has been registered,
so an observation captured before a merge still reaches the surviving person. `unlink_identity`
reverses one face-and-voice merge, returning the restored ID or `None` when the merge is not
reversible; it resets the pair's accumulated evidence rather than suppressing the pair, so a voice
and face that keep co-occurring are corroborated and merged again. `reinforce`
records explicit positive feedback and returns the number of existing distinct memories updated.
`list` uses an opaque keyset cursor. `delete` is idempotent and reports whether the record existed.
`reindex` rebuilds Zvec from authoritative SQLite embeddings without calling the embedder and
returns the number of memories rebuilt. `optimize` merges and flushes staged index vectors.
Repeated `close()` calls are harmless.

### Cross-modal identity binding

A face and a voice become one identity only after the same pair co-occurs in
`identity_link_min_assets` distinct assets (default `2`). A single asset cannot separate "this
face spoke" from "this face was listening to someone off camera", which is the ordinary case in
egocentric capture: the wearer talks while somebody else fills the frame. Binding on one asset
therefore attaches the wearer's voice to whoever happened to be visible. Set
`identity_link_min_assets=1` to bind on first co-occurrence.

Binding is recorded, not guessed at read time, so it is durable and reversible through
`unlink_identity`. Recognizer yield per asset is reported under the `mindbridge.identity.*` span
attributes, including when nothing was detected.

### AsyncMemory

`AsyncMemory` has the same constructor, class methods, operation names, keyword parameters,
defaults, and result values as `Memory`. Finite operations are awaited; `close()` is asynchronous.
Its stream boundary is:

```text
add_stream(
    contents: AsyncIterable[ContentInput | StreamInput],
) -> AsyncIterator[MemoryRecord]
```

The facade runs the synchronous embedded consistency core with `asyncio.to_thread`; it does not
turn synchronous provider clients into native async clients.

`AsyncOmniPrefetch` coalesces evolving query snapshots for one turn:

```text
AsyncOmniPrefetch(
    memory: AsyncMemory,
    *,
    limit: int = 10,
    memory_type: MemoryType | None = None,
    reference_at: datetime | None = None,
    occurred_from: datetime | None = None,
    occurred_until: datetime | None = None,
)

latest: PrefetchResult | None
submit(query: ContentInput) -> int
async finalize(query: ContentInput | None = None) -> PrefetchResult
async close() -> None
```

Only one search runs at once and only the newest queued snapshot survives. `finalize` returns the
exact final revision and closes the helper. Snapshots reject mutable `Path` atoms; use `Blob` or
`AssetRef`.

### Public values

The principal immutable values are:

| Value | Fields |
| --- | --- |
| `Blob` | `data`, `media_type`, `name` |
| `AssetRef` | `id`, `modality`, `media_type`, `size_bytes`, `sha256`, `name`, `path`; `is_resolved` property |
| `StreamInput` | `content`, `occurred_at`, `occurred_end`, `metadata`, `memory_type`, `context`, `transcript`, `description` |
| `MemoryRecord` | `id`, `content`, `created_at`, `occurred_at`, `occurred_end`, `metadata`, `assets`, `modality`, `memory_type`, `context` |
| `SearchHit` | all visible memory fields plus `score` |
| `AnswerResult` | `answer`, `hits`, `abstained`, `abstention_reason` |
| `Page` | `items`, `next_cursor` |
| `SpeakerSegment` | `asset_id`, `start_ms`, `end_ms`, `text`, `speaker_id`, `speaker_name`, `identity_score` |
| `FaceObservation` | `asset_id`, `bounding_box`, `identity_id`, `identity_name`, `identity_score`, `observed_at_ms` |
| `SpatialContext` | `frame_id`, `anchor`, `x`, `y`, `z`, `orientation_xyzw`, `position_uncertainty_m` |
| `ObservationContext` | `basis`, `source_id`, `confidence`, `valid_from`, `valid_until`, `spatial` |
| `MemoryContext` | `kind`, `basis`, `confidence`, `valid_from`, `valid_until`, `recorded_at`, `visible`, `retired_at`, `lineage_id`, `source_id`, `subject`, `predicate`, `value`, `evidence_ids`, `supersedes_id`, `model_id`, `recipe`, `spatial`, `cue_modality`, `valence`, `arousal` |
| `RetrievalScope` | `valid_at`, `known_at`, `near`, `radius_m` |
| `StreamEvent` | `phase`, `item`, `stream_id` |
| `StreamCommit` | `record`, `prefetch`, `retrieval_error`, `stream_id` |
| `PCMChunk` | `data`, `sample_rate_hz`, `channels`, `sample_width_bytes`, `stream_id`, `occurred_at` |
| `VADPacket` | `active`, `stream_id`, `occurred_at` |
| `ASRPartial` | `text`, `stream_id`, `occurred_at` |
| `AcousticBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `VisionFrame` | `image`, `stream_id`, `occurred_at` |
| `VisionPartial` | `text`, `stream_id`, `occurred_at` |
| `SceneBoundary` | `boundary`, `stream_id`, `occurred_at` |
| `PrefetchResult` | positive `revision`, `hits` |
| `TracedSearchResult` | `hits`, `trace` |
| `RetrievalTrace` | `candidates`, `candidate_limit`, `exhaustive`, `ambiguous` |
| `RetrievalCandidateTrace` | `memory_id`, `index_ids`, `dense_relevance`, `dense_confidence`, `lexical_relevance`, `lexical_rerank_bonus`, `lexical_match`, `gate_confidence`, `base_relevance`, `reinforcement_factor`, `temporal_factor`, `retention_factor`, `final_score`, `rank`, `rejected_by` |

`abstained` reports that the answerer returned the exact sentence the grounding prompt reserves for
having no usable evidence. It is not a measure of how often a model declined to answer: a model that
refuses in its own words is not abstaining by this definition, and on one EgoLifeQA slice 12 of 51
answers read as refusals without using the sentence. Treat it as a protocol signal and measure
refusal rate separately if that is the quantity needed.

Enum values are:

| Enum | Values |
| --- | --- |
| `Modality` | `text`, `image`, `video`, `audio`, `omni` |
| `MemoryType` | `semantic`, `episodic`, `procedural` |
| `IndexQuantization` | `none`, `fp16`, `int8`, `rabitq` |
| `AbstentionReason` | `no_evidence`, `insufficient_evidence` |
| `RetrievalRejection` | `stale_index`, `occurrence_range`, `missing_memory`, `memory_type`, `minimum_relevance`, `ambiguity`, `limit` |
| `EmbedTask` | `retrieval.query`, `retrieval.document` |
| `MemoryKind` | `observation`, `entity`, `event`, `state`, `relation`, `affect`, `trait`, `response_policy` |
| `EvidenceBasis` | `observation`, `user_statement`, `model_inference`, `response_feedback` |
| `SpatialAnchor` | `observer`, `subject` |
| `StreamPhase` | `update`, `final`, `cancel` |
| `AudioBoundary` | `start`, `end`, `cancel` |
| `VisionBoundary` | `start`, `end`, `cancel` |

### Backend protocols

Backends are runtime-checkable, thread-safe protocols. Their required methods are:

```text
EmbeddingBackend.embed(
    inputs: Sequence[ModelInput],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]

GenerationBackend.answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> AnswerResult

StreamingGenerationBackend.stream_answer(
    question: ModelInput,
    hits: Sequence[SearchHit],
) -> Iterator[str]

TranscriptionBackend.transcribe(
    assets: Sequence[AssetRef],
) -> tuple[str, ...]

SpeechBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[SpeechAnalysis, ...]

FaceBackend.analyze(
    assets: Sequence[AssetRef],
) -> tuple[FaceAnalysis, ...]

VisionDescriptionBackend.describe(
    inputs: Sequence[ModelInput],
) -> tuple[str, ...]

FormationBackend.form(
    inputs: Sequence[FormationInput],
) -> tuple[tuple[FormationProposal, ...], ...]
```

Required properties are `embedding_capabilities`, `embedding_model`, `embedding_space`, and
`embedding_dimension` for embedding; `transcription_capabilities`, `transcription_model`, and
`transcription_space` for transcription and speech; `face_capabilities`, `face_model`,
`face_space`, and `face_analysis_space` for faces; `vision_capabilities` and `vision_model` for
visual description; `formation_capabilities`, `formation_model`, and `formation_space` for
formation; and `generation_capabilities` for generation. Every base protocol except the optional
streaming extension implements `close()`.

`form` receives one `FormationInput` per committed source and returns one proposal tuple per
input, in the same order. A former never writes storage: the kernel validates each proposal
against the source modality and spatial frame, assigns identity, links evidence, and commits.

`ModelInput` contains normalized `text` and resolved `assets`. Speech adapters return
`SpeechAnalysis(turns, speakers)` using `SpeechTurn` and `SpeakerEmbedding`; face adapters return
`FaceAnalysis(faces)` using `FaceEmbedding`.

| Backend value | Fields |
| --- | --- |
| `ModelInput` | `text`, `assets`; derived `modality` and `modalities` properties |
| `SpeechTurn` | `start_ms`, `end_ms`, `text`, `speaker_label` |
| `SpeakerEmbedding` | `speaker_label`, `values` |
| `SpeechAnalysis` | `turns`, `speakers` |
| `FaceEmbedding` | `face_label`, `values`, `bounding_box`, `observed_at_ms` |
| `FaceAnalysis` | `faces` |
| `FormationInput` | `memory_id`, `content`, `context` |
| `FormationProposal` | `kind`, `content`, `basis`, `subject`, `predicate`, `value`, `confidence`, `valid_from`, `valid_until`, `spatial`, `cue_modality`, `valence`, `arousal` |

### Bundled adapters

The public construction signatures are:

```text
SentenceTransformersEmbedder(
    encoder,
    *,
    model_id: str,
    revision: str,
    dimension: int | None = None,
    batch_size: int = 32,
)

SentenceTransformersEmbedder.load(
    model_id: str,
    *,
    revision: str,
    dimension: int | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> SentenceTransformersEmbedder

JinaOmniEmbedder(
    *,
    dimension: int = 1024,
    device: str | None = None,
    batch_size: int = 32,
)

FunASRTranscriber(
    recipe: FunASRRecipe = DEFAULT_FUNASR_RECIPE,
    *,
    device: str = "auto",
)

FunASRRecipe(
    model_id: str,
    vad_model: str,
    speaker_model: str | None,
    model_revision: str | None = None,
    vad_revision: str | None = None,
    speaker_revision: str | None = None,
    punctuation_model: str | None = None,
    punctuation_revision: str | None = None,
    vad_max_single_segment_ms: int | None = None,
    hub: str = "ms",
    trust_remote_code: bool = False,
)

OpenCVFaceAnalyzer(
    detector_model: str | Path,
    recognizer_model: str | Path,
    *,
    score_threshold: float = 0.9,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
    frame_interval_ms: int = 1000,
    max_video_frames: int = 300,
)
```

The direct `SentenceTransformersEmbedder` constructor accepts a caller-owned encoder implementing
`supports`, `get_embedding_dimension`, `encode_query`, and `encode_document`; `load` constructs that
encoder and requires a 40-character immutable commit revision. `JinaOmniEmbedder.load` has the same
keyword parameters as its constructor and eagerly loads the pinned model. Jina loading sets
`trust_remote_code=True` while pinning both model and code revisions. Its weights are CC BY-NC 4.0;
that license covers the weights, not MindBridge. `DEFAULT_FUNASR_MODEL_ID` and
`DEFAULT_FUNASR_RECIPE` publish the default speech recipe.
`FunASRRecipe.auto_model_arguments() -> dict[str, object]` returns its standard FunASR composition.

`OpenAIModels` can fill embedding, generation, and transcription slots independently:

```text
OpenAIModels(
    client: OpenAI | None = None,
    *,
    embedding_client: OpenAI | None = None,
    generation_client: OpenAI | None = None,
    transcription_client: OpenAI | None = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_space: str | None = None,
    embedding_dimension: int = 1536,
    embedding_request_format: Literal["input", "messages"] = "input",
    generation_model: str = "gpt-5-mini",
    transcription_model: str = "whisper-1",
    transcription_space: str | None = None,
    transcription_prompt: str | None = None,
    transcription_keywords: Sequence[str] | None = None,
    transcription_languages: Sequence[str] | None = None,
    embedding_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    generation_capabilities: frozenset[Modality] = frozenset({Modality.TEXT}),
    transcription_capabilities: frozenset[Modality] = frozenset({Modality.AUDIO}),
    generation_seed: int | None = None,
    generation_temperature: float | None = None,
    generation_max_tokens: int | None = None,
    generation_min_video_seconds: float | None = None,
    generation_video_limit: int | None = 8,
    generation_extra_body: Mapping[str, object] | None = None,
)
```

The common `client` fills any operation-specific client left unset. A missing client fails only
when that operation is called. `close()` does not close caller-supplied OpenAI clients. Provider
selection, extras, model identity, and credential behavior are documented once in
[configuration](../configuration.md).

`embedding_request_format` selects an `input` or `messages` request for compatible embedding
endpoints. Transcription prompt, keywords, and languages form the provider hint.
`generation_min_video_seconds` converts shorter videos to four ordered stills when image generation
capability is available; `generation_video_limit` caps retrieved evidence videos but not question
media. `generation_extra_body` is forwarded to the SDK request.

Its operation signatures widen the base protocols only by accepting plain text where useful:

```text
embed(
    inputs: Sequence[ModelInput | str],
    task: EmbedTask = EmbedTask.DOCUMENT,
) -> tuple[tuple[float, ...], ...]
answer(question: ModelInput | str, hits: Sequence[SearchHit]) -> AnswerResult
stream_answer(
    question: ModelInput | str,
    hits: Sequence[SearchHit],
) -> Generator[str, None, tuple[SearchHit, ...]]
transcribe(assets: Sequence[AssetRef]) -> tuple[str, ...]
close() -> None
```

OpenAI requests inline at most 20 MiB of base64 data per media item and 64 MiB in aggregate. Because
base64 expands input, that is roughly 15 MiB per source file and 48 MiB per call on disk. Grounded
answer text is capped at 4 MiB. Answer fitting reserves media capacity for the question, keeps
highest-ranked evidence that fits, and leaves overflow evidence as text when possible; use a
provider-specific upload adapter for larger media.

Named recipes are also available as a small, closed construction API:

```text
from mindbridge import recipes

recipes.names() -> tuple[str, ...]
recipes.slots(name: str) -> tuple[str, ...]
recipes.require_slot(name: str, slot: str) -> None
recipes.probe(name: str) -> str
recipes.describe(name: str) -> dict[str, object]
recipes.embedder(name: str, *, load: bool = False) -> EmbeddingBackend
recipes.answerer(name: str, *, load: bool = False) -> GenerationBackend
recipes.transcriber(
    name: str,
    *,
    load: bool = False,
) -> SpeechBackend | TranscriptionBackend
```

Each construction function returns an object the caller owns and closes. `load=True` additionally
exercises the recipe's published probe (`weights`, `import`, or `client`).

## Errors and limits

All stable exceptions derive from `MindBridgeError`:

| Exception | `code` | Default `reason` |
| --- | --- | --- |
| `MindBridgeError` | `mindbridge_error` | `None` |
| `ValidationError` | `validation_error` | `input_invalid` |
| `MemoryNotFoundError` | `memory_not_found` | `memory_not_found` |
| `SpeakerNotFoundError` | `speaker_not_found` | `speaker_not_found` |
| `IdentityNotFoundError` | `identity_not_found` | `identity_not_found` |
| `ModelError` | `model_error` | `None` |
| `ModelOutputTruncatedError` | `model_output_truncated` | `output_truncated` |
| `StorageError` | `storage_error` | `None` |
| `IndexUnavailableError` | `index_unavailable` | `None` |

Every error may carry `reason`, `stage`, and `subject`. `retryable` is true only when `reason` is
`connection_failed`, `data_dir_in_use`, `flush_failed`, `index_missing`, `rate_limited`, or
`timeout`; it is never inferred from the message. `ModelOutputTruncatedError` is a deterministic
`ModelError`, and `IndexUnavailableError` is a `StorageError`.

Stable input bounds are:

| Bound | Value |
| --- | --- |
| Ordered parts in one `ContentInput` | 128 |
| Normalized text in one operation, including combined text atoms | 65,536 characters |
| Serialized metadata for one memory | 262,144 UTF-8 bytes |
| One local media asset | 512 MiB |
| `limit` for search, answer, and listing operations | 1 through 100 |
| Identity or speaker name | 255 printable characters |

The 512 MiB value is the local ingestion ceiling, not a promise that every model backend accepts an
asset that large. Backend request limits apply before model work.

Two degradations keep a write alive rather than failing it, and both are recorded on the operation
span. Media the embedding model will not accept inline drops the retrieval key holding it: the
memory is still stored with its media and stays reachable through the keys that did embed, counted
by `mindbridge.embedding.elided_parts`. A memory whose every key is refused still fails, because
nothing would find it. Separately, a video the embedding model cannot fit in its context is embedded
as four ordered stills, counted by `mindbridge.embedding.video_sampled_inputs`; only a rejection
that declares the length constraint triggers it, and the video itself is still stored and still
reaches the answering model on its own route.

`add_many` has no separate SDK item-count cap, but each item is subject to the same content and
metadata bounds. REST and MCP deliberately impose narrower transport limits. A `Memory` is bound
to the process that opened it and cannot be used after `fork`; open a new instance with a different
`data_dir` in the child.
