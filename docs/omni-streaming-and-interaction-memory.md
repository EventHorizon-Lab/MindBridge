# Omni streaming and interaction memory

MindBridge separates changing capture state from durable memory. `StreamEvent` is the explicit form
of that separation:

1. An `UPDATE` is a complete current snapshot used only for speculative search.
2. A `FINAL` is a completed observation eligible for exactly one durable write.
3. A `CANCEL` or end-of-stream before finality writes nothing.

`AsyncAudioStream` and `AsyncVisionStream` map canonical capture values onto that contract.

This page owns those lifecycle boundaries. Sensor capture, stream reconnection, provider packet
decoding, pixel conversion, frame selection, and turn detection remain application or adapter
work: the application decides when an observation is complete and normalizes it to `ContentInput`.

## Choose the smallest stream API

| Need | API |
| --- | --- |
| Lazily add completed independent observations | `Memory.add_stream()` or `AsyncMemory.add_stream()` |
| Acknowledge completed observations without waiting for models | `Memory.capture()` with `Memory.settle()`, or any stream API with `capture=True` |
| Search while one query snapshot is changing | `AsyncOmniPrefetch` |
| Associate speculative snapshots with final commits | `AsyncCaptureStream` |
| Normalize PCM, VAD, and ASR state | `AsyncAudioStream` |
| Normalize encoded frames and visual descriptions | `AsyncVisionStream` |

Use `add_stream()` unless the application benefits from speculative recall. Use the capture
reducers only when explicit finality is available. Every reducer takes an `AsyncMemory`, written
`async_memory` below; the synchronous snippets assume an open `memory` plus already-read immutable
media bytes.

## Add completed observations

`Memory.add_stream()` consumes a lazy iterable of `ContentInput` or `StreamInput` values. Each item
uses the ordinary `add()` path and is durable and searchable before the next item is requested. The
stream is not one transaction: a later source or item failure leaves the committed prefix intact.

Use `StreamInput` when observations need independent event time, metadata, memory type, context,
transcript, or visual description:

```python
from datetime import datetime, timedelta, timezone

from mindbridge import Blob, MemoryType, StreamInput

started = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)
observation = StreamInput(
    (
        "The user is looking for the red toolbox.",
        Blob(frame_bytes, "image/png", "frame.png"),
        Blob(audio_bytes, "audio/wav", "speech.wav"),
    ),
    occurred_at=started,
    occurred_end=started + timedelta(seconds=2),
    metadata={"source": "companion_robot", "sequence": 17},
    memory_type=MemoryType.EPISODIC,
)

for source_record in memory.add_stream((observation,)):
    print(source_record.id)
```

The async form accepts an `AsyncIterable` and yields with `async for`. Both forms preserve source
backpressure and identify a failing item as zero-based `contents[index]` in the public error.

Mutable ASR hypotheses, incomplete media, and repeatedly overwritten file paths are not completed
observations. Keep them out of durable `add_stream()` input.

## Fast capture

Use `capture()` instead of `add()` when the observation is final but the caller cannot wait for
transcription, embedding, indexing, and formation — a device that must acknowledge a burst of
finals, or a turn loop that would otherwise block on the slowest model. `capture()` returns after
the SQLite commit; the record becomes searchable when the host calls `settle()`.

```python
for observation in burst:
    memory.capture(observation)

while memory.settle(limit=32):
    pass
```

Loop on what `settle()` returns rather than on `pending_captures()`: a record that reached the
retry ceiling stays queued on purpose, so a loop that waits for the queue to empty never ends.
`pending_captures()` is how you then see which record it is and why, and `awaiting` says whether
a queued record has no vectors yet or is already searchable and owes only formation.
`settle(memory_ids=...)` runs named records alone and ignores the ceiling for them, so a parked
capture is retried by hand rather than by raising the ceiling for everything. One settlement runs
at a time per `Memory`: a concurrent call waits instead of running the same models twice.

**Contract:** keep `add()` where a caller needs the record searchable on return, and keep the
enrichment loop explicit. Nothing else settles for you, so an application that never calls
`settle()` accumulates durable memories that `search()`, `ask()`, and `compile()` cannot see. See
the [Python SDK reference](api/python-sdk.md#memory-operations) for the exact failure semantics.

## Prefetch a changing query

`AsyncOmniPrefetch` wraps an open `AsyncMemory` and overlaps retrieval with an unfinished turn
without storing its partial input. Every submission is the complete current query snapshot, not a
delta:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(async_memory, limit=8)
prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))

result = await prefetch.finalize((final_text, final_frame, final_audio_window))
hits = result.hits
```

Only one search runs at a time. While it runs, a newer submission replaces the snapshot that has
not started yet. `latest` returns the newest completed `PrefetchResult` and its revision without
waiting.

`finalize` returns only a result for the exact final snapshot. It reuses an already completed
revision when the values match and that revision succeeded; otherwise it waits for a search of the
final snapshot. It then closes that per-turn prefetcher.

Snapshots accept immutable text, `Blob`, and `AssetRef` values. Raw `Path` values are rejected
because the file may change after submission. If a turn is abandoned, `await prefetch.close()`
drops queued work and waits for any search already dispatched through `asyncio.to_thread`.
Cancelling that task would not stop its synchronous model or index call, so the helper drains it.

Prefetch never stores or reinforces a snapshot. Call `reinforce` only after the application has
observed positive use or feedback.

## Capture-event reducer

`AsyncCaptureStream` provides the lifecycle that raw `add_stream()` lacks. Every `StreamEvent`
carries a `stream_id` (`"default"` when omitted), so interleaved microphones, cameras, or sessions
keep independent speculative state. The same ID comes back on the resulting `StreamCommit`.

| Phase | Payload | Retrieval | Durable write |
| --- | --- | --- | --- |
| `UPDATE` | Complete current immutable `ContentInput` snapshot | Speculative and coalesced | Never |
| `FINAL` | Exact final `ContentInput` or `StreamInput` | Exact-final confirmation | Once |
| `CANCEL` | None | That stream alone is discarded and drained | Never |
| EOF before `FINAL` | — | Every open stream is discarded and drained | Never |

```python
from mindbridge import AsyncCaptureStream, StreamEvent, StreamPhase


async def camera_events():
    yield StreamEvent(StreamPhase.UPDATE, left_frame, stream_id="camera-left")
    yield StreamEvent(StreamPhase.UPDATE, right_frame, stream_id="camera-right")
    yield StreamEvent(StreamPhase.FINAL, left_clip, stream_id="camera-left")
    yield StreamEvent(StreamPhase.FINAL, right_clip, stream_id="camera-right")


capture = AsyncCaptureStream(async_memory, limit=8)
async for commit in capture.consume(camera_events()):
    print(commit.stream_id, commit.record.id, commit.prefetch, commit.retrieval_error)
```

`max_streams` defaults to 32 and bounds the active prefetch workers. One search runs at a time per
ID; different IDs may search concurrently. A final or cancel boundary frees that ID. The two packet
reducers additionally accept `context=`, either a fixed `ObservationContext` or a callable read once
per closed observation, because a capture stream outlives the observations it commits.

`AsyncCaptureStream`, `AsyncAudioStream`, `AsyncVisionStream`, and `add_stream()` all accept
`capture=True`, which commits each final through `capture()` instead of `add()`. That is the
complete path from continuous observation to acknowledgement: speculative `UPDATE` retrieval, a
`FINAL` acknowledged after the SQLite commit, and enrichment deferred to `settle()`. Every
`StreamCommit` then reports `pending_settlement=True`, and the record stays out of `search()`
until the host settles it. A `StreamInput` transcript or description is folded in at capture time,
so the deferred commit lands on the same content-addressed record the strong path would have
written. The default is unchanged: without the flag, a final still commits through `add()` and is
searchable when the commit yields.

```python
audio = AsyncAudioStream(async_memory, limit=8, capture=True)
async for commit in audio.consume(audio_packets()):
    assert commit.pending_settlement

while await async_memory.settle(limit=32):
    pass
```

If final retrieval fails, the observation still commits and `StreamCommit.retrieval_error` carries
a stable public error code, or `retrieval_failed` for an unclassified exception. Cancellation before
the final write starts stores nothing. Once that write starts, cancellation waits for the commit and
then propagates, so no worker writes after a cancelled task has returned.

`AsyncAudioStream` and `AsyncVisionStream` accept a `memory_type` and `context`. The context may be
one fixed `ObservationContext` for a static sensor or a zero-argument sampler for a moving sensor;
the sampler is read once when each observation closes, so its metric pose or symbolic `place_id`
belongs to that durable observation. Speculative updates do not persist the sampled context.

## Native audio protocol

`AsyncAudioStream` is the canonical speech-capture adapter. It accepts four immutable values:

- `PCMChunk`: interleaved WAV-compatible linear PCM plus sample rate, channels, sample width,
  timestamp, and `stream_id`;
- `VADPacket`: the complete current voice-active state;
- `ASRPartial`: the complete current ASR hypothesis, not a provider delta;
- `AcousticBoundary`: explicit `START`, `END`, or `CANCEL` finality.

```python
from mindbridge import (
    ASRPartial,
    AcousticBoundary,
    AsyncAudioStream,
    AudioBoundary,
    PCMChunk,
    VADPacket,
)


async def audio_packets():
    yield VADPacket(True, stream_id="headset")
    yield PCMChunk(first_pcm, sample_rate_hz=16_000, stream_id="headset")
    yield ASRPartial("where is the red", stream_id="headset")
    yield PCMChunk(second_pcm, sample_rate_hz=16_000, stream_id="headset")
    yield ASRPartial("where is the red toolbox", stream_id="headset")
    yield AcousticBoundary(AudioBoundary.END, stream_id="headset")


audio = AsyncAudioStream(async_memory, limit=8)
async for commit in audio.consume(audio_packets()):
    print(commit.stream_id, commit.record.id)
```

PCM accumulates independently per ID and is wrapped in a standard WAV container. A native-audio
embedding backend receives changing audio snapshots directly. A text-only backend receives ASR
partials for speculative retrieval; the final hypothesis binds to the WAV through
`StreamInput.transcript`, stays in the durable record, and drives text embedding. Without an
external hypothesis, the configured transcription backend is the fallback. With neither route the
operation fails rather than dropping audio.

`VADPacket(active=False)` and `AudioBoundary.END` finalize; `CANCEL` and incomplete EOF never write.
A PCM format change inside a segment, or a buffer beyond the local asset limit, fails validation.
Normalize vendor VAD and ASR payloads to these four values instead of teaching the kernel provider
packet schemas.

## Native vision protocol

`AsyncVisionStream` is the canonical live-frame adapter. It accepts three immutable values:

- `VisionFrame`: one encoded image `Blob` plus timestamp and `stream_id`;
- `VisionPartial`: the complete current caption, OCR, or detector description, not a provider delta;
- `SceneBoundary`: explicit `START`, `END`, or `CANCEL` finality.

```python
from mindbridge import (
    AsyncVisionStream,
    Blob,
    SceneBoundary,
    VisionBoundary,
    VisionFrame,
    VisionPartial,
)


async def vision_packets():
    yield VisionFrame(Blob(first_jpeg, "image/jpeg"), stream_id="camera-left")
    yield VisionPartial("a red toolbox beside the door", stream_id="camera-left")
    yield VisionFrame(Blob(latest_jpeg, "image/jpeg"), stream_id="camera-left")
    yield SceneBoundary(VisionBoundary.END, stream_id="camera-left")


vision = AsyncVisionStream(async_memory, limit=8)
async for commit in vision.consume(vision_packets()):
    print(commit.stream_id, commit.record.id)
```

State and prefetch work stay independent per ID. Each scene retains only its latest frame as the
durable keyframe, which bounds memory without imposing a hidden codec or sampler. Finalize an
encoded video through `AsyncCaptureStream` when a full clip is required.

With native image embedding, frame changes drive speculative retrieval directly. With a text-only
embedder, `VisionPartial` drives retrieval and its final value attaches to the keyframe through
`StreamInput.description`. Without a partial, a configured `VisionDescriptionBackend` describes the
final frame once, before finalization, so speculative retrieval has a query to work with. With none
of those routes, finalization fails instead of discarding the frame.

That early call is the text-only case only. It is a scheduling choice, not the condition for
describing at all: `add` and `add_many` call a configured describer for **every** embedder whenever
a visual asset still has no description, and union the text into the stored and indexed document
regardless of embedder capability, because what the lexical document contains is a separate question
from which modality the embedder accepts. Configuring no describer stays an exact no-op on both
paths.

## Interaction-derived records

Without a configured former, interaction memory is an application convention over the existing
three memory types, not another store or a fourth type:

| Derived information | Memory type |
| --- | --- |
| Explicit preference or supported stable fact | `SEMANTIC` |
| One situated interaction with event time | `EPISODIC` |
| Response guidance supported by feedback | `PROCEDURAL` |

Store the completed source observation first. Continuing with the `source_record` returned above,
application analysis can then add an ordinary record whose metadata cites the source:

```python
from mindbridge import MemoryType

guidance = memory.add(
    "When deadline pressure is present, acknowledge it before proposing steps.",
    metadata={
        "basis": "response_feedback",
        "evidence_ids": [source_record.id],
        "confidence": 0.8,
    },
    memory_type=MemoryType.PROCEDURAL,
)
```

On this manual path MindBridge stores that metadata but does not interpret its provenance,
confidence, or evidence IDs; emotion, trait, and policy inference remain application work.
Configuring a `FormationBackend` instead moves the same information into typed, kernel-validated
records, as described next.

Keep the source observation when a derived interpretation changes. Add corrected evidence or delete
the incorrect derived record instead of presenting a rewrite as the original observation.

Applications can query roles separately when one role must not crowd out another:

```python
semantic = memory.search(query, memory_type=MemoryType.SEMANTIC, limit=2)
episodic = memory.search(query, memory_type=MemoryType.EPISODIC, limit=2)
procedural = memory.search(query, memory_type=MemoryType.PROCEDURAL, limit=1)
```

Procedural records are evidence for the application; MindBridge never executes their text.
Metadata and evidence IDs do not create an authorization or isolation boundary.

## Automatic memory formation

Configure an optional `FormationBackend` to turn a committed observation into typed semantic
proposals. The former cannot write storage and is not trusted to choose IDs, evidence, or conflict
semantics:

```python
from mindbridge import Memory, OpenAIModels

models = OpenAIModels(
    embedding_client=embedding_client,
    generation_client=generation_client,
)

with Memory("./data", embedder=models, former=models) as memory:
    source = memory.add(observation.content, context=observation.context)
```

The raw source commits first. Formation then validates every proposal against the source modality
and spatial frame, assigns deterministic derived IDs, links evidence, versions state, embeds the new
records, and records the durable formation recipe in one SQLite transaction. If formation fails, the
source stays durable and repeating the same add can complete the missing projection. `OpenAIModels`
sends model content but never stable memory or CAS IDs, nor exact spatial values.

The typed kinds refine the existing semantic, episodic, and procedural `MemoryType` roles; they do
not create eight stores. Their meanings and assigned memory types are in
[typed context and formation](memory-types-time-and-decay.md#typed-context-and-formation).

## Affect and personality evidence

Affect is situated evidence. A proposal may carry cue modality, valence, arousal, confidence,
validity, spatial context, and source evidence. The cue modality must exist in the source
observation, so a text-only inference cannot claim to have heard vocal tension. Preserve text,
prosody, face, posture, and environment cues separately and fuse them late.

A trait is a long-horizon claim and has a stricter visibility rule:

- a trusted explicit `USER_STATEMENT` is visible immediately;
- a `MODEL_INFERENCE` stays hidden from active search until two independent source observations
  support the same normalized subject/predicate/value claim;
- repeated extraction from one observation or shared `source_id` counts once;
- confidence uses the maximum within one source and noisy-OR across independent sources;
- deleting source evidence recomputes confidence and can hide or remove the trait;
- explicit corrections use state-like bitemporal replacement instead of overwriting history.

`get` and `list` may expose a hidden trait with `MemoryContext.visible=False` for audit, while
ordinary search and answer exclude it. Developers can inspect why a profile did not activate without
letting one uncertain emotional episode personalize future behavior.

Response policies are typed procedural evidence, never executable code. They require explicit
feedback provenance before an application should use them to adapt companion behavior.

A spatial pose recorded on a stream observation is retrieved with `RetrievalScope`; see
[spatial scope](memory-types-time-and-decay.md#spatial-scope) for the same-frame rule and
[valid time and transaction time](memory-types-time-and-decay.md#valid-time-and-transaction-time)
for the historical axes. Formation introduces no graph database, and the gate a graph projection
would have to pass is in the [benchmark protocol](benchmarking.md#mandatory-controls).

## Observability

Every completed stream item emits an ordinary `mindbridge.add` operation span. Every speculative
revision that starts emits an ordinary `mindbridge.search` span. This keeps model request, token,
latency, and failure accounting comparable with non-streaming calls. See
[operations](operations.md#telemetry) for the attribute contract.

Useful end-to-end measurements for a capture reducer are:

- fraction of turns with a completed matching prefetch at finality;
- final-boundary-to-context latency at p50, p95, and p99;
- fraction of finals that require another search;
- searches started per turn and maximum concurrent searches per reducer;
- retrieval quality for the final snapshot, not just earlier partials;
- `CANCEL`, EOF, or pre-commit cancellation that produced a durable record, whose target is zero.
