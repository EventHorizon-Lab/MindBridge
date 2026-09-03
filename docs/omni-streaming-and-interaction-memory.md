# Omni streaming and interaction memory

MindBridge separates changing capture state from durable memory:

1. An `UPDATE` is a complete current snapshot used only for speculative search.
2. A `FINAL` is a completed observation eligible for one durable write.
3. A `CANCEL` or end-of-stream before finality writes nothing.

This page owns those lifecycle boundaries. Camera and microphone I/O, reconnection, provider packet
decoding, frame selection, and turn detection remain application or adapter work.

## Choose the smallest stream API

| Need | API |
| --- | --- |
| Lazily add completed independent observations | `Memory.add_stream()` or `AsyncMemory.add_stream()` |
| Acknowledge final observations without waiting for models | `Memory.capture()` with `Memory.settle()` |
| Search while one query snapshot is changing | `AsyncOmniPrefetch` |
| Associate speculative snapshots with final commits | `AsyncCaptureStream` |
| Normalize PCM, VAD, and ASR state | `AsyncAudioStream` |
| Normalize encoded frames and visual descriptions | `AsyncVisionStream` |

**Guidance:** Use `add_stream()` unless the application benefits from speculative recall. Use the
capture reducers only when explicit finality is available.

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

**Contract:** Mutable ASR hypotheses, incomplete media, and changing file paths are not completed
observations. Keep them out of durable `add_stream()` input.

## Fast capture

Use `capture()` instead of `add()` when the observation is final but the caller cannot wait for
transcription, embedding, indexing, and formation — a device that must acknowledge a burst of
finals, or a turn loop that would otherwise block on the slowest model. `capture()` returns after
the SQLite commit; the record becomes searchable when the host calls `settle()`.

```python
for observation in burst:
    memory.capture(observation)

while memory.pending_captures():
    memory.settle(limit=32)
```

Keep `add()` where a caller needs the record searchable on return, and keep the enrichment loop
explicit: nothing else settles for you, so an application that never calls `settle()` accumulates
durable memories that `search()` and `ask()` cannot see. See the
[Python SDK reference](api/python-sdk.md#memory-operations) for the exact failure semantics.

## Prefetch a changing query

`AsyncOmniPrefetch` wraps an open `AsyncMemory`. Each submission is the complete current query
snapshot, not a delta:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(async_memory, limit=8)
prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))

result = await prefetch.finalize((final_text, final_frame, final_audio_window))
hits = result.hits
```

Only one search runs at a time. While it runs, a newer submission replaces the queued snapshot.
`latest` returns the newest completed `PrefetchResult` without waiting. `finalize()` returns only a
result for the exact final snapshot: it reuses a matching successful revision or searches again,
then closes that per-turn prefetcher.

Snapshots accept immutable text, `Blob`, and `AssetRef` values. Raw `Path` values are rejected
because their bytes may change after submission. If a turn is abandoned, `await prefetch.close()`
drops queued work and drains any synchronous search already dispatched through a worker thread.

**Contract:** Prefetch never stores or reinforces a snapshot. Reinforce only after the application
observes positive use or feedback.

## Capture-event reducer

`AsyncCaptureStream` reduces `StreamEvent` values. Each event carries a `stream_id` (`"default"`
when omitted), so interleaved microphones, cameras, or sessions keep independent prefetch state.

| Phase | Payload | Retrieval | Durable write |
| --- | --- | --- | --- |
| `UPDATE` | Complete immutable `ContentInput` snapshot | Speculative and coalesced | Never |
| `FINAL` | Exact final `ContentInput` or `StreamInput` | Exact-final confirmation | Once |
| `CANCEL` | None | That stream is discarded and drained | Never |
| EOF before `FINAL` | — | Open streams are discarded and drained | Never |

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

`max_streams` defaults to 32 and bounds active prefetch workers. One search runs at a time per ID;
different IDs may search concurrently. A final or cancel boundary frees the ID.

If final retrieval fails, the observation still commits and `StreamCommit.retrieval_error` carries
a stable public error code, or `retrieval_failed` for an unclassified exception. Cancellation
before the final write starts stores nothing. Once that write starts, cancellation waits for it and
then propagates, so no worker writes after the cancelled task has returned.

## Native audio protocol

`AsyncAudioStream` accepts four immutable values:

- `PCMChunk`: interleaved WAV-compatible linear PCM plus format, timestamp, and `stream_id`;
- `VADPacket`: the complete current voice-active state;
- `ASRPartial`: the complete current hypothesis, not a provider delta;
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

PCM accumulates independently per ID and is stored in a standard WAV container. A native-audio
embedder receives changing audio snapshots. A text-only embedder uses ASR partials for speculative
retrieval; the final transcript remains attached to the WAV and drives text embedding. A configured
speech backend can transcribe final audio when no external hypothesis exists. With no supported
route, finalization fails instead of dropping audio.

`VADPacket(active=False)` and `AudioBoundary.END` finalize a segment. `CANCEL` and incomplete EOF do
not write. A PCM format change inside a segment, or a buffer beyond the local asset limit, fails
validation.

**Guidance:** Normalize vendor VAD and ASR payloads to these public values outside the kernel.

## Native vision protocol

`AsyncVisionStream` accepts three immutable values:

- `VisionFrame`: one encoded image `Blob` plus timestamp and `stream_id`;
- `VisionPartial`: the complete current caption, OCR, or detector description;
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

Each scene keeps only its latest frame as the durable keyframe. Use `AsyncCaptureStream` with an
encoded video when the complete clip must be stored. Native image embedding searches frame
snapshots directly. A text-only embedder uses `VisionPartial`; without one, an injected
`VisionDescriptionBackend` can describe the final frame once. With none of those routes,
finalization fails instead of discarding the image.

## Interaction-derived records

Without a former, interaction memory is an application convention over the existing three memory
types, not another store or a fourth type:

| Derived information | Memory type |
| --- | --- |
| Explicit preference or supported stable fact | `SEMANTIC` |
| One situated interaction with event time | `EPISODIC` |
| Response guidance supported by feedback | `PROCEDURAL` |

Store the completed source observation first. Application analysis can then add an ordinary record
whose metadata cites the source:

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

**Contract:** On this manual path, MindBridge stores metadata but does not interpret its provenance,
confidence, or evidence IDs. Procedural text is returned as evidence and is never executed.

For kernel-validated provenance, visibility, conflict handling, affect, traits, and response policy,
inject a `FormationBackend`; see
[typed context and formation](memory-types-time-and-decay.md#typed-context-and-formation). Formation
runs only after a completed source commits and never learns from abandoned speculative snapshots.

Streaming uses the ordinary add and search telemetry. See [operations](operations.md#telemetry) for
the span contract.
