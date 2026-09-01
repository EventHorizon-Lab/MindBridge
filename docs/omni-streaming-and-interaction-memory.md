# Omni streams, speculative recall, and interaction records

MindBridge distinguishes three states in a continuous application:

1. A changing query snapshot used only for speculative search.
2. A completed observation that can be written durably.
3. An application-derived record that cites the observation it came from.

`StreamEvent` is the explicit form of those states: an `UPDATE` snapshot may drive speculative
retrieval, a `FINAL` persists exactly once, and `CANCEL` or end-of-stream writes nothing.
`AsyncAudioStream` and `AsyncVisionStream` map canonical capture values onto that contract.

MindBridge does not capture sensors, reconnect streams, select frames, parse partial containers, or
detect turn boundaries. Camera and microphone I/O, provider packet decoding, and pixel conversion
remain adapter work. The application decides when an observation is complete and normalizes it to
`ContentInput`.

The snippets assume an open memory plus already-read immutable media bytes. They show lifecycle and
provenance boundaries; media capture and decoding remain application work.

## Add completed observations

`Memory.add_stream` consumes a lazy iterable of `ContentInput` or `StreamInput` values. Each item
uses the ordinary `add` path and is durable and searchable before the next item is pulled. The
stream is not one transaction: a later item or source failure leaves the committed prefix intact.

Use `StreamInput` for per-observation event time, metadata, or memory type:

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

The async form accepts an `AsyncIterable` and yields through `async for`. Both forms preserve
source backpressure and name a failing item's zero-based `contents[index]` in the public error.
Mutable ASR hypotheses, incomplete media, and repeatedly overwritten file paths are not completed
observations; keep them out of durable ingestion.

## Prefetch a changing query

`AsyncOmniPrefetch` wraps an open `AsyncMemory` (named `async_memory` below) and overlaps retrieval
with an unfinished turn without storing its partial input. Every submission is a complete current
snapshot, not a delta:

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

Speculative search does not reinforce returned memories. Call `reinforce` only after the
application has observed positive use or feedback.

## Capture-event reducer

`AsyncCaptureStream` provides the lifecycle that raw `add_stream` lacks. Every `StreamEvent` carries
a `stream_id` (`"default"` when omitted), so microphones, cameras, and sessions may interleave
without sharing speculative state. The same ID comes back on the resulting `StreamCommit`.

| Phase | Association | Payload | Retrieval | Durable write |
| --- | --- | --- | --- | --- |
| `UPDATE` | One `stream_id` | Complete current `ContentInput` snapshot | Speculative and coalesced | Never |
| `FINAL` | One `stream_id` | Exact final `ContentInput` or `StreamInput` | Exact-final confirmation | Once |
| `CANCEL` | One `stream_id` | No payload | Only that stream is discarded/drained | Never |
| EOF before `FINAL` | All open IDs | — | Pending work is discarded/drained | Never |

```python
from mindbridge import AsyncCaptureStream, StreamEvent, StreamPhase


async def camera_events():
    yield StreamEvent(StreamPhase.UPDATE, left_frame, stream_id="camera-left")
    yield StreamEvent(StreamPhase.UPDATE, right_frame, stream_id="camera-right")
    yield StreamEvent(StreamPhase.FINAL, left_clip, stream_id="camera-left")
    yield StreamEvent(StreamPhase.FINAL, right_clip, stream_id="camera-right")


capture = AsyncCaptureStream(memory, limit=8)
async for commit in capture.consume(camera_events()):
    print(commit.stream_id, commit.record.id, commit.prefetch, commit.retrieval_error)
```

`max_streams` defaults to 32 and bounds the active prefetch workers. One worker runs at a time per
ID; different IDs may search concurrently. A final or cancel boundary frees that ID.

If final retrieval fails, the final observation still commits and `StreamCommit.retrieval_error`
carries a stable public code (or `retrieval_failed` for an unclassified exception). Task
cancellation before the final commit writes nothing; once the exact-final write starts, cancellation
waits for that commit and then propagates, so no worker writes after a cancelled task returned.

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


audio = AsyncAudioStream(memory, limit=8)
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


vision = AsyncVisionStream(memory, limit=8)
async for commit in vision.consume(vision_packets()):
    print(commit.stream_id, commit.record.id)
```

State and prefetch work stay independent per ID. Each scene retains only its latest frame as the
durable keyframe, which bounds memory without imposing a hidden codec or sampler. Finalize an
encoded video through `AsyncCaptureStream` when a full clip is required.

With native image embedding, frame changes drive speculative retrieval directly. With a text-only
embedder, `VisionPartial` drives retrieval and its final value attaches to the keyframe through
`StreamInput.description`. Without a partial, a configured `VisionDescriptionBackend` describes the
final frame once. With none of those routes, finalization fails instead of discarding the frame.

## Interaction-derived records

Without a configured former, interaction memory is an application convention over existing
records, not another store or a fourth `MemoryType`:

| Derived information | Type |
| --- | --- |
| Explicit preference or supported stable fact | `SEMANTIC` |
| One situated interaction with event time | `EPISODIC` |
| Response guidance supported by feedback | `PROCEDURAL` |

Store the completed source observation first. Continuing with the `source_record` returned above, a
later analysis can write an ordinary record whose metadata cites the source ID:

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

MindBridge stores that metadata but does not interpret its provenance or confidence. On this
path emotion, trait, and policy inference remain application work; the kernel does not run them
automatically. Configuring a `FormationBackend` instead moves the same information into typed,
kernel-validated records, as described next. Keep the source observation when a derived interpretation changes. Add corrected
evidence or delete the incorrect derived record instead of presenting a rewrite as the original
observation.

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

The typed roles are `OBSERVATION`, `ENTITY`, `EVENT`, `STATE`, `RELATION`, `AFFECT`, `TRAIT`, and
`RESPONSE_POLICY`. They refine the existing semantic, episodic, and procedural `MemoryType` roles;
they do not create eight stores.

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

## Spatial and historical retrieval

`RetrievalScope` asks what was valid in the world, what MindBridge knew at the time, or what was
nearby in one coordinate frame:

```python
from mindbridge import RetrievalScope, SpatialAnchor, SpatialContext

scope = RetrievalScope(
    valid_at=event_time,
    known_at=audit_time,
    near=SpatialContext(
        frame_id="home/map",
        anchor=SpatialAnchor.SUBJECT,
        x=2.0,
        y=1.0,
    ),
    radius_m=0.75,
)
hits = memory.search("Where was the red toolbox?", scope=scope)
```

Spatial search is same-frame and same-anchor only, and position uncertainty expands the conservative
intersection test; MindBridge never guesses a coordinate transform. The `valid_at` and `known_at`
axes are defined in
[memory types, time, and decay](memory-types-time-and-decay.md#valid-time-and-transaction-time).
The `occurred_from`/`occurred_until` filters remain separate and select raw event occurrence.

## Graph evidence gate

Formation introduces no graph database. Typed lineage and evidence links cover deterministic state
maintenance without another service. A graph candidate is justified only when public-path benchmarks
show that gold evidence is commonly retrieved at a larger K but displaced at the product K, the
failure spans independent memory units, and a prototype improves retrieval and answer metrics
without violating latency, storage, rebuild, and privacy budgets.

If that gate passes, start with an additive entity/relation projection derived from authoritative
SQLite records and rebuilt through the existing outbox lifecycle. It must not become a second source
of truth, create logical account scopes, or learn from abandoned speculative queries. See the
[competitive review](competitive-memory-systems.md) for the source audit and roadmap.

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
