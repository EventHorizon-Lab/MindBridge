# Omni streaming, speculative recall, and interaction memory

MindBridge separates four lifecycles that a continuous embodied agent must not confuse:

1. A changing working snapshot used only for speculative search.
2. A final observation that is safe to store durably.
3. A typed interpretation formed from that committed source.
4. A long-horizon trait or response policy supported by independent evidence.

The same boundary applies to text, image, video, audio, and combined omni input. MindBridge does not
own cameras, microphones, stream containers, VAD implementations, provider-specific ASR packets,
or pixel conversion. It owns canonical audio packets and immutable encoded-frame, visual-partial,
and scene-boundary values; capture adapters normalize hardware and provider payloads into those
values or immutable `ContentInput` snapshots.

## Completed observations

`Memory.add_stream` consumes already completed observations lazily. Each item follows ordinary
`add`, and is durable and searchable before the next item is requested. The stream is not one
transaction: a later failure leaves the committed prefix intact.

Use `StreamInput` when an item needs its own time, metadata, role, or typed observation context:

```python
from datetime import datetime, timedelta, timezone

from mindbridge import (
    Blob,
    EvidenceBasis,
    MemoryType,
    ObservationContext,
    SpatialAnchor,
    SpatialContext,
    StreamInput,
)

observed_at = datetime(2026, 8, 31, 9, tzinfo=timezone.utc)
observation = StreamInput(
    (
        "The user is looking for the red toolbox.",
        Blob(frame_bytes, "image/png", "frame.png"),
        Blob(audio_bytes, "audio/wav", "speech.wav"),
    ),
    occurred_at=observed_at,
    occurred_end=observed_at + timedelta(seconds=2),
    metadata={"sequence": 17},
    memory_type=MemoryType.EPISODIC,
    context=ObservationContext(
        basis=EvidenceBasis.OBSERVATION,
        source_id="companion-camera-1:17",
        confidence=0.92,
        spatial=SpatialContext(
            frame_id="home/map",
            anchor=SpatialAnchor.OBSERVER,
            x=1.4,
            y=2.1,
            position_uncertainty_m=0.08,
        ),
    ),
)

for record in memory.add_stream((observation,)):
    print(record.id, record.context)
```

An async owner consumes an `AsyncIterable` with `async for`. Both paths preserve backpressure and
use the same SQLite, outbox, and Zvec consistency path as `add`. Mutable ASR hypotheses, incomplete
container fragments, and overwritten frame paths are not completed observations and must not be
passed to `add_stream`.

The CLI `add-stream` command provides per-line commit behavior for finite JSONL imports. It emits one
JSON document at EOF, so Python remains the unbounded-source interface.

## Speculative omni recall

`AsyncOmniPrefetch` hides retrieval latency inside an ongoing turn without persisting partial
input. Every submission is the complete current snapshot, not a delta:

```python
from mindbridge import AsyncOmniPrefetch

prefetch = AsyncOmniPrefetch(memory, limit=8)
prefetch.submit((partial_text, latest_frame, current_audio_window))
prefetch.submit((newer_text, latest_frame, current_audio_window))
result = await prefetch.finalize((final_text, final_frame, final_audio_window))
```

Only one search runs at a time. A newer queued snapshot replaces the older queued snapshot, so
partials do not create a thread storm. `latest` exposes the newest completed `PrefetchResult`
without waiting. `finalize` reuses a result only when its revision represents the exact final
snapshot; otherwise it searches the final value before returning.

Snapshots accept immutable text, `Blob`, and resolved `AssetRef` values. Mutable `Path` values are
rejected for `UPDATE` because their bytes could change after submission. `close()` abandons queued
work and drains one already-running synchronous search. Search does not reinforce hits.

## Capture-event reducer

`AsyncCaptureStream` provides the common lifecycle missing from raw `add_stream`. Every
`StreamEvent` has a `stream_id` (`"default"` when omitted), so events from microphones, cameras, or
sessions may be interleaved without sharing speculative state. The same ID is returned on the
resulting `StreamCommit`.

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

`max_streams` defaults to 32 and bounds the number of active prefetch workers. One worker runs at a
time per ID; different IDs may search concurrently. A final or cancel boundary frees that ID.

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
partials for speculative retrieval; the final hypothesis is bound to the WAV through
`StreamInput.transcript`, remains in the durable record, and drives text embedding. If no external
ASR hypothesis exists, the configured transcription backend remains the fallback. If neither route
exists, the operation fails before dropping audio.

`VADPacket(active=False)` and `AudioBoundary.END` finalize; `CANCEL` and incomplete EOF never write.
PCM format changes inside a segment and buffers beyond the local asset limit fail validation.
Microphone APIs and provider-specific VAD or ASR payloads remain thin adapter work: normalize them
to these four values rather than teaching the memory kernel vendor packet schemas.

## Native vision protocol

`AsyncVisionStream` is the canonical live-frame adapter. It accepts three immutable values:

- `VisionFrame`: one encoded image `Blob` plus timestamp and `stream_id`;
- `VisionPartial`: the complete current caption, OCR, or detector description, not a provider
  delta;
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

State and prefetch work remain independent per ID. Each scene retains only its latest frame as the
durable keyframe; this bounds memory without imposing a hidden video codec or sampler. Applications
that need a full clip may finalize an encoded video through `AsyncCaptureStream` instead.

With native image embedding, frame changes drive speculative retrieval directly. With a text-only
embedder, `VisionPartial` drives retrieval and its final value is attached to the keyframe through
`StreamInput.description`. If no partial exists, a configured `VisionDescriptionBackend` describes
the final frame once. If none of those routes exists, finalization fails instead of discarding the
frame. Camera SDK values and RGB/YUV conversion remain adapter work.

If final retrieval fails, the final observation still commits and `StreamCommit.retrieval_error`
contains a stable public code (or `retrieval_failed` for an unclassified exception). Consumer task
cancellation before final commit writes nothing. Once the exact-final write starts, `FINAL` is the
commit point: cancellation waits for the write and then propagates, so no worker can write after a
cancelled task has already returned.

Useful end-to-end measurements are:

- fraction of turns with a completed matching prefetch at finality;
- final-boundary-to-context latency at p50, p95, and p99;
- fraction of finals that require another search;
- searches started per turn and maximum concurrent searches per reducer;
- retrieval quality for the final snapshot, not just earlier partials;
- `CANCEL`, EOF, or pre-commit task cancellation that produced a durable record, whose target is
  zero.

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
and spatial frame, creates deterministic derived IDs, links evidence, versions state, embeds new
records, and records the durable formation recipe in one SQLite transaction. If formation fails,
the source remains durable and retrying the same add can complete the missing projection.

The typed roles are `OBSERVATION`, `ENTITY`, `EVENT`, `STATE`, `RELATION`, `AFFECT`, `TRAIT`, and
`RESPONSE_POLICY`. They refine the existing semantic, episodic, and procedural `MemoryType` roles;
they do not create eight stores.

## Affect and personality evidence

Affect is situated evidence. A proposal may carry cue modality, valence, arousal, confidence,
validity, spatial context, and source evidence. The cue modality must exist in the source
observation, so a text-only inference cannot claim to have heard vocal tension. Applications should
preserve text, prosody, face, posture, and environment cues separately and fuse them late.

A trait is a long-horizon claim and has a stricter visibility rule:

- a trusted explicit `USER_STATEMENT` is visible immediately;
- a `MODEL_INFERENCE` remains hidden from active search until two independent source observations
  support the same normalized subject/predicate/value claim;
- repeated extraction from one observation or shared `source_id` counts once;
- confidence uses the maximum within one source and noisy-OR across independent sources;
- deleting source evidence recomputes confidence and can hide or remove the trait;
- explicit corrections use state-like bitemporal replacement instead of overwriting history.

`get` and `list` may expose a hidden trait with `MemoryContext.visible=False` for audit. Ordinary
search and answer exclude it. This lets developers inspect why a profile did not activate without
allowing one uncertain emotional episode to personalize future behavior.

Response policies are typed procedural evidence, never executable code. They require explicit
feedback provenance before an application should use them to adapt companion behavior.

## Retrieval by valid, known, and spatial state

Use `RetrievalScope` to ask what was valid in the world, what MindBridge knew at the time, or what
was nearby in one coordinate frame:

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

Spatial search is same-frame and same-anchor only. Position uncertainty expands the conservative
intersection test. MindBridge does not guess coordinate transforms. `valid_at` is world time;
`known_at` is transaction time. The older `occurred_from`/`occurred_until` filters still select raw
event occurrence and remain separate.

## Graph evidence gate

No graph database is introduced by formation. Typed lineage and evidence links cover deterministic
state maintenance without another service. A graph candidate is justified only when public-path
benchmarks show that gold evidence is commonly retrieved at a larger K but displaced at the product
K, the failure spans independent memory units, and a prototype improves retrieval/answer metrics
without violating latency, storage, rebuild, and privacy budgets.

If that gate passes, start with an additive entity/relation projection derived from authoritative
SQLite records and rebuilt through the existing outbox lifecycle. It must not become a second source
of truth, create logical account scopes, or learn from abandoned speculative queries. See the
[competitive review](competitive-memory-systems.md) for the source audit and roadmap.
