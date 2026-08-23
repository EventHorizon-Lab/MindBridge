# Edge deployment

The edge path is platform-neutral. It runs on NVIDIA Jetson, D-Robotics RDK, Rockchip RK,
Intel/OpenVINO x86, generic ARM hosts, and on workstations where the "edge" is a 4090 or 5090.
Only the capture backend and the inference runtime change; the observation timeline, identity
gates, and forget semantics are identical everywhere.

```bash
uv sync --extra edge
```

## What the device owns

| Responsibility | Owner |
| --- | --- |
| Camera decoding, encoding, frame rate, resolution | Platform capture stack (GStreamer, FFmpeg, optionally DeepStream) |
| VAD, motion, and scene gating | Platform capture stack |
| Anonymous face and voice identity | MindBridge edge |
| Durable outbox and retry state | MindBridge edge |
| Recent-memory cache for offline recall | MindBridge edge |
| Deletion reconciliation | MindBridge edge |

MindBridge orchestrates upstream libraries and does not reimplement their networks. It adds no
cross-platform abstraction layer of its own, which is why a new platform costs a runtime swap
rather than a port.

## The privacy boundary

This is the part worth reading twice.

**Leaves the device:** anonymous identity IDs, time ranges, optional voice transcripts, identity
scope, normalized face bounding boxes, and the media the deployment chose to upload.

**Never leaves the device:** raw face and voice embeddings, and the device encryption key.

The local identity store normalizes and AES-256-GCM encrypts every bounded sample, and matches
only across equal model and dimension spaces. `device_identity_key` is exactly 32
bytes, loaded from the device TPM or a secret manager. AWS credentials and the MindBridge API key
are never written to SQLite.

Face boxes are **0..1 normalized** `(left, top, right, bottom)`, not pixels. A detector that
leaks pixel coordinates has shipped here before, and the failure is indirect: the annotator draws
name boxes off-screen and the downstream model quietly stops identifying people.

## Capture handoff

Capture and encode with whatever stack the platform provides. When `splitmuxsink` or the robot
capture supervisor closes a segment, hand the completed file to the durable boundary:

```python
from pathlib import Path

from mindbridge.core import IdentityKind, ModelReference, derive_observation_id
from mindbridge.edge import (
    LocalIdentitySample,
    SQLiteIdentityMemory,
    SQLiteObservationOutbox,
    enqueue_captured_media,
)

outbox = SQLiteObservationOutbox(Path("/var/lib/mindbridge/edge.db"))
observation_id = derive_observation_id(
    "tenant_01", "front_camera", "robot-boot-20260811T120000Z", 7
)

identities = SQLiteIdentityMemory(
    Path("/var/lib/mindbridge/edge.db"),
    device_id="front_camera",
    encryption_key=device_identity_key,
)
face = identities.recognize_and_remember(
    LocalIdentitySample(
        tenant_id="tenant_01",
        kind=IdentityKind.FACE,
        source_observation_id=observation_id,
        sample_id="face-track-7-1",
        embedding=insightface_embedding,
        model_reference=ModelReference(model_id="insightface/buffalo_l"),
    ),
    minimum_similarity=calibrated_face_threshold,
)

request = enqueue_captured_media(
    outbox,
    Path("/var/lib/mindbridge/media/segment-000007.mp4"),
    tenant_id="tenant_01",
    device_id="front_camera",
    boot_id="robot-boot-20260811T120000Z",
    sequence=7,
    bucket="mindbridge-media",
    occurred_at=segment_started_at,
    ended_at=segment_ended_at,
    observed_at=capture_completed_at,
    clock_offset_ms=estimated_clock_offset_ms,
    identity_observations=(face.to_observation_input(start_ms=120, end_ms=2810),),
)
```

The handoff computes the SHA-256 and size, derives a deterministic tenant-scoped object key and
idempotency key, then commits the request and the absolute local path to a mode-`0600`,
WAL-enabled SQLite outbox in one transaction.

`kind` defaults to `MediaKind.VIDEO` and accepts `MediaKind.AUDIO` for a microphone-only capture
or `MediaKind.IMAGE` for a still frame. The sensor follows from it, so audio-only segments record
against `SensorKind.MICROPHONE`. The `audio_path` sidecar applies only to video, and a still image
carries no `duration_ms`.

Modality routing downstream is driven entirely by the declared `kind`, which `MediaObjectInput`
cross-checks against the URI extension when the extension is recognized. Declaring one kind and
pointing at another container is refused at the boundary rather than surfacing later as a decode
failure in a worker.

## Native hot paths

The example above accepts an embedding from an existing robot vision stack. Native paths do not
need to reopen a completed media file — feed timestamped BGR frames and 16 kHz mono PCM16 chunks
directly:

```python
import asyncio

from mindbridge.edge.identity_diarization import FunASRStreamingTranscriber

streaming_asr = FunASRStreamingTranscriber.load(device="auto")

faces = await asyncio.to_thread(
    face_encoder.encode_frame,
    bgr_frame,
    timestamp_ms=frame_timestamp_ms,
    duration_ms=frame_duration_ms,
)
partial = await streaming_asr.push_pcm16(pcm16_chunk, is_final=is_last_audio_chunk)
```

The partial transcript is provisional. The platform capture stack still owns the bounded rolling
fragment; when its event gate closes, `FunASRSpeechPipeline` performs VAD, quality ASR,
punctuation, diarization, and CAM++ centroid extraction in one upstream call.

`recognize_identities_in_av_segment()` combines that result with InsightFace and the optional
provider-neutral audiovisual active-speaker pipeline, then returns only cloud-safe intervals ready
for `enqueue_captured_media()`.

`device="auto"` selects an available accelerator. An explicit accelerator request **fails** rather
than silently using CPU — a silent CPU fallback turns a capacity problem into a latency mystery.

## Platform runtimes

Install InsightFace/ONNX Runtime and FunASR/ModelScope from the device image matching the target
platform SDK — JetPack/CUDA, D-Robotics OpenExplorer, RKNN Toolkit, OpenVINO, or a plain CUDA/CPU
host.

The generic `uv.lock` intentionally pins **no** vendor accelerator wheel. ONNX is the default
portable artifact. Compiled engines (TensorRT, RKNN, OpenVINO IR, BPU `.bin`) are built and cached
per platform and are never reused across device images.

NeMo is not part of the current pipeline.

## Syncing

Drain a bounded batch with the standard Boto3 credential chain and the typed SDK:

```bash
export MINDBRIDGE_API_KEY=...
uv run --extra edge mindbridge edge sync \
  --database /var/lib/mindbridge/edge.db \
  --api-base-url https://memory.example.com \
  --bucket mindbridge-media \
  --region us-east-1 \
  --recent-retention-hours 24 \
  --limit 100
```

One shot by design. Use the robot service manager or a systemd timer for retry scheduling and
backoff — a daemon owning its own retry loop is a second scheduler to reason about, and the one
you already have restarts on failure.

```ini
[Unit]
Description=MindBridge edge sync
After=network-online.target

[Service]
Type=oneshot
Environment=MINDBRIDGE_API_KEY=...
ExecStart=/usr/local/bin/mindbridge edge sync \
  --database /var/lib/mindbridge/edge.db \
  --api-base-url https://memory.example.com \
  --bucket mindbridge-media
```

Paired with a `systemd.timer`, this gives you restart, backoff, and jitter without writing any of
it.

What a run does:

- A failed run keeps the row, its sanitized error code, and its attempt count.
- Once media has uploaded, later retries send only the idempotent observation metadata.
- A cloud receipt advances the per-boot sync watermark, records its processing job, and removes
  the outbox row **atomically**.

## Offline recall

Successful jobs cache their memories locally for `--recent-retention-hours`. Evidence still
present on the device uses an offline `file://` reference, so recall works with no network:

```python
from pathlib import Path

from mindbridge.edge import SQLiteRecentMemory

recent = SQLiteRecentMemory(Path("/var/lib/mindbridge/edge.db"))
memories = recent.list_memories("tenant_01")
```

Local media deletion stays an explicit rolling-cache policy. MindBridge does not decide when your
disk is full.

## Deletion reconciliation

Deletion has to reach a device that was offline when it was issued. `--tenant-id` pulls
tombstones before submitting:

```bash
mindbridge edge sync --tenant-id tenant_01 --tenant-id tenant_02 ...
```

Tombstones remove matching cache rows **before** the deletion cursor advances, so an interruption
mid-reconcile re-processes rather than skips. Forgetting an observation also removes the identity
samples learned from that source.

A stale or foreign cursor is rejected by the API rather than answered with an empty page,
specifically so a device can never read truncation as completion.

## Storage layout

One SQLite file holds three things: the observation outbox, the deletion inbox, and the recent
memory cache. Mode `0600`, WAL enabled.

Each store operation opens its own connection and closes it again, so a process that runs for weeks
holds no growing set of descriptors on this file. WAL is what makes that cheap and what lets a
reader run while a writer commits.

Put it on persistent storage that survives reboot — it is the durability boundary for everything
captured but not yet acknowledged. Losing it loses observations that the cloud never saw.

## Related

- [Configuration](configuration.md) — `MINDBRIDGE_API_KEY` and the AWS chain.
- [CLI](api/cli.md#mindbridge-edge-sync) — every flag.
- [Edge identity model selection](edge-identity-sota.md) — model choice and validation.
