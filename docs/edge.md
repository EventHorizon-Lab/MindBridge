# Edge deployment

The edge path is platform-neutral. It runs on NVIDIA Jetson, D-Robotics RDK, Rockchip RK,
Intel/OpenVINO x86, generic ARM hosts, and on workstations where the "edge" is a 4090 or 5090.
Only the capture backend and the inference runtime change; the observation timeline, identity
gates, and forget semantics are identical everywhere.

```bash
uv sync --extra edge
```

On Linux/Windows x86_64 and Apple Silicon with macOS 14+, `edge` includes the sync/storage stack
plus FunASR, Torch/TorchAudio, InsightFace, ONNX Runtime, and OpenCV. Linux ARM device images keep
the model wheels supplied by JetPack, RKNN, OpenVINO, or BPU. PyTorch 2.13 has no Intel macOS
wheel, so Intel Macs get the orchestration stack but must supply a compatible model runtime.

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
fragment; when its event gate closes, a speech backend performs VAD, quality ASR, punctuation,
diarization, and CAM++ centroid extraction.

`recognize_identities_in_av_segment()` combines that result with InsightFace and the optional
provider-neutral audiovisual active-speaker pipeline, then returns only cloud-safe intervals ready
for `enqueue_captured_media()`.

## Choosing a speech engine

`recognize_identities_in_av_segment()` takes any `SpeechAnalyzer` — the contract is timed speech
spans plus the speaker centroids those spans belong to, and nothing about which model or engine
produced them. `load_speech_analyzer()` picks one:

```python
from mindbridge.edge.identity_diarization import load_speech_analyzer

# Resolves by environment: CUDA with vLLM installed → vllm, every other platform → automodel.
speech = load_speech_analyzer(device="auto")

# Or name one.
speech = load_speech_analyzer(engine="vllm", device="cuda")
speech = load_speech_analyzer(engine="automodel", recipe="sensevoice")
```

| Engine | Runs on | Batching | Speaker turns bounded by |
| --- | --- | --- | --- |
| `automodel` | anywhere the `edge` extra installs | per VAD span | the recipe's segmentation |
| `vllm` | CUDA only | batched across VAD spans | CTC character alignment |

Both engines fill the whole contract, including CAM++ centroids, so *within one model* resolving
from the environment can only change throughput and span precision — never whether the device can
still answer who spoke. Across models it would change the transcript, so the recipe constrains the
choice too: the vLLM engine implements Fun-ASR-Nano's architecture only, and a host configured for
`paraformer` or `sensevoice` gets `automodel` however much CUDA it has. Naming
`engine="vllm"` for a model it cannot serve is refused rather than honoured with a different model.

Servability is declared on the recipe (`vllm_servable`), not inferred from the model id — whether
a checkpoint is that architecture does not follow from its name, so a local conversion of those
weights can opt in and a Paraformer checkpoint cannot be mistaken for one.

vLLM also has to be importable, not merely wanted: installing it is deliberate, so its presence is
the signal. A GPU host that never installed it stays on `automodel` rather than failing at load.
Naming `engine="vllm"` on a device without CUDA fails loudly — an explicit accelerator request is
not a suggestion here.

### AutoModel recipes

A recipe is the composition, not just the model id, because a FunASR model id alone does not say
whether the checkpoint predicts timestamps or punctuates its own output — and those answers decide
whether timed speech and speaker centroids can be produced at all:

| Recipe | Model | VAD | Punctuation | Speaker turns split on | vLLM |
| --- | --- | --- | --- | --- | --- |
| `fun-asr-nano` (default) | Fun-ASR-Nano | FSMN-VAD, 30 s ceiling | the model's own | VAD spans | yes |
| `paraformer` | SeACo-Paraformer (Mandarin) | FSMN-VAD | CT-Transformer | punctuation | no |
| `sensevoice` | SenseVoiceSmall | FSMN-VAD, 30 s ceiling | none — nothing to align it to | VAD spans | no |

Every recipe composes CAM++ for the centroid, because a voiceprint has nothing to match against
without it. A model MindBridge has not measured is enabled by declaring its own recipe, which is
refused if it drops VAD or the speaker model:

```python
speech = load_speech_analyzer(
    recipe=FunASRRecipe(model_id="iic/SenseVoiceSmall", vad_max_single_segment_ms=30_000),
)
```

The default recipe runs with `trust_remote_code=True`, which FunASR needs for Fun-ASR-Nano, and
an unset revision resolves upstream to `master`. Pin `revision=` once a deployment has measured a
checkpoint; it reaches both engines. The vLLM engine can only honour a pin on the ModelScope hub —
upstream's HuggingFace path ignores it — so that combination is refused rather than silently
unpinned.

### Streaming

Streaming is deliberately not part of a recipe or an engine choice. It is a separate load because
it is not a capability every FunASR model has — neither SenseVoice nor Fun-ASR-Nano publishes a
causal variant, so `FunASRStreamingTranscriber.load()` needs a checkpoint that does.

`device="auto"` selects an available accelerator. An explicit accelerator request **fails** rather
than silently using CPU — a silent CPU fallback turns a capacity problem into a latency mystery.

## Platform runtimes

Generic Linux/Windows x86_64 and Apple Silicon hosts get InsightFace/ONNX Runtime,
FunASR/ModelScope, and their Torch runtime from `edge`. The portable `onnxruntime` dependency
provides CPU face inference; CUDA or TensorRT face providers still come from a platform image.
JetPack, D-Robotics OpenExplorer, RKNN Toolkit, OpenVINO, and BPU images provide the builds
matching their platform instead.

The optional annotated active-speaker path also calls the system `ffmpeg` executable and needs a
build with `drawtext`, `libx264`, and AAC support. This is an operating-system dependency, not a
Python wheel; the normal platform capture image should provide it.

The lock resolves the same PyPI releases that a built wheel declares, but carries no JetPack,
RKNN, OpenVINO, or BPU wheel. ONNX is the default portable artifact. Compiled engines (TensorRT,
RKNN, OpenVINO IR, BPU `.bin`) are built and cached per platform and are never reused across device
images.

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
