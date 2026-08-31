# Edge deployment

MindBridge uses the same embedded multimodal API on an edge device: one process owns SQLite, the
content-addressed media store, and Zvec. Text, image, video, and audio enter through `Memory`; the
library does not require a separate edge service.

## Device requirements

Verify these properties on the target hardware:

- Python 3.10 through 3.14 is available.
- A compatible Zvec wheel exists for the operating system and CPU architecture.
- The filesystem supports durable SQLite WAL, atomic rename, directory fsync, and advisory locks.
- Storage covers original media, authoritative FP32 embeddings, WAL, and the derived index.
- Memory covers model payload preparation and aggregate-plus-atomic embedding batches.
- Network/model latency fits the product's responsiveness and offline requirements.

With image embedding capability configured, use an explicit local directory and one owner:

```python
from pathlib import Path

from mindbridge import JinaOmniEmbedder, Memory

with Memory(
    "/var/lib/device-agent/memory",
    embedder=JinaOmniEmbedder(),
) as memory:
    memory.add(["Calibration frame", Path("/var/lib/device-agent/capture/frame.png")])
```

The source capture can be removed after a successful add because MindBridge stores immutable bytes
under `assets/`. On POSIX, opening the store enforces `0700` on the top-level data directory.

## Models on or near the device

The Jina embedder runs through Sentence Transformers on the device selected by
`JinaOmniEmbedder.load(device=...)`. A standard model such as Qwen3-VL uses
`SentenceTransformersEmbedder`. `FunASRTranscriber` delegates Fun-ASR-Nano, FSMN-VAD, and CAM++
execution to `funasr.AutoModel`. Generation runs through a separately supplied adapter and provider
client.

The OpenAI adapter inlines at most 20 MiB per base64-encoded media item and 64 MiB per embedding or
generation call, roughly 15 MiB per file and 48 MiB in aggregate on disk. For larger video on a
device, use a provider-specific adapter with its native upload or streaming mechanism.

For another embedding runtime, implement `EmbeddingBackend` and pass it through
`Memory(..., embedder=backend)`. Generation and transcription use independent `answerer=` and
`transcriber=` arguments. MindBridge does not ship a provider registry, GPU scheduler, or
quantization policy.

Declare exact capabilities. A visual-language model without native audio can receive transcript
text plus retained image/video assets when the transcription operation supports the audio. If a
route is unavailable, MindBridge fails instead of discarding sensor data.
Keep the backend's `transcription_space` stable for the ASR model and transcript-affecting recipe in
one directory; changing it is a fail-fast compatibility event and requires a new directory.

## Capture and network boundaries

MindBridge accepts a lazy stream of completed files, bytes, and ordered omni observations through
`Memory.add_stream`. It does not own cameras, microphones, capture reconnection, segmentation, or
sensor drivers. The application decides when each observation is complete and can attach its time
range with `StreamInput`; MindBridge makes it durable and searchable before pulling the next.
Mutable working snapshots may use `AsyncOmniPrefetch` without entering durable storage. Local
face-and-voice identity remains available for completed image/audio/video assets.

Edge applications usually prefer `Path` or `Blob` so ingestion does not depend on external
storage. Fetch remote media in application code, where the platform's existing HTTP client,
credential, firewall, and retry policy already apply.

Apply disk encryption, device access control, and retention rules according to the sensitivity of
stored media, transcript text, metadata, and embeddings. Separate applications or benchmark jobs
use separate data directories; metadata is not isolation.

The returned retrieval unit remains one memory; composite inputs add aggregate and atomic
text/media vectors, and long text adds overlapping contextual keys. All keys collapse by maximum
relevance and are bounded to 128 non-aggregate vectors per record. Budget for the extra vectors.
Do not assume audio/video segmentation, generated semantic keys, or learned reranking. The optional
OpenCV face adapter performs bounded video frame sampling only for face observations.
