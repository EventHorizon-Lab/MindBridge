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
- Memory covers model payload preparation and one aggregate embedding batch.
- Network/model latency fits the product's responsiveness and offline requirements.

With image embedding capability configured, use an explicit local directory and one owner:

```python
from pathlib import Path

from mindbridge import Memory

with Memory("/var/lib/device-agent/memory") as memory:
    memory.add(["Calibration frame", Path("/var/lib/device-agent/capture/frame.png")])
```

The source capture can be removed after a successful add because MindBridge stores immutable bytes
under `assets/`. On POSIX, opening the store enforces `0700` on the top-level data directory.

## Models on or near the device

The default Jina embedder runs through Sentence Transformers on the device selected by
`JinaOmniEmbedder.load(device=...)`. A standard model such as Qwen3-VL uses
`SentenceTransformersEmbedder`. Fun-ASR-Nano, FSMN-VAD, and CAM++ run locally by default;
generation can run remotely, on a gateway, or locally through a compatible server such as vLLM.
The optional `face` extra runs InsightFace through an available ONNX Runtime CPU, CUDA, or
TensorRT provider. `buffalo_s` is the smallest built-in model-pack choice; keep `buffalo_l` when
camera quality makes detection accuracy more important than the lower-compute detector.

The built-in `data` transport caps aggregate raw media at 64 MiB per embedding or generation call
before base64 expansion. For larger video on a device, use `file` transport with a trusted local
model server that can read the CAS, or implement a custom backend with native file streaming.

For another embedding runtime, implement `EmbeddingBackend` and pass it through
`Memory(..., embedder=backend)`. A combined cloud implementation uses `models=backend` without an
explicit embedder. MindBridge does not ship a plugin registry, GPU scheduler, or quantization
policy.

Declare exact capabilities. A visual-language model without native audio can receive transcript
text plus retained image/video assets when the transcription operation supports the audio. If a
route is unavailable, MindBridge fails instead of discarding sensor data.
Keep the backend's `transcription_space` stable for the ASR model and transcript-affecting recipe in
one directory; changing it is a fail-fast compatibility event and requires a new directory.

## Capture and network boundaries

MindBridge accepts completed files and bytes; it does not own cameras, microphones, streaming
capture, liveness/anti-spoofing, or sensor drivers. It provides local face and voice identity with
optional names for completed media assets; the application still decides when capture is complete
and when cross-modal evidence is strong enough to merge two identities.

HTTPS media download is disabled until exact public hostnames are configured. Edge applications
usually prefer `Path` or `Blob` so ingestion does not depend on external storage. If URLs are
enabled, apply outbound firewall policy in addition to `MINDBRIDGE_ALLOWED_URL_HOSTS`.

Apply disk encryption, device access control, consent, and retention rules according to the
sensitivity of stored media, transcript text, metadata, and biometric embeddings. Separate
applications or benchmark jobs use separate data directories; metadata is not isolation.

The current retrieval unit remains one memory and one aggregate embedding. Face recognition samples
video at its configured rate, but retrieval does not create per-frame vectors. Budget and measure
that path; do not assume streaming tracking, audio segmentation, per-asset retrieval vectors, or
reranking.
