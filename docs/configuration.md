# Configuration and composition

MindBridge has no product-wide provider configuration object. Construct model clients and adapters
where the application already manages dependencies and secrets.

## Memory settings

`Memory` accepts only local semantic and consistency settings:

```python
Memory(
    data_dir=".mindbridge",
    *,
    embedder=embedder,
    answerer=None,
    transcriber=None,
    face_analyzer=None,
    index_speech=False,
    index_quantization=IndexQuantization.NONE,
    minimum_relevance=0.55,
    ambiguity_margin=0.01,
    decay_half_life_days=None,
    speaker_similarity=0.78,
    speaker_margin=0.05,
    face_similarity=0.363,
    face_margin=0.05,
    tracer=None,
)
```

- `embedder` is required because its model, dimension, and vector-space recipe are durable data
  contracts.
- `answerer` is optional. `ask()` raises `ModelError` when it is absent.
- `transcriber` is optional. Configure it when audio fallback, add-time transcript indexing, or
  `speech()` is required. A `TranscriptionBackend` transcribes every stored asset whose modality it
  declares, so a media memory carries retrievable text even where the embedder accepts the media
  natively.
- `face_analyzer` is optional. Configure it when `faces()` or visual identity grounding is required.
- `index_speech` opts a configured `SpeechBackend` into add-time transcript and speaker-identity
  indexing; the default leaves analysis lazy.
- `index_quantization` defaults to quality-first `NONE`; `FP16`, rotated `INT8`, and x86_64-only
  `RABITQ` are explicit, lossy capacity choices that rebuild only the derived Zvec projection.
- `minimum_relevance` drops weak dense evidence; `ambiguity_margin` drops unresolved top-two ties
  unless the winner has a lexical or temporal anchor. Set either to `0` to disable that gate.
- `decay_half_life_days` controls query-time soft decay; `None` disables it.
- Face and speaker thresholds are local matching semantics, not provider settings. Each modality
  applies its own top-two margin before enrolling a new identity.
- `tracer` optionally injects an OpenTelemetry tracer; `None` uses the global no-op or
  application-configured provider.

## Provider clients

Use the provider SDK directly:

```python
from openai import OpenAI

from mindbridge import OpenAIModels

client = OpenAI(
    api_key="...",
    base_url="https://api.openai.com/v1",
    timeout=30.0,
    max_retries=3,
)
models = OpenAIModels(
    generation_client=client,
    generation_model="gpt-5-mini",
)
```

Provider-specific Chat Completions fields remain explicit. For example, an OpenAI-compatible Qwen
endpoint can bound output with `generation_max_tokens=512` and disable its thinking template with
`generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.

The SDK may read its own environment variables. MindBridge does not duplicate key lookup, URL
normalization, proxy support, retry policy, connection pooling, or sync/async conversion.

Separate SDK clients can be supplied for embedding, generation, and transcription when their
provider settings differ:

```python
models = OpenAIModels(
    embedding_client=embedding_client,
    generation_client=generation_client,
    transcription_client=transcription_client,
)
```

All clients remain caller-owned and must be closed by the caller.

## Explicit capabilities

Adapters declare atomic modalities for each operation. `OpenAIModels` defaults to text embedding,
text generation, and audio transcription. Override only when the selected model and endpoint
actually accept another modality:

```python
from mindbridge import Modality, OpenAIModels

models = OpenAIModels(
    generation_client=client,
    generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO}),
)
```

MindBridge compares the routed input with these declarations before inference. Unsupported audio
can be replaced with a transcript when a transcriber exists. Unsupported image or video evidence
is never silently discarded. Transcription is routed by `transcription_capabilities` rather than by
the gaps in the embedding or generation adapter, so a transcriber that declares video contributes a
video's speech on every path that reads a transcript.

## Durable spaces

Every embedding adapter exposes:

- `embedding_model`
- `embedding_space`
- `embedding_dimension`
- `embedding_capabilities`

Every transcription adapter exposes `transcription_model`, `transcription_space`, and
`transcription_capabilities`. MindBridge persists the embedding model, vector space, dimension,
transcription space, configured face spaces, and index recipe in SQLite. Opening the directory with
incompatible values fails immediately.

`reindex()` rebuilds Zvec from stored vectors. It does not re-embed or retranscribe content. Use a
new directory when changing a semantic recipe.

## Remote content

There is no URL allowlist or built-in downloader. This is deliberate: DNS policy, redirects,
credentials, retries, streaming, and SSRF controls belong to the application's HTTP stack. Pass
downloaded bytes as `Blob` or a local file as `Path`.

REST and MCP accept base64 data URLs and stored asset IDs. They reject HTTP(S) URLs and filesystem
paths at their trust boundaries.

## Local adapters

```python
from mindbridge import FunASRTranscriber, JinaOmniEmbedder, OpenCVFaceAnalyzer

embedder = JinaOmniEmbedder(device="cuda", batch_size=8)
speech = FunASRTranscriber(device="cuda")
faces = OpenCVFaceAnalyzer("./models/yunet.onnx", "./models/sface.onnx")
```

Jina delegates query/document inference to Sentence Transformers. FunASR delegates execution and
device selection to `funasr.AutoModel`. Face analysis delegates local decoding, YuNet detection,
alignment, and SFace encoding to OpenCV; model files are caller-provided and never downloaded by
MindBridge.

## Benchmark-only environment settings

The benchmark executable has a private `ModelConfig` for reproducible runs. It may read
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MINDBRIDGE_GENERATION_API_KEY`,
`MINDBRIDGE_GENERATION_BASE_URL`, `MINDBRIDGE_GENERATION_MODEL`,
`MINDBRIDGE_GENERATION_MODALITIES`, and `MINDBRIDGE_TIMEOUT_SECONDS`. These are benchmark harness
inputs, not Python SDK configuration.
