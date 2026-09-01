# Configuration and composition

MindBridge offers two composition paths over the same memory kernel:

- `Memory.from_config` validates pure data and constructs bundled adapters.
- `Memory(...)` accepts already-constructed protocol implementations for custom models and
  caller-owned SDK clients.

Provider names exist only in the declarative composition layer. The kernel continues to depend on
typed capabilities rather than provider branches.

## Declarative composition

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/assistant",
    "embedding": {
        "provider": "openai",
        "model": "text-embedding-3-small",
    },
    "generation": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "modalities": ["text", "image"],
        "temperature": 0.1,
    },
    "speech": {
        "provider": "funasr",
        "device": "cuda",
    },
    "face": {
        "provider": "opencv",
        "detector_model": "./models/yunet.onnx",
        "recognizer_model": "./models/sface.onnx",
    },
    "settings": {
        "index_speech": True,
        "minimum_relevance": 0.55,
        "evidence_budget_chars": 16000,
    },
}

with Memory.from_config(config) as memory:
    memory.add("Remember this")
```

The same input can be validated ahead of time with `MindBridgeConfig.model_validate(config)`.
Unknown providers, unknown fields, invalid ranges, and missing required values raise a field-specific
error before storage opens. `AsyncMemory.from_config` accepts the same configuration.

Hosts that need constructed adapters separately from opening storage can call
`resolve_memory_config(config)`. It returns a `MemoryComposition`; the caller owns it and must call
`close()` unless it transfers the plugins to one `Memory`. Ordinary applications should keep using
`Memory.from_config` so ownership transfers automatically.

Supported declarative slots are intentionally small:

| Slot | Bundled providers |
| --- | --- |
| `embedding` | `openai`, `jina-omni`, `sentence-transformers` |
| `generation` | `openai` |
| `speech` | `openai`, `funasr` |
| `face` | `opencv` |

OpenAI slots accept `base_url`, `timeout`, and `max_retries`; the official SDK resolves its standard
credentials. Generation also accepts atomic `modalities` (`text`, `image`, `video`, and `audio`) so
request validation matches an OpenAI-compatible model's actual inputs. Use direct injection for
secret managers, existing clients, custom proxies, or third-party providers. MindBridge owns and
closes clients created by `from_config`.

### Multimodal embedding over an OpenAI-compatible server

`embedding` declares the same two facts about a self-hosted embedding model:

| Field | Default | Meaning |
| --- | --- | --- |
| `modalities` | `["text"]` | Atomic modalities the embedding model accepts. |
| `request_format` | `"input"` | `"input"` posts the standard `input` array; `"messages"` posts chat-style content parts, which is how OpenAI-compatible servers carry media. |

Routing reads the declaration, so it is what decides whether an image or video memory is stored or
rejected. Leave `modalities` at its default and every non-text write fails with
`unsupported_modality` — correctly, because nothing would have embedded the pixels. Audio is the one
modality with a documented fallback: when a `speech` backend is configured, audio is transcribed and
the text is embedded, so a model without audio support still accepts audio memories.

```python
config = {
    "embedding": {
        "provider": "openai",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "tencent/WeMM-Embedding-2B",
        "dimension": 2048,
        "modalities": ["text", "image", "video"],
        "request_format": "messages",
    },
    "speech": {"provider": "funasr", "device": "cuda"},
}
```

`embedding_space` records the request format, so switching `request_format` changes the durable
embedding identity and forces a rebuild instead of silently mixing two spaces.

## Memory settings

`MemorySettings` is the readable name for the value-only local policy; `MemoryConfig` remains a
compatible alias. The `settings` section accepts these values:

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
    evidence_budget_chars=None,
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
  indexing; the default leaves analysis lazy. Enable it whenever memories carry speech: without it
  a clip-shaped video memory's indexed document can be a source identifier and nothing more. See
  [the Python SDK reference](api/python-sdk.md) for the measured effect.
- `index_quantization` defaults to quality-first `NONE`; `FP16`, rotated `INT8`, and x86_64-only
  `RABITQ` are explicit, lossy capacity choices that rebuild only the derived Zvec projection.
- `minimum_relevance` drops weak dense evidence. `ambiguity_margin` drops unresolved top-two ties
  only when `search()` or `ask()` uses `limit=1`, unless the winner has a lexical or temporal anchor.
  Larger limits preserve qualified candidates for the caller or answerer. Set either to `0` to
  disable that gate.
- `evidence_budget_chars` widens what `ask()` grounds on. The `limit` hits are always kept; the
  budget then admits further ranked memories while the evidence fits, charging each media asset its
  modality's text equivalent because a media part costs a model far more than its record's text.
  `None` keeps grounding at exactly `limit`.
- `decay_half_life_days` controls query-time soft decay; `None` disables it.
- Face and speaker thresholds are local matching semantics, not provider settings. Each modality
  applies its own top-two margin before enrolling a new identity.
- `tracer` optionally injects an OpenTelemetry tracer; `None` uses the global no-op or
  application-configured provider.

## Explicit provider clients

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
`generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. If an endpoint also
requires a minimum video duration, set `generation_min_video_seconds` to that documented boundary;
shorter videos use four ordered stills when image input and the `openai` extra are available.
`generation_video_limit` defaults to eight distinct retrieved videos per answer; overflow videos
retain their text or transcript evidence. Set a positive integer to calibrate the limit, or `None`
only when the provider context and latency budget can safely accept every retrieved video.

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

### OpenAI Transcribe

Use the existing transcription plugin with OpenAI's completed-file Transcriptions API:

```python
from openai import OpenAI

from mindbridge import OpenAIModels

client = OpenAI()
transcriber = OpenAIModels(
    transcription_client=client,
    transcription_model="gpt-transcribe",
    transcription_prompt="A household conversation about reminders and medication.",
    transcription_keywords=("MindBridge", "AC-42"),
    transcription_languages=("en", "zh-cn"),
)
```

MindBridge sends each resolved audio asset to `/audio/transcriptions` as multipart form data and
returns the response text through `TranscriptionBackend`. `transcription_prompt` supplies
unstructured context; `transcription_keywords` and `transcription_languages` map to OpenAI's
`keywords[]` and `languages[]` fields. See the official
[file transcription guide](https://developers.openai.com/api/docs/guides/speech-to-text).

The `whisper-1` default remains unchanged for compatibility; select `gpt-transcribe` explicitly.
When no explicit `transcription_space` is supplied, MindBridge hashes prompt, keyword, and language
settings into the derived space so stored transcripts from different recipes cannot mix. The hash
does not expose the prompt or keywords in store metadata.

[vLLM multimodal pooling models](https://github.com/vllm-project/vllm/blob/main/examples/pooling/embed/vision_embedding_online.py)
extend `/embeddings` with top-level chat `messages`. Select that wire format explicitly so text
queries and media documents use the same model recipe:

```python
from openai import OpenAI

from mindbridge import Modality, OpenAIModels

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
embedder = OpenAIModels(
    embedding_client=client,
    embedding_model="openai/clip-vit-base-patch32",
    embedding_dimension=512,
    embedding_request_format="messages",
    embedding_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
)
```

MindBridge still uses the caller-owned official OpenAI SDK client; it does not add a provider
registry or its own HTTP transport. The `messages` format becomes part of the derived durable
embedding space, so changing formats against an existing `data_dir` fails rather than mixing
vectors from different input recipes. `embedding_dimension` declares the model's output size for
validation; MindBridge does not send it as vLLM's optional Matryoshka truncation control.

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

For transcript-only workloads, copy `DEFAULT_FUNASR_RECIPE` with `speaker_model=None` and
`speaker_revision=None`. This skips CAM++ and returns timed turns without speaker identities; the
benchmark runner uses that recipe because its tasks do not score speaker identity.

## Benchmark-only environment settings

The benchmark executable has a private `ModelConfig` for reproducible runs. It may read
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MINDBRIDGE_GENERATION_API_KEY`,
`MINDBRIDGE_GENERATION_BASE_URL`, `MINDBRIDGE_GENERATION_MODEL`,
`MINDBRIDGE_GENERATION_MODALITIES`, and `MINDBRIDGE_TIMEOUT_SECONDS`. These are benchmark harness
inputs, not Python SDK configuration. `mindbridge-bench eval --config memory.json` also accepts the
declarative schema above, replaces its `data_dir` for per-unit isolation, and records the effective
composition in the benchmark artifacts.
