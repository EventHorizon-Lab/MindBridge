# Configuration

MindBridge separates embedding, generation, and speech analysis. `Memory()` uses pinned Jina v5
Omni embedding and Fun-ASR-Nano speech locally by default; `Config` controls the OpenAI-compatible
generation/combined backend, media policy, and timeouts.

## Embedding backends

Install the default local runtime with `mindbridge[local]`. It contains Sentence Transformers,
FunASR, and the media decoders used by Jina and FunASR. The base package remains dependency-light
for a deployment that passes explicit cloud backends.

Choose a different standard Sentence Transformers model explicitly:

```python
from mindbridge import Memory, SentenceTransformersEmbedder

embedder = SentenceTransformersEmbedder.load(
    "Qwen/Qwen3-VL-Embedding-2B",
    revision="9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
    device="cuda",
    batch_size=1,
)

memory = Memory("./data/qwen", embedder=embedder)
```

`revision` must be a full immutable 40-character commit. `dimension` defaults to the model's
native value and may be shortened only to a dimension advertised by the loaded model as
Matryoshka-trained. `device` is passed to Sentence Transformers. `batch_size` applies to one
ordered `encode_query` or `encode_document` call; video workloads should start at one.

The generic adapter discovers text, image, video, and audio support with the model's `supports()`
method and emits only standard dict/message input. `JinaOmniEmbedder` is separate because the
pinned Jina model currently uses a provider-specific tuple contract and remote model code.
MindBridge restores Sentence Transformers' global methods after loading Jina so a Qwen instance
in the same process remains standard.

## Environment variables

`Memory()` calls `Config.from_environment()` when no `Config` is supplied.

### Shared defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Fallback credential for OpenAI-compatible model operations |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Fallback OpenAI-compatible base URL |
| `MINDBRIDGE_TIMEOUT_SECONDS` | `120` | HTTP timeout in seconds |
| `MINDBRIDGE_DECAY_HALF_LIFE_DAYS` | unset | Positive decay half-life; unset disables decay |
| `MINDBRIDGE_MEDIA_TRANSPORT` | `data` | Send model media as `data` URLs or local `file` URLs |
| `MINDBRIDGE_ALLOWED_URL_HOSTS` | empty | Comma-separated HTTPS hosts allowed for media ingestion |

An operation-specific key or URL overrides the shared value. Embedding values apply when an
explicit combined `OpenAIHTTP`/`ModelBackend` owns embedding; the default Jina adapter does not read
them.

| Operation | API key | Base URL |
| --- | --- | --- |
| Embedding | `MINDBRIDGE_EMBEDDING_API_KEY` | `MINDBRIDGE_EMBEDDING_BASE_URL` |
| Generation | `MINDBRIDGE_GENERATION_API_KEY` | `MINDBRIDGE_GENERATION_BASE_URL` |
| Transcription | `MINDBRIDGE_TRANSCRIPTION_API_KEY` | `MINDBRIDGE_TRANSCRIPTION_BASE_URL` |

Base URLs must be HTTP or HTTPS URLs without credentials, query, or fragment. MindBridge
normalizes each URL to one trailing `/v1`.

A key is not required merely to open a directory or call `get`, `list`, or `delete`. The first
operation that needs an unconfigured model credential raises `ModelError`.

Memory decay is a local retrieval setting and adds no model call. When enabled, MindBridge records
bounded access reinforcement in SQLite and applies a soft search-time factor; it never deletes or
filters a durable record. See
[memory types, temporal reasoning, and decay](memory-types-time-and-decay.md) for the formula and
side effects.

### OpenAI-compatible models and capabilities

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINDBRIDGE_EMBEDDING_MODEL` | `text-embedding-3-small` | Combined HTTP backend embedding model |
| `MINDBRIDGE_EMBEDDING_SPACE` | `<model>:<dimension>:l2-v1` | Combined HTTP backend vector-space ID |
| `MINDBRIDGE_EMBEDDING_DIMENSION` | `1536` | Combined HTTP backend vector length |
| `MINDBRIDGE_GENERATION_MODEL` | `gpt-5-mini` | Grounded answer model identifier |
| `MINDBRIDGE_TRANSCRIPTION_MODEL` | `whisper-1` | ASR model identifier |
| `MINDBRIDGE_TRANSCRIPTION_SPACE` | `<transcription-model>:asr-v1` | Stable ASR model/preprocessing recipe ID |
| `MINDBRIDGE_EMBEDDING_MODALITIES` | `text` | Combined HTTP backend embedding modalities |
| `MINDBRIDGE_GENERATION_MODALITIES` | `text` | Comma-separated accepted modalities |
| `MINDBRIDGE_TRANSCRIPTION_MODALITIES` | `audio` | Comma-separated accepted modalities |

The transcription variables configure `OpenAIHTTP` and apply when an explicit combined backend is
passed as `models=`. They do not replace the default local `FunASRTranscriber`; pass
`transcriber=` for that.

Capability values are `text`, `image`, `video`, and `audio`. `omni` is accepted as shorthand for
all four, but the stored configuration expands it into atomic modalities. Declare what the actual
endpoint accepts; MindBridge never infers capability from a model name.

For a combined HTTP backend that embeds and answers from text plus visual input, with audio handled
by ASR:

```bash
export MINDBRIDGE_EMBEDDING_MODALITIES=text,image,video
export MINDBRIDGE_GENERATION_MODALITIES=text,image,video
export MINDBRIDGE_TRANSCRIPTION_MODALITIES=audio
```

Audio is then transcribed when embedding or generation lacks native audio support. The transcript
is combined with retained image/video parts. If the embedding operation accepts audio, it receives
the native asset instead.

### Remote media allowlist

Remote ingestion is opt-in. With the default empty allowlist, every `URL` input is rejected.
List exact hostnames without schemes, paths, ports, or wildcards:

```bash
export MINDBRIDGE_ALLOWED_URL_HOSTS=media.example,cdn.example
```

MindBridge validates every redirect, requires HTTPS, rejects credentials and fragments, resolves
the hostname, and refuses a hop if any address is non-public. It connects to a verified public IP
from that result while retaining the hostname for HTTP and TLS SNI, then repeats the process after a
redirect. A concrete expected MIME matches response `Content-Type` exactly; a range such as
`video/*` matches only that family. An allowlisted hostname is a fetch permission, so keep the set
narrow.

## Programmatic configuration

Pass an immutable `Config` when generation/transcription settings should not come from process
state. The local embedder owns its own identity and capabilities:

```python
from mindbridge import Config, Memory, ModelCapabilities, Modality

config = Config(
    generation_api_key="generation-key",
    generation_base_url="https://generation.example.com/v1",
    generation_model="example-vlm",
    transcription_api_key="transcription-key",
    transcription_base_url="https://speech.example.com/v1",
    transcription_model="example-asr",
    transcription_space="example-asr-english-v1",
    capabilities=ModelCapabilities(
        embedding=frozenset({Modality.TEXT}),
        generation=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        transcription=frozenset({Modality.AUDIO}),
    ),
    allowed_url_hosts=frozenset({"media.example"}),
    timeout_seconds=30,
)

with Memory("./data/example", config=config) as memory:
    memory.add("Configuration is explicit.")
```

Keys are excluded from the `Config` representation. Avoid logging credentials or putting them in
base URLs.

## OpenAI-compatible and custom backends

The built-in backend calls:

| Operation | Route |
| --- | --- |
| Embedding | `POST /v1/embeddings` |
| Generation | `POST /v1/chat/completions` |
| Transcription | `POST /v1/audio/transcriptions` |

Generation requests ask the provider for an SSE stream and consume it into the existing completed
`AnswerResult`. This preserves the Python, REST, and MCP response contracts while allowing
benchmarks to distinguish provider time-to-first-content-token from full generation time. A
provider that returns a normal JSON completion remains supported, but has no observable TTFT.

It can point directly at OpenAI-compatible deployments, including a vLLM deployment when that
deployment exposes the required compatible routes and content-part shapes. To make that combined
backend own embedding as well as generation and transcription, pass it as `models` and omit
`embedder`:

```python
from mindbridge import Config, Memory, ModelCapabilities, Modality, OpenAIHTTP

config = Config(
    embedding_model="example-embedding",
    embedding_space="example-embedding-recipe-v1",
    embedding_dimension=1024,
    capabilities=ModelCapabilities(
        embedding=frozenset({Modality.TEXT}),
        generation=frozenset({Modality.TEXT}),
        transcription=frozenset({Modality.AUDIO}),
    ),
)
cloud = OpenAIHTTP(config)

with Memory("./data/cloud", config=config, models=cloud) as memory:
    memory.add("Cloud embedding is explicit.")
```

Another provider can implement the public `ModelBackend` protocol and pass one instance explicitly:

```python
from mindbridge import Config, Memory, ModelBackend

backend: ModelBackend = MyJinaOrHuggingFaceBackend()
config = Config(
    embedding_model=backend.embedding_model,
    embedding_space=backend.embedding_space,
    embedding_dimension=backend.embedding_dimension,
    transcription_space=backend.transcription_space,
    capabilities=backend.capabilities,
)

with Memory("./data/custom", config=config, models=backend) as memory:
    memory.add("The backend boundary is explicit.")
```

`ModelBackend` has four operations: `embed`, `answer`, `transcribe`, and `close`, plus immutable
embedding identity, `transcription_space`, and capability properties. There is no registry,
dynamic plugin loader, telemetry wrapper, or hidden fallback backend. A task-aware implementation receives
`EmbedTask.DOCUMENT` for memories and `EmbedTask.QUERY` for queries. `Memory` takes ownership of
the supplied instance and calls `close()` during shutdown. Calls may overlap across threads; a
custom backend must be thread-safe until `close()` begins. A standalone embedding implementation
uses the narrower `EmbeddingBackend` protocol and is passed as `embedder=`.

The default `FunASRTranscriber` is lazy, uses `device="auto"`, and keeps the portable `automodel`
engine. Pass `FunASRTranscriber(device="cpu")` or `FunASRTranscriber(device="cuda")` explicitly to
control placement. For CUDA batch throughput, pin a driver-compatible vLLM release and install
`mindbridge[local,vllm]`, then pass `FunASRTranscriber(engine="vllm", device="cuda")`. MindBridge
still composes FSMN-VAD and CAM++ around vLLM, preserving timed speaker identity. A custom
`SpeechBackend` supplies timed turns plus speaker centroids and is passed as `transcriber=`.
`speaker_similarity` and `speaker_margin` on `Memory` are the two CAM++ identity calibration
thresholds.

The `vllm` extra is isolated from `local` because it owns a CUDA-specific Torch stack. Select and
pin a compatible release using the
[FunASR vLLM guide](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md); ordinary local
installations do not download vLLM or CUDA wheels.

### Media transport and model-call size

With `MINDBRIDGE_MEDIA_TRANSPORT=data`, the built-in backend limits the aggregate raw bytes of media
in each embedding or generation call to 64 MiB before base64 expansion. For a batch embedding call,
the aggregate covers every submitted sample. For an answer call, one distinct asset is serialized
only once even when the question and multiple hits refer to it.

For `answer`, the built-in backend also limits the serialized question and grounding evidence to
4 MiB. That evidence includes each hit's content, `memory_type`, `occurred_at`, `created_at`,
metadata, and asset IDs. Lower `limit` or use a custom backend when a workload intentionally needs
a larger context.

Use `file` only with a trusted co-located model server that can read MindBridge's local `file:` URLs.
It avoids inline encoding and is appropriate for larger video. A custom `ModelBackend` can instead
implement provider-native file upload or streaming. The 64 MiB guard is a built-in `data` transport
boundary, not a promise that an upstream provider accepts that much media.

## Data directory

Choose storage explicitly at the call site:

```python
memory = Memory(data_dir="/var/lib/mindbridge/assistant")
```

The default is `.mindbridge` relative to the process working directory. The directory must be
writable and may have only one live owner. Different directories are independent; metadata cannot
partition a shared directory. On POSIX, opening the store sets the top-level directory mode to
`0700`, including when the directory already existed. Run MindBridge as the intended owner rather
than relying on group access to that path.

## Stored compatibility metadata

On first open, SQLite records:

- Embedding model identifier and explicit vector-space identifier.
- Transcription-space identifier.
- Embedding dimension.
- Zvec index recipe.

Later opens compare the active backend/configuration with these values. A mismatch fails at
startup to prevent mixed vector spaces. Local adapters derive `space_id` from adapter recipe,
model, immutable revision, effective dimension, normalization, and query/document semantics;
there is no manual override. Combined cloud backends must set a new `embedding_space` whenever any
of those inputs changes. Re-encode into a new directory instead of editing the stored ID. The same
rule applies to `transcription_space`: it identifies the ASR model and all
transcript-affecting preprocessing, language, prompting, or decoding choices. A directory fails fast if the
active value changes, because cached transcripts and add-time derived text must use one recipe.

The generation model, URL allowlist, media transport, and HTTP timeout can change without
invalidating stored embeddings or transcript caches.

## REST service configuration

`create_app` accepts a data directory and an optional inbound service key:

```python
from mindbridge.api import create_app

app = create_app(
    data_dir="/var/lib/mindbridge/assistant",
    api_key="service-secret",
)
```

The service key protects `/v1` routes and is distinct from outbound model credentials. `/healthz`
remains unauthenticated. See [deployment](deployment.md).
