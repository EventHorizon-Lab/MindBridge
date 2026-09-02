# Configuration

Configuration has one job: compose model capabilities and local policy around one `Memory`.
Storage and retrieval semantics stay the same whichever composition path you choose.

| Path | Choose it when |
| --- | --- |
| `Memory.from_config()` | Bundled providers and environment-based credentials are enough |
| `Memory.from_plugins()` | The application already groups constructed backends and settings |
| `Memory(...)` | The application injects individual backends or owns SDK clients directly |

**Contract:** All three paths open the same embedded kernel. `EmbeddingBackend` is required;
generation, speech, face, vision-description, and formation capabilities are optional.

## Install only the surfaces you use

| Extra | Adds |
| --- | --- |
| `local` | Jina and Sentence Transformers embedding, plus FunASR speech |
| `openai` | Official OpenAI SDK adapter and media handling |
| `face` | Local OpenCV face analysis |
| `server` | FastAPI and Uvicorn REST serving |
| `mcp` | MCP server transport |
| `observability` | OpenTelemetry SDK export support |
| `benchmarks` | Benchmark datasets, scorers, and telemetry |
| `all` | Every optional surface |

The base package already contains local storage, public values, backend protocols, and Zvec.

```bash
uv add "mindbridge[local,openai]"
```

## Declarative configuration

`Memory.from_config()` validates pure data and constructs adapters from a closed provider catalog.
`data_dir` defaults to `.mindbridge`; `embedding` is the only required slot.

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/assistant",
    "embedding": {
        "provider": "jina-omni",
        "device": "cuda",
        "batch_size": 16,
    },
    "generation": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "temperature": 0.1,
    },
    "settings": {
        "minimum_relevance": 0.55,
        "ambiguity_margin": 0.01,
        "evidence_budget_chars": 12000,
    },
}

with Memory.from_config(config) as memory:
    memory.add("Remember this.")
```

`AsyncMemory.from_config()` accepts the same shape. Use `MindBridgeConfig.model_validate()` when a
host needs validation before opening storage.

**Contract:** Unknown fields and providers are rejected. Numeric and Boolean values are strict;
invalid values are not silently coerced. Declarative configuration owns every adapter it creates
and closes it with the memory.

### Provider fields

| Slot and provider | Required | Optional defaults |
| --- | --- | --- |
| `embedding: jina-omni` | — | `dimension=1024`, `device=None`, `batch_size=32` |
| `embedding: sentence-transformers` | `model`, `revision` | `dimension=None`, `device=None`, `batch_size=32` |
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, `modalities=[text]`, `request_format=input`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `video_limit=8`, `extra_body=None`, plus connection fields |
| `speech: funasr` | — | `device=auto` |
| `speech: openai` | — | `model=whisper-1`, `space=None`, plus connection fields |
| `face: opencv` | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

Every OpenAI slot accepts `base_url=None`, `timeout=None`, and `max_retries=None`. The official SDK
applies its own defaults and reads standard credentials such as `OPENAI_API_KEY`. Declarative
configuration has no credential field, so `api_key` is rejected. Inject a caller-owned SDK client
when credentials or transport policy must come from another source.

Embedding and generation capability declarations contain only the atomic `text`, `image`, `video`,
and `audio` modalities. OpenAI embedding requires at least one declared modality. Positive fields
must be greater than zero; retry counts are non-negative; temperature is from 0 through 2; seed is
from 0 through `2**63 - 1`; face thresholds are from 0 through 1.

`generation.video_limit` caps retrieved evidence videos in one answer request. Question media has
priority, and `None` disables the evidence-video count.

## Embedding choices

The embedding recipe is durable identity, not a per-query preference:

- `jina-omni` is the pinned multimodal recipe used in project examples. Its weights are CC BY-NC
  4.0. Loading executes pinned upstream code with `trust_remote_code=True`; review that code and
  license as part of the application's trust boundary. Supported dimensions are `32`, `64`, `128`,
  `256`, `512`, and `1024`.
- `sentence-transformers` loads an application-selected model at a required immutable 40-character
  commit revision. The model declares its supported modalities.
- `openai` defaults to text embedding. For an OpenAI-compatible multimodal server, declare its
  `modalities` and choose `request_format=input` for the standard embeddings array or
  `request_format=messages` for chat-style media parts. The default embedding space includes that
  request recipe.

**Contract:** Changing a model, revision, dimension, or input recipe changes the vector space.
MindBridge refuses unrecognized store metadata mismatches instead of mixing spaces.

**Guidance:** Start a new data directory when intentionally changing embedding space. Reuse an
existing directory only for a supported bundled migration; see
[store metadata mismatch](troubleshooting.md#store-metadata-mismatch).

Audio has one fallback: a configured speech backend can transcribe it for a text-only embedder.
Unsupported image or video input fails instead of being stored without visual evidence.

## Local memory settings

The `settings` mapping validates as `MemorySettings`. `MemoryConfig` is the compatible public name
used by `Memory.from_plugins()`.

| Field | Default | Effect |
| --- | --- | --- |
| `index_speech` | `False` | Persist configured speech analysis during `add` |
| `index_quantization` | `none` | Zvec projection mode: `none`, `fp16`, `int8`, or `rabitq` |
| `minimum_relevance` | `0.55` | Reject evidence below this confidence |
| `ambiguity_margin` | `0.01` | Withhold an unresolved top-two tie when `limit=1` |
| `evidence_budget_chars` | `None` | Let `ask` admit evidence beyond `limit` while it fits this budget |
| `decay_half_life_days` | `None` | Apply optional query-time retention decay |
| `speaker_similarity` | `0.78` | Voice identity match threshold |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |
| `identity_link_min_assets` | `2` | Distinct assets required before voice and face identities merge |

Thresholds and margins accept values from 0 through 1. A zero relevance floor disables weak-hit
rejection; a zero ambiguity margin disables the corresponding tie rejection. Half-life and
evidence budget values must be positive when set, and `identity_link_min_assets` is positive.

`evidence_budget_chars=None` grounds `ask()` on exactly `limit` hits. A positive budget keeps those
hits, then admits more ranked evidence while it fits. Media is charged a modality-specific text
equivalent because its generation cost is not represented by record text length.

Pass an OpenTelemetry tracer through the separate `tracer=` argument; it is not a setting. See
[operations](operations.md#telemetry) for exporter setup and
[memory types, time, and decay](memory-types-time-and-decay.md#decay-and-reinforcement) for ranking
semantics.

## Direct adapter injection

Direct injection is the boundary for custom protocols, custom SDK clients, proxies, or credentials
that do not belong in declarative data:

```python
from openai import OpenAI

from mindbridge import JinaOmniEmbedder, Memory, OpenAIModels

with OpenAI(timeout=30.0, max_retries=3) as client:
    models = OpenAIModels(
        generation_client=client,
        generation_model="gpt-5-mini",
    )
    with Memory(
        "./data/assistant",
        embedder=JinaOmniEmbedder(),
        answerer=models,
    ) as memory:
        print(memory.ask("What should I remember?").answer)
```

`Memory` closes the adapters passed to it. `OpenAIModels.close()` leaves a caller-supplied OpenAI
client open, so the outer client context remains the application's responsibility.

`vision_describer=` and `former=` are direct-injection-only capability slots; no declarative
provider enables them implicitly. A `VisionDescriptionBackend` supplies final-frame text fallback.
A `FormationBackend` proposes typed memories after the source observation commits.

Use `MemoryPlugins` with `Memory.from_plugins()` when these constructed adapters should travel as
one value. `resolve_memory_config()` is for hosts that need a `MemoryComposition` before opening
storage; close the composition unless its plugins are transferred to a `Memory`. Ordinary
applications should prefer `Memory.from_config()`.

For every constructor and protocol field, use the [Python SDK reference](api/python-sdk.md). For
benchmark-only environment variables, use [benchmarking](benchmarking.md).
