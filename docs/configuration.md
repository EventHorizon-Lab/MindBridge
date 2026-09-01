# Configuration

MindBridge has two composition paths over the same `Memory` kernel:

- `Memory.from_config()` validates data and constructs bundled adapters.
- `Memory(...)` accepts constructed protocol implementations and SDK clients.

Use declarative configuration for bundled providers. Use direct injection when the application
owns clients, credentials, proxies, custom models, or adapter behavior.

## Install only the surfaces you use

The base package provides storage, public values, and adapter protocols. Optional extras add the
heavy or protocol-specific runtimes:

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

For example:

```bash
uv add "mindbridge[local,openai]"
```

## Declarative configuration

`embedding` is the only required capability. `data_dir` defaults to `.mindbridge`; all other
capability slots are optional.

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
    },
}

with Memory.from_config(config) as memory:
    memory.add("Remember this.")
```

`AsyncMemory.from_config()` accepts the same shape. `MindBridgeConfig.model_validate()` can
validate it before opening storage. Unknown fields, unknown providers, and out-of-range values are
rejected rather than ignored or coerced.

Bundled provider fields are:

| Slot and provider | Required fields | Optional fields and defaults |
| --- | --- | --- |
| `embedding: jina-omni` | — | `dimension=1024`, `device=None`, `batch_size=32` |
| `embedding: sentence-transformers` | `model`, `revision` | `dimension=None`, `device=None`, `batch_size=32` |
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `video_limit=8`, `extra_body=None`, plus connection fields |
| `speech: funasr` | — | `device=auto` |
| `speech: openai` | — | `model=whisper-1`, `space=None`, plus connection fields |
| `face: opencv` | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

Every OpenAI slot accepts `base_url=None`, `timeout=None`, and `max_retries=None`. The official SDK
then applies its own defaults and reads its standard credentials, including `OPENAI_API_KEY`.
Declarative configuration intentionally has no credential field; an `api_key` entry is rejected.
Inject a caller-owned SDK client when credentials must come from another source. Atomic generation
modalities are `text`, `image`, `video`, and `audio`.

Jina dimensions are `32`, `64`, `128`, `256`, `512`, or `1024`. Batch sizes and model token/video
limits are positive; OpenAI timeouts are positive, retry counts are non-negative, temperature is
from 0 through 2, and seed is from 0 through `2**63 - 1`. Face thresholds are from 0 through 1.
`generation.video_limit` caps retrieved evidence videos in one answer request; question media has
priority, and `None` disables that evidence-video count.

`resolve_memory_config()` is available to hosts that need a `MemoryComposition` before opening
storage. The caller must close that composition unless its plugins are transferred to one
`Memory`; ordinary applications should prefer `Memory.from_config()`.

## Embedding choices

The embedding model defines durable vector identity, so every memory requires one:

- `jina-omni` is the pinned multimodal default used in the examples. Its weights are licensed
  CC BY-NC 4.0 for non-commercial use. Loading it sets `trust_remote_code=True`, with both model
  and code revisions pinned to the same immutable upstream commit recorded by MindBridge. Review
  that upstream code as part of the application's dependency trust boundary.
- `sentence-transformers` loads a model and immutable 40-character commit revision selected by the
  application. Its supported modalities come from that model.
- `openai` uses the official SDK and defaults to text embedding. Use direct injection to declare
  additional capabilities or a non-standard request format.

Changing model, revision, dimension, or input recipe changes the embedding space. Do not point the
new configuration at an existing directory unless it is an explicitly supported bundled upgrade;
see [metadata mismatch](troubleshooting.md#store-metadata-mismatch).

## Local memory settings

The `settings` mapping is the value-only `MemorySettings` policy (`MemoryConfig` is a compatible
alias):

| Field | Default | Meaning |
| --- | --- | --- |
| `index_speech` | `False` | Persist configured speech analysis during `add` |
| `index_quantization` | `none` | Zvec projection mode: `none`, `fp16`, `int8`, or `rabitq` |
| `minimum_relevance` | `0.55` | Reject evidence below this confidence |
| `ambiguity_margin` | `0.01` | Withhold an unresolved top-two tie when `limit=1` |
| `decay_half_life_days` | `None` | Optional positive half-life for query-time decay |
| `speaker_similarity` | `0.78` | Voice identity match threshold |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |

Thresholds and margins accept values from `0` through `1`. `minimum_relevance=0` disables the
weak-evidence floor; a zero ambiguity margin disables the corresponding tie rejection. A zero
speaker or face similarity is merely the most permissive threshold. `decay_half_life_days` must be
positive when set. A tracer is passed as the separate `tracer=` argument to
`Memory.from_config()` or `Memory(...)`, not inside `settings`.

## Direct adapter injection

Construct adapters directly when the application must own an SDK client or use a custom protocol
implementation:

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

`Memory` closes the adapter objects passed to it. `OpenAIModels.close()` deliberately leaves a
caller-supplied SDK client open, so the outer client context remains the application's
responsibility. Credentials, retries, timeouts, proxies, and connection pooling also remain SDK
configuration.

Two capability slots are reachable only this way, or through `MemoryPlugins`; no declarative
provider selects either implicitly:

- `former=` takes a `FormationBackend`, which proposes typed memories after a raw observation
  commits. Configuring the declarative `generation` slot never enables automatic formation.
- `vision_describer=` takes a `VisionDescriptionBackend`. `AsyncVisionStream` calls it at
  finality only when the embedder lacks native image support and no external `VisionPartial`
  is available.

For every constructor and protocol field, see the [Python API](api/python-sdk.md). For benchmark
environment variables, see [benchmarking](benchmarking.md).
