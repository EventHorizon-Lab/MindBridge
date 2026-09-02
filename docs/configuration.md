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
        "evidence_budget_chars": 12000,
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
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, `modalities=[text]` (at least one), `request_format=input`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `video_limit=8`, `extra_body=None`, plus connection fields |
| `formation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `extra_body=None`, plus connection fields |
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

## Automatic memory formation

The `formation` slot is omitted by default and must stay omitted unless the deployment wants it.
Configuring it adds one LLM round-trip to every write, after the raw observation has already
committed. Derived memories are a union with their sources, never a replacement: the raw
observation stays exactly as it was written and stays retrievable on its own.

```python
config = {
    "embedding": {"provider": "jina-omni"},
    "formation": {
        "provider": "openai",
        "model": "gpt-5-mini",
        "modalities": ["text", "image"],
        "max_tokens": 4096,
    },
}
```

What that buys is the typed plane: entity, event, state, relation, affect, trait, and
response-policy memories, each carrying a validity interval, optional metric spatial pose, and
optional emotional valence and arousal, all linked back to the source memory as evidence.

Two fields decide whether it does anything at all:

- `modalities` is the formation capability set. An observation whose modalities are not covered is
  skipped silently and by design, so a text-only declaration means image and video sources never
  form anything. Declare what the endpoint actually accepts.
- `max_tokens` bounds a JSON response. Formation treats a truncated response as an error rather
  than parsing it partially, so too small a budget fails the whole batch.

The bundled adapter answers and forms with the same completion controls, so this slot repeats
them instead of borrowing the `generation` slot; formation usually wants its own model, token
budget, or endpoint, and configuring `generation` alone never enables formation. `video_limit` is
absent because it caps answer evidence only. Formation is idempotent per source memory and
formation recipe, so a model failure leaves the raw observation durable and the formation
retryable. See [memory types, time, and decay](memory-types-time-and-decay.md) for the typed
kinds and their visibility rules.

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
- `openai` uses the official SDK and defaults to text embedding. An OpenAI-compatible server may
  host a multimodal embedding model, so `modalities` declares which atomic modalities it accepts and
  `request_format` selects how media is carried: `input` posts the standard array, `messages` posts
  chat-style content parts. Routing reads that declaration, so it decides whether an image or video
  memory is stored or rejected — left at the default, every non-text write fails with
  `unsupported_modality`, correctly, because nothing would have embedded the pixels. Audio is the
  one modality with a fallback: with a `speech` backend configured it is transcribed and the text is
  embedded. `embedding_space` records the request format, so changing it forces a rebuild rather
  than silently mixing two spaces.

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
| `evidence_budget_chars` | `None` | Widen `ask` grounding while the evidence fits this budget; raises a floor, never a ceiling; `None` grounds on exactly `limit` |
| `decay_half_life_days` | `None` | Optional positive half-life for query-time decay |
| `reinforce_on_answer` | `true` | Count the evidence `ask()` cited, so retrieval favours it later |
| `speaker_similarity` | `0.78` | Voice identity match threshold |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |
| `identity_link_min_assets` | `2` | Distinct assets a voice-and-face pair must share before they merge |

Thresholds and margins accept values from `0` through `1`. `minimum_relevance=0` disables the
weak-evidence floor; a zero ambiguity margin disables the corresponding tie rejection. A zero
speaker or face similarity is merely the most permissive threshold. `decay_half_life_days` must be
positive when set. `evidence_budget_chars` keeps the `limit` hits unconditionally and then admits
further ranked memories while the evidence fits, charging each media asset its modality's text
equivalent because a media part costs a model far more than its record's text. Because the `limit`
hits are never dropped, this setting can only enlarge a prompt: to bound one, lower `limit` and
leave the budget at `None`. Setting it also removes `limit`'s effect on prompt size, since the
budget refills the window to the same width whatever `limit` was. A tracer is passed
as the separate `tracer=` argument to
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

The `formation` slot is also reachable from the command line as `--former NAME`, which fills it
from the same recipe table as `--embedder` and `--answerer`.

One capability slot is reachable only this way, or through `MemoryPlugins`; no declarative
provider selects it implicitly:

- `vision_describer=` takes a `VisionDescriptionBackend`. `AsyncVisionStream` calls it at
  finality only when the embedder lacks native image support and no external `VisionPartial`
  is available.

`former=` takes a `FormationBackend` directly for a custom or non-bundled adapter; the bundled
OpenAI adapter is reachable through the `formation` slot above.

For every constructor and protocol field, see the [Python API](api/python-sdk.md). For benchmark
environment variables, see [benchmarking](benchmarking.md).
