# Configuration

Configuration composes model capabilities and local policy around one `Memory`. Storage and
retrieval semantics stay the same whichever path you choose.

| Path | Choose it when |
| --- | --- |
| `Memory.from_config()` | Bundled providers and pure-data configuration are enough |
| `Memory.from_plugins()` | The application already groups constructed backends and settings |
| `Memory(...)` | The application injects individual backends or owns SDK clients directly |

**Contract:** All three paths open the same embedded kernel. `EmbeddingBackend` is required;
generation, speech, face, vision-description, formation, and consolidation capabilities are
optional.

## Install only the surfaces you use

The base package contains local storage, public values, backend protocols, and Zvec. Extras add
model runtimes and transports:

| Extra | Adds |
| --- | --- |
| `local` | Jina and Sentence Transformers embedding, plus FunASR speech |
| `openai` | Official OpenAI SDK 3.x adapter, its `httpx2` transport, and media handling |
| `face` | Local OpenCV face analysis |
| `server` | FastAPI and Starlette REST adapter; bring your own ASGI server |
| `mcp` | MCP server transport |
| `observability` | OpenTelemetry SDK export support |
| `benchmarks` | Benchmark download, parsing, scoring, and telemetry dependencies; datasets are downloaded separately |
| `all` | The exact union of every optional surface |

For example:

```bash
uv add "mindbridge[local,openai]"
```

## Declarative configuration

`Memory.from_config()` validates pure data and constructs adapters from a closed provider
catalog. `data_dir` defaults to `.mindbridge`; `embedding` is the only required slot.

```python
from mindbridge import Memory

config = {
    "data_dir": "./data/assistant",
    "embedding": {
        "provider": "sentence-transformers",
        "model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
        "device": "cuda",
        "batch_size": 16,
    },
    "generation": {
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07",
        "response_format": "json_schema",
    },
    "settings": {
        "minimum_relevance": 0.10,
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
invalid values are not silently coerced. Declarative composition owns and closes every adapter it
creates.

### Loading a configuration file

`Memory.from_config()` accepts a `MindBridgeConfig` or a mapping, not a filename. File decoding
belongs to the host. JSON needs only the Python standard library:

```python
import json
from pathlib import Path

from mindbridge import Memory

config = json.loads(Path("config.json").read_text(encoding="utf-8"))
with Memory.from_config(config) as memory:
    print(memory.capabilities.embedding_model)
```

For YAML, install a YAML parser in the host and decode with `yaml.safe_load()` before passing
the mapping. MindBridge's product configuration does not depend on a YAML parser. The benchmark
runner's `--config` flag handles YAML itself through the `benchmarks` extra.

### Provider fields

In this table, connection fields are `base_url`, `api_key`, `timeout`, and `max_retries`:

| Slot and provider | Required | Optional defaults |
| --- | --- | --- |
| `embedding: sentence-transformers` | `model` | `revision=None`, `dimension=None`, `device=None`, `batch_size=32` |
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, `modalities=[text]` (at least one), `request_format=input`, `image_max_pixels=None`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini-2025-08-07`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `response_format=json_schema`, `store=False`, `stream=False`, `answer_style=default`, `video_limit=8`, `min_video_seconds=None`, `image_max_pixels=None`, `offline_gazetteer=None`, `extra_body=None`, plus connection fields |
| `formation: openai` | — | `model=gpt-5-mini-2025-08-07`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `response_format=json_schema`, `store=False`, `stream=False`, `image_max_pixels=None`, `extra_body=None`, plus connection fields |
| `vision: openai` | — | `model=gpt-5-mini-2025-08-07`, `modalities=[image]` (image and/or video only), `video_frames=4` (2–64), `image_max_pixels=None`, `temperature=None`, `seed=None`, `max_tokens=None`, `response_format=json_schema`, `store=False`, `stream=False`, `extra_body=None`, plus connection fields |
| `consolidation: openai` | — | `model=gpt-5-mini-2025-08-07`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `response_format=json_schema`, `store=False`, `stream=False`, `extra_body=None`, plus connection fields |
| `speech: funasr` | — | `device=auto`, `ncpu=4`, `vad_device=None`, `speaker_device=None` |
| `speech: openai` | — | `model=whisper-1`, `space=None`, plus connection fields |
| `face: opencv` | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

If `api_key` is unset, the official SDK uses its standard credential lookup, including
`OPENAI_API_KEY`. Each OpenAI slot builds a separate client and can instead receive its own
`api_key`, endpoint, timeout, and retry policy. `api_key` is a Pydantic `SecretStr`, so dumps and
representations mask it, but a configuration file still contains a secret. Prefer environment
lookup or a caller-owned client when the file could be committed or shared.
Every `base_url` must be an absolute `http://` or `https://` URL; a bare host name is rejected
during configuration validation rather than at the first model request.

Use an SDK base URL ending in `/v1`, not the complete `/chat/completions` URL: the SDK appends the
operation path. The default `response_format="json_schema"` sends a strict JSON Schema for formation,
consolidation, visual descriptions, media diagnosis, and recall planning. Free-text answers remain
text. A compatible endpoint that supports only JSON mode must explicitly select
`response_format="json_object"`; this guarantees JSON syntax, not response shape or meaning.
Both modes retain local validation, and a strict schema is never silently downgraded.

Every chat-completion request, including streaming and structural calls, explicitly sends the
configured storage policy, defaulting to `store=False`. Set the slot's strict boolean `store`
field to `True` only when provider-side
application-state storage is intended. Direct callers use `generation_store`. This setting does
not disable provider abuse-monitoring retention or establish Zero Data Retention; review the
provider's account-level [data controls](https://developers.openai.com/api/docs/guides/your-data).
Local deletion and retention cannot erase copies already held by a model provider.

The bundled generation default pins the previously evaluated snapshot rather than a moving alias.
Explicit model names remain caller-owned: pin a compatible model's deployment revision and
re-evaluate it before changing that deployment. Returned model identity and request IDs remain
available on traces. Changing the default model or output protocol changes formation, vision and
consolidation recipe identities; replay source data into a fresh directory when those stored
representations were built with the previous defaults.

Sampling controls are model-dependent. Known reasoning models do not receive incompatible automatic
`temperature` or `seed` settings; explicitly configuring unsupported controls raises
`ValidationError` before inference. The public `max_tokens` field remains the output budget and is
sent as `max_completion_tokens` for reasoning models, including their reasoning tokens; other
compatible models retain `max_tokens`. Leave headroom for reasoning when budgeting visible
answers. Model snapshots make quality comparisons reproducible; changing a model or prompt still
requires the [quality gate](operations.md#model-quality-gate).

`generation.answer_style` accepts `default` or `concise`. The default keeps the existing reader
prompt unchanged. `concise` asks the reader to return only the entities, relations, values or list
items the question requests, without explanations, citations or introductory phrases, and to
honor a requested output format. It still resolves relative times internally, but includes dates
or durations only when asked for them. The evidence policy and insufficient-evidence marker keep
their existing meanings. This is an opt-in presentation policy, not an established quality
improvement; evaluate it with the chosen model and task. Formation, visual description and
consolidation do not accept this answer-only control.

`generation.offline_gazetteer` optionally supplies external-world map knowledge while answering.
It is disabled by default and requires the `gazetteer` extra in addition to `openai`:

```yaml
generation:
  provider: openai
  model: your-vision-model
  modalities: [text, image]
  offline_gazetteer:
    index_path: /absolute/path/named-geometries.sqlite3
    index_sha256: "<complete lowercase SHA256>"
    radius_m: 500
    limit: 8
    budget_chars: 16000
```

The path and complete index SHA are required. Radius must be positive and at most 500 metres,
candidate limit is 1–8, and the additional serialized-character budget must be a positive integer.
The defaults are 500, 8 and 16000. This accepts the explicitly identified named-geometry v1
single-region or v2 multi-region SQLite formats described in the
[Python SDK](api/python-sdk.md#offline-gazetteer), with the same configuration fields. Every
source's URL, source-file SHA, snapshot date, region bounds and OpenStreetMap attribution/license
are mandatory and fixed by
the index SHA. Supply a complete, checkpointed main SQLite snapshot. The adapter verifies that
file once and opens it read-only with SQLite's immutable option; adjacent WAL/SHM files are not
read or included in the input. Keep the main file immutable: replacement or changes to its size
or modification time reject further lookups, including cached ones. It does not download or
build indexes.
Missing geometry dependencies or invalid provenance fail configuration before a model request.
V2 validates each feature's source index and rejects duplicate OSM identities. Lookups filter
by each source's own bounds and apply one distance order and candidate limit across eligible
regions. Answer metadata identifies the admitted sources individually through `map_sources`;
headers and repeated metadata occurrences count toward the same global character budget.
V1 request fields are unchanged, and the feature remains disabled by default.

Only original image GPS that passes the existing asset-integrity check is used. Nearby named
nodes, roads, waterways and areas are untrusted candidates; real line/polygon distances and
polygon holes determine proximity. A covering or nearest feature does not identify the photo's
subject, a mentioned place or a street address. Address tags describe the map feature only.
Map snapshot time does not establish historical presence at capture time, and rounded distances
do not express GPS accuracy. Outside-region and empty results do not prove that no place exists.

Candidates are added beside the attachment's original `untrusted_source_metadata`; native pixels,
GPS, capture time, source IDs and caller metadata retain their values. Only attachments actually
retained for the answer receive map context. Whole candidate prefixes fit the single answer-wide
budget in attachment order; repeated per-memory/query metadata occurrences are all charged.
No name or address is truncated to fit. Omitted candidates are counted in scalar tracing.
Geometry lookup is local and sends no coordinates over a network; the answer provider receives
the selected context alongside the existing source metadata.

This answer-only resource does not change `place_id`, durable records, embeddings, visual
descriptions, formation or retrieval. Existing stores need no replay. Comparisons with it enabled
must explicitly report the additional map source/index condition; it is not a memory-only baseline.

The OpenAI embedding, generation, formation, and vision slots accept `image_max_pixels`, an integer from
`1` through `64_000_000`, default `None`. When enabled, the adapter checks image headers before
decoding and rejects sources above 64 million pixels without changing Pillow's global limit.
Images exceeding the configured pixel bound or the 20 MiB inline item bound are EXIF-oriented,
converted to RGB JPEG at quality 85 with 4:2:0 subsampling, and resized proportionally with
LANCZOS. If necessary, the pixel budget is repeatedly halved until the encoded image fits.
Images already within both bounds retain their original request bytes. Request fitting counts
the actual outbound representation, including its base64 size; the 64 MiB aggregate bound still
applies. Resizing can reduce OCR and fine-detail accuracy, so choose a bound for the task.

Conversion is cached only within a model request. Stored files, asset IDs, digests, and record
references remain unchanged. An undecodable or oversized image fails embedding, formation, or description;
answer fitting can omit it from retrieved evidence, but a required question attachment raises an
error. Missing or changed source files still raise integrity errors. The pixel setting joins the
embedding-space identity even when `space` is explicit, and joins the visual-description and formation recipes.
Enabling or changing it for a persisted capability requires a fresh data directory and replay;
`None` preserves the previous identity.

The adapter recognizes GPT-5/GPT-6 reasoning names and snapshots and the `o1`, `o3`, and `o4`
families. GPT-5 Chat names and unrelated compatible model names retain generic sampling controls.
Later reasoning models that support non-reasoning requests may use temperature with
`extra_body={"reasoning_effort": "none"}`; the original GPT-5 family, GPT-6 Astra, and GPT-6.1 Sol
cannot use this combination. GPT-5.1 and GPT-5.2 aliases and dated snapshots default to `none`, so
temperature also works without explicitly setting effort on those models. Seeds are rejected for
known reasoning models. `extra_body` cannot
override the model, messages, response format, streaming controls, or storage policy; configure
their named options.

Atomic capability declarations contain only `text`, `image`, `video`, and `audio`. OpenAI
embedding requires at least one modality. Jina dimensions are `32`, `64`, `128`, `256`, `512`, or
`1024`. Positive fields must be greater than zero; retry counts are non-negative; temperature is
from 0 through 2; seed is from 0 through `2**63 - 1`; face thresholds are from 0 through 1.

`generation.video_limit` caps retrieved evidence videos in one answer request; question media has
priority, and `None` disables that count. `generation.min_video_seconds` sends a shorter video as
four ordered stills and requires image support. When the provider rejects an answer request by
declaring a technical duration constraint such as a video "too short", without a different
semantic error code, the request is sent once more with that modality's evidence media omitted
and the omission declared in the prompt. Content-safety refusals, including recognized safety
codes and compatible gateways' "inappropriate content" inspection errors, instead raise
`ModelError(reason="model_refused")` without removing media or retrying. Explicit semantic error
codes take precedence over message matching. Formation and consolidation have neither field
because they shape answer evidence, not proposals.

Answer evidence declares each record's original `source_modality`: `text`, `image`, `video`,
`audio`, or `omni` for a record containing multiple media kinds. Derived text records remain
`text`; this field does not infer the types of their supporting sources. Each attached question
or memory asset also has a `media_modalities` entry keyed by its numbered attachment alias.
These entries describe the original asset types actually supplied, so video frames remain
identified as video sources, with an ordered-still sampling label. Omitted media has no attachment
entry and is reported separately; a video record retaining only its caption still declares its
original video source. Attachment aliases, ordering, stored assets, and media limits are unchanged.

`generation.media_policy="on_demand"` enables one text-only diagnostic call before each answer
that has eligible memory media. It selects original attachments only for a missing detail or
conflicting descriptions. `generation.media_max_items` defaults to `2` and accepts integers from
`1` to `8`; the existing byte and video limits still apply. At most 64 candidate attachments are
offered to diagnosis. Question attachments remain intact, and all retained memory text and
provenance remain available. Invalid or failed diagnosis adds no media; omission metadata remains
in the answer request. A provider media rejection does not trigger another diagnosis. The default
`media_policy="all"` preserves existing generation. Diagnosis has its own model span and request
and token costs. SDK construction uses `generation_media_policy` and `generation_media_max_items`
on `OpenAIModels`. This mechanism has local tests, not benchmark evidence of accuracy or cost gains.

`vision` takes the same completion fields as `formation`, but its `modalities` accepts only `image`
and `video`, because it is the visual capability set rather than a generation one. `generation`,
`formation`, `vision`, and `consolidation` are separate slots and separate clients: setting one
never enables another. See [visual descriptions](#visual-descriptions) for the write-path cost of
the vision slot.

## Embedding choices

The embedding model defines durable vector identity, so every memory requires one:

- `sentence-transformers` loads an application-selected model. An optional `revision` pins a
  40-character immutable commit hash. When omitted, MindBridge resolves the current Hub revision
  before loading and retains that commit in the vector-space identity; Jina uses its bundled pin.
  Resolution requires Hub access. Specify a commit for reproducible or cached offline loading.
  The model declares its supported modalities. The pinned Jina Omni checkpoint used in project
  examples supports text, images, audio, and video, with 1024 dimensions by default. Its weights
  are CC BY-NC 4.0 and loading executes pinned upstream code with `trust_remote_code=True`;
  review that code and license as part of the application's trust boundary.
- `openai` defaults to text embedding. For an OpenAI-compatible multimodal server, declare its
  `modalities` and use `request_format=input` for the embeddings array or
  `request_format=messages` for chat-style media parts. The request format is part of the embedding
  space.

Audio has one fallback: a configured speech backend can transcribe it for a text-only embedder.
Unsupported image or video input fails instead of being stored without visual evidence.

**Contract:** Changing a model, revision, dimension, or input recipe changes the vector space.
MindBridge refuses store metadata mismatches instead of mixing spaces.

**Guidance:** Start a new data directory when intentionally changing embedding space; see
[store metadata mismatch](troubleshooting.md#store-metadata-mismatch).

### Pinned Jina checkpoint

Use `SentenceTransformersEmbedder` or `provider: sentence-transformers` with
`model: jinaai/jina-embeddings-v5-omni-small-retrieval`, as in the example above. The bundled pin is
`e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4`. Only this Jina revision is supported; another revision
fails validation before loading remote code.

Without a caller-supplied encoder, the constructor and declarative configuration keep this pinned
Jina checkpoint lazy. `SentenceTransformersEmbedder.load(...)` loads it immediately. Other models
load immediately and continue to use `trust_remote_code=False`. The Jina input recipe, embedding
space ID are fixed by the checkpoint and adapter recipe. A current-recipe Jina directory
reopens without re-embedding. The CLI `--embedder jina-omni` selects a model recipe that constructs
the same Sentence Transformers backend.

### EmbeddingGemma 2

Install `mindbridge[local]` and select
[Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) through the existing
Sentence Transformers provider:

```json
{
  "data_dir": "./data/embeddinggemma2",
  "embedding": {
    "provider": "sentence-transformers",
    "model": "google/embeddinggemma-2",
    "revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
    "dimension": 768,
    "device": "cpu",
    "batch_size": 8
  }
}
```

The local extra requires Sentence Transformers 6.1 or newer and Transformers 5.18 or newer.
The adapter loads this model in FP32; `device` can also be `cuda`. Queries and documents use the
checkpoint's separate retrieval prompts for text automatically; media alone receives no prefix.
Video decoding uses the existing PyAV dependency and preserves the model's frame sampling.
This text-only prompt recipe has a distinct space ID from generic ST batch prompting.
Text, images, audio, video, and ordered
combinations use the same embedding space. Supported dimensions are `128`, `256`, `512`, and
`768`; shortened vectors are normalized again. The model has an 8,192-token shared context;
split larger inputs before embedding them.

Applications injecting their own Sentence Transformer encoder must use FP32 or BF16, never FP16.
This checkpoint uses Apache 2.0 weights and loads with `trust_remote_code=False`. It supplies
embeddings only; generation, transcription, speaker recognition, and face recognition remain
separate backends. Use a new data directory when switching from another embedding model.

#### EmbeddingGemma 2 through vLLM

Install `mindbridge[openai]` on the application side and deploy vLLM separately using the
[serving instructions](deployment.md#embeddinggemma-2-model-server). Select its OpenAI-compatible
embeddings endpoint with the existing `openai` provider:

```json
{
  "data_dir": "./data/embeddinggemma2-vllm",
  "embedding": {
    "provider": "openai",
    "model": "google/embeddinggemma-2",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "EMPTY",
    "dimension": 768,
    "modalities": ["text", "image", "audio", "video"],
    "request_format": "messages"
  }
}
```

Use the exact model name `google/embeddinggemma-2`: in messages mode, MindBridge recognizes this
name and prefixes query text with `"task: search result | query: "` and document text with
`"title: none | text: "`. Pure media receives no prefix; mixed inputs preserve text and attachment
order. The request enables the checkpoint's BOS/EOS special tokens and sends inline data URLs
for images, audio, and video. Server aliases do not enable this model-specific recipe.

The dimension above validates the native 768-dimensional response; messages mode does not request
Matryoshka truncation. Keep the checkpoint's own chat template on the server. This recipe has its
own vector-space identity, separate from generic messages requests and the local Sentence
Transformers recipe. Use a fresh data directory when switching; existing unprompted EmbeddingGemma
2 messages stores fail the metadata check. A custom `space` overrides that identity and must
describe the actual server checkpoint and input recipe. Pin the server's model revision for
reproducible deployment. HTTP fixture tests verify request construction and SDK wiring; live vLLM
numerical parity has not been tested in this checkout.

## Local speech with FunASR

The `local` and `all` extras install FunASR from the immutable main-branch commit
[`940b0d2463141a2e9ace8a9fd965b4aea93f2db5`](https://github.com/modelscope/FunASR/commit/940b0d2463141a2e9ace8a9fd965b4aea93f2db5).
Git is required when installing these extras. This includes the upstream speaker-overlap,
CAM++ clustering, Nano punctuation-timestamp, and resampled speaker-input fixes that are absent
from the 1.4.16 PyPI release. The default model recipe remains pinned Fun-ASR-Nano, FSMN-VAD,
and CAM++; Nano covers Chinese, English, and Japanese, not the separate MLT checkpoint's
31-language contract.

```yaml
speech:
  provider: funasr
  device: cuda:0
  ncpu: 2
  vad_device: cpu
  speaker_device: cpu
```

`device` accepts `auto`, `cpu`, `cuda`, or `cuda:<index>`. With `auto`, FunASR chooses its
default accelerator and falls back to CPU if unavailable. `vad_device` and `speaker_device`
accept `cpu`, `cuda`, or `cuda:<index>`; `None` inherits the resolved ASR device. Explicit
submodel placement preserves the pinned VAD segment limit. Placement is selected when the
models first load and does not change during inference. CPU submodels can reduce GPU memory
use; measure latency on the target hardware before selecting them.

`ncpu` is a strictly positive integer. Upstream applies it to both PyTorch and BLAS for the
entire process when loading models; another `AutoModel` can replace that budget. Embedding
and other numerical work in the same process share these limits. It does not enable concurrent
calls to one transcriber: MindBridge serializes inference over its loaded FunASR pipeline.

The transcription-space identity includes the pinned FunASR source revision and the complete
model recipe. Device placement and CPU budgets do not create another semantic space. This
revision replaces the previous speech space: opening an old directory with the new transcriber
fails with a transcription-space metadata mismatch. Allocate a fresh `data_dir` and replay the
original inputs; old speaker assignments are not reused or migrated.

## Automatic memory formation

The optional `formation` slot builds the bundled OpenAI-compatible `FormationBackend`. It proposes
entity, event, state, relation, affect, trait, and response-policy records after the source
observation commits. Derived records are additive: the raw source stays durable and independently
retrievable.

```python
from mindbridge import Memory

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
            "revision": "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
        },
        "formation": {
            "provider": "openai",
            "model": "qwen3-8b",
            "base_url": "http://127.0.0.1:8080/v1",
            "api_key": "unused-by-this-local-server",
            "modalities": ["text"],
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    }
) as memory:
    memory.add("Ada said she prefers tea, and she sounded relieved.")
```

An OpenAI-compatible local server can fill this slot without another adapter. It must implement
`POST <base_url>/chat/completions`, accept `response_format={"type": "json_object"}`, and return one
standard choice with a non-empty `message.content`, or a provider refusal. Select `json_schema`
instead when the server supports the strict schema contract. A `finish_reason` of `length` raises
`ModelOutputTruncatedError`; increase `max_tokens` instead of accepting partial proposals.
An explicit provider refusal raises `ModelError(reason="model_refused")` and is not retried as
malformed JSON. Truncation, refusal, and schema validation apply to streamed responses as well.

Strict formation schemas require exactly one item per input observation and restrict
`observation_id` to that batch's aliases. An item may contain no proposals; these constraints do
not require inventing a fact. The parser still rejects duplicate or missing observations and
invalid evidence references. This batch-envelope policy is part of the strict formation schema;
the other schema recipes retain their existing identities. The provider must
support `minItems`, `maxItems`, and string enums; select `json_object` explicitly when it does not.

Formation stays off when the slot is omitted. Configuring `generation` does not enable it, and the
two slots use separate clients so they may select different models, endpoints, and credentials.
`modalities` controls which observations reach the former; unsupported observations are skipped.
Formation is one completion per `add`, or one batched completion per `add_many`, on the write path.
A failed call leaves the source observation committed, and retry tracking prevents duplicate
proposals for sources already formed by the same recipe.

The formation request carries a source observation's spatial `frame_id` and observer/subject
`anchor`, so a model can preserve the coordinate system, but does not disclose metric `x`/`y`
coordinates to the provider.

MindBridge validates the response envelope and every proposal. Invalid envelopes fail the call;
individual proposals the adapter cannot read are dropped and counted by
`mindbridge.formation.dropped_proposals`, and proposals the kernel refuses to ground are counted
over the whole operation by `mindbridge.formation.refused_proposals`. See
[memory types, time, and decay](memory-types-time-and-decay.md) for the resulting typed records and
visibility rules.

`formation_history_max_rows` enables a bounded, read-only text history window. Its default is
`0` (off); choose at most `128` rows and set `formation_history_budget_chars` (default `6000`)
to bound their combined content. The window ends at the target observation's recording time,
excludes this batch's targets and forgotten records, and includes retirement status for typed
records. History is background, never another formation target. The window retains whole records;
an oversized recent record stops selection instead of being skipped to expose older claims.
`history_truncated` means counterevidence may be missing. This is a recent-window policy, not
semantic history search or a guarantee that natural-language retractions are understood.

The OpenAI adapter aliases background records as `history_N`, preserves their event and validity
times, and permits a proposal to cite only its target, this batch, and its own active history.
History selects semantic versions recorded by the target's capture time, including retired
versions as background. Records and contexts share one database read snapshot without loading
media metadata. A typed record with no version known by that cutoff does not consume the row
window. Each target owns its grounding snapshot, even when another target reads a different
version of the same record. Within one request, the adapter serializes identical complete,
ordered history snapshots once and gives their targets explicit read-only group references.
Different versions, provenance, truncation flags, or model-visible values remain separate;
sharing does not authorize another target to cite that history. Content budgets still exclude
protocol overhead. This presentation policy changes the formation recipe in both JSON modes,
without changing the public formation input or its grounding checks.

With history enabled, a named event, state, relation, trait, or affect about an entity in that
history must jointly cite a still-standing entity record naming the subject. Retired names cannot
be reused by citing only the new observation. Missing or competing naming witnesses withhold that
proposal; a supported anonymous event is still accepted. The check does not rewrite source text,
interpret arbitrary withdrawal prose, or change host naming and consent authority. Forgotten
records stay excluded from history. Current identity binding also refuses competing visible names
for the same identity instead of selecting one by recording order.

This is version selection, not complete historical replay: stored text, metadata and in-place
confidence/visibility projections retain the store's existing semantics.
The kernel checks cited records again in the commit transaction. If a cited record was forgotten,
or its latest recorded version became hidden or retired during formation, derived records do not
commit and a `ModelError` with `reason="response_invalid"`, `stage="form"` reports that the
observation remains stored.
The formation prompt change changes `formation_space`; existing derived records remain available.
Reprocessing the same source under the new recipe may create additional derived records.

## Visual descriptions

The `vision` slot is omitted by default and, like `formation`, must stay omitted unless the
deployment wants it. Configuring it adds one chat completion per write batch that carries a
not-yet-described image or video, and buys the derived text that lets the lexical half of
retrieval reach a visual memory at all.

```python
config = {
    "embedding": {
        "provider": "sentence-transformers",
        "model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
        "revision": "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
    },
    "vision": {
        "provider": "openai",
        "model": "gpt-5-mini-2025-08-07",
        "modalities": ["image"],
    },
}
```

A memory whose content is one image has no words. Its full-text document is empty, so BM25 cannot
match it, the lexical re-ranking bonus scores it zero, and the only route that reaches it is the
dense one -- however capable the embedder is. Speech-bearing video escapes this through
`index_speech`; nothing covers a silent image. The caption is unioned into that document beside
whatever the caller wrote, each section carrying a `[visual description:<asset_id>]` marker so
`get` shows which assets were described, and the asset is still embedded natively: the derived
text is added to the record, never substituted for it.

A caption is labelled lines rather than prose -- what is shown, readable text transcribed exactly,
counts, place cues, a visible date or clock, and index tags -- because a question days later asks
for the sign or the count, and free prose names whichever of those the model found interesting. A
visual described while a speech backend is configured is also shown that clip's diarised
transcript, and answers with `Fact:` lines: short declarative statements that stay true after the
clip ends ("Lily is allergic to peanuts", "the yoga mat lives in the storage room"). Those land in
their own `[facts:<asset_id>]` section, so the distillation is correctable separately from the
observation, and where the dialogue itself states a diarised speaker's name that fact also
registers the name against the recognized identity, so every later clip resolving to the same
person is indexed under it.

These controls bound what it costs and what it can do:

- `modalities` is the visual capability set and accepts only `image` and `video`. An asset outside
  it is left undescribed, silently and by design.
- Video is described from ordered stills decoded locally, never by uploading the file.
  `vision.video_frames` selects an integer from `2` through `64`, default `4`, and each clip costs
  that many image parts; that is why the default is `[image]` and video is opted into.
  Sampling covers the clip in order, including its last frame. A non-default count participates
  in the visual-description recipe and cache identity; the default retains the four-frame recipe.
  A clip with no readable video stream is left without a caption rather than falling back to
  sending it. The
  request marks such a visual with its still count (`Visual 2, as 4 ordered stills:`) and asks for
  one caption per visual, never one per still: asked over an unmarked clip, a measured endpoint
  returned four separate descriptions on every attempt, the one-caption-per-input contract below
  rejected the reply whole, and `modalities: [image, video]` therefore stored no caption at all
  while still paying for the request.
- The caption is derived text, so the request carries pixels and an ordinal only -- no memory ID,
  file name, or store path -- and a reply that does not return exactly one non-empty caption per
  visual is rejected whole rather than mislabelling a memory with another's contents. The 20 MiB
  per-item and 64 MiB aggregate inline limits apply to these outbound image data URLs, including
  decoded video stills, rather than to a video file that is never uploaded. One malformed reply
  is retried once -- as it is for formation and consolidation, which parse JSON from the same
  endpoint -- because an endpoint can answer `200 OK` with invalid JSON and an SDK retry policy
  never sees that. For visual descriptions, the retry adds a short correction that reports only
  the rejected reply's JSON shape and the required caption count; it never sends the rejected
  caption text back or accepts a wrong-length list. A second failure, or any other failure, leaves
  the memory
  stored **without** a caption rather than failing the write. Losing derived text must never lose
  an observation the caller handed over. The batch that finally gives up -- the one that actually
  cost a memory its caption -- is counted on the vision span as `mindbridge.vision.failed_batches`,
  so the loss is measurable rather than silent. The failure is
  not negatively cached, but re-adding the same already-embedded memory is idempotent and does not
  revisit its caption. Repairing an existing captionless record therefore requires rebuilding it;
  the retry improves new writes and does not backfill old ones.
- A throttled or overloaded endpoint is waited out before that fail-open, not straight through it.
  A batch refused as `rate_limited`, with a timeout, on a dropped connection, or with a 5xx is
  described again after 1 s, 4 s, and 16 s; a refusal that an identical request cannot clear --
  `quota_exhausted`, `auth_failed`, `model_refused`, `request_rejected` -- is not retried at all, with one narrow
  exception: a `request_rejected` whose provider message says generation itself was aborted
  while producing the JSON reply ("Model output became abnormal ... generation was aborted ...
  Please retry") is retried the same way, because an identical request there routinely succeeds
  seconds later; every other `request_rejected` -- a rejected image, an unsupported request --
  still fails open on the first attempt. A connection the peer drops while a response body or
  stream is being read is classified `connection_failed` like one dropped during the request,
  whichever HTTP transport the installed SDK runs on. The SDK client's
  own `max_retries` budget is spent inside each of those attempts, so `max_retries` on this slot
  raises the per-attempt budget and is worth setting above the SDK default for a long ingest. A
  provider that throttles a corpus throttles it for minutes, which is long enough for a
  fail-open to store an entire ingest with no captions behind one log line. An attempt that fails
  but still has a wait left is not a lost caption, only a paid retry, so it is counted apart on
  `mindbridge.vision.retried_batches` rather than summed into `failed_batches` -- the two answer
  different questions, "how many memories lost their caption" against "what did a long ingest pay
  waiting one out".
- Captions are not reproducible. Models supporting sampling controls receive `temperature` 0 and
  a fixed `seed` unless the slot sets its own; known reasoning models receive neither automatic
  control. A measured compatible endpoint returned four different completions for four identical
  requests at those values. A caption becomes indexed text, so anything that needs two
  ingests of one corpus to build the same documents has to cache descriptions by asset digest;
  the benchmark harness does exactly that.

Description tokens are reported under their own model module, so a deployment that meters usage
can separate what captioning costs from what answering costs.

`settings.query_vision=True` also uses the configured image-capable vision backend during
`search`, `ask`, and `compile`. It describes query images temporarily and adds independent text
search routes for the full description and its exact `Text:` OCR line, alongside the original
question and native media routes. Dense/hybrid modes additionally embed these descriptions when
the embedder supports text; lexical mode uses them only for full-text retrieval. It does not
replace the original query media or add support for a modality the normal route cannot accept.
The switch defaults to `False` and has no effect without an applicable vision backend. Query
descriptions, including failed attempts, are cached only inside that operation. They create no
durable description, identity, or memory, and a description failure keeps the original search.

## Agentic memory management

The optional `consolidation` slot builds the bundled OpenAI-compatible `ConsolidationBackend`,
which is what `consolidate()` needs. It reads the same completion knobs as `formation`:

```python
with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {
            "provider": "sentence-transformers",
            "model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
            "revision": "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
        },
        "consolidation": {
            "provider": "openai",
            "model": "gpt-5-mini-2025-08-07",
            "modalities": ["text", "image"],
            "response_format": "json_schema",
            "max_tokens": 4096,
        },
    }
) as memory:
    report = memory.deliberate()
```

`deliberate()` is that loop: it asks `consolidation_candidates()` what the store's own state says
is due, consolidates each row under the row's trigger, and repeats until nothing is due. The
primitives stay available for a host that wants to schedule the two halves itself.

Consolidation stays off when the slot is omitted, and `consolidate()` then raises
`ModelError(reason="backend_not_configured")` while every other operation is unchanged. Declaring
it flips `consolidate` on in the capability document `mindbridge doctor` prints. `/healthz` and the
MCP server instructions never list it, whatever is declared: neither surface has a route or a tool
for consolidation, because the control plane stays with the process that owns the memory, and each
surface publishes only the operations it can serve. Unlike formation it is not on the write path:
it runs only when a host calls `consolidate()`.

### Running the loop inside a server process

A REST or MCP deployment cannot call `deliberate()` over the wire — no route and no tool reaches
the control plane — and one physical `data_dir` has one live owner, so nothing else can call it
either. The signals the served routes and tools record, above all the `QUERY_FAILURE` rows a
failed `search`, `ask`, or `compile` writes, would then be collected and never read. Pass
`deliberate_every` to `create_app` or `build_mcp_server` to have the serving process run the loop
itself:

```python
app = create_app(memory=memory, deliberate_every=900.0)
server = build_mcp_server(memory, deliberate_every=900.0)
```

It is a number of seconds, off by default, and it means the same thing on both surfaces. The round
runs in a worker thread inside the server lifespan, so it never blocks a request or a tool call,
and each round logs one line at `INFO` on the `mindbridge.api.deliberation` logger:
`deliberation: rounds=… weighed=… applied=… rejected=…`. A failed round logs a warning and the
next one still runs. Passing it without a configured `consolidation` backend, or with a
non-positive interval, raises `ValidationError` at construction rather than failing on the first
round.

`modalities` becomes the media the backend attaches to its request. Evidence assets in a declared
modality travel natively, so an image or audio memory with no derived description is readable by
the loop rather than an empty `content` string it can only propose forgetting; assets in an
undeclared modality are left out instead of failing the pass.

When `consolidation` declares exactly the same values as `generation` or `formation`, the
composition reuses that adapter rather than opening a second client against the same endpoint.
Any difference — a different model, endpoint, or credential — builds its own. See
[the memory management loop](memory-types-time-and-decay.md#memory-management-loop) for what a
consolidator may propose and what the kernel refuses.

## Local memory settings

The opt-in `settings.query_vision` flag, `generation.answer_style`, per-OpenAI-slot
`image_max_pixels`, and `vision.video_frames` controls are documented in
[grounded multimodal recall](grounded-multimodal-recall.md). Their defaults preserve the existing
query path and reader style; enabling query vision adds model work on image queries.

The `settings` mapping is the value-only `MemoryConfig` policy:

| Field | Default | Meaning |
| --- | --- | --- |
| `index_speech` | `True` | Persist configured speech analysis during `add` |
| `index_quantization` | `none` | Zvec 0.8 projection mode: `none`, `fp16`, unrotated `int8`, or `rabitq`; changing modes rebuilds from stored embeddings |
| `retrieval_mode` | `hybrid` | Instance candidate policy: `hybrid`, `dense`, or lexical-only `lexical`; lexical ranks by native normalized full-text relevance (RRF when both text fields apply); use physically isolated stores when comparing modes |
| `minimum_relevance` | `0.10` | Floor on query-relevant evidence before retention and reinforcement ranking |
| `ambiguity_margin` | `0.01` | Withhold an unresolved top-two tie when `limit=1`, including lexical matches |
| `evidence_budget_chars` | `24000` | Soft budget for widening `ask` grounding; mandatory `limit` hits survive even when over budget. `None` disables ranked-tail expansion; planning and linked expansion may still add records. No provider token guarantee |
| `recall_planning` | `False` | Ask the answerer for a recall plan during `ask`, and ground on the set, sequence or entity it names; off means `ask` runs the one search it always has. An answerer without `plan_recall` cannot plan: the store logs one warning when it opens and every question runs that same search |
| `recall_query_expansion` | `False` | Experimental execution of `similar.query` when planning is on. At most two additional searches per `ask`, shared across steps and replans; each caches up to 100 ranked hits and may add at most `limit` novel hits before the replaceable ranked tail. Also enables bounded evidence-driven retries of strict abstentions. See the controls below |
| `query_vision` | `False` | Describe query images through the configured image-capable vision backend and add transient description/OCR search routes. Retains native routes, caches only inside the operation, and writes no memory or description rows |
| `recall_set_budget_chars` | `30000` | Soft character allowance used to pack a plan's additions after charging its mandatory window, narrowed by `evidence_budget_chars` when set. Also bounds experimental semantic additions; a point plan without query expansion keeps its normal ranked window. Read only when `recall_planning` is on |
| `recall_set_max_rows` | `60` | Matched rows a set, sequence or entity plan may add at all, whatever the character budget leaves room for; the rows past it are reported as not shown, so the set is declared incomplete. Bounds the matched set only. Read only when `recall_planning` is on |
| `recall_rounds` | `2` | Plan-and-answer rounds one `ask` may spend. A later round requires an abstention under `best_effort`, or under `strict` when `recall_query_expansion=True`. `1` disables retries. Read only when `recall_planning` is on |
| `evidence_expansion` | `False` | Widen `ask` grounding with exact query-media source context and existing structural links. An exact source in the scoped ranking can add its immediate corpus neighbors before semantic additions and the ranked tail; capture, person, place, lineage and evidence links spend the remaining budget. Adds no model call |
| `evidence_expansion_max_rows` | `24` | Joint cap on added exact-source context and structural rows, within the remaining character and supplemental media budgets. Read only when `evidence_expansion` is on |
| `decay_half_life_days` | `None` | Optional positive half-life for query-time decay |
| `reinforce_on_answer` | `True` | Count the evidence `ask()` cited, so retrieval favours it later |
| `independent_evidence` | `False` | Experimental capture-grounded AND/OR corroboration for formation and consolidation; fixed when a store is created. See [independent evidence](memory-types-time-and-decay.md#experimental-independent-evidence) |
| `formation_history_max_rows` | `0` | Read-only text history records per formation target; `0` disables, maximum `128` |
| `formation_history_budget_chars` | `6000` | Combined history content characters per target; positive, whole records only, excludes protocol overhead |
| `speaker_similarity` | `0.78` | Voice identity match threshold (uncalibrated; see below) |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |
| `identity_link_min_assets` | `2` | Distinct assets a voice-and-face pair must share before they merge |
| `memory_budget_records` | `None` | Active records this instance is meant to hold; the whole definition of the `PRESSURE` trigger, which derives nothing without it |
| `query_failure_window_seconds` | `3600.0` | How far back the `QUERY_FAILURE` trigger counts near-equal empty recalls |
| `query_failure_history` | `512` | How many empty recalls the store keeps at all; the oldest fall out |
| `retention` | empty | What `apply_retention()` may delete; spell it as the top-level `retention` section below |

Changing `retrieval_mode` does not change stored embeddings or full-text data, so after closing an
instance you may reopen the same `data_dir` under another mode without reindexing. Separate local
stores are required for simultaneous or isolated comparisons because one `data_dir` has one live
owner. `lexical` skips only query embedding; ingestion still requires and writes the configured
embedder for durable index consistency.

`recall_query_expansion` requires `recall_planning=True` and is disabled by default. Without it,
`similar(query, k)` reuses the original question's ranking; changing query text or increasing `k`
does not initiate another search. With it, each distinct query caches a 100-hit ranking and the
step reads its requested prefix. The original query reuses its initial ranking unless `k`
requests greater depth; deepening consumes one of the same two additional-search slots. Failed
embedding calls are cached and consume a slot; a third distinct search contributes no new rows.
The cache and ceiling reset at the next `ask`. Whitespace at the ends is ignored for cache keys.
Successful routes remain eligible across internal rounds, at the greatest depth actually
requested, in discovery order. A later plan can name only its new query without losing an earlier
bridge fact. Retained routes reuse their cache and continue to share the same per-query window,
character allowance and media cap; they do not add another grounding budget.
Multiple routes offer one novel record each in turn, skipping duplicates, before any route
offers its next record against that shared budget.

All searches preserve the caller's scope, memory type, reference time and original inferred
temporal intent; rewrites cannot widen visibility or replace query media. Semantic additions
deduplicate by record ID and stop at `limit` novel rows per query.
Their combined character allowance is `min(recall_set_budget_chars, evidence_budget_chars)`
when the latter is set, otherwise `recall_set_budget_chars`. The original `limit` window and its
modality floor remain mandatory, even if already over budget, and matched set rows keep their
own budget, media caps and completeness accounting. New semantic evidence spends the remaining
budget before the optional original ranked tail; linked rows follow the tail. Thus useful
new-query evidence can replace supplemental ranked records without removing the initial window.
If no semantic row fits, the original grounding is restored. Added semantic media stop at twice
`limit` rows. These are bounded rankings, never complete sets; semantic or tail packing cannot
readmit matched rows omitted by a set's caps.

With this switch on, an abstained `strict` answer can also trigger another internal round within
`recall_rounds`. The next planner receives a JSON note of at most `6000` characters, containing
the missing-evidence status, local `R` source aliases, source IDs, event dates and up to `1500`
characters per read record, deduplicated by record ID. Omitted records and truncated excerpts are
reported. The note explicitly marks its evidence as untrusted data, and the kernel still
validates every proposed read. Ordinary planning retains its best-effort-only retry rule and
count/date summary. A replaceable round is buffered, so streaming releases only the answer that
stands. No new model or service is configured by the switch. Mechanism tests alone do not
establish natural-language gains.

`minimum_relevance` uses the selected route's relevance scale: dense cosine in `dense`, native
normalized full-text relevance in `lexical`, and the documented hybrid fusion in `hybrid`. Keep
the default when comparing modes, or calibrate a changed threshold separately for each mode.

A memory declares its composition through `Memory.capabilities`, which reports the modalities,
model identities, and configured backends the routing layer reads, including
`consolidation_model` when a consolidator is injected. `GET /healthz` and the MCP server
instructions publish that same view.

Thresholds and margins accept values from `0` through `1`. Relevance is floored at `0`, so
`minimum_relevance=0` admits every candidate and disables the weak-evidence floor; a zero ambiguity margin disables the corresponding tie rejection. A zero
speaker or face similarity is merely the most permissive threshold.

`minimum_relevance` gates the signals the query asked about, and only those. Retrieval relevance
and temporal proximity are inside the gate: a caller who asks "in 2024" made the year part of the
question, so overlapping it counts as evidence and missing it does not. Reinforcement and
`decay_half_life_days` retention are outside it, because the query never mentioned them. They
still shape `SearchHit.score` and the result order, so **an admitted hit can report a `score`
below the floor you set**.

That asymmetry is deliberate. Both factors are bounded below by `0.3`, so with retention inside
the gate a perfectly relevant memory would decay to `0.30`, and to `0.09` once a dated question's
window also missed it — under the `0.10` default. A floor that included retention would turn
"prefer recent" into "hide old" for precisely the deployment that enabled decay and then asked
about last year. Ranking an old event last is fine; ceasing to return it is not.

Use `search_with_trace` to read the gated quantity as `gate_relevance`, beside the
`retention_factor` and `temporal_factor` that moved `score` away from it. `decay_half_life_days` must be
positive when set. `evidence_budget_chars` keeps the `limit` hits unconditionally and then admits
further ranked memories while the evidence fits, charging each media asset its modality's text
equivalent because a media part costs a model far more than its record's text. Because the `limit`
hits are never dropped, this setting can only enlarge the grounding set. Lowering `limit` and
setting the budget to `None` reduces the basic ranked window; recall planning and linked expansion
can still add records when enabled. These controls do not bound the serialized provider input.
With a budget, lowering `limit` may leave the evidence window unchanged because packing
can refill it. Increasing the budget can enlarge the answer prompt.

Evidence selection and provider capacity are separate constraints. Character counts and media
text equivalents omit system instructions, question text, provenance, attachment labels, message
framing, and model-specific media tokenization. The bundled OpenAI adapter checks the complete
answer text (system prompt, serialized user text, and attachment labels) against 4 MiB before
reading media, and applies separate inline-media byte and video-count limits. These are payload
safety bounds, not token/context limits. `generation.max_tokens` bounds output, including
reasoning for models that use it; it does not cap input. Reserve output capacity when sizing a
request against the provider's context window. Exact input preflight needs that endpoint's own
counting protocol for the actual request; MindBridge does not provide a universal token preflight.
Provider-declared context overflow raises permanent `ModelError(reason="context_length_exceeded")`
(HTTP 413), without retrying an answer after dropping evidence. The independent
[`compile_context` budget](context-compilation.md#budget) remains a hard character ceiling on
the bundle it renders; it also does not promise provider tokens.

A set, sequence or entity recall plan grounds on its own matched set *in addition to* the ranked
window -- the same `limit` hits and modality floor the unplanned path grounds -- and
`recall_set_budget_chars` bounds what the plan added with the same text-equivalent cost. It is a
ceiling on supplemental packing: when `evidence_budget_chars` is also set, the lower of the two
applies across question shapes after charging the mandatory ranked window. Neither bound may
trim the mandatory `limit` window. With semantic query expansion enabled, supplemental ranked
rows can instead yield budget to new-query evidence. Media rows carry one further bound.
Grounding a media row runs face and
speech recognition over it, which are writes and are paid again on a replan round, so the matched
set keeps at most twice `limit` media rows however many its predicate matched -- the ranked window,
whose width is `limit`, is outside that cap. Text rows are unaffected, and the matched rows either
bound dropped are reported to the reader as matched records not shown. `recall_set_max_rows`
is the same bound on the other axis: a corpus of short records fits hundreds of matched rows
inside 30 000 characters, and the rows past the cap are dropped in favour of the earliest ones --
the read's own chronological order, not a second ranking -- and counted as not shown.
For a `sequence` plan, the final selected evidence is presented in corpus time order, using
SQLite insertion order when event times tie, across matched, ranked and supplemental records;
this presentation order does not establish a complete event history or answer set.

`evidence_expansion` widens the window without another model call. If an original query attachment
has exactly the same asset ID and SHA as a source in the already scoped ranking, the kernel can
include that source and the immediately preceding and following corpus records. At most `limit`
source occurrences are followed; repeated attachments do not duplicate records. These additions
share the character and supplemental media budgets with semantic additions, and are considered
before the replaceable ranked tail. If no new context fits, the original grounding path is used.
The original grounding window remains protected. Nearby
records retain the caller's scope and memory type; adjacency does not establish a shared event or
a complete answer set. A similar-looking image with different bytes does not activate this read.

After those additions and the ranked tail, the kernel reads the
records that window is *structurally* linked to -- committed under the same capture, about the
same recognized person, labelled with the same symbolic place, in the same claim lineage, or
joined to it by an evidence edge -- and admits them into whatever budget the ranking left unspent.
Exact-source context and structural additions share `evidence_expansion_max_rows`. Predicate rows
omitted by their own caps cannot reenter through these routes. Required evidence is charged before
supplemental admission, so only a required window that itself exceeds the character budget may
exceed that soft budget. Each structural edge is one read over a column the kernel already wrote,
indexed for every edge but the
capture, whose `memory_semantics.source_id` carries no index yet and is scanned instead.
The linked media rows are capped at twice `limit` on top of `evidence_expansion_max_rows`, the
same bound a recall program's matched media rows hit and for the same reason: a media row reaching
the answer call pays face and speech recognition, again on every replan round, and a capture on
the photo corpora this is recommended for is a whole day of clips. The cap bounds only what
expansion added -- the ranked window and the budget's own tail are the ranking, and the ranking is
never trimmed for carrying media.
Each edge is bounded on its own by the selectivity rule below, and an edge that links to more than
a fifth of the corpus contributes nothing rather than thinning the window: on a two-speaker
transcript every record is about both speakers, so the identity edge selects the corpus and says
nothing about the question, while the capture edge beside it stays selective. Gating the edges
together would let the first discard the second.

Structural expansion is off by default. If the ranked tail fills the character budget, no linked
rows can be added; exact query-media context still has its earlier admission priority. With
`evidence_budget_chars=None`, linked rows can follow the mandatory window within the row and
supplemental media caps.

Evaluate expansion on your own corpus before enabling it. Compare complete support, answer
quality and evidence cost against both the narrow window and a budget of ranked rows. Existing
links can reach records below the window without establishing a complete answer set or improving
quality. `benchmarks/edge_power.py` measures potential support coverage without running a reader.

What expansion adds is reported to the reader as records linked to the evidence rather than records
a predicate matched, so nothing it adds licenses a count, and the count is the number of rows
expansion actually admitted rather than the number of window rows that happen to be linked.

A read is only allowed to be complete while its predicate is selective. A step whose predicate
selected more than a fifth of the active corpus, or more than four times `limit` rows on a corpus
too small for that fifth to mean anything, contributes no rows at all: the reader is told what the
predicate matched and that what it holds is the question's top-ranked records instead. Measured on
LoCoMo, where every record reads "[date] Caroline said: ...", an entity step that degraded to
matching the name as text selected about half the corpus and cost 19 points of accuracy on exactly
the questions it ran on. A plan whose every step is non-selective grounds precisely what no plan
would have grounded.

Pass an OpenTelemetry tracer through the separate `tracer=` argument of `Memory.from_config()` or
`Memory(...)`; it is not a setting. See [operations](operations.md#telemetry) for exporter setup
and [memory types, time, and decay](memory-types-time-and-decay.md#decay-and-reinforcement) for
ranking semantics.

`index_speech` is on by default. It only has an effect when `transcriber` is a `SpeechBackend`
(`speech: funasr`), which has already run its analysis by the time `add` reaches the index, so
reusing that text costs no additional model call and no additional token; with no speech backend, or
with a plain transcription backend such as `speech: openai`, the setting does nothing. Set it to
`False` to keep speaker identities out of the index and out of `add`-time identity matching.

### The identity thresholds are not calibrated

`face_similarity` has provenance: `0.363` is upstream SFace's own `_threshold_cosine`, adopted
verbatim. **`speaker_similarity` does not.** `0.78` has no upstream source, and it is knowingly
wrong in the safe direction — it sits above CAM++'s same-speaker cosine, so voice identities
fragment: one run turned 1667 voice segments into 1743 separate identities. Expect voice identity to
be unreliable at the default until it is calibrated.

It is left high on purpose. The failure modes are not symmetric: too high fragments one person into
many, which costs recall, while too low merges two people into one identity, which hands one
person's memories to another. Fragmentation is recoverable; a false merge is a correctness and
privacy failure.

The obvious fix does not work. The pinned CAM++ recipe publishes its own threshold — `yesOrno_thr`
is `0.31` in the model's `configuration.json`, and ModelScope's speaker-verification pipeline uses
exactly that value on a raw cosine, the same quantity MindBridge compares. But `0.31` is calibrated
for **one pair** of embeddings, and MindBridge accepts on the **maximum over up to 20 stored
exemplars**. A maximum over more samples can only rise, so a pair threshold is a lower bound on the
correct max-over-many threshold, never the value itself: measured against this matcher, mutually
orthogonal random impostor vectors already reach `0.28` at 20 exemplars. `speaker_margin` does not
compensate, because it only breaks ties between two candidate identities — with a single enrolled
identity it never applies, and with several the maximum inflates only the winner.

To calibrate it for a deployment, embed a labelled set of that deployment's own audio with the
configured recipe, then plot two distributions of the score MindBridge actually decides on — the
maximum over an identity's exemplar bank — one for same-speaker pairs and one for
different-speaker pairs. Set the threshold between them, and keep `speaker_margin` non-zero. Around
30 utterances each from 4-8 speakers in the real acoustic environment is enough to see whether a gap
exists. The same procedure applies to `face_similarity` if SFace's published value proves wrong on a
deployment's cameras.

## Retention policy

`retention` is its own top-level section rather than a `settings` field, because every other
setting shapes what recall returns and can be changed back, and this one deletes.

```python
config = {
    "embedding": {
        "provider": "sentence-transformers",
        "model": "jinaai/jina-embeddings-v5-omni-small-retrieval",
        "revision": "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
    },
    "retention": {
        "media_days": 30,
        "forgotten_days": 90,
        "capture_failure_days": 7,
    },
}
```

| Field | Default | Effect |
| --- | --- | --- |
| `media_days` | `None` | Delete media assets recorded longer ago than this, and every memory that still references them |
| `forgotten_days` | `None` | Physically delete records that `forget()` moved out of recall longer ago than this |
| `capture_failure_days` | `None` | Abandon capture-queue rows enqueued longer ago than this that have failed at least once; the memory itself stays |

Every field is optional and every one defaults to `None`. `None` means "no policy", not "keep for
zero days": nothing is ever deleted because a clock ticked and no age was declared. Ages are
positive numbers of days, measured from the material's own recorded time rather than from the last
time it was read, so traffic cannot reset a policy.

Nothing here runs on its own. The policy is inert until a host calls
[`Memory.apply_retention()`](api/python-sdk.md#data-subject-rights) or `mindbridge
apply-retention`, which is what keeps physical deletion an explicit act with an auditable report.
Start with `apply_retention(dry_run=True)`, which names exactly what a real pass would remove and
removes nothing.

`MemoryConfig` carries the same value under the name `retention`, which is what
`Memory.from_plugins()` and the `Memory(retention=...)` constructor argument read. Declaring it in
both places is refused rather than silently resolved.

## Direct adapter injection

Direct injection is the boundary for custom protocols, caller-owned SDK clients, proxies, and
connection pooling:

```python
from openai import OpenAI

from mindbridge import SentenceTransformersEmbedder, Memory, OpenAIModels

with OpenAI(timeout=30.0, max_retries=3) as client:
    models = OpenAIModels(
        generation_client=client,
        generation_model="gpt-5-mini-2025-08-07",
    )
    with Memory(
        "./data/assistant",
        embedder=SentenceTransformersEmbedder(
            model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
            revision="e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
        ),
        answerer=models,
    ) as memory:
        memory.add("The spare key is in the blue toolbox.")
        print(memory.ask("Where is the spare key?").answer)
```

`Memory` closes the adapter objects passed to it. `OpenAIModels.close()` leaves a caller-supplied
OpenAI client open, so the outer client context remains the application's responsibility.

`former=` accepts a custom `FormationBackend` and `consolidator=` a `ConsolidationBackend`;
declarative `formation: openai` and `consolidation: openai` supply the bundled adapter for those
slots. `vision_describer=` accepts a `VisionDescriptionBackend`, for a custom or non-bundled adapter;
declarative `vision: openai` supplies the bundled one. `add` and `add_many` call it for every
embedder whenever a visual asset has no description yet, and `AsyncVisionStream` additionally calls
it at finality when the embedder lacks native image support and no external `VisionPartial`
supplied a description, so speculative retrieval has a query before finality.

`OpenAIModels` implements both reasoning protocols on its generation client, so one adapter can
fill `answerer=`, `former=`, and `consolidator=`:

```python
models = OpenAIModels(generation_model="gpt-5-mini-2025-08-07")
memory = Memory(
    "./data/assistant",
    embedder=SentenceTransformersEmbedder(
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        revision="e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4",
    ),
    answerer=models,
    former=models,
    consolidator=models,
)
```

`Memory` closes each distinct adapter once, so passing one object to several slots is safe.
Omitting `consolidator=` leaves `consolidate()` unavailable and every other operation unchanged;
see [the memory management loop](memory-types-time-and-decay.md#memory-management-loop) for what a
consolidator is allowed to propose.

For an OpenAI-compatible gateway that times out while waiting for a complete JSON response, set
`stream: true` on an `openai` generation, formation, consolidation, or vision configuration. It
collects the provider's SSE response before parsing it, requires the terminal usage chunk, and is
off by default. This transport setting does not change `ask_stream()`, which already delivers
answer deltas to the caller.

The benchmark's `benchmark.run.stream_answers` separately chooses the public SDK answer surface:
`true` (the default) uses `AsyncMemory.ask_stream()`, and `false` uses buffered `AsyncMemory.ask()`.
For sixteen concurrent answers against one store, set it to `false` and set
`benchmark.run.request_concurrency: 16`. See [benchmark concurrency](benchmarking.md) for admission
limits and timing semantics.

Use `MemoryPlugins` with `Memory.from_plugins()` when constructed adapters should travel as one
value. `resolve_memory_config()` is for hosts that need a `MemoryComposition` before opening
storage; close it unless its plugins are transferred to a `Memory`. Ordinary applications should
prefer `Memory.from_config()`.

For exact constructor, protocol, and configuration types, use the
[Python SDK reference](api/python-sdk.md). For benchmark-only environment variables, use
[benchmarking](benchmarking.md).
