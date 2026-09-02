# Configuration

Configuration has one job: compose model capabilities and local policy around one `Memory`.
Storage and retrieval semantics stay the same whichever composition path you choose.

| Path | Choose it when |
| --- | --- |
| `Memory.from_config()` | Bundled providers and SDK-side credentials are enough |
| `Memory.from_plugins()` | The application already groups constructed backends and settings |
| `Memory(...)` | The application injects individual backends or owns SDK clients directly |

All three paths open the same embedded kernel. An `EmbeddingBackend` is required; generation,
speech, face, vision-description, and formation capabilities are optional.

## Install only the surfaces you use

Optional extras add the heavy or protocol-specific runtimes:

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

The base package already contains local SQLite storage, the Zvec index, public values, and the
backend protocols.

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
        "minimum_relevance": 0.10,
        "ambiguity_margin": 0.01,
        "evidence_budget_chars": 12000,
    },
}

with Memory.from_config(config) as memory:
    memory.add("Remember this.")
```

`AsyncMemory.from_config()` accepts the same shape. Use `MindBridgeConfig.model_validate()` when a
host needs validation before opening storage. Unknown fields, unknown providers, and out-of-range
values are rejected rather than ignored or coerced; numeric and Boolean values are strict.
Declarative configuration owns every adapter it creates and closes it with the memory.

### Provider fields

Bundled provider fields are, with "connection fields" meaning `base_url`, `api_key`, `timeout`,
and `max_retries`:

| Slot and provider | Required fields | Optional fields and defaults |
| --- | --- | --- |
| `embedding: jina-omni` | — | `dimension=1024`, `device=None`, `batch_size=32` |
| `embedding: sentence-transformers` | `model`, `revision` | `dimension=None`, `device=None`, `batch_size=32` |
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, `modalities=[text]` (at least one), `request_format=input`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `video_limit=8`, `min_video_seconds=None`, `extra_body=None`, plus connection fields |
| `formation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `extra_body=None`, plus connection fields |
| `speech: funasr` | — | `device=auto` |
| `speech: openai` | — | `model=whisper-1`, `space=None`, plus connection fields |
| `face: opencv` | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

Every OpenAI slot accepts `base_url=None`, `api_key=None`, `timeout=None`, and `max_retries=None`.
The official SDK then applies its own defaults and, for an unset `api_key`, reads its standard
credentials including `OPENAI_API_KEY`.

Each slot builds its own client, so each takes its own `api_key`. That is what lets one
composition point `embedding` at a local server and `generation` at a hosted one: a single
environment variable cannot describe two endpoints with different credentials. The field is a
Pydantic `SecretStr`, so the value is masked in `model_dump()`, in `repr()`, and therefore in
anything that serialises a configuration -- but it is still a secret in a file. Prefer the SDK's
environment lookup, or an injected caller-owned client, wherever the file would be committed or
shared. Atomic generation modalities are `text`, `image`, `video`, and `audio`.

Jina dimensions are `32`, `64`, `128`, `256`, `512`, or `1024`. Batch sizes and model token/video
limits are positive; OpenAI timeouts are positive, retry counts are non-negative, temperature is
from 0 through 2, and seed is from 0 through `2**63 - 1`. Face thresholds are from 0 through 1.
`generation.video_limit` caps retrieved evidence videos in one answer request; question media has
priority, and `None` disables that evidence-video count. `generation.min_video_seconds` answers a
shorter video as four ordered stills instead, which is what an endpoint that rejects very short
clips needs; it requires the model to accept images and is unset by default. `formation` has no
`video_limit` or `min_video_seconds` because
it shapes an answer rather than a formation proposal; the slot otherwise takes the same fields as
`generation`, since the adapter derives its formation model and space from exactly those values.
`generation` and `formation` are separate slots and separate clients: setting one never enables the
other. See [automatic formation over a local server](#automatic-formation-over-a-local-server) for
what that slot costs and what a local endpoint has to implement.

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

## Automatic formation over a local server

`formation` is the only supplier of MindBridge's model-derived semantics: bitemporal validity windows,
affect, traits, entities, relations, and proposed spatial pose all arrive as `FormationProposal`
values from this one slot. Its single bundled implementation is the OpenAI SDK adapter, and that
adapter reads `base_url` like every other OpenAI slot — so **an OpenAI-compatible local server
already fills the slot, with no additional adapter and no additional dependency.** `llama-server`
from llama.cpp, vLLM's OpenAI server, and Ollama's compatible endpoint all qualify.

```python
import os

from mindbridge import Memory

# The official SDK performs its own credential lookup even for an endpoint that checks nothing.
os.environ.setdefault("OPENAI_API_KEY", "unused-by-a-local-server")

with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
        "formation": {
            "provider": "openai",
            "model": "qwen3-8b",
            "base_url": "http://127.0.0.1:8080/v1",
            "temperature": 0.0,
            "max_tokens": 2048,
        },
    }
) as memory:
    memory.add("Ada said she prefers tea, and she sounded relieved.")
```

A local endpoint that authenticates nobody still needs a credential in the environment. The
official SDK resolves its own key when the client is constructed, and declarative configuration has
no credential field by design, so `resolve_memory_config()` and `Memory.from_config()` raise
`openai.OpenAIError` when `OPENAI_API_KEY` is unset — before any request reaches the local server.
Any non-empty placeholder satisfies it. Inject a caller-owned client instead when the placeholder is
unacceptable.

### What the local server must implement

- `POST <base_url>/chat/completions` with the standard request and response shape. Formation uses
  chat completions only; it never calls embeddings, transcription, or streaming.
- A reply with exactly one choice at `index` 0, carrying `finish_reason` and a non-empty
  `message.content` string. Anything else is refused rather than partially stored.
- `response_format={"type": "json_object"}`, which every formation request sends. A server that
  ignores the field still works if the model returns one JSON object anyway, but nothing recovers a
  prose reply.
- Enough output budget for the whole batch. `finish_reason == "length"` raises
  `ModelOutputTruncatedError` instead of storing a truncated set, so raise `max_tokens` rather than
  retrying.
- `seed`, `temperature`, `max_tokens`, and `extra_body` are forwarded only when configured, so a
  server that rejects one of them stays usable as long as you leave it unset.

MindBridge supplies the system prompt and the reply schema; the local model does not need to know
them in advance, but it does have to obey them. Validation is strict per proposal: one unknown
field, one unusable `kind`, or one out-of-range `confidence` discards that whole proposal and the
count is published on `mindbridge.formation.dropped_proposals`, so the loss is visible rather than
silent. Damage to the envelope itself -- unparseable JSON, a missing observation, an unexpected
top-level field -- still fails the whole call, and a `finish_reason` of `length` raises rather than
storing a truncated set. A small local model is where that bites, so pin `temperature` at `0.0` and
check a handful of observations before trusting an unattended deployment.

### `formation` is not `generation`

They are separate slots over separate clients, which is deliberate rather than incidental:

- Configuring one never enables the other, in either direction.
- Each slot builds its own SDK client. Pointing `generation` and `formation` at the same URL therefore
  opens two connection pools against one server. That costs one extra pool and changes nothing
  about correctness — and it is what lets the two slots differ, which is the useful arrangement
  here: a capable remote answerer on the read path, a local former on the write path, or the
  reverse. Inject one caller-owned `OpenAI` client into `OpenAIModels` when a single pool matters
  more than declarative configuration.
- `formation` accepts no `video_limit` or `min_video_seconds`: both shape the media an answer
  request carries, and a formation proposal is not an answer.

### Reachable is not default

`formation` defaults to `None`, and that default is not a placeholder for an unfinished feature.
Formation is one chat completion per `add` — per batch for `add_many` — on the write path, and the
call happens *after* the raw observation commits. A rejected reply therefore raises from `add` with
the observation already durable and searchable; the store records which observations a formation
space has already covered, so a retry re-forms only what is still missing rather than duplicating
proposals.

An internal controlled comparison of competing write paths measured *not* extracting on the write
path ahead by 15.9 and 22.0 points on its two arms. The same comparison found that keeping the raw
observation **and** the derived records loses nothing, while replacing the raw observation with
extraction does lose. MindBridge's former is that additive shape, so the penalty is not what makes
the slot opt-in — cost is, together with the deployment choice of which model sees every
observation. Turn it on for the semantics it supplies, on evidence from your own corpus; see
[benchmarking](benchmarking.md) for what a quality claim has to report. Leaving it off costs
nothing: no model call, no optional dependency, and no change to any other operation.

## Local memory settings

The `settings` mapping is the value-only `MemorySettings` policy (`MemoryConfig` is a compatible
alias):

| Field | Default | Meaning |
| --- | --- | --- |
| `index_speech` | `True` | Persist configured speech analysis during `add` |
| `index_quantization` | `none` | Zvec projection mode: `none`, `fp16`, `int8`, or `rabitq` |
| `minimum_relevance` | `0.10` | Floor on evidence relevance: the cosine the dense route reports, or the demoted full-text contribution when only the lexical route matched, times the observation's own confidence |
| `ambiguity_margin` | `0.01` | Withhold an unresolved top-two tie when `limit=1` |
| `evidence_budget_chars` | `None` | Widen `ask` grounding while the evidence fits this budget; raises a floor, never a ceiling; `None` grounds on exactly `limit` |
| `decay_half_life_days` | `None` | Optional positive half-life for query-time decay |
| `reinforce_on_answer` | `true` | Count the evidence `ask()` cited, so retrieval favours it later |
| `speaker_similarity` | `0.78` | Voice identity match threshold (uncalibrated; see below) |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |
| `identity_link_min_assets` | `2` | Distinct assets a voice-and-face pair must share before they merge |

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
hits are never dropped, this setting can only enlarge a prompt: to bound one, lower `limit` and
leave the budget at `None`. Setting it also removes `limit`'s effect on prompt size, since the
budget refills the window to the same width whatever `limit` was.

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

`Memory` closes the adapter objects passed to it. `OpenAIModels.close()` deliberately leaves a
caller-supplied SDK client open, so the outer client context remains the application's
responsibility. Credentials, retries, timeouts, proxies, and connection pooling also remain SDK
configuration.

Two capability slots behave differently from the declarative catalog:

- `former=` takes a `FormationBackend` directly, for a custom or non-bundled adapter that proposes
  typed memories after a raw observation commits. The declarative `formation` slot above builds the
  bundled OpenAI adapter for the same slot, and the command line fills it with `--former NAME`
  from the same recipe table as `--embedder` and `--answerer`. Configuring `generation` never
  enables automatic formation, because a former is a model call per observation on the write path.
  It stays opt-in on every path.
- `vision_describer=` takes a `VisionDescriptionBackend` and is reachable only this way, or through
  `MemoryPlugins`; no declarative provider selects it, because MindBridge bundles no
  implementation of that protocol. `add` and `add_many` call it for every embedder whenever a
  visual asset has no description yet. `AsyncVisionStream` additionally calls it at finality, when
  the embedder lacks native image support and no external `VisionPartial` supplied a description,
  so speculative retrieval has a query before finality.

Use `MemoryPlugins` with `Memory.from_plugins()` when these constructed adapters should travel as
one value. `resolve_memory_config()` is for hosts that need a `MemoryComposition` before opening
storage; close that composition unless its plugins are transferred to a `Memory`. Ordinary
applications should prefer `Memory.from_config()`.

For every constructor and protocol field, see the [Python SDK reference](api/python-sdk.md). For
benchmark-only environment variables, see [benchmarking](benchmarking.md).
