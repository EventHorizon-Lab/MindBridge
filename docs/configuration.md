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
| `openai` | Official OpenAI SDK adapter and media handling |
| `face` | Local OpenCV face analysis |
| `server` | FastAPI and Uvicorn REST serving |
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
host needs validation before opening storage.

**Contract:** Unknown fields and providers are rejected. Numeric and Boolean values are strict;
invalid values are not silently coerced. Declarative composition owns and closes every adapter it
creates.

### Provider fields

In this table, connection fields are `base_url`, `api_key`, `timeout`, and `max_retries`:

| Slot and provider | Required | Optional defaults |
| --- | --- | --- |
| `embedding: jina-omni` | — | `dimension=1024`, `device=None`, `batch_size=32` |
| `embedding: sentence-transformers` | `model`, `revision` | `dimension=None`, `device=None`, `batch_size=32` |
| `embedding: openai` | — | `model=text-embedding-3-small`, `dimension=1536`, `space=None`, `modalities=[text]` (at least one), `request_format=input`, plus connection fields |
| `generation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `video_limit=8`, `min_video_seconds=None`, `extra_body=None`, plus connection fields |
| `formation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `extra_body=None`, plus connection fields |
| `vision: openai` | — | `model=gpt-5-mini`, `modalities=[image]` (image and/or video only), `temperature=None`, `seed=None`, `max_tokens=None`, `extra_body=None`, plus connection fields |
| `consolidation: openai` | — | `model=gpt-5-mini`, `modalities=[text]`, `temperature=None`, `seed=None`, `max_tokens=None`, `extra_body=None`, plus connection fields |
| `speech: funasr` | — | `device=auto` |
| `speech: openai` | — | `model=whisper-1`, `space=None`, plus connection fields |
| `face: opencv` | `detector_model`, `recognizer_model` | `score_threshold=0.9`, `nms_threshold=0.3`, `top_k=5000`, `frame_interval_ms=1000`, `max_video_frames=300` |

If `api_key` is unset, the official SDK uses its standard credential lookup, including
`OPENAI_API_KEY`. Each OpenAI slot builds a separate client and can instead receive its own
`api_key`, endpoint, timeout, and retry policy. `api_key` is a Pydantic `SecretStr`, so dumps and
representations mask it, but a configuration file still contains a secret. Prefer environment
lookup or a caller-owned client when the file could be committed or shared.

Atomic capability declarations contain only `text`, `image`, `video`, and `audio`. OpenAI
embedding requires at least one modality. Jina dimensions are `32`, `64`, `128`, `256`, `512`, or
`1024`. Positive fields must be greater than zero; retry counts are non-negative; temperature is
from 0 through 2; seed is from 0 through `2**63 - 1`; face thresholds are from 0 through 1.

`generation.video_limit` caps retrieved evidence videos in one answer request; question media has
priority, and `None` disables that count. `generation.min_video_seconds` sends a shorter video as
four ordered stills and requires image support. Formation and consolidation have neither field
because they shape answer evidence, not proposals.

`vision` takes the same completion fields as `formation`, but its `modalities` accepts only `image`
and `video`, because it is the visual capability set rather than a generation one. `generation`,
`formation`, `vision`, and `consolidation` are separate slots and separate clients: setting one
never enables another. See [visual descriptions](#visual-descriptions) for the write-path cost of
the vision slot.

## Embedding choices

The embedding model defines durable vector identity, so every memory requires one:

- `jina-omni` is the pinned multimodal recipe used in project examples. Its weights are CC BY-NC
  4.0. Loading executes pinned upstream code with `trust_remote_code=True`; review that code and
  license as part of the application's trust boundary.
- `sentence-transformers` loads an application-selected model at a required immutable 40-character
  commit revision. The model declares its supported modalities.
- `openai` defaults to text embedding. For an OpenAI-compatible multimodal server, declare its
  `modalities` and use `request_format=input` for the embeddings array or
  `request_format=messages` for chat-style media parts. The request format is part of the embedding
  space.

Audio has one fallback: a configured speech backend can transcribe it for a text-only embedder.
Unsupported image or video input fails instead of being stored without visual evidence.

**Contract:** Changing a model, revision, dimension, or input recipe changes the vector space.
MindBridge refuses unrecognized store metadata mismatches instead of mixing spaces.

**Guidance:** Start a new data directory when intentionally changing embedding space. Reuse an
existing directory only for a supported bundled migration; see
[store metadata mismatch](troubleshooting.md#store-metadata-mismatch).

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
        "embedding": {"provider": "jina-omni"},
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
standard choice with a non-empty `message.content`. A `finish_reason` of `length` raises
`ModelOutputTruncatedError`; increase `max_tokens` instead of accepting partial proposals.

Formation stays off when the slot is omitted. Configuring `generation` does not enable it, and the
two slots use separate clients so they may select different models, endpoints, and credentials.
`modalities` controls which observations reach the former; unsupported observations are skipped.
Formation is one completion per `add`, or one batched completion per `add_many`, on the write path.
A failed call leaves the source observation committed, and retry tracking prevents duplicate
proposals for sources already formed by the same recipe.

MindBridge validates the response envelope and every proposal. Invalid envelopes fail the call;
invalid individual proposals are dropped and counted by
`mindbridge.formation.dropped_proposals`. See
[memory types, time, and decay](memory-types-time-and-decay.md) for the resulting typed records and
visibility rules.

## Visual descriptions

The `vision` slot is omitted by default and, like `formation`, must stay omitted unless the
deployment wants it. Configuring it adds one chat completion per write batch that carries a
not-yet-described image or video, and buys the derived text that lets the lexical half of
retrieval reach a visual memory at all.

```python
config = {
    "embedding": {"provider": "jina-omni"},
    "vision": {
        "provider": "openai",
        "model": "gpt-5-mini",
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

Three things bound what it costs and what it can do:

- `modalities` is the visual capability set and accepts only `image` and `video`. An asset outside
  it is left undescribed, silently and by design.
- Video is described from four ordered stills decoded locally, never by uploading the file, so it
  costs four image parts per memory; that is why the default is `[image]` and video is opted into.
  A clip with no readable video stream fails the write rather than falling back to sending it. The
  request marks such a visual with its still count (`Visual 2, as 4 ordered stills:`) and asks for
  one caption per visual, never one per still: asked over an unmarked clip, a measured endpoint
  returned four separate descriptions on every attempt, the one-caption-per-input contract below
  rejected the reply whole, and `modalities: [image, video]` therefore stored no caption at all
  while still paying for the request.
- The caption is derived text, so the request carries pixels and an ordinal only -- no memory ID,
  file name, or store path -- and a reply that does not return exactly one non-empty caption per
  visual is rejected whole rather than mislabelling a memory with another's contents. One
  malformed reply is retried once, because an endpoint can answer `200 OK` with invalid JSON and
  an SDK retry policy never sees that; a second failure, or any other failure, leaves the memory
  stored **without** a caption rather than failing the write. Losing derived text must never lose
  an observation the caller handed over. Those batches are counted on the vision span as
  `mindbridge.vision.failed_batches`, so the loss is measurable rather than silent.
- Captions are not reproducible. The request pins `temperature` 0 and a fixed `seed` unless the
  slot sets its own, but a measured endpoint returned four different completions for four
  identical requests at those values. A caption becomes indexed text, so anything that needs two
  ingests of one corpus to build the same documents has to cache descriptions by asset digest;
  the benchmark harness does exactly that.

Description tokens are reported under their own model module, so a deployment that meters usage
can separate what captioning costs from what answering costs.

## Agentic memory management

The optional `consolidation` slot builds the bundled OpenAI-compatible `ConsolidationBackend`,
which is what `consolidate()` needs. It reads the same completion knobs as `formation`:

```python
with Memory.from_config(
    {
        "data_dir": "./data/assistant",
        "embedding": {"provider": "jina-omni"},
        "consolidation": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "modalities": ["text", "image", "audio"],
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
it flips `consolidate` on in the capability document that `/healthz`, the MCP server
instructions, and `mindbridge doctor` publish. Unlike formation it is not on the write path: it
runs only when a host calls `consolidate()`.

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

The `settings` mapping is the value-only `MemorySettings` policy (`MemoryConfig` is a compatible
alias):

| Field | Default | Meaning |
| --- | --- | --- |
| `index_speech` | `True` | Persist configured speech analysis during `add` |
| `index_quantization` | `none` | Zvec projection mode: `none`, `fp16`, `int8`, or `rabitq` |
| `minimum_relevance` | `0.10` | Floor on query-relevant evidence before retention and reinforcement ranking |
| `ambiguity_margin` | `0.01` | Withhold an unresolved top-two tie when `limit=1` |
| `evidence_budget_chars` | `None` | Widen `ask` grounding while the evidence fits this budget; raises a floor, never a ceiling; `None` grounds on exactly `limit` |
| `decay_half_life_days` | `None` | Optional positive half-life for query-time decay |
| `reinforce_on_answer` | `True` | Count the evidence `ask()` cited, so retrieval favours it later |
| `speaker_similarity` | `0.78` | Voice identity match threshold (uncalibrated; see below) |
| `speaker_margin` | `0.05` | Voice identity ambiguity margin |
| `face_similarity` | `0.363` | Face identity match threshold |
| `face_margin` | `0.05` | Face identity ambiguity margin |
| `identity_link_min_assets` | `2` | Distinct assets a voice-and-face pair must share before they merge |
| `memory_budget_records` | `None` | Active records this instance is meant to hold; the whole definition of the `PRESSURE` trigger, which derives nothing without it |
| `query_failure_window_seconds` | `3600.0` | How far back the `QUERY_FAILURE` trigger counts near-equal empty recalls |
| `query_failure_history` | `512` | How many empty recalls the store keeps at all; the oldest fall out |
| `retention` | empty | What `apply_retention()` may delete; spell it as the top-level `retention` section below |

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

## Retention policy

`retention` is its own top-level section rather than a `settings` field, because every other
setting shapes what recall returns and can be changed back, and this one deletes.

```python
config = {
    "embedding": {"provider": "jina-omni"},
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

`MemoryConfig`/`MemorySettings` carries the same value under the name `retention`, which is what
`Memory.from_plugins()` and the `Memory(retention=...)` constructor argument read. Declaring it in
both places is refused rather than silently resolved.

## Direct adapter injection

Direct injection is the boundary for custom protocols, caller-owned SDK clients, proxies, and
connection pooling:

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
models = OpenAIModels(generation_model="gpt-5-mini")
memory = Memory(
    "./data/assistant",
    embedder=JinaOmniEmbedder(),
    answerer=models,
    former=models,
    consolidator=models,
)
```

`Memory` closes each distinct adapter once, so passing one object to several slots is safe.
Omitting `consolidator=` leaves `consolidate()` unavailable and every other operation unchanged;
see [the memory management loop](memory-types-time-and-decay.md#memory-management-loop) for what a
consolidator is allowed to propose.

Use `MemoryPlugins` with `Memory.from_plugins()` when constructed adapters should travel as one
value. `resolve_memory_config()` is for hosts that need a `MemoryComposition` before opening
storage; close it unless its plugins are transferred to a `Memory`. Ordinary applications should
prefer `Memory.from_config()`.

For exact constructor, protocol, and configuration types, use the
[Python SDK reference](api/python-sdk.md). For benchmark-only environment variables, use
[benchmarking](benchmarking.md).
