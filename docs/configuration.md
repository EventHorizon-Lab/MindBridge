# Configuration

Credentials are configured entirely through environment variables, and are never accepted as a
CLI flag or read from a file — so a recorded invocation, a process list, a systemd unit, and the
repository itself never carry a secret. Everything that is not a credential lives in
`mindbridge.toml`, which is committed, commented, and diffable.

Each setting's environment variable overrides the file, so a container or a CI job changes one
value without rebuilding anything. `mindbridge config check --role <role>` reports which source
won for each setting, and every setting a role still needs, in one pass rather than one per
restart.

A credential key inside `mindbridge.toml` is refused when the file loads, and so is a key no
reader looks up: a typo that is ignored is a value that silently reverts to its default.

Configuration is validated at startup, not at first request. A deployment with a wrong value
fails to start rather than failing one call an hour later.

## Which process reads what

| Variable | Source | API | MCP | Worker | Consolidate | Lifecycle | Edge sync |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `MINDBRIDGE_CONFIG_FILE` | env | ○ | ○ | ○ | ○ | ○ | ○ |
| `MINDBRIDGE_DATABASE_URL` | env | ● | ● | ● | ● | ● | |
| `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` | file | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_TASK_BROKER_URL` | env | ● | | ● | | | |
| `MINDBRIDGE_OBJECT_STORAGE_BUCKET` | file | ● | ● | ● | ● | ◐ | |
| `MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` | file | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL` | file | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_GENERATOR_*` | both | ● | ● | ● | ● | | |
| `MINDBRIDGE_EMBEDDER_*` | both | ● | ● | ● | ● | | |
| `MINDBRIDGE_EMBEDDING_*` | file | ○ | ○ | ○ | ○ | | |
| `MINDBRIDGE_MEDIA_EMBEDDER_*` | file | | | ○ | | | |
| `MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON` | file | | | ○ | | | |
| `MINDBRIDGE_WORKER_CONCURRENCY` | file | | | ○ | | | |
| `MINDBRIDGE_WORKER_VRAM_BUDGET_GIB` | file | | | ○ | | | |
| `MINDBRIDGE_CONSOLIDATION_TENANT_IDS` | file | | | ○ | | | |
| `MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS` | file | | | ○ | | | |
| `MINDBRIDGE_TENANT_API_KEYS_JSON` | env | ● | | | | | |
| `MINDBRIDGE_API_KEY` | env | | | | | | ● |
| `MINDBRIDGE_AML_*` | both | ○ | | | | | |
| `MINDBRIDGE_LOG_LEVEL` | file | ○ | ○ | ○ | ○ | ○ | ○ |
| `MINDBRIDGE_LOG_FORMAT` | file | ○ | ○ | ○ | ○ | ○ | ○ |
| `MINDBRIDGE_TIMING_SUMMARY` | file | ○ | ○ | ○ | ○ | ○ | ○ |

● required ○ optional ◐ required only with `--reclaim-orphan-clips`

**Source** is where a setting is configured: `file` in `mindbridge.toml`, `env` in the
environment only, `both` for a group whose API key is a credential and whose remaining keys are
structure. Every `file` row is also settable from the environment, which overrides the file.

## The configuration file

`mindbridge.toml` is read from the working directory. `MINDBRIDGE_CONFIG_FILE` names a different
path, and a path it names that is not a file is an error rather than a fall back — there is no
parent-directory search and no XDG lookup, because a configuration file found somewhere nobody
named is worse than none at all.

No file at all is not an error either. A deployment that sets everything in the environment
behaves exactly as it did before this file existed.

| Condition | Behaviour |
| --- | --- |
| `MINDBRIDGE_CONFIG_FILE` set, path missing | Error naming the path. |
| File is not valid TOML | Error naming the file and the parse position. |
| File contains a credential key | Error naming the key and the variable it belongs in. |
| File has an unknown section or key | Error naming it. |
| Both file and environment set a value | The environment wins. `config check` shows which. |
| No file at all | Not an error. |

A key `k` in section `s` configures `MINDBRIDGE_<S>_<K>`, and a key at the top level configures
`MINDBRIDGE_<K>`. That derivation is the whole mapping; the tables in this document are written
from it rather than beside it.

## Storage

### `MINDBRIDGE_DATABASE_URL`

**Required.** PostgreSQL DSN, e.g.
`postgresql://mindbridge:password@db.internal:5432/mindbridge`.

The login must be able to `SET ROLE mindbridge_runtime`. Grant that role to the API login when
your migration user and runtime user differ. Never give the runtime login `SUPERUSER` or
`BYPASSRLS` — either one disables tenant row-level security completely.

### `MINDBRIDGE_DATABASE_MAX_POOL_SIZE`

Optional, default `32`, minimum `1`. The ceiling on pooled connections, read identically by
**every** process that opens a pool.

This is not a per-process budget. Four processes at 32 ask for 128 connections against a default
`max_connections` of 100. Size it against your server's actual limit and the number of MindBridge
processes you run.

A value near ten is a trap: one recall alone peaks near ten connections, because a lexical search
runs concurrently with three vector searches, then four memory searches, and a reflection round
runs several such waves at once. The pool opens only one connection eagerly, so a higher ceiling
costs nothing until load asks for it.

### `MINDBRIDGE_TASK_BROKER_URL`

**Required** for the API and the worker. Redis DSN, e.g. `redis://redis.internal:6379/0`. The
consolidation and lifecycle sweeps need no broker.

### `MINDBRIDGE_OBJECT_STORAGE_BUCKET`

**Required** wherever media is read or written. Media URIs must take the tenant-safe shape
`s3://<bucket>/tenants/<tenant_id>/<key>`.

### `MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL`

Optional. Omit for AWS S3 itself; set it for MinIO, Ceph, or another S3-compatible store.

### `MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL`

Optional, defaults to the endpoint above. Set it only when a hosted model must fetch signed
evidence over a different address than the one the deployment reads and writes through. Signed
URLs then name the public address while every internal read, write, and delete stays on the
direct one, instead of leaving and re-entering the network.

Two constraints:

- Signatures cover the host, so both names must reach the same bucket.
- It moves **every** signed URL, including the derived-clip URLs a self-hosted embedder fetches.
  Set it only where that endpoint can also reach the public name.

### AWS credentials and region

MindBridge owns neither. Boto3's own chain resolves them from `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_DEFAULT_REGION`, `~/.aws/config`, or instance
metadata — exactly as for every other tool on the host, so one AWS configuration serves them
all.

S3-compatible stores that ignore the region still need one set. `AWS_DEFAULT_REGION=us-east-1`
is the conventional choice.

## Models

Three slots. Each is selected by a lowercase plugin name and configured either through the
bundled per-field variables or through one explicit JSON object.

| Slot | Plugin variable | Default plugin |
| --- | --- | --- |
| Generator | `MINDBRIDGE_GENERATOR_PLUGIN` | `openai` |
| Text embedder | `MINDBRIDGE_EMBEDDER_PLUGIN` | `openai` |
| Media embedder | `MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN` | inherits `MINDBRIDGE_EMBEDDER_PLUGIN` |

`openai` means "any OpenAI-compatible endpoint", including one you serve yourself, and it is the
default for **all three** slots. `sentence-transformers` loads a local model into the process;
`jina` remains its compatibility alias. See
[optional local media embedder](#optional-local-media-embedder-worker-only) for what that
costs and when the worker refuses to run it.

### Generator

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_GENERATOR_API_KEY` | yes | — |
| `MINDBRIDGE_GENERATOR_ENDPOINT` | yes | — |
| `MINDBRIDGE_GENERATOR_MODEL_ID` | no | `qwen3.8-max` |

`request_timeout_seconds` has no fallback variable — it lives in
`MINDBRIDGE_GENERATOR_CONFIG_JSON` and defaults to **1800**. It is worth knowing because it does
double duty: the worker derives its whole Celery task budget from it, adding a fixed 300-second
allowance for the encoding and graph write after the model call. Raising it is how a deployment on
a slow generator moves both deadlines at once. See
[deployment](deployment.md#how-long-one-observation-may-take).

**It has a ceiling of 1920 seconds, and the worker refuses to start above it.** One delivery may
stay alive for this timeout plus 480 seconds — the 300-second post-model allowance, the 60-second
hard-limit margin, and the 120 seconds the broker waits before re-delivering — and the job
ledger's stale-claim window is a fixed 2400. Past the ceiling the two disagree about the same
attempt: the delivery is still paying for its model call while the ledger already treats the row
as reclaimable, so a concurrent delivery or one `mindbridge jobs --republish` buys the same
observation twice. The boot failure names the largest value that fits.

### Shared embedder

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_EMBEDDER_API_KEY` | yes | — |
| `MINDBRIDGE_EMBEDDER_ENDPOINT` | yes | — |
| `MINDBRIDGE_EMBEDDER_MODEL_ID` | no | `jinaai/jina-embeddings-v5-omni-small-retrieval` |

The default `openai` plugin names the wire adapter, not the model runtime. The committed endpoint
points to `mindbridge sentence-transformers serve`, which loads Jina v5 Omni by default. The API,
MCP, consolidation jobs, and both worker slots use that one service by default.

### Optional local media embedder (worker only)

The served shape is the default; the local model is an explicit opt-in.

**Served (recommended).** One variable, because the media slot then reuses the text embedder's
endpoint — it has to write into the same embedding space anyway:

```bash
export MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=openai
```

**In-process (`sentence-transformers`, optional).** Loads the encoder into the worker itself:

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID` | no | `jinaai/jina-embeddings-v5-omni-small-retrieval` |
| `MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION` | no | `12949877f0092093f366c6450340011320152a05` |
| `MINDBRIDGE_MEDIA_EMBEDDER_DEVICE` | no | automatic selection |
| `MINDBRIDGE_WORKER_VRAM_BUDGET_GIB` | no | `3.7` — one model copy per child |

These variables are read only when a `MINDBRIDGE_MEDIA_EMBEDDER_*` override is present. Setting
`MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=sentence-transformers` opts the worker into loading a second,
local model. Without that explicit override, media inherits the shared endpoint. The old `jina`
plugin name selects the same adapter.

`MODEL_REVISION` is the upstream commit this plugin loads. Bundled Jina Omni also defaults
`trust_remote_code` to `true` and pins that code to the same commit. Other models default it to
`false`; a model that needs repository code must opt in with `trust_remote_code=true` in the
`[media_embedder]` object or `MINDBRIDGE_MEDIA_EMBEDDER_CONFIG_JSON`. Only the local plugin takes a
revision. The endpoint-backed plugin talks to a server that resolves its own model.

Changing this value is a deliberate act, and it needs `MINDBRIDGE_EMBEDDING_SPACE_ID` changed
with it. Vectors are comparable only within a space, `SPACE_ID` is what declares that
boundary, and it does not vary with the checkout — so a new revision written under the old
`SPACE_ID` puts two encoders' output in one cosine search with nothing able to report it.

An explicit `DEVICE` that is unavailable fails rather than silently falling back to CPU.

The in-process plugin costs **3.7 GiB of resident weights per prefork child** on the measured RTX
5090. The worker **refuses to start** when a
pool of more than one child would hold more than `MINDBRIDGE_WORKER_VRAM_BUDGET_GIB`, while either
embedder slot names `sentence-transformers` or `jina`, because a prefork child holds its own copy
of the model and
`--max-memory-per-child` cannot bound VRAM. The pool size is whichever of `--concurrency` and
`--autoscale` is larger. Only `prefork` and `solo` are supported: `solo` holds one copy,
while thread and greenlet pools are refused because the worker runtime owns one synchronous event
loop per process. `MINDBRIDGE_MEDIA_EMBEDDER_DEVICE=cpu` holds no VRAM and is exempt from the
device-memory calculation.

`MINDBRIDGE_WORKER_VRAM_BUDGET_GIB` defaults to **3.7 GiB, one model copy per child**, which is
what keeps a second resident copy a decision rather than an accident. Raise it on a card that can
genuinely hold more — `MINDBRIDGE_WORKER_VRAM_BUDGET_GIB=48` admits six children with one loaded
slot each. It exists because a guard with no way to say yes gets routed around, and the route an
operator reaches for is `--autoscale`, which the guard cannot see from `--concurrency` alone.
**The estimate it bounds counts resident weights and CUDA contexts only, not activation memory**:
the evaluation's six children measured 30.2 GB against an estimated 22.2, so leave room. A value
that is not a **finite** positive number fails startup rather than disabling the guard — `Infinity`
parses, and every estimate compares below it, so a typo in the one variable that raises the guard
would otherwise switch it off.

Switching an existing deployment from `jina` to `openai` is not free: video vectors from the two
backends agree only to cosine **0.944**, because they sample different numbers of frames from the
same clip. Text agrees to 0.99994 and images to 0.985. Re-encode media evidence, or accept that
old and new video vectors are not strictly comparable.

### Plugin sections

A plugin's whole configuration is one object, and `mindbridge.toml` is where it is written. Its
keys are the plugin's own — `reasoning_effort` below belongs to the bundled OpenAI generator, not
to MindBridge:

```toml
[generator]
plugin = "openai"
endpoint = "https://generator.example.com/v1"
model_id = "qwen3.8-max"
reasoning_effort = "low"
```

Also `[embedder]`, `[media_embedder]`, and `[media_sampling]`. `[embedding]`'s two keys are
folded into `[embedder]` and `[media_embedder]` when they are read, so one space is stated once.

An individual `MINDBRIDGE_<SECTION>_<KEY>` variable overrides one key of a section, in the type
the file declared for it: `MINDBRIDGE_GENERATOR_REQUEST_TIMEOUT_SECONDS=900` against a file
saying `request_timeout_seconds = 1800` resolves to the number 900, not the string. The API key
arrives by the same mechanism, which is why it never has to appear in the file.

`MINDBRIDGE_GENERATOR_CONFIG_JSON` still exists and still **replaces** the whole section rather
than merging into it. An opaque object that is half overridden is not something a plugin schema
can validate, so the environment either supplies the object or overrides its keys one at a time.

The rule that keeps this surface bounded: **a variable exists for a credential or for model
identity** — what a deployment cannot start without. Every other knob is a key of a section, not
a variable of its own. Without that rule, the variable list grows with every setting any plugin
ever gains.

Configs are validated with `extra="forbid"`. An unrecognized key fails startup rather than being
ignored, which is the difference between "that setting had no effect" and "that setting was
never applied and nobody noticed". The three names migration `0021` retired are the one closed
exception, described in [plugin-architecture.md](plugin-architecture.md).

Anthropic, Gemini, local runtimes, and experimental adapters need no OpenAI-specific variables —
set the plugin name and provide its JSON. See [plugin-architecture.md](plugin-architecture.md).

## Embedding space

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_EMBEDDING_SPACE_ID` | no | `jinaai/jina-embeddings-v5-omni-small-retrieval-1024` |
| `MINDBRIDGE_EMBEDDING_DIMENSION` | no | `1024` |
| `MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY` | no | `0.0` |

`SPACE_ID` names the compatibility space the selected embedder writes into and queries. It is
separate from the encoder's own identity because several independently served encoders can write
into one comparable space.

`EMBEDDING_DIMENSION` is one width shared by the pgvector column and every encoder in the
deployment. The local adapter accepts the selected model's native width, or a smaller width only
when that exact value appears in the model's advertised Matryoshka dimensions. Changing the model,
revision, or dimension requires a new `SPACE_ID` and re-encoding every searchable object; the
bundled default space is rejected with any other model or revision. Give every process the same
new values. Startup refuses tenants that still have an object type only in the old space, so a
model switch cannot silently turn recall empty.

`MINIMUM_EMBEDDING_SIMILARITY` accepts −1.0 to 1.0. The default of 0.0 admits every
non-antipodal candidate and lets fusion and the answer stage do the filtering, which is usually
right: a similarity floor discards candidates a graph hop or a lexical match would have rescued.

**What raising it buys.** At 0.0 a recall returns exactly `limit` memories whenever the tenant
holds at least that many, so it has no way to report "I hold nothing about this" — it reports the
`limit` least dissimilar rows instead. Raising the floor lets recall return fewer rows, and zero
rows, and an empty recall abstains rather than answering from whatever came back. That is a
change in what the system can express, not a ranking improvement: on a nine-benchmark evaluation
a retrieval hit did not predict answer correctness on three of the corpora, so do not expect a
floor to raise accuracy by putting better rows in the page.

**Where it helps, and where it does not.** Long-horizon, sparsely-covered corpora — hours or days
of footage where most of the store is unrelated to any one question — are the shape where the
`limit` least dissimilar rows are actively misleading. On densely covered corpora, where almost
everything retrieved is at least on topic, a floor only removes candidates fusion was already
ordering correctly. Leave it at 0.0 unless you have measured your deployment in the first shape,
and change it one deployment at a time: there is no value that is right for both.

**It only binds the dense channel.** The floor is applied to every vector search. The full-text
channel fused beside them has no floor of its own — it matches on the query's lexemes ORed
together plus a substring test, neither of which produces a similarity to compare — so a floored
text query returns what the vector searches admitted *plus* whatever shares a word with the
question. A media-only query has no full-text channel and is bounded by the floor alone.

**Startup probe.** The API probes every tenant in `MINDBRIDGE_TENANT_API_KEYS_JSON` and refuses
to serve when one holds vectors the configured space cannot reach. Pointing a deployment at a new
embedder without re-embedding therefore fails loudly instead of returning empty recalls. The
probe reports each stranded object type separately, so memory records the server wrote itself
cannot vouch for evidence, events, and claims the worker wrote in another space. Vectors in
several spaces are accepted while a re-embedding is in progress, for every object type: an
`embedding_id` hashes the space it was produced for, so one object holds one vector per space
rather than colliding on the primary key. One object may also hold several vectors within one
space, where it is embedded in pieces rather than whole — an evidence span longer than the
encoder's audio window is cut into clips, and migration `0027` gives each its own row. Migration `0026` records which recipe derived each
stored ID, and the first write to touch a row an older recipe named re-keys it in place — so the
first pass after that migration pays for one re-encode per object it has not seen under its new
name, and every pass after it does not.

The stdio MCP process does not require the tenant key map for authentication. When
`MINDBRIDGE_TENANT_API_KEYS_JSON` is present, the shared runtime uses its tenant IDs for this
probe; when it is absent, MCP has no tenants to probe at startup.

## Authentication

### `MINDBRIDGE_TENANT_API_KEYS_JSON`

**Required for the REST API.** The factory refuses to build without it; there is no anonymous
mode.

```json
{
  "tenant_01": ["at-least-32-characters-long"],
  "tenant_02": ["key-a-being-retired", "key-b-taking-over"]
}
```

Each key is bound to an explicit tenant allowlist. Two or more keys per tenant is how you rotate
without downtime. One isolated benchmark deployment can authorize all its generated tenants with
the same key — but every generated tenant ID has to be in the mapping before the API starts.

Blank or short keys (under 32 characters) fail startup. No plaintext is retained, only a digest.

### `MINDBRIDGE_API_KEY`

Read by `mindbridge edge sync` and the benchmark runners as the bearer token for the API they
call. It is a client credential, not a server one.

## Media sampling (worker)

`[media_sampling]` is optional; unset keys keep the defaults shown:

```toml
[media_sampling]
frames_per_second = 1.0
max_pixels = 200704
image_max_pixels = 1003520
generation_proxy = true
proxy_audio = true
```

An unrecognized key or a value of the wrong type fails startup, and so does a frame rate outside
`0 < fps <= 20`. The upper bound is the media layer's own: past 20 fps every span widened to the
sampling floor exceeds the proxy frame ceiling below, so the knob would silently switch off the
feature it is tuning. `Infinity` is well-formed JSON and is refused here rather than downstream.

**Frame rate sets the entire write cost of a video deployment** — one clip cut, one encoder call,
and one stored object per sampled window. It is the first thing to change if ingest is too
expensive.

`generation_proxy` decides what a generation request downloads. Every request already carries the
frame rate and pixel budget the model must apply, so handing over the untouched source makes the
model fetch frames it discards on arrival. With the proxy on, video is cut once to that budget
and the sampled copy is what perception reads. Turn it off for a generator that reads the same
storage the worker does, where the encode costs more than the transfer it removes.

`proxy_audio` decides whether the copy carries the source's audio track. Keep it on for a
generator that listens: a video-only proxy silently takes speech away from every question that
depends on what was said. Turn it off for one that does not. Whether yours does is worth
measuring rather than assuming — send the same clip twice, once with its audio track and once
without, and compare `prompt_tokens`. Against the endpoint used for the 2026-08-21 evaluation the
count was identical at 1009 either way, so the track was never ingested, while the file was
336 KiB with it against 212 KiB without: an encode and a transfer bought nothing. Note also that
such an endpoint will still answer "what was said" with fluent invented dialogue rather than
saying it heard nothing, so a silent deployment does not announce itself.

Four constraints on the proxy, all of which have bitten before:

- **Video only.** `image_max_pixels` governs stored image clips, not what the model is sent. An
  image reaches the model at full resolution because the request carries no pixel budget for
  images at all.
- **Its ceiling is a frame count, not a duration.** Past roughly forty sampled frames the encode
  fails, on the flush that drains the encoder rather than on any one frame. At the 30-second
  segments every ingest path here uses, anything above about 1.3 fps exceeds it. Raising
  `frames_per_second` therefore trades the proxy away; lower it, or segment shorter, to keep
  both. **Turning `proxy_audio` off does not raise this ceiling** — a silent source with the
  audio disabled fails at the same frame count, so the limit is not the audio interleave it was
  previously documented as. What has been ruled out, and what has not, is recorded next to
  `MAX_PROXY_SAMPLED_FRAMES` in `application/evidence_clips.py`.
- **Best-effort.** A span over budget is skipped before its source is read, and anything the
  encoder or object storage refuses degrades the same way — the observation behaves exactly as
  it did before this knob existed rather than paying for a doomed encode.
- **It costs one extra read of the source**, because clip derivation reads it again after the
  model call rather than holding the whole recording in memory across it. That read is internal,
  so a deployment whose object storage is only reachable over a slow public address should set
  `MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL` and keep the direct endpoint local.

A proxy that did not come out smaller is discarded. The copy is lent for one model call and
deleted when it returns, including on a failed attempt: nothing registers it, so it is never
cited as provenance, and leaving it behind would put a re-encoded copy of an observation's
picture and speech beyond the reach of `forget()`.

## Worker throughput

`MINDBRIDGE_WORKER_CONCURRENCY` is optional and defaults to **1**: how many observations one
worker may have in flight at once.

One observation is one model call and then some encoding, so a worker against remote model
endpoints spends most of its budget waiting on the network. At the default it waits on one
observation at a time, which is the single largest throughput lever a deployment has.

It defaults to 1 because the ceiling is not the network in every deployment. Celery's prefork
pool builds the media embedder once per child, so:

- **Models served over the network** (an OpenAI-compatible endpoint for the media embedder):
  raise it. Each child holds HTTP clients, not weights.
- **A media embedder that loads its model in-process** (the bundled `jina` plugin): this
  multiplies device memory rather than overlapping anything. Leave it at 1 and add worker
  processes on separate hosts instead.

Each child also opens its own database pool, so `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` is a
per-child ceiling here, not a per-deployment one. Values outside 1–32, and anything that is not
an integer, fail startup rather than being clamped.

## Built-in consolidation schedule

| Variable | File key | Required | Default |
| --- | --- | --- | --- |
| `MINDBRIDGE_CONSOLIDATION_TENANT_IDS` | `[consolidation] tenant_ids` | no | unset — no schedule is registered at all |
| `MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS` | `[consolidation] interval_seconds` | no | `3600` |

Neither is a credential, so both belong in `mindbridge.toml` and the environment overrides it:

```toml
[consolidation]
tenant_ids = "tenant_01,tenant_02"
interval_seconds = 3600
```

Read by `celery -A mindbridge.celery_app:app beat` and by the worker that consumes
`mindbridge_consolidation`; both processes need the same tenant list, because beat's ticks address
whichever list the worker was given. Leaving `TENANT_IDS` unset registers no schedule and no task,
so a deployment that has not opted in runs exactly the worker it ran before.

One tick sweeps **one** tenant, chosen by rotation, so the interval is a ceiling on
consolidation's entire share of the generator rather than a per-tenant cadence — adding tenants
lengthens the rotation instead of raising load. Scheduled sweeps skip entity resolution, which is
the only sweep that opens media and spends a generator call per candidate pair; run that from the
CLI on its own cadence.

Tenants are enumerated from this variable rather than discovered: a tenant that is created and not
added here silently stops consolidating. See
[operations](operations.md#built-in-consolidation-schedule).

## Logs and timings

Every process writes structured logs to stderr. This is unconditional and needs no collector:
each instrumented operation logs its own duration and outcome on completion, so the read and
write paths are attributable from `docker logs` alone.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINDBRIDGE_LOG_LEVEL` | `INFO` | Standard level name. `WARNING` silences the per-operation stream and keeps failures. |
| `MINDBRIDGE_LOG_FORMAT` | auto | `json` for a log shipper, `text` for a terminal. Unset picks `text` on a TTY and `json` otherwise. |
| `MINDBRIDGE_TIMING_SUMMARY` | unset | Set `1` to log a ranked per-operation cost summary at process exit. |

An unusable level or format fails startup; falling back silently to `INFO` is worse to debug
than refusing to start.

Each record carries `operation`, `duration_ms`, `self_ms`, `outcome`, and — whenever a span is
active — `trace_id` and `span_id`, so a reported `trace_id` joins the logs to the trace. Only
the MindBridge logger namespace is configured, never the root logger: this package is embedded
as a library as well as run as a service.

`MINDBRIDGE_TIMING_SUMMARY=1` is the answer to "where did the time go". It emits one row per
operation, ranked by **self time** — duration minus nested instrumented operations — because
ranked by total time the outermost operation always wins and explains nothing. An operation that
only gathers others reports near-zero self time by construction; that is the intended reading,
not a missing measurement.

`mindbridge-bench` logs that summary at the end of every run, successful or not, without the
variable: a measurement run's own cost breakdown is part of its result.

## Telemetry

OpenTelemetry activates per signal, according to the exporter that signal is configured for.
The default exporter is `otlp`, which needs an endpoint and stays a no-op without one. `console`
needs no endpoint and renders the same instruments into the process's own output, which is how a
box with no collector reads its stage timings and token counts. `none` turns the signal off.

| Variable | Purpose |
| --- | --- |
| `OTEL_TRACES_EXPORTER`, `OTEL_METRICS_EXPORTER` | `otlp` (default), `console`, or `none`. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector address, e.g. `http://otel-collector:4318`. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Signal-specific override. |
| `OTEL_SERVICE_NAME` | Override the per-process default. |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Standard sampler configuration. |
| `OTEL_SDK_DISABLED` | Set `true` for an explicit process-level opt-out. |

Two measurements deliberately do not depend on any of this, because losing them to a sampling
decision would misreport what happened rather than merely fail to describe it. What a model
charged for an observation is written to that observation's job row. What a stage discarded or
rewrote from a model's answer — dropped events, entities and claims, and claim types resolved
through an alias — is also logged at `WARNING`, so a low memory count can be told apart from a
model whose output was mostly thrown away.

Each process has a distinct default `service.name`, so they are already separable without
configuration. MindBridge captures no authorization headers, request bodies, prompts, memory
text, or media.

## Benchmark harness

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINDBRIDGE_AML_API_KEY` | unset | Enables `POST /aml/add` and `/aml/search`. Leave off in production. |
| `MINDBRIDGE_AML_TENANT_PREFIX` | `bench_aml` | Tenant prefix the AML harness generates under. |
| `MINDBRIDGE_BENCH_JUDGE_ENDPOINT` | unset | OpenAI-compatible endpoint the judge is called on. Required by the seven benchmarks whose answers are free text; `--predict-only` runs them without one. |
| `MINDBRIDGE_BENCH_JUDGE_API_KEY` | unset | Bearer token for that endpoint. Environment-only, so a recorded invocation never carries it. |
| `MINDBRIDGE_BENCH_JUDGE_MODEL` | `gpt-4o-2024-11-20` | Judge model, defaulting to the one lmms-eval's MM-Vet task uses. Recorded in every manifest it scores: two runs under different judges are not comparable. |

## Development and test

| Variable | Purpose |
| --- | --- |
| `MINDBRIDGE_TRACEBACK` | Set `1` to keep full tracebacks behind CLI failures instead of the one-line form. |
| `MINDBRIDGE_TEST_DATABASE_URL` | Disposable test database. Its name **must** end in `_test`; the fixture refuses to rebuild anything else. |
| `MINDBRIDGE_REQUIRE_INTEGRATION` | Set `1` to turn a missing test database into a failure instead of a skip. |

Without `MINDBRIDGE_TEST_DATABASE_URL` the entire integration suite skips, so a green run may
never have touched the production store. See [contributing](../CONTRIBUTING.md#quality-gates).
