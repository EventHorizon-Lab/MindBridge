# Configuration

Every process is configured entirely through environment variables. There is no config file, and
credentials are never accepted as CLI flags — so a recorded invocation, a process list, and a
systemd unit never carry a secret.

Configuration is validated at startup, not at first request. A deployment with a wrong value
fails to start rather than failing one call an hour later.

## Which process reads what

| Variable | API | MCP | Worker | Consolidate | Lifecycle | Edge sync |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| `MINDBRIDGE_DATABASE_URL` | ● | ● | ● | ● | ● | |
| `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_TASK_BROKER_URL` | ● | | ● | | | |
| `MINDBRIDGE_OBJECT_STORAGE_BUCKET` | ● | ● | ● | ● | ◐ | |
| `MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL` | ○ | ○ | ○ | ○ | ○ | |
| `MINDBRIDGE_GENERATOR_*` | ● | ● | ● | ● | | |
| `MINDBRIDGE_EMBEDDER_*` | ● | ● | ● | ● | | |
| `MINDBRIDGE_EMBEDDING_*` | ○ | ○ | ○ | ○ | | |
| `MINDBRIDGE_MEDIA_EMBEDDER_*` | | | ○ | | | |
| `MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON` | | | ○ | | | |
| `MINDBRIDGE_TENANT_API_KEYS_JSON` | ● | | | | | |
| `MINDBRIDGE_API_KEY` | | | | | | ● |
| `MINDBRIDGE_AML_*` | ○ | | | | | |

● required ○ optional ◐ required only with `--reclaim-orphan-clips`

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
| Media embedder | `MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN` | `jina` |

### Generator

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_GENERATOR_API_KEY` | yes | — |
| `MINDBRIDGE_GENERATOR_ENDPOINT` | yes | — |
| `MINDBRIDGE_GENERATOR_MODEL_ID` | no | `qwen3.8-max` |

The credentials are required and the model ID is not: an endpoint has no sensible default, while
the model behind it does.

`request_timeout_seconds` has no fallback variable — it lives in
`MINDBRIDGE_GENERATOR_CONFIG_JSON` and defaults to **1800**. It is worth knowing because it does
double duty: the worker derives its whole Celery task budget from it, adding a fixed 300-second
allowance for the encoding and graph write after the model call. Raising it is how a deployment on
a slow generator moves both deadlines at once. See
[deployment](deployment.md#how-long-one-observation-may-take).

### Text embedder

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_EMBEDDER_API_KEY` | yes | — |
| `MINDBRIDGE_EMBEDDER_ENDPOINT` | yes | — |
| `MINDBRIDGE_EMBEDDER_MODEL_ID` | no | `jinaai/jina-embeddings-v5-omni-small-retrieval` |

The worker's text slot deliberately reads these same names rather than a parallel family. It has
to land in the space the API queries, and a second name is a second thing that can silently
disagree. A worker that genuinely needs a different endpoint sets a different value for the same
name — each process has its own environment.

### Media embedder (worker only)

| Variable | Required | Default |
| --- | --- | --- |
| `MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID` | no | `jinaai/jina-embeddings-v5-omni-small-retrieval` |
| `MINDBRIDGE_MEDIA_EMBEDDER_DEVICE` | no | automatic selection |

`MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID` is a Hugging Face repository ID; `MINDBRIDGE_EMBEDDER_MODEL_ID`
is an endpoint-side alias. They frequently hold the same string and are still not the same field
— do not consolidate them.

An explicit `DEVICE` that is unavailable fails rather than silently falling back to CPU.

### Plugin JSON

Every slot accepts one explicit JSON object that **replaces** the per-field variables entirely:

```bash
export MINDBRIDGE_GENERATOR_CONFIG_JSON='{
  "api_key": "...",
  "endpoint": "https://generator.example.com/v1",
  "model_id": "qwen3.8-max",
  "reasoning_effort": "low"
}'
```

Also `MINDBRIDGE_EMBEDDER_CONFIG_JSON` and `MINDBRIDGE_MEDIA_EMBEDDER_CONFIG_JSON`.

The rule that keeps this surface bounded: **fallback variables cover credentials and model
identity only** — what a deployment cannot start without. Every other knob lives in the
`*_CONFIG_JSON` object. Without that rule, the variable list grows with every setting any plugin
ever gains.

Configs are validated with `extra="forbid"`. An unrecognized key fails startup rather than being
ignored, which is the difference between "that setting had no effect" and "that setting was
never applied and nobody noticed".

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
into one comparable space, while a different encoder may not.

`EMBEDDING_DIMENSION` is one width shared by the pgvector column and every encoder in the
deployment. It accepts only widths Jina v5 was trained to truncate to — 32, 64, 128, 256, 512,
768, 1024 — because any other value is an untrained truncation that quietly degrades recall.
Changing it requires re-embedding, so set it once per deployment and give every process the same
value.

`MINIMUM_EMBEDDING_SIMILARITY` accepts −1.0 to 1.0. The default of 0.0 admits every
non-antipodal candidate and lets fusion and the answer stage do the filtering, which is usually
right: a similarity floor discards candidates a graph hop or a lexical match would have rescued.

**Startup probe.** The API probes every tenant in `MINDBRIDGE_TENANT_API_KEYS_JSON` and refuses
to serve when one holds vectors the configured space cannot reach. Pointing a deployment at a new
embedder without re-embedding therefore fails loudly instead of returning empty recalls. The
probe reports each stranded object type separately, so memory records the server wrote itself
cannot vouch for evidence, events, and claims the worker wrote in another space. Vectors in
several spaces are accepted while a re-embedding is in progress.

The stdio MCP process has no configured tenant list and therefore cannot run this probe.

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

`MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON` is optional; unset keys keep the defaults shown:

```bash
export MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON='{
  "frames_per_second": 1.0,
  "max_pixels": 200704,
  "image_max_pixels": 1003520,
  "generation_proxy": true
}'
```

An unrecognized key or a value of the wrong type fails startup.

**Frame rate sets the entire write cost of a video deployment** — one clip cut, one encoder call,
and one stored object per sampled window. It is the first thing to change if ingest is too
expensive.

`generation_proxy` decides what a generation request downloads. Every request already carries the
frame rate and pixel budget the model must apply, so handing over the untouched source makes the
model fetch frames it discards on arrival. With the proxy on, video is cut once to that budget
and the sampled copy is what perception reads. Turn it off for a generator that reads the same
storage the worker does, where the encode costs more than the transfer it removes.

Four constraints on the proxy, all of which have bitten before:

- **Video only.** `image_max_pixels` governs stored image clips, not what the model is sent. An
  image reaches the model at full resolution because the request carries no pixel budget for
  images at all.
- **Its ceiling is a frame count, not a duration.** Past roughly forty sampled frames the MP4
  muxer refuses to interleave a sparse video track with continuous audio. At the 30-second
  segments every ingest path here uses, anything above about 1.3 fps exceeds it. Raising
  `frames_per_second` therefore trades the proxy away; lower it, or segment shorter, to keep
  both.
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

## Telemetry

OpenTelemetry activates only when a standard common or signal-specific OTLP endpoint is set.
Without one it stays a no-op.

| Variable | Purpose |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector address, e.g. `http://otel-collector:4318`. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Signal-specific override. |
| `OTEL_SERVICE_NAME` | Override the per-process default. |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Standard sampler configuration. |
| `OTEL_SDK_DISABLED` | Set `true` for an explicit process-level opt-out. |

Each process has a distinct default `service.name`, so they are already separable without
configuration. MindBridge captures no authorization headers, request bodies, prompts, memory
text, or media.

## Benchmark harness

| Variable | Default | Purpose |
| --- | --- | --- |
| `MINDBRIDGE_AML_API_KEY` | unset | Enables `POST /aml/add` and `/aml/search`. Leave off in production. |
| `MINDBRIDGE_AML_TENANT_PREFIX` | `bench_aml` | Tenant prefix the AML harness generates under. |

## Development and test

| Variable | Purpose |
| --- | --- |
| `MINDBRIDGE_TRACEBACK` | Set `1` to keep full tracebacks behind CLI failures instead of the one-line form. |
| `MINDBRIDGE_TEST_DATABASE_URL` | Disposable test database. Its name **must** end in `_test`; the fixture refuses to rebuild anything else. |
| `MINDBRIDGE_REQUIRE_INTEGRATION` | Set `1` to turn a missing test database into a failure instead of a skip. |

Without `MINDBRIDGE_TEST_DATABASE_URL` the entire integration suite skips, so a green run may
never have touched the production store. See [contributing](../CONTRIBUTING.md#quality-gates).
