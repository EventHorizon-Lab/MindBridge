# Deployment

Running MindBridge in production: what to start, in what order, and what each process needs. For
the variables themselves see [configuration](configuration.md); for the runtime shape see
[architecture](architecture.md#process-topology).

## Topology

Eight process roles. Only the first six are long-running, and two of those are optional.

| Role | Command | Extras |
| --- | --- | --- |
| API | `uvicorn mindbridge.server:create_app --factory` | `server` |
| Jina embedding service | `mindbridge jina serve` | `server` + `cloud-models` |
| Memory worker | `celery -A mindbridge.celery_app:app worker` | `server` + `media` (+ `cloud-models` only for an in-process media encoder, which brings `media` with it) |
| MCP (optional) | `mindbridge mcp` | `server` |
| Consolidation beat (optional) | `celery -A mindbridge.celery_app:app beat` | `server` |
| Consolidation worker (optional) | `celery -A mindbridge.celery_app:app worker -Q mindbridge_consolidation` | `server` |
| Consolidation | `mindbridge consolidate --tenant-id ...` | `server` |
| Lifecycle | `mindbridge lifecycle --tenant-id ...` | `server` |

Install only what a process runs. Each line below is a whole environment for one role, not a
step in a sequence — `uv sync` is exact, so running the next one uninstalls the last one's
packages:

```bash
uv sync                                      # Core types and Python SDK
uv sync --extra edge                         # Edge sync and identity runtime
uv sync --extra server                       # API, MCP, scheduled sweeps
uv sync --extra server --extra cloud-models  # Jina SentenceTransformers service
uv sync --extra benchmarks                   # Benchmark harness
uv sync --all-extras                         # Every role at once
```

The API image should not carry `cloud-models`; it pulls torch and the API loads no model. A host
that runs more than one role names their extras together in one `uv sync`, or extends what is
installed with `uv sync --inexact --extra ...`.

`--all-extras` is a uv flag. Installers that have no equivalent take the `all` extra, which is
the same set by name:

```bash
uv pip install '.[all]'
python -m pip install '.[all]'
```

## Datastores

PostgreSQL 18 with pgvector 0.8 or newer. Filtered recall relies on pgvector iterative scans, so
this is a hard floor, not a preference. Redis is the task broker. Object storage is S3 or any
S3-compatible store.

The checked-in `compose.yaml` pins the development versions:

```bash
docker compose up -d postgres redis
```

## Migrations

Numbered SQL in `migrations/`, applied in order:

```bash
for migration in migrations/*.sql; do
  psql "$MINDBRIDGE_DATABASE_URL" -v ON_ERROR_STOP=1 -f "$migration"
done
```

Apply migrations before starting the processes that read the schema they add. There is no
automatic migration on startup, deliberately: a process that migrates on boot turns a rolling
restart into an uncoordinated schema race.

Six migrations need a decision from you rather than just an apply:

**`0005` — tenant row-level security.** Creates the non-login `mindbridge_runtime` role, grants
the migration user membership, and enables **forced** RLS on every table carrying a `tenant_id`.
Each store transaction sets one tenant locally. When migrations and the runtime use different
logins, grant `mindbridge_runtime` to the runtime login. Never grant it `SUPERUSER` or
`BYPASSRLS` — either silently disables tenant isolation while everything still appears to work.

**`0007` — embedding spaces.** Separates encoder identity from compatible search-space identity.
Existing vectors stay isolated under their former model space and **must be rebuilt** before
serving the new aligned Omni/Text space.

**`0015` — evidence clips.** Narrows `observations.sensor` to `camera` and `microphone`. It
fails rather than rewriting data if a historical row used `gaze`, `imu`, or `robot_state`.
Resolve those rows explicitly before applying it.

**`0018` — drops the HNSW vector index.** This looks like a regression and is not. Under RLS the
planner always has a tenant predicate, so it chose an exact scan and never read the HNSW index —
which still cost roughly 18× on write and held over a gigabyte. Read plans are unchanged.

**`0024` — stores the recall tsvector.** The only migration here that needs a **window**, not just
a decision: `ADD COLUMN ... GENERATED` rewrites `memory_records` under `ACCESS EXCLUSIVE`, so every
read and write of that table blocks for the length of the rewrite — 5.2 s measured on 150 MB, and
proportional to the largest `memory_records` you hold. Size the window against that table rather
than against this number, and apply it while the API is drained rather than mid-serve. It also
drops `memory_records_summary_fts_idx`, which had 0 scans across two complete evaluations because
the substring arm of the recall query gives the planner nothing to use it for.

**`0025` — repairs two things `0021` left behind.** It deletes every `observe` row in
`idempotency_keys`. `0021` removed a field from `ObserveRequest`, which changed the digest of a
request whose bytes did not change, while the idempotency key itself stayed stable — so a device
retrying an observation the server already accepted got `409` forever instead of `DUPLICATE`. The
digest cannot be recomputed here without reproducing one serializer's escaping rules in SQL, so
the claims go instead: losing one costs a reprocess, and the reprocess is idempotent because the
write still dedupes on the derived `observation_id`. **Any `observe` retry in flight across the
upgrade is reprocessed rather than deduplicated by its key.** It also widens the `embeddings`
unique key to include `space_id`, which `0021` dropped along with the revision; without it one
object cannot hold vectors in two spaces, so the re-embedding described under `0007` fails with
`embedding conflict could not be resolved`. Widening a key cannot make existing rows collide, so
there is nothing to resolve before applying it.

## API

Structure goes in `mindbridge.toml`, committed alongside the deployment:

```toml
[database]
max_pool_size = 32

[object_storage]
bucket = "mindbridge-media"
endpoint_url = "https://objects.example.com"

[embedding]
dimension = 1024
space_id = "jina-v5"

[generator]
plugin = "openai"
endpoint = "https://generator.example.com/v1"
model_id = "qwen3.8-max"

[embedder]
plugin = "openai"
endpoint = "https://embeddings.example.com/v1"
model_id = "jinaai/jina-embeddings-v5-omni-small-retrieval"
```

Credentials go in the environment, never in that file — a systemd unit reads them with
`EnvironmentFile=`, a container with `env_file:`, and neither puts them on disk beside the code:

```bash
MINDBRIDGE_DATABASE_URL=postgresql://mindbridge:password@db.internal:5432/mindbridge
MINDBRIDGE_TASK_BROKER_URL=redis://redis.internal:6379/0
MINDBRIDGE_GENERATOR_API_KEY=...
MINDBRIDGE_EMBEDDER_API_KEY=...
MINDBRIDGE_TENANT_API_KEYS_JSON={"tenant_01":["at-least-32-random-characters"]}
```

Confirm both halves resolved before starting anything. This reports every missing setting in one
pass and names the source of each resolved value, without printing any of them:

```bash
mindbridge config check --role api
```

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

uv run --extra server uvicorn mindbridge.server:create_app --factory
```

Stateless — scale horizontally behind any load balancer. Use `/healthz` for liveness. It
deliberately makes no readiness claim about dependencies, so do not gate traffic on it expecting
one.

Startup does real validation and will refuse to serve on: a missing required variable, an API key
under 32 characters, an unrecognized plugin config key, or a tenant holding vectors the
configured embedding space cannot reach. All four are better as a failed deploy than as a
degraded service.

### Embedding endpoint

The API sends multimodal recall queries and explicit memory text to one OpenAI-compatible Jina v5
Omni endpoint. The bundled service loads the model through SentenceTransformers, preserving its
native text, image, video, audio, and mixed-input preprocessing:

```bash
export MINDBRIDGE_EMBEDDER_API_KEY=replace-with-at-least-32-random-characters

uv run --extra server --extra cloud-models mindbridge jina serve \
  --host 0.0.0.0 --port 8001 --device cuda \
  --media-origin https://media.example.com
```

Repeat `--media-origin` for every exact object-storage origin that signs media downloads. The
service downloads those URLs itself with a 64 MiB limit and rejects redirects and every origin
not named here, so the model never receives an unrestricted remote URL.

Point `MINDBRIDGE_EMBEDDER_ENDPOINT` at this service's `/v1` base URL and use the same API key in
the API and worker. `/health` is the readiness probe; `/v1/models` and `/v1/embeddings` follow the
OpenAI-compatible contract used by MindBridge.

**Running this service separately is the largest measured ingest lever there is.** In the
2026-08-24 evaluation, moving the encoder out of the worker and into this service took throughput
from **51 to 489 clips/hour**, and rebuilding the shared broker queue as a round robin so no one
tenant could occupy the pool took it to **626**. The reason the first number is so low is not the
GPU: a prefork child holds its own copy of the encoder, so concurrency multiplies device memory
instead of overlapping anything, and the card measured 1-5% utilised at 93% allocated. With the
encoder served, the worker holds no model and its concurrency is bounded by this endpoint and the
connection pool.

## Memory worker

Shares storage and generator variables with the API. Its text slot reads the same
`MINDBRIDGE_EMBEDDER_*` contract the API queries with, and the recommended media slot reuses that
same endpoint, so the whole worker is one variable away from the API's configuration:

```toml
[media_embedder]
plugin = "openai"
```

```bash
uv run --extra server --extra media \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO --concurrency=8
```

No `cloud-models`, no GPU, no torch: with both embedder slots served, the worker loads no model at
all and its concurrency is bounded by the endpoint and the database rather than by a card.

Note that the command passes no `-Q`. It consumes nine queues that way: `mindbridge` and
`mindbridge.0` through `mindbridge.7`. **Do not narrow that with `-Q`** — a worker started
`-Q mindbridge` consumes only the pre-shard queue, and ingest then stops with nothing logged
anywhere. `-Q` belongs to the consolidation worker, which has a queue of its own.

Observations are published to the shard their tenant id hashes to, because with one queue a single
tenant's backlog is every other tenant's wait: during the 2026-08-24 evaluation one benchmark spent
five hours at zero coverage behind 400 clips belonging to another, and median queue wait across the
run rose from 0.7 s with one producer to 12,018 s with nine. Shards bound that but do not remove
it — kombu polls the queues round robin, so one tenant can no longer starve every other tenant,
only the tenants that hash to its own shard. Their number is a constant in the code rather than a
setting, because a publisher writing to a shard the worker does not consume stops ingest silently.

`mindbridge` itself receives no new work. It stays in the set so that an upgrade drains whatever was
queued before the shards existed.

`media` is still required, and is easy to lose sight of precisely because serving removes
everything else. Clip derivation runs in the worker whatever the embedder slots say, and its PyAV,
Pillow, and SoundFile decoders are declared only in that extra. They are imported lazily, so a
`server`-only install starts cleanly, passes an import probe, and then fails the first observation
that carries media -- as `ModelUnavailableError`, which is in `autoretry_for`, so it retries with
backoff before failing. The in-process command below does not need it spelled out because
`cloud-models` depends on `mindbridge[media]`.

Both embedder slots must resolve to one embedding space. The worker compares the two declared
spaces before processing and fails the job rather than writing media and text vectors that cannot
be compared.

### Optional in-process media encoder

The alternative is the bundled `jina` plugin, which loads Jina v5 Omni into the worker process:

```bash
uv sync --extra server --extra cloud-models
export MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=jina
export MINDBRIDGE_MEDIA_EMBEDDER_DEVICE=cuda

uv run --extra server --extra cloud-models \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO --concurrency=1
```

**The worker refuses to start when a pool of more than one child would hold more resident encoder
weight than the deployment allows, while either embedder slot names `jina`.** A prefork child owns
its plugins, so the model is loaded once per child and six children need about 22 GiB of VRAM. The
pool size is whatever Celery settles on, from `--concurrency` or from `--autoscale`; a pool that
shares one process, `--pool=threads` or `solo`, holds one copy however wide it runs and is not
refused. `--max-memory-per-child` bounds resident host memory and structurally cannot bound VRAM;
during a nine-benchmark evaluation that combination reached 30.2 of the card's 32.6 GB with the GPU
at 1-5% utilisation, then produced 479 CUDA out-of-memory errors and a kernel `global_oom` on the
host. Scale in-process encoding with one worker process per assigned GPU at concurrency 1, or serve
the encoder and stop budgeting VRAM per child. A card that can genuinely hold more copies says so
in
[`MINDBRIDGE_WORKER_VRAM_BUDGET_GIB`](configuration.md#optional-local-media-embedder-worker-only),
which is the
supported way to raise the limit — the estimate it bounds counts resident weights only, so leave
room for activation memory.

### How long one observation may take

This is not a separate setting. Perception is one generator call per observation, so the worker
sizes its Celery soft limit, hard limit, and broker re-delivery window from the generator's own
`request_timeout_seconds`, plus a fixed 300-second allowance for the encoding and graph write that
follow. A slow deployment raises one value and the budget follows:

```toml
[generator]
endpoint = "https://generator.example.com/v1"
model_id = "qwen3.8-max"
request_timeout_seconds = 1800
```

Omitting the key is fine: the bundled generator's own default of 1800 seconds applies, and the
budget is derived from that same number, so the two cannot disagree.

Sizing them independently is what makes a slow generator look like a *broken* write path rather
than a slow one. Perception can spend thousands of output tokens on a busy 30-second clip; if the
task limit expires first, the overrun is retried as though it were transient and the same call is
paid for again until the retries run out. Nothing is written either way. Set the generator's
deadline to what its slowest clip actually needs and let the budget follow.

The worker inspects original AV once, writes evidence-grounded Event/Entity/Claim records
atomically, and cuts one derived clip per grounded span before encoding it locally. Event and
Claim text is batched through the remote embedder.

Sampling is the cost lever: see
[media sampling](configuration.md#media-sampling-worker). Frame rate sets the entire write cost
of a video deployment.

## Separating read and write traffic

Perception and answering compete for one generator endpoint by default, and they starve each
other badly. Measured in the 2026-08-24 evaluation: per-observation work time went from **74.5 s**
at one producer to **626.8 s** at nine, median queue wait from **0.7 s to 12 018 s**, and a
populated recall from **10.8 s idle to 206 s** under the same nine-way contention. The endpoint
was not down — a trivial completion answered in 0.28 s throughout — it was saturated with
multimodal perception.

**No configuration change ships for this, because none is needed.** The API, the MCP process, and
the worker each read `MINDBRIDGE_GENERATOR_*` from their own environment and construct their own
client, so pointing them at different endpoints is a matter of exporting a different value in
each unit file:

```bash
# API and MCP units: the interactive endpoint
export MINDBRIDGE_GENERATOR_ENDPOINT=https://generator-read.example.com/v1

# Worker units: the batch endpoint
export MINDBRIDGE_GENERATOR_ENDPOINT=https://generator-write.example.com/v1
```

The same applies to `MINDBRIDGE_EMBEDDER_*`, and to `MINDBRIDGE_GENERATOR_CONFIG_JSON` when a
deployment wants different deadlines or retry budgets on the two paths. Both endpoints must serve
the same generator for answers to be comparable across paths; the embedder additionally has to
resolve to one embedding space, which the startup probe enforces.

Two things this does *not* fix, so plan around them:

- **One saturated endpoint behind two URLs is still one saturated endpoint.** Split the traffic
  only when the two URLs reach different capacity.
- **Ingest has no fairness between tenants.** Every observation goes to one `mindbridge` queue,
  consumed in order. In the evaluation one benchmark sat five hours at zero coverage purely on
  queue position — 400 clips behind a frontier advancing at tens of messages per hour — and
  started processing six minutes after the queue was rebuilt as a round robin across benchmarks.
  A tenant that bulk-imports will delay every other tenant for as long as its backlog lasts, and
  nothing in the deployment prevents that. Rate-limit bulk ingest at the caller.

## Scheduled jobs

Neither sweep needs a broker or a local model. Both are idempotent under concurrent runs, so use
whatever CronJob, systemd timer, or Celery beat the deployment already has.

```bash
uv run --extra server mindbridge consolidate --tenant-id tenant_01
uv run --extra server mindbridge lifecycle --tenant-id tenant_01
```

**Consolidation is the one you cannot skip.** Nothing above a single clip exists until it runs:
no Episode, no Claim, no Summary, so recall can only ever return the individual moments a query
matched. Two variables put it on the Celery beat schedule the worker app already carries:

```bash
export MINDBRIDGE_CONSOLIDATION_TENANT_IDS=tenant_01,tenant_02
export MINDBRIDGE_CONSOLIDATION_INTERVAL_SECONDS=3600   # optional, this is the default

celery -A mindbridge.celery_app:app beat --loglevel=INFO
celery -A mindbridge.celery_app:app worker -Q mindbridge_consolidation --concurrency=1
```

One tick sweeps one tenant by rotation, and the sweep is routed off the observation queue, so it
takes at most one concurrent generator call and never a worker slot from ingest. Both processes
need the same tenant list. The contract, the two off-switches, and what a running sweep costs are
in [operations](operations.md#built-in-consolidation-schedule).

Each prints one JSON object on stdout — capture it, since it is the record of what the sweep
actually did.

A reasonable starting cadence: consolidation hourly to daily depending on ingest volume,
lifecycle daily. Entity resolution is the expensive part of consolidation — it opens media and
spends a generator call per candidate pair — so a deployment that wants cheap frequent runs plus
occasional expensive ones can split them:

```bash
# frequent
mindbridge consolidate --tenant-id tenant_01 --skip-entity-resolution
# occasional
mindbridge consolidate --tenant-id tenant_01
```

Orphan clip reclamation is separate because it deletes objects:

```bash
mindbridge lifecycle --tenant-id tenant_01 --reclaim-orphan-clips --dry-run
mindbridge lifecycle --tenant-id tenant_01 --reclaim-orphan-clips
```

`--dry-run` writes nothing at all. Run it first.

See [operations](operations.md) for what these sweeps do and how to calibrate them.

## MCP

```bash
uv run --extra server mindbridge mcp
```

Deploy remote MCP only behind authenticated process or gateway isolation. The shipped command
exposes stdio deliberately rather than an unauthenticated HTTP listener.

The stdio process has no configured tenant list, so it cannot run the embedding-space startup
probe. It will not catch a mid-re-embedding deployment for you.

## Security posture

| Control | Where |
| --- | --- |
| Tenant isolation | Forced PostgreSQL RLS (migration `0005`) plus an API-key allowlist. |
| Credentials | Environment only. Never a CLI flag, never in SQLite on the edge. |
| API keys | Minimum 32 characters; digest-only storage; multiple keys per tenant for rotation. |
| Media access | Short-lived signed URLs. No public bucket. |
| Telemetry | No auth headers, bodies, prompts, memory text, or media captured. |
| Edge | Raw face and voice embeddings and the device key never leave the device. |

Full detail and the reporting process are in [SECURITY.md](../SECURITY.md).

## Capacity notes

These are shapes to plan against, not benchmarks:

- **Recall is connection-hungry.** One recall peaks near ten PostgreSQL connections. Size
  `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` against your server's `max_connections` divided across
  every MindBridge process, not per process.
- **Ingest cost is set by frame rate.** One clip cut, one encoder call, and one stored object per
  sampled window.
- **The worker is network-bound once its encoder is served.** With `MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=openai`
  a worker holds no model, so concurrency is bounded by the endpoint and the connection pool. With
  the in-process encoder it is bounded by one card, one child at a time, and it spends most of its
  wall clock waiting on the generator anyway — measured GPU utilisation was 1-5% against 93%
  allocation.
- **`search` mode is much cheaper than `answer`.** It skips the generator entirely.

## Backup and deletion

Two obligations that are easy to get wrong together.

Back up PostgreSQL and object storage consistently — evidence pointers in the database are
meaningless without the objects, and orphaned objects are cost without value.

But a backup that outlives a `forget()` reintroduces deleted content on restore. Tombstones are
content-free by design and survive the content precisely so that a restore can be reconciled
against them. Decide the backup retention window and the deletion propagation window together,
and rehearse the restore-then-reconcile path before you need it.

## What is not automated

Honest gaps, so you plan for them rather than discover them:

- **No migration-on-startup.** Apply them yourself, before the rolling restart.
- **Scheduling is built in for consolidation only.** Lifecycle and orphan reclamation are
  commands; their schedule is yours.
- **No per-tenant quotas and no ingest fairness.** Nothing limits how much one tenant ingests,
  and one shared queue serves every tenant in arrival order.
- **No automatic re-embedding.** Changing embedding space or dimension is a manual rebuild; the
  startup probe only stops you from serving a half-migrated deployment.
