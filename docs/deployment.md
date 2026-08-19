# Deployment

Running MindBridge in production: what to start, in what order, and what each process needs. For
the variables themselves see [configuration](configuration.md); for the runtime shape see
[architecture](architecture.md#process-topology).

## Topology

Five process roles. Only the first three are long-running.

| Role | Command | Extras |
| --- | --- | --- |
| API | `uvicorn mindbridge.server:create_app --factory` | `server` |
| Memory worker | `celery -A mindbridge.celery_app:app worker` | `server` + `cloud-models` |
| MCP (optional) | `mindbridge mcp` | `server` |
| Consolidation | `mindbridge consolidate --tenant-id ...` | `server` |
| Lifecycle | `mindbridge lifecycle --tenant-id ...` | `server` |

Install only what a process runs:

```bash
uv sync                                      # Core types and Python SDK
uv sync --extra edge                         # Any edge host
uv sync --extra server                       # API, MCP, scheduled sweeps
uv sync --extra server --extra cloud-models  # GPU memory worker
```

The API image should not carry `cloud-models`; it pulls torch and the API loads no model.

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

Four migrations need a decision from you rather than just an apply:

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

## API

```bash
export MINDBRIDGE_DATABASE_URL=postgresql://mindbridge:password@db.internal:5432/mindbridge
export MINDBRIDGE_DATABASE_MAX_POOL_SIZE=32
export MINDBRIDGE_TASK_BROKER_URL=redis://redis.internal:6379/0
export MINDBRIDGE_OBJECT_STORAGE_BUCKET=mindbridge-media
export MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL=https://objects.example.com

export MINDBRIDGE_GENERATOR_PLUGIN=openai
export MINDBRIDGE_GENERATOR_API_KEY=...
export MINDBRIDGE_GENERATOR_ENDPOINT=https://generator.example.com/v1
export MINDBRIDGE_GENERATOR_MODEL_ID=qwen3.8-max
export MINDBRIDGE_GENERATOR_MODEL_REVISION=deployment-2026-08-11

export MINDBRIDGE_EMBEDDER_PLUGIN=openai
export MINDBRIDGE_EMBEDDER_API_KEY=...
export MINDBRIDGE_EMBEDDER_ENDPOINT=https://embeddings.example.com/v1
export MINDBRIDGE_EMBEDDER_MODEL_ID=jinaai/jina-embeddings-v5-omni-small-retrieval
export MINDBRIDGE_EMBEDDER_MODEL_REVISION=12949877f0092093f366c6450340011320152a05
export MINDBRIDGE_EMBEDDING_SPACE_ID=jina-v5
export MINDBRIDGE_EMBEDDING_SPACE_REVISION=deployment-space-v1
export MINDBRIDGE_EMBEDDING_DIMENSION=1024

export MINDBRIDGE_TENANT_API_KEYS_JSON='{"tenant_01":["at-least-32-random-characters"]}'
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

The API sends both multimodal recall queries and explicit memory text to one OpenAI-compatible
Jina v5 Omni pooling endpoint, which encodes each side with its own retrieval prompt. It loads no
model itself. A self-hosted endpoint can use the upstream validated vLLM path:

```bash
vllm serve jinaai/jina-embeddings-v5-omni-small-retrieval \
  --revision 12949877f0092093f366c6450340011320152a05 \
  --trust-remote-code
```

## Memory worker

Shares storage and generator variables with the API. Its text slot reads the same
`MINDBRIDGE_EMBEDDER_*` contract the API queries with, so only the media slot needs
worker-specific variables:

```bash
export MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=jina
export MINDBRIDGE_MEDIA_EMBEDDER_DEVICE=cuda
export MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID=jinaai/jina-embeddings-v5-omni-small-retrieval
export MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION=12949877f0092093f366c6450340011320152a05

uv run --extra server --extra cloud-models \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO
```

**One prefork child is the safe default**, because each child owns a full embedding model. Scale
with one worker process per assigned GPU rather than raising concurrency inside a process.

Both embedder slots must resolve to one embedding space. The worker compares the two declared
spaces before processing and fails the job rather than writing media and text vectors that cannot
be compared.

The worker inspects original AV once, writes evidence-grounded Event/Entity/Claim records
atomically, and cuts one derived clip per grounded span before encoding it locally. Event and
Claim text is batched through the remote embedder.

Sampling is the cost lever: see
[media sampling](configuration.md#media-sampling-worker). Frame rate sets the entire write cost
of a video deployment.

## Scheduled jobs

Neither sweep needs a broker or a local model. Both are idempotent under concurrent runs, so use
whatever CronJob, systemd timer, or Celery beat the deployment already has.

```bash
uv run --extra server mindbridge consolidate --tenant-id tenant_01
uv run --extra server mindbridge lifecycle --tenant-id tenant_01
```

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
- **The worker is GPU-bound, the API is not.** They scale independently, which is why the media
  slot is the only in-process model.
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
- **No built-in scheduler.** The sweeps are commands; scheduling is yours.
- **No per-tenant quotas.** Nothing limits how much one tenant ingests.
- **No automatic re-embedding.** Changing embedding space or dimension is a manual rebuild; the
  startup probe only stops you from serving a half-migrated deployment.
