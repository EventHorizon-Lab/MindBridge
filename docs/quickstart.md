# Quickstart

A local MindBridge that accepts writes and answers recalls, in about fifteen minutes. This path
uses explicit `remember()` writes; ingesting real audio and video additionally needs the memory
worker, which is covered in [deployment](deployment.md).

## Before you start

| Requirement | Why |
| --- | --- |
| Python 3.10 or 3.11 | 3.10 is the floor because JetPack, RDK, and RKNN edge images still ship it. |
| [uv](https://docs.astral.sh/uv/) | The lockfile is authoritative; `pip` will not reproduce it. |
| Docker with Compose | Runs the pinned PostgreSQL 18 + pgvector and Redis. |
| An OpenAI-compatible **generator** endpoint | Answers recalls and judges consolidation candidates. |
| An OpenAI-compatible **embedder** endpoint | Encodes queries and text. Must serve Jina v5 Omni or another model at the dimension you configure. |
| S3-compatible object storage | Holds evidence media. MinIO is fine locally; AWS S3 needs no endpoint URL. |

MindBridge loads no model in-process on the API path. Both model slots are remote endpoints you
point it at, so this quickstart needs somewhere to point. If you have neither, serve the
reference embedder yourself:

```bash
vllm serve jinaai/jina-embeddings-v5-omni-small-retrieval \
  --revision 12949877f0092093f366c6450340011320152a05 \
  --trust-remote-code
```

## 1. Install

```bash
git clone https://github.com/EventHorizon-Lab/MindBridge.git
cd MindBridge
uv sync --extra server
```

`--extra server` pulls FastAPI, psycopg, Celery, and the MCP server. It deliberately excludes
`cloud-models`, which carries torch — the API does not need it.

## 2. Start the datastores

```bash
docker compose up -d postgres redis
```

The compose file pins `pgvector/pgvector:0.8.2-pg18-trixie`. Filtered recall relies on pgvector
iterative scans, so 0.8 or newer is a hard requirement rather than a preference.

## 3. Apply the migrations

Numbered SQL, applied in order, to a fresh database:

```bash
for migration in migrations/*.sql; do
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U mindbridge -d mindbridge < "$migration"
done
```

Migration `0005` is the one to understand before you deviate: it creates a non-login
`mindbridge_runtime` role and turns on **forced** row-level security for every table carrying a
`tenant_id`. Each store transaction sets one tenant locally. If your migration user and your API
login differ, grant `mindbridge_runtime` to the API login — and never give that login
`SUPERUSER` or `BYPASSRLS`, which would silently disable tenant isolation.

## 4. Configure the server

Copy the credential template:

```bash
cp .env.example .env
```

Fill in the two API keys and generate a tenant key:

```bash
openssl rand -hex 24
```

Everything that is not a credential is already in the committed `mindbridge.toml` — endpoints,
model IDs, revisions, the embedding space, and the pool size. Edit that file rather than
exporting variables. Its `[generator]` and `[embedder]` endpoints point at localhost; change
them to wherever you serve those models.

Two things bite first-time users:

- **API keys must be at least 32 characters.** A shorter one fails at startup, not at the first
  request. `openssl rand -hex 24` gives 48.
- **`MINDBRIDGE_TENANT_API_KEYS_JSON` is required.** The REST factory refuses to build without
  it. There is no anonymous mode; only `/healthz` is public.

Boto3's own chain resolves S3 credentials and region. MindBridge holds no copy of either, so
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` work exactly as they do
for every other tool on the host. S3-compatible stores that ignore the region still need one
set, so add it to `.env` as well.

Nothing in MindBridge loads `.env` — every deployment target already does it natively. Pass it
to whatever runs the process, and check the result before starting anything:

```bash
uv run --env-file .env mindbridge config check --role api
```

That reports every missing setting in one pass, and names whether each resolved value came from
the environment or from `mindbridge.toml`. It never prints a value.

## 5. Run the API

```bash
uv run --env-file .env --extra server uvicorn mindbridge.server:create_app --factory
```

```bash
curl -s localhost:8000/healthz
```

`/healthz` reports liveness only. It deliberately makes no claim about PostgreSQL, Redis, or the
model endpoints — a health check that lies about its dependencies is worse than one that does
not mention them.

Startup runs one check that is easy to misread as a failure: for every tenant in
`MINDBRIDGE_TENANT_API_KEYS_JSON`, it probes whether that tenant already holds vectors the
configured embedding space cannot reach, and refuses to serve if so. That is what stops a
changed embedder from turning every recall into a silent empty result. On a fresh database it
passes trivially.

## 6. Write and recall

```python
import asyncio
from datetime import datetime, timezone

from mindbridge import MemoryType, MindBridge, RecallQuery, RecallRequest, RememberRequest

API_KEY = "the key printed in step 4"


async def main() -> None:
    async with MindBridge.connect(base_url="http://localhost:8000", api_key=API_KEY) as memory:
        written = await memory.remember(
            RememberRequest(
                tenant_id="tenant_01",
                summary="The red screwdriver went into the blue toolbox on the workbench.",
                memory_type=MemoryType.EPISODIC,
                occurred_at=datetime.now(timezone.utc),
            )
        )
        print(written.status, written.memory_id)

        result = await memory.recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="Where did the red screwdriver end up?"),
            )
        )
        print(result.answer)
        print(result.confidence, [m.memory_id for m in result.memories])


asyncio.run(main())
```

`written.status` is `created` the first time and `duplicate` on an identical resend. Retries are
safe without being silent: omit `idempotency_key` and one is derived from the content, so the
second call returns the same memory rather than a second copy.

The same call over HTTP:

```bash
curl -s localhost:8000/v1/recall \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"tenant_01","query":{"text":"Where did the red screwdriver end up?"}}'
```

## What you have not exercised yet

This quickstart never ran the perception path. To ingest actual audio and video you need the
memory worker, which loads a local media embedder and therefore wants a GPU:

```bash
uv sync --extra server --extra cloud-models
```

Point the local encoder at the GPU in `mindbridge.toml`:

```toml
[media_embedder]
plugin = "jina"
device = "cuda"
```

```bash
uv run --env-file .env --extra server --extra cloud-models \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO
```

`POST /v1/observations` then returns a `processing_job_id`; memory does not exist when that
receipt returns. Poll `GET /v1/jobs/{job_id}` until `succeeded`, or follow it as a stream. See
[deployment](deployment.md#memory-worker).

## Next

- [Concepts](concepts.md) — what an Observation, Event, Claim, and Memory each are, and why
  those are four things rather than one.
- [REST API](api/rest.md) — every endpoint and the full error-code table.
- [Configuration](configuration.md) — the rest of the variables and what breaks without them.
- [Troubleshooting](troubleshooting.md) — if the server refused to start or recall came back
  empty.
