# MindBridge

MindBridge is an Agentic Native Embodied Memory System: Memory-as-a-Service for machines that can see and hear.

## Documentation

- [Technical implementation architecture](docs/technical-architecture.md)

## Development

MindBridge supports Python 3.10 and 3.11. Python 3.10 is kept as the compatibility floor for Jetson deployments.

Install the project and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
```

Run the required local quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
git diff --check
```

## Local PostgreSQL

The production store uses PostgreSQL 18 with pgvector. Start the pinned development database with Docker Compose and apply the initial migration once:

```bash
docker compose up -d postgres redis
docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U mindbridge -d mindbridge < migrations/0001_initial.sql
```

Run the PostgreSQL contract tests against a disposable database whose name ends in `_test`:

```bash
docker compose exec postgres createdb -U mindbridge mindbridge_test
export MINDBRIDGE_TEST_DATABASE_URL=postgresql://mindbridge:mindbridge@localhost:5432/mindbridge_test
uv run pytest -W error tests/integration/test_postgres_store.py
```

The integration fixture refuses to rebuild a database without the `_test` suffix.

## Cloud embedding smoke

Cloud multimodal embedding dependencies are optional so the API and edge packages stay light:

```bash
uv sync --extra cloud-models
uv run --extra cloud-models python -m mindbridge.benchmarks.jina_smoke \
  --revision 12949877f0092093f366c6450340011320152a05
```

The checked-in result is [benchmarks/manifests/jina-omni-small-smoke.json](benchmarks/manifests/jina-omni-small-smoke.json).

## Run the MaaS API

The production factory reads secrets from the process environment. S3 credentials use Boto3's
standard AWS credential chain and are not copied into MindBridge configuration:

```bash
export MINDBRIDGE_DATABASE_URL=postgresql://mindbridge:password@localhost:5432/mindbridge
export MINDBRIDGE_OBJECT_STORAGE_BUCKET=mindbridge-media
export MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL=https://objects.example.com
export MINDBRIDGE_TASK_BROKER_URL=redis://localhost:6379/0
export MINDBRIDGE_VLM_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_VLM_ENDPOINT=https://vlm.example.com/api/v1/chat/completions
export MINDBRIDGE_VLM_MODEL_ID=qwen3.8-max

uv run uvicorn mindbridge.api:create_production_app --factory
```

`MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` is optional for AWS S3. Media URIs must use the tenant-safe
shape `s3://<bucket>/tenants/<tenant_id>/<object>`.

## Run the memory Worker

The Worker shares the storage variables above and additionally pins the fallback VLM revision. Jina
v5 Omni Small is pinned by default; set `MINDBRIDGE_JINA_DEVICE` only when automatic device selection
is unsuitable:

```bash
export MINDBRIDGE_VLM_MODEL_REVISION=deployment-2026-08-11
export MINDBRIDGE_JINA_DEVICE=cuda

uv run --extra cloud-models celery -A mindbridge.celery_app:app worker --loglevel=INFO
```

One prefork child is the safe default because each child owns a full embedding model. Scale with one
Worker process per assigned GPU instead of increasing concurrency inside a process.
