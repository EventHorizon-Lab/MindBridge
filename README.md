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

The production store uses PostgreSQL 18 with pgvector. Start the pinned development database and
apply every migration in order to a fresh database:

```bash
docker compose up -d postgres redis
for migration in migrations/*.sql; do
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U mindbridge -d mindbridge < "$migration"
done
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

## Benchmark dataset smoke

LoCoMo and M3-Bench annotations are consumed from their official repositories by thin schema
adapters. Pin both repositories before producing the checked-in manifest:

```bash
git clone https://github.com/snap-research/locomo.git .benchmarks/locomo
git -C .benchmarks/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
git clone https://github.com/ByteDance-Seed/m3-agent.git .benchmarks/m3-agent
git -C .benchmarks/m3-agent checkout 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c

uv run python -m mindbridge.benchmarks.dataset_smoke \
  --locomo .benchmarks/locomo/data/locomo10.json \
  --locomo-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --m3-robot .benchmarks/m3-agent/data/annotations/robot.json \
  --m3-web .benchmarks/m3-agent/data/annotations/web.json \
  --m3-revision 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c
```

Large M3-Bench media stays outside Git. Acquire it through the official Hugging Face client rather
than a MindBridge downloader:

```bash
uvx --from huggingface-hub hf download ByteDance-Seed/M3-Bench \
  --repo-type dataset \
  --revision 2672152eee36b25ccb38fdbc3b72135347adbb63 \
  --include 'videos/robot/*' \
  --local-dir .benchmarks/m3-bench
```

The resulting annotation identity and counts are recorded in
[benchmarks/manifests/dataset-adapters-smoke.json](benchmarks/manifests/dataset-adapters-smoke.json).

Run LoCoMo against the deployed production API. The command writes the official conversation-level
prediction shape and a sidecar manifest containing source, code, model, Prompt, retrieval, and output
identities. `MINDBRIDGE_API_KEY` is optional for an unauthenticated local deployment and is never
written to the manifest.

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run python -m mindbridge.benchmarks.locomo_cli \
  --dataset .benchmarks/locomo/data/locomo10.json \
  --output .benchmarks/results/locomo-mindbridge.json \
  --api-base-url http://localhost:8000 \
  --source-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --code-revision "$(git rev-parse HEAD)" \
  --answer-model-revision serving-fingerprint
```

Use `--sample-id` for a smoke subset. Existing results are preserved unless `--overwrite` is
explicitly supplied. The prediction field is `mindbridge_prediction`, ready for the official
LoCoMo evaluation functions.

Run M3-Bench through the same deployed API after the official videos have been split into
30-second clips with FFmpeg and uploaded with the standard S3 tooling. MindBridge does not contain
a second media downloader, clipper, or uploader. `--prepared-media` is a JSON array that records
the already-addressable objects:

```json
[
  {
    "video_id": "bedroom_01",
    "timeline_origin": "2000-01-01T00:00:00Z",
    "clips": [
      {
        "clip_index": 0,
        "media_object": {
          "media_object_id": "m3_bedroom_01_0",
          "kind": "video",
          "uri": "s3://mindbridge-media/tenants/benchmark_m3_bedroom_01/0.mp4",
          "sha256": "<64 lowercase hexadecimal characters>",
          "size_bytes": 12345678,
          "created_at": "2026-08-11T00:00:00Z",
          "duration_ms": 30000
        }
      }
    ]
  }
]
```

Clip indices must be contiguous and zero-based. The runner ingests and waits for each durable job
before answering questions whose official `before_clip` equals that index, so a question cannot see
future video. Questions without `before_clip` run after the complete video.

```bash
uv run python -m mindbridge.benchmarks.m3_cli \
  --dataset .benchmarks/m3-agent/data/annotations/robot.json \
  --prepared-media .benchmarks/m3-prepared-robot.json \
  --output .benchmarks/results/m3-robot-mindbridge.jsonl \
  --api-base-url http://localhost:8000 \
  --subset robot \
  --source-revision 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c \
  --media-revision 2672152eee36b25ccb38fdbc3b72135347adbb63 \
  --code-revision "$(git rev-parse HEAD)" \
  --perception-model-revision serving-fingerprint \
  --answer-model-revision serving-fingerprint
```

Use `--video-id` for a smoke subset. The JSONL uses the official `id`, `question`, `answer`, `type`,
`before_clip`, and `response` fields and adds MindBridge retrieval diagnostics. Its sidecar manifest
pins annotation/media hashes and revisions, code, both Omni calls, Jina, Prompt versions, retrieval
settings, and output hash.

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
export MINDBRIDGE_EMBEDDING_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_EMBEDDING_ENDPOINT=https://embeddings.example.com/v1/embeddings
export MINDBRIDGE_EMBEDDING_MODEL_ID=jinaai/jina-embeddings-v5-omni-small-retrieval
export MINDBRIDGE_EMBEDDING_MODEL_REVISION=12949877f0092093f366c6450340011320152a05

uv run uvicorn mindbridge.api:create_production_app --factory
```

Agents can start the same production kernel over the official MCP stdio transport. Tool input and
structured output schemas are generated from the same Pydantic contracts used by REST and Python:

```bash
uv run mindbridge-mcp
```

The stable tools are `memory_observe`, `memory_remember`, `memory_recall`, `memory_get`,
`memory_feedback`, and `memory_forget`. Deploy remote MCP only behind authenticated process or
gateway isolation; the initial command intentionally exposes stdio rather than an unauthenticated
HTTP listener.

Applications and Benchmark runners use the same typed REST contract through the asynchronous
Python SDK:

```python
from mindbridge import AsyncMindBridge
from mindbridge.contracts import RecallQuery, RecallRequest

memory = AsyncMindBridge.connect(base_url="http://localhost:8000")
try:
    result = await memory.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is my tool?"))
    )
finally:
    await memory.close()
```

`MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` is optional for AWS S3. Media URIs must use the tenant-safe
shape `s3://<bucket>/tenants/<tenant_id>/<object>`.

`POST /v1/observations` returns a durable processing job. Poll
`GET /v1/jobs/{job_id}?tenant_id=<tenant_id>` until it reaches `succeeded` before issuing a recall
that depends on its derived events.

The API sends recall queries to an OpenAI-compatible Jina v5 Omni pooling endpoint; it never loads
the 1.56B-parameter model itself. A self-hosted endpoint can be started with the upstream validated
vLLM path:

```bash
vllm serve jinaai/jina-embeddings-v5-omni-small-retrieval \
  --revision 12949877f0092093f366c6450340011320152a05 \
  --trust-remote-code
```

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

## Run the Jetson/robot edge path

Capture and encode with the installed NVIDIA/GStreamer stack. When `splitmuxsink` or the robot
capture supervisor closes a segment, hand that completed file to the small durable boundary:

```python
from pathlib import Path

from mindbridge.edge import SQLiteObservationOutbox, enqueue_captured_video

outbox = SQLiteObservationOutbox(Path("/var/lib/mindbridge/edge.db"))
request = enqueue_captured_video(
    outbox,
    Path("/var/lib/mindbridge/media/segment-000007.mp4"),
    tenant_id="tenant_01",
    device_id="front_camera",
    boot_id="robot-boot-20260811T120000Z",
    sequence=7,
    bucket="mindbridge-media",
    occurred_at=segment_started_at,
    ended_at=segment_ended_at,
    observed_at=capture_completed_at,
    clock_offset_ms=estimated_clock_offset_ms,
)
```

The handoff computes the SHA-256 and size, generates a deterministic tenant-scoped object key and
idempotency key, then commits the request and absolute local path to a mode-`0600`, WAL-enabled
SQLite Outbox. GStreamer/DeepStream remains responsible for camera decoding, encoding, frame rate,
resolution, VAD/motion/scene gates, and hardware calibration.

Drain a bounded batch with the standard Boto3 credential chain and the typed MindBridge SDK:

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run python -m mindbridge.edge.sync_cli \
  --database /var/lib/mindbridge/edge.db \
  --api-base-url https://memory.example.com \
  --bucket mindbridge-media \
  --region us-east-1 \
  --limit 100
```

Use the robot service manager or a systemd timer for retry scheduling and backoff. A failed run keeps
the row, its sanitized error code, and attempt count. Once media has uploaded, later retries send
only the idempotent observation metadata. A cloud receipt advances the per-boot sync watermark and
removes the Outbox row atomically; local media deletion remains an explicit rolling-cache policy.
AWS/S3 credentials and the MindBridge API key are never stored in SQLite.
