# MindBridge

MindBridge is an Agentic Native Embodied Memory System: Memory-as-a-Service for machines that can see and hear.

## Documentation

- [Technical implementation architecture](docs/technical-architecture.md)

## Development

MindBridge supports Python 3.10 and 3.11. Python 3.10 is kept as the compatibility floor for Jetson deployments.

Install the project and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups --extra edge --extra server
```

Deployment installs only the process it runs:

```bash
uv sync                                      # Core types and Python SDK
uv sync --extra edge                         # Jetson / robot host
uv sync --extra server                       # API, MCP, PostgreSQL jobs
uv sync --extra server --extra cloud-models  # GPU memory Worker
```

Run the required local quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -W error
git diff --check
```

`tests/benchmarks/golden_recall.json` is the deterministic retrieval gate. It exercises dense
evidence recall, exact text recall, temporal exclusion, and unsupported-query abstention through
the production kernel and PostgreSQL/pgvector path; the normal integration test command runs it.

## Local PostgreSQL

The production store uses PostgreSQL 18 with pgvector 0.8+; filtered HNSW recall relies on iterative
scans. Start the pinned development database and apply every migration in order to a fresh database:

```bash
docker compose up -d postgres redis
for migration in migrations/*.sql; do
  docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U mindbridge -d mindbridge < "$migration"
done
```

Migration `0005` creates a non-login `mindbridge_runtime` role, grants the migration user
membership, and enables forced tenant RLS on every table containing `tenant_id`. Each store
transaction sets one tenant locally. When migrations and the API use different database users,
grant `mindbridge_runtime` to the API login; never give that login `SUPERUSER` or `BYPASSRLS`.
Migration `0007` separates the encoder identity from its compatible search-space identity. Existing
vectors remain isolated under their former model space and must be rebuilt before serving the new
aligned Omni/Text space.

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

LoCoMo, M3-Bench, EgoLifeQA, and SuperMemory-VQA are consumed through thin adapters over pinned
official files. Use Git for code releases and the Hugging Face CLI for Hub datasets; MindBridge does
not ship another downloader:

```bash
git clone https://github.com/snap-research/locomo.git .benchmarks/locomo
git -C .benchmarks/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
git clone https://github.com/ByteDance-Seed/m3-agent.git .benchmarks/m3-agent
git -C .benchmarks/m3-agent checkout 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c
uvx --from huggingface-hub hf download lmms-lab/EgoLife \
  EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --repo-type dataset \
  --revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --local-dir .benchmarks/egolife
uvx --from huggingface-hub hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA \
  data/json/all_qa.json \
  --repo-type dataset \
  --revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f \
  --local-dir .benchmarks/supermemory-vqa

uv run python -m mindbridge.benchmarks.dataset_smoke \
  --locomo .benchmarks/locomo/data/locomo10.json \
  --locomo-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --m3-robot .benchmarks/m3-agent/data/annotations/robot.json \
  --m3-web .benchmarks/m3-agent/data/annotations/web.json \
  --m3-revision 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c \
  --egolife .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --egolife-revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --supermemory .benchmarks/supermemory-vqa/data/json/all_qa.json \
  --supermemory-revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f
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
identities. `MINDBRIDGE_API_KEY` identifies the exact benchmark tenant and is never written to the
manifest.

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run python -m mindbridge.benchmarks.locomo_cli \
  --dataset .benchmarks/locomo/data/locomo10.json \
  --output .benchmarks/results/locomo-mindbridge.json \
  --api-base-url http://localhost:8000 \
  --source-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --run-id locomo-001 \
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
  --run-id m3-robot-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --perception-model-revision serving-fingerprint \
  --answer-model-revision serving-fingerprint
```

Use `--video-id` for a smoke subset. The JSONL uses the official `id`, `question`, `answer`, `type`,
`before_clip`, and `response` fields and adds MindBridge retrieval diagnostics. Its sidecar manifest
pins annotation/media hashes and revisions, code, both Omni calls, Jina, Prompt versions, retrieval
settings, and output hash.

EgoLifeQA uses its official `DAYn` plus `HHMMSSFF` clock, whose final two digits are hundredths of
a second in the released annotations. Its prepared manifest contains one subject, a `DAY1 00:00`
timeline origin, and chronological non-overlapping video clips. A clip whose end crosses a question
time is withheld until a later question, so no future frames or audio enter memory:

```json
{
  "subject_id": "A1_JAKE",
  "timeline_origin": "2000-01-01T00:00:00Z",
  "clips": [
    {
      "day": 1,
      "start_timecode": "11100000",
      "media_object": {
        "media_object_id": "egolife_a1_day1_11100000",
        "kind": "video",
        "uri": "s3://mindbridge-media/egolife/day1/11100000.mp4",
        "sha256": "<64 hexadecimal characters>",
        "size_bytes": 14379440,
        "created_at": "2026-08-12T00:00:00Z",
        "duration_ms": 30000
      }
    }
  ]
}
```

```bash
uv run python -m mindbridge.benchmarks.egolife_cli \
  --dataset .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --prepared-media .benchmarks/egolife-prepared-a1.json \
  --output .benchmarks/results/egolife-a1.json \
  --api-base-url http://localhost:8000 \
  --dataset-revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --evaluator-revision 7a97157908757cc898c26835b718653055ecc5f5 \
  --run-id egolife-a1-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --perception-model-revision serving-fingerprint \
  --answer-model-revision serving-fingerprint
```

SuperMemory-VQA runs one participant per invocation. Each prepared video records its official Unix
start and chronological segments. `media_objects` are sent to `observe`; an optional aligned
`transcript` is sent to `remember` because the public release intentionally withholds raw audio.
Either field may be omitted, but every segment must contain at least one. Only segments that end no
later than the earliest official question span are ingested. The output reports Ans-F1, QA-Acc, and
QA-MRR and contains no ground-truth fields:

```json
{
  "subject": 1,
  "videos": [
    {
      "video_id": "Person_1_session_8_03102026_glasses_1264",
      "started_at": "2026-03-10T22:04:28Z",
      "segments": [
        {
          "start_seconds": 0,
          "duration_ms": 30000,
          "media_objects": [],
          "transcript": "User: Okay, it started."
        }
      ]
    }
  ]
}
```

```bash
uv run python -m mindbridge.benchmarks.supermemory_cli \
  --dataset .benchmarks/supermemory-vqa/data/json/all_qa.json \
  --prepared-media .benchmarks/supermemory-prepared-person-1.json \
  --output .benchmarks/results/supermemory-person-1.json \
  --api-base-url http://localhost:8000 \
  --subject 1 \
  --dataset-revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f \
  --source-revision 8123980820ffa23a3452faa6bd8ce5dff0f03164 \
  --run-id supermemory-person-1-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --perception-model-revision serving-fingerprint \
  --answer-model-revision serving-fingerprint
```

Every benchmark `run_id` must be unique for that deployment. It is included in the tenant ID and
sidecar manifest, preventing a rerun from exposing an earlier question to future memories retained
by a previous run.

## Run the MaaS API

The production factory reads secrets from the process environment. The database login must be able
to `SET ROLE mindbridge_runtime`. S3 credentials use Boto3's standard AWS credential chain and are
not copied into MindBridge configuration:

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
export MINDBRIDGE_TEXT_EMBEDDING_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_TEXT_EMBEDDING_ENDPOINT=https://text-embeddings.example.com/v1/embeddings
export MINDBRIDGE_TEXT_EMBEDDING_MODEL_ID=jinaai/jina-embeddings-v5-text-small-retrieval
export MINDBRIDGE_TEXT_EMBEDDING_MODEL_REVISION=6856e76bb72982e58de0620458a4e8b3614da340
export MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY=0.0
export MINDBRIDGE_TENANT_API_KEYS_JSON='{"tenant_01":["replace-with-at-least-32-random-characters"]}'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

uv run --extra server uvicorn mindbridge.api:create_production_app --factory
```

OpenTelemetry is activated only when a standard common or signal-specific OTLP endpoint is set;
without one it remains a no-op. The API, MCP process, Worker, edge sync, consolidation, and lifecycle
commands use distinct default `service.name` values. Override them per process with
`OTEL_SERVICE_NAME` when the deployment needs a namespace. The official FastAPI, HTTPX, Psycopg,
Celery, and Botocore
instrumentations propagate W3C context through REST, model calls, PostgreSQL, S3, and queued jobs.
MindBridge does not capture authorization headers, request bodies, prompts, memory text, or media in
telemetry. Response `trace_id` values use `trace_<32-hex W3C trace ID>` so the suffix maps directly
to the configured backend. Set `OTEL_SDK_DISABLED=true` for an explicit process-level opt-out.

Agents can start the same production kernel over the official MCP stdio transport. Tool input and
structured output schemas are generated from the same Pydantic contracts used by REST and Python:

```bash
uv run --extra server mindbridge-mcp
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

memory = AsyncMindBridge.connect(
    base_url="http://localhost:8000",
    api_key="replace-with-at-least-32-random-characters",
)
try:
    result = await memory.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is my tool?"))
    )
finally:
    await memory.close()
```

For a grounded follow-up, pass selected IDs from the previous result in
`RecallRequest.memory_ids`. Those IDs become the strict candidate scope; MindBridge still applies
tenant, lifecycle, deletion, and evidence checks, but does not search unrelated memory.

Set `mode="enumerate"` for exhaustive count/timeline queries. This path scans the complete
structured-filter scope, verifies candidates against original media in bounded Omni batches, and
returns every occurrence chronologically; scopes above 1,000 candidates fail explicitly instead of
silently truncating.

Every REST API key is bound to an explicit tenant allowlist. The JSON value maps each tenant ID to
one or more keys, so a key can be rotated without downtime and one isolated benchmark deployment can
authorize its generated tenants with the same key. Blank or short keys fail startup. All `/v1`
operations reject a body or query `tenant_id` outside the authenticated allowlist. Only `/healthz`
is public; benchmark runs must add every generated tenant ID to the mapping before starting the API.

`MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` is optional for AWS S3. Media URIs must use the tenant-safe
shape `s3://<bucket>/tenants/<tenant_id>/<object>`.

`POST /v1/observations` returns a durable processing job. Poll
`GET /v1/jobs/{job_id}?tenant_id=<tenant_id>` until it reaches `succeeded` before issuing a recall
that depends on its derived events.

The API sends multimodal recall queries to an OpenAI-compatible Jina v5 Omni pooling endpoint and
explicit memory text to a separate Jina v5 Text Small retrieval endpoint. Both are pinned to one
declared 1024-dimensional compatibility space, while every vector still records the encoder that
actually produced it. The API loads neither model. Self-hosted endpoints can use the upstream
validated vLLM path:

```bash
vllm serve jinaai/jina-embeddings-v5-omni-small-retrieval \
  --revision 12949877f0092093f366c6450340011320152a05 \
  --trust-remote-code

vllm serve jinaai/jina-embeddings-v5-text-small-retrieval \
  --revision 6856e76bb72982e58de0620458a4e8b3614da340 \
  --port 8001
```

## Run the memory Worker

The Worker shares the storage, VLM, and Text Small endpoint variables above. It inspects original AV
once, writes evidence-grounded Event/Entity/Claim graph records atomically, encodes raw evidence with
the pinned local Jina v5 Omni Small model, and batches Event/Claim text through the OpenAI-compatible
Text Small endpoint. Set `MINDBRIDGE_JINA_DEVICE` only when automatic device selection is unsuitable:

```bash
export MINDBRIDGE_VLM_MODEL_REVISION=deployment-2026-08-11
export MINDBRIDGE_JINA_DEVICE=cuda

uv run --extra server --extra cloud-models \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO
```

One prefork child is the safe default because each child owns a full embedding model. Scale with one
Worker process per assigned GPU instead of increasing concurrency inside a process.

Run evidence-verified Episode, semantic Claim, and hierarchical Summary consolidation as one
tenant-scoped scheduled job.
It reuses the database, object-storage, VLM, Text Small, and shared embedding-space variables above;
no task broker or local Jina Omni model is required:

```bash
uv run --extra server mindbridge-consolidate --tenant-id tenant_01
```

Each sweep fixes one `evaluated_at`, scans bounded candidate pages, and lets the Omni/VLM inspect
exact source AV. Episode writes atomically claim child Events. Claim writes atomically create
evidence-unioned semantic Claims or durable `contradicts`/`supersedes` edges; supersession also
versions the represented MemoryRecord. Summary writes form a single-parent Memory tree, inspect
original AV before grouping, and remain recursively expandable to their source memories.
`--page-size`, `--maximum-gap-seconds`, and
`--minimum-similarity` calibrate Episodes; the corresponding `--claim-*` options calibrate Claims.
The corresponding `--summary-*` options calibrate hierarchical Memory grouping.
Schedule the command with the deployment's existing CronJob/systemd/Celery beat control plane;
concurrent runs are idempotent.

Run automatic decay as a tenant-scoped scheduled job. A complete run uses stable bounded pages and
one fixed evaluation instant; concurrent feedback or deletion wins through optimistic guards:

```bash
uv run --extra server mindbridge-lifecycle --tenant-id tenant_01
```

Schedule this command with the deployment's existing CronJob/systemd/Celery beat control plane.
The strength coefficients and hot/cold thresholds are explicit CLI options so hardware cadence and
retention policy can be calibrated without changing model weights or code.

## Run the Jetson/robot edge path

Capture and encode with the installed NVIDIA/GStreamer stack. When `splitmuxsink` or the robot
capture supervisor closes a segment, hand that completed file to the small durable boundary:

```python
from pathlib import Path

from mindbridge.core import IdentityKind, ModelReference, derive_observation_id
from mindbridge.edge import (
    LocalIdentitySample,
    SQLiteIdentityMemory,
    SQLiteObservationOutbox,
    enqueue_captured_video,
)

outbox = SQLiteObservationOutbox(Path("/var/lib/mindbridge/edge.db"))
observation_id = derive_observation_id(
    "tenant_01", "front_camera", "robot-boot-20260811T120000Z", 7
)
identities = SQLiteIdentityMemory(
    Path("/var/lib/mindbridge/edge.db"),
    device_id="front_camera",
    encryption_key=device_identity_key,
)
face = identities.recognize_and_remember(
    LocalIdentitySample(
        tenant_id="tenant_01",
        kind=IdentityKind.FACE,
        source_observation_id=observation_id,
        sample_id="face-track-7-1",
        embedding=insightface_embedding,
        model_reference=ModelReference(
            model_id="insightface/buffalo_l",
            revision="1.0.1",
        ),
    ),
    minimum_similarity=calibrated_face_threshold,
)
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
    identity_observations=(face.to_observation_input(start_ms=120, end_ms=2810),),
)
```

The handoff computes the SHA-256 and size, generates a deterministic tenant-scoped object key and
idempotency key, then commits the request and absolute local path to a mode-`0600`, WAL-enabled
SQLite Outbox. GStreamer/DeepStream remains responsible for camera decoding, encoding, frame rate,
resolution, VAD/motion/scene gates, and hardware calibration.

InsightFace or 3D-Speaker/SpeakerLab remains responsible for producing face or voice embeddings;
MindBridge does not reimplement those models. `device_identity_key` is exactly 32 bytes loaded from
the device TPM or secret manager. The local store normalizes and AES-256-GCM encrypts every bounded
sample, matches only equal model/revision/dimension spaces, and sends only anonymous IDs and time
ranges in `ObserveRequest`. The raw embedding and encryption key never enter the Outbox or cloud.
Forgetting an Observation also removes identity samples learned from that source before the edge
deletion cursor advances.

Drain a bounded batch with the standard Boto3 credential chain and the typed MindBridge SDK:

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run --extra edge python -m mindbridge.edge.sync_cli \
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
