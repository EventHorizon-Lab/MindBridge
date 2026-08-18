# MindBridge

MindBridge is an Agentic Native Embodied Memory System: Memory-as-a-Service for machines that can see and hear.

## Documentation

- [Technical implementation architecture](docs/technical-architecture.md)
- [Model plugin architecture and author contract](docs/plugin-architecture.md)
- [RTX 5090 benchmark and lifecycle validation](docs/benchmark-report-5090.md)
- [RTX 5090 reproducibility manifest](benchmarks/manifests/benchmark-5090-clean-007.json)
- [Edge identity model selection and validation](docs/edge-identity-sota.md)
- [SOTA baselines for the supported benchmarks](docs/benchmarks-sota.md)

## Development

MindBridge supports Python 3.10 and 3.11. Python 3.10 is kept as the compatibility floor because
several edge platform images (JetPack, D-Robotics RDK, Rockchip RKNN) still ship it.

Install the project and development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups --extra edge --extra server
```

Deployment installs only the process it runs:

```bash
uv sync                                      # Core types and Python SDK
uv sync --extra edge                         # Any edge host: Jetson, RDK, RK, OpenVINO x86, dGPU
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

Validate Markdown structure and links with the same tool versions used by CI:

```bash
docker run --rm -v "$PWD:/workdir:ro" davidanson/markdownlint-cli2:v0.23.0 \
  "**/*.md" "!.git/**" "!.venv/**" "!.pytest_cache/**" "!.benchmarks/**"
docker run --rm -v "$PWD:/input:ro" -w /input lycheeverse/lychee:0.23.0 \
  --no-progress --root-dir /input './*.md' './docs/**/*.md'
```

`tests/benchmarks/golden_recall.json` is the deterministic retrieval gate. It exercises dense
evidence recall, exact text recall, temporal exclusion, and unsupported-query abstention through
the production kernel and PostgreSQL/pgvector path; the normal integration test command runs it.

Without `MINDBRIDGE_TEST_DATABASE_URL` the whole integration suite — Golden Recall included —
skips, so a green run may never have touched the production store. CI and any change that affects
recall, consolidation, or deletion must therefore require it explicitly:

```bash
MINDBRIDGE_REQUIRE_INTEGRATION=1 uv run pytest -W error
```

With that variable set, a missing test database fails the run instead of skipping it.

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
aligned Omni/Text space. Migration `0015` narrows `observations.sensor` to `camera` and `microphone`,
the only sensors that can carry the image, video, or audio evidence every observation requires. It
fails instead of rewriting data if a historical row used `gaze`, `imu`, or `robot_state`; resolve
those rows explicitly before applying it.

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

## Agent Memory Leaderboard offline harness

[AML](https://agentmemories.ai/) does not distribute a benchmark you run yourself. It calls two
endpoints you host — `Add` writes a history, `Search` returns ranked evidence — and owns the answer
model, the prompts, the judges, and `top_k`. MindBridge serves that contract at `POST /aml/add` and
`POST /aml/search`, registered only when `MINDBRIDGE_AML_API_KEY` is set, so no existing deployment
grows an AML surface by accident. Set `MINDBRIDGE_AML_TENANT_PREFIX` too; the routes derive each
tenant from `user_id` alone and never accept a caller-supplied tenant.

The offline harness drives those same endpoints locally, then scores with AML's own published
`answer` and `evaluate` stages, vendored verbatim under
[benchmarks/aml/](benchmarks/aml/PINNED.md) and pinned by sha256. Those files are never edited and
are excluded from `ruff`, because formatting them would break the byte-for-byte match that makes the
scores comparable at all.

### What these numbers are, and are not

They measure progress between MindBridge versions. They are **not** leaderboard scores:

- AML evaluates refined and held-out splits that are not distributed; the harness uses the public
  upstream splits.
- AML's answer and judge models are undisclosed. The harness points `ANSWER_*` and `JUDGE_*` at
  whatever you configure.
- A real submission must run Add and Search on `gpt-4o-mini` — mandatory on both the industry and
  academic boards. Offline runs are free to use any model, so an offline number tells you the
  architecture's ceiling, not the score you would post.

### Datasets

Six of AML's seven textual benchmarks. ScriptMem is absent on purpose: its public release ships
questions, gold answers, and scoring code, but every `conversation` field contains only a
`format_example` placeholder — the four source scripts are not distributed, so there is nothing for
a memory system to retrieve and an offline number would measure nothing. A real submission is
unaffected; AML runs ScriptMem server-side against its own copy.

```bash
git clone https://github.com/snap-research/locomo.git .benchmarks/locomo
git -C .benchmarks/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
git clone https://github.com/mohammadtavakoli78/BEAM.git .benchmarks/beam
git -C .benchmarks/beam checkout 3e12035532eb85768f1a7cd779832b650c4b2ef9
uvx --from huggingface-hub hf download xiaowu0162/longmemeval --repo-type dataset \
  --local-dir .benchmarks/longmemeval
uvx --from huggingface-hub hf download bowen-upenn/PersonaMem-v1 --repo-type dataset \
  --local-dir .benchmarks/personamem-v1
uvx --from huggingface-hub hf download bowen-upenn/PersonaMem-v2 --repo-type dataset \
  --local-dir .benchmarks/personamem-v2
uvx --from huggingface-hub hf download tencent/CL-bench --repo-type dataset \
  --local-dir .benchmarks/clbench
```

Use `longmemeval_s`. `longmemeval_oracle` ships only each question's gold sessions, so retrieval has
nothing to discriminate against and any score from it is meaningless; `longmemeval_m` is 2.7 GB.

Per-benchmark field mappings, including the places a dataset's key differs from what its pipeline
reads, are recorded in
[the dataset schema reference](docs/superpowers/specs/2026-08-17-aml-dataset-schemas.md).

### Retrieval and scoring

`--dataset` is repeated once per positional argument the benchmark's loader takes, in order:
`locomo`, `longmemeval`, and `clbench` take one path; `beam` takes `chat` then `questions`;
`personamem-v1` takes `questions_csv` then `contexts_jsonl`; `personamem-v2` takes `benchmark_csv`
then `data_root`.

```bash
uv run python -m mindbridge.benchmarks.aml.cli \
  --benchmark locomo \
  --dataset .benchmarks/locomo/data/locomo10.json \
  --output .benchmarks/results/aml-locomo.jsonl \
  --api-base-url "$MINDBRIDGE_API_BASE_URL" \
  --code-revision "$(git rev-parse HEAD)" \
  --deployment-config .benchmarks/deployment.json \
  --run-id smoke-1 \
  --tenant-prefix bench_aml \
  --top-k 100 \
  --concurrency 4
```

`--tenant-prefix` has no default and must match the deployment's `MINDBRIDGE_AML_TENANT_PREFIX`. The
manifest's tenant map is derived client-side from the value you pass, so a mismatch produces a
manifest recording tenants the server never used. Reruns against an existing `--output` resume, and
refuse to start if the sidecar manifest disagrees on benchmark, run id, deployment, or recall limit.

Then score through the vendored pipeline, unmodified:

```bash
export ANSWER_API_BASE="$MINDBRIDGE_GENERATOR_ENDPOINT"
export ANSWER_API_KEY="$MINDBRIDGE_GENERATOR_API_KEY"
export ANSWER_MODEL="qwen3.8-max"
export JUDGE_API_BASE="$MINDBRIDGE_GENERATOR_ENDPOINT"
export JUDGE_API_KEY="$MINDBRIDGE_GENERATOR_API_KEY"
export JUDGE_MODEL="qwen3.8-max"
export JUDGE_VERSION="qwen3.8-max"

uv run python benchmarks/aml/pipelines/locomo-refined/pipeline.py answer \
  --input .benchmarks/results/aml-locomo.jsonl \
  --output .benchmarks/results/aml-locomo-answers.jsonl

uv run python benchmarks/aml/pipelines/locomo-refined/pipeline.py evaluate \
  --input .benchmarks/results/aml-locomo.jsonl \
  --answers .benchmarks/results/aml-locomo-answers.jsonl \
  --output .benchmarks/results/aml-locomo-scores.jsonl
```

### Disable thinking mode on the answer endpoint

The vendored pipelines send exactly `{"model", "messages", "temperature": 0}` and expose no switch
for thinking mode. Pointed at a thinking model such as Qwen3.8-Max, the answer stage can return
reasoning text, the judge scores that text, and the run reads as a memory-system failure rather than
a misconfigured harness. Default `enable_thinking` to false on the endpoint itself rather than
patching the pinned files, and confirm it before trusting any score:

```bash
curl -s "$MINDBRIDGE_GENERATOR_ENDPOINT/chat/completions" -H "Authorization: Bearer $MINDBRIDGE_GENERATOR_API_KEY" -H 'Content-Type: application/json' -d '{"model":"qwen3.8-max","messages":[{"role":"user","content":"Reply with the single word: ok"}],"temperature":0}'
```

`choices[0].message.content` must be exactly `ok`, with no reasoning text and no empty content.

## Benchmark dataset smoke

LoCoMo, M3-Bench, Video-MME, EgoLife (EgoLifeQA), EgoTempo, EgoMemReason, MEMLENS, MM-Lifelong,
and SuperMemory-VQA are consumed through thin adapters over pinned official files. Use Git for code
releases and the Hugging Face CLI for Hub datasets; MindBridge does not ship another downloader:

```bash
git clone https://github.com/snap-research/locomo.git .benchmarks/locomo
git -C .benchmarks/locomo checkout 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376
git clone https://github.com/ByteDance-Seed/m3-agent.git .benchmarks/m3-agent
git -C .benchmarks/m3-agent checkout 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c
uvx --from huggingface-hub hf download lmms-eval/Video-MME \
  videomme/test-00000-of-00001.parquet \
  --repo-type dataset \
  --revision ead1408f75b618502df9a1d8e0950166bf0a2a0b \
  --local-dir .benchmarks/video-mme
uvx --from huggingface-hub hf download lmms-lab/EgoLife \
  EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --repo-type dataset \
  --revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --local-dir .benchmarks/egolife
git clone https://github.com/google-research-datasets/egotempo.git .benchmarks/egotempo
git -C .benchmarks/egotempo checkout 7022ba77b4d89f51cf34e499767995ccd5c90c7a
uvx --from huggingface-hub hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA \
  data/json/all_qa.json \
  --repo-type dataset \
  --revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f \
  --local-dir .benchmarks/supermemory-vqa
uvx --from huggingface-hub hf download Ted412/EgoMemReason \
  annotations_public.jsonl \
  --repo-type dataset \
  --revision 7e581505b9dce0e85193a27ae689ff899d0bc507 \
  --local-dir .benchmarks/egomem-reason
uvx --from huggingface-hub hf download xiyuRenBill/MEMLENS \
  dataset_32k.json agent_subset_195.json \
  --repo-type dataset \
  --revision afa101a1907cc37db40b50d649547964387b96b7 \
  --local-dir .benchmarks/memlens
uvx --from huggingface-hub hf download MM-Lifelong/MM-Lifelong \
  day/test.json week/test.json month/train.json month/val.json \
  --repo-type dataset \
  --revision 248aa82039a574e63a2e524746a7cd8f32330443 \
  --local-dir .benchmarks/mm-lifelong

uv run --extra benchmarks python -m mindbridge.benchmarks.dataset_smoke \
  --locomo .benchmarks/locomo/data/locomo10.json \
  --locomo-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --m3-robot .benchmarks/m3-agent/data/annotations/robot.json \
  --m3-web .benchmarks/m3-agent/data/annotations/web.json \
  --m3-revision 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c \
  --video-mme .benchmarks/video-mme/videomme/test-00000-of-00001.parquet \
  --video-mme-revision ead1408f75b618502df9a1d8e0950166bf0a2a0b \
  --egolife .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --egolife-revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --egotempo .benchmarks/egotempo/egotempo_openQA.json \
  --egotempo-revision 7022ba77b4d89f51cf34e499767995ccd5c90c7a \
  --egomem .benchmarks/egomem-reason/annotations_public.jsonl \
  --egomem-revision 7e581505b9dce0e85193a27ae689ff899d0bc507 \
  --memlens .benchmarks/memlens/dataset_32k.json \
  --memlens-revision afa101a1907cc37db40b50d649547964387b96b7 \
  --mm-day .benchmarks/mm-lifelong/day/test.json \
  --mm-week .benchmarks/mm-lifelong/week/test.json \
  --mm-month-train .benchmarks/mm-lifelong/month/train.json \
  --mm-month-val .benchmarks/mm-lifelong/month/val.json \
  --mm-lifelong-revision 248aa82039a574e63a2e524746a7cd8f32330443 \
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

uvx --from huggingface-hub hf download ByteDance-Seed/M3-Bench \
  intermediate_outputs/robot.tar.gz.00 \
  intermediate_outputs/robot.tar.gz.01 \
  intermediate_outputs/robot.tar.gz.02 \
  memory_graphs/robot.tar.gz \
  --repo-type dataset \
  --revision 2672152eee36b25ccb38fdbc3b72135347adbb63 \
  --local-dir .benchmarks/m3-bench
```

Acquire the released SuperMemory-VQA RGB videos the same way; raw audio is not part of the public
release:

```bash
uvx --from huggingface-hub hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA \
  --repo-type dataset \
  --revision 1d228e0f10049a8a84c458dded2aa25b1e21ce8f \
  --include 'data/video/*' \
  --local-dir .benchmarks/supermemory-vqa
```

The resulting annotation identity and counts are recorded in
[benchmarks/manifests/dataset-adapters-smoke.json](benchmarks/manifests/dataset-adapters-smoke.json).

Run LoCoMo against the deployed production API. The command writes the official conversation-level
prediction shape and a sidecar manifest containing source, code, model, Prompt, retrieval, and output
identities. `MINDBRIDGE_API_KEY` identifies the exact benchmark tenant and is never written to the
manifest.

Every benchmark also requires a secret-free deployment snapshot. It records the actual capability
slots, plugin distribution versions, model revisions, embedding space, and inference options used by
the server and Worker. Before inference begins, the runner freezes the validated snapshot and the
SHA-256 of those same bytes; credential-like keys are rejected.
This is a run artifact, not a named Profile. For example, save the following as
`.benchmarks/deployment.json` and replace the distribution versions and model revisions with those
from the deployed processes:

```json
{
  "server_generator": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "qwen3.8-max",
      "model_revision": "serving-fingerprint",
      "reasoning_effort": "low"
    }
  },
  "server_embedder": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "model_revision": "12949877f0092093f366c6450340011320152a05",
      "space_id": "jina-v5",
      "space_revision": "deployment-space-v1",
      "dimension": 1024
    }
  },
  "worker_generator": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "qwen3.8-max",
      "model_revision": "serving-fingerprint"
    }
  },
  "worker_media_embedder": {
    "plugin": "jina",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "revision": "12949877f0092093f366c6450340011320152a05",
      "device": "cuda"
    }
  },
  "worker_text_embedder": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "model_revision": "12949877f0092093f366c6450340011320152a05"
    }
  }
}
```

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run python -m mindbridge.benchmarks.locomo_cli \
  --dataset .benchmarks/locomo/data/locomo10.json \
  --output .benchmarks/results/locomo-mindbridge.json \
  --api-base-url http://localhost:8000 \
  --source-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --deployment-config .benchmarks/deployment.json \
  --run-id locomo-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --recall-limit 50 \
  --request-timeout-seconds 1800
```

Use `--sample-id` for a smoke subset. The example explicitly selects the experimental Top-50 recall
budget; the benchmark default remains the product-wide Top-20 budget until held-out or full-split
evidence justifies changing it. Existing results are preserved unless `--overwrite` is supplied. The
prediction field is `mindbridge_prediction`, ready for the official LoCoMo evaluation functions;
retrieved dialogue IDs use its matching
`mindbridge_prediction_context` field.

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
        },
        "identity_observations": [
          {
            "identity_id": "person_device_01",
            "kind": "face",
            "start_ms": 1200,
            "end_ms": 4800,
            "confidence": 0.91,
            "model_id": "insightface/buffalo_l",
            "model_revision": "1.0.1",
            "scope": "device",
            "visual_bbox_xyxy": [0.12, 0.08, 0.46, 0.82]
          }
        ],
        "caption": "[Event] The person places the red cup on the left shelf."
      }
    ]
  }
]
```

Clip indices must be contiguous and zero-based. Every clip before the final clip must be exactly 30
seconds, and the final clip must not exceed 30 seconds. A clip may carry both raw media and the
released memory-graph caption: the observation receipt's source `evidence_ids` are attached to its
released `[Event]` episodic view and `[Inference]` semantic view, so recall can find the text and then
reinspect the same video. The runner ingests and waits for each durable job before answering
questions whose official `before_clip` equals that index, so a question cannot see future video.
Questions without `before_clip` run after the complete video.

Optional `identity_observations` come from the existing Edge InsightFace/FunASR path. Only anonymous,
timed IDs, transcripts, and face boxes enter the manifest; biometric vectors remain device-local.
For M3-Bench Robot, a prepared-media producer may instead reuse the pinned released face/voice
intermediates and memory-graph character mapping with the upstream matching thresholds; it must
discard their embeddings and base64 samples after emitting the same cloud-safe contract.
The Omni worker extracts speech, visible text, objects, actions, and relations from the raw AV into
Event/Entity/Claim records that share the source `EvidenceSpan`. Recall therefore retrieves released
text and structured cues together, then reopens each distinct source object before answering.

```bash
uv run python -m mindbridge.benchmarks.m3_cli \
  --dataset .benchmarks/m3-agent/data/annotations/robot.json \
  --prepared-media .benchmarks/m3-prepared-robot.json \
  --output .benchmarks/results/m3-robot-mindbridge.jsonl \
  --api-base-url http://localhost:8000 \
  --subset robot \
  --source-revision 0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c \
  --media-revision 2672152eee36b25ccb38fdbc3b72135347adbb63 \
  --deployment-config .benchmarks/deployment.json \
  --run-id m3-robot-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --request-timeout-seconds 1800
```

Use `--video-id` for a smoke subset. The runner rejects a `--subset` that does not match the
official Robot timing fields or their absence from Web. The JSONL uses the official `id`,
`question`, `answer`, `type`, `before_clip`, and `response` fields and adds MindBridge retrieval
diagnostics. Its sidecar manifest pins annotation/media hashes and revisions, code, both Omni calls,
Jina, Prompt versions, retrieval settings, and output hash.

EgoLifeQA uses its official `DAYn` plus `HHMMSSFF` clock, whose final two digits are frame counts at
the release videos' 20 FPS. Frame counts are carried into later seconds, matching the official
frame-index conversion and its few non-normalized annotations. Its prepared manifest contains one
subject, a `DAY1 00:00` timeline origin, and chronological non-overlapping video clips. A clip whose
end crosses a question time is withheld until a later question, so no future frames or audio enter
memory:

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
      },
      "caption": "Visual Jake passes the phone to Alice.\nAudio Jake asks everyone to mark it."
    }
  ]
}
```

For the official memory-layer protocol, a clip may instead carry the released multimodal
`DenseCaption`/`Transcript` text plus its duration and omit `media_object`:

```json
{
  "day": 1,
  "start_timecode": "11100000",
  "duration_ms": 30000,
  "caption": "Visual: Jake passes the phone to Alice. Audio: Jake asks everyone to mark it."
}
```

When both forms are present, the released visual and audio lines remain separately retrievable but
share the raw observation's `evidence_ids`; answering reopens that video. The run manifest reports
raw-media and caption clip counts separately. Use raw media for the end-to-end perception gate; use
the pinned official captions for a reproducible comparison with EgoRAG-style memory results.

```bash
uv run python -m mindbridge.benchmarks.egolife_cli \
  --dataset .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --prepared-media .benchmarks/egolife-prepared-a1.json \
  --output .benchmarks/results/egolife-a1.json \
  --api-base-url http://localhost:8000 \
  --dataset-revision 143fb319be7aa5ae210c936bf4f0f3a86092afb0 \
  --evaluator-revision 7a97157908757cc898c26835b718653055ecc5f5 \
  --deployment-config .benchmarks/deployment.json \
  --run-id egolife-a1-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --request-timeout-seconds 1800
```

SuperMemory-VQA runs one participant per invocation. Each prepared video records its official Unix
start and chronological segments. `media_objects` are sent to `observe`; an optional aligned
`transcript` is sent to `remember` because the public release intentionally withholds raw audio.
Either field may be omitted, but every segment must contain at least one. If both are present, the
transcript receives the observation's `evidence_ids`, binding searchable speech to the same raw
video/audio rather than creating an evidence-free parallel memory. Following the official protocol,
prepared media must split at every selected question span's end; the runner rejects a missing
boundary before ingestion. It then includes that completed segment and no later media. The output
reports Ans-F1, QA-Acc, and QA-MRR and contains no ground-truth fields:

When an official question boundary extends past the released MP4 container, keep that segment's
published transcript and attach only the available media duration. The resulting `EvidenceSpan`
ends at the real container tail; media may be shorter than its segment but must never exceed it.
At the pinned dataset revision, 82 of 83 referenced sessions have an MP4; the remaining
`Person_1_session_2_01312026_glasses_1275` session is VRS-only and remains transcript-only unless
the caller prepares that VRS with the upstream Project Aria tooling.

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
          "identity_observations": [],
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
  --deployment-config .benchmarks/deployment.json \
  --run-id supermemory-person-1-001 \
  --code-revision "$(git rev-parse HEAD)" \
  --request-timeout-seconds 1800
```

EgoMemReason reuses the same causal EgoLife clip contract shown above. Its prepared-media file is a
JSON array containing one `EgoLifePreparedStream` object for each selected identity. The runner
withholds clips that cross each official `query_time`, supports the released A-J option range, and
writes the exact answer-key-free leaderboard submission shape:

```bash
uv run python -m mindbridge.benchmarks.egomem_cli \
  --dataset .benchmarks/egomem-reason/annotations_public.jsonl \
  --prepared-media .benchmarks/egomem-prepared.json \
  --output .benchmarks/results/egomem-submission.json \
  --api-base-url http://localhost:8000 \
  --dataset-revision 7e581505b9dce0e85193a27ae689ff899d0bc507 \
  --evaluator-revision 2ea98f7002bfad785532b186964cd779b6cd0ed6 \
  --deployment-config .benchmarks/deployment.json \
  --run-id egomem-001 \
  --code-revision "$(git rev-parse HEAD)"
```

MEMLENS follows the official memory-agent protocol: every question gets a fresh tenant, sessions
are consumed in release order, and the question date is supplied only at answer time. Download
`release_images.tar.gz` from the same pinned dataset revision for multimodal runs, upload the
extracted images with the standard storage tooling, and map official relative paths to immutable
image objects:

```json
{
  "images": [
    {
      "source_file": "needle_images/2658faf8f6e6.jpg",
      "media_object": {
        "media_object_id": "memlens_2658faf8f6e6",
        "kind": "image",
        "uri": "s3://mindbridge-media/memlens/2658faf8f6e6.jpg",
        "sha256": "<64 hexadecimal characters>",
        "size_bytes": 123456,
        "created_at": "2026-08-14T00:00:00Z"
      }
    }
  ]
}
```

Use `--agent-subset-index` for the canonical 195-question memory-agent comparison. The output
contains the official judge fields under `data` and can be passed directly to the pinned
`xrenaf/MEMLENS` `llm_judge.py`:

```bash
uv run python -m mindbridge.benchmarks.memlens_cli \
  --dataset .benchmarks/memlens/dataset_32k.json \
  --prepared-images .benchmarks/memlens-prepared-images.json \
  --agent-subset-index .benchmarks/memlens/agent_subset_195.json \
  --output .benchmarks/results/memlens-32k.json \
  --api-base-url http://localhost:8000 \
  --context-window 32k \
  --dataset-revision afa101a1907cc37db40b50d649547964387b96b7 \
  --evaluator-revision 77f3ab9a52fa2d6a17978e2dffe80438a4ecced2 \
  --deployment-config .benchmarks/deployment.json \
  --run-id memlens-32k-001 \
  --code-revision "$(git rev-parse HEAD)"
```

For the official text-only ablation, add `--text-only`, omit `--prepared-images`, and use a
deployment file without Worker plugins; image placeholders remain `[image]` and no generated image
caption is injected.

MM-Lifelong prepared segments use the split-wide clock of the official `total_intervals` field.
`start_seconds` must therefore be a global Day, Week, or Month offset, not a source-video-local
timestamp. A segment can carry raw audio/video, a pinned caption, or both:

```json
{
  "split": "month_val",
  "timeline_origin": "2000-01-01T00:00:00Z",
  "segments": [
    {
      "segment_id": "video_14_00000",
      "start_seconds": 0,
      "duration_ms": 30000,
      "media_objects": [
        {
          "media_object_id": "mm_lifelong_14_00000",
          "kind": "video",
          "uri": "s3://mindbridge-media/mm-lifelong/14/00000.mp4",
          "sha256": "<64 hexadecimal characters>",
          "size_bytes": 12345678,
          "created_at": "2026-08-14T00:00:00Z",
          "duration_ms": 30000
        }
      ],
      "caption": "The streamer enters the station."
    }
  ]
}
```

The JSONL retains `question`, `answer`, and `pred.answer` for the released accuracy judge, adds
`pred.intervals`, and records the official Ref@300 bucket Jaccard score per question and in the
manifest:

```bash
uv run python -m mindbridge.benchmarks.mm_lifelong_cli \
  --dataset .benchmarks/mm-lifelong/month/val.json \
  --prepared-media .benchmarks/mm-lifelong-month-val-prepared.json \
  --output .benchmarks/results/mm-lifelong-month-val.jsonl \
  --api-base-url http://localhost:8000 \
  --split month_val \
  --source-revision 248aa82039a574e63a2e524746a7cd8f32330443 \
  --deployment-config .benchmarks/deployment.json \
  --run-id mm-lifelong-month-val-001 \
  --code-revision "$(git rev-parse HEAD)"
```

### Video-MME and EgoTempo

Both runners use the same prepared-video manifest. `video_id` is the official Video-MME numeric ID
or the exact EgoTempo `clip_id`; EgoTempo segment times start at zero in the trimmed clip. Split long
media and subtitles into ordered, non-overlapping segments. A segment may contain addressable media,
a transcript, or both:

```json
[
  {
    "video_id": "001",
    "timeline_origin": "2026-08-14T00:00:00Z",
    "segments": [
      {
        "segment_id": "001-0000",
        "start_seconds": 0,
        "duration_ms": 30000,
        "media_objects": [
          {
            "media_object_id": "video-mme-001-0000",
            "kind": "video",
            "uri": "s3://mindbridge-media/video-mme/001/0000.mp4",
            "sha256": "<64 hexadecimal characters>",
            "size_bytes": 12345678,
            "created_at": "2026-08-14T00:00:00Z",
            "duration_ms": 30000
          }
        ],
        "transcript": "Optional subtitle text aligned to this segment."
      }
    ]
  }
]
```

Video-MME writes the released nested evaluator shape and an answered-only local accuracy matching
the official parser, broken out per `short`/`medium`/`long` cell. Parquet loading is isolated in the
`benchmarks` extra.

`--transcript-source` is mandatory because Video-MME publishes separate with- and without-subtitle
tables, while a prepared segment's `transcript` may hold either MindBridge's own ASR or the released
subtitle track. Declaring `none` while the prepared media carries transcripts is refused, as is
declaring `asr` or `official_subtitles` when it carries none. `--duration` scopes a run to the cells
being reported; the overall number is saturated, so the long cell is the one worth quoting:

```bash
uv run --extra benchmarks python -m mindbridge.benchmarks.video_mme_cli \
  --dataset .benchmarks/video-mme/videomme/test-00000-of-00001.parquet \
  --prepared-media .benchmarks/video-mme-prepared.json \
  --output .benchmarks/results/video-mme-long.json \
  --api-base-url http://localhost:8000 \
  --dataset-revision ead1408f75b618502df9a1d8e0950166bf0a2a0b \
  --evaluator-revision afd52cfe3dde5b3685e0d4f760c10c756860c758 \
  --deployment-config .benchmarks/deployment.json \
  --duration long \
  --transcript-source none \
  --run-id video-mme-001 \
  --code-revision "$(git rev-parse HEAD)"
```

EgoTempo writes the official `V`, `Q`, `QA`, `A`, `C`, and `M` fields. Run its pinned
`gemini_eval.ipynb` for the released semantic judge rather than substituting a local metric:

```bash
uv run python -m mindbridge.benchmarks.egotempo_cli \
  --dataset .benchmarks/egotempo/egotempo_openQA.json \
  --prepared-media .benchmarks/egotempo-prepared.json \
  --output .benchmarks/results/egotempo.json \
  --api-base-url http://localhost:8000 \
  --source-revision 7022ba77b4d89f51cf34e499767995ccd5c90c7a \
  --evaluator-revision 7022ba77b4d89f51cf34e499767995ccd5c90c7a \
  --deployment-config .benchmarks/deployment.json \
  --run-id egotempo-001 \
  --code-revision "$(git rev-parse HEAD)"
```

The official notebook reads every `.json` file in its configured `input_dir`. Copy only the
prediction artifact, not its `.manifest.json` sidecar, into a run-specific directory and point the
notebook there:

```bash
mkdir -p .benchmarks/egotempo-judge/egotempo-001
cp .benchmarks/results/egotempo.json \
  .benchmarks/egotempo-judge/egotempo-001/
```

The Ego-Life benchmark remains available through the existing `egolife_cli` and official
EgoLifeQA schema; no second alias with divergent behavior is maintained.

Every benchmark `run_id` must be unique for that deployment. It is included in the tenant ID and
sidecar manifest, preventing a rerun from exposing an earlier question to future memories retained
by a previous run.

## Recording an official scorer's result

LoCoMo, MM-Lifelong, EgoTempo, and EgoMemReason are scored outside MindBridge. A run manifest is
written before any of those scorers execute, so it can only pin inputs. Record their output in a
`*.score.json` sidecar instead, which re-hashes the predictions and refuses numbers that belong to
a different run:

```bash
uv run python -m mindbridge.benchmarks.official_score \
  --predictions .benchmarks/results/locomo.json \
  --manifest .benchmarks/results/locomo.json.manifest.json \
  --scorer-output .benchmarks/results/locomo-scorer-stdout.json \
  --scorer-repository snap-research/locomo \
  --scorer-revision 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376 \
  --scorer-command "python evaluation/evaluate.py --data locomo.json" \
  --judge-model gpt-4o-mini \
  --answer-backbone qwen3.8-max \
  --scored-question-count 1540 \
  --metric f1=45.65
```

`--judge-model` and `--answer-backbone` are the fields that make two LoCoMo numbers comparable or
not, and the LoCoMo run manifest additionally records `category_question_counts` so a reader can
tell a four-category result from a five-category one. See
[SOTA baselines for the supported benchmarks](docs/benchmarks-sota.md) for what each number has to
beat.

## Run the MaaS API

The production factory reads secrets from the process environment. The database login must be able
to `SET ROLE mindbridge_runtime`. S3 credentials use Boto3's standard AWS credential chain and are
not copied into MindBridge configuration:

```bash
export MINDBRIDGE_DATABASE_URL=postgresql://mindbridge:password@localhost:5432/mindbridge
export MINDBRIDGE_OBJECT_STORAGE_BUCKET=mindbridge-media
export MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL=https://objects.example.com
export MINDBRIDGE_TASK_BROKER_URL=redis://localhost:6379/0
export MINDBRIDGE_GENERATOR_PLUGIN=openai
export MINDBRIDGE_GENERATOR_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_GENERATOR_ENDPOINT=https://generator.example.com/v1
export MINDBRIDGE_GENERATOR_MODEL_ID=qwen3.8-max
export MINDBRIDGE_GENERATOR_MODEL_REVISION=deployment-2026-08-11
export MINDBRIDGE_EMBEDDER_PLUGIN=openai
export MINDBRIDGE_EMBEDDER_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_EMBEDDER_ENDPOINT=https://embeddings.example.com/v1
export MINDBRIDGE_EMBEDDER_MODEL_ID=jinaai/jina-embeddings-v5-omni-small-retrieval
export MINDBRIDGE_EMBEDDER_MODEL_REVISION=12949877f0092093f366c6450340011320152a05
export MINDBRIDGE_EMBEDDING_SPACE_ID=jina-v5
export MINDBRIDGE_EMBEDDING_SPACE_REVISION=deployment-space-v1
export MINDBRIDGE_EMBEDDING_DIMENSION=1024
export MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY=0.0
export MINDBRIDGE_TENANT_API_KEYS_JSON='{"tenant_01":["replace-with-at-least-32-random-characters"]}'
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1

uv run --extra server uvicorn mindbridge.server:create_app --factory
```

These variables are the normal deployable default, not a Profile. To select any installed adapter,
set its lowercase plugin name and provide the matching `MINDBRIDGE_GENERATOR_CONFIG_JSON` or
`MINDBRIDGE_EMBEDDER_CONFIG_JSON`. An explicit JSON
object is authoritative, so Anthropic, Gemini, local runtimes, and experimental adapters do not need
OpenAI-specific variables. See the [plugin author contract](docs/plugin-architecture.md).

The bundled variables above cover credentials and model identity — what a deployment cannot start
without. A plugin's remaining optional settings are reachable through its `*_CONFIG_JSON` object
instead of one variable each, which is what keeps this list from growing with every knob a plugin
gains. Tuning the bundled OpenAI generator therefore looks like:

```bash
export MINDBRIDGE_GENERATOR_CONFIG_JSON='{
  "api_key": "replace-with-a-runtime-secret",
  "endpoint": "https://generator.example.com/v1",
  "model_id": "qwen3.8-max",
  "model_revision": "deployment-2026-08-11",
  "reasoning_effort": "low"
}'
```

`MINDBRIDGE_EMBEDDING_SPACE_ID` and `MINDBRIDGE_EMBEDDING_SPACE_REVISION` name the search space the
selected Embedder writes into and queries. `MINDBRIDGE_EMBEDDING_DIMENSION` is the one vector width
shared by the pgvector index and every encoder in the deployment; it defaults to 1024 and accepts
only a width Jina v5 was trained to truncate to (32, 64, 128, 256, 512, 768, or 1024). Changing it
requires re-embedding, so set it once per deployment and give every process the same value.
Startup probes every tenant in
`MINDBRIDGE_TENANT_API_KEYS_JSON` and refuses to serve when one holds vectors that space cannot
reach, so pointing the server at a new embedding model without re-embedding fails loudly instead of
returning empty recalls. The probe reports each stranded object type separately, so memory records
the server wrote itself cannot vouch for evidence, events, and claims the Worker wrote in another
space. Vectors in several spaces are accepted while a re-embedding is in progress. The stdio MCP
process has no configured tenant list and therefore cannot run this probe.

OpenTelemetry is activated only when a standard common or signal-specific OTLP endpoint is set;
without one it remains a no-op. The API, MCP process, Worker, edge sync, consolidation, and lifecycle
commands use distinct default `service.name` values. Override them per process with
`OTEL_SERVICE_NAME` when the deployment needs a namespace. The official FastAPI, HTTPX, Psycopg,
Celery, and Botocore
instrumentations propagate W3C context through REST, model calls, PostgreSQL, S3, and queued jobs.
MindBridge does not capture authorization headers, request bodies, prompts, memory text, or media in
telemetry. Response `trace_id` values use `trace_<32-hex W3C trace ID>` so the suffix maps directly
to the configured backend. Set `OTEL_SDK_DISABLED=true` for an explicit process-level opt-out.

The `mindbridge.stage.duration` histogram reports bounded `stage` values for edge capture-to-upload
and acknowledgement, cloud job claim and searchable readiness, and recall first-answer and
completion latency. Generator spans also report media count, JSON retry, time to first token, token
usage, and the bounded recall phase/round; none of these attributes contain user content or IDs.

Agents can start the same production kernel over the official MCP stdio transport. Tool input and
structured output schemas are generated from the same Pydantic contracts used by REST and Python:

```bash
uv run --extra server mindbridge-mcp
```

The stable tools are `memory_observe`, `memory_remember`, `memory_recall`, `memory_get`,
`memory_feedback`, and `memory_forget`. Deploy remote MCP only behind authenticated process or
gateway isolation; the initial command intentionally exposes stdio rather than an unauthenticated
HTTP listener. `memory_get` and the matching Python/REST operation return short-lived signed
EvidenceSpan URLs with the memory, so an Agent does not need a second private storage call.

Applications and Benchmark runners use the same typed REST contract through the asynchronous
Python SDK:

```python
from mindbridge import MindBridge
from mindbridge.contracts import RecallQuery, RecallRequest

async with MindBridge.connect(
    base_url="http://localhost:8000",
    api_key="replace-with-at-least-32-random-characters",
) as memory:
    result = await memory.recall(
        RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is my tool?"))
    )
```

For a grounded follow-up, pass selected IDs from the previous result in
`RecallRequest.memory_ids`. Those IDs become the strict candidate scope; MindBridge still applies
tenant, lifecycle, deletion, and evidence checks, but does not search unrelated memory.

`observe` returns immediately with a `processing_job_id`. Poll it with `get_observation_job`, or
follow it as an event stream — useful because deriving memory from raw media takes far longer than
the request that submitted it:

```python
receipt = await memory.observe(request)
async for event in memory.stream_observation_job("tenant_01", receipt.processing_job_id):
    print(event.job.state, event.job.memory_ids)
```

Every event carries the complete job view rather than a delta, so resuming after a dropped
connection needs only the last ID received — pass it as `last_event_id` and the server skips states
you already have. The stream ends when the attempt settles; because a failed attempt can be retried
later by the stale-job sweep, `failed` means "this attempt is done", not "this job is finished".
Treat `event_id` as opaque. State changes occurring between two server reads are coalesced, so you
always observe the newer state but not necessarily every intermediate `attempt`.

Set `mode="enumerate"` for exhaustive count/timeline queries. This path scans the complete
structured-filter scope, verifies candidates against original media in bounded Generator batches, and
returns every occurrence chronologically; scopes above 1,000 candidates fail explicitly instead of
silently truncating.

Every REST API key is bound to an explicit tenant allowlist. The JSON value maps each tenant ID to
one or more keys, so a key can be rotated without downtime and one isolated benchmark deployment can
authorize its generated tenants with the same key. Blank or short keys fail startup. All `/v1`
operations reject a body or query `tenant_id` outside the authenticated allowlist. Only `/healthz`
is public; benchmark runs must add every generated tenant ID to the mapping before starting the API.

`MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL` is optional for AWS S3. Media URIs must use the tenant-safe
shape `s3://<bucket>/tenants/<tenant_id>/<object>`.

MindBridge owns no S3 region setting. Boto3's own chain resolves it from `AWS_REGION`,
`AWS_DEFAULT_REGION`, `~/.aws/config`, or instance metadata, exactly as it resolves credentials, so
one AWS configuration serves MindBridge and every other tool in the deployment. S3-compatible stores
that do not care about the value still need one set — `AWS_DEFAULT_REGION=us-east-1` is the
conventional choice.

`POST /v1/observations` returns a durable processing job. Poll
`GET /v1/jobs/{job_id}?tenant_id=<tenant_id>` until it reaches `succeeded` before issuing a recall
that depends on its derived events.

The API sends both multimodal recall queries and explicit memory text to one OpenAI-compatible
Jina v5 Omni pooling endpoint, which encodes each side with its own retrieval prompt. Vectors carry
the declared 1024-dimensional compatibility space plus the encoder that actually produced them. The
API loads no model. A self-hosted endpoint can use the upstream validated vLLM path:

```bash
vllm serve jinaai/jina-embeddings-v5-omni-small-retrieval \
  --revision 12949877f0092093f366c6450340011320152a05 \
  --trust-remote-code
```

## Run the memory Worker

The Worker shares storage and Generator variables with the server. It inspects original AV once,
writes evidence-grounded Event/Entity/Claim graph records atomically, and cuts one derived clip per
grounded event span before encoding it with the default local Jina plugin, so each vector covers the
event's own slice of the recording rather than the whole file. Event and Claim text is batched through the
default OpenAI-compatible Embedder. Its text encoder reads the same `MINDBRIDGE_EMBEDDER_*` contract
the API queries with, so only the media slot needs Worker-specific variables:

```bash
export MINDBRIDGE_MEDIA_EMBEDDER_PLUGIN=jina
export MINDBRIDGE_MEDIA_EMBEDDER_DEVICE=cuda
export MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID=jinaai/jina-embeddings-v5-omni-small-retrieval
export MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION=12949877f0092093f366c6450340011320152a05

uv run --extra server --extra cloud-models \
  celery -A mindbridge.celery_app:app worker --loglevel=INFO
```

One prefork child is the safe default because each child owns a full embedding model. Scale with one
Worker process per assigned GPU instead of increasing concurrency inside a process.
`MINDBRIDGE_MEDIA_EMBEDDER_DEVICE` is optional and falls back to automatic device selection.

The Worker's text slot is deliberately not a separate variable family. It must land in the space the
API queries, so it reads `MINDBRIDGE_EMBEDDER_PLUGIN`, `MINDBRIDGE_EMBEDDER_API_KEY`,
`MINDBRIDGE_EMBEDDER_ENDPOINT`, `MINDBRIDGE_EMBEDDER_MODEL_ID`, and
`MINDBRIDGE_EMBEDDER_MODEL_REVISION` exactly as the API does. Each process has its own environment,
so a Worker that genuinely needs a different endpoint sets a different value for the same name rather
than a second name that can silently disagree.

Non-default Worker adapters use `MINDBRIDGE_MEDIA_EMBEDDER_CONFIG_JSON` and
`MINDBRIDGE_EMBEDDER_CONFIG_JSON`; explicit objects replace the bundled fallback variables.
Both slots must resolve to one embedding space. The Worker compares the two declared spaces before
processing and fails the job instead of writing media and text vectors that cannot be compared.

Run evidence-verified Episode, semantic Claim, and hierarchical Summary consolidation as one
tenant-scoped scheduled job.
It reuses the database, object-storage, Generator, Embedder, and shared embedding-space variables above;
no task broker or local Jina Omni model is required:

```bash
uv run --extra server mindbridge-consolidate --tenant-id tenant_01
```

Each sweep fixes one `evaluated_at`, scans bounded candidate pages, and lets the selected Generator inspect
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

Derived evidence clips are uploaded before the transaction that registers them, so an interrupted
attempt can leave an object no record references. Clip keys are content addressed, which keeps a
retry from multiplying that object, and `--reclaim-orphan-clips` deletes whatever is already there.
The flag also reads the object storage variables, so enable it only where those are configured:

```bash
uv run --extra server mindbridge-lifecycle --tenant-id tenant_01 --reclaim-orphan-clips
```

## Run the edge path

The edge path is platform-neutral. It runs on NVIDIA Jetson, D-Robotics RDK, Rockchip RK,
Intel/OpenVINO x86, generic ARM hosts, and on workstations where the "edge" is a 4090/5090/A100.
Only the capture backend and inference runtime change; the Observation timeline, identity gates,
and forget semantics are identical everywhere.

Capture and encode with whatever GStreamer/FFmpeg stack the platform provides. When `splitmuxsink`
or the robot capture supervisor closes a segment, hand that completed file to the small durable
boundary:

```python
from pathlib import Path

from mindbridge.core import IdentityKind, ModelReference, derive_observation_id
from mindbridge.edge import (
    LocalIdentitySample,
    SQLiteIdentityMemory,
    SQLiteObservationOutbox,
    enqueue_captured_media,
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
request = enqueue_captured_media(
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
SQLite Outbox. The platform capture stack (GStreamer, optionally DeepStream) remains responsible for
camera decoding, encoding, frame rate, resolution, VAD/motion/scene gates, and hardware calibration.

`kind` defaults to `MediaKind.VIDEO` and accepts `MediaKind.AUDIO` for a microphone-only capture or
`MediaKind.IMAGE` for a still frame; the sensor follows from it, so audio-only segments are recorded
against `SensorKind.MICROPHONE`. The `audio_path` sidecar applies only to video, and a still image
carries no `duration_ms`. Modality routing downstream is driven entirely by the declared `kind`,
which `MediaObjectInput` cross-checks against the URI extension when the extension is recognized —
declaring one kind and pointing at another container is refused at the boundary rather than
surfacing later as a decode failure in a worker.

The lower-level example accepts an embedding from an existing robot vision stack. Native hot paths
do not need to reopen a completed media file: feed timestamped BGR frames to
`InsightFaceVideoEncoder.encode_frame()` and arbitrary 16 kHz mono PCM16 chunks to the causal
FunASR cache:

```python
import asyncio

from mindbridge.edge.identity_diarization import FunASRStreamingTranscriber

streaming_asr = FunASRStreamingTranscriber.load(device="auto")
faces = await asyncio.to_thread(
    face_encoder.encode_frame,
    bgr_frame,
    timestamp_ms=frame_timestamp_ms,
    duration_ms=frame_duration_ms,
)
partial = await streaming_asr.push_pcm16(pcm16_chunk, is_final=is_last_audio_chunk)
```

The partial transcript is provisional. The platform capture stack still owns the bounded rolling fragment;
when its Event gate closes, `FunASRSpeechPipeline` performs VAD, quality ASR, punctuation,
diarization, and CAM++ centroid extraction in one upstream call.
`recognize_identities_in_av_segment()` combines that result with InsightFace and the optional
provider-neutral audiovisual active-speaker Pipeline, then returns only cloud-safe intervals ready for
`enqueue_captured_media()`. `device=auto` selects an available accelerator, and an explicit
accelerator request fails instead of silently using CPU.

Install InsightFace/ONNX Runtime and FunASR/ModelScope from the device image that matches the target
platform SDK — JetPack/CUDA, D-Robotics OpenExplorer, RKNN Toolkit, OpenVINO, or a plain CUDA/CPU
host. The generic `uv.lock` intentionally pins no vendor accelerator wheel. ONNX is the default
portable artifact; compiled engines (TensorRT, RKNN, OpenVINO IR, BPU `.bin`) are built and cached
per platform and are never reused across device images. NeMo is not part of the current pipeline.
MindBridge orchestrates upstream libraries but does not reimplement their networks, and adds no
cross-platform abstraction layer of its own.

`device_identity_key` is exactly 32 bytes loaded from
the device TPM or secret manager. The local store normalizes and AES-256-GCM encrypts every bounded
sample, matches only equal model/revision/dimension spaces, and sends only anonymous IDs and time
ranges, optional voice transcripts, identity scope, and face boxes in `ObserveRequest`. The raw
embedding and encryption key never enter the Outbox or cloud.
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
  --recent-retention-hours 24 \
  --limit 100
```

Use the robot service manager or a systemd timer for retry scheduling and backoff. A failed run keeps
the row, its sanitized error code, and attempt count. Once media has uploaded, later retries send
only the idempotent observation metadata. A cloud receipt advances the per-boot sync watermark,
records its processing job, and removes the Outbox row atomically. Later runs cache successful job
memories for the configured TTL; evidence still present on the device uses an offline `file://`
reference. Read them without network access through
`SQLiteRecentMemory(Path("/var/lib/mindbridge/edge.db")).list_memories("tenant_01")`. Tombstones
remove matching cache rows before the deletion cursor advances. Local media deletion remains an
explicit rolling-cache policy. AWS/S3 credentials and the MindBridge API key are never stored in
SQLite.
