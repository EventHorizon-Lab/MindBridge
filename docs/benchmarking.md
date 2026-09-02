# Benchmarking

MindBridge provides one evaluation runner and two focused utilities. Use the evaluation runner for
quality claims; use the utilities only for their narrower artifact or storage purpose.

| Command | Use it for | Do not infer |
| --- | --- | --- |
| `mindbridge-bench eval` | Pinned datasets, official or explicitly identified scorers, confidence intervals, and run comparisons | A score is leaderboard-comparable without checking its dataset, judge, and validity fields |
| `mindbridge-bench locomo-refined` | Raw LoCoMo-Refined predictions for another evaluator | An integrated benchmark score |
| `mindbridge-bench local-index` | SQLite-to-Zvec ingestion, recall, latency, throughput, and disk use | Embedding or answer quality |

Never point a benchmark at an application's live `data_dir`. One physical directory has one live
MindBridge owner, and each independent benchmark unit needs its own directory.

## Install the benchmark environment

From the repository root, install the benchmark, local-model, and OpenAI-compatible model extras:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --extra benchmarks --extra local --extra openai
uv run --frozen mindbridge-bench --help
```

Media tasks also require `ffmpeg` and `ffprobe`. M3-Bench web media requires `yt-dlp`. These are
system tools, not Python dependencies managed by this project.

## Discover tasks before downloading

The catalog is executable documentation. List its groups, concrete task names, pinned revisions,
and local readiness without downloading anything:

```bash
uv run --frozen mindbridge-bench eval --list-tasks
```

`--tasks` accepts a concrete task, a group such as `m3-bench`, a comma-separated selection, or a
glob such as `video-*`. Prefer the listing over copying a task name or revision from this page.

If inputs are already present, validate their schema and digests without creating a run directory,
loading a model, or contacting a provider:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --check-integrity \
  --no-download
```

The JSON response reports `unit_count`, `question_count`, `dataset_sha256`, and
`evaluation_sha256` for each selected task.

## Run a smoke evaluation

Configure an OpenAI-compatible generation endpoint, then start with one task and one example:

```bash
export MINDBRIDGE_GENERATION_API_KEY="..."

uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --model-args generation_model=gpt-5-mini \
  --limit 1 \
  --seed 42 \
  --output-path .benchmarks/results/locomo-smoke
```

The default generation endpoint is OpenAI-compatible. These environment variables configure it:

| Variable | Meaning |
| --- | --- |
| `MINDBRIDGE_GENERATION_API_KEY` | API key; falls back to `OPENAI_API_KEY` |
| `MINDBRIDGE_GENERATION_BASE_URL` | Base URL; falls back to `OPENAI_BASE_URL` |
| `MINDBRIDGE_GENERATION_MODEL` | Generation model; defaults to `gpt-5-mini` |
| `MINDBRIDGE_GENERATION_MODALITIES` | Comma-separated atomic modalities, or `omni` |
| `MINDBRIDGE_TIMEOUT_SECONDS` | Generation timeout; defaults to 300 seconds |

`--model-args` overrides `generation_model`, `base_url`, `timeout_seconds`, and
`generation_min_video_seconds`. `--gen-kwargs` accepts deterministic generation controls; the
runner enforces temperature zero, sampling off, and a seed matching `--seed`.

Open-ended tasks use a judge. Without separate judge settings, the judge reuses the generation
endpoint. Configure an official or deliberately chosen judge independently:

```bash
export MINDBRIDGE_JUDGE_MODEL="Qwen/Qwen3-14B"
export MINDBRIDGE_JUDGE_BASE_URL="https://judge.example/v1"
export MINDBRIDGE_JUDGE_API_KEY="..."

uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --limit 10 \
  --seed 42 \
  --output-path .benchmarks/results/locomo-qwen3-14b
```

The runner also accepts `--judge-model-args model=...,base_url=...,api_key=...,timeout_seconds=...`.
Avoid putting a real key on a shared machine's command line. A non-official judge is allowed, but
the affected metric is recorded with `official_metric: false`.
`MINDBRIDGE_JUDGE_TIMEOUT_SECONDS` configures the judge timeout when the command-line override is
not used.

## Configure the memory implementation

Pass `--config memory.json` to evaluate the same `MindBridgeConfig` composition used by
`Memory.from_config`:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks longmemeval-s \
  --config memory.json \
  --limit 10 \
  --output-path .benchmarks/results/longmemeval-configured
```

The runner replaces the configured `data_dir` with one isolated directory per benchmark unit.
`--device` overrides configured local embedding and speech devices. The
[Python SDK construction contract](api/python-sdk.md#construction) describes the same composition
boundary.

`--limit` accepts `-1`, a fraction between zero and one, or an absolute example count. Use an
integer for a count; a non-integral value above one is truncated to an integer by the current
loader selection. Its unit is adapter-specific: OpenEQA limits episodes and retains every question
in a selected episode, while EgoTempo limits questions. Use `--check-integrity` or the completed
`results.json` instead of assuming that `--limit` equals `question_count`.

## Catalog and primary metrics

This table summarizes the current scorer registry. The task listing remains authoritative for
concrete variants and source revisions.

| Benchmark | Selector | Primary result | Official judge when required |
| --- | --- | --- | --- |
| LoCoMo-Refined | `locomo-refined` | `llm_judge` | `qwen3-14b` |
| M3-Bench | `m3-bench` | `accuracy` | `gpt-4o-2024-11-20` |
| Video-MME | `video-mme` | `accuracy` | -- |
| Video-MME-v2 | `video-mme-v2` | `rating` | -- |
| EgoLifeQA | `egolifeqa` | `accuracy` | -- |
| EgoMemReason | `egomemreason` | `submission` | Official server scoring |
| EgoTempo | `egotempo` | `accuracy` | `gemini-1.5-flash` |
| MemLens | `memlens` | `accuracy` | `qwen3-235b-judge` |
| MM-Lifelong | `mm-lifelong` | `answer_accuracy` | `gpt-5` |
| SuperMemory-VQA | `supermemory-vqa` | `qa_accuracy` | -- |
| ATM-Bench | `atm-bench` | `accuracy` | `gpt-5-mini` |
| LongMemEval | `longmemeval-s` | `accuracy` | `gpt-4o-2024-08-06` |
| CL-Bench | `clbench` | `solving_rate` | `gpt-5.1` |
| BEAM | `beam` | `llm_judge_score` | `gpt-4.1-mini` |
| PersonaMem-v3 | `personamem-v3` | `personamem_score` | `gpt-5.5` |
| OpenEQA | `openeqa` | `llm_match` | `gpt-4-1106-preview` |
| Mem-Gallery | `mem-gallery` | `f1` | `qwen2.5-72b-instruct` |

EgoMemReason's public release has no answer key. A complete valid run writes an upload-ready
`egomemreason_submission.json`; partial runs remain development artifacts. Video-MME-v2's grouped
`rating` is on a 0--100 scale. Other 0--5 diagnostics are named explicitly in `results.json`.

Protocol boundaries that affect interpretation are explicit rather than approximated:

- `longmemeval-s`, `clbench`, `beam`, and `personamem-v3` are text-only and need no media
  preparation.
- CL-Bench has no separate question field. The adapter splits the final user turn at its last
  blank-line paragraph break and marks oversized residual questions with `question_unsliced`.
- BEAM reports its per-rubric `llm_judge_score`. The `event_ordering` composite is absent because
  its semantic-alignment call has a different shape that the runner does not issue.
- PersonaMem-v3 reads the released fields and causally masks future events. Task families requiring
  structured actions, response-threaded clusters, or paired-row deltas carry no official headline;
  `profile.json` is scorer-side ground truth and is never ingested as memory.
- SuperMemory-VQA reports `qa_accuracy`; `qa_mrr` is unavailable because the answer backend does
  not expose answer-option scores.
- OpenEQA reports normalized `llm_match` plus `llm_match_score_1_5`. Its fixed-history adapter is
  not the active-navigation A-EQA protocol.

Review the upstream repository and license printed by `--list-tasks` before download. Dataset terms
remain independent of MindBridge's license; the copied scorer licenses and protocol notes are in
the [scorer notices](../src/mindbridge/benchmarks/_official/NOTICE.md).

## Acquire and prepare data

The runner downloads missing pinned annotations and supported media by default. Start with one task
and a small limit: the `all` group can require hundreds of gigabytes.

Use `--no-download` for an offline run. Override operator-managed inputs explicitly:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks video-mme \
  --task-data video-mme=/datasets/video-mme/test.parquet \
  --media-root video-mme=/datasets/video-mme/videos \
  --allow-unverified-data \
  --limit 10 \
  --output-path .benchmarks/results/video-mme-local
```

`--allow-unverified-data` applies only to a selected task whose annotation path was overridden with
`--task-data`; it bypasses that pinned dataset digest check. Media and all resolved inputs still
contribute to `input_sha256` and `evaluation_sha256` in the result.

Long videos are prepared as deterministic bounded clips and cached under
`.benchmarks/.prepared/`. When preparation runs, the effective manifest is written to
`OUTPUT/media-manifest.json`. A supplied `--media-manifest FILE` may provide prepared clips or text,
but causal tasks require source intervals; only observations ending at or before a question cutoff
are ingested. Relative paths are resolved from the supplied manifest.

Some datasets need operator action:

- EgoTempo requires Ego4D authorization and AWS credentials.
- OpenEQA publishes questions separately from episode histories. Supply HM3D or ScanNet frame
  directories through `--media-root`, or place them under the catalog path shown by
  `--list-tasks`. The runner accepts either a split directory or its `data/frames` parent.
- A partial OpenEQA extraction fails with the count of missing selected episodes. OpenEQA limits
  episodes, encodes frame histories at one frame per second, and adapts EM-EQA only; it does not
  register the active-navigation A-EQA protocol.

## Read and compare results

A completed evaluation directory contains:

- `samples.jsonl`: one prediction, metric set, evidence intervals, retrieval diagnostics, and
  structured failure fields per question;
- `results.json`: run and implementation identities, input digests, primary and secondary metric
  summaries, cluster confidence intervals, performance, token use, abstentions, and
  `samples_sha256`;
- `media-manifest.json`: only when the runner prepared media;
- `egomemreason_submission.json`: only for a complete valid EgoMemReason run.

Check these task fields before reporting a number:

| Field | Reporting rule |
| --- | --- |
| `primary_metric` and `score.mean` | Name and value of the headline result |
| `official_metric` | Must be true for an official-protocol claim |
| `score_valid` | Must be true; answer or ingest failures make it false |
| `question_count` and `score.cluster_count` | Report both sample size and independent memory units |
| `score.confidence_interval_95` | Absent with fewer than two independent clusters |
| `dataset_sha256`, `evaluation_sha256`, `scorer_protocol`, and `judge_model` | Establish whether two runs are comparable |

Prompts, references, and raw judge responses are retained only with `--log-samples`. Treat that
option as sensitive because artifacts can contain licensed source content, retrieved evidence, and
model responses.

Compare a candidate with an equivalent baseline using stable sample IDs:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --compare .benchmarks/results/baseline \
  --fail-on-regression \
  --regression-threshold 0.01 \
  --output-path .benchmarks/results/candidate
```

Comparison rejects incompatible sample schemas, evaluation inputs, scorer protocols, judge
identities, or scored sample IDs. `--fail-on-regression` exits nonzero when the upper endpoint of
the candidate-minus-baseline 95% cluster interval is less than the negative threshold; any answer
or ingest failure also produces a nonzero exit.

Use `--use-cache .benchmarks/response-cache` to reuse deterministic generation and judge responses
across isolated reruns. The cache namespace includes implementation, model, task, and runner
identities; it is an optimization, not a result artifact.

For a publishable run, keep immutable model identifiers and report hardware, task selection,
digests, recall limit, judge, seed, and whether the index was cold, warm, or optimized. Partial or
unverified runs are suitable for development, not leaderboard comparison.

## Produce raw LoCoMo-Refined predictions

Use the dedicated command only when another evaluator needs the official prediction shape rather
than integrated scores:

```bash
uv run --frozen mindbridge-bench locomo-refined \
  --dataset .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --output .benchmarks/results/locomo-raw/predictions.jsonl \
  --data-root .benchmarks/data \
  --run-id locomo-raw-001 \
  --limit 1
```

It writes the requested JSONL and a sibling `.manifest.json` containing dataset and prediction
digests, model identities, counts, platform details, and relative isolated-store paths. Existing
artifacts are protected unless `--overwrite` is passed; reuse of an existing run requires
`--resume`.

## Measure the local index

The storage microbenchmark is the narrow exception allowed to call local adapters directly:

```bash
uv run --frozen mindbridge-bench local-index \
  --data-dir .benchmarks/local-index/trial-001 \
  --rows 1000 \
  --dimension 128 \
  --queries 20 \
  --k 10 \
  --seed 42 \
  --quantization none
```

`--data-dir` must be absent or empty. The command prints one JSON object with `ingest_seconds`,
`optimize_seconds`, exact-search `recall_at_k`, p50/p95/p99 query latency, `query_qps`, and SQLite,
Zvec, and total disk bytes. Use a fresh directory for each quantization mode.

## Preserve isolation and artifacts

The evaluation runner allocates one physical store per independent unit:

```text
.benchmarks/data/
└── benchmark-<encoded-task>/
    └── run-<encoded-run-id>/
        ├── unit-<encoded-unit-a>/  # SQLite, Zvec, and lock
        └── unit-<encoded-unit-b>/  # a different physical store
```

Harness labels belong to the filesystem, not the product API. Do not encode benchmark, account,
request, or user scope as hidden fields or metadata. Distinct directories may run concurrently;
the same directory has one live owner. CUDA evaluations also take a per-device process lock unless
an external scheduler owns admission control.

Custom behavior benchmarks must call the public `mindbridge` SDK. The local-index command is the
only documented direct-adapter exception. Benchmark directories may contain licensed data,
embeddings, prompts, and responses, so restrict their permissions and publish only artifacts the
upstream terms allow.
