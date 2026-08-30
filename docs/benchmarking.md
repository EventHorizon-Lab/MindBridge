# Benchmarking

Benchmark correctness starts with storage isolation. MindBridge has no logical benchmark scope, so
every concurrently executing unit must own a different physical data directory.

## Evaluation command

Install the local model runtime and Parquet/download support:

```bash
uv sync --extra local --extra openai --extra benchmarks
```

`mindbridge-bench eval` follows the task-selection shape used by `lmms-eval`:

```bash
mindbridge-bench eval --tasks list

mindbridge-bench eval \
  --model mindbridge \
  --model-args pretrained=gpt-5-mini \
  --judge-model-args model=qwen3.8-27b,base_url=http://judge.example/v1,api_key=EMPTY \
  --tasks locomo-refined,video-mme \
  --batch-size auto \
  --limit 10 \
  --seed 42
```

Open-ended tasks are judged inside this command. EgoMemReason is the server-scored exception: the
same evaluation command automatically writes its official submission file after a complete run.
`--judge-model-args` follows the same comma-separated style as `--model-args` and accepts `model`,
`base_url`, `api_key`, and `timeout_seconds`. Without it, the judge reuses the generation model,
endpoint, key, and timeout. The equivalent environment variables are
`MINDBRIDGE_JUDGE_MODEL`, `MINDBRIDGE_JUDGE_BASE_URL`, `MINDBRIDGE_JUDGE_API_KEY`, and
`MINDBRIDGE_JUDGE_TIMEOUT_SECONDS`. Use `--judge-concurrency` to bound parallel judge requests.
For OpenAI-compatible models that expose a thinking template, bounded deterministic generation can
be selected with `--gen-kwargs max_tokens=512,enable_thinking=false`; both controls are recorded in
the cache namespace and result artifact. `--model-args generation_min_video_seconds=2` explicitly
records a provider's minimum video duration and enables the shared four-still preflight.

Missing annotations and media are downloaded by default. Public releases use immutable Git or
Hugging Face revisions; annotations are also checked against a published SHA-256 when available.
ZIP volumes are extracted with traversal/link checks and resume at the first missing or truncated
entry. `--limit` and `--offset` narrow downloads when the release exposes individual files;
Video-MME v1 is the exception because its 94 GiB release has no public video-to-volume index.
Archives stay beside extracted files so later runs do not download them again.
The `all` group is intentionally unbounded and spans hundreds of gigabytes; use `--limit` and
individual task groups for smoke runs, and budget both archive and extracted sizes.

Automatic preparation uses `ffmpeg` and `ffprobe` to cache deterministic 30-second, 1 fps,
at-most-640×360 clips under `.benchmarks/.prepared/`. M3-Bench web videos use `yt-dlp` (resolved
through `uvx` when available) with conservative request pacing. EgoTempo uses the official
Ego4D CLI and therefore still requires an accepted Ego4D agreement plus AWS credentials;
the command performs every step after that one-time authorization. Set
`MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS` to tune M3 pacing.

Pass `--no-download` for a fully offline run. Existing extracted archives and prepared clips are
reused. A task/media override also remains available for an operator-managed corpus:

```bash
mindbridge-bench eval \
  --tasks video-mme \
  --task-data video-mme=/datasets/video-mme/test.parquet \
  --media-root video-mme=/datasets/video-mme/videos \
  --allow-unverified-data
```

`--allow-unverified-data` is required when an override does not match the catalog digest. That
choice is recorded by the resulting dataset digest, so it is visible rather than silently mixed
with the pinned task.

The catalog covers these benchmark families:

| Task or group | Concrete tasks |
| --- | --- |
| `locomo-refined` | LoCoMo-Refined |
| `m3-bench` | robot and web |
| `video-mme`, `video-mme-v2` | Video-MME and Video-MME-v2 |
| `egolife`, `egomem`, `egotempo` | EgoLifeQA A1/Jake, EgoMemReason, and EgoTempo |
| `memlens` | 32K, 64K, 128K, and 256K on the released 195-question agent subset |
| `mm-lifelong` | day test, week test, month train, and month validation |
| `supermemory-vqa` | subject 1 |
| `atm-bench` | main and hard, each with raw-media and `-sgm` representations |
| `mem-gallery` | all released topics |

`all` expands to every concrete task in fixed catalog order. Multiple names and wildcard patterns
are de-duplicated and run in catalog order. As in `lmms-eval`, `--limit` accepts `-1`, an absolute
count, or a fraction and `--offset` starts from a later document. Video-MME-v2 keeps whole
four-question groups because its official rating is defined at that boundary.

LoCoMo-Refined and MemLens use their released caption/text tracks. Mem-Gallery is multimodal and
requires its dialogue and question images; ATM-Bench exposes raw-media and schema-guided-memory
tasks separately so their scores are never mixed.

## Prepared media and causal evaluation

The default downloader prepares long video streams and writes the effective versioned manifest
into the result directory. Supply `--media-manifest` only to replace that generated view with an
operator-prepared one; relative paths are resolved from the manifest file:

```json
{
  "version": 1,
  "tasks": {
    "egolifeqa": {
      "units": {
        "A1_JAKE": [
          {
            "path": "clips/jake-day1-0000.mp4",
            "source_id": "jake-day1-0000",
            "start_seconds": 0,
            "end_seconds": 30
          }
        ]
      }
    }
  }
}
```

Run it with `--media-manifest prepared-media.json`. A part may contain `text` instead of `path`,
or both. EgoLifeQA, EgoMemReason, SuperMemory-VQA, and timestamped M3-Bench questions reject
untimestamped media. The runner adds only parts whose `end_seconds` is at or before the query
cutoff, preventing future observations from leaking into an answer. M3-Bench uses seconds from the
start of its video, EgoLifeQA and EgoMemReason use seconds from their release's day-one origin, and
SuperMemory-VQA uses Unix UTC seconds; manifest intervals must use the matching time base.

Prepared clips keep video I/O bounded. Video-MME/v2, M3-Bench, EgoTempo, MM-Lifelong, and
SuperMemory-VQA are physically segmented; released EgoLife 30-second clips are reused. ATM's raw
videos are already only seconds long and Mem-Gallery uses images. `--batch-size auto` uses a
conservative media batch and a
larger text batch, capped by `--max-batch-size`; a failed batch is bisected until it fits or the
individual item is reported as failed. `--unit-concurrency` runs physically isolated
units in parallel, while one shared `--request-concurrency` limit bounds answer calls across all
units in the task. One shared Jina/FunASR model pool is reused across those stores, so weights are
not loaded once per case.

The runner stores source memories as episodic, preserves released wall-clock event spans as typed
bounds, and retains source-relative clip intervals as provenance metadata. LoCoMo-Refined,
MemLens, and Mem-Gallery use the public batched API for released event times; ATM-Bench raw media
uses the same official filename capture-time parser as SGM, while email keeps its released
timestamp. Temporal retrieval is therefore measured instead of relying only on timestamps embedded
in display text.
ATM-Bench keeps each released email or SGM record as one parent memory and lets MindBridge's derived
context keys cover long fields. Mem-Gallery preserves released image IDs beside their images.
Benchmark questions keep the original question as the first ordered text atom and retain answer
instructions as later atoms, exercising the same public mixed-content API as applications.

The benchmark's OpenAI adapter reserves at most 20 MiB per base64-encoded media item and 64 MiB per
answer request for question media and top-ranked evidence — roughly 15 MiB per file and 48 MiB in
aggregate on disk — making prepared clips important. A benchmark against larger question media
needs a provider-specific harness adapter with that provider's native upload mechanism.

## Reproducibility and result trust

An evaluation fixes and records the dataset repository, revision, annotation, auxiliary,
manifest, and resolved-memory digests, adapter and scorer versions, MindBridge/Zvec/Python
versions, generation and judge models/endpoints, pinned Jina revision, batch sizes, retrieval
limit, and seed. Generation requests use the run seed and temperature `0`; judge requests preserve
each benchmark's published transport and sampling settings. Model providers can still change
weights behind an unversioned model name, so use an immutable model identifier for publishable
runs.

Before task spans begin, the runner performs one local `retrieval.query` embedding warmup and
records it under `model.embedding_warmup`. This removes lazy weight loading from per-question ask
latency without hiding it in ingestion or generation metrics.

Every run writes atomically to its output directory:

- `samples.jsonl` contains predictions, parsed options, all per-sample native metrics, scorer
  protocol, judge identity/cache state, exact retrieved source intervals, retrieval diagnostics,
  and failures. `--log-samples` additionally retains prompts, references, and raw judge responses.
- `results.json` contains pins, aggregate metrics, a SHA-256 of the samples, cluster-robust
  standard errors, deterministic cluster-bootstrap 95% confidence intervals, and per-task
  performance/token aggregates.
- A complete, valid EgoMemReason run also writes `egomemreason_submission.json` in the official
  500-row upload format.

The terminal summary shows each task's total seconds, average milliseconds per question, total
tokens, and average tokens per question; incomplete provider usage is shown as `—`.

Upload the EgoMemReason file manually to the
[official scorer](https://huggingface.co/spaces/Ted412/EgoMemReason). A run narrowed with `--limit`
or `--offset` records the submission as `partial` and does not emit an upload file. An incomplete
unrestricted run or malformed prediction records `invalid` and exits nonzero. The evaluator never
uploads predictions automatically.

The core `lmms-eval` response-cache path shape is supported. A directory keeps a shared
`cache.db` plus an audit shard at `runs/<run-id>/cache.db`. Each successful deterministic answer is
written through to the shared cache as soon as it completes, so a fresh isolated rerun can recover
after interruption; the run shard is also merged when the run closes:

```bash
mindbridge-bench eval \
  --tasks video-mme \
  --use_cache .benchmarks/response-cache \
  --output_path .benchmarks/results/video-mme
```

Questions are clustered by independent memory unit, not treated as independent observations.
Confidence intervals and regression significance are `null` when a task has fewer than two
independent units; the runner does not manufacture precision from questions sharing one memory.

Each task's `performance` object reports total wall time, mean time per selected question, every
MindBridge operation/stage/model span, generation TTFT, total and mean provider tokens, modality
breakdowns, and per-module usage. It also includes judge time/tokens because those are part of the
actual evaluation cost. Missing provider usage makes `total_tokens` and `average_tokens` null while
retaining an exact `reported_total_tokens` lower bound. Full field semantics are documented in
[performance and token observability](observability.md#benchmark-output).

The scorer uses each release's native protocol:

| Benchmark | Integrated metrics |
| --- | --- |
| LoCoMo-Refined | refined Qwen3-14B judge, token F1, and BLEU-1 |
| M3-Bench | GPT-4o entailment accuracy |
| Video-MME / v2 | exact MCQ accuracy; v2 grouped nonlinear rating |
| EgoLifeQA | exact MCQ accuracy |
| EgoMemReason | official 500-row submission JSON; private server scoring required |
| EgoTempo | Gemini correct/incorrect accuracy and 0–5 judge score |
| MemLens | task-specific binary judge accuracy, including update/refusal rubrics |
| MM-Lifelong | GPT-5 mapped answer accuracy and quantized `ref_at_300` in `[0, 1]` |
| SuperMemory-VQA | QA accuracy and answerability F1; QA-MRR is explicitly unavailable until the answer backend exposes option scores |
| ATM-Bench | native number/list/open-end scoring, Recall@K/GT, Hit@1, and strict/partial Joint@K |
| Mem-Gallery | stemmed F1, BLEU-1/2/4, EM, LLM judge, and retrieval Precision/Recall/Hit@10 |

`results.json` marks every metric independently. An official prompt run with a different judge
model remains useful for iteration but is marked `official_metric: false` and records both the
configured and required official model; it must not be presented as leaderboard-comparable.

Compare identical samples against a prior run and optionally fail CI only on a statistically
supported regression:

```bash
mindbridge-bench eval \
  --tasks locomo-refined \
  --compare .benchmarks/results/baseline \
  --fail-on-regression \
  --regression-threshold 0.01
```

The comparison validates the dataset digest, scorer protocol, and judge identity; joins by stable
sample ID; and reports paired cluster-bootstrap confidence intervals plus win/tie/loss counts. Any
answer error or incomplete ingest also makes the command fail instead of quietly lowering a score.

## Isolation contract

Use one hierarchy under a disposable root:

```text
.benchmarks/
└── benchmark-label/
    └── run-label/
        ├── case-a/    # its own SQLite, Zvec, and lock
        └── case-b/    # its own SQLite, Zvec, and lock
```

The labels belong to the harness and filesystem only. They must not be passed into `Memory.add`,
stored as hidden product fields, or used to filter a shared database.

`BenchmarkRun` creates collision-safe path components and atomically allocates unit directories:

```python
from mindbridge import JinaOmniEmbedder, Memory
from mindbridge.benchmarks.isolation import BenchmarkRun

run = BenchmarkRun(".benchmarks", "retrieval", "trial-001")

for case_id in ("case-a", "case-b"):
    data_dir = run.unit_dir(case_id)
    with Memory(data_dir, embedder=JinaOmniEmbedder()) as memory:
        memory.add(f"Evidence for {case_id}")
```

Actual path components are encoded rather than using raw labels, preventing traversal and naming
collisions. Without `resume=True`, a non-empty run directory fails immediately. Unit creation also
fails on reuse, so two workers cannot silently select the same store.

For parallel execution, allocate every `unit_dir` before or within its worker and keep one
`Memory` owner alive per leaf. Distinct leaves can run at the same time.

## Clean-run rules

A publishable result should record:

- MindBridge and Zvec versions.
- Python version and platform.
- CPU, memory, and storage medium.
- Embedding and generation model identity.
- Dataset revision and case selection.
- Random seed and retrieval limit.
- Whether the index was cold, warm, or optimized.
- The isolated data root layout.

Do not reuse a populated directory for a clean-run score. Resume mode is for deliberate recovery
or continuation and must be reported as such.

## Local-index microbenchmark

The built-in synthetic benchmark measures the SQLite-to-Zvec adapter path without model-network
variance:

```bash
python -m mindbridge.benchmarks.local_index_benchmark \
  --data-dir .benchmarks/local-index/trial-001 \
  --rows 1000 \
  --dimension 128 \
  --queries 20 \
  --k 10 \
  --seed 42 \
  --quantization none
```

`--data-dir` must be empty. The command emits one JSON object with:

- Row, dimension, query, `k`, seed, and quantization parameters.
- Ingest and optimize seconds.
- Recall at `k` against Zvec exact search.
- Query latency p50, p95, and p99 in milliseconds.
- Query throughput in QPS.
- SQLite, Zvec, and total disk bytes.

Synthetic vectors isolate storage and index behavior. They do not measure embedding quality,
grounded-answer quality, or remote model latency.

Run `none`, `fp16`, `int8`, and `rabitq` against separate empty directories to compare recall,
latency, and bytes. RaBitQ requires an x86_64 AVX2 host and dimensions from 64 through 4095. The
exact-search baseline uses Zvec's retained FP32 vectors; a product decision should also pass the
end-to-end behavior benchmarks because synthetic recall does not measure memory-answer quality.

## End-to-end retrieval benchmarks

Behavior benchmarks should exercise only the public SDK:

1. Allocate a new directory for the case.
2. Construct `Memory` with the model configuration under test.
3. Add the case corpus through `add` or `add_many`.
4. Call `optimize()` if the benchmark protocol specifies an optimized index.
5. Query through `search` or `ask`.
6. Score returned public values.
7. Close the instance before archiving artifacts.

Do not import `LocalStore` or `ZvecIndex` in an end-to-end benchmark. The synthetic local-index
microbenchmark is the deliberate exception because those adapters are exactly what it measures.

## Comparing performance

Report quality and speed together. A faster approximate search is not an improvement if recall
falls outside the declared tolerance. Keep ingestion, optimization, and query phases separate;
combining them hides write amplification and one-time build costs.

Latency percentiles require enough queries to be meaningful. Warm-up queries should not be mixed
with measured queries, and concurrent throughput should state concurrency and directory count.

## Artifact safety

Benchmark directories contain source text and embeddings. Treat them as dataset artifacts, apply
appropriate access controls, and delete them according to dataset terms. Never point a benchmark
at an application's live `data_dir`.
