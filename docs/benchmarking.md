# Benchmarking

MindBridge ships three benchmark commands:

- `mindbridge-bench eval` runs pinned memory datasets through the public SDK.
- `mindbridge-bench locomo-refined` writes raw predictions for an operator-provided LoCoMo-Refined
  dataset.
- `mindbridge-bench local-index` measures SQLite and Zvec without model or network variance.

Every run requires a new physical data directory for each independent unit. Never benchmark
against an application's live `data_dir`.

## Install and inspect tasks

From a repository checkout, install the model and dataset extras used by the evaluation harness:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --extra benchmarks --extra local --extra openai
```

List the current task groups, pinned revisions, and local-data readiness:

```bash
mindbridge-bench eval --list-tasks
```

The catalog currently covers LoCoMo-Refined, M3-Bench, Video-MME and Video-MME-v2,
EgoLifeQA, EgoMemReason, EgoTempo, MemLens, MM-Lifelong, SuperMemory-VQA, ATM-Bench,
Mem-Gallery, LongMemEval, CL-Bench, BEAM, PersonaMem-v3, and OpenEQA. Use the listing command
instead of copying task names from this page; it is generated from the catalog used by the
runner.

Review each printed repository and revision before downloading data. Upstream terms vary:
LoCoMo-Refined and PersonaMem-v3 are CC BY-NC 4.0, MM-Lifelong is academic-only and restricts
redistribution or modification without prior approval, and CL-Bench carries an evaluation-only
license that forbids training, fine-tuning or distilling on the corpus. MindBridge's license does
not replace those terms; see the
[scorer notices](../src/mindbridge/benchmarks/_official/NOTICE.md).

## Run an evaluation

Set credentials for the OpenAI-compatible generation endpoint, then select a task or group:

```bash
export OPENAI_API_KEY="..."

mindbridge-bench eval \
  --tasks locomo-refined \
  --model-args generation_model=gpt-5-mini \
  --limit 10 \
  --seed 42 \
  --output-path .benchmarks/results/smoke
```

`--limit` accepts an absolute unit count, a fraction between zero and one, or `-1` for all units.
It is not always a question count: one memory unit may contain several questions. Use the emitted
`question_count` when reporting sample size.

Pass `--config memory.json` to use the same composition accepted by `Memory.from_config`; the
runner replaces `data_dir` with isolated per-unit directories. See [configuration](configuration.md)
for the JSON schema. `--device` overrides configured local-model devices, and `--model-args`
overrides generation model, base URL, timeout, or the provider's minimum video duration.

The harness also reads `MINDBRIDGE_GENERATION_API_KEY`, `MINDBRIDGE_GENERATION_BASE_URL`,
`MINDBRIDGE_GENERATION_MODEL`, `MINDBRIDGE_GENERATION_MODALITIES`, and
`MINDBRIDGE_TIMEOUT_SECONDS`. The request timeout defaults to 300 seconds: a request the server
never answers otherwise holds its task for the whole timeout while the remaining workers idle, and
the run reports the stall as elapsed time rather than as a failure. Raise it only for a model whose
legitimate responses exceed it.

Open-ended tasks use a judge. Configure it separately when needed:

```bash
mindbridge-bench eval \
  --tasks locomo-refined \
  --judge-model-args model=qwen3.8-27b,base_url=http://judge.example/v1,api_key=EMPTY
```

Without `--judge-model-args`, the judge reuses the generation endpoint. Run
`mindbridge-bench eval --help` for concurrency, cache, generation, and comparison options.

Use `mindbridge-bench locomo-refined --help` when another evaluator needs raw LoCoMo-Refined
predictions and a manifest instead of the integrated scores and confidence intervals from `eval`.

## Datasets and prepared media

By default, the runner downloads missing pinned inputs and verifies published digests when one is
available. The `all` group can require hundreds of gigabytes; start with one task and `--limit`.
Long videos are prepared as deterministic bounded clips and cached under `.benchmarks/.prepared/`.
Preparation requires `ffmpeg` and `ffprobe`; M3-Bench web media also uses `yt-dlp`. EgoTempo
requires prior Ego4D authorization and AWS credentials, and OpenEQA requires operator-supplied
episode histories.

Use `--no-download` for a fully offline run. Operator-managed inputs require explicit task paths:

```bash
mindbridge-bench eval \
  --tasks video-mme \
  --task-data video-mme=/datasets/video-mme/test.parquet \
  --media-root video-mme=/datasets/video-mme/videos \
  --allow-unverified-data
```

`--allow-unverified-data` is required when an override does not match the catalog digest. The
result records the resolved dataset and memory digests so such a run cannot be silently confused
with the pinned release.

For a prepared causal stream, pass `--media-manifest FILE`. Causal tasks require source intervals;
the runner ingests only observations ending at or before the question cutoff. Relative paths are
resolved from the manifest file. When the runner prepares media itself, it writes
`OUTPUT/media-manifest.json`; use that generated file as the format reference before supplying an
operator-authored replacement.

## OpenEQA episode histories

`openeqa-hm3d` and `openeqa-scannet` read one pinned question file --
`data/open-eqa-v0.json`, 1,636 questions over 152 episodes -- but its episode histories are
published separately and are not downloadable from here. Extract them into the catalog's own
location, `.benchmarks/openeqa/data/frames/<split>/`, and the task runs with no extra flag;
otherwise point at them:

```bash
mindbridge-bench eval \
  --tasks openeqa-hm3d \
  --media-root openeqa-hm3d=/datasets/open-eqa/data/frames/hm3d-v0 \
  --limit 1
```

A partial extraction fails rather than scoring the episodes that happen to be present, and the
message names how many of the selected episodes are absent.

| Task | Episodes | Questions | Episode histories |
| --- | --- | --- | --- |
| `openeqa-hm3d` | 63 | 557 | 12 GB of RGB frames from the tarball the upstream `data/README.md` links, or re-extracted from HM3D with the Habitat simulator |
| `openeqa-scannet` | 89 | 1,079 | ScanNet's own signed terms of use, then `data/scannet/extract-frames.py` for 62 GB and roughly eight hours |

Either layout is accepted: the split directory itself, or the parent `data/frames` the upstream
README documents.

Four choices decide whether a number here is comparable with the leaderboard:

- **Only EM-EQA is adapted.** A-EQA scores an agent that navigates the scene to gather its own
  history, which a memory system answering from a fixed episode cannot express. Its 184-question
  subset is entirely HM3D and is not registered as a task.
- **The frame sequence is encoded at one frame per second.** OpenEQA publishes no video encoding
  for evaluation -- upstream's `data/frames2videos.py` writes at 30 fps for its web viewer, not
  for scoring -- so the adapter chooses one, and 1 fps is the rate at which preparation's own
  `fps=1` resample keeps every extracted frame. Each episode then becomes 30-second segments
  through the same pipeline the video tasks use. `_OPENEQA_FRAME_RATE` in
  `benchmarks/prepare_media.py` is the knob for deliberately thinning a scene's history.
- **`--limit` counts episodes, not questions.** One episode is one physically isolated store fed
  by hundreds of frames, and every question over it answers against the same ingested scene.
- **The headline is `llm_match`, reported 0-1.** It is the official LLM-Match protocol: the
  `mmbench` prompt, or `mmbench-extra` for the 263 questions that publish `extra_answers`, judged
  by `gpt-4-1106-preview` for a mark of 1-5. The raw mark is kept beside it as
  `llm_match_score_1_5`. Upstream prints the same quantity multiplied by 100. Two upstream
  behaviours are reproduced rather than corrected: a prediction is cut after its last period when
  that period is not already its final character, and a mark outside 1-5 is clipped instead of
  rejected.

## Text-only memory benchmarks

Four tasks read no media, so they need neither `ffmpeg` nor a preparation pass:

| Task | Unit | Corpus | Official headline |
| --- | --- | --- | --- |
| `longmemeval-s` | one question | its own 50-session haystack | `accuracy`, the yes/no answer-check judge |
| `clbench` | one task | the reference document behind its question | `solving_rate`, the binary rubric judge |
| `beam-100k` … `beam-10m` | one conversation | the whole transcript | `llm_judge_score`, mean over rubric items |
| `personamem-v3` | one persona | five engagement logs plus a calendar stream | `personamem_score`, 0-1 |

Three of them need a note before a number is quoted:

- **CL-Bench publishes no `question` field.** Each record's final turn mixes a reference document
  -- up to ~150,000 characters -- with the query in one string, and the loader splits it at the
  last blank-line paragraph break. 1,322 of the 1,899 records split cleanly (median question 434
  characters); 130 end up with a question of 2,000 characters or more and carry
  `question_unsliced` in their metadata. Filter on that field before reporting.
- **BEAM's `event_ordering` category loses part of its official metric.** Upstream combines the
  per-rubric judge score with `tau_norm x f1`, whose alignment step is a second, differently
  shaped model call the runner does not issue. The per-rubric `llm_judge_score` is reported and
  the composite is left absent rather than approximated.
- **PersonaMem-v3 is scored on the families the pinned release supports.** Its evaluation
  repository has drifted from the released data -- the repository's slate scorer reads a `slate`
  and `origin_by_idx` the release does not publish -- so the reproduced protocols read only
  released fields: the unified personalization rubric (13 task types), the four task-specific
  judges, and the deterministic ranking family, whose headline is the graded nDCG@5 that
  `task_registry.PRIMARY_METRIC` names. The proactive decision judge, the two repetition-fatigue
  cluster tasks, `new_suggestions_chatbot`, `local_recommendation_geo_shift`,
  `active_mistake_prevention` and `short_vs_long_term_lifecycle` are answered and reported but
  carry no official headline -- the last one ranks a slate like the other three, but upstream
  scores it with a delta across two paired rows that a single row cannot carry. The cluster rows
  are dropped at load because their runner threads each response into the next prompt. The rubric judge's evidence block is also narrower than upstream's `build_source_a`,
  which queries the persona backend for the same-day avoid slice and the privacy flags, so
  hard-rule checks that depend on those under-fire relative to a full-harness run.

PersonaMem-v3 masks history causally: each query is answered against only the events that happened
strictly before its timestamp, which the runner applies through the same cutoff machinery the
causal video tasks use. `profile.json` is the scorer-side ground-truth persona and is neither
downloaded nor read as memory.

`ScriptMem` is deliberately absent. Its public release ships questions, gold answers and a scorer,
but every `conversation` field holds only a `format_example` placeholder -- the four source scripts
are withheld for copyright -- so there is nothing for a memory system to retrieve and an offline
number would measure the generator's prior knowledge of the scripts rather than its memory.

## Reported performance and resource metrics

Every task row in `results.json` carries a `performance` block with the five metric families
[the design principles](design-principles.md#end-to-end-memory-and-search-speed) require. Each
block names the span it was measured from, so a number is never separated from its definition.

| Block | Quantity | Measured from |
| --- | --- | --- |
| `ingest` | Accepted input to durable, searchable memory; item and call latency; sustained items per second | `mindbridge.benchmark.ingest`, one span per accepted `add_many` batch |
| `search` | Retrieval latency at p50, p95, and p99, plus queries per second | `mindbridge.retrieve`, one span per `ask` over the whole retrieval leg |
| `answer` | End-to-end answer latency at p50, p95, and p99, and time to first token | `mindbridge.ask` and `mindbridge.model.generation` |
| `asr` | Audio seconds per wall second and transcription inference latency | `mindbridge.model.transcription` |
| `nodes` | Count, total time, mean, and p50/p95/p99 for every traced stage | every operation, stage, and model span |

`ingest` measures accepted input to durable and searchable memory rather than the time until the
call returned. That is exact rather than approximate: `add` and `add_many` return only after the
SQLite commit, the Zvec flush, and the search-index outbox acknowledgement, so the wall clock of
the traced call already includes index visibility.

`answer.latency_ms` and the per-task `answer_latency_ms` are timed after concurrency admission, so
they are response latency and not queue depth. `answer.time_to_first_token_ms` is `null` unless the
selected generation backend actually streamed a first token; a wall-clock number is never
substituted for one.

`search` is measured from `mindbridge.retrieve`, one stage span per `ask` covering the whole
retrieval leg: query embedding, content preparation, the index lookup, and ranking. It is not the
index lookup alone, which appears separately under `nodes` as `mindbridge.index.search`. Because
nodes are keyed by span name, the distribution never mixes that narrower stage in, and it excludes
`search()`, which opens `mindbridge.search` instead -- so a run that only calls `search()`, and the
harness's `search` fallback after a failed `ask`, report nothing under `search`.

`asr` is omitted when no transcription ran. Its `real_time_factor` is audio seconds divided by
transcription wall seconds, so values above one mean faster than real time.

The run-level `resources` block records CPU seconds and utilization, peak resident bytes, storage
growth split into media, row, and vector bytes, and per-device peak GPU utilization and memory when
`nvidia-smi` answers. CPU and memory come from `resource.getrusage`, so they need no sampling; only
GPU utilization is polled. `storage.media_share` is the fraction of growth that is source media,
which is the term that dominates long-run storage.

## Mandatory controls

A score is not interpretable on its own, so `results.json` reports three controls per task in a
`controls` block and the console table renders each of them as its own column. Each of these has
independently invalidated a conclusion on this project.

| Control | Why it is mandatory |
| --- | --- |
| `random_ranker` | Retrieval recall can be high by chance. A uniformly random ranker over a small candidate pool already reaches R@10 near 1.0. |
| `blind` | A no-memory arm can already score well, so a headline number can look strong while measuring nothing about memory. |
| `recall_at_20` next to `recall_at_1` | R@20 is the measured retrieval ceiling on this harness, so a change in R@1 with no change in R@20 is noise. |

`controls.missing` lists the absent controls, `controls.interpretable` is false whenever any is
absent, and the run-level `controls_complete` is false if any task is uninterpretable. The console
table renders every absent control as `MISSING` and prints one `UNINTERPRETABLE:` line per affected
task on standard error. A task row that carries no `controls` block cannot be rendered at all.

The random-ranker row is the exact expectation `min(1, k / candidate_pool_size)` for a uniform
ranker over the same candidate pool, not a shuffled sample, so it adds no variance and no run time.
Measured recall and the random-ranker expectation are only available for tasks whose adapter
carries gold evidence source IDs; when it does not, `retrieval.gold_evidence_key` is `null`,
`retrieval.unavailable_reason` says so, and the controls are reported as missing rather than
quietly omitted.

## Gold evidence per benchmark family

A gold evidence label is a set of memory **source IDs** — the `source_id` an adapter gives each
stored memory, which comes back on every retrieved hit. Only a family that can name those IDs can
have `recall_at_1`, `recall_at_20`, or a random-ranker control at all; the rest print `MISSING`,
which is the honest state and not a defect to paper over.

| Family | What the release publishes | Verdict |
| --- | --- | --- |
| `locomo-refined` | `qa[].evidence`, a list of `dia_id` values | Exact: `dia_id` is the stored source ID |
| `longmemeval` | `has_answer` on the answering turn, and the coarser `answer_session_ids` | Exact at turn level |
| `atm-bench` | `evidence_ids` naming emails and media records | Exact |
| `mem-gallery` | `clue_ids`, the clue round IDs | Exact |
| `mm-lifelong` | `total_intervals`, and `clue_intervals[].video_id` on the week and month splits | Interval-level only, and already reported as the official `ref_at_300`. The clue video IDs cannot be joined: prepared clips are keyed by file stem, not by release video ID |
| `supermemory-vqa` | `question_evidence.time_spans[].video_id`, kept as `source_video_ids` | Source-video level only. The join exists — `prepare_media` writes `<video_id>-video-#####` — but scoring it needs a group recall ("any clip of each gold video"), a different operator from the exact set recall above. Not implemented |
| `m3-bench` | `timestamp` and `before_clip` | Not derivable: both say when the question is asked, not where the answer is |
| `memlens` | Nothing beyond the answer | Not derivable |
| `clbench` | `context_id`, which names the whole unit | Not derivable: a label equal to the unit cannot separate rankers |
| `beam` | Rubrics and reference answers; `turns[].id` is a turn's own index and no question refers to one | Not derivable |
| `personamem-v3` | Slate-internal `_origin` and `_held_out_persona_item` | Unresolved. Those fields are deliberately excluded from the rendered slate because they are the answer; whether `_origin` names a `source_object_id` that matches an `event_id` needs a check against the corpus |
| `egolifeqa`, `egomemreason` | Query time only | Not derivable |
| `egotempo` | One clip per unit | Degenerate: a one-candidate pool cannot separate rankers |
| `openeqa` | `episode_history`, which is the unit | Not derivable |
| `video-mme`, `video-mme-v2` | Nothing beyond the answer | Not derivable |

So retrieval quality is measurable on **4 of the 17 families** in the catalog. `recall_at_20` — the
premise behind treating index content rather than ranking as the lever — is checkable on those four
and on no others, and two of them were only wired in after that premise was already load-bearing.
Treat it as a four-sample generalisation.

The two exact labels wired here are joined differently, because the risk differs. LongMemEval marks
the answering turn as the memories are built, so its label is exact by construction; a turn over
the part limit becomes several `_B####` blocks and every block of a marked turn is gold.
LoCoMo-Refined publishes a separate list that has to be matched onto the stored turns, so an
evidence ID naming no stored turn is counted in `retrieval.unresolved_gold_evidence_ids` instead of
being dropped. That count is the join's health: were a release's label vocabulary not the source-ID
vocabulary after all, recall would otherwise read as a plausible number over whichever IDs happened
to match.

Produce the blind control with a second run that ingests nothing and answers every question
through the same public path, then pass it back in:

```bash
mindbridge-bench eval --tasks locomo-refined --blind \
  --output-path .benchmarks/results/blind

mindbridge-bench eval --tasks locomo-refined \
  --blind-baseline .benchmarks/results/blind \
  --output-path .benchmarks/results/memory
```

`--blind-baseline` rejects a document that did not come from a `--blind` run and rejects one whose
`evaluation_sha256` differs, so a memory-backed run cannot be presented as the control.

## Noise floor

Each task row carries a `noise_floor` block with the per-question standard deviation, the
cluster-robust standard error, and `minimum_meaningful_difference`: the larger of the measured
three-point floor and the two-run interval implied by that standard error. A difference smaller
than that is inside the run-to-run noise band and is not a result. `--compare` rows repeat
`noise_floor` and add `below_noise_floor` for the observed delta.

## Results and reproducibility

Each completed `eval` output directory contains:

- `samples.jsonl`: one prediction and its native metrics, evidence intervals, retrieval diagnostics,
  and structured failure fields per sample.
- `results.json`: dataset and implementation pins, aggregate metrics, confidence intervals,
  performance, token usage, abstentions, mandatory controls, the noise floor, resource usage, and a
  digest of `samples.jsonl`.
- `egomemreason_submission.json`: only for a complete, valid EgoMemReason run.

Prompts, references, and raw judge responses are retained only with `--log-samples`. Treat that
option as sensitive: benchmark artifacts can contain source content, retrieved evidence, and model
responses.

The runner fixes seeds and generation temperature, records model endpoints and scorer protocols,
and marks whether each metric used the required official judge. A provider may still change an
unversioned model, so publishable runs should use immutable model identifiers and report hardware,
dataset selection, retrieval limit, and whether the index was cold, warm, or optimized.

A quality claim has to identify the dataset and revision, the official split and evaluator, the
input route, the model and runtime revisions, the retrieval settings, the hardware, and the
measured latency and resource cost. `results.json` records each of those:

| Required field | Where it is recorded |
| --- | --- |
| Dataset and revision | `tasks[].source_repository`, `source_revision`, `dataset_path`, `dataset_sha256`, `input_sha256`, `media_source` |
| Official split and evaluator | `tasks[].evaluation_sha256`, `primary_metric`, `official_metric`, `scorer_protocol`, `official_judge_model`, `judge_model_official` |
| Input route | `tasks[].input_modalities` and `performance.token_usage.calls_by_input_modality` |
| Model and runtime revisions | `model.*`, `environment.mindbridge_version`, `zvec_version`, `runtime_versions`, `python_version`, `platform` |
| Retrieval settings | `recall_limit`, `tasks[].retrieval.recall_limit`, and the full `model.memory_config` dump |
| Hardware | `environment.hardware` and the `resources` block |
| Latency and resource cost | `tasks[].performance`, `tasks[].answer_latency_ms`, and `resources` |
| Replay inputs | `run_id`, `seed`, `seeds`, `bootstrap_samples`, `limit`, `offset`, `batch_size`, `blind`, `blind_baseline` |

Scores are comparable only against runs of this harness at the same runner version, dataset
revision, and scorer protocol. Every task row therefore carries `cross_harness_comparable: false`
and a `comparability_note`. Vendor and third-party numbers for the same dataset are not
comparable: LoCoMo has ranged from 28.0 to 92.5 across harnesses on identical data. Report which
harness produced a number, and never place two harnesses' numbers in one column.

The task-family table used for metric breakdowns comes only from
`mindbridge.benchmarks.official_scorers.task_family`. A second copy in the runner previously
drifted and crashed report generation for four benchmarks; a unit test now pins every declared
breakdown family against that single table.

Questions sharing one memory are clustered as one independent unit. Confidence intervals and
regression significance remain unavailable when fewer than two independent units are present.
Partial, failed, or unverified runs remain useful for development but are not leaderboard-comparable.

Compare a run with an equivalent baseline using stable sample IDs:

```bash
mindbridge-bench eval \
  --tasks locomo-refined \
  --compare .benchmarks/results/baseline \
  --fail-on-regression \
  --regression-threshold 0.01
```

The comparison rejects incompatible dataset, scorer, or judge identities. Any answer or ingest
failure also makes `--fail-on-regression` exit nonzero.

Use `--use-cache .benchmarks/response-cache` to persist deterministic generation responses across
isolated reruns. The cache is an optimization, not a substitute for the result artifacts.

## Isolation contract

The built-in runner allocates this shape atomically:

```text
.benchmarks/data/
└── benchmark-run/
    ├── unit-a/  # SQLite, Zvec, and lock
    └── unit-b/  # a different physical store
```

Labels belong to the harness and filesystem only. Do not pass them into `Memory.add`, store them as
hidden product fields, or treat metadata as an isolation boundary. Distinct unit directories may
run concurrently; the same directory has one live owner. Evaluation CUDA runs also take a
per-device process lock unless an external scheduler owns admission control.

Custom behavior benchmarks should use only the public SDK: create a directory, construct `Memory`,
ingest through `add` or `add_many`, query through `search` or `ask`, score public return values, and
close the instance before archiving artifacts.

## Local-index microbenchmark

The synthetic benchmark isolates the SQLite-to-Zvec storage path:

```bash
mindbridge-bench local-index \
  --data-dir .benchmarks/local-index/trial-001 \
  --rows 1000 \
  --dimension 128 \
  --queries 20 \
  --k 10 \
  --seed 42 \
  --quantization none
```

`--data-dir` must be empty. The JSON result reports ingest and optimization time, recall at `k`
against exact search, query latency percentiles and throughput, plus SQLite, Zvec, and total bytes.
Run each quantization mode against a separate directory.

This command deliberately measures local adapters directly, which is the narrow storage
microbenchmark exception in `AGENTS.md`, not a second product API. Its JSON therefore labels
itself with `scope: storage_microbenchmark` and an `excludes` list. Its `ingest_seconds` is a
synthetic-vector storage number and is not the product ingest figure: it never embeds, routes a
modality, prepares media, grounds an answer, or touches `Memory`. The product ingest latency and
throughput come from the `ingest` block of an `eval` run, which drives the public SDK.

## Artifact safety

Benchmark directories may contain licensed dataset content, embeddings, prompts, and responses.
Apply the upstream terms reviewed before download, restrict artifact permissions, and remove the
artifacts when the experiment no longer needs them. Do not publish raw samples merely because the
aggregate metric is publishable.
