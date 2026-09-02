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

## Baseline arms

A MindBridge score on its own is unattributable: it does not say how much of the answer came from
memory rather than from the generator's prior, and a retrieval score does not say whether the
ranking carried any information. `--arms` runs the baselines that answer those questions beside
the product, sharing one ingest per unit:

```bash
mindbridge-bench eval \
  --tasks atm-bench-easy \
  --arms mindbridge,blind,full-context,random \
  --full-context-chars 24000
```

| Arm | Answers from | Retrieval | Reports |
| --- | --- | --- | --- |
| `mindbridge` | `Memory.ask` over retrieved evidence | the product's | every metric |
| `blind` | the generator's prior, with no evidence | none | answer metrics only |
| `full-context` | the corpus stuffed into one prompt, oldest first, under `--full-context-chars` | none | answer metrics only |
| `random` | nothing; it generates no answer | a seeded uniform shuffle of the same ranked candidates | retrieval metrics only |

The default is `mindbridge` alone. Each arm tags its samples and its task row with `arm`, and
`results.json` records every selected arm's definition -- prompt version, budget, and random seed
-- under `arms`. Sample IDs of a non-default arm are prefixed with its name, so `samples.jsonl`
stays one row per answered question per arm.

Three properties of the baselines are load-bearing when quoting them:

- **The two generating baselines are outside the product path by construction.** `Memory.ask`
  abstains before it reaches the model when no hit survives grounding, so neither could exist
  through it. They call the configured generation model with a harness-owned prompt, versioned as
  `mindbridge_blind_v1` and `mindbridge_full_context_v1`, and are scored and judged by the same
  scorers as the product arm. No baseline number is ever stamped `official_metric`.
- **They are text-only.** Media in a question is dropped from the blind prompt, and media in a
  corpus is not stuffed, so on a video or audio task both arms are lower bounds.
- **`random` shuffles the retriever's own candidate pool.** It holds pool membership fixed and
  randomizes order, which isolates the ranking. It is not a random draw from the whole corpus: on
  a corpus small enough for the pool to cover it, a random ranker can score near-perfect recall,
  and that is exactly the number worth printing next to the product's.

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

## Results and reproducibility

Each completed `eval` output directory contains:

- `samples.jsonl`: one prediction and its native metrics, evidence intervals, retrieval diagnostics,
  and structured failure fields per sample, per arm.
- `results.json`: dataset and implementation pins, arm definitions, aggregate metrics, confidence
  intervals, performance, token usage, abstentions, and a digest of `samples.jsonl`.
- `egomemreason_submission.json`: only for a complete, valid EgoMemReason run, from the
  `mindbridge` arm.

Three result fields carry a caveat that decides whether they can be quoted:

- **`official_metric` means the pinned upstream protocol publishes that metric and, for a judged
  one, that the required judge produced it.** The retrieval and joint diagnostics -- `retrieval_*`
  and `joint_*` -- are MindBridge's own and are never official, however faithful the rest of the
  run was. `official_scorers.py` holds the per-family registry that decides this.
- **`abstentions` undercounts.** It matches one exact English refusal sentence the answer backend
  emits, so a model that refuses in its own words is not counted. A measured EgoLifeQA slice
  reported 2 of 51 while 14 of 51 answers read as refusals. Treat it as a lower bound and read the
  predictions before drawing a conclusion about refusal rates.
- **`retrieval_*` scores the retriever's ranked candidate list, not the answer's evidence.** The
  runner takes the top `retrieval_candidate_limit` (100) hits in score order through `search`, so
  a miss there is a retrieval failure. What the answer actually grounded on is separate:
  `evidence` is the answer's hits, and `dropped_hits` counts what the answerer's inline context
  budget removed. A gold that is in the candidate list but not in `evidence` is budget loss, not
  retrieval loss. A replayed answer from `--use-cache` carries no retrieval metrics: the cache
  stores answers, not candidate lists. `ref_at_300` stays a property of the answer's evidence.

`performance` is aggregated per task across every arm of that task, not per arm.

Prompts, references, and raw judge responses are retained only with `--log-samples`. Treat that
option as sensitive: benchmark artifacts can contain source content, retrieved evidence, and model
responses.

The runner fixes seeds and generation temperature, records model endpoints and scorer protocols,
and marks whether each metric used the required official judge. A provider may still change an
unversioned model, so publishable runs should use immutable model identifiers and report hardware,
dataset selection, retrieval limit, and whether the index was cold, warm, or optimized.

Questions sharing one memory are clustered as one independent unit. Confidence intervals and
regression significance remain unavailable when fewer than two independent units are present.
Partial, failed, or unverified runs remain useful for development but are not leaderboard-comparable.

`--compare` is a regression guard, not a baseline. It pairs the current `mindbridge` arm against a
prior MindBridge run over identical dataset, scorer, and judge identities, using stable sample IDs;
it cannot say how much of a score came from memory. Use `--arms` for that.

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

This command deliberately measures local adapters directly. It does not measure embedding quality,
grounded-answer quality, or provider latency; use `eval` for those claims.

## Artifact safety

Benchmark directories may contain licensed dataset content, embeddings, prompts, and responses.
Apply the upstream terms reviewed before download, restrict artifact permissions, and remove the
artifacts when the experiment no longer needs them. Do not publish raw samples merely because the
aggregate metric is publishable.
