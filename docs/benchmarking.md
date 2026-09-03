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

## Run a bounded evaluation

Configure an OpenAI-compatible generation endpoint, then start with one task and one memory unit:

```bash
export MINDBRIDGE_GENERATION_API_KEY="..."

uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --model-args generation_model=gpt-5-mini \
  --limit 1 \
  --seed 42 \
  --output-path .benchmarks/results/locomo-one-conversation
```

For LoCoMo-Refined, `--limit 1` selects one conversation and then evaluates every question in it.
The first conversation in the pinned dataset currently has 138 questions, so this is a bounded run,
not a one-question or one-request smoke test.

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

## Configure the evaluation

Pass `--config eval.yaml` to declare the product composition and harness settings together. The
document is YAML, which also accepts JSON. Its top level is the composition accepted by
`Memory.from_config` plus one harness-owned `benchmark` mapping:

```yaml
embedding:
  provider: openai
  base_url: http://127.0.0.1:8000/v1
  api_key: ...
  model: local-embedder
  dimension: 1536
generation:
  provider: openai
  base_url: https://your-endpoint.example/v1
  api_key: ...
  model: qwen3.8-flash
  min_video_seconds: 2
speech:
  provider: funasr
  device: cuda
benchmark:
  judge:
    model: qwen3.8-flash
    base_url: https://judge.example/v1
    api_key: ...
    timeout_seconds: 600
  download:
    benchmarks_root: /corpus/.benchmarks
    hf_home: /corpus/huggingface
    hf_endpoint: https://huggingface.co
    youtube_sleep_seconds: 30
  run:
    tasks: locomo-refined
    arms: mindbridge
    limit: 10
    seed: 42
```

The `benchmark` mapping carries what `MindBridgeConfig` has no field for: judging, corpus
acquisition, and, under `benchmark.run`, run tunables. Model endpoints, credentials, modalities,
timeouts, token ceilings, and `extra_body` stay in the product block that owns them. See the
annotated [example configuration](examples/eval.example.yaml) for every field and its purpose.

`benchmark.run` mirrors these flags: `tasks`, `arms`, `full_context_chars`, `limit`, `offset`,
`seed`, `bootstrap_samples`, `batch_size`, `max_batch_size`, `unit_concurrency`,
`request_concurrency`, `judge_concurrency`, `recall_limit`, `device`, `device_lock`, `use_cache`,
`run_id`, `output_path`, `overwrite`, `log_samples`, `predict_only`, `download`,
`allow_unverified_data`, `verbosity`, `quiet`, `compare`, `fail_on_regression`,
`regression_threshold`, `media_manifest`, `task_data`, `media_root`, and `num_fewshot`. Omitted
keys retain the unset-flag defaults.

`--blind` and `--blind-baseline` remain command-line-only because they label or attach a whole
no-memory control run rather than an arm sweep. `--blind` cannot be combined with an arm selection
in the file. `--config`, the literal `--model mindbridge`, and the `--list-tasks` and
`--check-integrity` action modes also remain command-line-only. The three `*-args` strings are
shorthand: `--model-args` writes generation endpoint settings, `--gen-kwargs` writes
`generation.max_tokens` and `generation.extra_body`, and `--judge-model-args` writes
`benchmark.judge`.

A reproducible run pins generation temperature to zero and uses the run seed, so
`generation.temperature` and `generation.seed` are rejected. Disable model thinking through the
generation block instead:

```yaml
generation:
  provider: openai
  extra_body:
    chat_template_kwargs:
      enable_thinking: false
```

Omitted sections use the same defaults as a run without `--config`: a `jina-omni` embedder and an
`openai` generation endpoint. The runner always replaces `data_dir` with isolated per-unit
directories.

```bash
uv run --frozen mindbridge-bench eval \
  --tasks longmemeval-s \
  --config eval.yaml \
  --limit 10 \
  --output-path .benchmarks/results/longmemeval-configured
```

Settings resolve in this order: command-line flag, configuration file, environment, built-in
default. `--device` overrides configured local embedding and speech devices. Without a separate
judge setting, the judge reuses the generation endpoint. The request timeout defaults to 300
seconds. The [configuration reference](configuration.md) documents the product schema, and the
[Python SDK construction contract](api/python-sdk.md#construction) describes the composition
boundary.

Download settings mirror `MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS`, `HF_HOME`, and `HF_ENDPOINT`.
The resolved values are published to the process environment because `huggingface_hub` reads its
cache and endpoint when first imported.

`--benchmarks-root` defaults to the checkout's `.benchmarks` directory, resolved from the
repository rather than the process directory; `--data-root` defaults to
`<benchmarks-root>/data`. Run `mindbridge-bench eval --help` for the full concurrency, cache,
generation, and comparison options.

`--limit` accepts `-1`, a fraction between zero and one, or an absolute adapter-unit count. Use an
integer for a count; a non-integral value above one is truncated to an integer by the current
loader selection. Its unit is adapter-specific: OpenEQA limits episodes and retains every question
in a selected episode, while EgoTempo limits questions. Use `--check-integrity` or the completed
`results.json` instead of assuming that `--limit` equals `question_count`.

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

## Supported benchmark categories and primary metrics

The executable catalog currently contains 17 benchmark families expanded to 31 concrete tasks.
They fall into three dataset categories and one local systems microbenchmark. Classification follows
the primary workload; several suites deliberately overlap categories.

| Category | What it measures | Use it when |
| --- | --- | --- |
| Behavioral and long-term memory | Multi-session recall, temporal reasoning, knowledge and preference updates, abstention, personalization, and long-context learning | Comparing the behavior of complete memory compositions on text or captioned histories |
| Multimodal personal memory | Retrieval and answering over personal archives or conversations containing text, images, video, and email | Testing cross-modal evidence grounding in user histories |
| Embodied, video, and spatial memory | Causal video histories, egocentric lifelog reasoning, temporal localization, grouped video reasoning, and fixed-scene episodic QA | Testing long-running agents, wearables, robots, video assistants, or spatial memories |
| Local storage and retrieval | Direct SQLite-to-Zvec vector ingestion, exact-search recall, query latency, throughput, and disk growth | Isolating the embedded index from embedding, generation, and application behavior |

All dataset benchmarks enter through:

```bash
uv run --frozen mindbridge-bench eval --tasks <selector>
```

Every `mindbridge` arm needs a generation provider. The tables name the additional judge required
for an official primary result; `--predict-only` skips that judge and therefore cannot produce the
judged primary result. A judge may reuse the generation endpoint, but a publication-comparable
metric requires the named model. Selectors, variants, source revisions, and local readiness come
from `mindbridge.benchmarks.task_catalog`; `--list-tasks` remains authoritative.

### Behavioral and long-term memory

| Benchmark and selector | What it measures and when to use it | Primary result and scoring requirement | Data requirement |
| --- | --- | --- | --- |
| LoCoMo-Refined (`locomo-refined`) | Multi-session dialogue QA, temporal questions, captioned-image turns, and exact source-ID retrieval; use for conversational long-term memory | `llm_judge`; judge `qwen3-14b` | Pinned GitHub JSON; automatic |
| MemLens (`memlens`: 32K/64K/128K/256K) | Information extraction, multi-session and temporal reasoning, knowledge updates, and refusal over dated conversations; use for scaling context length | `accuracy`; judge `qwen3-235b-judge` | Pinned Hugging Face JSON and 195-question subset; automatic; published captions need no runtime media |
| LongMemEval (`longmemeval-s`) | User, assistant, and preference recall plus multi-session reasoning, updates, abstention, and exact turn-level retrieval; use for established long-term dialogue behaviors | `accuracy`; judge `gpt-4o-2024-08-06` | Pinned Hugging Face JSON; automatic |
| BEAM (`beam`: 100K/500K/1M/10M) | Very-long dialogue with contradiction resolution, ordering, extraction, updates, summarization, and temporal reasoning; use for length scaling | `llm_judge_score`; judge `gpt-4.1-mini` | Pinned GitHub tier directories; automatic |
| PersonaMem-v3 (`personamem-v3`) | Causally masked cross-app personalization, preference shifts, sycophancy, privacy, hallucination, and candidate ranking; use for personal-agent behavior | `personamem_score`; judged families use `gpt-5.5`, ranking rows are deterministic | Pinned Hugging Face backend JSON; automatic; scorer-only `profile.json` is excluded |
| CL-Bench (`clbench`) | Learning a long reference document and following open-ended instructions; use for task-local context learning, not gold-source retrieval | `solving_rate`; judge `gpt-5.1` | Pinned Hugging Face JSONL; automatic |

### Multimodal personal memory

| Benchmark and selector | What it measures and when to use it | Primary result and scoring requirement | Data requirement |
| --- | --- | --- | --- |
| ATM-Bench (`atm-bench`: main/hard, raw/SGM) | Email, image, and video memory; needle-in-a-haystack, number/list, open-ended answers, and exact evidence-ID retrieval; use for personal archives | `accuracy`; `open_end` uses judge `gpt-5-mini`, number/list rows are deterministic | Pinned Hugging Face QA, email, media, and SGM artifacts; automatic; SGM variants use processed text instead of runtime media |
| Mem-Gallery (`mem-gallery`) | Multi-session persona dialogue, image-grounded QA, temporal/knowledge/recall points, and exact clue-round retrieval; use for conversational image memory | `f1`; deterministic; optional official `llm_judge` uses `qwen2.5-72b-instruct` | Pinned Hugging Face dialogue JSON and images; automatic |

### Embodied, video, and spatial memory

| Benchmark and selector | What it measures and when to use it | Primary result and scoring requirement | Data requirement |
| --- | --- | --- | --- |
| EgoLifeQA (`egolifeqa`) | Multi-day causal video/audio memory, names, last occurrences, and temporal questions; use for wearable lifelogs | `accuracy`; deterministic choice scorer | Pinned Hugging Face annotations and EgoLife media; automatic |
| EgoMemReason (`egomemreason`) | Multi-day egocentric reasoning with 4--10 choices; use to generate an official-server submission because public labels are withheld | `submission`; no local judge | Pinned Hugging Face annotations and EgoLife media; automatic |
| EgoTempo (`egotempo`) | Open-ended temporal QA over Ego4D clips; use for temporal grounding rather than multi-session retrieval | `accuracy`; judge `gemini-1.5-flash` | Pinned GitHub annotations; media needs Ego4D authorization and AWS credentials |
| MM-Lifelong (`mm-lifelong`: day/week/month) | Day-to-month video memory, multi-interval clues, temporal localization, and open-ended answers; use for duration scaling | `answer_accuracy`; judge `gpt-5` | Pinned Hugging Face annotations and split media; automatic |
| SuperMemory-VQA (`supermemory-vqa`) | Causal multi-video memory, skill breakdowns, answerability, and unanswerable cases; use for lifelong video QA | `qa_accuracy`; deterministic choice scorer | Pinned Hugging Face annotations, transcripts, and video; automatic |
| M3-Bench (`m3-bench`: robot/web) | Causal long-video memory and open-ended QA; use for robot and web-video histories | `accuracy`; judge `gpt-4o-2024-11-20` | Pinned GitHub annotations; robot media from Hugging Face, web media through `yt-dlp` |
| Video-MME (`video-mme`) | Short, medium, and long cross-domain video understanding; use as a video-generation baseline, not as evidence of long-term memory alone | `accuracy`; deterministic choice scorer | Pinned Hugging Face Parquet and ZIP media; automatic |
| Video-MME-v2 (`video-mme-v2`) | Four-question relevance/logic groups with level and reasoning-head breakdowns; use when grouped consistency matters | `rating` (0--100); deterministic grouped scorer | Pinned Hugging Face Parquet and media volumes; automatic |
| OpenEQA (`openeqa`: HM3D/ScanNet) | Open-ended EM-EQA over fixed scene histories: spatial, recognition, localization, and world knowledge; use for spatial episodic memory | `llm_match`; judge `gpt-4-1106-preview` | Pinned GitHub questions; operator supplies extracted HM3D or licensed ScanNet frames |

### Local storage and retrieval microbenchmark

`uv run --frozen mindbridge-bench local-index` writes synthetic vectors directly to SQLite and
Zvec. It needs no dataset, generation model, or judge and reports exact-search recall, ingestion,
optimization and query latency, throughput, and disk use. It is the sole direct-adapter exception;
its result supports local-index claims only, not end-to-end memory, embedding, or answer quality.

### Result boundaries

One `eval` result mixes upstream scores with MindBridge-specific diagnostics. Interpret each field
at its declared boundary:

| Boundary | What it covers | What the result can establish |
| --- | --- | --- |
| Upstream protocol | A catalog task's pinned release, adapter, scorer, and required judge | Only a metric marked `official_metric: true` is an upstream-protocol result |
| MindBridge behavior | Public-SDK ingest, retrieval, answering, baseline arms, failures, latency, resources, and retrieval diagnostics | Product behavior under the recorded composition; custom diagnostics are not official metrics |
| Dataset and adapter | Download, schema normalization, digest verification, causal cutoffs, and unit/question counts | Input readiness and identity, not memory quality |

`mindbridge-bench locomo-refined` is an artifact utility, not another benchmark category: it emits
raw LoCoMo-Refined predictions for an external evaluator and does not produce an integrated score.
Likewise, `--list-tasks` and `--check-integrity` discover and validate inputs; they do not measure
memory quality.

Coverage is deliberately asymmetric:

- **Single-hop and multi-hop:** LongMemEval declares single-session and multi-session types;
  MemLens and BEAM declare multi-session reasoning. Other datasets may need several memories, but
  the catalog does not relabel them as multi-hop without an upstream type.
- **Time, updates, and conflicts:** LongMemEval, MemLens, and BEAM expose temporal or knowledge
  update types; BEAM also exposes contradiction resolution. PersonaMem-v3 adds causal preference
  shifts and sycophancy behavior. LoCoMo-Refined explicitly removed LoCoMo's adversarial category
  5, so it must not be cited as adversarial coverage.
- **Retrieval versus generation:** every dataset row measures generated answers. Exact
  MindBridge source-ID recall is available only for LoCoMo-Refined, LongMemEval, ATM-Bench, and
  Mem-Gallery. PersonaMem-v3's official candidate-ranking metrics rank answer slates and are not
  source-ID recall for the MindBridge retriever.
- **Open-ended versus open-domain:** open-ended scoring appears in many rows, while explicit
  cross-domain or world-knowledge breakdowns come from Video-MME and OpenEQA. Do not infer
  open-domain coverage from free-form answer format alone.

EgoMemReason's public release has no answer key. A complete valid run writes an upload-ready
`egomemreason_submission.json`; partial runs remain development artifacts. Video-MME-v2's grouped
`rating` is on a 0--100 scale. Other 0--5 diagnostics are named explicitly in `results.json`.

Protocol boundaries that affect interpretation are explicit rather than approximated:

- `locomo-refined`, `memlens`, `longmemeval-s`, `clbench`, `beam`, and `personamem-v3` need no
  runtime media preparation; LoCoMo-Refined and MemLens ingest the releases' published captions.
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

The following adapter and measurement details determine what the artifacts mean.

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

## Benchmarks without runtime media preparation

Six benchmark families read text, structured annotations, or published captions without opening
runtime media, so they need neither `ffmpeg` nor a preparation pass:

| Task | Unit | Corpus | Official headline |
| --- | --- | --- | --- |
| `locomo-refined` | one conversation | multi-session dialogue plus published image captions | `llm_judge`, the official correctness judge |
| `memlens-32k` … `memlens-256k` | one question | dated conversation sessions plus published image captions | `accuracy`, the official question-type judge |
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
  and structured failure fields per sample, per arm.
- `results.json`: dataset and implementation pins, arm definitions, aggregate metrics, confidence
  intervals, performance, token usage, abstentions, mandatory controls, the noise floor, resource
  usage, and a digest of `samples.jsonl`.
- `egomemreason_submission.json`: only for a complete, valid EgoMemReason run, from the
  `mindbridge` arm.

Three result fields carry a caveat that decides whether they can be quoted:

- **`official_metric` means the pinned upstream protocol publishes that metric and, for a judged
  one, that the required judge produced it.** The retrieval and joint diagnostics -- `retrieval_*`
  and `joint_*` -- are MindBridge's own and are never official, however faithful the rest of the
  run was. `official_scorers.py` holds the per-family registry that decides this.
- **`abstentions` undercounts.** It counts two things: the opaque marker the answer backend emits
  when it declines, and -- for a task whose own prompt mandates a refusal wording -- an answer
  equal to that wording. A model that refuses in its own free wording, on a task that mandates
  none, is still not counted. Measured under the older exact-sentence detector, an EgoLifeQA slice
  reported 2 of 51 while 14 of 51 answers read as refusals; treat the field as a lower bound and
  read the predictions before drawing a conclusion about refusal rates.
- **`retrieval_*` scores the retriever's ranked candidate list, not the answer's evidence.** The
  runner takes the top `retrieval_candidate_limit` (100) hits in score order through `search`, so
  a miss there is a retrieval failure. What the answer actually grounded on is separate:
  `evidence` is the answer's hits, and `dropped_hits` counts what the answerer's inline context
  budget removed. A gold that is in the candidate list but not in `evidence` is budget loss, not
  retrieval loss. A replayed answer from `--use-cache` carries no retrieval metrics: the cache
  stores answers, not candidate lists. `ref_at_300` stays a property of the answer's evidence.

`performance` is aggregated per task across every arm of that task, not per arm.

Before reporting a number, check these task fields:

| Field | Reporting rule |
| --- | --- |
| `primary_metric` and `score.mean` | Name and value of the headline result |
| `official_metric` | Must be true for an official-protocol claim |
| `score_valid` | Must be true; answer or ingest failures make it false |
| `question_count` and `score.cluster_count` | Report both sample size and independent memory units |
| `score.confidence_interval_95` | Present as `null` with fewer than two independent clusters |
| `dataset_sha256`, `evaluation_sha256`, `scorer_protocol`, and `judge_model` | Establish whether two runs are comparable |

Prompts, references, and raw judge responses are retained only with `--log-samples`. Treat that
option as sensitive because artifacts can contain licensed source content, retrieved evidence, and
model responses.

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
regression significance remain unavailable when fewer than two independent units are present;
`score.confidence_interval_95` remains present in the JSON with the value `null`.
Partial, failed, or unverified runs remain useful for development but are not leaderboard-comparable.

`--compare` is a regression guard, not a baseline. It pairs the current `mindbridge` arm against a
prior MindBridge run over identical dataset, scorer, and judge identities, using stable sample IDs;
it cannot say how much of a score came from memory. Use `--arms` for that.

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

This command deliberately measures local adapters directly, which is the narrow storage
microbenchmark exception in `AGENTS.md`, not a second product API. Its JSON therefore labels
itself with `scope: storage_microbenchmark` and an `excludes` list. Its `ingest_seconds` is a
synthetic-vector storage number and is not the product ingest figure: it never embeds, routes a
modality, prepares media, grounds an answer, or touches `Memory`. The product ingest latency and
throughput come from the `ingest` block of an `eval` run, which drives the public SDK.

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
