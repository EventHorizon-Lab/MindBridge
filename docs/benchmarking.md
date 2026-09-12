# Benchmarking

MindBridge provides one evaluation runner and three focused utilities. Use the evaluation runner for
quality claims; use the utilities only for their narrower artifact or storage purpose.

| Command | Use it for | Do not infer |
| --- | --- | --- |
| `mindbridge-bench eval` | Pinned datasets, official or explicitly identified scorers, confidence intervals, and run comparisons | That a score is leaderboard-comparable without checking its dataset, judge, and validity fields |
| `mindbridge-bench locomo-refined` | Raw LoCoMo-Refined predictions for another evaluator | An integrated benchmark score |
| `mindbridge-bench local-index` | SQLite-to-Zvec ingestion, recall, latency, throughput, and disk use | Embedding or answer quality |
| `mindbridge-bench control-plane` | Slow-loop quality on a seeded synthetic scenario: consolidation precision, contradiction recovery, false retirement, and rollback success | Retrieval or answer quality; the scenario is sized so one deliberation window covers it |

Never point a benchmark at an application's live `data_dir`. One physical directory has one live
MindBridge owner, and each independent benchmark unit needs its own new directory.

## Install and inspect tasks

From a repository checkout, install the model and dataset extras used by the evaluation harness:

```bash
uv sync --locked --default-index https://pypi.org/simple \
  --extra benchmarks --extra local --extra openai
```

Media tasks also need `ffmpeg` and `ffprobe`, and M3-Bench web media needs `yt-dlp`. These are
system tools, not Python dependencies this project manages.

List the current task groups, pinned revisions, and local-data readiness:

```bash
mindbridge-bench eval --list-tasks
```

The catalog currently covers LoCoMo-Refined, ES-MemEval, M3-Bench, Video-MME-v2, WorldMemArena,
EgoTempo, MemLens, MM-Lifelong, SuperMemory-VQA, ATM-Bench, Mem-Gallery, LongMemEval, CL-Bench,
BEAM, PersonaMem-v3, and OpenEQA. Use the listing command
instead of copying task names from this page; it is generated from the catalog used by the
runner.

Review each printed repository and revision before downloading data. Upstream terms vary:
LoCoMo-Refined and PersonaMem-v3 are CC BY-NC 4.0, MM-Lifelong is academic-only and restricts
redistribution or modification without prior approval, and CL-Bench carries an evaluation-only
license that forbids training, fine-tuning or distilling on the corpus. MindBridge's license does
not replace those terms; see the
[scorer notices](../src/mindbridge/benchmarks/_official/NOTICE.md).

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

## Run an evaluation

Set credentials for the OpenAI-compatible generation endpoint, then select a task or group:

```bash
export OPENAI_API_KEY="..."

uv run --frozen mindbridge-bench eval \
  --tasks locomo-refined \
  --output-path .benchmarks/results/locomo
```

The resulting `results.jsonl` reports `unit_count`, `question_count`, `dataset_sha256`, and
`evaluation_sha256` for each selected task.

Some datasets omit the clock used to interpret relative phrases such as "this morning". For a
reproducible run, supply a timezone-aware fallback without changing questions that already carry a
dataset or corpus date:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks m3-bench-robot \
  --fallback-reference-at 2026-09-08T00:00:00Z \
  --output-path .benchmarks/results/m3-fixed-clock
```

The fallback is passed only as the question's `reference_at`; it does not write event times or
knowledge-visibility timestamps. Results record the normalized clock and the number of questions
that used it. Invalid or timezone-naive values fail before provider clients are created.

ES-MemEval's QA task is selected through its family name; `--limit 1` means one seeker and all of
that seeker's questions:

```bash
uv run --frozen mindbridge-bench eval \
  --tasks es-memeval \
  --limit 1 \
  --output-path .benchmarks/results/es-memeval-qa-smoke
```

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
  --output-path .benchmarks/results/smoke
```

`--limit` accepts an absolute unit count, a fraction between zero and one, or `all` (equivalently
`-1`) for every unit. It is not always a question count: one memory unit may contain several
questions. Use the emitted `question_count` when reporting sample size.
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
  server_metrics:
    generation_url: https://your-endpoint.example/metrics
    embedding_url: http://127.0.0.1:8000/metrics
    timeout_seconds: 5
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
    repeat_index: 0
```

The `benchmark` mapping carries what `MindBridgeConfig` has no field for: judging, corpus
acquisition, optional process-global server telemetry, performance budgets, and, under
`benchmark.run`, run tunables. Model endpoints, credentials, modalities, timeouts, token ceilings,
and `extra_body` stay in the product block that owns them. See the
annotated [example configuration](examples/eval.example.yaml) for every field and its purpose.

`benchmark.run` mirrors these flags: `tasks`, `arms`, `full_context_chars`, `compile_max_items`,
`compile_max_chars`, `ingest`, `limit`, `offset`, `seed`, `bootstrap_samples`, `batch_size`,
`repeat_index`, `max_batch_size`, `unit_concurrency`,
`request_concurrency`, `judge_concurrency`, `recall_limit`, `device`, `device_lock`, `use_cache`,
`run_id`, `output_path`, `overwrite`, `log_samples`, `predict_only`, `stream_results`,
`download`,
`allow_unverified_data`, `verbosity`, `quiet`, `compare`, `fail_on_regression`,
`regression_threshold`, `media_manifest`, `task_data`, `media_root`, and `num_fewshot`. Omitted
keys retain the unset-flag defaults.

`--blind` and `--blind-baseline` remain command-line-only because they label or attach a whole
no-memory control run rather than an arm sweep. `--resume` is command-line-only for the same
reason: it describes one invocation's recovery, not the sweep a file declares. `--blind` cannot be combined with an arm selection
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

That is exactly what `--gen-kwargs enable_thinking=false` writes, so a file that declares it does
not need the flag.

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

Unless `--quiet` is set, `eval` reports ingest, sample, and judge progress to stderr. On a
terminal those are live progress bars with an ETA. While a sample is still rebuilding, ingesting,
deliberating, or answering, the sample bar keeps its honest completed-sample count but refreshes
its elapsed time and labels the active phase. Concurrent units are summarized by phase rather than
letting a completed unit leave a stale label behind.

A unit writes every memory a question is allowed to have seen before it answers that question, so
a task whose questions carry no cutoff writes its whole corpus first and the sample bar honestly
reads zero for as long as that takes -- hours, on a video task. The ingest bar beneath it is what
shows those hours moving: it counts memories written across every unit of the task, against the
number those units will reach at their latest cutoff. A resumed run starts it at the checkpoint
rather than at zero, and a unit answered entirely from the response cache completes its share
without writing, so the bar still finishes.

When stderr is a file or a pipe, where a redrawn bar is unreadable, the same counts and ETA are
written as one line at most once a minute, plus the first and the last completion, so a stalled
run says so immediately and the log always ends on the final count.

`--verbosity` sets the log level for the run and claims the root handler before an imported
dependency can raise it: `funasr`, `modelscope`, `numba` and others each turn their own logging
up when imported. It applies to MindBridge's own loggers; a dependency has to reach `WARNING` to
be heard, so a successful HTTP request logs nothing and a benchmark run does not carry anyone
else's INFO. `--verbosity DEBUG` is the exception and opens the whole process, per-request
transport lines included. The results table is printed regardless; `--quiet` (or
`--verbosity ERROR`) is what suppresses it.

`--limit` accepts `-1`, a fraction between zero and one, or an absolute adapter-unit count. Use an
integer for a count; a non-integral value above one is truncated to an integer by the current
loader selection. Its unit is adapter-specific: OpenEQA limits episodes and retains every question
in a selected episode, while EgoTempo limits questions. Use `--check-integrity` or the completed
`results.jsonl` instead of assuming that `--limit` equals `question_count`.

## RTX 5090, Qwen3.8, and WeMM reference baseline

Two checked-in profiles encode the reference composition with the explicit non-secret `EMPTY`
sentinel, so an ambient `OPENAI_API_KEY` is never forwarded to these endpoints:

- [text profile](examples/baselines/rtx5090-qwen38-wemm9b-text.yaml) for the text memory suites;
- [media profile](examples/baselines/rtx5090-qwen38-wemm9b-media.yaml) for native image, video,
  and local FunASR paths.

They use generation model `Qwen3.8-27B` at `http://xyrobot-vl.xyrobot.com/v1` and embedding model
`tencent/WeMM-Embedding-9B` at `https://xyrobot-embed.xyrobot.com/v1`, dimension 4096. A bare host
such as `xyrobot-embed.xyrobot.com` is deliberately rejected: every endpoint must name its scheme.
Both product clients disable hidden SDK retries. Adapter-level fallbacks remain explicitly counted,
so request counts, token completeness, and failure status describe the observable provider
attempts. The VLM keeps the baseline's declared HTTP transport; switching it to HTTPS is a different
network/TTFT baseline. Run one independent invocation per repeat:

```bash
for repeat in 0 1 2; do
  uv run --frozen mindbridge-bench eval \
    --config docs/examples/baselines/rtx5090-qwen38-wemm9b-text.yaml \
    --repeat-index "$repeat" \
    --run-id "rtx5090-text-r${repeat}" \
    --output-path ".benchmarks/results/rtx5090-text-r${repeat}"
done
```

Each invocation creates new physical stores. It performs one query-embedding warmup after model
construction and before telemetry starts, performs no generation warmup, and records the actual
warmup count. A vision-description cache, when configured, is shared only inside that invocation,
so a later repeat cannot silently skip write-path model calls. Do not use `--use-cache` for a
performance baseline; cached samples are excluded from performance denominators and cached runs
are rejected by performance comparison.

The RTX 5090 metadata and sampled GPU values describe the benchmark client. The configured VLM
and embedding URLs are remote services, so their GPU hardware is not attributed to the 5090.
Their configured `/metrics` endpoints instead provide process-global vLLM deltas and are marked
non-exclusive because other traffic may occur in the same interval. For a publishable baseline,
reserve both endpoints and the local GPU, record three or more repeats, and compare the same
repeat protocol rather than treating one run as noise-free.

## Baseline arms

A MindBridge score on its own is unattributable: it does not say how much of the answer came from
memory rather than from the generator's prior, and a retrieval score does not say whether the
ranking carried any information. `--arms` runs the baselines that answer those questions beside
the product, sharing one ingest per unit:

```bash
mindbridge-bench eval \
  --tasks atm-bench-easy \
  --arms mindbridge,blind,full-context,random,compile \
  --full-context-chars 24000 \
  --compile-max-items 24 --compile-max-chars 16000
```

| Arm | Answers from | Retrieval | Reports |
| --- | --- | --- | --- |
| `mindbridge` | `Memory.ask` over retrieved evidence | the product's | every metric |
| `blind` | the generator's prior, with no evidence | none | answer metrics only |
| `full-context` | the corpus stuffed into one prompt, oldest first, under `--full-context-chars` | none | answer metrics only |
| `random` | nothing; it generates no answer | a seeded shuffle of an independently fetched top-100 pool for gold-labelled questions (`recall_limit` otherwise) | retrieval metrics only |
| `compile` | `Memory.compile`'s rendered bundle, under `--compile-max-items`/`--compile-max-chars` | the compiler's own selection | answer metrics, plus `compile_bundle_chars` and `compile_bundle_items` per sample |

The default is `mindbridge` alone. Each arm tags its samples and its task row with `arm`, and
`results.jsonl` records every selected arm's definition -- prompt version, budget, and random seed
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
- **`random` reuses the retriever, not necessarily the product arm's pool membership.** At the same
  causal store state it independently requests 100 candidates for a gold-labelled question
  (`recall_limit` otherwise), then applies a seeded shuffle. When the product answer ranks fewer
  candidates, treat this as a random-order control at its recorded depth, not a strict ranking-only
  A/B over identical membership.

`compile` measures downstream task success from `Memory.compile()`'s bundle rather than
`Memory.ask()`: it calls `compile(question, budget=ContextBudget(max_items=..., max_chars=...))`,
renders the bundle with `bundle.render()`, and feeds that rendered text to the same generator call
the `full-context` arm uses (`compile()` never generates text itself, so a compiled answer has to
leave the product path the same way the two text baselines do). Reading it beside `blind`,
`full-context`, and `mindbridge` is exactly the no-memory/full-context/retrieval-only comparison
[Context OS gate 4](context-os.md#evolution-gates) asks for. Its per-sample `compile_bundle_chars`
and `compile_bundle_items` (in `metrics`) are the bundle's own size, the numerator half of "useful
evidence per token" -- pair them with the question's answer-quality metric to compute it; a run
does not compute the ratio itself, since only the caller knows which quality metric is the
numerator.

## Ingest mode

`--ingest add` (the default) ingests through `Memory.add_many`/`Memory.add`, the strong path: a
memory is searchable the instant ingest returns. `--ingest capture` ingests through
`Memory.capture()` followed by `Memory.settle()` instead, so a run actually exercises the fast
plane's acknowledge-then-enrich path end to end -- this is the only way capture acknowledgement and
time-to-searchable, in [Reported performance and resource metrics](#reported-performance-and-resource-metrics)
below, come from a real run instead of reading `0` in every task's `performance` block. Every arm
still answers against the same fully settled store either way; `--ingest` changes how the store got
there, not what an arm can see once ingest finishes.

## Supported benchmark categories and primary metrics

The executable catalog currently contains 16 benchmark families expanded to 30 concrete tasks.
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
| MemLens (`memlens`: 32K/64K/128K/256K) | Information extraction, multi-session and temporal reasoning, knowledge updates, and refusal over dated conversations; use for scaling context length | `accuracy`; judge `qwen3-235b-judge` | Pinned Hugging Face JSON, 195-question subset, and the release's 229 MB `release_images/` tree; automatic. A turn's image is ingested as media when its file is present, so a run needs a model that declares the image capability; without the images the published captions are ingested alone |
| LongMemEval (`longmemeval-s`) | User, assistant, and preference recall plus multi-session reasoning, updates, abstention, and exact turn-level retrieval; use for established long-term dialogue behaviors | `accuracy`; judge `gpt-4o-2024-08-06` | Pinned Hugging Face JSON; automatic |
| ES-MemEval (`es-memeval`, task `es-memeval-qa`) | Personalized long-term emotional-support dialogue QA across information extraction, temporal reasoning, conflict detection, abstention, and user modeling | `llm_judge` (the 0--2 rubric normalized to 0--1); judge `gpt-4o`; published `f1` is also reported; the semantically transcribed judge is marked non-official | Pinned GitHub EvoEmo JSON; automatic; upstream declares no license at the pinned revision |
| BEAM (`beam`: 100K/500K/1M/10M) | Very-long dialogue with contradiction resolution, ordering, extraction, updates, summarization, and temporal reasoning; use for length scaling | `llm_judge_score`; judge `gpt-4.1-mini` | Pinned GitHub tier directories; automatic |
| PersonaMem-v3 (`personamem-v3`) | Causally masked cross-app personalization, preference shifts, sycophancy, privacy, hallucination, and candidate ranking; use for personal-agent behavior | `personamem_score`; judged families use `gpt-5.5`, ranking rows use the deterministic `ndcg_at_5` formula frozen at upstream commit `ad80a3b1b322` | Pinned Hugging Face backend JSON; automatic; scorer-only `profile.json` is excluded |
| CL-Bench (`clbench`) | Learning a long reference document and following open-ended instructions; use for task-local context learning, not gold-source retrieval | `solving_rate`; judge `gpt-5.1` | Pinned Hugging Face JSONL; automatic |

LongMemEval assigns every stored turn or split block an opaque, per-question source-order ID such
as `M000000`. Release session labels, including answer and abstention markers, remain evaluator-only;
`has_answer` is mapped onto the same opaque IDs for retrieval scoring. This mapping is part of the
adapter version, so response caches from the earlier source-ID scheme are not reused.

### Multimodal personal memory

| Benchmark and selector | What it measures and when to use it | Primary result and scoring requirement | Data requirement |
| --- | --- | --- | --- |
| ATM-Bench (`atm-bench`: main/hard, raw/SGM) | Email, image, and video memory; needle-in-a-haystack, number/list, open-ended answers, and exact evidence-ID retrieval; use for personal archives | `accuracy`; `open_end` uses judge `gpt-5-mini`, number/list rows are deterministic | Pinned Hugging Face QA, email, media, and SGM artifacts; automatic; SGM variants use processed text instead of runtime media; raw variants need a `vision:` slot so image and video memories carry a text document, otherwise their only indexed text is the source ID |
| Mem-Gallery (`mem-gallery`) | Multi-session persona dialogue, image-grounded QA, temporal/knowledge/recall points, and exact clue-round retrieval; use for conversational image memory | `f1`; deterministic; optional official `llm_judge` uses `qwen2.5-72b-instruct` | Pinned Hugging Face dialogue JSON and images; automatic |

### Embodied, video, and spatial memory

| Benchmark and selector | What it measures and when to use it | Primary result and scoring requirement | Data requirement |
| --- | --- | --- | --- |
| WorldMemArena (`worldmemarena`) | Causal multimodal agent and lifelong sessions with checkpoint QA, updates, temporal reasoning, visual recall/search, and cross-modal reasoning | `correct_ratio`; the official configurable Correct/Hallucination/Omission judge, plus `answer_f1` and `answer_bleu1` | Pinned Hugging Face JSON and images; automatic |
| EgoTempo (`egotempo`) | Open-ended temporal QA over Ego4D clips; use for temporal grounding rather than multi-session retrieval | `accuracy`; judge `gemini-1.5-flash` | Pinned GitHub annotations; media needs Ego4D authorization and AWS credentials |
| MM-Lifelong (`mm-lifelong`: day/week/month-val; `mm-lifelong-month-train` by name) | Day-to-month video memory, multi-interval clues, temporal localization, and open-ended answers; use for duration scaling | `answer_accuracy`; judge `gpt-5` | Pinned Hugging Face annotations and split media; automatic |
| SuperMemory-VQA (`supermemory-vqa`) | Causal multi-video memory, skill breakdowns, answerability, and unanswerable cases; use for lifelong video QA | `qa_accuracy`; deterministic choice scorer | Pinned Hugging Face annotations, transcripts, and video; automatic |
| M3-Bench (`m3-bench`: robot/web) | Causal long-video memory and open-ended QA; use for robot and web-video histories | `accuracy`; judge `gpt-4o-2024-11-20` | Pinned GitHub annotations; robot media from Hugging Face, web media through `yt-dlp` |
| Video-MME-v2 (`video-mme-v2`) | Four-question relevance/logic groups with level and reasoning-head breakdowns; use when grouped consistency matters | `rating` and auxiliary `accuracy` (both 0--100); deterministic grouped scorer | Pinned Hugging Face Parquet and media volumes; automatic |
| OpenEQA (`openeqa`: HM3D/ScanNet) | Open-ended EM-EQA over fixed scene histories: spatial, recognition, localization, and world knowledge; use for spatial episodic memory | `llm_match` (0--100); judge `gpt-4-1106-preview` | Pinned GitHub questions; operator supplies extracted HM3D or licensed ScanNet frames |

### Local storage and retrieval microbenchmark

`uv run --frozen mindbridge-bench local-index` writes synthetic vectors directly to SQLite and
Zvec. It needs no dataset, generation model, or judge and reports exact-search recall, ingestion,
optimization and query latency, throughput, and disk use. It is the sole direct-adapter exception;
its result supports local-index claims only, not end-to-end memory, embedding, or answer quality.

### Control-plane behaviour benchmark

`uv run --frozen mindbridge-bench control-plane --config CONFIG --data-dir DIR` measures the slow
loop rather than recall. It ingests a seeded synthetic long run through `Memory.add_many` --
persons with a standing gold fact, one fact observed twice, and a preference stated then flipped
with a later `occurred_at` -- and then asserts each side of every flip as a host-authored
`RELATION` claim over its own observation, through `Memory.apply` with a `CONSOLIDATE` operation.
That step is what makes the flip resolvable at all: `CORRECT` and consolidation forgetting only
reach a *derived* record, so a correction of the raw observation is refused `not_derived` however
right it is, and a scenario without a former has no other way to hold a claim. `RELATION` is the
kind whose two sides both stand until the loop retires one -- a `STATE` is superseded by lineage
reconciliation as soon as the second claim lands, and a model-inferred `TRAIT` stays invisible
until a second evidence group supports it -- so the pair is what the `CONTRADICTION` trigger of
`consolidation_candidates()` reports: two disagreeing visible claims in one lineage.

Before that step existed the metric was structurally unreachable: every flip was two raw
observations, so `CORRECT` was refused `not_derived` on all of them and `contradiction_recovery`
could only ever read `0.0`. The benchmark configures no former, so any `CONTRADICTION` candidate an
earlier run reported came from claims the loop's own accepted `CONSOLIDATE` operations minted
mid-run, never from the injected observation pairs.

Two failing recalls per person follow, asking about a window after everything the scenario
recorded. Those are the whole of the `QUERY_FAILURE` signal, and they come last on purpose: a
candidate is dropped while nothing about it has changed since an operation last weighed it, and
the claims above are applied operations over the very records that query is nearest to.

It then runs `deliberate()` with the configured consolidator and scores the operation log against
the ground truth it injected:

| Metric | Definition | Undefined when |
| --- | --- | --- |
| `consolidation_precision` | Applied `CONSOLIDATE` operations whose cited evidence is exactly one injected duplicate group, over every applied `CONSOLIDATE` | The loop applied no consolidation |
| `contradiction_recovery` | Injected preference flips where the newer claim is in force and the older is out of recall -- forgotten, or its version retired by the `CORRECT` that resolves it -- over every injected flip | Never; the denominator is the scenario |
| `false_retirement` | Retired records that were gold-standing, over every record the run retired | The run retired nothing |
| `rollback_success` | Applied operations `rollback()` reversed, newest first; `state_restored` separately reports whether every ingested record came back into recall | The loop applied nothing |
| `deliberation.model_calls` | Backend round trips. `ConsolidationBackend` reports no token or currency cost, so this is the loop's whole cost proxy | Never |

Each applied operation is judged with `record_outcome()` before anything is rolled back, so the
operation log itself carries the verdict and `outcomes.confirmed` / `outcomes.refuted` are
derivable from the log alone. A `CORRECT` that retires the claim actually in force is judged
`REFUTED`, so the log does not call the wrong half of a flip a success. The scenario's own
`apply()` rows are not judged: they are the ground truth, not something the loop proposed. An
undefined rate is reported as `null`, never as `1.0`.

`--config` is a MindBridge configuration file declaring `embedding` and `consolidation`; a
`benchmark:` section, so an `eval` config can be reused verbatim, is ignored. The benchmark pins
`minimum_relevance=0` and `reinforce_on_answer=False` itself, for the same reproducibility reason
`eval` does.

### Running the slow loop during an evaluation

`mindbridge-bench eval --deliberate` runs `deliberate()` after each causal cutoff's ingest and
before that cutoff's questions, so "does the loop change QA scores" is measurable on the existing
tasks. It is off by default and refused when the configuration declares no `consolidation`
section. The run report carries `deliberation.enabled` and `deliberation.operations_applied`, and
the flag is part of the response-cache namespace, so a cached run is never reused across it.

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
  MindBridge source-ID recall is available for LoCoMo-Refined, LongMemEval, ATM-Bench, and
  Mem-Gallery. WorldMemArena image evidence can be joined exactly, but its `mp_*`
  labels name scorer-authored memory points rather than raw turns and remain unresolved.
  PersonaMem-v3's official candidate-ranking metrics rank answer slates and are not
  source-ID recall for the MindBridge retriever.
- **Open-ended versus open-domain:** open-ended scoring appears in many rows, while explicit
  cross-domain or world-knowledge breakdowns come from OpenEQA. Do not infer
  open-domain coverage from free-form answer format alone.

Video-MME-v2's grouped `rating` and question-level `accuracy` are both on a 0--100 scale. Other
0--5 diagnostics are named explicitly in `results.jsonl`.

Protocol boundaries that affect interpretation are explicit rather than approximated:

- `locomo-refined`, `memlens`, `longmemeval-s`, `es-memeval-qa`, `clbench`, `beam`, and
  `personamem-v3` need no
  runtime media preparation; LoCoMo-Refined ingests the release's published captions, and
  MemLens ingests its captions plus the release's own image files where they are present.
  MEMLENS hides 65.7% of its answers inside those images, so a caption-only run is a
  different protocol and must be reported as one; `--media-root` pointed at an empty
  directory selects it deliberately.
- CL-Bench has no separate question field. The adapter splits the final user turn at its last
  blank-line paragraph break and marks oversized residual questions with `question_unsliced`.
- BEAM reports its per-rubric `llm_judge_score`; `event_ordering` additionally runs the official
  pairwise event-equivalence calls and reports `precision`, `recall`, `f1`, `tau_norm`, and
  `final_score`. As in upstream `report_results.py`, its category result uses `tau_norm`.
- PersonaMem-v3 reads the released fields and causally masks future events. Task families requiring
  structured actions, response-threaded clusters, or paired-row deltas carry no official headline;
  `profile.json` is scorer-side ground truth and is never ingested as memory.
- SuperMemory-VQA reports `qa_accuracy`; `qa_mrr` is unavailable because the answer backend does
  not expose answer-option scores.
- OpenEQA reports 0--100 `llm_match` plus `llm_match_score_1_5`. Its fixed-history adapter is
  not the active-navigation A-EQA protocol.
- WorldMemArena support covers its official checkpoint-QA answer protocol. Its separate memory
  snapshot recall/correctness, update-handling, interference, and LLM evidence-coverage calls are
  reported unavailable because MindBridge does not export the per-session snapshot those scorers
  require. Gold memory-point summaries are never ingested as system memories.

Review the upstream repository and license printed by `--list-tasks` before download. Dataset terms
remain independent of MindBridge's license; the copied scorer licenses and protocol notes are in
the [scorer notices](../src/mindbridge/benchmarks/_official/NOTICE.md).

## Acquire and prepare data

The runner downloads missing pinned annotations and supported media by default, and verifies a
published digest when one is available. Start with one task and a small limit: the `all` group can
require hundreds of gigabytes. Long videos are prepared as deterministic bounded clips cached under
`.benchmarks/.prepared/`; preparation needs `ffmpeg` and `ffprobe`, and M3-Bench web media also
needs `yt-dlp`.

Prepared-video cache versions are isolated. A cache entry is reusable only when `ffprobe` finds a
video stream with at least one real frame; an audio-only or otherwise incomplete derivative is
rebuilt in the current cache version. The normal one-frame-per-second output stays unchanged when
it has at least two frames. A shorter output is re-encoded from the same bounded source interval
with its native cadence, audio mapping, and geometry so no synthetic frame or event time is
introduced. A consumer whose video processor needs two frames reports a genuine one-frame input as
unsupported at its own boundary rather than changing the generic prepared-media representation.
During a cache-version migration, a legacy one-frame derivative is rebuilt instead of reused because
the former one-frame-per-second sampling may have discarded real source frames.

M3-Bench-web's official release publishes YouTube URLs rather than a durable web-video archive.
When `yt-dlp` identifies a video as permanently unavailable (for example, private, removed, or
copyright-blocked), acquisition continues and records the unit and exact reason under
`unavailable_units` in the generated media manifest. Results then report incomplete
`dataset_coverage` and set `score_valid` and `score_comparable_to_full_dataset` to `false`; the
remaining samples are useful for development but are not a full-dataset benchmark score. Network,
authentication, throttling, and unknown download failures still stop the run instead of being
misreported as upstream data loss. Supplying an operator-managed copy with `--media-root` retains
the unit and produces complete coverage. An intentional `--limit` or `--offset` slice can have
complete selected coverage, but still reports `score_comparable_to_full_dataset: false`.

Use `--no-download` for an offline run. Override operator-managed inputs explicitly:

```bash
mindbridge-bench eval \
  --tasks video-mme-v2 \
  --task-data video-mme-v2=/datasets/video-mme-v2/test.parquet \
  --media-root video-mme-v2=/datasets/video-mme-v2/videos \
  --allow-unverified-data
```

`--allow-unverified-data` is required when an override does not match the catalog digest. The
result records the resolved dataset and memory digests so such a run cannot be silently confused
with the pinned release.

For a prepared causal stream, pass `--media-manifest FILE`. Causal tasks require source intervals;
the runner ingests only observations ending at or before the question cutoff. Relative paths are
resolved from the manifest file. When the runner prepares media itself, it writes one manifest
object to `OUTPUT/media-manifest.jsonl`; use that generated file as the format reference before
supplying an operator-authored replacement.

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
- **The headline is `llm_match`, reported 0--100.** It is the official LLM-Match protocol: the
  `mmbench` prompt, or `mmbench-extra` for the 263 questions that publish `extra_answers`, judged
  by `gpt-4-1106-preview` for a mark of 1-5. The raw mark is kept beside it as
  `llm_match_score_1_5`. Two upstream behaviours are reproduced rather than corrected: a
  prediction is cut after its last period when
  that period is not already its final character, and a mark outside 1-5 is clipped instead of
  rejected.

## Benchmarks without runtime media preparation

Seven benchmark families read text, structured annotations, published captions, or single
images without opening timed media, so they need neither `ffmpeg` nor a preparation pass:

| Task | Unit | Corpus | Official headline |
| --- | --- | --- | --- |
| `locomo-refined` | one conversation | multi-session dialogue plus published image captions | `llm_judge`, the official correctness judge |
| `memlens-32k` … `memlens-256k` | one question | dated conversation sessions, published image captions, and the release's image files | `accuracy`, the official question-type judge |
| `longmemeval-s` | one question | its own 50-session haystack | `accuracy`, the yes/no answer-check judge |
| `es-memeval-qa` | one seeker | every dated seeker/supporter session and all of that seeker's QA items | non-official adapted `llm_judge`, the GPT-4o 0--2 rubric normalized to 0--1 |
| `clbench` | one task | the reference document behind its question | `solving_rate`, the binary rubric judge |
| `beam-100k` … `beam-10m` | one conversation | the whole transcript | Category metrics: `tau_norm` for event ordering, `llm_judge_score` otherwise |
| `personamem-v3` | one persona | five engagement logs plus a calendar stream | `accuracy_pct_micro`, 0--100 when scorer coverage is complete |

Four of them need a note before a number is quoted:

- **ES-MemEval support covers the QA task.** The upstream suite also contains summarization and
  dialogue-generation experiments, whose multi-turn and event-based scoring contracts do not fit
  this QA runner. The adapter matches the published session-level RAG corpus, reports the upstream
  set-overlap `f1`, and retains the raw judge mark as `judge_score_0_2`. Because the pinned upstream
  repository has no declared license, the judge prompt is a compact semantic transcription rather
  than copied text, and both judge metrics carry `official_metric: false`. The paper's BERTScore is
  not run because it requires a separate learned evaluator and model download. Verify permission
  before downloading or using EvoEmo.

- **CL-Bench publishes no `question` field.** Each record's final turn mixes a reference document
  -- up to ~150,000 characters -- with the query in one string, and the loader splits it at the
  last blank-line paragraph break. 1,322 of the 1,899 records split cleanly (median question 434
  characters); 130 end up with a question of 2,000 characters or more and carry
  `question_unsliced` in their metadata. Filter on that field before reporting.
- **BEAM's `event_ordering` category has a distinct official path.** The runner issues the
  pairwise equivalence calls used for semantic alignment, then computes the same F1, normalized
  Kendall tau, and `final_score = tau_norm x f1`. The upstream report selects `tau_norm` for this
  category and `llm_judge_score` for the other nine.
- **PersonaMem-v3 is scored on the families the pinned release supports.** Its evaluation
  repository has drifted from the released data, so the reproduced protocols read only released
  fields: the unified personalization rubric (13 task types), the four task-specific judges, and
  the deterministic ranking family. Ranking freezes the formula at upstream commit
  `ad80a3b1b322`: positive items have gain +2, fillers +1, and hard negatives -2, with the
  hidden-persona filler gain set to zero only for historical slates that have no hard negatives.
  The resulting `ndcg_at_5` is the ranking headline. `target_only_ndcg@5` remains a local
  diagnostic. The deprecated `ndcg_graded@5` key is a compatibility alias for that target-only
  diagnostic and is not classified as an upstream metric. The proactive decision judge, the two repetition-fatigue
  cluster tasks, `new_suggestions_chatbot`, `local_recommendation_geo_shift`,
  `active_mistake_prevention` and `short_vs_long_term_lifecycle` are answered and reported but
  carry no official headline -- therefore the aggregate `accuracy_pct_micro` is explicitly
  unavailable instead of being calculated over a biased subset. The last one ranks a slate like
  the other three, but upstream scores it with a delta across two paired rows that a single row
  cannot carry. The cluster rows are dropped at load because their runner threads each response
  into the next prompt. Judge evidence is reconstructed from the release's frozen source-A slices,
  including same-day avoids, privacy flags, update contradictions, style references, and friend
  records.

PersonaMem-v3 masks history causally: each query is answered against only the events that happened
strictly before its timestamp, which the runner applies through the same cutoff machinery the
causal video tasks use. `profile.json` is the scorer-side ground-truth persona and is neither
downloaded nor read as memory.

`ScriptMem` is deliberately absent. Its public release ships questions, gold answers and a scorer,
but every `conversation` field holds only a `format_example` placeholder -- the four source scripts
are withheld for copyright -- so there is nothing for a memory system to retrieve and an offline
number would measure the generator's prior knowledge of the scripts rather than its memory.

## Reported performance and resource metrics

Every task and arm row in `results.jsonl` carries a `performance` object. Distributions retain every
observation and report exact count, average, p50, p95, and p99 values. Throughput divides successful
work by the union of successful active intervals, so concurrency overlap is counted once and idle
gaps do not inflate the denominator. TTFT and token-per-call distributions also report
`observed_count`; missing observations make `complete` false and the exact all-call average null.

| Block | Boundary and quantities |
| --- | --- |
| `duration_seconds` | Gap-free active wall-time unions for the product and judge phases. Its average denominator is the union of sample IDs measured in either phase, so a cached product answer judged by an uncached call is counted once. |
| `ingest` | Attempt, success, and error counts; attempted and accepted items; durable/searchable batch latency; compute and active-wall throughput. Speech analysis for a chunk usually ran ahead of it on the lent speech backend while the previous chunk was embedding, so a batch's durable latency excludes most of that model time; the transcription spans still carry it. |
| `search_e2e` | Run-global, post-answer warm-store replay of public `Memory.search(limit=recall_limit)`. The second pass starts only after every selected task has finished its formal answers. Its caller span starts before request admission and `sdk_operation` is the nested SDK boundary. Replay nodes and tokens remain isolated under `diagnostic`. |
| `ask_retrieval_core` | The complete retrieval prerequisite inside `Memory.ask`: content preparation, temporal/scope handling, query speech, embedding, index lookup, and ranking. |
| `answer` | Caller end-to-end completion latency and TTFT, plus nested SDK operation latency, generation TTFT, generation first-chunk time, and throughput. |
| `asr` | Audio duration, inference duration, call success/error counts, standard real-time factor, speedup, and latency distribution. |
| `capture` | Capture-acknowledgement latency from `mindbridge.capture`; empty unless `--ingest capture` runs. |
| `time_to_searchable_ms` | Elapsed time from capture commit to the `settle()` call that made the record searchable; empty unless `--ingest capture` runs. |
| `formation` | `settle()` latency for model-dependent enrichment; empty unless `--ingest capture` runs. |
| `compile` | `Memory.compile` latency and the compiled bundle's character, item, and media-item distributions; empty unless the `compile` arm runs. |
| `recall` | Recall-planning activation: `plan_count`, `shapes` (plans counted by shape), `fallback_count`, `replan_count`, `incomplete_count`, `non_selective_steps`, and the `exhaustive_rows` distribution, plus the stage's own latency; `plan_count` is `0` unless the run sets `recall_planning`. |
| `nodes` | Count, compute time, active time, throughput, average, p50/p95/p99, status, parent operation, purpose, model identity, response identity, fingerprint, embedding task, batch size, and modalities for every operation, stage, and model span. |
| `token_usage` | Total and per-module request counts, exactness, input/output/cached/reasoning tokens, modality totals, per-call distribution, and observed output tokens per model-compute second. |

`ingest` ends only after the SQLite commit and the Zvec apply, so an accepted item is searchable
when its measured call completes; the Zvec flush and outbox acknowledgement are batched behind the
call and are not part of its latency. Failed attempts stay in attempt
latency and error counts but never enter accepted-item throughput.

The three first-output clocks are intentionally distinct:

- `answer.end_to_end_time_to_first_token_ms` starts when the benchmark caller launches the answer,
  before request-concurrency admission, and stops at the first non-empty text delta;
- `answer.generation_time_to_first_token_ms` is the generation model's request-to-first-token
  clock;
- `answer.generation_time_to_first_chunk_ms` includes an empty first provider chunk when one is
  emitted.

The caller clock is `null` when no non-empty delta was observed. `answer.sdk_operation` exposes the
nested `mindbridge.ask` service boundary. `AsyncMemory` includes executor queueing in its operation
TTFT; the outer caller metric additionally includes time spent waiting for the harness request
semaphore. Per-sample `latency_ms` starts after admission and therefore remains response latency.

`capture`, `time_to_searchable_ms`, and `formation` name what [Context OS](context-os.md#fast-context-plane)
calls the fast plane's acknowledge-then-enrich contract: `capture()` acknowledges after the SQLite
commit, so its own span *is* capture-acknowledgement latency; `time_to_searchable_ms` is the delay
until a `settle()` call actually made a record searchable; `formation` is the model-dependent work
one `settle()` call paid, distinct from any scheduling delay before it ran. All three report a
`count` of `0` under the default `--ingest add`, because nothing calls `capture()`/`settle()` on
that path -- that is the harness telling the truth about what it measured, not a bug. Speculative
first-hit latency (the fifth quantity Context OS names) has no dedicated span today: the streaming
prefetch path (`AsyncOmniPrefetch`) issues an ordinary `Memory.search()`, indistinguishable in
telemetry from any other `search()` call, and this harness does not exercise streaming ingest at
all, so it is not measurable by `eval` yet.

`recall` is how a run proves the planner ran. Every planning failure resolves to the fallback
plan on purpose -- no planning capability, a planner error, a plan the kernel will not run -- so
scores alone cannot separate "planning is off" from "planning ran and decided nothing". A
`plan_count` equal to the question count with a `fallback_count` of `0` is an active planner; a
`plan_count` of `0` under `recall_planning` means the answerer never declared
`RecallPlanningBackend`. `incomplete_count` counts the plans whose exhaustive reads filled their
row bound, which is when a set answer may not state a total. `non_selective_steps` counts the
reads that matched too much of the corpus to enumerate and so contributed no rows, which is the
other way a plan fails to add anything -- and the one that looks like an active planner in every
other counter. Each sample additionally carries
`recall_shape`, the shape of the plan its own answer was grounded on.

`compile`'s bundle-size attributes are harness-owned (`compile()` itself carries no such attribute)
so that adding this measurement never touched product code. Pair `compile_bundle_chars` and
`compile_bundle_items` (in a sample's `metrics`) with that sample's answer-quality metric to compute
"useful evidence per token" for one question; the run does not compute the ratio itself.

`search_e2e` and `ask_retrieval_core` must not be conflated. The former is a standalone,
post-answer warm-store replay of each fresh product question through public `Memory.search`; all
selected tasks finish their formal answers before this run-global second pass begins. It is the
metric used by `retrieval_e2e_latency_p95` performance budgets. It does not claim to be the
retrieval performed by the answer. `ask_retrieval_core` measures that real in-answer retrieval
path. The old `performance.search` field remains only as a deprecated alias of
`ask_retrieval_core`.

`search_e2e.planned_count` is the number of eligible fresh product questions, including requests
whose persisted unit could not be reopened. `attempt_count`, `success_count`, and `error_count`
make that accounting explicit. `complete` is false when any planned replay lacks a successful
measurement; such a run is `completed_with_errors` and cannot supply a
`retrieval_e2e_latency_p95` regression comparison. Store-open time is setup and never enters a
successful caller-latency observation.

`asr.real_time_factor` is successful transcription compute seconds divided by the audio duration
reported by the same successful calls, so lower is better and values below one are faster than real
time. `asr.realtime_speedup` is the inverse. Both headline ratios are `null` unless every
successful span reports a positive finite duration; `ratio_complete`, `ratio_call_count`, and the
`reported_subset_*` ratios expose partial coverage without presenting it as complete. Failed-call
latency remains visible in `inference_latency_ms` but cannot contaminate either ratio. Local FunASR
uses the complete input audio-stream duration, including silence and tail, and counts the input
again when an invalid batch reply triggers per-asset fallback inference. `invocation_count` counts
transcription spans while `request_count` includes those internal fallback model calls;
`call_count` remains a deprecated alias of `invocation_count`.

Token totals are never estimated. `complete` covers total tokens, while input, output, cached
input, and reasoning output each have their own `*_complete` flag. An unavailable component is
`null`, not zero; its `reported_*` field is only the known lower bound. An incomplete per-call
distribution has a separately named `retained_average`; its exact all-call average is null when a
logical call omitted usage or one span aggregated multiple requests without per-request totals.
Post-answer replay model usage is reported under
`diagnostic.token_usage`; judge usage stays visible in the top-level total and its own module.

The run-level `resources` object has two attribution scopes:

- client CPU uses process CPU time and the current CPU affinity; memory is the process-lifetime
  resident high-water mark; storage growth is measured over the selected run directories;
- local GPU utilization, memory, and power are sampled from `nvidia-smi`. Averages use
  time-weighted integration; peaks between polls may be missed. These are system-device readings,
  not per-process values, and are marked non-exclusive.

When `benchmark.server_metrics` supplies `/metrics` URLs, `resources.model_servers` records
whitelisted vLLM counter and histogram deltas plus start/end gauges. Its window begins after the
embedding warmup, includes product answers and the post-answer public-search replay, and ends before
judging.
Those metrics are process-global and explicitly warn that shared traffic may be included. If an
endpoint is not configured or cannot be read, the artifact says `unavailable` rather than
inventing remote GPU or server utilization. `storage.media_share` remains the fraction of storage
growth due to source media.

## Mandatory controls

A score is not interpretable on its own, so `results.jsonl` reports three controls per task in a
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
quietly omitted. `performance.token_usage` also carries a `product` block: the tokens MindBridge
itself spent (embedding, generation, transcription, description), summed only over modules whose
usage is complete. The console table prints those with a trailing `*` when the run total is null
because the judge omitted usage on some requests; the cost axis needs the product number and it is
measured. Recall scores the ranked candidate list that `Memory.ask` already produced before
grounding, never the narrower evidence the generator saw and never a second search. That list is
captured against the same causal store state as the answer. Its configured depth is 100 when
`evidence_budget_chars` is set and otherwise `min(100, recall_limit * 3)`; both the configured depth
and the returned count are recorded. The random arm requests 100 for gold-labelled questions and
`recall_limit` otherwise. New response-cache entries preserve the ranked IDs; a legacy cache entry
without one is counted in
`retrieval.unranked_labelled_question_count` instead of being scored as zero.

## Gold evidence per benchmark family

A gold evidence label is a set of memory **source IDs** — the `source_id` an adapter gives each
stored memory, which comes back on every retrieved hit. Only a family that can name those IDs can
have `recall_at_1`, `recall_at_20`, or a random-ranker control at all; the rest print `MISSING`,
which is the honest state and not a defect to paper over.

| Family | What the release publishes | Verdict |
| --- | --- | --- |
| `locomo-refined` | `qa[].evidence`, a list of `dia_id` values | Exact: `dia_id` is the stored source ID |
| `longmemeval` | `has_answer` on the answering turn, and the coarser `answer_session_ids` | Exact at turn level |
| `es-memeval` | QA `evidence` turn IDs; event IDs are scorer-only annotations | Exact at the published session retrieval granularity: a session is gold when it contains at least one labelled visible turn |
| `atm-bench` | `evidence_ids` naming emails and media records | Exact |
| `mem-gallery` | `clue_ids`, the clue round IDs | Exact |
| `worldmemarena` | Gold `image_id` and scorer-authored `memory_id` points | Exact for image IDs; `mp_*` memory points are unresolved rather than leaked into memory |
| `mm-lifelong` | `total_intervals`, and `clue_intervals[].video_id` on the week and month splits | Interval-level only, and already reported as the official `ref_at_300`. The clue video IDs cannot be joined: prepared clips are keyed by file stem, not by release video ID |
| `supermemory-vqa` | `question_evidence.time_spans[].video_id`, kept as `source_video_ids` | Source-video level only. The join exists — `prepare_media` writes `<video_id>-video-#####` — but scoring it needs a group recall ("any clip of each gold video"), a different operator from the exact set recall above. Not implemented |
| `m3-bench` | `timestamp` and `before_clip` | Not derivable: both say when the question is asked, not where the answer is |
| `memlens` | Nothing beyond the answer | Not derivable |
| `clbench` | `context_id`, which names the whole unit | Not derivable: a label equal to the unit cannot separate rankers |
| `beam` | Rubrics and reference answers; `turns[].id` is a turn's own index and no question refers to one | Not derivable |
| `personamem-v3` | Slate-internal `_origin` and `_held_out_persona_item` | Unresolved. Those fields are deliberately excluded from the rendered slate because they are the answer; whether `_origin` names a `source_object_id` that matches an `event_id` needs a check against the corpus |
| `egotempo` | One clip per unit | Degenerate: a one-candidate pool cannot separate rankers |
| `openeqa` | `episode_history`, which is the unit | Not derivable |
| `video-mme-v2` | Nothing beyond the answer | Not derivable |

So exact retrieval quality is measurable on five families in the catalog, plus the image-labelled
subset of WorldMemArena. Its `recall_at_20` remains a diagnostic for those compatible source-ID
labels, not a universal benchmark metric.

The two exact labels wired here are joined differently, because the risk differs. LongMemEval marks
the answering turn as the memories are built, so its label is exact by construction; a turn over
the part limit becomes several opaque, source-ordered blocks and every block of a marked turn is
gold.
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

- `config.yaml`: a resolved comparison manifest containing the effective product, judge, download,
  server-observation, and run settings after file, environment, and command-line precedence. API
  credentials and unconstrained provider-specific `extra_body` values are omitted. This generated
  manifest describes the completed run; it is not intended to be passed back to `--config` as an
  input document.
- `samples.jsonl`: one prediction and its native metrics, evidence intervals, retrieval diagnostics,
  `ranked_source_ids_complete`, and structured failure fields per sample, per arm.
- `results.jsonl`: one aggregate record with dataset and implementation pins, arm definitions,
  aggregate metrics, confidence intervals, performance, token usage, abstentions, mandatory
  controls, the noise floor, resource usage, and a digest of `samples.jsonl`.

A run in progress also holds `samples.partial.jsonl`, appended as each task finishes answering and
removed when the real artifacts land. It is a crash copy, not an artifact: it carries no results
document and no digest, and a leftover file means the run it belongs to did not finish. Read it to
recover the answers of the tasks that completed before an interruption.

The evaluator treats embedding HTTP 401, 403, 404, 405, 408, 429, 5xx, and connection or timeout
failures as a shared-service outage. It stops recursive ingest isolation after the first such
failure, and a query-time failure marked with the `embed` stage stops queued questions in that
unit. Answers already completed remain in the result; the failed and remaining questions retain
their planned rows as structured errors. Input-specific HTTP 400, 413, 415, and 422 failures still
use item isolation, and a generation-stage provider failure remains an ordinary per-question
error.

Before any of that classification, a connection, timeout, rate-limit, or HTTP 5xx failure from the
embedding, generation, or judge endpoint is retried with exponential backoff (capped at 30 seconds
between attempts) for up to ten minutes. Such a failure describes the network at that second, not
the answer, write, or verdict it interrupted, and one fourteen-hour run lost its `longmemeval-s`
score to eight connection resets during judging. Only the attempt that succeeded is timed. An
outage longer than the budget still produces the structured errors described above and still
invalidates the score, so a dead endpoint is never hidden.

The standard CLI does not persist each question inside one unfinished task. The frozen long-run
EgoLife protocol adds that narrower behavior with an attempt-owned benchmark source overlay; it
does not change `mindbridge-bench eval`. Its private `samples.generation.journal.jsonl` writes and
fsyncs one checksum-protected record after each completed question. A record contains the composite
attempt, run, task, unit, question, and arm identity; raw prediction or structured error; the
canonical expected choice; released day, question type, and audio-needed strata; and cumulative
usage through that row. Missing questions have no journal row. The journal is never read by the
runner and never serves as an answer or judge cache.

Offline recovery accepts complete, checksum-valid records and may ignore one unterminated final
line. It rejects earlier corruption and every duplicate composite identity, including identical
duplicates. A finalized sample supersedes its matching journal row only when the raw prediction and
structured error fields agree; a conflict invalidates recovery. Scoring retains the frozen roster's
full denominator, assigning zero to missing and error outcomes. The journal can recover completed
quality rows and cumulative usage, while provider work still in flight at process termination is
outside the last durable usage snapshot.

### Reporting cadence

A multi-task run answers its tasks in order. By default, each task is judged and its interim table is
printed before the next task starts, so a six-task run shows its first score after the first task
instead of waiting for all six inference phases. The interim tables use the same arithmetic as the
final document, minus the standalone search replay, which still runs once at the end, and every
answer is judged exactly once.

`--no-stream-results` (or `benchmark.run.stream_results: false`) restores the former run-global
judging pass when a protocol requires every inference phase to finish before any judging begins. It
also defers every task table until the end. Under the default cadence, client resource sampling and
optional model-server counter deltas split around each judge pass, so judge work is excluded from
the product measurement window. `--stream-results` remains accepted as an explicit spelling of the
default.

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
- **Two benchmarks ask the product for a committed answer rather than a refusal.**
  `mindbridge.benchmarks.prompts.task_answer_policy` maps each task to the `answer_policy` the
  runner passes into `Memory.ask`, and `m3-bench-robot` and the four `mm-lifelong-*` splits are
  `best_effort`. This is protocol alignment on the request side, not a scorer change -- their
  official evaluations give no credit for "unknown" (both are judged against a reference answer
  with no abstention class; MM-Lifelong's judge grades semantic similarity on 0--5) and they ship
  no genuinely unanswerable item, so an abstention there is a lost point rather than a correct
  report. Every other task keeps the product default `strict`,
  because abstaining is part of what they measure: LongMemEval and MEMLENS carry abstention
  abilities, ATM-Bench scores abstention as a class, LoCoMo's category 5 is adversarial, and
  Mem-Gallery's `AR` mandates its own refusal wording. Under `best_effort` the answer is still
  reported with `abstained` set when the evidence was thin, so `abstentions` still counts it; only
  the prediction changes from a refusal to the answerer's best guess. Each `results.jsonl` task
  row and each `samples.jsonl` sample row records the `answer_policy` its product arm requested
  (`null` for the baseline arms, which do not call `ask`), so a `best_effort` run is not
  silently comparable with an earlier `strict` run of the same task. `--answer-policy`, or
  `benchmark.run.answer_policy`, overrides the whole table for one run when the policy itself is
  what is being measured; the resolved `config.yaml` records it as `answer_policy_override`.
- **A task whose query prompt mandates a format or a refusal wording puts that wording into
  retrieval, not only into generation.** `EvalQuestion.content` is what the runner passes to
  `Memory.ask`, and `ask` takes one content input for both legs, so the instruction is matched
  against memory alongside the question. MEMLENS is the extreme case: every one of its queries
  carries `Answer with the exact amount (e.g., "$45.00") only.` and `answer exactly "Insufficient
  information".`, so a retrieval rule keyed on literals the asker named -- numbers, dates, quoted
  spans -- engages on 100 % of its questions on the strength of the template rather than the
  question. The bare question is preserved as `EvalQuestion.source_question` and is what the
  scorers and judges read, but no public surface routes it to retrieval while keeping the template
  for generation. Read any retrieval-side result on a templated task as measuring the template
  too, the same way the `[source_id: ...]` marker the runner prefixes to every stored text
  memory is part of what the full-text index sees.
- **`retrieval_*` scores the retriever's ranked candidate list, not the answer's evidence.** The
  runner observes the list already produced inside `Memory.ask`, before grounding and generation;
  it issues no scoring search. The artifact records that answer's configured candidate depth, and
  `retrieval_candidates` records the actual returned count for each sample. A miss there is a
  retrieval failure. What the answer actually grounded on is separate:
  `evidence` is the answer's hits, and `dropped_hits` counts what the answerer's inline context
  budget removed. A gold that is in the candidate list but not in `evidence` is budget loss, not
  retrieval loss. Current response-cache entries retain the ranked candidate IDs; legacy entries
  that predate that field report the retrieval metric as unranked. `ref_at_300` stays a property of
  the answer's evidence.

`performance` is aggregated separately for each task and arm, and each task row carries only the
measurements attributed to its own arm.

Before reporting a number, check these task fields:

| Field | Reporting rule |
| --- | --- |
| `primary_metric` and `score.mean` | Name and value of the headline result |
| `official_metric` | Must be true for an official-protocol claim |
| `score_valid` | Must be true; ingest failures, failed retrieval diagnostics, and unavailable units make it false. An answer or judge failure scores zero, stays in the mean, and is counted in `error_count` |
| `question_count` and `score.cluster_count` | Report both sample size and independent memory units |
| `score.confidence_interval_95` | Present as `null` with fewer than two independent clusters |
| `dataset_sha256`, `evaluation_sha256`, `scorer_protocol`, and `judge_model` | Establish whether two runs are comparable |

Prompts, references, and raw judge responses are retained only with `--log-samples`. Treat that
option as sensitive: benchmark artifacts can contain source content, retrieved evidence, and model
responses.

The runner fixes seeds and generation temperature, records model endpoints and scorer protocols,
and marks whether each metric used the required official judge. `measurement_protocol` records
fresh-store state, actual embedding warmups, repeat index, cache exclusions, and uncontrolled
remote-server state. A provider may still change an unversioned model, so publishable runs should
use immutable model identifiers and report hardware, dataset selection, retrieval limit, and the
measurement protocol.

A quality claim has to identify the dataset and revision, the official split and evaluator, the
input route, the model and runtime revisions, the retrieval settings, the hardware, and the
measured latency and resource cost. `results.jsonl` records each of those:

| Required field | Where it is recorded |
| --- | --- |
| Dataset and revision | `tasks[].source_repository`, `source_revision`, `dataset_path`, `dataset_sha256`, `input_sha256`, `media_source` |
| Official split and evaluator | `tasks[].evaluation_sha256`, `primary_metric`, `official_metric`, `scorer_protocol`, `official_judge_model`, `judge_model_official` |
| Input route | `tasks[].input_modalities` and `performance.token_usage.calls_by_input_modality` |
| Model and runtime revisions | `model.*`, `environment.mindbridge_version`, `zvec_version`, `runtime_versions`, `python_version`, `platform` |
| Retrieval settings | `recall_limit`, `tasks[].retrieval.ranked_candidate_limit`, and the full `model.memory_config` dump |
| Hardware | `environment.hardware` and the `resources` block |
| Latency and resource cost | `tasks[].performance`, `tasks[].answer_latency_ms`, and `resources` |
| Replay inputs | `run_id`, `seed`, `seeds`, `bootstrap_samples`, `repeat_index`, `measurement_protocol`, `limit`, `offset`, `batch_size`, `blind`, `blind_baseline`, `arms.ingest`, `arms.definitions.compile` |

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
mindbridge-bench eval \
  --tasks locomo-refined \
  --compare .benchmarks/results/baseline \
  --fail-on-regression \
  --regression-threshold 0.01
```

The comparison rejects incompatible dataset, scorer, or judge identities. Any answer or ingest
failure also makes `--fail-on-regression` exit nonzero.

Performance budgets use the same `--compare` artifact but apply only after stricter comparability
checks: schema and runner, task/input digests, models and full memory composition, concurrency,
warmup protocol, runtime versions, acceleration runtime, and hardware must match; the candidate
must carry a product row for every task the baseline measured, so a narrowed `--tasks` selection
is rejected instead of leaving the dropped task's budget unevaluated; and neither result may use a
response cache or contain a product or retrieval-diagnostic error.

```bash
mindbridge-bench eval \
  --config docs/examples/baselines/rtx5090-qwen38-wemm9b-text.yaml \
  --compare .benchmarks/results/rtx5090-text-r0 \
  --performance-budget answer_e2e_ttft_p95=0.10 \
  --performance-budget answer_e2e_latency_p95=0.10 \
  --performance-budget retrieval_e2e_latency_p95=0.15 \
  --performance-budget tokens_per_call=0.05 \
  --performance-budget answer_throughput=0.10 \
  --fail-on-regression \
  --run-id rtx5090-text-candidate \
  --output-path .benchmarks/results/rtx5090-text-candidate
```

The fraction is the maximum tolerated relative regression. Lower is better for latency and token
budgets; higher is better for throughput. The rows are written to `performance_comparisons`, and a
breach exits nonzero with `--fail-on-regression`. The same mapping can be stored under
`benchmark.performance_budgets`; it requires `benchmark.run.compare` or `--compare`. A TTFT budget
is rejected if either artifact's TTFT distribution is incomplete.

Use `--use-cache .benchmarks/response-cache` to persist deterministic generation responses across
isolated reruns. The cache is an optimization, not a substitute for the result artifacts.

Everything that decides what a request asked for is part of the cache namespace, so a cached
answer is only ever reused for the same question. That includes `--answer-policy` and the policy
each task resolves to, because an arm that asks for a committed answer must not be handed the
refusal a strict run already cached and report it as its own.

## Resume an interrupted run

A long run that is killed - by an out-of-memory reaper, a lost session, or a Ctrl-C - keeps both
halves of its work when it is started with a fixed `--run-id` and a response cache:

```bash
mindbridge-bench eval \
  --tasks locomo-refined \
  --run-id locomo-sweep-01 \
  --use-cache .benchmarks/response-cache \
  --resume
```

`--use-cache` returns the answers and judge scores already produced; `--resume` returns the stores
already ingested. Rerun the identical command after an interruption. The two recover different
work and are independent, but a resumed run without the cache still re-answers every question.

`--resume` requires the `--run-id` of the run it continues, because a generated identifier names a
directory no earlier run wrote to. Checkpoints are written on every run, so the interrupted run
does not need to have been started with `--resume`. The crash copy that interrupted run left
behind is the one artifact `--resume` may find in the output directory without `--overwrite`;
`results.jsonl` and `samples.jsonl` still refuse, because a run that wrote them finished.

Each unit's store is reused only when the checkpoint beside it still describes the run being
started. The checkpoint names the task and dataset revision, the embedding, transcription, and
memory configuration, the device, the ingest mode, and whether the run consolidates with
`--deliberate`; anything else rebuilds that unit from zero. A store that was ingested past the
earliest cutoff still holding unanswered questions is also rebuilt, because reusing it would
answer those questions with memories they must not have seen yet. So is a unit whose store directory was emptied or deleted while its checkpoint stayed
behind. Rebuilding empties that unit directory and zeroes its checkpoint together, and it first
opens the store it is about to destroy, so a unit another run still owns fails with the usual
in-use error instead of being deleted underneath it.

The checkpoint is written after each committed batch, never before, so an interruption inside a
batch costs at most one duplicated batch rather than silently dropped evidence. Write failures
recorded before the interruption are carried into the resumed run's `unwritten` column.

A resumed run is not a performance baseline: it skips ingest another invocation paid for and
inherits that run's warm description cache. The result document records `resume`, and comparing
either side of a `--compare` pair that carries it is refused.

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

It writes the requested JSONL and a sibling `.manifest.jsonl` containing one manifest record with
dataset and prediction digests, model identities, counts, platform details, and relative
isolated-store paths. Existing artifacts are protected unless `--overwrite` is passed; reuse of an
existing run requires `--resume`.

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

The evaluation runner allocates one physical store per independent unit, atomically. Each path
component is the label base32-encoded, so no dataset identifier reaches the filesystem verbatim:

```text
.benchmarks/data/
└── benchmark-<encoded-task>/
    └── run-<encoded-run-id>/
        ├── unit-<encoded-unit-a>/  # SQLite, Zvec, and lock
        └── unit-<encoded-unit-b>/  # a different physical store
```

Harness labels belong to the filesystem, not the product API. Do not pass them into `Memory.add`,
store them as hidden product fields, or treat metadata as an isolation boundary. Distinct unit
directories may run concurrently; the same directory has one live owner. Evaluation CUDA runs also
take a per-device process lock unless an external scheduler owns admission control.

Custom behavior benchmarks must use only the public SDK: create a directory, construct `Memory`,
ingest through `add` or `add_many`, query through `search` or `ask`, score public return values, and
close the instance before archiving artifacts. The local-index command is the only documented
direct-adapter exception.

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
