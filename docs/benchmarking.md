# Benchmarking

MindBridge ships a reproducible evaluation harness under `mindbridge-bench`. This page covers how
to run it and — more importantly — what its output does and does not license you to claim.

## The stance

Two rules govern every number produced here.

**Benchmarks run through the production API.** There is no evaluation-only path. A harness that
bypasses the product measures something the product does not do, and every optimisation it
motivates is aimed at the wrong target. This costs iteration speed and weakens single-change
attribution; both are accepted.

**Released-text memory-layer evaluation and raw audiovisual reproduction are separate claims.**
A score obtained by feeding a benchmark's released transcripts through the memory layer says
something real about the memory layer. It says nothing about multimodal capability, and it may
not be presented as a multimodal result.

A number may be quoted as a benchmark score only when all four hold:

1. An official complete split, not a subset.
2. A pinned deployment snapshot of the code and models that answered.
3. A published run manifest committed alongside the results.
4. Replayable outputs.

Subset and diagnostic runs are legitimate engineering instruments. They are not scores. Comparison
targets are in [benchmarks-sota.md](benchmarks-sota.md).

## Current baseline status

**No complete public-set baseline currently stands.**

The four full public-benchmark baselines from 2026-08-13 were retired and deleted on 2026-08-19.
The reasons are worth recording because they are ordinary and recurrent:

- The LoCoMo numbers came from the original corpus, which LoCoMo-Refined has since replaced.
- The M3-Bench Web score mixed 908 v6 shards with 12 v7 shards and included selective reruns.
- Later subset runs overturned two of its stated conclusions — on SuperMemory over-abstention and
  on EgoLifeQA clip-level retrieval hit rate.

`.benchmarks/results/` now retains only diagnostic runs from 2026-08-18 onward, and old manifests
can no longer be verified. **A new baseline must land its manifest alongside its result file to be
citable at all.**

Incremental evidence from the current code counts as functional evidence, not as a score.

## Two traps worth knowing before you interpret anything

These come from diagnostic runs whose result files were retired on 2026-08-19, so the figures
behind them are no longer citable. The methodological lessons stand and are cheap to re-establish.

**Run a blind-answer control.** Answering with the memory system disabled entirely is not a
formality: on EgoLifeQA it scored high enough that one whole question category was substantially
solvable with no memory access at all. A score without that floor cannot distinguish "the memory
worked" from "a large model guessed well". Re-measure the control on any split you report.

**Retrieval hit rate is not the objective.** On the same benchmark, clip-level retrieval hit rate
showed no association with answer correctness. What decided outcomes was landing in the right
hour, not the right clip. Confirm that your retrieval metric actually predicts your answer metric
before optimising against it.

## Running a benchmark

```bash
uv run mindbridge-bench
uv run mindbridge-bench m3 --help
```

The dispatcher exposes twelve official runners: `locomo-refined`, `m3`, `egolife`, `egomem`,
`egotempo`, `memlens`, `mm-lifelong`, `atm`, `mem-gallery`, `supermemory`, `video-mme`, and
`video-mme-v2`. `aml` is a separate offline replay harness; `eval`, `score`, `datasets`, `jina`,
and `bakeoff` support those runs.

Most individual runners need nothing past the core install because they drive the product through
its own API. `video-mme`, `video-mme-v2`, and `datasets` need `--extra benchmarks`; use that extra
for `eval` when it must download releases or prepare media. `jina` and `bakeoff` load the local
embedder and need `--extra cloud-models`. A command whose extra is missing names it and exits
instead of failing part-way through a run.

Runners need a live API and a bearer token in `MINDBRIDGE_API_KEY`. Every generated tenant ID must
be in the deployment's `MINDBRIDGE_TENANT_API_KEYS_JSON` **before** the API starts — one key can
authorize all of them.

Runners write predictions and a manifest to `--output` and print nothing on stdout. Progress goes
to stderr; `-q` silences it.

Long runs are fragile in a specific way: an upstream hiccup mid-stream can escape as an unhandled
error, and results land only after the whole run completes. Shard by `--run-id` and run the shards
in parallel so a failure costs one shard rather than the sweep.

## Running several benchmarks in one command

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run --extra benchmarks mindbridge-bench eval --tasks released-text --limit 2
```

That runs four catalog tasks across three benchmark families — LoCoMo-Refined, MEMLENS 32K, and
both ATM-Bench SGM splits — against one deployment, downloading each official release it does not
already have. Nothing has to be cloned or written first. `--tasks` names entries in a catalog
shipped with MindBridge; `--list-tasks` prints every name it accepts and what obtaining it would
still take. The current `all` group expands to 22 task invocations across the twelve runners:

```bash
uv run mindbridge-bench eval --list-tasks
```

```text
groups (--tasks expands these), inputs resolved against .benchmarks:
  released-text           locomo-refined, memlens-32k, atm-main-sgm, atm-hard-sgm
  aml                     aml-locomo-refined, aml-longmemeval-s, aml-clbench, ...
  all                     locomo-refined, m3-robot, m3-web, egolife, ...

tasks:
  locomo-refined          locomo-refined  download
  m3-robot                m3              download, prepare
  m3-web                  m3              download, acquire, prepare
  egolife                 egolife         download, prepare
  memlens-32k             memlens         download
  aml-clbench             aml             download
  aml-beam                aml             needs .benchmarks/beam/chats
  ...
```

The stages are named in the order the sweep performs them. `ready` runs now. `download` is a fetch
of the pinned official release the sweep performs itself. `acquire` is media that comes from
outside the release — Ego4D behind its signed agreement, or 920 YouTube URLs — and is the one
stage with a prerequisite this machine may not hold; see [Media no snapshot
supplies](#media-no-snapshot-supplies). `prepare` means the sweep will stage media into the
deployment's bucket. A named path is a prepared-media manifest with no producer yet and must be
supplied by the operator, which no catalog task now needs.

The listing reads the tables and the filesystem and nothing else: it never downloads, never
imports an acquirer, and never checks a credential, so it stays instant and safe to run before
deciding to spend an evening.

Task names are comma-separated and the flag repeats, so `--tasks m3-robot,m3-web --tasks egolife`
is three tasks. A group expands to several, and a task named twice still runs once.

Everything else has a default worth having. Only the task names do not:

| Flag | Default |
| --- | --- |
| `--run-id` | `sweep-<UTC timestamp>` — unique, so it isolates tenants and never meets its own output |
| `--benchmarks-root` | `.benchmarks`, the layout the download commands above create |
| `--output-dir` | `<benchmarks-root>/results/<run-id>` |
| `--api-base-url` | `http://localhost:8000` |
| `--deployment-config` | `<benchmarks-root>/deployment.json` |

Name `--run-id` yourself when the run has to be found again by name rather than by date.

The individual runners still require the URL and the deployment file explicitly. They are single
measurements; this command is the one you reach for while iterating.

### What the catalog holds

One task per required choice. A runner that forces you to pick — ATM-Bench's `--split`, MEMLENS's
`--context-window`, M3-Bench's `--subset`, Video-MME's `--transcript-source` — gets one entry per
value, which is why MEMLENS appears four times and LoCoMo-Refined once. Optional filters are not
enumerated: a run scoped to Video-MME's `long` band or three Mem-Gallery topics is a task of your
own, and `--limit` already covers a smoke run.

### How releases are obtained

A sweep downloads every file its tasks read and does not already have, then holds each one to a
digest this repository committed. Three things about that are deliberate.

**Only the files a task reads.** ATM-Bench's Hub repository is 3.2 GB and Mem-Gallery's is
530 MB, but a run consumes five JSON files and one directory of them — about 40 MB between them,
against 302 GB of full releases. Media is fetched separately and only for a task that stages it,
narrowed to the units `--limit` selects where the release allows narrowing — a `--limit 1` EgoLife
run takes one day rather than the release's 477 GiB. MEMLENS is the exception worth planning disk
for among the annotations: its annotation *is* the corpus, so its four context windows are 98 MB,
191 MB, 369 MB and 732 MB, and `--tasks all` fetches about 1.4 GB rather than 40 MB.

**Pinned.** Each release is fetched at a fixed commit — every one of them, asserted by
`tests/unit/benchmarks/test_releases.py` rather than left to review, because a branch name makes
one task name mean different bytes on different days and nothing in the run would say so.

**Verified where a digest names it.** Most annotations have a `source_sha256` in
[benchmarks/manifests/dataset-adapters-smoke.json](../benchmarks/manifests/dataset-adapters-smoke.json),
recorded when `mindbridge-bench datasets` last ran. A download whose bytes differ stops the run,
names both digests, and **deletes the file** — left in place it would be verified once and then
skipped as already-present by every later sweep. Re-run that smoke and record the new digest
before measuring against it. The releases the smoke manifest keys by something other than a file
name have no digest here and rest on the pin alone; `mindbridge-bench datasets` is what checks a
corpus already on disk.

Downloads are skipped for files already present, so a second sweep fetches nothing.
`--benchmarks-root` chooses where they land — it defaults to `.benchmarks`, and the layout is the
same one the manual commands in "Benchmark dataset smoke" below produce, so an existing corpus is
found as-is. `--no-download` refuses to fetch and fails on an absent release instead — before the
first task starts, not when its turn comes. A missing manual prepared-media manifest is reported
but does not prevent unrelated ready tasks from running; producer-backed tasks prepare their own
run-specific manifest.

### Prepared media

Media benchmarks read a prepared-media manifest: a JSON file naming clips already in object
storage, with their durations and the identity spans over them. A sweep produces that manifest
itself for the benchmarks below, staging into the deployment's own bucket:

| Benchmark | Produced from | Media obtained by |
| --- | --- | --- |
| `mem-gallery` | the release's own images | the sweep |
| `m3-robot` | the official videos, cut into 30-second clips | the sweep |
| `m3-web` | the official videos, cut into 30-second clips | the sweep — see below |
| `video-mme`, `video-mme-v2` | the official videos, cut into 30-second segments | the sweep |
| `egotempo` | the pre-trimmed clip each question names | the sweep — see below |
| `egolife` | the release's own 30-second clips, staged verbatim | the sweep |
| `egomem` | EgoLife's clips, on EgoMemReason's own horizons | the sweep |
| `supermemory` | the official recordings, cut to each question's horizon | the sweep |
| `mm-lifelong` | the official videos on the split-wide clock | the sweep |
| `atm-main`, `atm-hard` | the release's own image and video archive | the sweep |

ATM-Bench's `sgm` arms are absent deliberately: they ingest the release's pre-processed captions
and open no manifest, so nothing is staged for them. That is declared rather than inferred from
the flag being absent — absence is what makes the sweep *add* the flag, which would have staged
about 3 GB neither arm reads.

EgoLife's clips are staged as the release published them rather than re-cut. They are already the
30-second split both shapes want, and they carry an AAC track that a re-encode is the documented
way to lose. SuperMemory-VQA's recordings carry no audio at all — that is the public release, not
the clipper.

Two properties are forced rather than chosen, and they are worth knowing before reading a
manifest.

**A manifest belongs to one run.** A media URI is accepted only under
`tenants/<tenant_id>/`, and a benchmark tenant is `<tenant-prefix>_<unit>_<run-id>`. The same
clips staged under another `--run-id` are unreadable, so preparation happens inside the run
rather than once beside the corpus. `--limit` is what keeps that affordable: it bounds the units
prepared as well as the units answered.

**The selection is the runner's own.** Each producer parses the task's arguments with the
runner's parser and selects with the runner's helper, so it cannot prepare a unit the run will
skip or miss one the run will read.

Clips are cut by the same encoder the product stores evidence with, audio track included, so
what a benchmark ingests is what the product would have produced. Preparation is skipped when the
manifest already exists, and a preparation that fails is that task's failure — the sweep reports
it and runs the remaining tasks.

Bucket credentials are Boto3's own (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`); the bucket and
endpoint come from the same `[object_storage]` configuration the deployment reads, so a benchmark
cannot stage into a bucket the deployment will not look in.

### Media no snapshot supplies

Naming a task fetches the media it reads, at the revision `releases.py` pins, with no size cap and
no prompt — Video-MME's 20 archives are 94 GiB and the sweep will take all of them. Two media sets
are not files in a pinned repository at all, so the sweep obtains them a different way. It still
obtains them: `--list-tasks` marks these two `acquire` rather than `download`, and the difference
is the prerequisite, not the automation.

- **`egotempo`** — its videos are Ego4D, released under a signed access agreement no unattended
  download can accept, so **you need Ego4D access before this can run**. Request it at
  <https://ego4d-data.org>, then let the sweep fetch each question's `source_video_id` with the
  `ego4d` CLI and cut `clip_start_seconds..clip_end_seconds` into
  `<benchmarks-root>/egotempo/videos/<clip_id>.mp4`. The name is the `clip_id` because the runner
  requires the prepared `video_id` to equal it. Without the agreement the run stops with those
  same instructions, which is also what to follow to place the clips by hand.
- **`m3-web`** — its 920 videos are web sources the release distributes as the `video_url` of each
  entry in `m3-agent/data/annotations/web.json` rather than as files, so the sweep downloads them
  with `yt-dlp` into `<benchmarks-root>/m3-bench/videos/web/<video_id>.mp4`. Each annotation key is
  the YouTube ID, so the file name needs no template — though many of those IDs begin with a
  hyphen, so inspecting the corpus by hand needs `./-4OBcRHX1Bc.mp4` or a `--` first. **Budget
  roughly 90 GB and about 20 hours of wall clock for the whole split**; a `--limit`ed run asks for
  only the videos it will read.

Both are narrowed per unit, so `--tasks m3-web --limit 1` acquires one video rather than 920.
`MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS` (default `30`) is the wait before each download, jittered
up to twice that, with metadata requests paced at a quarter of it. Raising it is the only lever
against YouTube's bot detection, and a value that is not a number is refused rather than defaulted.

**Those figures are measured, and estimating them was 30x wrong.** The constraint is YouTube's
per-connection bandwidth, about 0.85 MB/s, not the pacing: the sleeps are 1–2% of the total, so
**raising the pacing is nearly free and lowering it buys nearly nothing.** Treat that variable as
insurance against being cut off, not as a throughput dial. And do not size this from `yt-dlp`'s
`filesize_approx` — it is bitrate times duration, meaningless for the fragmented DASH formats
YouTube serves, and it read 26x low on a video that arrived as 712 MB.

**`--format-sort res:360` is why 90 GB and not 550 GB, and it is not a quality compromise.** The
release's sources are 1920x1080, and taking the best available measured 596 MB and 496 seconds per
video across three real downloads — about 550 GB and five days for the split. But every stored clip
is capped at `DEFAULT_VIDEO_MAX_PIXELS`, **200,704 pixels**: a staged clip measures 596x336, so
roughly 94% of a 1080p download is discarded before anything reaches the bucket. 640x360 is 230,400
pixels, still above that ceiling, so a 360p download and a 1080p one are scaled to the same
596x336 clip while the download is about six times smaller. Not byte-identical — the scaler is fed
a different source, so clip digests differ across the change — but the same clip at the same size,
which is all the encoder and the model ever see. Do not "fix" the sort key back to best-available:
it would cost five days and change no dimension of what gets staged.

**That is a coupling, and it only holds while `DEFAULT_VIDEO_MAX_PIXELS` stays at or below
230,400.** Raise the ceiling above it and 360p becomes the binding limit instead: clips would be
upscaled from a source that no longer carries the detail, and the repair is re-downloading the
corpus. `tests/unit/benchmarks/test_acquire_youtube.py` reads that constant and fails if it rises,
because nothing else would surface it — but if you are the one raising it, the corpus is what you
have to plan for.

**A dead video does not report the same way twice.** One removed entry answered `Private video.
Sign in if you've been granted access` on two attempts and `Please sign in.` on two more, minutes
apart, for the same URL — so a run must recognise a login wall by more than one wording or it
mistakes the second phrasing for the address itself being blocked, and stops on videos it should
walk past.

**An out-of-date `yt-dlp` presents as HTTP 403 from YouTube, and 412 from bilibili.** That looks
exactly like bot detection and cannot be fixed by pacing, which is why the acquirer prefers
`uvx yt-dlp@latest` and why its failure message distinguishes the two by whether anything
succeeded first — a stale binary fails on the very first video. Recent versions also warn that no
JavaScript runtime was found and that some formats may be missing; installing `deno` silences it.

`--no-download` refuses the annotation fetch, the media fetch, and the acquisition rather than
performing any of them, so it fails on an absent release instead of quietly obtaining 94 GiB. For
the two sets above it names both the flag and the prerequisite, because only one of those is
knowable without going and looking: the flag is certain, and whether you hold an Ego4D signature
is not.

M3-Bench's released memory graphs would give a caption-only manifest with no video at all, which
is the cheaper released-text arm. They are distributed as Python pickles, and unpickling a
downloaded file executes whatever it contains, so that path stays a deliberate manual step.

### Smoke runs

`--limit N` runs the first N of each benchmark's own units — conversations for LoCoMo-Refined,
videos for M3-Bench, topics for Mem-Gallery. It is deliberately not a count of questions: what
one of these runs costs is dominated by ingesting the unit, so limiting the questions inside a
unit that was ingested anyway saves almost nothing. On LoCoMo-Refined, `--limit 2` is 2 of 10
conversations and 210 of 1,382 questions. It composes with a benchmark's own ID flags rather
than competing with them — narrow first, then truncate — and every runner accepts it on its own,
not only through a sweep.

`--tasks all --limit 1` exercises the whole harness. It is a smoke run, not an evaluation.

### What the sweep owns

**`--output` and `--run-id` are derived per task, and a suite setting either is refused.** Each
task writes into a directory of its own: `predictions.jsonl` for line-oriented runners or
`predictions.json` for array-based runners, with the manifest, prepared media, and score beside
it. This keeps a 22-task `all` sweep in 22 directories instead of one flat artifact pile. The run
ID is not cosmetic. A tenant is `<tenant-prefix>_<unit-id>_<run-id>`, so two parameterisations of
one benchmark — MEMLENS at two context windows, ATM-Bench at both splits — share everything but
the run ID. Without a per-task one the second task would write into the first task's tenants and
then answer from its memories.

Every tenant a run writes to has to be in the deployment's `MINDBRIDGE_TENANT_API_KEYS_JSON`
before the API starts, and these run IDs are ones the sweep makes up. `--dry-run` prints the exact
invocation behind each task, so they are readable before anything runs:

```bash
uv run mindbridge-bench eval --tasks released-text --run-id sweep-001 --dry-run
```

`--api-base-url` and `--deployment-config` are forwarded to every task. So are `--limit`,
`--recall-limit`, `--request-concurrency`, `--unit-concurrency`, and
`--request-timeout-seconds`, but only when given — otherwise each runner keeps the default it
declares rather than a copy pinned here that goes stale. They are placed before a task's own
arguments, so a benchmark needing its own recall budget sets it in the task.

`--device-id`, `--poll-interval-seconds`, and `--processing-timeout-seconds` are forwarded the
same way, but only to the tasks whose runner takes them — every benchmark here except
LoCoMo-Refined, which ingests no media and would reject them. That is what lets one sweep mixing
text and media raise the processing deadline for the runs it applies to. A flag narrower still,
`--prepared-media` among them, belongs in the task rather than on the sweep.

### Keeping the worker busy

Two flags decide how much work a run keeps in flight, and the product of them is the number that
matters:

| Flag | Default | What it bounds |
| --- | --- | --- |
| `--request-concurrency` | 4 | in-flight API requests inside one unit |
| `--unit-concurrency` | 4 | units of the benchmark running at once |

A run holds up to `--unit-concurrency` × `--request-concurrency` requests in flight, so the
defaults keep 16 outstanding against a Worker whose own concurrency is what bounds GPU memory.
Raising these does not make the Worker do more at once; it makes it less likely to find its queue
empty. Set both to 1 for a serial run, which is what to reach for when reading a failure.

`--request-concurrency` alone cannot keep the queue full, and this is worth knowing before tuning
it: a unit ingests before it answers, and the answer phase never reaches the Worker at all. Past
the size of one unit's fan-out the flag stops buying anything, because what is left idle is the
gap between units rather than the room inside one. `--unit-concurrency` is the flag that closes
that gap — with units overlapping, one unit's answer phase runs against another's ingest.

### What it records

Tasks run one at a time. Running them concurrently against one deployment would have them
contending for the same worker and would corrupt every timing the runs report; to use more
hardware, run separate sweeps with different `--run-id`s against separate deployments.

That is a claim about tasks, not about the work inside one. A task's units — a video, a topic, a
conversation — overlap, and nothing a run reports is per unit, so there is no per-unit timing for
them to contaminate. See [Keeping the worker busy](#keeping-the-worker-busy).

A task that fails does not stop the sweep, so a benchmark dying four hours in costs its own result
rather than the others. An interrupt does stop it, and still writes the summary. `--output-dir`
receives one `suite-summary.json` recording, per task, the derived run ID, the output path, the
argv behind it, the exit code, how long it took, and whether it produced predictions at all — a
task that exits 0 without writing them is recorded as failed rather than as a result nobody can
open. The sweep exits 1 if any task failed and 130 if it was interrupted.

### What it prints

When the sweep finishes it writes a results table to stdout — the run's output, so `-q`, which
silences progress on stderr, leaves it alone:

```text
Task            Benchmark       Status     Wall   Metric                         Value  Source
──────────────────────────────────────────────────────────────────────────────────────────────
video-mme       Video-MME       completed  41:02  accuracy                      0.6129  runner
                                                  strict_accuracy               0.6000  runner
                                                  question_count                   900  runner
                                                  by_duration.long.accuracy     0.4900  runner
locomo-refined  LoCoMo-Refined  completed  08:03  llm_judge                     0.5810  judge
egomem          EgoMemReason    completed  18:03  —                                  —  not scored
atm-main-sgm    ATM-Bench       failed     00:03  —                                  —  —

sweep-20260826-141233: 4 tasks · 3 completed · 1 failed · 1:07:08 total
not scored: egomem — these are scored outside MindBridge; attach the result with
  mindbridge-bench score --predictions ... --manifest ... ...
```

The column after Metric is lmms-eval's `↑`/`↓`: which direction is good, declared with the
metric rather than left to the reader. `Source` says who produced the number, and there are four:

| Source | Who | Benchmarks |
| --- | --- | --- |
| `runner` | exact option match, computed locally with no network call | EgoLifeQA, SuperMemory-VQA, Video-MME, Video-MME-v2 |
| `judge` | a judge model, called from inside the run | LoCoMo-Refined, M3-Bench, MEMLENS, ATM-Bench, Mem-Gallery, EgoTempo, MM-Lifelong |
| `official` | an external scorer, attached afterwards by `mindbridge-bench score` | any run that has been scored |
| `bypass` | nothing; `--predict-only` was passed | — |

`not scored` is a completed run with neither an in-run metric nor an official score sidecar. That
is the normal EgoMemReason state because its answers are held out by its leaderboard. A judged run
under `--predict-only` is different: it reports the visible `bypass=999` sentinel.

## Scoring, and what copying lmms-eval costs

Which of those four applies is declared per benchmark in `src/mindbridge/benchmarks/scoring.py`,
the way lmms-eval declares a `metric_list` per task. That table is the whole policy: a runner
names its benchmark and hands over either the numbers it computed or the answers a judge has to
read, and never decides who scores it.

**Free-text benchmarks are judged inside the run.** Seven of them pair a natural-language gold
answer with a natural-language prediction, so correctness is a judgement rather than a
comparison. lmms-eval makes that call in-framework — MM-Vet calls `gpt-4o-2024-11-20` from inside
`process_results` — and so does this. Configure it with:

```bash
export MINDBRIDGE_BENCH_JUDGE_ENDPOINT=https://your-endpoint/v1
export MINDBRIDGE_BENCH_JUDGE_API_KEY=replace-with-a-runtime-secret
export MINDBRIDGE_BENCH_JUDGE_MODEL=gpt-4o-2024-11-20
```

The model defaults to MM-Vet's own so a number here is comparable to one lmms-eval would produce.
The endpoint has no default: a judged benchmark with none configured refuses to start rather than
scoring the whole split zero — checked before the run, and once for the whole sweep, because a
judged run that finishes and then cannot score writes no predictions at all. The key is optional,
since a local judge does not want one. Seventeen of the current 22 catalog tasks use this judge, so
`--tasks all` requires it unless `--predict-only` is set.

**Three consequences, all of them upstream's and all deliberate:**

- **A judge that cannot be read scores the answer `0.0`.** Three attempts, each with a stricter
  instruction, then the floor. So an upstream outage reports as a low benchmark score, not as a
  failed run. The one thing added here is the tally beside it: `scoring.judge_failure_count`
  records how many answers were floored, and the results table prints it under the table as *a
  floor, not a measurement*. Without that count a run whose judge was down is afterwards
  indistinguishable from a run that answered badly.
- **The judge is MindBridge's choice, not the benchmark's.** Two runs under different judges are
  not comparable to each other, and neither is comparable to a leaderboard. `judge_model` is
  recorded in every manifest for exactly this reason. An official number still comes only from
  the benchmark's own scorer through `mindbridge-bench score`, which lands as `official`.
- **`--predict-only` reports `999`.** lmms-eval's `bypass_agg` returns that literal, in the same
  column as a real accuracy, and so does this. It is a sentinel, not a score; the table says so
  underneath. Use it to write predictions with no judge configured at all:

```bash
uv run --extra benchmarks mindbridge-bench eval --tasks released-text --predict-only
```

One departure worth naming: MM-Vet escalates `temperature += 0.5` between attempts, and
`GenerateRequest` here carries no temperature — generation is deliberately deterministic. Retrying
an identical request would return an identical unparseable answer, so the retry escalates the
*instruction* instead. Same patience, same floor.

The sweep summary file itself still carries no scores; every number above lives in the manifest or
the score sidecar of the task that produced it.

Runner and MindBridge-judge metrics are available when a task completes. An official scorer still
runs after the sweep, and `--report` renders the table again from a directory an earlier run wrote,
so an attached official number reaches the screen without running anything twice:

```bash
uv run mindbridge-bench eval --report .benchmarks/results/sweep-20260826-141233
```

It reads the artifacts beside each task's predictions, so a results directory copied off the
machine that produced it renders the same way there.

---

## Agent Memory Leaderboard offline harness

[AML](https://agentmemories.ai/) does not distribute a benchmark you run yourself. It calls two
endpoints you host — `Add` writes a history, `Search` returns ranked evidence — and owns the answer
model, the prompts, the judges, and `top_k`. MindBridge serves that contract at `POST /aml/add` and
`POST /aml/search`, registered only when `MINDBRIDGE_AML_API_KEY` is set, so no existing deployment
grows an AML surface by accident. Set `MINDBRIDGE_AML_TENANT_PREFIX` too; the routes derive each
tenant from `user_id` alone and never accept a caller-supplied tenant.

The offline harness drives those same endpoints locally, then scores with AML's own published
`answer` and `evaluate` stages, vendored verbatim under
[benchmarks/aml/](../benchmarks/aml/PINNED.md) and pinned by sha256. Those files are never edited and
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
unaffected; AML runs ScriptMem server-side against its own copy. `mindbridge-bench aml
--benchmark scriptmem` offers the choice and refuses it with that reason, so it reads as a decision
rather than a typo.

CL-Bench, LongMemEval and PersonaMem-v1 are in the release table, so `mindbridge-bench eval`
fetches them itself and these commands are only for obtaining them out-of-band. BEAM and
PersonaMem-v2 are not, and have to be fetched by hand: BEAM's corpus is 200 files discovered by
glob under `chats/{tier}/{conv}/`, and PersonaMem-v2's `data/` is 3.9 GB of which a run reads one
history variant. Registering either would make a sweep fetch far more than the task reads, or
announce a fetch that writes nothing. `--list-tasks` reports both as `needs <path>`.

Pin every one of them. Without `--revision` the same task name means different bytes on different
days, which is the drift that makes two scores incomparable:

```bash
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git .benchmarks/locomo-refined
git -C .benchmarks/locomo-refined checkout 887091190789e8d6760e70b9edd696539923dc4f

git clone https://github.com/mohammadtavakoli78/BEAM.git .benchmarks/beam
git -C .benchmarks/beam checkout 3e12035532eb85768f1a7cd779832b650c4b2ef9

uvx --from huggingface-hub hf download xiaowu0162/longmemeval --repo-type dataset \
  --revision 2ec2a557f339b6c0369619b1ed5793734cc87533 --local-dir .benchmarks/longmemeval
uvx --from huggingface-hub hf download bowen-upenn/PersonaMem-v1 --repo-type dataset \
  --revision fd7c30f071d5c2ee2a211506783be222d7b6002e --local-dir .benchmarks/personamem-v1
uvx --from huggingface-hub hf download bowen-upenn/PersonaMem-v2 --repo-type dataset \
  --revision 0622e56d1cc6f1bc990a5100a6ec4022a60e66a6 --local-dir .benchmarks/personamem-v2
uvx --from huggingface-hub hf download tencent/CL-bench --repo-type dataset \
  --revision b28a5832a09b0d96c0cf4c22e90d7c60ede25b80 --local-dir .benchmarks/clbench
```

Use `longmemeval_s`. `longmemeval_oracle` ships only each question's gold sessions, so retrieval has
nothing to discriminate against and any score from it is meaningless; `longmemeval_m` is 2.7 GB.

Per-benchmark field mappings, including the places a dataset's key differs from what its pipeline
reads, are recorded in
[the dataset schema reference](superpowers/specs/2026-08-17-aml-dataset-schemas.md).

### Retrieval and scoring

`--dataset` is repeated once per positional argument the benchmark's loader takes, in order:
`locomo-refined`, `longmemeval`, and `clbench` take one path; `beam` takes `chat` then
`questions`;
`personamem-v1` takes `questions_csv` then `contexts_jsonl`; `personamem-v2` takes `benchmark_csv`
then `data_root`.

The replay reads its bearer token from `MINDBRIDGE_AML_API_KEY`, which is the client
credential for `--api-base-url` and is separate from the `MINDBRIDGE_API_KEY` the other
runners use:

```bash
export MINDBRIDGE_AML_API_KEY=...
uv run mindbridge-bench aml \
  --benchmark locomo-refined \
  --dataset .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --output .benchmarks/results/aml-locomo-refined.jsonl \
  --api-base-url "$MINDBRIDGE_API_BASE_URL" \
  --deployment-config .benchmarks/deployment.json \
  --run-id smoke-1 \
  --tenant-prefix bench_aml \
  --recall-limit 100 \
  --request-concurrency 4
```

This runner takes the same flags as every other one -- `--recall-limit`, `--request-concurrency`,
`--request-timeout-seconds`, `--limit`, `--overwrite`, `--predict-only` -- because that shared
vocabulary is what lets `eval` dispatch it. `--predict-only` is accepted and does nothing: an AML
run never scores in-runner, so there is no metric for it to replace. `--overwrite` deletes the
predictions and manifest rather than truncating them, since a resumed run reads the ids already on
disk to decide what is left to do.

`--overwrite` is refused outright when the sidecar describes *this same run*. Deleting the rows
does not unwrite the `/aml/add` calls that produced them, and the tenant is derived from
`--run-id`, so replaying would add every case's memories a second time into a tenant that already
holds them and score the run against a doubled corpus. Drop the flag to resume where the run
stopped, or pass a new `--run-id` to measure a clean tenant. `--recall-limit` is also range-checked
before anything is deleted or ingested: the wire contract accepts 1 to 100, and finding that out
from a `422` mid-run would already have cost the previous result.

`--request-concurrency` bounds requests in flight across every case, not cases replayed at once: a
case adds its chunks and searches its questions concurrently, with the add phase completing before
any of that case's searches run. Each in-flight `/aml/add` can hold up to eight pooled connections
while it writes, so the server needs roughly `8 x --request-concurrency` connections -- the default
4 matches the default `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` of 32. Raise both together, and raise
PostgreSQL's `max_connections` to match, or writes queue inside the pool until they time out.

`--tenant-prefix` must match the deployment's `MINDBRIDGE_AML_TENANT_PREFIX` and defaults to that
same variable read locally -- unset, with no flag, the run is refused rather than given a literal
default. The manifest's tenant map is derived client-side from the value you pass, so a mismatch
produces a manifest recording tenants the server never used. Reruns against an existing `--output`
resume, and refuse to start if the sidecar manifest disagrees on benchmark, run id, deployment, or
recall limit.

To run several in one sweep, name them through the catalog instead. `--tasks aml` expands to all
eight AML tasks and needs no judge configured, because none of them is judged in-runner:

```bash
uv run mindbridge-bench eval --tasks aml --api-base-url "$MINDBRIDGE_API_BASE_URL"
```

An interrupted AML task resumes: rerun the sweep with the same `--run-id` and the runner continues
from the rows already written. The preflight that refuses an existing output exempts AML for that
reason, and only AML — every other runner writes its predictions once at the end, so an existing
output there is a finished result. Do not reach for `--overwrite` to get past an interruption; it
is for replacing a *different* run's output, and the runner refuses it in the resume case.

The tasks are `aml-locomo-refined`, `aml-longmemeval-s`, `aml-clbench`, `aml-personamem-v1-32k`,
`aml-personamem-v1-128k`, `aml-personamem-v1-1M`, `aml-personamem-v2`, and `aml-beam`. PersonaMem-v1
appears three times because its context windows are different question sets rather than
repackagings -- 589, 2727 and 2674 questions over 37, 110 and 33 shared contexts. `aml-beam`
and `aml-personamem-v2` need their corpora fetched by hand first, per the commands above.

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
  --input .benchmarks/results/aml-locomo-refined.jsonl \
  --output .benchmarks/results/aml-locomo-refined-answers.jsonl

uv run python benchmarks/aml/pipelines/locomo-refined/pipeline.py evaluate \
  --input .benchmarks/results/aml-locomo-refined.jsonl \
  --answers .benchmarks/results/aml-locomo-refined-answers.jsonl \
  --output .benchmarks/results/aml-locomo-refined-scores.jsonl
```

### Shard the scoring stages

Both vendored stages are a strictly serial `for` loop over one LLM call each -- roughly 5,000
questions through `answer` and again through `evaluate` -- so a full sweep spends hours holding one
request open at a time while the serving GPU idles. The pinned files can never be edited, but they
do not have to be: both stages take `--input`/`--output`, so shard the input and run one process per
shard. Nothing inside the pinned files changes, and the concatenated output is byte-identical to a
serial run's.

```bash
split -n l/16 -d --additional-suffix=.jsonl \
  .benchmarks/results/aml-locomo-refined.jsonl .benchmarks/results/shard-
```

```bash
ls .benchmarks/results/shard-*.jsonl | xargs -P 16 -I {} sh -c 'uv run python benchmarks/aml/pipelines/locomo-refined/pipeline.py answer --input {} --output {}.answers'
```

Then evaluate each shard against its own answers and concatenate. `evaluate` requires its `--input`
and `--answers` to hold exactly the same ids, which is why the shards must be paired rather than
scored against the whole answer file:

```bash
ls .benchmarks/results/shard-*.jsonl | xargs -P 16 -I {} sh -c 'uv run python benchmarks/aml/pipelines/locomo-refined/pipeline.py evaluate --input {} --answers {}.answers --output {}.scores'
```

```bash
cat .benchmarks/results/shard-*.jsonl.scores > .benchmarks/results/aml-locomo-refined-scores.jsonl
```

Pick the shard count from what the answer endpoint will actually serve concurrently, not from core
count. `answer` resumes from its own output, so a killed shard restarts where it stopped.

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

LoCoMo-Refined, M3-Bench, Video-MME, Video-MME-v2, EgoLife (EgoLifeQA), EgoTempo, EgoMemReason,
MEMLENS, MM-Lifelong, SuperMemory-VQA, ATM-Bench, and Mem-Gallery are consumed through thin
adapters over pinned official files.

`mindbridge-bench eval --tasks ...` fetches each of these itself, at the same revisions, so the
commands below are for populating a corpus without running a sweep — a mirror, an offline machine,
or the whole of a release rather than the files one task reads. They also stay the reference for
what `--benchmarks-root` is expected to contain.

```bash
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git .benchmarks/locomo-refined
git clone https://github.com/ByteDance-Seed/m3-agent.git .benchmarks/m3-agent
uvx --from huggingface-hub hf download lmms-eval/Video-MME \
  videomme/test-00000-of-00001.parquet \
  --repo-type dataset \
  --local-dir .benchmarks/video-mme
uvx --from huggingface-hub hf download MME-Benchmarks/Video-MME-v2 \
  test.parquet subtitle.zip \
  --repo-type dataset \
  --revision 6e4bebb03202e1ddbf3d37703e560e51c5aa2d64 \
  --local-dir .benchmarks/video-mme-v2
uvx --from huggingface-hub hf download lmms-lab/EgoLife \
  EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --repo-type dataset \
  --local-dir .benchmarks/egolife
git clone https://github.com/google-research-datasets/egotempo.git .benchmarks/egotempo
uvx --from huggingface-hub hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA \
  data/json/all_qa.json \
  --repo-type dataset \
  --local-dir .benchmarks/supermemory-vqa
uvx --from huggingface-hub hf download Ted412/EgoMemReason \
  annotations_public.jsonl \
  --repo-type dataset \
  --local-dir .benchmarks/egomem-reason
uvx --from huggingface-hub hf download xiyuRenBill/MEMLENS \
  dataset_32k.json agent_subset_195.json \
  --repo-type dataset \
  --local-dir .benchmarks/memlens
uvx --from huggingface-hub hf download MM-Lifelong/MM-Lifelong \
  day/test.json week/test.json month/train.json month/val.json \
  --repo-type dataset \
  --local-dir .benchmarks/mm-lifelong
uvx --from huggingface-hub hf download Jingbiao/ATM-Bench \
  --repo-type dataset \
  --revision 78e826dc07e97466b2f54443831ef9a83ab8b27c \
  --local-dir .benchmarks/atm-bench
uvx --from huggingface-hub hf download Ethan-Bei/Mem-Gallery \
  --repo-type dataset \
  --revision af912daba984e896e253016b7c7e334ef92c2a6f \
  --local-dir .benchmarks/mem-gallery

uv run --extra benchmarks mindbridge-bench datasets \
  --locomo-refined .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --m3-robot .benchmarks/m3-agent/data/annotations/robot.json \
  --m3-web .benchmarks/m3-agent/data/annotations/web.json \
  --video-mme .benchmarks/video-mme/videomme/test-00000-of-00001.parquet \
  --video-mme-v2 .benchmarks/video-mme-v2/test.parquet \
  --egolife .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --egotempo .benchmarks/egotempo/egotempo_openQA.json \
  --egomem .benchmarks/egomem-reason/annotations_public.jsonl \
  --memlens .benchmarks/memlens/dataset_32k.json \
  --mm-day .benchmarks/mm-lifelong/day/test.json \
  --mm-week .benchmarks/mm-lifelong/week/test.json \
  --mm-month-train .benchmarks/mm-lifelong/month/train.json \
  --mm-month-val .benchmarks/mm-lifelong/month/val.json \
  --supermemory .benchmarks/supermemory-vqa/data/json/all_qa.json \
  --atm .benchmarks/atm-bench/data/atm-bench/atm-bench.json \
  --atm-hard .benchmarks/atm-bench/data/atm-bench/atm-bench-hard.json \
  --mem-gallery-dialog .benchmarks/mem-gallery/data/dialog
```

Both ATM-Bench and Mem-Gallery are pinned by revision because the digests in this smoke are only
meaningful against a fixed revision. ATM-Bench is 3.2 GB including the raw media; Mem-Gallery is
530 MB.

Large M3-Bench media stays outside Git. A sweep naming `m3-robot` fetches it, so this is for
populating a corpus ahead of time — a mirror, or an offline machine:

```bash
uvx --from huggingface-hub hf download ByteDance-Seed/M3-Bench \
  --repo-type dataset \
  --include 'videos/robot/*' \
  --local-dir .benchmarks/m3-bench

uvx --from huggingface-hub hf download ByteDance-Seed/M3-Bench \
  intermediate_outputs/robot.tar.gz.00 \
  intermediate_outputs/robot.tar.gz.01 \
  intermediate_outputs/robot.tar.gz.02 \
  memory_graphs/robot.tar.gz \
  --repo-type dataset \
  --local-dir .benchmarks/m3-bench
```

The released SuperMemory-VQA RGB videos are fetched the same way, and a sweep naming
`supermemory-subject-1` narrows the fetch to the recordings its questions reach. Raw audio is not
part of the public release, so these files carry no audio track at all:

```bash
uvx --from huggingface-hub hf download OSU-AIoT-MLSys-Lab/SuperMemory-VQA \
  --repo-type dataset \
  --include 'data/video/*' \
  --local-dir .benchmarks/supermemory-vqa
```

The resulting annotation identity and counts are recorded in
[benchmarks/manifests/dataset-adapters-smoke.json](../benchmarks/manifests/dataset-adapters-smoke.json).

Run LoCoMo-Refined against the deployed production API. The command writes the official
`predictions.jsonl` shape and a sidecar manifest containing source, model, Prompt, retrieval, and
output identities. `MINDBRIDGE_API_KEY` identifies the exact benchmark tenant and is never written to the
manifest.

Every benchmark also requires a secret-free deployment snapshot. It records the actual capability
slots, plugin distribution versions, models, embedding space, and inference options used by
the server and Worker. Before inference begins, the runner freezes the validated snapshot and the
SHA-256 of those same bytes; credential-like keys are rejected.
This is a run artifact, not a named Profile. For example, save the following as
`.benchmarks/deployment.json` and replace the distribution versions and model IDs with those from
the deployed processes:

```json
{
  "server_generator": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "qwen3.8-max",
      "reasoning_effort": "low"
    }
  },
  "server_embedder": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "space_id": "jina-v5",
      "dimension": 1024
    }
  },
  "worker_generator": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "qwen3.8-max"
    }
  },
  "worker_media_embedder": {
    "plugin": "jina",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval",
      "device": "cuda"
    }
  },
  "worker_text_embedder": {
    "plugin": "openai",
    "distribution": "mindbridge",
    "version": "0.1.0",
    "config": {
      "model_id": "jinaai/jina-embeddings-v5-omni-small-retrieval"
    }
  }
}
```

```bash
export MINDBRIDGE_API_KEY=replace-with-a-runtime-secret
uv run mindbridge-bench locomo-refined \
  --dataset .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --output .benchmarks/results/locomo-refined-mindbridge.jsonl \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id locomo-refined-001 \
  --recall-limit 50 \
  --request-timeout-seconds 1800
```

Use `--sample-id` for a smoke subset. The example explicitly selects the experimental Top-50 recall
budget; the benchmark default remains the product-wide Top-20 budget until held-out or full-split
evidence justifies changing it. Existing results are preserved unless `--overwrite` is supplied. The
rows are `{"qa_id", "predicted_answer"}` keyed by the release's own `qa_id`, so the file
is fed straight to `./scripts/run_eval.sh` in a LoCoMo-Refined checkout; retrieved dialogue
IDs ride along in the `mindbridge_prediction_context` field that evaluator ignores.
LoCoMo-Refined is CC BY-NC 4.0, so the corpus itself is not licensed for commercial use.

Run M3-Bench through the same deployed API. The `eval` sweep cuts selected official videos into
30-second clips with MindBridge's production media encoder, uploads them into the deployment's
bucket, and writes a run-specific manifest automatically. The individual `m3` runner performs no
preparation; when invoked directly, `--prepared-media` must point to a JSON array of already
addressable objects:

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
          "uri": "s3://mindbridge-media/tenants/benchmark_m3_bedroom_01_m3-robot-001/0.mp4",
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
uv run mindbridge-bench m3 \
  --dataset .benchmarks/m3-agent/data/annotations/robot.json \
  --prepared-media .benchmarks/m3-prepared-robot.json \
  --output .benchmarks/results/m3-robot-mindbridge.jsonl \
  --api-base-url http://localhost:8000 \
  --subset robot \
  --deployment-config .benchmarks/deployment.json \
  --run-id m3-robot-001 \
  --request-timeout-seconds 1800
```

Use `--video-id` for a smoke subset. Web video IDs are YouTube IDs, and 14 of them begin with
`-`, so pass those as `--video-id=-bMyTZYVzgw`: in the separated form `argparse` reads the ID
as the next option. The runner rejects a `--subset` that does not match the
official Robot timing fields or their absence from Web. The JSONL uses the official `id`,
`question`, `answer`, `type`, `before_clip`, and `response` fields and adds MindBridge retrieval
diagnostics. Its sidecar manifest pins annotation and media hashes, both Omni calls, Jina, Prompt
versions, retrieval settings, and output hash.

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
        "uri": "s3://mindbridge-media/tenants/benchmark_egolife_A1_JAKE_egolife-a1-001/day1/11100000.mp4",
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
uv run mindbridge-bench egolife \
  --dataset .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --prepared-media .benchmarks/egolife-prepared-a1.json \
  --output .benchmarks/results/egolife-a1.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id egolife-a1-001 \
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
In the released dataset, 82 of 83 referenced sessions have an MP4; the remaining
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
uv run mindbridge-bench supermemory \
  --dataset .benchmarks/supermemory-vqa/data/json/all_qa.json \
  --prepared-media .benchmarks/supermemory-prepared-person-1.json \
  --output .benchmarks/results/supermemory-person-1.json \
  --api-base-url http://localhost:8000 \
  --subject 1 \
  --deployment-config .benchmarks/deployment.json \
  --run-id supermemory-person-1-001 \
  --request-timeout-seconds 1800
```

EgoMemReason reuses the same causal EgoLife clip contract shown above. Its prepared-media file is a
JSON array containing one `EgoLifePreparedStream` object for each selected identity. The runner
withholds clips that cross each official `query_time`, supports the released A-J option range, and
writes the exact answer-key-free leaderboard submission shape:

```bash
uv run mindbridge-bench egomem \
  --dataset .benchmarks/egomem-reason/annotations_public.jsonl \
  --prepared-media .benchmarks/egomem-prepared.json \
  --output .benchmarks/results/egomem-submission.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id egomem-001
```

MEMLENS follows the official memory-agent protocol: every question gets a fresh tenant, sessions
are consumed in release order, and the question date is supplied only at answer time. Download
`release_images.tar.gz` from the same dataset release for multimodal runs, upload the
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
        "uri": "s3://mindbridge-media/tenants/benchmark_memlens_question-001_memlens-32k-001/2658faf8f6e6.jpg",
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
uv run mindbridge-bench memlens \
  --dataset .benchmarks/memlens/dataset_32k.json \
  --prepared-images .benchmarks/memlens-prepared-images.json \
  --agent-subset-index .benchmarks/memlens/agent_subset_195.json \
  --output .benchmarks/results/memlens-32k.json \
  --api-base-url http://localhost:8000 \
  --context-window 32k \
  --deployment-config .benchmarks/deployment.json \
  --run-id memlens-32k-001
```

For the official text-only ablation, add `--text-only`, omit `--prepared-images`, and use a
deployment file without Worker plugins; image placeholders remain `[image]` and no generated image
caption is injected.

ATM-Bench replays a single person's photo, video, and email archive — 3,759 images, 533 videos,
and 6,742 emails, against the official `main` (1,013 questions) and `hard` (31 questions) splits.
The two splits are disjoint in their questions: `hard` is not a subset of `main`. Their evidence
is not disjoint the same way — of the 6,742 emails, 430 evidence references resolve to only 362
distinct emails, 5.4% of the archive: 354 cited from `main`, 13 from `hard`, and 5 of those from
both splits. The rest of the archive is distractor mass that still gets ingested.

The release carries two clocks: an image's `timestamp` is local wall clock and matches its
filename stem, but a video's is true UTC and can sit an hour off the stem for half the year. The
adapter reads capture time from the filename stem for every modality instead, so raw media and
the official schema-guided (SGM) records share one timeline. The runner refuses a `raw` run whose
staged capture time disagrees with the release's own SGM record for the same item, and refuses a
staged video that no SGM record gives a duration for.

`--media-source` picks which of ATM's two representations of the archive gets written: `raw`
(the default) sends the archive's own image and video bytes through MindBridge's own perception;
`sgm` writes the release's own schema-guided caption text instead, skipping perception entirely.
Neither is ATM's own "Raw" baseline, which puts the image directly into the answerer's context —
MindBridge's `raw` arm always goes through structured memory first, so the published column
comparable to it is ATM's **SGM** column, not its **Raw** column.

`--prepared-media` stages the archive's own bytes for a `raw` run, one entry per item, keyed by
and required to equal the official `YYYYMMDD_HHMMSS` filename stem:

```json
{
  "media": [
    {
      "media_id": "20250223_130249",
      "media_object": {
        "media_object_id": "20250223_130249",
        "kind": "image",
        "uri": "s3://mindbridge-media/tenants/benchmark_atm_archive_atm-hard-raw-01/20250223_130249.jpg",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size_bytes": 100686,
        "created_at": "2025-02-23T13:02:49Z"
      }
    }
  ]
}
```

```bash
uv run mindbridge-bench atm \
  --dataset .benchmarks/atm-bench/data/atm-bench/atm-bench-hard.json \
  --split hard \
  --media-source raw \
  --emails .benchmarks/atm-bench/data/raw_memory/email/emails.json \
  --prepared-media .benchmarks/atm-prepared-media.json \
  --sgm-image .benchmarks/atm-bench/data/processed_memory/image_batch_results.json \
  --sgm-video .benchmarks/atm-bench/data/processed_memory/video_batch_results.json \
  --output .benchmarks/results/atm-hard-raw.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id atm-hard-raw-01
```

`--split` does not filter questions — that is `--dataset`'s job, by pointing at either release
file — it only records which split the loaded file is, so the manifest cannot silently mislabel a
run. `--sgm-video` is required on a `raw` run whenever a staged item is a video, since only the SGM
record carries its duration; `--sgm-image` is not required there but still worth passing, since it
extends the same clock-agreement guard to staged images. `--question-id` is repeatable and narrows
a run to specific IDs; omit it for the whole split. The output is a JSON array of `{id, question,
qtype, answer, prediction, evidence_ids, retrieved_evidence_ids, ...}` objects in the shape the
official `JingbiaoMei/ATM-Bench` evaluator reads, with MindBridge's own retrieval diagnostics
appended in fields that evaluator ignores. ATM-Bench is scored outside MindBridge the same way;
record the official evaluator's verdict with `mindbridge-bench score`, described in "Recording an
official scorer's result" below.

Mem-Gallery replays one topic's whole multi-session dialogue as its own tenant, then answers that
topic's own questions over it — the release is twenty independent personas, and a shared tenant
would let one persona's memory answer another's question. `--dataset` names the release's
`data/dialog` directory, not a single file; the runner loads every topic file inside it and
narrows to `--topic` selections when given.

The `eval` sweep stages every image selected topics reference and writes `--prepared-images`
automatically. A direct `mem-gallery` invocation must supply that manifest itself. It covers both
the 1,003 images embedded in dialogue rounds and the 487 question images asked alongside a query;
a missing image refuses the run before ingestion starts. Images are keyed by the release's own
relative path (`image_key`); `media_object_id` is a separate, caller-assigned field, but for a
dialogue image it must be set to the official `image_id` (for example `D2:IMG_001`) because `VS`
("visual search") questions ask the model to name the matching `image_id` directly as its answer,
and that ID is what ties a retrieved image back to the release's own answer key. That official ID
is release-relative, not archive-unique — `D1:IMG_001` alone names a different picture in all
twenty topics — so `media_object_id` only has to be unique within the one topic that answers a
`VS` question with it, never across the whole staged manifest; `image_key`, which is always
topic-prefixed, is what the manifest actually requires to be unique. Question images carry no
official ID, so their `media_object_id` may be assigned freely:

```json
{
  "images": [
    {
      "image_key": "../image/Baking_Dessert_Daily_Life_Skill/D1_IMG_001.jpg",
      "media_object": {
        "media_object_id": "D1:IMG_001",
        "kind": "image",
        "uri": "s3://mindbridge-media/tenants/benchmark_mem_gallery_Baking_Dessert_Daily_Life_Skill_mem-gallery-baking-01/D1_IMG_001.jpg",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size_bytes": 123456,
        "created_at": "2026-08-14T00:00:00Z"
      }
    }
  ]
}
```

```bash
uv run mindbridge-bench mem-gallery \
  --dataset .benchmarks/mem-gallery/data/dialog \
  --prepared-images .benchmarks/mem-gallery-prepared-images.json \
  --topic Baking_Dessert_Daily_Life_Skill \
  --output .benchmarks/results/mem-gallery-baking.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id mem-gallery-baking-01
```

`--topic` is repeatable and defaults to all twenty. The output is a JSON array of `{question_id,
topic, point, question, answer, prediction, clue, retrieved_ids, ...}` objects in the shape the
official `YuanchenBei/Mem-Gallery` evaluator reads. `point` is the official nine-way task code
(`FR`, `MR`, `TR`, `VR`, `TTL`, `VS`, `CD`, `KR`, `AR`); scoring is broken out per `point`, and
`AR` specifically rewards abstaining with the release's own refusal text (`Not mentioned.`), so
its score is not comparable to the other eight without reading it on its own. Mem-Gallery is
scored outside MindBridge too; record the official evaluator's verdict with
`mindbridge-bench score`, described in "Recording an official scorer's result" below.

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
          "uri": "s3://mindbridge-media/tenants/benchmark_mm_lifelong_month_val_mm-lifelong-month-val-001/14/00000.mp4",
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
uv run mindbridge-bench mm-lifelong \
  --dataset .benchmarks/mm-lifelong/month/val.json \
  --prepared-media .benchmarks/mm-lifelong-month-val-prepared.json \
  --output .benchmarks/results/mm-lifelong-month-val.jsonl \
  --api-base-url http://localhost:8000 \
  --split month_val \
  --deployment-config .benchmarks/deployment.json \
  --run-id mm-lifelong-month-val-001
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
            "uri": "s3://mindbridge-media/tenants/benchmark_video_mme_001_video-mme-001/0000.mp4",
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
uv run --extra benchmarks mindbridge-bench video-mme \
  --dataset .benchmarks/video-mme/videomme/test-00000-of-00001.parquet \
  --prepared-media .benchmarks/video-mme-prepared.json \
  --output .benchmarks/results/video-mme-long.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --duration long \
  --transcript-source none \
  --run-id video-mme-001
```

### Video-MME-v2

A separate benchmark from Video-MME, not a newer split of it. 800 videos, 3,200 questions, options
A through H, no short/medium/long bands. Its unit of scoring is a **group**: the four questions over
one video, scored together.

The runner reuses the prepared-video manifest shape above, keyed on the official numeric `video_id`.
Media is 97.8 GiB across 40 zips, so acquire it separately from the annotations:

```bash
uvx --from huggingface-hub hf download MME-Benchmarks/Video-MME-v2 \
  --include "videos/*" \
  --repo-type dataset \
  --revision 6e4bebb03202e1ddbf3d37703e560e51c5aa2d64 \
  --local-dir .benchmarks/video-mme-v2
```

Two numbers are reported, reproducing the released `_rating.json` and `_acc.json` cell for cell:

- `metrics.rating` is the leaderboard number, averaged over whole **groups** on a 0-100 scale. A
  `relevance` group scores on how many of its four are correct, quadratically: one of four is worth
  6.25, four of four is worth 100. A `logic` group scores on the longest unbroken correct prefix of
  its dependency chain, so a group that misses question 1 earns nothing for a correct 2 through 4.
- `metrics.accuracy` is plain per-question accuracy with the same breakdowns.

Both break out by `group_type`, `level`, `second_head`, and `third_head`. Note that the rating's
taxonomy cells are keyed on each group's **fourth** question, which is what the released scorer
reads; `level`, `second_head`, and `third_head` all vary inside a group, so the two views key the
same field differently on purpose.

Scores are on the released 0-100 scale, unlike Video-MME's 0-1 fractions, and the naming of the
accuracy fields is inverted relative to it: here `overall` counts every question and scores an
abstention wrong, matching the released `_acc.json`, while `answered_accuracy` is the answered-only
figure the same script only prints. Quote `rating.overall` against the leaderboard; quote
`accuracy.overall` only alongside it, because the gap between them is what the benchmark was rebuilt
to expose.

`--video-id` scopes a run and always carries all four of a video's questions, because a partial
group has no defined rating. There is deliberately no `--level` counterpart to Video-MME's
`--duration`: `level` varies between the questions of a group, so filtering on it would either split
a group or silently mean "groups whose fourth question is level N". `--group-type` is available and
is safe, being constant across a group. `--transcript-source` behaves as it does for Video-MME:

```bash
uv run --extra benchmarks mindbridge-bench video-mme-v2 \
  --dataset .benchmarks/video-mme-v2/test.parquet \
  --prepared-media .benchmarks/video-mme-v2-prepared.json \
  --output .benchmarks/results/video-mme-v2.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --transcript-source none \
  --run-id video-mme-v2-001
```

The released subtitles are word-level JSONL (`{"text": ..., "start_time": ..., "end_time": ...}`),
one file per video in `subtitle.zip`. Grouping those words into segment transcripts happens in
whatever prepares the media manifest; the runner only checks that what it was handed agrees with
the `--transcript-source` the run declares.

EgoTempo writes the official `V`, `Q`, `QA`, `A`, `C`, and `M` fields. Run its pinned
`gemini_eval.ipynb` for the released semantic judge rather than substituting a local metric:

```bash
uv run mindbridge-bench egotempo \
  --dataset .benchmarks/egotempo/egotempo_openQA.json \
  --prepared-media .benchmarks/egotempo-prepared.json \
  --output .benchmarks/results/egotempo.json \
  --api-base-url http://localhost:8000 \
  --deployment-config .benchmarks/deployment.json \
  --run-id egotempo-001
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

Use `mindbridge-bench score` whenever an official evaluator or leaderboard produces a verdict
outside MindBridge. This includes attaching an EgoMemReason leaderboard result and replacing a
MindBridge-selected judge score with the benchmark's own protocol for a citable comparison. The
run manifest is written before an external scorer executes, so it can only pin inputs. The score
command records the verdict in a `*.score.json` sidecar, re-hashes the predictions, and refuses
numbers that belong to a different run:

```bash
uv run mindbridge-bench score \
  --predictions .benchmarks/results/sweep-001/locomo-refined/predictions.jsonl \
  --manifest .benchmarks/results/sweep-001/locomo-refined/predictions.jsonl.manifest.json \
  --scorer-output .benchmarks/results/sweep-001/locomo-refined/scorer-summary.json \
  --scorer-repository mem-eval-suite/LoCoMo_refined \
  --scorer-command "./scripts/run_eval.sh --metrics llm f1 bleu --llm-judge refined" \
  --judge-model Qwen/Qwen3-14B \
  --answer-backbone qwen3.8-max \
  --scored-question-count 1382 \
  --metric llm=82.65
```

The sidecar lands beside the predictions, which is where the results table looks for it:
`mindbridge-bench eval --report .benchmarks/results/sweep-001` then prints that `llm` figure
against the run, attributed to the official scorer.

`--judge-model` and `--answer-backbone` are the fields that make two LoCoMo-Refined numbers
comparable or not: the official judge is `Qwen/Qwen3-14B` under the `refined` prompt, and
`run_eval.sh` also accepts the looser `original` LoCoMo judge. The run manifest additionally
records `category_question_counts`, because 802 of the release's 1,382 questions are category 4
and a subset run can skew that mix without saying so. See
[SOTA baselines for the supported benchmarks](benchmarks-sota.md) for what each number has to
beat.

## Cloud embedding smoke

Cloud multimodal embedding dependencies are optional, so the API and edge packages stay light:

```bash
uv sync --extra cloud-models
uv run --extra cloud-models mindbridge-bench jina
```

The checked-in result is
[benchmarks/manifests/jina-omni-small-smoke.json](../benchmarks/manifests/jina-omni-small-smoke.json).

## Reading results honestly

A checklist for anyone about to quote a number from this harness.

| Question | If the answer is no |
| --- | --- |
| Complete official split? | It is a diagnostic run. Say so. |
| Manifest committed beside the result file? | It is not citable. |
| Blind-answer control run? | You cannot separate memory from model. |
| Deployment snapshot committed with the run? | It is not reproducible. |
| Released text, or raw audiovisual? | Text-only results are not multimodal results. |
| Thinking mode confirmed off on the answer endpoint? | The judge may be scoring reasoning traces. |

Two failure modes have produced wrong conclusions in this repository before, both worth guarding
against:

- **Bucketing that is confounded with question type.** A LoCoMo span-count analysis appeared to
  show 81% → 52% degradation. After controlling for question category, almost all of it was
  category mix rather than span count.
- **An ablation whose branch never ran.** Prove the code path under test actually executed before
  reading its metrics. A self-referential assertion or a no-op ablation produces a clean-looking
  number that means nothing.

## Related

- [CLI reference](api/cli.md#mindbridge-bench) — the runner table.
- [SOTA baselines](benchmarks-sota.md) — comparison targets per benchmark.
- [Architecture](architecture.md) — the production path these runs exercise.
