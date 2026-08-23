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

Most runners need nothing past the core install because they drive the product through its own
API. `video-mme` and `datasets` need `--extra benchmarks`; `jina` and `bakeoff` load the local
embedder and need `--extra cloud-models`. A runner whose extra is missing names it and exits
instead of failing part-way through a run.

Runners need a live API and a bearer token in `MINDBRIDGE_API_KEY`. Every generated tenant ID must
be in the deployment's `MINDBRIDGE_TENANT_API_KEYS_JSON` **before** the API starts — one key can
authorize all of them.

Runners write predictions and a manifest to `--output` and print nothing on stdout. Progress goes
to stderr; `-q` silences it.

Long runs are fragile in a specific way: an upstream hiccup mid-stream can escape as an unhandled
error, and results land only after the whole run completes. Shard by `--run-id` and run the shards
in parallel so a failure costs one shard rather than the sweep.

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
unaffected; AML runs ScriptMem server-side against its own copy.

```bash
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git .benchmarks/locomo-refined
git clone https://github.com/mohammadtavakoli78/BEAM.git .benchmarks/beam
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
  --top-k 100 \
  --concurrency 4
```

`--concurrency` bounds requests in flight across every case, not cases replayed at once: a case
adds its chunks and searches its questions concurrently, with the add phase completing before any
of that case's searches run. Each in-flight `/aml/add` can hold up to eight pooled connections while
it writes, so the server needs roughly `8 x --concurrency` connections -- the default 4 matches the
default `MINDBRIDGE_DATABASE_MAX_POOL_SIZE` of 32. Raise both together, and raise PostgreSQL's
`max_connections` to match, or writes queue inside the pool until they time out.

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

LoCoMo-Refined, M3-Bench, Video-MME, EgoLife (EgoLifeQA), EgoTempo, EgoMemReason, MEMLENS,
MM-Lifelong, and SuperMemory-VQA are consumed through thin adapters over pinned official files. Use Git for code
releases and the Hugging Face CLI for Hub datasets; MindBridge does not ship another downloader:

```bash
git clone https://github.com/mem-eval-suite/LoCoMo_refined.git .benchmarks/locomo-refined
git clone https://github.com/ByteDance-Seed/m3-agent.git .benchmarks/m3-agent
uvx --from huggingface-hub hf download lmms-eval/Video-MME \
  videomme/test-00000-of-00001.parquet \
  --repo-type dataset \
  --local-dir .benchmarks/video-mme
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

uv run --extra benchmarks mindbridge-bench datasets \
  --locomo-refined .benchmarks/locomo-refined/data/raw/locomo_refined.json \
  --m3-robot .benchmarks/m3-agent/data/annotations/robot.json \
  --m3-web .benchmarks/m3-agent/data/annotations/web.json \
  --video-mme .benchmarks/video-mme/videomme/test-00000-of-00001.parquet \
  --egolife .benchmarks/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json \
  --egotempo .benchmarks/egotempo/egotempo_openQA.json \
  --egomem .benchmarks/egomem-reason/annotations_public.jsonl \
  --memlens .benchmarks/memlens/dataset_32k.json \
  --mm-day .benchmarks/mm-lifelong/day/test.json \
  --mm-week .benchmarks/mm-lifelong/week/test.json \
  --mm-month-train .benchmarks/mm-lifelong/month/train.json \
  --mm-month-val .benchmarks/mm-lifelong/month/val.json \
  --supermemory .benchmarks/supermemory-vqa/data/json/all_qa.json
```

Large M3-Bench media stays outside Git. Acquire it through the official Hugging Face client rather
than a MindBridge downloader:

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

Acquire the released SuperMemory-VQA RGB videos the same way; raw audio is not part of the public
release:

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

LoCoMo-Refined, MM-Lifelong, EgoTempo, and EgoMemReason are scored outside MindBridge. A run manifest is
written before any of those scorers execute, so it can only pin inputs. Record their output in a
`*.score.json` sidecar instead, which re-hashes the predictions and refuses numbers that belong to
a different run:

```bash
uv run mindbridge-bench score \
  --predictions .benchmarks/results/locomo-refined.jsonl \
  --manifest .benchmarks/results/locomo-refined.jsonl.manifest.json \
  --scorer-output .benchmarks/results/locomo-refined-scorer-summary.json \
  --scorer-repository mem-eval-suite/LoCoMo_refined \
  --scorer-command "./scripts/run_eval.sh --metrics llm f1 bleu --llm-judge refined" \
  --judge-model Qwen/Qwen3-14B \
  --answer-backbone qwen3.8-max \
  --scored-question-count 1382 \
  --metric llm=82.65
```

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
