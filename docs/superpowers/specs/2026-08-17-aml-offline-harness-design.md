# AML Offline Harness Design

Run the six Agent Memory Leaderboard textual benchmarks against MindBridge
locally, before submitting to the AML industry board.

Date: 2026-08-17

## Why

MindBridge has eleven benchmark runners, none of which touch AML. AML does not
run benchmarks for participants to download: it calls two endpoints the
participant hosts (`Add`, `Search`), then answers and scores under its own
fixed pipeline. Submitting blind means learning the score after the run.

The platform publishes its answer and judge stages at
[AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard).
The only unpublished piece is the retrieval driver that sits between the
dataset and those stages. Writing that driver is what makes an offline run
possible.

## Two facts that constrain the work

**gpt-4o-mini is mandatory for participant Add and Search internals**, on both
the industry and academic boards. Offline runs are free to use any model;
a real submission is not. Offline numbers produced with Qwen3.8-Max therefore
measure the architecture's ceiling, not the submission's score.

**The 2026 challenge submission window closed on 7 August 2026.** AML continues
to operate and accepts systems for later leaderboard releases. This work
targets a later release, not a deadline.

## Scope

Six textual benchmarks: LoCoMo, LongMemEval-S, PersonaMem (v1 and v2), BEAM,
CL-Bench, ScriptMem.

Out of scope: the coding track (SWEContextBench is unreleased — no data, no
verifier, no pipeline) and the multimodal track (ATM-Bench, Mem-Gallery are
not in the public evaluation release).

## Architecture

```text
6 loaders ──► AmlCase ──► driver ──► POST /aml/add    ──► extract ──► remember()
 (datasets)  (common IR)     │                                        (MindBridge)
                             └──► POST /aml/search ──► recall()
                                        │
                                        ▼
                        vendored pipeline.py: answer → evaluate → manifest
```

### 1. Submission surface — `src/mindbridge/api/aml.py`

Two routes on the existing FastAPI app. This is the artifact AML will call
during a real submission, so the offline harness exercises the same code the
platform will.

`POST /aml/add`

- Accepts `{request_id, messages[], user_id, session_id}`.
- Feeds the chunk to the configured `Generator` (`qwen3.8-max`, `json_mode`) to
  extract atomic facts, preferences, and rules.
- Writes each extracted item through `kernel.remember()`.
- Echoes `request_id`, `user_id`, and `session_id` byte for byte.

AML requires that a 200 means the content is already searchable. `remember()`
already satisfies this: it awaits `_index_memory` before returning, so the
embedding is written inline. No job polling is involved — the asynchronous
job path belongs to `/v1/observations`, not to `remember()`.

`POST /aml/search`

- Accepts `{query, options?, user_id, top_k}`.
- Issues `RecallRequest(mode=RecallMode.SEARCH, limit=top_k,
  include_evidence=False)`.
- Maps `RecallResult.memories` to `{id: memory_id, content: summary,
  created_at}`, preserving rank order. `score` is omitted: `MemoryView` carries
  `salience` and `strength`, neither of which is a query-relevance score, and
  AML treats the field as optional and the array order as authoritative.
  `EvidenceView` is media-shaped (`media_url`) and irrelevant to text
  benchmarks.

### Tenant derivation and authorization

`TenantApiKeyAuthenticator` proves an **exact set** of tenant IDs per key.
AML supplies arbitrary `user_id` values (`eval:<run_id>:locomo:conv-0`) that
cannot be enumerated in advance, so the AML routes need their own rule.

One AML key authorizes one tenant **namespace**. The route derives
`tenant_id = f"{prefix}:{sha256(user_id).hexdigest()[:32]}"` and never accepts
a caller-supplied tenant. A caller therefore cannot name a tenant outside the
namespace, and distinct `user_id`s cannot collide into one tenant — which is
what AML's cross-user retrieval prohibition requires. Hashing also keeps the
identifier inside the 255-character `Identifier` limit regardless of what AML
sends. The readable `user_id` to `tenant_id` mapping is recorded in the run
manifest for debugging.

Health check: `GET /healthz` is unchanged. AML's health endpoint is
configurable and only defaults to `/health`, so the submission form carries
`/healthz`. If the platform turns out to reject a custom path — the smoke test
would surface that immediately — rename it then. Do not add an alias: an alias
is the only option here that adds permanent surface, and the OpenAPI snapshot,
telemetry `excluded_urls`, and docs would all have to describe two equivalent
endpoints forever.

### 2. Dataset loaders — `src/mindbridge/benchmarks/aml/loaders/`

One loader per benchmark, each normalising to a shared `AmlCase`:

- `user_id`
- `chunks[]` — messages split at AML's documented boundary (20 messages or
  2,000 words)
- `questions[]` — carrying whichever fields that benchmark's official pipeline
  reads: `gold_answer`, `rubrics`, `options`, `qa_type`, `system_prompt`

Upstream sources, all public:

| Benchmark | Source |
| --- | --- |
| LoCoMo | `snap-research/locomo` (already vendored at `.benchmarks/locomo`) |
| LongMemEval-S | `xiaowu0162/longmemeval` |
| PersonaMem v1/v2 | `bowen-upenn/PersonaMem` |
| BEAM | `mohammadtavakoli78/BEAM@3e12035532eb85768f1a7cd779832b650c4b2ef9` |
| CL-Bench | `tencent/CL-bench` |
| ScriptMem | `memorax-ai/ScriptMem` |

AML evaluates against refined and held-out splits that are not distributed.
Offline runs use the public upstream splits, so absolute scores are not
comparable to leaderboard entries.

### 3. Driver — `src/mindbridge/benchmarks/aml/driver.py`

Per case: add every chunk, then search once per question at `top_k=100`, then
emit the JSONL each official pipeline expects. Field names differ per
benchmark and are not negotiable — LoCoMo and LongMemEval read
`speaker_1_memories`, CL-Bench reads `retrieval.selected`, ScriptMem keys on
`qa_id` plus `dataset`.

### 4. Scoring — vendored, unmodified

`benchmarks/aml/pipelines/` holds the six published pipelines at
`AML-memory/agent-memory-leaderboard@5761ed58502d24153115cbdc010e44957cb18c3a`,
recorded with a per-file sha256 in the run manifest.

Run `pipeline.py answer` then `pipeline.py evaluate` with `ANSWER_*` and
`JUDGE_*` pointing at the Qwen3.8-Max endpoint. Their transport is httpx
against `/chat/completions` with `Authorization: Bearer` — an OpenAI-compatible
call that needs no adaptation.

**These files are not edited.** Their transport being httpx rather than the
OpenAI SDK does not affect scores: all six send exactly
`{"model", "messages", "temperature": 0}`, byte-identical to what
`AsyncOpenAI.chat.completions.create` produces with the same arguments. The
reason to leave them alone is auditability and upgrade cost — unmodified,
sha256-pinned files make "same source as the leaderboard" a claim a reviewer
can check, and an AML pipeline bump becomes a re-pin rather than a re-port.

Code we write ourselves uses `AsyncOpenAI` throughout. Note the distinction
from routing these stages through the repository's `OpenAIGenerator`, which
would add `stream=True`, `modalities`, `max_tokens`, and `stream_options` to
the body — that changes the request and would move the scores.

### Thinking-mode hazard

The published pipelines accept `--max-tokens` but never place it in the
payload, and offer no way to disable thinking mode. Qwen3.8-Max is a thinking
model, and this repository has already been bitten by it: `ask` requires an
explicit `enable_thinking: false`.

Left alone, the answer stage can return reasoning text, the judge scores that
text, and the run reads as a memory-system failure rather than a harness
misconfiguration.

**Resolution: `enable_thinking` defaults to false on the Qwen3.8-Max endpoint,
which we operate.** No vendored file changes, no deviation from the published
pipeline. The deployment snapshot records the endpoint default so the run stays
reproducible.

## Performance

Add is the bottleneck: roughly 5,000 questions sit behind tens of thousands of
chunks, each costing one extraction call plus N `remember()` writes.

- Bounded concurrency via semaphore, reusing the existing `request_concurrency`
  knob.
- Chunks within one `user_id` run serially to preserve temporal order; chunks
  across users run concurrently.
- Extraction results cache to disk keyed by chunk hash, making reruns free.
- Driver output is append-only JSONL, matching the resume behaviour the
  official pipelines already implement through their `done` set.

## Repository conventions this work must satisfy

- The extraction prompt is a `PromptSpec` in `mindbridge.prompts.ALL_PROMPTS`,
  with its sha256 registered in `tests/contracts/test_prompt_catalog.py`.
  Prompt text cannot change without a version bump.
- New routes change `tests/contracts/snapshots/openapi.json`; the snapshot is
  regenerated in the same commit that adds them.
- The run manifest follows the existing runner fields (`source_repository`,
  `source_sha256`, `deployment`, `run_id`, `tenant_prefix`, `recall_limit`,
  `request_concurrency`).

## Testing

- One shape test per loader against a three-record fixture.
- One contract test for the add/search pair: byte-exact echo of all three IDs,
  a search immediately after a 200 returns the added content, and a search
  under a different `user_id` returns empty.

## Output

A run manifest matching the existing benchmark runner convention
(`source_repository`, `source_sha256`, `deployment`, `run_id`, `tenant_prefix`,
`recall_limit`, `request_concurrency`),
written to `benchmarks/manifests/`.

## What this cannot tell us

Offline scores measure relative progress between MindBridge versions. They are
not leaderboard scores: the splits differ, AML's answer and judge models are
undisclosed, and a real submission must run Add and Search on gpt-4o-mini
rather than Qwen3.8-Max.
