# Recall programs, identity-anchored keys, and completeness-aware grounding

Design for autoresearch round r0910 (2026-09-10). Baseline `af98692e` (mindbridge-v03 tip).
Models: Qwen3.8-27B (`qwen3.8-27b` at inner-prism) answerer/judge, tencent/WeMM-Embedding-2B (2048-d),
FunASR speech. Priority benchmarks: ATM-Bench (main, hard; sgm and raw), M3-Bench-Robot; then MemLens,
PersonaMem-v3, LoCoMo-Refined.

## 1. Problem, from measurement

Per-question forensics of the current baseline artifacts (12,052 samples) and three literature tracks:

| Finding | Evidence |
| --- | --- |
| Retrieval ranking is not the binding constraint on point questions | median gold rank is 1 on every gold-labelled task; any-gold@ranked36 is 74–98% |
| Abstention is a pure loss | m3-robot refuses 48% (accuracy when answering 56%, blind scores 18% on the refused ones); MemLens refuses 54% of non-refusal questions; ATM-main refused 114 questions with all gold in the window |
| Set-valued questions have no retrieval formulation | ATM-Hard gold sets are 3–17 ids, window is 12: all-gold@window 0% for `list_recall` while recall at depth 36 is 87%; MemLens counting/arithmetic/duration and m3 counting sit at the blind rate |
| Sequence questions have no formulation | MemLens `previnfo` ("what did I say before X") abstains 89%; LoCoMo adjacency gold shares zero terms with the question |
| Multi-evidence composition across time is where memory systems lose the tier | ATM-Hard: every embedding-RAG system scores 12–18; agents that grep the whole Schema-Guided store in many rounds score 35–63 with the same weights; the paper's own "rewrite + re-retrieve on dense top-10" bought −0.1 |
| Media reach the index only through their description | ATM SGM text vs raw media: +16 to +24pp for every agent, +21.7pp at oracle; MemLens: 65.7% of questions need the image, frontier LVLMs fall below 2% without it; our adapter stores BLIP captions and never downloaded the images |
| Stable identity across clips is the multimodal lever | M3-Agent ablations: semantic memory −17.1, identity equivalence −11.2, reasoning loop −11.7; its un-RL'd controller (20.7) is below MindBridge (26.1) |
| Union keys are safe, replacement loses | Fidelity Before Structure: chunk ∪ artifact = chunk (p=0.39); SimpleMem: set union dense ∪ BM25 ∪ symbolic with id dedup, not weighted fusion |

Known nulls we must not re-buy: RRF fusion (−8.7pp R@1 locally), cross-encoder rerank, widening k on point
questions, one-hop graph expansion (recall up, answers flat, three independent measurements), knee pruning,
dated evidence lines, thinking mode on LoCoMo/LongMemEval.

## 2. Goals and non-goals

Goals: stronger memory first. (1) Give `ask()` retrieval semantics beyond top-k similarity — completeness
for set, sequence and entity shapes, executed deterministically by the authoritative store. (2) Put
identity-anchored, decontextualized facts about media into the index as additional keys. (3) Ground on an
evidence set whose size follows the question shape, rendered chronologically with completeness stated.
(4) Stop refusing when the caller asks for a best-effort answer.

Non-goals: rerankers, graph stores, RRF, new dependencies, replacing verbatim content, benchmark-specific
prompt strings inside the product, training.

## 3. Architecture

### 3.1 Recall primitives (store layer, `infrastructure/local/store.py`)

All read `memory_records` with `forgotten_at IS NULL`, respect the same scope arguments as
`read_memories(active_only=True)`, and are bounded by an explicit `max_rows`.

| Primitive | Semantics | Implementation |
| --- | --- | --- |
| `similar(query, k, filters)` | existing hybrid search | unchanged (`_search_prepared`) |
| `match(terms, any_of, time, modality, memory_type, max_rows)` | every active memory whose content contains the terms (case-folded substring, NFKC), optionally within an `occurred_at` range | SQL `instr(lower(content), ?)` over `memory_records`; content already carries transcripts, captions and speech prose, so media are reachable |
| `window(time, modality, memory_type, max_rows)` | every active memory in an `occurred_at` range | SQL range on `occurred_at`/`occurred_end` |
| `neighbors(memory_ids, before, after)` | the memories adjacent in corpus order | order = `(occurred_at, rowid)`; rowid is insertion order, which every adapter writes in turn order. Verify `memory_records` is a rowid table; if not, add an insertion sequence |
| `entity(name_or_identity_id, ...)` | memories attributed to an identity | alias lookup → identity id → existing `identity_ids` pushdown; when no identity registry matches, degrade to `match(name)` |

Union with id dedup, never weighted fusion. Results carry `(memory_id, occurred_at, source_op, score|None)`.

### 3.2 Recall planner (read path, one small model call)

`ask()` asks the generation backend for a recall plan before retrieving. Input: the question, the reference
time, and a corpus digest (active record count, time span, modalities present, whether identities exist).
Output: strict JSON.

```json
{"shape": "point|set|sequence|entity|composite",
 "steps": [
   {"op": "similar", "query": "...", "k": 12, "time": null, "modality": null},
   {"op": "match", "terms": ["Cairo", "flight"], "any_of": true, "time": ["2024-09-01", "2024-10-01"], "max_rows": 200},
   {"op": "window", "time": ["2024-09-20", "2024-09-30"], "modality": "image", "max_rows": 200},
   {"op": "neighbors", "of": "step:0", "before": 2, "after": 2},
   {"op": "entity", "name": "Lily"}
 ]}
```

Rules: the plan is validated against a schema; any invalid plan, missing capability, or backend error falls
back to `{"shape": "point", "steps": [{"op": "similar", ...}]}`, i.e. today's behaviour. `point` plans use
`similar` only (k widening on point questions is a measured null). Rounds are bounded: plan → execute →
answer; if the answer abstains and the policy is best-effort, one replan round receives the first-round
evidence labels and the abstention, then the final answer is forced. Route by capability: a backend
without `plan_recall` yields the fallback plan. Off by default (`MemoryConfig.recall_planning=False`);
`ask(..., plan=...)` may later accept a caller-supplied plan so agents can drive the primitives directly
(same shape over MCP).

### 3.3 Completeness-aware grounding

Replace the fixed `limit` window with a shape-dependent budget: point → today's `limit` hits; set,
sequence, entity, composite → every row produced by exhaustive ops up to `recall_set_budget_chars`
(default 30,000 chars), then `similar` rows by rank until the budget is spent. Exhaustive rows are ordered
chronologically, similarity rows follow by rank, all labelled `E1..En` (existing `evidence_label`). The
prompt states the plan and its completeness ("the evidence is every record matching X in [range]"; or
"N further matches were omitted"), so counting and list answers are licensed only when the set is complete.
Media attachment and elision rules are unchanged.

Two bounds keep a set read from being the corpus. A step whose predicate selected more than
`_RECALL_NON_SELECTIVE_SHARE` (0.2) of the active records, or more than four times the ask's own
`limit` where that fifth is smaller, is non-selective: it contributes no rows, is counted on the stage span as
`mindbridge.recall.non_selective_steps`, and the note tells the reader what the predicate matched and that
it holds the ranking instead (measured on LoCoMo, an `entity` step that degraded to matching a name as text
selected ~300 of ~600 records and cost 0.721 -> 0.528 accuracy on those questions). The rows a selective
read did return are additionally capped at `MemoryConfig.recall_set_max_rows` (default 60), chronologically,
with the remainder counted into the same "not shown" shortfall that declares the set incomplete.

### 3.4 Answer policy and answer shaping (reader)

Port `answer_policy` (`strict` | `best_effort`) from branch `claude/mindbridge-memory-research-3f4fd0`
(commits `1d0c4ba2`, `74eff71e`: SDK, REST, MCP, harness mapping, run record). Default stays `strict`; the
harness maps per task (protocol alignment: every reference system answers). Separately, one general line
in the grounded prompt: answer the whole question (every component asked), prefer a short phrase to a
sentence unless asked to explain, list only items the evidence supports. No dataset strings in `src/`.

The shaping line was implemented, measured, and removed. Under `strict` it cost LoCoMo
0.747 -> 0.545 with abstention rising 9.9 % -> 36.2 %, MemLens 0.300 -> 0.283 with abstention
32 % -> 52 %, and ATM-hard-sgm abstention 35 % -> 48 %: asking for the shortest complete answer
taught the reader to refuse rather than to answer short. `strict` is byte-identical to the prompt
that predates `answer_policy`, and `best_effort` differs from it only by its own abstention
instruction.

### 3.5 Identity-anchored decontextualized keys (write path)

Extend the existing opt-in vision describer so a media memory gains, as additional content sections
(indexed lexically via part 0 and densely via the aggregate and atomic keys; verbatim asset stays
authoritative):

- `[visual description:<asset>]` structured: what is shown, visible text (OCR), counts, place cues,
  visible date/time cues, people (by identity id when known), tags.
- `[facts:<asset>]` durable statements distilled from frames plus the transcript when present ("Lily is
  marketing staff", "the yoga mat lives in the storage room", "speaker_2 is called Lily").
- Name binding: when the distillation names a diarised speaker, call `register_identity(identity_id,
  name)` so `_speech_retrieval_text` renders `Lily: …` for every later clip that resolves to the same
  identity. Names carry across clips through the existing identity graph.

Cache by asset sha256 in `visual_descriptions` (existing). Write-path only (mirror
`_pending_visual_descriptions`; never `_embedding_content`, which the query path shares). One model
call per media item at write time; ATM has 4,292 media items, the m3 dev slice 1,700 clips.

### 3.6 MemLens pixels (harness)

Download `needle_images/`, and make the MemLens adapter attach the image as a media part alongside the
caption when the file exists. Harness-only; measures the product's existing multimodal path.

### 3.7 Reasoning effort (configuration lever, reported separately)

`enable_thinking: true` on the answer call (the endpoint returns `reasoning_content` separately). Measured
on ATM-Hard only where the literature shows +23pp on identical weights; reported as a reader lever, not a
memory gain.

## 4. Invariants

SQLite authoritative, Zvec rebuildable; new primitives read SQLite only. Write-path sections change the
embedded content → `_LEGACY_INDEX_RECIPES` bump so old stores re-embed (opt-in describer means default
stores are unaffected). Planner and grounding changes are inside `_ask_operation`, within the operation
span and asset leases; `_reinforce_answered` and `_note_query_failure` key off the final hit set. REST, MCP,
CLI translate only. Every new behaviour is off by default and has a test that fails if the fallback path
changes.

## 5. Evaluation protocol (pre-registered)

- Dev/holdout slices per `autoresearch/orchestrator-260905/tools/launch.sh`: ATM-main dev 0:300 / holdout
  300:713 (sgm for read-path arms, raw for write-path arms); ATM-hard whole (31 questions, 1 question =
  3.2pp: report, never decide on it alone); m3-robot dev 0:25 (314 q) / holdout 25:75; memlens-32k dev 0:60 /
  holdout 60:135; locomo dev 0:4 / holdout 4:6 (regression guard); personamem 0:8 (regression guard).
- Every arm is paired against the HEAD baseline on the same questions (`tools/paired.py`, cluster
  bootstrap CI95, W/L/T). Read-path arms reuse the baseline stores; write-path arms re-ingest.
- Accept a mechanism when its dev delta is positive with CI excluding zero on a priority benchmark and no
  priority or guard benchmark regresses with CI excluding zero; confirm on holdout before reporting.
- Levers are reported in three classes: memory mechanism (3.1–3.3, 3.5), protocol alignment (3.4 policy,
  3.6), reader configuration (3.7). Judge is Qwen3.8-27B: all headline numbers are internal, not
  leaderboard-comparable.
- `reinforce_on_answer: false`, `recall_limit 12`, seed 0 everywhere. No source edits under a running
  eval's checkout (pinned detached worktrees).

## 6. Risks

Planner adds one call per `ask()` (latency and tokens; accepted: stronger before faster). Set ops on
point questions add noise — guarded by shape. Distillation hallucinations enter the index — kept as
separate sections so the reader sees provenance, and the verbatim asset remains attached. Name binding
errors propagate across clips — not bounded by `identity_link_min_assets`, which gates face↔voice
linking (a different join) rather than naming; a wrong name is bounded instead by the never-overwrite
guard (a standing name is never replaced by a later assertion) and is reversible through `rollback()`,
since binding lands as an ordinary `IDENTIFY` operation. 31-question ATM-Hard is noisy — decisions use
ATM-main dev and m3 dev.

## 7. Work packages

| WP | Owner | Files | Depends on |
| --- | --- | --- | --- |
| A infra + baselines | bench-infra agent | autoresearch/orchestrator-260910/**, .benchmarks/** | — |
| B answer policy port → recall primitives, planner, grounding | recall agent (own worktree) | store.py, memory.py (ask path), new recall.py, models/openai_sdk.py (planner), types.py, api/*, benchmarks/eval.py, prompts.py, tests | — |
| C identity-anchored keys | describe agent (own worktree) | memory.py (write path), models/openai_sdk.py (describer), store.py (identity naming), tests | — |
| D MemLens images | memlens agent (own worktree) | benchmarks/eval_adapters.py, task_catalog.py, .benchmarks/memlens | — |
| E review | reviewer agents | read-only | B, C |
| F integration + arms | lead | merge B, C, D; launch arms | A–E |
