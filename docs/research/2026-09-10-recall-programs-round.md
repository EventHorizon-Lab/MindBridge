# Recall programs round — 2026-09-10/11

Design: [recall programs spec](../superpowers/specs/2026-09-10-recall-programs-design.md). Code: branch
`claude/mindbridge-memory-optimization-738edb` (tip after this round). Every number below is an internal,
paired, per-question comparison on the same slice with the same store and seed, judged by `qwen3.8-27b`
(the answering model), so nothing here is comparable to a published leaderboard row. Answerer
`qwen3.8-27b`, embedder `tencent/WeMM-Embedding-2B` (2048-d), speech FunASR; `recall_limit 12`,
`reinforce_on_answer false`. CI95 is the harness cluster bootstrap; single-unit ATM splits have none.

## Premises corrected before building

- "PI + Qwen3.8-27B = 49.9 on ATM-Bench-Hard" is not a memory system. Pi is a coding-agent harness in the
  benchmark's agent track that greps the pre-built Schema-Guided Memory JSON in unbounded rounds at
  reasoning effort xhigh (same weights: low 26.6, medium 30.9, xhigh 49.9). Published embedding-RAG memory
  systems score 12–18 on Hard; MindBridge was already at 21–23. Hard has 31 questions, so one question
  is 3.2 points.
- MemLens: the adapter stored the release's BLIP-2 captions and the evidence images had never been
  downloaded; 65.7% of its questions need the image. The published Claude row is Sonnet 4.5 (36.5 at
  32k), and text-only memory agents cap at 30–33.
- Forensics on 12,052 baseline samples: median gold rank is 1 on every gold-labelled task; the largest
  measured loss is abstention (m3-robot refused 48% of questions and scored 56% when it answered; ATM-main
  refused 114 questions with all gold in the window); set-valued questions (ATM-Hard list_recall gold sets
  of 3–17 against a 12-item window) and sequence questions had no retrieval formulation.

## What landed (all opt-in; additions are union keys, never replacements)

1. Recall programs: `MemoryConfig.recall_planning` asks the generation backend for one small JSON plan
   (point, set, sequence, entity, composite) executed as bounded SQLite reads (`match_memories`,
   `memories_in_window`, `neighbor_memories`, identity lookup), unioned with id dedup, ordered
   chronologically, and always added on top of the question's own ranked window. A selectivity guard
   drops predicates that select more than 20% of the corpus (and more than four times `limit`);
   exhaustive rows are capped (`recall_set_max_rows`, 60) and budgeted (`recall_set_budget_chars`,
   30,000). The prompt states completeness so totals are licensed only for complete sets. Any planning
   failure falls back to today's path byte-for-byte. Activation is reported in `performance.recall`
   and per sample as `recall_shape`.
2. Answer policy `strict | best_effort` on `ask`, `ask_stream`, REST, MCP, and the harness
   (`--answer-policy`); the strict prompt is byte-identical to the previous release.
3. Identity-anchored keys on the write path behind the `vision:` slot: labelled description lines,
   a `[facts:<asset>]` section of durable statements distilled from stills plus the transcript, speaker
   name binding through the identity registry, and a retry for the gateway's transient constrained-JSON
   400. `vision_space` moved to v2.
4. MemLens evidence images (4,695 files, full coverage) attached as media parts by the adapter.
5. Harness: the lent generation proxy exposes the pool's real optional capabilities (planning had been
   silently inactive under the harness), the response cache keys on the answer policy, and recall
   activation counters are recorded.

## Measured outcomes

Dev slices, paired against the HEAD baseline run (`af98692e`) on the same questions. "S0" is the strict
control on the new code; "P1" is best effort; "R1/R2" add planning to S0/P1; "W1" adds the vision slot.

| Lever | Task (n) | Baseline | Arm | Paired delta [CI95] | W/L/T |
| --- | --- | --- | --- | --- | --- |
| best_effort (P1) | m3-bench-robot dev (314) | 0.2780 | 0.4551 | +0.176 [+0.134, +0.218] | 62/7/243 |
| best_effort (P1) | memlens-32k dev (60) | 0.3000 | 0.4667 | +0.167 [+0.083, +0.267] | 10/0/50 |
| best_effort (P1) | locomo-refined dev (525) | 0.7467 | 0.7619 | +0.015 [+0.003, +0.034] | 28/20/477 |
| best_effort (P1) | atm-bench-main-sgm dev (300) | 0.7210 | 0.7518 | +0.031 | 17/7/276 |
| best_effort (P1) | atm-bench-hard-sgm (31) | 0.1515 | 0.2007 | +0.049 | 4/0/27 |
| planning, strict (R1 vs S0) | atm-bench-hard-sgm (31) | 0.1555 | 0.2243 | +0.069 | — |
| planning, best effort (R2 vs P1) | atm-bench-hard-sgm (31) | 0.2007 | 0.2701 | +0.069 | 5/3/23 |
| planning, strict (R1 vs S0) | atm-bench-main-sgm (300) | 0.7093 | 0.6920 | −0.017 | 5/11/284 |
| planning, best effort (R2 vs P1) | atm-bench-main-sgm (300) | 0.7518 | 0.7619 | +0.011 | 10/6/283 |
| planning, strict (R1 vs baseline) | locomo-refined dev (525) | 0.7467 | 0.7448 | −0.002 [−0.006, +0.005] | 11/12/502 |
| planning, best effort (R2 vs P1) | locomo-refined dev (525) | 0.7619 | 0.7638 | +0.002 [−0.004, +0.009] | 12/11/502 |
| planning, best effort (R2 vs P1) | memlens-32k dev (60) | 0.4667 | 0.4667 | +0.000 [−0.050, +0.050] | 1/1/58 |
| planning, best effort (R2 vs P1) | m3-bench-robot dev (312) | 0.4551 | 0.4519 | −0.003 [−0.032, +0.025] | 9/10/293 |
| vision keys, best effort (W1P1 vs P1) | m3-bench-robot dev (313) | 0.4551 | 0.4505 | −0.003 [−0.043, +0.037] | 20/21/271 |
| vision keys, strict (raw media) | atm-bench-main dev (300) | 0.5959 | 0.5920 | −0.004 | 20/21/259 |
| vision keys, strict (raw media) | atm-bench-hard (31) | 0.2116 | 0.1932 | −0.018 | 3/4/24 |
| MemLens pixels, strict | memlens-32k dev (60) | 0.3000 | 0.2833 | −0.017 [−0.100, +0.067] | 3/4/53 |

Holdout confirmations of the best-effort policy (strict control and best effort on the same held-out
slice, same code):

| Task (holdout slice) | n | Strict | Best effort | Paired delta [CI95] | W/L/T |
| --- | --- | --- | --- | --- | --- |
| m3-bench-robot 25:75 | 962 | 0.2630 [0.227, 0.299] | 0.3867 [0.348, 0.426] | +0.124 [+0.095, +0.151] | 146/27/789 |
| locomo-refined 4:6 | 857 | 0.7036 | 0.7421 | +0.039 [+0.029, +0.048] | 52/19/786 |
| memlens-32k 60:135 | 75 | 0.480 | 0.680 | +0.200 [+0.093, +0.307] | 17/2/56 |
| atm-bench-main-sgm 300:713 | 713 | 0.6474 | 0.6870 | +0.040 | 48/19/646 |

Rejected by measurement: an always-on answer-shaping sentence (locomo 0.747 → 0.545 through abstention,
ATM-main-sgm −10.4, 5W/37L) was removed; thinking mode on the answer call (ATM-main-sgm −7.5, 19W/41L, 18×
latency; +3.1 on the 31-question Hard split only) is not enabled.

## Reading the results

- The best-effort policy is protocol alignment, not a memory change: every reference system answers
  every question, and abstained answers score zero under all five scorers. It is the only lever that
  meets the pre-registered rule (positive with CI excluding zero on a priority benchmark, confirmed on
  holdout, no guard regression). It stays a caller choice with `strict` as the default.
- Recall programs do what the design says (gold in the grounded set rose 64 → 79 of 194 on ATM-Hard, and
  every question where a set plan added gold improved) but pay only where set-shaped questions dominate.
  On the point-dominated benchmarks the planner chooses `point` for most questions and the result is
  neutral, at one extra generation call per question and, under best effort, up to 2.4× prompt tokens
  from replans. It does not meet the acceptance rule on a priority benchmark and remains off by default.
- Write-path keys move retrieval, not answers: on raw ATM-main, recall@12 +2.1 and any-gold@12 +2.7
  (16W/4L) with accuracy unchanged, because this reader already sees the raw media. They stay opt-in for
  text-only readers and lexical reach.
- Pixels alone do not help MemLens with this reader; the MemLens gain came from the policy.

## Defects found by measurement (each fixed with a failing test first)

The planner was inactive under the harness because the lent proxy hid `plan_recall` from the static
protocol check; set plans replaced the ranked window instead of adding to it; `entity` degraded to a
speaker-name match that selected half of a conversation corpus; over-long plan terms escaped `ask()` as
a bare `ValueError`; a cut similarity row cancelled the completeness statement; the name-binding reindex
flattened per-section keys; describe retries were counted as failed batches; the describer context was
unbounded; two tests passed vacuously.

## Limits

The judge is the answering model; ATM-Hard is 31 questions; the endpoint is not deterministic at
temperature 0 (planner-inactive replicates differed by ±2 points with W/L up to 8/1 at n=300); W1
naming rarely fires on m3-robot because dialogue seldom states names; ATM-Hard open-ended questions
scored 0 of 13 in every arm, including the baseline.
