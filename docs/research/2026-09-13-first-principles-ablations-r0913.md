# First-principles ablations — round r0913, 2026-09-13

Baseline code `599ae4b8` (`origin/mindbridge-v03` after PRs #186–#191), pinned at
`.benchmarks/worktrees/r0913-base`. Answerer and proxy judge `qwen3.8-27b`, embedder
`tencent/WeMM-Embedding-2B` (2048-d), `recall_limit 12`, `evidence_budget_chars 24000`,
`reinforce_on_answer false`. Every number is internal to this harness and judge, not comparable to a
published leaderboard row. **Artifact paths** are relative to
`autoresearch/orchestrator-260913-0024/`, gitignored (`.gitignore:22`) — its reports, scripts and
JSON live on the machine that ran the round, not in the repository. Said once here.

## 1. Question and method

Memory failure decomposes as: not encoded retrievably (representation) ∪ encoded but not retrieved
(ranking/selection) ∪ retrieved but unused (reader/policy) ∪ used but stale. Prior rounds closed the
reader/policy loss (`docs/research/2026-09-10-recall-programs-round.md`) and priced the evidence
window (`docs/research/2026-09-12-baseline-loss-decomposition.md`); the two open losses were
representation on the embodied/video route and set/count/sequence questions top-k ranking cannot
answer. This round tested only levers aimed at those two.

Five hypotheses were pre-registered in `PROTOCOL.md` before any code was written: H1 event-boundary
segmentation, H2 a STATE timeline emitted by the vision slot, H3 aggregate recall primitives, H4
closed-loop utility learning, H5 an `rtree` spatial index. Each carried a **Phase-0 offline
diagnostic** able to falsify it from existing stores and samples before any product code existed.
The panel is `mm-lifelong-day-test` (200), `m3-bench-robot` dev, `memlens-32k` dev 60 / holdout 135,
`locomo-refined` dev 525 / holdout 857, `atm-bench-hard-sgm` (31) — dev tunes, holdout confirms.

The acceptance rule was fixed before data: (1) paired per-question delta with cluster-bootstrap CI95
excluding 0 on the target, confirmed on holdout, against a ±2 pp noise floor; (2) no other family
regresses; (3) **activation counters must show the lever fired** — zero activation is an invalid
arm, not a neutral result (the r0910 lesson, where the planner ran 0 times and looked neutral); (4)
a declared cost budget; (5) no benchmark vocabulary in `src/mindbridge/` outside `benchmarks/`; (6)
fidelity before structure — every new key is a union addition; (7) opt-in unless 1–6 pass on ≥2
families.

Anti-reward-hacking controls: byte-identical answer-policy prompts in both arms, a fixed-length
merge control for H1, a selectivity guard for H3, blind and analytic random-ranker controls in every
retrieval table, prompt `sha256` recorded for every write arm, and every eval detached at ≤3
concurrent processes machine-wide. Separate adversarial reviewers with Bash access re-derived every
headline from the raw artifacts with their own scripts (`review/*.py`, none reusing an author
script) and reviewed each defect fix. Seven amendments were pre-registered mid-round, each before
the data it governs: 1 (Phase-0 verdicts, Phase-1 levers), 2 (H2 re-scoped to occurrence questions),
3 (H2 verdict), 4 (L3a verdict), 5 (L3c no-go, F1), 6 (F1 verified, F3), 7 (F3b).

## 2. H1 — event-boundary segmentation

**Claim.** Merging consecutive clips into events at embedding change-points, indexed as an extra
retrieval unit, raises gold coverage over the flat 30 s baseline *and* over fixed-length merges of
equal mean size (the control separating "bigger unit" from "event-aligned unit").

**Phase-0 diagnostic** (`reports/d1-h1-event-segmentation.md`): offline, zero generation calls.
Vectors from the ablation store's `embeddings` table (2,831 clips); the 200 query strings
re-embedded through the product's own `OpenAIModels.embed`; boundaries are local minima of
consecutive-clip cosine at `mean − k·std`. Every arm ranks *clips*, not units, so a bigger unit is
never rewarded for being bigger. Reproduction fidelity against the stored ranking `overlap@36 =
0.855`, `spearman = 0.894` (`d1/step1.json:baseline_reproduction.max_agg_vid`). Oracle arms
(`d1/step23.json:oracle`, n=200): `flat` 2,831 units, hit@12 89/200, hit@36 119/200, gold_cov@12
0.2074; `event_vid_k0.5/max` 521 units of 5.43 clips, 61/200, 96/200, 0.1866;
`fixed_matched_k0.5_size5/max` 567 units of 4.99 clips, 64/200, 100/200, 0.1859;
`event_vid_k1.0/max` 375 units of 7.55 clips, 61/200, 83/200, 0.1874. Paired bootstrap over those,
5000 resamples, seed 1234 (`d1/step23.json:paired_bootstrap`):

| a | b | metric | delta_pp | ci95_pp | W/L |
| --- | --- | --- | --- | --- | --- |
| event_vid_k0.5/max | flat | h12 | −14.0 | [−20.0, −8.0] | 8/36 |
| event_vid_k0.5/max | fixed_matched_k0.5_size5/max | h12 | −1.5 | [−5.0, 2.0] | 4/7 |
| event_vid_k1.0/max | flat | h12 | −14.0 | [−20.5, −8.0] | 9/37 |
| event_vid_k1.0/max | fixed_matched_k1.0_size8/max | h12 | +1.0 | [−3.0, +5.5] | 10/8 |

Gold-edge alignment is weakly above a uniform-placement null (precision 0.294 vs chance 0.196 at
`k=1.0, tol=0`, p ≤ 0.0035, `d1/step23.json:gold_edge_alignment`) — not "aligned". All ten question
types have flat ≥ event on hit@12 (`d1/step23.json:by_question_type_hit@12`), and `ref@300` rises
while `hit@12` falls only because bigger units concentrate the 12-clip budget into fewer 300 s
buckets. The union form is additionally a structural no-op: `parent_index_signals`
(`src/mindbridge/kernel/ranking.py:492-527`) takes `max` over every embedding key mapping to a clip,
and an event vector is at best a proxy for its members' own scores.

**Verdict: falsified.** The adversarial reviewer re-derived every oracle row byte-for-byte and then
attacked the *emission rule* — the author's `rank_clips` dumps all members of a winning unit into
the budget. Under five emission rules (`review/h1_recheck.py`):

| emission rule | event_vid_k0.5 hit@12 | vs flat (hit@12) | vs flat (hit@36) |
| --- | --- | --- | --- |
| authors (all members, unit order) | 61/200 | −14.00 [−20.0, −8.0] | −11.50 [−17.5, −6.0] |
| best_member (one clip per unit) | 82/200 | −3.50 [−6.0, −1.5] | −5.00 [−10.0, 0.0] |
| round-robin across units | 82/200 | −3.50 [−6.0, −1.5] | −5.00 [−10.0, 0.0] |
| topM units → rank members by own score | 88/200 | −0.50 [−4.5, +3.5] | −11.00 [−17.0, −5.5] |
| unit-max boost (λ=0.98 / 1.0) | 69 / 56 | −10.00 / −16.50 | −7.0 / −11.0 |

**Rating: CONFIRMED, magnitude overstated.** Strongest attack: the headline −14.0 pp is largely the
emission rule, not event units; the honest range is −0.5 to −3.5 pp on hit@12 and −5 to −11 pp on
hit@36. No rule at any `k` on any metric produces an event > flat CI excluding 0, and `event −
fixed_matched` includes 0 nearly everywhere, so the mechanism claim stays unsupported — the
falsification is **five emission rules, best reaches parity**. Salvage: change-points cost 1.3 ms
over 2,831 clips (`d1/step23.json:cost.changepoint_ms_2831_clips_numpy`) and compress 5.4–7.6 clips
per group — a write-path economy if any per-clip LLM lever had survived. None did.

## 3. H2 — state timeline from the vision slot

**Claim.** The vision slot emits STATE assertions (subject/predicate/value with `valid_from`) for
deltas between consecutive clips, so counting/first/last/order questions read a bitemporal timeline.

**Phase-0, first pilot** (`reports/d3-h2-state-timeline.md`, 6 questions, 25 clips, 51 new vision
calls): four text representations — transcript, caption, state-delta, union — scored **0/6 each**
under best_effort against the shipped raw-clip baseline's **1/6**. The `mm-lifelong` "Counting"
subset turned out to ask for numbers printed on a UI panel, so H2's mechanism was never loaded. Two
fidelity losses surfaced instead: q178's gold `153643` transcribed under the wrong field label
(`总血量`), and q193's gold token reaching the union arm's notes with the reader still wrong.

**Mechanism census** (`reports/d3b-h2-mechanism-census.md`, 0 model calls, all 200 day-test
questions classified from `metadata.reference_intervals`): `anchored_window` 67 (mean score 0.15,
abstain 0.93), `cross_clip_occurrence` 61 (0.08 / 0.89), `other` 38 (0.14 / 0.76),
`single_frame_reading` 34 (0.15 / 0.85). 44 of the 61 cross-clip occurrence questions have a gold
span wider than the reader's `video_limit` of 8, so raw-clip reading cannot structurally cover them.
Amendment 2 re-scoped H2 to that slice and pre-registered the rule: **E (occurrence log) must beat A
(caption) on ≥3 of 8, and A must beat T (transcript)**; otherwise H2 is falsified as a
representation lever regardless of cost.

**Decisive pilot** (`reports/d3c-h2-occurrence-pilot.md`, 8 questions × 4 arms, 33 clips, 96 of 100
attempted vision calls recorded):

every arm scores **0/8**, and 0 of 4 on the rule-derived count, at 273 (T, transcript), 3,393 (A,
the product caption prompt byte-identical), 1,208 (E, occurrence log) and 4,425 (U, union in one
call) mean notes chars per question. Shipped raw-clip baseline on the same eight: **1/8**
(`samples.jsonl` `metrics.answer_accuracy`); all 32 question × arm cells scored 0. Why the
occurrence log is empty of occurrences (`d3/arm_c_e.jsonl`, `d3/arm_c_u.jsonl`): 116 of 172 emitted
entities across both arms — **67 %** (E 59 %, U 77 %) — are generic nouns (`character`, `creature`,
`enemy`, `主角`), and no event named a death, a spell cast or a defeat. Against the pre-registered
count rules, on q34 the ACTION pattern matches 10 E-events and the ENTITY pattern 0; on q0 the
ENTITY pattern matches 4 and the ACTION pattern 0 — the halves never co-occur on one event, in any
arm. 30 of the 32 wrong cells are representation failures (no gold evidence in the notes), 2 are
partial (right field, wrong digits), **0 are reader failures**.

Per-clip cost: A 1,631 tok / 4.64 s, E 1,414 tok / 3.77 s, U 2,127 tok / 6.95 s — all inside
Amendment 2's ≤2,200 tokens and ≤7 s/clip (≤6.2 M tokens for the store): affordable, unpurchasable.

**Verdict: H2 failed its pre-registered pilot.** Both clauses of the falsification condition hold,
so the decision is correct and binding — but this is a **rule outcome, not a statistical
falsification**. **Rating: WEAKENED.** Strongest attack (`reports/adversarial-review-r0913.md` §3):
`P(0 wins of 8 | true rate 0.25) = 0.1001`, `P(fewer than 3 of 8 | 0.25) = 0.679`, and the one-sided
95 % upper bound on the true rate given 0/8 is **0.312** — consistent with E genuinely beating A up
to ~31 % of the time. The reviewer re-scored all 32 cells and agreed with 0 on every one, reproduced
the 30/32 representation split with its own token grep (`review/h2_notes_grep.py`), and noted that
the raw-clip control is confounded *against* the report: the baseline had 0 gold clips in its top-12
on 6 of 8 questions while the pilot arms had oracle retrieval, so it won q107 on less gold evidence
than any text arm. The mechanism finding survives the weakening and is the round's most transferable
result: **the vision model cannot name what it is looking at** — a perception failure upstream of
any representation.

## 4. H3 — aggregate recall primitives

**Claim.** `count / first / last / distinct / order` over predicate-selected sets inside recall
programs improve set/count/sequence subsets, at zero model calls beyond the planner's one.

**Phase-0 diagnostic** (`reports/d2-h3-aggregate-primitives.md`). H3's chain is *planner fires →
plan selects the full predicate set → completeness note licenses a count → reader answers*, and it
never breaks first at the aggregate op. memlens breaks at link 1 (the planner never fired —
`baseline.yaml:31-37` + `plugins.py:140`) and then link 2 (the gold-covering predicate is
non-selective: spot-checks return 68/117 and 49/116 rows against a ceiling of 48); mm-lifelong and
m3-bench-robot break at link 0, no predicate exists (`visual_descriptions` = 0 rows, ASR median 8
chars, action keywords hitting 0–2 of 2,831 clips); locomo breaks at link 1 only
(`r0912-a0-locomo/config.yaml:132` `recall_planning: false`, `plan_count = 0`).

A new `count`/`sum` op answers **0 of 34 memlens and 0 of 42 mm-lifelong** count questions
(`d2/feasibility.md`): memlens counts money inside free text or entities net of disposal, never
rows; mm-lifelong has no action text to predicate on. A live probe at HEAD showed the planner
firing, `shape=set ops=[match] complete=true`, answering a count correctly **with no aggregate op**
(`d2/activation.json`). So no op was built; the minimal lever was **L3a = turn the planner on**.

**L3a** (`reports/p2-l3a-memlens-dev.md`, `reports/p2-l3a-locomo-dev.md`). All four arms run code
`777fa3ee`, configs differing in `recall_planning` alone, `evaluation_sha256` matched:

| axis | memlens dev60 base | plan | locomo dev525 base | plan |
| --- | ---: | --- | ---: | --- |
| mean | 0.2833 | 0.3000 | 0.8019 | 0.7676 |
| paired delta | — | **+1.67 pp** [−3.33, +6.71], W/L/T 2/1/57 | — | **−3.43 pp** [−5.47, −1.20], W/L/T 12/30/483 |
| abstained | 19 | 20 | 22 | 39 |
| `plan_count` | 0 | 60, fallback 0, `non_selective_steps` 0 | 0 | 525, ops `similar 340, entity 174, match 27, neighbors 1, window 1` |
| evidence rows/q | 27.32 | 23.17 | 99.7 | 80.1 (`set` plans 26.5) |
| product tokens/q | 44,348.2 | 44,203.7 (−0.3 %) | 13,692.67 | 12,013.08 (**−12.3 %**) |
| `ask` latency mean ms | 2,678.1 | 3,745.1 (+39.8 %) | 4,354.93 | 4,880.56 (+12.1 %) |

**The memlens columns are void as the panel's memlens.** Both L3a memlens arms were launched with a
`--media-root memlens-32k=.benchmarks` workaround and silently ingested text-only
(`calls_by_input_modality` `{text: 190}`) against the later, valid `{image: 622, text: 985}`
(`reports/p3-f1-f2-verification.md` §6), so that pair reads only as *planner on versus off on a
text-only memlens*. It is inside the ±2 pp floor either way (the run's own
`noise_floor.minimum_meaningful_difference` is 16.3 pp at n=60); its count subset moved +5.26 pp on
n=19 carried entirely by one question (`q_cc5afaeb`), and count abstention went the wrong way, 3→4.
`non_selective_steps = 0` is a hard bound, so the pre-registered L3b trigger (≥20 % of memlens count
questions) is **NOT MET** and L3b was not built. Only 4 of 19 count questions have complete gold at
the depth the reader saw — window depth binds, not the op and not the guard.

On locomo no aggregate op ever fired and retrieval is unchanged (`recall@12` 0.8481 both arms,
`gold_in_window` 0.9658 vs 0.9649), so the loss is not retrieval: the plan **replaces** the reader's
evidence. Context shrank on n=195 (−8.72 pp), was unchanged on n=330 (−0.30 pp); 61 % of the loss is
new abstentions (20 new, 13 correct in base). Judge-noise control: 282 byte-identical predictions, 0
scored differently.

**Verdict: no aggregate op is needed; L3a is rejected under rule 2 and stays off by default.**
**Rating: CONFIRMED on memlens** (exact reproduction of +1.67 pp and W/L/T 2/1/57, plus a
numeral-level re-check of all 38 count answers finding 0 judge disagreements); **WEAKENED on
locomo.** Strongest attack (§6 of the review): **a 4-cluster bootstrap cannot support that CI** —
per-conversation deltas are −5.07 (n=138), −5.88 (n=136), −1.68 (n=179), +0.00 (n=72), and the df=3
t-interval is **[−7.59, +1.27] pp, which includes 0**. The mechanism holds instead: splitting on
whether the arms' top-12 ranked set is identical puts the entire loss on the *byte-identical* 477
questions (−3.98 pp) and leaves the 48 jittered ones positive (+2.08 pp), and the placebo row — the
314 questions where the planner kept all 100 rows — moves **−0.32 pp, W/L 4/5**. Adopted here: *sign
consistent across 3/4 conversations, 0/4 positive, mechanism isolated*, with rule 2's CI test
structurally unusable at K=4.

**L3c** (depth-by-shape widening, `reports/l3c-depth-by-shape-ceiling.md`) was sized **NO-GO**
before running: `evidence_budget_chars 24000` already refills the unplanned path (memlens base 27.32
rows/q, locomo 99.68), L3c's 30,000-char ceiling leaves memlens gold coverage at 15/23, its 60-row
cap *removes* gold from 5 of 58 locomo aggregate questions, and its predicted +0.56 pp needs ~10⁵
questions per arm for 80 % power on a 195-question task.

## 5. H4 and H5 — deferred, with reasons

**H4 closed-loop utility learning** (`reports/h4-affect-eval-sizing.md`): deferred for lack of a
target. Zero benchmark adapters carry an emotion/affect ground-truth field; ES-MemEval's upstream is
EvoEmo but the adapter drops the per-turn emotion labels at ingest
(`src/mindbridge/benchmarks/es_memeval.py:117-121`), and `docs/affective-memory.md:227` already says
no affect benchmark exists. No kernel primitive aggregates affect over time (`grep -rn
"trajector\|change.point\|baseline" src/mindbridge/kernel` → nothing). Cheapest real path: an
adapter change recovering those labels for ~30–50 subject/topic pairs. No mechanism work.

**H5 `rtree` spatial index** (`reports/h5-spatial-sizing.md`): deferred for lack of a slice.
`place_id` is a real SQL filter, but metric `near`/`radius_m` is a per-row Python compare
(`_selection.py:159-200`); `rtree` is compiled into the venv and unused; and **no adapter in
`src/mindbridge/benchmarks/` constructs a `SpatialContext`, sets `place_id`, or passes
`near=`/`radius_m=`**. An rtree swap is latency-only on a path no benchmark exercises, so rule 1
cannot accept it. Prerequisite: an adapter emitting `SpatialContext` (OpenEQA's spatial category)
plus a micro-benchmark at ≥10⁵ posed rows.

## 6. What the round found instead — six results that outlive the five hypotheses

1. **On the embodied route the binding constraint is perception, not representation.** 30 of 32
   wrong cells in the occurrence pilot had no gold evidence in the notes at all; 67 % of emitted
   entities are generic nouns; transcribed numerals land under the wrong label or with wrong digits.
   Until a per-clip representation carries a named entity and a correct numeral, nothing above it —
   H2's timeline, H3's ops — has anything to aggregate. The only levers that move it are a stronger
   vision model and frame budget (a model choice, not a memory mechanism), or letting the reader
   watch more raw clips, which is a retrieval-completeness problem.
2. **The read side violated the round's own Fidelity-Before-Structure rule.** A recall plan's
   grounding floor was built as "exactly what the unplanned path grounds" but without the caller's
   `evidence_budget_chars`, so the floor was `limit` rows while the unplanned path keeps every row
   the budget pays for (`kernel/answering.py:967-974`) — structured selection *replaced* similarity
   evidence instead of adding to it. The plan drops ≥1 base evidence row on 366 of 524 locomo
   questions (mean 29.1 lost, max 88); 29 of the 30 paired losses and all 20 new abstentions sit on
   shrunk windows; restoring those 29 to base scores moves the slice to 0.8015 against base 0.8019 —
   the −3.43 pp is **entirely** replacement. Fixed as **F1** (`4e879a4a`).
3. **Completeness was vacuously claimed for similarity-only programs.** `RecallResult.complete`
   quantified over exhaustive reads alone, so a plan that made none left the quantifier empty and
   returned `True`; the reader's note then said "a count or a list over those records is complete"
   about a similarity top-12 — the exact licence the note exists to withhold. With `similar` on 43
   of 69 memlens steps and `p50(exhaustive_rows) = 0`, ≥30 of 60 plans took that path. Fixed as
   **F2** (`d3f35bed`). The reviewer's probe (`review/completeness_probe.py`) reproduced the note
   verbatim and showed it did not cost accuracy here (count questions: base 3 correct / 13
   confidently wrong / 3 abstained, plan 4 / 11 / 4) — a defect, not the cause of the neutral.
4. **The planner is token-neutral or token-negative but always latency-positive.** memlens: product
   tokens −0.3 %, `ask` latency +39.8 %. locomo: product tokens −12.3 % (the plan narrows the
   reader's context, the opposite sign to Amendment 1's declared +874 tokens/q), `ask` +12.1 %,
   search p95 +55.1 %. No axis clears rule 4's ≥10 % cut, and that token saving is bought with
   evidence the reader needed; F1/F3 later flip the sign to +11 to +15 % (below).
5. **Four tooling defects that would have poisoned later rounds**, all found in Phase 0 and fixed
   before any A/B (`tools/README.md`): `retrieval_replay.py` flattened MemLens's nested
   `evidence_groups` through a scalar-only helper, so a group stringified as `"['turn_a',
   'turn_b']"` — truthy, never raising, never matching — and **reported recall 0 for every MemLens
   question, forever** (now uses the harness's own `gold_source_groups` and `_random_group_hit`,
   with a nested-group self-check). `preflight.py` compared the raw comma-split `run.tasks` against
   concrete task rows, so a config naming a *group* reported it missing though every task under it
   was present (now calls `task_catalog.expand`). `preflight.py` passed a pydantic `SecretStr` to
   the raw OpenAI client, and `str(SecretStr(...))` is the redacted `"**********"` — it
   authenticates with the placeholder and fails as a 401 that looks like a bad credential (now
   `.get_secret_value()`). And both panel configs declared an unconditional `run.media_root` for
   `m3-bench-robot`, making **every single-task launch of a different task** fail with `path
   override names an unselected task` (no opt-out exists); it moved into `launch.sh`'s own table.
6. **A silent ingest divergence made one panel leg meaningless.** The L3a memlens arms carried a
   `--media-root` workaround that resolved to no images, so they ingested text-only (`{text: 190}`
   embedding calls) while every later memlens run ingested the release images (`{image: 622, text:
   985}`); the stores differ in `input_sha256.memory` and the harness **did not fail on the missing
   media**. The F12/F3 memlens pairs share one image ingest and are valid. Images did not lift the
   base score — 0.2667 with images against 0.2833 text-only — and grounding rows/q fall 27.3 → 16.8
   because longer records fill the same 24,000-char budget.

Rule 5 was independently verified clean: `git diff 599ae4b8..HEAD` touches **zero lines under
`src/mindbridge/` outside `benchmarks/`** at the telemetry commit, and the D3/D3c write prompts
contain no game, benchmark, task or dataset vocabulary.

## Defect fixes F1–F3b and their verification

Six commits on `r0913/recall-activation` (`.benchmarks/worktrees/r0913-recall`), each with a failing
test first: `777fa3ee` telemetry (`performance.recall.ops`); `4e879a4a` **F1**, restore the caller's
`evidence_budget_chars` on the planned grounding floor; `d3f35bed` **F2**, a similarity-only program
is never a complete set; `fb52ff7c` the F1/F2 review fix-up (the owning reference
`docs/configuration.md`, a test whose 500-char budget never bound over a 71-char corpus,
`incomplete_count` gated on non-`similar` ops so it stops double-reporting `fallback_count`);
`9204cc9e` **F3**, the budget as a ceiling on the planned path too; `23f0b5dc` **F3b**, recount
`omitted` off the final grounded tuple, plus true-bound docs.

Pre-registered before each build: **Amendment 5** (F1/F2) — locomo plan arm within ±2 pp of base
0.8019 dev / 0.7725 holdout, memlens unchanged, tokens/q back toward base; **Amendment 6** (F3) —
accuracy ±2 pp, product tokens/q within ±3 % of base, shrunk-below-floor windows 0; **Amendment 7**
(F3b) — holdout abstentions fall back toward base's 40, accuracy ±2 pp, tokens inherited. Arms: base
= `recall_planning false`; pre-fix = the L3a plan arm; F12 = `4e879a4a`+`d3f35bed`+`fb52ff7c`; F3 =
`9204cc9e`; F3b = `23f0b5dc`. Sources `p3`, `p4`, `p5`: cluster t-CI is the df = K−1 interval the
adversarial review required in place of the K=4 bootstrap (`reports/p3-f1-f2-verification.md`,
`p4-f3-verification.md`, `p5-f3b-verification.md`).

| slice | arm | score | abstain | Δ vs base | bootstrap CI95 | cluster t-CI95 | W/L/T |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| locomo dev (525, K=4) | base | 0.8019 | 22 | — | — | — | — |
| | pre-fix | 0.7676 | 39 | −3.43 pp | [−5.47, −1.20] | [−7.59, +1.27] | 12/30/483 |
| | F12 | 0.7905 | 28 | −1.14 pp | [−1.87, −0.48] | [−2.48, +0.45] | 11/17/497 |
| | F3 | 0.7943 | 23 | −0.76 pp | [−3.75, +1.90] | [−5.76, +4.69] | 16/20/489 |
| | F3b | 0.8000 | 27 | −0.19 pp | [−2.08, +1.69] | [−3.65, +3.85] | 13/14/498 |
| locomo holdout (857, K=6) | base | 0.7725 | 40 | — | — | — | — |
| | F12 | 0.7643 | 56 | −0.82 pp | [−2.38, +0.81] | [−3.16, +1.77] | 20/27/810 |
| | F3 | 0.7596 | 56 | −1.28 pp | [−2.29, +0.00] | [−2.98, +0.77] | 23/34/800 |
| | F3b | 0.7643 | 55 | −0.82 pp | [−1.77, +0.12] | [−2.07, +0.57] | 19/26/812 |
| locomo pooled (1382, K=10) | F12 | — | — | −0.94 pp | [−1.90, +0.16] | [−2.14, +0.49] | 31/44/1307 |
| | F3 | — | — | −1.09 pp | [−2.39, +0.32] | [−2.55, +0.79] | 39/54/1289 |
| | F3b | — | — | −0.58 pp | [−1.46, +0.32] | [−1.63, +0.81] | 32/40/1310 |
| memlens dev60 image-ingest (60, K=60) | base | 0.2667 | 30 | — | — | — | — |
| | F12 | 0.2500 | 31 | −1.67 pp | [−5.00, +0.00] | [−5.00, +1.67] | 0/1/59 |
| | F3 | 0.2500 | 27 | −1.67 pp | [−5.00, +0.00] | [−5.00, +1.67] | 0/1/59 |
| | F3b | 0.2667 | 29 | +0.00 pp | [+0.00, +0.00] | [+0.00, +0.00] | 0/0/60 |

F3 and F12 score byte-identically on all 60 memlens questions (`p4` §1, W/L/T 0/0/60); that −1.67 pp
sits entirely on the identical-grounding subset (n=44) and is gone again under F3b — endpoint
nondeterminism, not the fix. Mechanism: shrunk windows 195/525 pre-fix → 2/525 (F12) → 2/525 (F3),
all ≤2 rows on questions whose base window was already under the cap; rows/q **99.7 base → 108.6 F12
→ 104.3 F3** on dev (holdout 99.4 → 109.3 → 104.5), `entity` plans 118.7 → 109.0; retrieval
unchanged in every arm (`recall@12` 0.8481, `gold_in_window` 0.9658). Cost, product tokens/q vs
base: F12 **+15.1 %** dev / +16.1 % holdout; F3 **+10.90 %** / **+11.28 %**; memlens F3 +2.28 %. A
token model fitting to 0.4 % (`p4` §3) splits F3's dev overshoot into ≈ **+6.3 pp planner call** and
≈ +4.6 pp wider reader window — the planner call alone is 2.1× the whole ±3 % budget.

Stated plainly: **the accuracy prediction was met; the cost prediction failed and cannot be met.**
Every locomo cut sits inside the ±2 pp floor with its cluster t-CI spanning 0, and the pre-fix
mechanism (shrunken windows, 61 % of the loss in abstentions) is closed; but the second LLM call is
structural, so no union policy reaches ±3 %. What the fixes buy is an honest opt-in feature: the
planned grounding is a union, not a replacement (F1); completeness is claimed only after an
exhaustive read ran (F2); the budget is a ceiling on both paths, not a second budget on top (F3);
and "further matched records are not shown" is counted off the final grounding (F3b).

**Amendment 7's abstention prediction is NOT MET** — holdout goes 56 → **55** against base's 40, 18
new and 13 correct in base (`p5` §3) — and the eviction hypothesis it rested on is **falsified**: 0
of those 18 lost a gold evidence row. The residual is planner-order dilution, the best gold row
sinking from rank 10.4 to **31.5** inside a 100 → 114 row window against a stayed-answered control
of 2.6 → 14.3 (`p5` §8, §8c). The `omitted`-note fix is unmeasurable at this n: F3 → F3b answers are
byte-identical on 340/525 dev against a base-vs-F3b floor of 284/525, and telemetry never observed
F-1 because `incomplete_count` is gated on `program.complete` at read time, not the grounding-time
`omitted` the fix changed (`p5` §2, §6). **F-series axes vs base on text**: accuracy **MET** (pooled
−0.58 pp, every CI spanning 0); abstentions **NOT MET** (+37.5 % holdout); tokens **NOT MET**
(+10.79 % dev / +11.41 % holdout); latency **NOT MET** (+26.4 / +27.4 %, an upper bound — base arms
ran uncontended, each fix trio launched together). **L3a stays rejected as the default.**

**Code reviews.** F1/F2 (`reports/review-f1-f2.md`): F1 **REQUEST CHANGES, documentation only** —
the code closes the bug (a `set` plan grounded 10 rows / 2,375 chars pre-fix against the unplanned 8
rows / 1,865, *not* a superset; post-fix 16 rows and a superset), but the patch never reached
`docs/configuration.md`, the owning reference, where three sentences became false and the overrun
widened from "`limit` rows" to "a second whole budget" (1.89× the declared ceiling at defaults). F2
**APPROVE WITH NOTE**; both blocking items landed in `fb52ff7c`. F3 (`reports/review-f3.md`):
**REQUEST CHANGES.** The mechanics hold under 4,000 randomised cells (floor never trimmed, no
duplicate ids, no double charging, `point` plans byte-identical 36/36), but (F-1) the refill
re-admits matched rows *after* `omitted` is counted, so a plan grounding 70/70 matched rows still
tells the reader "10 further matched records are not shown … do not state a total" — a regression
against `fb52ff7c`, and the reviewer's candidate cause of the F3 holdout abstentions, which `p5` §8
then falsified; (F-2) the true ceiling is `max(evidence_budget, first plan row) + floor`, so
wherever `floor ≥ budget` F3 is byte-identical to the F1 defect it fixes; (F-3) a plan carrying a
`similar` step refills with no budget set. F-1 is what `23f0b5dc` fixes; the 2× media leak is open.

## 7. Cost ledger

Model calls per diagnostic; "generation" counts planner, answer, vision and judge calls.

| diagnostic | generation calls | embedding calls | wall clock | artifact |
| --- | --- | --- | --- | --- |
| D1 (H1 segmentation) | 0 | 200 query re-embeds | offline; change-point pass 1.3 ms over 2,831 clips | `reports/d1-h1-event-segmentation.md` |
| D2 (H3 chain) | 2 (planner `plan` 836+38 tok, answer `chat` 508+9 tok) | 10 writes + 1 query | seconds | `d2/activation.json` |
| D3 (H2 first pilot) | 51 vision (a 14, b 12, c 25) + 48 answers + 24 judge = 123 | 0 | vision 3.77–6.60 s/clip | `d3/arm_{a,b,c}.jsonl`, `d3/judge_best_effort.jsonl` |
| D3b (mechanism census) | 0 | 0 | offline | `d3/mechanism_census.json` |
| D3c (H2 occurrence pilot) | 96 vision recorded of 100 attempted + 32 answers + 32 judge = 160 | 0 | answers 35.1 s, rubric 55.9 s; vision 3.77–6.95 s/clip | `d3/judge_c.jsonl` |
| H4, H5 and L3c sizing; adversarial review | 0 | 0 | offline | `reports/h4-affect-eval-sizing.md`, `reports/h5-spatial-sizing.md`, `analysis/l3c_*.txt`, `review/*.py` |

Read-side tokens for the vision pilots: D3 24,015 + 22,072 answer and 15,373 judge; D3c 33,521 +
1,296 answer and 17,449 + 2,568 rubric. Eval runs, detached via `tools/launch.sh`, logs in `runs/`;
same-second launches contend, so wall rows carry that and tokens do not:

| run | n | score | wall s (log) | product tokens/q |
| --- | ---: | ---: | ---: | ---: |
| `r0913-l3a-base-memlens-dev` (text-only ingest) | 60 | 0.2833 | 110 | 44,348.2 |
| `r0913-l3a-plan-memlens-dev` (text-only ingest) | 60 | 0.3000 | 123 | 44,203.7 |
| `r0913-l3a-base-locomo-dev` | 525 | 0.8019 | 225 | 13,692.7 |
| `r0913-l3a-plan-locomo-dev` | 525 | 0.7676 | 258 | 12,013.1 |
| `r0913-l3a-base-locomo-holdout` | 857 | 0.7725 | 367 | 13,721.1 |
| `r0913-f12-base-memlens-dev` | 60 | 0.2667 | 114 | 48,824.1 |
| `r0913-f12-plan-memlens-dev` | 60 | 0.2500 | 152 | 50,059.3 |
| `r0913-f12-plan-locomo-dev` | 525 | 0.7905 | 279 | 15,763.4 |
| `r0913-f12-plan-locomo-holdout` | 857 | 0.7643 | 460 | 15,926.6 |
| `r0913-f3-plan-memlens-dev` | 60 | 0.2500 | 142 | 49,935.5 |
| `r0913-f3-plan-locomo-dev` | 525 | 0.7943 | 282 | 15,184.8 |
| `r0913-f3-plan-locomo-holdout` | 857 | 0.7596 | 445 | 15,268.8 |
| `r0913-f3b-plan-memlens-dev` | 60 | 0.2667 | 123.3 | 49,936.7 |
| `r0913-f3b-plan-locomo-dev` | 525 | 0.8000 | 274.3 | 15,169.7 |
| `r0913-f3b-plan-locomo-holdout` | 857 | 0.7643 | 439.0 | 15,286.3 |

## 8. What is not claimed

- **One video store.** Every H1 and H2 number comes from 2,831 clips of a *single continuous* gaming
  video (`ablation-mmlifelong-day-20260912`). Boundary statistics, compression ratios, entity-naming
  failures and the raw-clip control are properties of this footage and this vision model.
- **Small n, few clusters.** The two H2 pilots (n=8, n=6) are falsification-grade only: one question
  is 12.5 pp, a 0/8 tie excludes nothing above ±30 pp, and the binomial bound puts the true
  E-beats-A rate up to 0.312. `locomo-refined` dev is four conversations, so its bootstrap resamples
  four values and is anti-conservative far below the ~30-cluster rule of thumb — the locomo verdicts
  rest on the cluster t-CI, the mechanism split and the placebo arm. `mm-lifelong-day-test` and
  every `atm-bench-*` split are single-unit: point estimates only.
- **Remote embedder jitter.** Query embeddings are non-deterministic at ~1e-3 per component, which
  is why D1's recomputed `flat` arm (89/200 hit@12) differs from the stored ranking (81/200) and why
  0 of 525 locomo ranked lists are byte-identical between arms. Segmentation is query-independent
  and the jittered questions were *positive*, so jitter is excluded as a cause — but no number here
  is reproducible to the digit.
- **Judge is the answering model.** `qwen3.8-27b` scores its own family's answers; the judge-noise
  controls (byte-identical predictions never scoring differently) bound *within-run* variance.
- **Ingest identity was not enforced by the harness.** A per-task `--media-root` resolving to nothing
  produces a text-only store and a clean-looking run (§6.6); a paired number is only as good as
  `input_sha256.memory` and `calls_by_input_modality` being equal across arms.
- **Store health was not controlled for.** The parallel round found a store that crossed the
  auto-optimise threshold during ingest returning 39 % of the exact top-100 (`COORDINATION.md`,
  03:40). Day-test never crossed it and is exact; no `index_completeness` was recorded here.

## 9. Next round candidates — each with the falsification test it needs *first*, no promises

1. **Hinted-term OCR control on the vision slot.** Re-ask the product caption prompt on a gold clip
   with the question's own on-screen term supplied as a hint, at ~0 tokens. Falsification test: if
   the label still comes out wrong, the fix is a model or frame-budget choice, not a prompt change.
2. **Video-aware selection predicate for occurrence questions.** 44 of 61 occurrence questions have
   gold spans wider than `video_limit` 8. Falsification test: an oracle arm handing the reader every
   clip of the run — if it does not beat the shipped 1/8 baseline, the route reduces to perception.
3. **"Plan the selection, keep the window."** The placebo row (−0.32 pp where the planner kept all
   100 rows) says the planner's structure is free when it adds rather than replaces. Falsification
   test: hold `len(memory_ids)` at the unplanned value while planning, on locomo dev *and* a
   ≥20-conversation slice; if the regression survives, recall programs should be retired.
4. **A locomo slice with enough clusters to decide anything.** Rule 2's CI test is unusable at K=4.
   Falsification test: two same-code control replicates over ≥20 conversations to measure the real
   cluster-level noise floor, before any lever is judged against it.
5. **Reader abstention under wider-but-honest grounding.** 18 of F3b's new holdout abstentions are
   mechanism, 13 of them correct in base, and none lost a gold row — the gold row is buried instead,
   rank 10.4 → 31.5 (`p5` §3, §8c). Falsification test: re-order the planned grounding so plan rows
   follow rather than precede the ranked window; if the refusals persist, the loss is width itself.
6. **`similar + neighbors` is the next vacuous completeness.** `neighbors` is in `_EXHAUSTIVE_OPS`,
   so a plan anchored on a similarity top-k still reports `complete=True` and licenses a total
   (`reports/review-f1-f2.md` §F2.3). Falsification test: build the plan and read the note; if it
   licenses a count, F2's rule needs "the anchors were matched", not just "a read ran".
7. **Window depth on memlens, by characters not rows.** Only 4 of 19 memlens count questions have
   complete gold at the depth the reader saw. Falsification test: the depth-60 variant predicts
   +3.33 pp and needs ~2,969 questions/arm for 80 % power on a 195-question task — so first ask
   whether any affordable slice can detect it.
