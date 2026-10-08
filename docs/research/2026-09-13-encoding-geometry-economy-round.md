# Encoding geometry and memory economy round r0913b — 2026-09-13

Historical branch report, recovered from commit `74c73d44`. The aggregate-only setting described
below remains experimental branch code and is not supported by the current SDK. The later
[memory-dynamics review](2026-09-14-memory-dynamics-round.md) qualifies the ANN and compaction
measurements because the original stores had not been checked for the Zvec index-binding defect.
Those measurements have not been revalidated for this archive.

Pre-registration: `autoresearch/orchestrator-260913-0040/PROTOCOL.md`, hashed at
`autoresearch/orchestrator-260913-0040/PROTOCOL.sha256`
(`fc276bdab2e00181967db266439948dfa75fdc4622df11040951cc15a187d57e`) before any number was read.
Every arm registered there is reported here. Numbers are quoted from the worker reports under
`autoresearch/orchestrator-260913-0040/reports/` and their JSON artefacts; nothing is recomputed
and no number appears without the file it came from. Base commit `599ae4b8`, worktree
`mindbridge-memory-optimization-6c0c02`.

**Round hazard, recorded in amendment 9 because it invalidated a gate run before it was found.**
The shared `/home/yons/thomas/MindBridge/.venv` editable-installs `mindbridge` from the **main
checkout** (`site-packages/_editable_impl_mindbridge.pth` → `/home/yons/thomas/MindBridge/src`), so
`pytest`, `mypy` and any script run from a worktree without `PYTHONPATH=<worktree>/src` exercises
the main checkout's code against the worktree's tests. The signature is a branch's own new test
failing on a base-code value (`assert {300, 4096} == {2048}`). Every gate and every measurement in
this note set `PYTHONPATH` explicitly and proved it with
`python -c "import mindbridge; print(mindbridge.__file__)"`.

## 1. Why this round, and how it differs

The brief was mechanism, not answer policy. Score movements in r0910 came from the reader
(`best_effort`, the 24,000-character evidence budget) and r0912 closed the contextual prefix, the
associative hop and the contiguity prior. The sibling round `autoresearch/orchestrator-260913-0024/`
owns event segmentation, the STATE timeline and aggregate recall; this round never re-tested those.
What was left unmeasured was the geometry of the keys MindBridge writes and the economy of what it
keeps, so the protocol took two families and one read-side family, all **zero model calls**: no arm
here can learn a benchmark, and a null quality result still yields a cost result.

Framing came from a literature survey in the session scratchpad (`scratchpad/lit-survey.md`): pattern
separation (Marr 1971; Leutgeb et al. 2007; Bakker et al. 2008; review Yassa & Stark 2011),
novelty-gated encoding (Lisman & Grace 2005; Kumaran & Maguire 2007), rational forgetting and
pruning (Anderson & Schooler 1991), and retrieval-induced forgetting (Anderson, Bjork & Bjork 1994).
Its own audit says the mnemoverse library it sampled covers none of those, nor event segmentation,
the spacing and testing effects, encoding specificity, the generation effect, emotional modulation
or gist-versus-verbatim, so every row comes from primary literature and the library pages reporting
the vendor's own results are either retracted or carry no numbers. The fifth principle, fidelity
before structure, is this round's own guardrail: decision rule 5 forbids removing or replacing a raw
record or its aggregate key, so E1b removes projections only and E2b may only add a key.

Pre-registration was hashed at 01:00. Nine amendments were appended between 01:10 and 06:10, each
dated and each recorded before the arm it governs was read; they are the audit trail and section 4
reports them in order.

## 2. Hypotheses

Predictions are quoted verbatim from `PROTOCOL.md`; the primary metric is in bold there.

| ID | Lever | Pre-registered prediction |
| --- | --- | --- |
| E1a | `index_quantization` fp16 / int8 / rabitq on real WeMM-2B keys | "**recall@12 and complete@12 within the measured ANN noise floor** (r0912: 6/1377 rank-11/12 swaps); Zvec bytes −40 % (fp16) / −65 % (int8); search p50 not slower. If int8 moves recall@12 by > 1 pp on either text corpus → not a default." |
| E1b | Atom-key economy on video: drop the transcript key and/or the video-asset key, keep the aggregate | "Offline max-over-part replay on mm-lifelong day: **hit@12 with aggregate-only within −2 pp of all keys**; if so, rows −60 %, index bytes −60 %, ingest embedding calls per clip 3→1. If the transcript key carries ≥ 2 pp of hit@12, keep it and drop only the video-asset key. Text corpora unaffected …; longmemeval chunk keys (29.5 % of rows) get the same test at n=60 as a text check." |
| E2a | Corpus mean-centring of document and query vectors | "**hit@12 on mm-lifelong day up ≥ 3 pp**; locomo recall@12 not down > 1 pp. Mechanism check: gain must be larger where anisotropy is larger (video > text). Null → closed." |
| E2b | Local-novelty residual key, added as a union key | "**State Change / Event Tracking / Temporal hit@12 up on mm-lifelong day dev, confirmed on holdout**; point questions not down > 1 pp; locomo … not down > 1 pp. If gains appear only with β where d' ≈ −mean … the mechanism is falsified." |
| E3 | Diversity-aware fill of the visible window (MMR over the depth-100 pool) | "**complete@12 on locomo multi-gold up; gold cov@12 on mm-lifelong set/count/sequence up**, both with paired CI excluding 0 and holdout confirmation; single-gold recall@12 not down > 1 pp on any corpus; p95 search latency ≤ 1.2×." |
| E4 | Multi-view lexical keys (generated questions into BM25 only) | "**Not run this round**: violates the zero-call rule; recorded as the strongest literature candidate (doc2query) for a future round with a declared per-record budget." |
| E5 | ACT-R base-level activation as prune policy | "**Not run**: requires a real usage log; benchmarks are single-shot. Recorded." |
| E6 | Prospective-memory trigger table (exact predicates) | "**Not run**: needs an agent interaction log to measure fire-rate. Recorded as the survey's top design gap." |

Two levers were registered after Phase 0 opened and are reported with the same discipline: **E1c**
index compaction (amendment 3) and **E1d** partial-key discount (amendment 4).

## 3. Corpora, metrics, statistics, kill rules

Retrieval-only, no generation model, product ranking depth 100, post-hoc cutoffs at 12 and 36,
exactly as r0912.

| corpus | store | questions | role |
| --- | --- | --- | --- |
| mm-lifelong day | `.benchmarks/data/ablation-mmlifelong-day-20260912/…/unit-mrqxsx3umvzxi` (P0-video) and the 2026-09-11 baseline day unit (P0-video2, P2-pre) | 200; dev = even `question_id`, holdout = odd | primary video target |
| mm-lifelong week | `.benchmarks/data/baseline-qwen38-27b-wemm9b-20260911/…/unit-o5swk227orsxg5a` | 200, same split | independent video replication (amendment 1) |
| locomo-refined | `.benchmarks/r0912-ab/stores/a0-locomo` + `out/a0-locomo.jsonl` | 1,377; dev = 5 conversations, holdout = 5 | text target, regression check |
| longmemeval-s | `.benchmarks/r0912-ab/stores/a0-longmemeval` + `out/a0-longmemeval.jsonl` | 60 | text holdout, chunk-key check |

Metrics: recall@12, complete@12, recall@36, complete@36 on text; hit@12, hit@36, gold\_cov@12,
gold\_cov@36 on video. Cost axes: Zvec and SQLite bytes, index rows, embedding calls per record,
search p50/p95. Statistics: paired bootstrap on the per-question delta (10,000 resamples, seed 42,
clustered by unit where more than one unit exists), exact McNemar on binary metrics, W/L always
shown, corpora never pooled.

Kill rule after Phase 0: an arm whose dev delta on its primary metric has a paired CI including 0
**and** a point estimate below 2 pp is closed without building; an arm passing on dev must pass its
holdout before Phase 1. Of the seven frozen decision rules the ones that decided arms here are
rule 1 (dev and holdout, no other corpus regressing beyond noise), rule 2 (single-gold recall@12
never down more than 1 pp), rule 3 (E1 levers accepted at quality-neutral if bytes −30 % or
p50 −10 %), rule 4 (no benchmark vocabulary in `src/mindbridge/` outside `benchmarks/`) and rule 5
(fidelity: raw records and aggregate keys are never removed).

## 4. Results

### E2a — corpus mean-centring: corpus-dependent, closed

Prediction: video hit@12 up at least 3 pp, locomo recall@12 not down more than 1 pp, gain larger
where anisotropy is larger.

| corpus | split | metric | base → arm | Δpp | CI95 | W/L | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mm-lifelong day | dev | hit@12 | 45.00 → 14.00 | −31.00 | [−40.0, −22.0] | 1/32 | `reports/p0-video.md` |
| mm-lifelong day | all | hit@12 | 40.00 → 16.00 | −24.00 | [−31.0, −17.0] | 7/55 | same |
| locomo-refined | dev | recall@12 | 0.8558 → 0.8825 | +2.67 | [+1.98, +3.39] | 41/15 | `reports/p0-text.md` |
| locomo-refined | holdout | recall@12 | 0.8183 → 0.8406 | +2.23 | [+1.18, +3.69] | 46/23 | same |
| longmemeval-s | all | recall@12 | 0.9542 → 0.9169 | −3.72 | [−7.17, −0.83] | 0/5 | same |

Verdict: fails decision rule 1 on cross-corpus grounds and is closed (amendment 2, 01:25). The
anisotropy mechanism check passes in direction: mean within-unit pairwise cosine of part-0 vectors
is 0.512 on locomo against 0.286 on longmemeval, and the more anisotropic corpus gains while the
less anisotropic one loses. On video it is not null but catastrophic, and the diagnostic separates
the halves: centring documents only is +4.0 pp on the three-key control but +0.0 pp (CI [−3.5,
+3.5], W/L 6/6) on the E1b winner's key set (`p0/video/e2a_on_e1b.json`), so it was only suppressing
the transcript keys E1b removes outright, and centring the query is what costs the 24 pp. No
adaptive variant was registered: choosing it per corpus would be fitting on the holdout corpus.

### E2b — local-novelty residual key: exact no-op, closed

Prediction: State Change / Event Tracking / Temporal hit@12 up on day dev, confirmed on holdout,
with the mechanism falsified if the neighbourhood-only control moves as much.

Added as a union key the residual is a literal no-op: deltas are exactly +0.00 pp on every split
and metric at β 0.3, 0.5 and 0.7 on video (`reports/p0-video.md`) and exactly 0.00 pp on
longmemeval, with the worst locomo union arm at −0.04 pp (`reports/p0-text.md`). The mechanism
diagnostic says why: the fraction of (query, clip) pairs where the residual beats every raw key is
0.0003 at β = 0.3 and 0.0000 at β = 0.5 and 0.7 (`p0/video/results.json → e2b_residual_activity`),
because it stays nearly parallel to its own raw key (cos 0.985 / 0.938 / 0.822 on video, ≥ 0.83 on
text). The protocol's own falsifier fires on the control: a neighbourhood-only key moves the metrics
as much as the residual and in the same direction (−4.5 pp hit@12 on video, −10.21 pp recall@12 on
locomo), so there is no novelty-specific effect to isolate. Closed by amendment 2. The `replace`
diagnostic is the fidelity rule doing real work: −0.5 to −2.5 pp on video, −4 to −24 pp on text.

### E3 — diversity-aware fill: killed on both families

Prediction: locomo multi-gold complete@12 up and video set/count/sequence gold\_cov@12 up, both
with CI excluding 0 and holdout confirmation.

| corpus | best arm | dev | holdout | source |
| --- | --- | --- | --- | --- |
| locomo-refined, multi-gold complete@12 | rank relevance, λ = 0.85 | +1.69 pp, CI [0.00, +3.54], W/L 3/0 | +0.00 pp, CI [−1.57, +1.67], W/L 1/1 | `reports/p0-text.md` |
| mm-lifelong day, target types (n=83) gold\_cov@12 | λ = 0.85 | −1.66 pp [−4.2, +0.4] | −3.51 pp [−9.4, +0.8] | `p0/video/e3_target.json` |

Every other λ is strongly negative (locomo cosine relevance at λ = 0.5 is −30.51 pp on dev; video
λ = 0.5 is −11.49 pp on the target types over all 200) and longmemeval regresses at every setting.
The dev best fails both clauses of the kill rule at once: CI includes 0 and the point estimate is
under 2 pp. Killed on video (amendment 1) and on text (amendment 2), and the two failure modes
agree. On locomo the gold records for one question are semantically similar turns of the same
conversation, so a diversity penalty evicts exactly the records that should co-occur; on video the
gold clips at ranks 13 to 36 are near-duplicates of each other (median adjacent cosine 0.808), so
doc-doc de-duplication evicts gold before distractors. The only directionally positive cell is video
hit@36 at λ = 0.85, +3.5 pp CI [−0.5, +7.5]: diversity helps the wide window, never the visible one.

### E1a — index quantisation: quality-neutral, but bytes go up

Prediction: quality within the noise floor, Zvec bytes −40 % (fp16) and −65 % (int8), p50 not
slower.

Quality is neutral on all three modes, measured against a freshly rebuilt `none` so compaction is
not credited to quantisation (`reports/p0-quant.md`): fp16 +0.000 pp recall@12 CI [0, 0], int8
+0.048 pp CI [+0.000, +0.154], rabitq +0.000 pp with 1,377/1,377 byte-identical ranked-ID lists.
int8 perturbs the ranking heavily (1,376/1,377 depth-100 orders differ) without the perturbation
reaching a gold group at these cutoffs. The cost premise is refuted:

| mode | Zvec bytes vs `none-rebuilt` | p50 | rule 3 |
| --- | --- | --- | --- |
| none, as r0912 left it | 150.3 MB (+57 %) | 51.5 ms | n/a |
| none-rebuilt | 95.8 MB | 18.2 ms | n/a |
| fp16 | 147.4 MB (+53.9 %) | 17.1 ms (−5.7 %) | reject |
| int8 | 147.9 MB (+54.5 %) | 17.0 ms (−5.8 %) | reject |
| rabitq | 177.5 MB (+85.5 %) | 18.1 ms (−1.1 %) | reject |

Zvec 0.7 keeps `DataType.VECTOR_FP32` in every mode and writes a second quantised structure beside
the full-precision one; per-unit evidence on `conv-41` (1,094 vectors) is
`embedding.index.3.proxima` 8,957,952 B plus `embedding.qindex.4.proxima` 5,156,864 B under fp16.
The knob trades resident search footprint and distance-computation cost for disk, the opposite of
what E1a wanted. Rejected on rule 3, both branches, all three modes (amendment 3, 01:45). The narrow
positive statement that survives: on real 2048-d WeMM keys fp16 and int8 cost nothing in retrieval
quality at k = 12/36, so the knob is safe for a RAM-bound deployment that states the trade itself.

### E1b — atom-key economy: the "drop the transcript key" rule failed to generalise

Prediction: aggregate-only within −2 pp of all keys on day; if the transcript key carries at least
2 pp, keep it and drop only the video-asset key.

P0-video, on the 2026-09-12 ablation day store (3 keys, 7,055 in the index, median transcript 8
characters), found the opposite of the prediction's shape: `{agg, vid}`, dropping only the
transcript key, was +4.00 pp dev [+1, +8], +5.00 pp holdout [+1, +10], +4.50 pp over all 200 with
McNemar p = 0.0039 and W/L 9/0, and Language Content Recall (n=6) went 0.167 → 0.667. Amendment 1
(01:10) registered the reversal and required an independent store before anything was built.
P0-video2 ran that store: the 2026-09-11 baseline week unit, 6,266 clips / 24,271 embeddings, four
keys, median transcript 66 characters.

| arm | corpus | dev Δpp | holdout Δpp | all Δpp | W/L | source |
| --- | --- | --- | --- | --- | --- | --- |
| drop transcript key | week | −1.0 [−7, +5] | −7.0 [−15, +1] | −4.0 [−9, +1] | 9/17 | `reports/p0-video2.md` |
| drop transcript key | day (4-key) | +4.0 [+1, +8] | +3.0 [−1, +8] | +3.5 [+0.5, +6.5] | 8/1 | same |
| drop context key | week | +0.0 | +1.0 [0, +3] | +0.5 [0, +1.5] | 1/0 | same |
| aggregate only | week | +2.0 [−4, +8] | −2.0 [−10, +6] | +0.0 [−5, +5] | 13/13 | same |
| aggregate only | day (4-key) | +4.0 [0, +9] | +1.0 [−5, +7] | +2.5 [−1, +6.5] | 10/5 | same |

The day result replicates in direction and size across two independent ingests of the same video;
the week result reverses it. The decisive cell is Language Content Recall, 0.167 → 0.667 on day
(n=6) and 0.867 → 0.533 on week (n=15) under the same edit. The promotion table explains both with
one statistic: on day the transcript key promotes clips with a gold share of 0.0154 against
replacements of 0.0411, so dropping it wins; on week it promotes 0.0608 against 0.0652, a coin flip
on precision, while 65 week questions have the transcript key winning a gold slot against 10 on day.
The sign is set by the quality of the derived text in that corpus, which the writer cannot know at
write time. The "no separate key for derived text below N characters" variant is refuted twice:
length bins are non-monotone in both corpora, and the threshold sweep is an exact no-op on day for
N ≤ 30, because a key that loses the max for its own clip cannot change a ranking. Closed as a rule
by amendment 4 (02:05).

The text check supports the economy direction: dropping the 12,337 longmemeval chunk keys (29.54 %
of index rows over 6,059 records) changes all four cutoff metrics by exactly 0.00 pp with W/L 0/0,
while gold rank improves in 39 of 109 gold groups and worsens in 0
(`p0/text/results_e1b_longmemeval.json`). This is not inertness: the chunk key wins the record's max
in 97.4 % of chunked (query, record) pairs and the depth-100 rankings are identical for 0 of 60
questions; what changes is which non-gold records get promoted. Ceiling recorded: these records are
2,052 to 3,736 characters, so this says nothing about records longer than the embedder's input
limit. Aggregate-only survived as the arm that replicates, hit@12-neutral on both video corpora at
−73 to −74 % index rows and four embedding projections per clip down to one, with per-type costs
that violate rule 2 on week single-gold and are why it can never be a default. Amendment 5 (02:30)
registered the product-path confirmation before it was run; §5 reports it.

### E1d — partial-key discount: dev pass, holdout exactly zero

Registered in amendment 4 from the mechanism table: under `max(aggregate, best_atom)` a partial key
that matches by chance outranks the whole record, so a derived projection should route but not
outrank its source. Arms were `max(agg, best_atom − δ)` for δ in {0.01, 0.02, 0.05} and
`agg + α·max(0, best_atom − agg)` for α in {0.25, 0.5, 0.75}, chosen on day and week dev by a rule
printed before any holdout number was read.

α = 0.25 was selected (dev +5.0 pp hit@12 on both corpora). On holdout it is **exactly +0.00 pp on
both**, W/L 6/6 on week and 5/5 on day, and longmemeval is exactly 0.00 pp on all four metrics and
both strata (`reports/p0-fusion.md`). No other δ or α passes either: the best holdout hit@12 across
all six settings is +2.0 pp on week (δ = 0.02, CI [0.0, +5.0]) with day at +0.0/+1.0, every holdout
CI including 0. E1d is closed as a default (amendment 5) but not refuted: rule 2 passes (week +0.00,
day +6.25, longmemeval +0.00 at corpus level) and the mechanism prediction is confirmed on both
corpora, the discount promoting gold at 1.9× (week) and 1.6× (day) the rate at which it displaces
it, net gold slots +28 and +7, while the hard-gate control inverts that ratio (displaced 0.1094
against promoted 0.0625 on week at m = 0.02) and is negative on three of four holdout cells. The
honest deflation is that 96.5 % (week) to 99.0 % (day) of per-question outcomes coincide with
aggregate-only. What is new sits in seven week questions: α = 0.25 keeps half the Language Content
Recall aggregate-only destroys (0.600 → 0.733 against a 0.867 baseline) and repairs its week
single-gold regression (−3.33 → 0.00) while holding the wide-window gains (week hit@36 +7.5
[+3, +12], cov@36 +5.12 [+2.13, +8.11]). Carried forward as a Phase-1 note on the economy setting.

### E1c — index compaction: accepted, then re-scoped twice

Registered in amendment 3 from P0-quant's control arm, which found that rebuilding the index from
SQLite at `quantization=none` cut Zvec bytes −36 % and p50 −65 % (51.5 → 18.2 ms) at exactly
0.0000 pp on every metric. P0-compact measured `optimize()` against a `none` control on five real
stores, warm against warm:

| store | embeddings | Zvec bytes | Δ p50 net of control | quality | source |
| --- | --- | --- | --- | --- | --- |
| memlens-32k | 217 | −26.6 % | −13.6 pp | identical, 0/50 | `reports/p0-compact.md` |
| locomo `conv-47` | 689 | −45.1 % | −18.0 pp | identical, 0/50 | same |
| memlens-256k | 1,359 | −46.6 % | −45.2 pp | identical, 0/50 | same |
| week-test | 24,271 | −23.8 % | −21.8 pp | improved: exhaustive overlap@100 0.770 → 0.913 | same |
| month-train | 50,642 | −0.0 % | −3.0 pp | identical, 0/50 | same |

Accepted under rule 3 on four stores spanning two orders of magnitude, null on the fifth
(amendment 6, 03:05). month-train is the load-bearing negative: its 50,642-embedding ingest crossed
the existing `_AUTO_OPTIMIZE_FLUSHES = 64` bound, so it is already a single segment and `optimize()`
on it is a 0.00 s no-op. The gap is not a large-store problem but a "fewer than about 65,000 rows
written in one session" problem. Amendment 7 (03:35) re-scoped E1c to correctness on
P2-pre's finding that the shipped week store returns 39 % of its true neighbours; amendment 8
(04:30) demoted that again after D1 attributed most of the loss to `ef`. **Amendment 9 (06:10)
withdrew the demotion**, because the ef review measured the two levers together on the shipped week
index against gold (`reports/rev-ann.md` §3): compaction is worth +6.5 pp of hit@12 at ef 300 and
**+16.5 pp at ef 2048**, and the ef fix is worth +16.5 pp fragmented and **+26.5 pp compacted**.
They are multiplicative, and neither substitutes for the other. E1c's final classification is
therefore both at once: a correctness lever on graph-backed stores, and — independently, on the
stores that never grow a graph — the bound on multi-session byte growth (§5a), which is the half
measured on more than one store.

### E4, E5, E6 — recorded, not run

E4 (generated questions into BM25 only, doc2query) violates the zero-call rule and is recorded as
the strongest literature candidate for a future round with a declared per-record budget. E5 (ACT-R
base-level activation as a prune policy) needs a real usage log; benchmarks are single-shot. E6
(prospective-memory trigger table with exact predicates) needs an agent interaction log to measure
fire-rate and is recorded as the survey's top design gap. None was measured and none is claimed.

## 5. What shipped

Three branches off `599ae4b8`, **none pushed to any remote**, all three merged into this branch
with `git merge --no-ff` so their history survives: `92b630c8` (compaction, tip `7c1d0c50`),
`ea4a536d` (atomic-keys, tip `002f5931`), `a4814736` (ann-completeness, tip `ab30accf`). The merged
head is the union of the three diffs and nothing else — for every file each branch touched,
`git diff <other-tip> HEAD -- <file>` returns exactly the other branches' changes; no symbol is
duplicated that `599ae4b8` did not already duplicate; and all fifteen tests the three branches add
are present. Gates on each branch and on the merged head are in the table at the end of this
section.

### (a) `r0913b/compaction` — merge at close, and stop `reindex()` un-merging itself

Commits `53e18e05`, `2c37b9be`, `7dbdfb03`, plus the reviewer's `dd7a8082` and `7c1d0c50` (tip);
worktree `.benchmarks/worktrees/r0913b-compaction`. Product change is 32 lines across three files, of which
14 are comment (`reports/p1-compaction.md`). `ZvecIndex.optimize_if_needed` gained one keyword,
`minimum_flushes`, defaulting to the existing `_AUTO_OPTIMIZE_FLUSHES = 64`, so every flush-path
caller is byte-identical; `Projection.flush_pending()`, which `Memory.close()` already calls, ends
with `optimize_if_needed(minimum_flushes=2)`. `reindex()` gained a `_merge_segments()` call after
its final drain. Failing tests first: `test_close_merges_the_segments_a_session_leaves_behind` and
`test_reindexing_leaves_the_index_merged` were red on the unmodified tree.

The adversarial review (`reports/rev-compaction.md`) corrected two of the three claims that
justified the lever, and one correction inverts the headline:

- **The base already compacts week-test at close.** `_flushes_since_optimization` is seeded from the
  directory at open (89), above the existing bound of 64, so any session that writes and closes
  merges that store on `599ae4b8`; P0-compact's `none` arm never saw it because it never wrote, so
  `flush()` — which is what calls `optimize_if_needed()` — never ran. Both arms end at 1 segment /
  239.5 MB / overlap@100 0.9128, byte for byte. The lever's real band is 2 to 63 persisted segments.
- **The strongest result is the multi-session bound**, which nobody had measured. Five consecutive
  `open → add one record → close` sessions on the 481 MB month-train store grow it 480.8 → 510.9 MB
  in seven segments on base and hold it at 480.7 MB in one segment on the branch, for a flat 1.4 s of
  close; on locomo the same loop is 26.7 → 46.8 MB in five to nine segments against 9.1 MB in one at
  0.45 s. Base grows about 5 MB and one segment per one-record session until the 64-flush bound fires.
- **The `open → add 1 → close` guard did not guard anything**: its test ran one session on an empty
  directory, where the condition is never reached, and every session after the first does merge. The
  behaviour is right (1.4 s worst case against unbounded growth) so `minimum_flushes=2` stays; the
  test was replaced with `test_repeated_short_sessions_do_not_grow_the_index` (red on `599ae4b8`:
  `assert 3 == 1`) plus `test_close_reports_a_failed_merge_and_still_releases_the_store`.

Phase-2 table, real stores through the product `close()` path (`p2/compact/<store>-<arm>.json`):

| store | arm | segments after close | `zvec/` | close | p50 | p95 | overlap@12 / @100 vs exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| locomo `conv-47` (689) | base | 4 / 4 / 1 | 21.6 MB | 0.23 s | 16.8 ms | 50.2 ms | 0.3650 / 0.8162 |
| locomo `conv-47` | branch | 1 / 4 / 1 | 9.2 MB (−57 %) | 1.15 s | 10.5 ms (−37 %) | 14.6 ms (−71 %) | 0.3650 / 0.8162 |
| week-test (24,271) | base | 1 / 1 / 1 | 239.5 MB | 6.78 s | 29.5 ms | 36.6 ms | 0.8050 / 0.9128 |
| week-test | branch | 1 / 1 / 1 | 239.5 MB | 2.72 s | 30.0 ms | 35.7 ms | 0.8050 / 0.9128 |
| month-train (50,642) | base | 2 / 2 / 2 | 485.8 MB | 20.37 s | 37.4 ms | 99.3 ms | 0.7917 / 0.8962 |
| month-train | branch | 1 / 1 / 2 | 480.7 MB | 13.81 s | 34.1 ms | 76.7 ms | 0.7917 / 0.8960 |

The `reindex()` fix holds on a real store: on locomo, base leaves 19.3 MB in two segments
(+16 % against the 16.7 MB it started at) and the following `close()` does not clean it up, while
the branch leaves 9.1 MB in one segment (−45 %), both with `doc_count == 689` and zero document
embeddings made.

The review's one behaviour change, made at the lead's direction and shipped as `7c1d0c50`: the
close-time merge now requires **this session to have flushed at least once** as well as two or more
persisted segments, because the counter is seeded from the directory and a search-only session would
otherwise merge on the previous owner's debt. Not benign — under the older condition a search-only
session on the shipped week unit rewrote 314.1 MB into 239.5 MB in a 32.5 s close, silently changing
the rankings D1 was measuring; under `7c1d0c50` it costs 0.029 s and leaves every segment byte
untouched, while a one-record writing session on month-train still pays 1.37 s warm and still holds
one segment. Failing test first, `test_a_session_that_only_reads_leaves_a_segmented_store_untouched`.
Honest limit: MindBridge has no read-only open, so the embedded RocksDB stores write fresh
`.log`/`MANIFEST`/`OPTIONS` at every open on `599ae4b8` too; the tests assert segment bytes
unchanged, not mtimes. Handed back unfixed: `deployment.md` tells operators to call `Memory.close()`
from a shutdown hook that now holds a multi-second native merge, and whether `minimum_flushes`
should be raised to 4 to 8.

### (b) `r0913b/atomic-keys` — opt-in `index_atomic_keys`

Commits `4125c5c5` and `8739bca9`; worktree `.benchmarks/worktrees/r0913b-atomic-keys`. 233
insertions / 5 deletions across 12 files, of which 26 lines are `src/`
(`reports/p1-atomic-keys.md`). `MemoryConfig.index_atomic_keys` defaults to `True`, which is
today's behaviour; setting it `False` makes `Embedding.inputs()` return the aggregate alone on the
write path while the query path keeps its multi-key behaviour through `stored=False`. The flag
folds into the index recipe as a `-aggregate` suffix on the `context-keys-v12` segment, and
`known_metadata_upgrade` now takes the expected marker so a key-layout change re-embeds in both
directions while a quantisation or tokenizer change still rebuilds the index only. Default `True`
reproduces the existing recipe string byte for byte, which is measured rather than asserted:
`test_atomic_retrieval_keys_are_derived_by_default` pins the pre-change key set literally and was
written and run against the unmodified tree.

The measured product-path trade, quoted verbatim from `reports/p2pre-econ.md` §4 and §5 (200
questions per corpus, control = all four keys, candidate = `object_part 0` only, both arms rebuilt
from the same SQLite, identical cached query vectors):

| corpus | metric | control | agg-only | Δ | CI95 / note |
| --- | --- | --- | --- | --- | --- |
| day | hit@12, all 200 | 41.50 | 44.50 | +3.00 pp | [−1.0, +7.0], W/L 11/5 |
| day | single-gold hit@12 (n=32) | 31.25 | 37.50 | +6.25 pp | [−9.38, +21.88] |
| week | hit@12, all 200 | 61.00 | 60.50 | −0.50 pp | [−5.5, +4.5], W/L 12/13 |
| week | hit@36, all 200 | 71.00 | 77.50 | +6.50 pp | [+1.5, +11.5], McNemar 0.024 |
| week | gold\_cov@36 | 36.61 | 41.36 | +4.76 pp | [+1.34, +8.10] |
| week | single-gold hit@12 (n=30) | 43.33 | 40.00 | −3.33 pp | rule 2 violation, never a default |
| week | Language Content Recall (n=15) | 0.867 | 0.600 | −26.7 pp | documented cost |
| week | Event Tracking (n=6) | 0.500 | 0.167 | −33.3 pp | documented cost |
| day | Language Content Recall (n=6) | 0.167 | 0.667 | +50.0 pp | opposite sign, same key |
| day | Counting (n=40) | 0.550 | 0.500 | −5.0 pp | documented cost |
| day | Attribute Recognition (n=6) | 0.500 | 0.333 | −16.7 pp | documented cost |

Cost: index rows −73.3 % (day) and −74.2 % (week), store bytes −68.6 % and −64.9 %, search p50
−73.7 % and −66.9 %, and 3.75 to 3.87 embedding projections per clip down to 1. (`p2pre-econ.md`
§5 stated "17,772 of 34,874 document embeddings never made". Recounted from the store copies it was
measured on — `sqlite3 mode=ro&immutable=1`, `SELECT COUNT(*) FROM embeddings` over
`.benchmarks/r0913b/econ/{day,week}/{control,agg_only}` — the four copies hold 10,603 / 2,831 and
24,271 / 6,266 rows, so it is **25,777 of 34,874 (−73.9 %)**; 17,772 matches neither per-corpus term
(day 7,772, week 18,005) nor their sum. The unit is a *projection*, one vector and one `embeddings`
row per key, not an embedder call: `Embedding.embed_document_parts`
(`src/mindbridge/kernel/embedding.py:144`) embeds every key of every memory in an ingest page in one
`embed()` call, and `OpenAICompatibleBackend.embed` marks `request_count = 1` per batch, so dropping
keys shrinks each request rather than removing any. `reports/p2pre-econ.md` carries a dated
correction and `docs/configuration.md` now states the recount.) Both latency columns are brute-force
scans,
so they are the cost of scanning four times fewer vectors, not an HNSW claim. The day control
reproduced the shipped ranking 200/200 at overlap@12 1.0000; the week control did not, for the
reason in §6, so the week row is a clean rebuilt-versus-rebuilt A/B and not a replay of the shipped
week path. The free-deletion path for on→off (`DELETE FROM embeddings WHERE object_part >= 1` plus a
rebuild, zero model calls, what P2-pre did by hand) is documented as a future note, not built.

### Adversarial review of `r0913b/atomic-keys`

`reports/rev-atomic-keys.md`, separate agent, four fix commits on the branch: `072efc36`,
`5796cf6a`, `b614bc71`, `002f5931` (final tip). Verdict: the lever itself is correct — default
byte-identical, recipe-aware, query path untouched, no benchmark vocabulary — and **neither
high-severity defect is in the 26 lines the implementer wrote**; both sit in the shared upgrade path
that a one-character boolean made reachable.

| # | severity | what | fix |
| --- | --- | --- | --- |
| H1 | high, **fixed** `072efc36` | `Embedding.reembed_memories` rewrites keys one 32-memory page per transaction while `memory.py:277` advances `index.recipe` only after the whole loop, and `ensure_store_metadata` short-circuits on `stored == value`. `SIGKILL` on the third batch of a real `memlens-32k` unit left rows 151 → 113, atoms 51 → 13, the marker claiming the full layout, and **no future open ever repairs it** — 87 retrieval keys gone silently and permanently. Pre-existing: the same hole swallows an interrupted embedder-space swap. | park the recipe on `REEMBEDDING_IN_PROGRESS` from the first page that changes rows; failing test `test_an_interrupted_key_layout_reembed_is_not_trusted_as_a_finished_one`. Re-verified on the killed copy: the revert re-embeds 151 and lands byte-identical to pristine |
| H2 | high, **fixed** `5796cf6a` | `index_atomic_keys` was absent from `cli.py`'s `_TUNING`, so every CLI operation against an aggregate-only store built `Memory(index_atomic_keys=True)` — a full re-embed at real model cost, undoing the operator's −73 %, with no way to say otherwise. Reproduced at 151 and 183 document embeds on two tiny units; on mm-lifelong month it is 50,642 | one `_TUNING` entry, one `BooleanOptionalAction`, one kwarg, one docs row |
| M1 | medium, **fixed** `b614bc71` | the shipped docs table put `search p50 −73.7 % / −66.9 %` beside the byte rows, reading as a production latency win; both arms were brute-force scans | row relabelled `search p50 (brute force)` with the caveat |
| M2 | medium, **documented** `b614bc71` | "both directions re-embed" is not "both directions restore": off → on is a re-derivation from the stored row, and the media unit round-trips 230 → 61 → **183** rows (atoms 169 → 0 → 122). Pre-existing — identical on `599ae4b8` | `docs/configuration.md` says so and names the measurement |
| L1 | low, not fixed | `-aggregate` is matched as a substring of the whole recipe, not of the `context-keys` segment | noted; legacy sets are frozen literals |

Upgrade safety was measured on two real `user_version 18` units: default open is **0 document
embeds** in 0.14–0.16 s, rows and atoms unchanged, and `COUNT(*) WHERE object_part >= 1` is 0 in
every off state. The query path is byte-identical by measurement as well as by construction — the
recorded `EmbedTask.QUERY` inputs for one 5,400-character multimodal query hash to
`92cb0f565eba8928` on `599ae4b8`, on the branch by default and with the flag off. The full 28 × 8
recipe decision matrix was generated against the branch: no pair of markers silently mixes layouts.
Scale caveat: on 100-record units `zvec/` barely moved (6.0 → 6.0 MB, 5.5 → 5.4 MB) while rows fell
34 % and 73 %, so the −65 to −69 % byte figures are the two mm-lifelong corpora and nothing smaller.

### (c) `r0913b/ann-completeness` — `ef_search` 300 → 2048

Commit `8ead911d`; worktree `.benchmarks/worktrees/r0913b-ann`. 20 lines in
`src/mindbridge/infrastructure/local/zvec_index.py` including the measurement comment:
`_DEFAULT_EF_SEARCH` becomes `_MAX_EF_SEARCH = 2048`, the validator accepts `1 <= ef_search <= 2048`
and the query clamps `ef = min(2048, max(selected_ef, limit))`. `ef_search` is a constructor keyword
no caller passes and `ZvecIndex.search`'s `ef=`/`exact=` are used only by
`benchmarks/local_index_benchmark.py:239`, so the constant is the product's behaviour. The failing
test written first, `test_dense_search_asks_for_the_largest_candidate_list_zvec_accepts` in
`tests/unit/infrastructure/local/test_zvec_index.py`, fails on unmodified HEAD on the value
(`{300} != {2048}`), not on a missing symbol.

Timeline on the week store (`reports/d1-ann.md` §1, `d1/probe-week-*.json`,
`d1/product-sweep.json`):

| checkpoint | completeness | segments (ipc / proxima) | raw recall@100 ef 300 | ef 2048 | `is_linear=True` |
| --- | --- | --- | --- | --- | --- |
| shipped, as the ingest left it | 0.7277 | 89 / 26 | 0.5300 | 0.9772 | 1.0000 |
| after `Memory.optimize()` | 1.0000 | 2 / 1 | 0.4834 | 0.9540 | 1.0000 |

On the grouped product path, exhaustive top-12 within depth 100 goes 0.2500 → 0.7867 on the shipped
index and 0.3333 → 0.9033 after `optimize()`, for p50 2.3 → 7.5 ms; month-train raw recall@100 goes
0.706 → 0.940 at p50 1.70 → 5.98 ms; memlens-32k (217), locomo (689) and memlens-256k (1,359) are
1.000 before and after at unchanged p50, because they have no graph and an exhaustive scan does not
consult `ef`. Attribution is upstream ANN quality at the candidate list MindBridge chose, amplified
by insert order (§6), not our flush or optimise ordering: reconstruction arm 7 reproduces the
product cadence and the shipped state at recall 0.9556, and eight reconstructions land between 0.80
and 0.96 at ef 300 while the shipped store sits at 0.5300. Synthetic corpora, including one matched
to the real anisotropy, are 1.000, so a deterministic fake embedder cannot show this and the guard
pins the parameter rather than a synthetic recall number. A second defect fell out of the clamp:
`ef = max(ef_search, limit)` was unclamped, so a deep enough search raised a `RuntimeError` from
Gandiva on any store with a graph — the review found the real trigger much closer to hand (R1).

### Adversarial review of `r0913b/ann-completeness`

`reports/rev-ann.md`, separate agent, one fix commit `ab30accf` (final tip). Verdict: ship it, and
the case is stronger than the branch's own report made it. Three findings, all resolved:

| # | severity | what | fix |
| --- | --- | --- | --- |
| R1 | high (pre-existing; the branch fixes it, and undersold it) | `Memory.search(limit=100, scope=RetrievalScope(near=…, radius_m=…))` raises `IndexUnavailableError` on **any** graph-backed store at base. A metric radius is a post-filter, so no candidate survives and `retrieval.py:445` doubles `candidate_limit` toward `limit × 129`; the third doubling asks 3,200 > 2,048 and Zvec refuses. Measured on 5 week questions: base **5 / 5 errors**, branch 0 | the branch's clamp; the review supplied the evidence, since the committed unit test covers the arithmetic on a fake collection |
| R2 | medium, **fixed** | the constructor validator was changed — new bound, new message — with **no test before or after** | `test_ef_search_is_rejected_outside_the_range_zvec_accepts`: `0`, `-1`, `2049`, `True` all raise; red on base's message |
| R3 | low, **fixed** | the 12-line justification comment attributed a post-`optimize()` recall (0.48) to "the shipped store" (0.53), quoted a raw-index p50 pair that overstates the product cost ~4×, and carried an internal round label with no precedent in `src/` | rewritten with product-path and scaling numbers |

Phase 2, quality: 200 recorded week prompts through `Memory.search(limit=100)`, shipped Zvec
directory copied verbatim and never rebuilt, cached query vectors, `document_inputs == 0` in every
arm (`rev-ann/results.json`):

| arm | `ef` | index | hit@12 | hit@36 | cov@12 | cov@36 | p50 | p95 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| the 2026-09-11 run's recorded ranking | — | — | 0.415 | 0.490 | 0.1651 | 0.2359 | — | — |
| base | 300 | as shipped | 0.235 | 0.320 | 0.1070 | 0.1574 | 24.12 ms | 32.93 |
| branch | 2048 | as shipped | **0.400** | 0.485 | 0.1833 | 0.2598 | 30.71 ms | 40.00 |
| ceiling, `exact=True` | — | as shipped | 0.430 | 0.535 | 0.1955 | 0.2902 | 53.25 ms | 61.99 |
| base + `optimize()` | 300 | compacted | 0.300 | 0.365 | 0.1181 | 0.1728 | 24.43 ms | 31.99 |
| branch + `optimize()` | 2048 | compacted | **0.565** | **0.650** | 0.2332 | 0.3303 | 33.39 ms | 44.22 |
| P2-pre, full rebuild from SQLite | — | rebuilt | 0.610 | 0.710 | 0.2542 | 0.3661 | — | — |

Paired bootstrap, 10,000 resamples, seed 42: base → branch **+16.5 pp [11.0, 22.0]**, W/L 36/3,
McNemar < 10⁻⁴; dev (even qid) +18.0, holdout (odd qid) +15.0, single-gold (n=30) +10.0
[−3.3, 23.3]; the remaining ANN residual against `exact` is +3.0 [1.0, 5.5]. No question type loses
in either configuration. **`base(shipped)` → `branch(opt)` is +33.0 pp [26.0, 40.0], W/L 70/4** —
this is the multiplicative result that withdrew amendment 8's demotion of E1c, and the reason the
published week baseline of 0.415 is recovered only by the pair, never by the ef fix alone (0.400).
Two corrections to `d1-ann.md` came with it: "base ≈ the shipped 0.415" conflated the 2026-09-11
run's *recorded* ranking with a replay through base code against the same index (0.235), and the
shipped index's exhaustive ceiling is 0.430, not 0.610, so the branch captures 85 % of what is
there and the residual to 0.610 is fragmentation, not ANN. The flat day store is unaffected:
**200/200 identical depth-100 hit ids**, every metric equal to four decimals, p50 37.8 → 37.3 ms.

Phase 2, latency: unit-norm isotropic 2048-d vectors through the product's own `ZvecIndex`, warm,
`limit=100`, 100 queries (`rev-ann/scaling.json`): p50 6.34 → 14.37 ms at 20k, 7.72 → 19.78 at 50k,
8.02 → 29.00 at 100k, **resident memory unchanged within 1 MB at every size**. `exact=True` at 20k is
68.8 ms, 4.8× ef 2048 and growing linearly. Relevant to E1c: `optimize()` on the same corpora costs
32 s / 158 s / 324 s, which is what optimise-at-close pays at scale. Above 10⁵ records nothing is
measured — the 40–50 ms suggested for 10⁶ is extrapolation on a synthetic manifold.

The strongest objection — *2048 is a ceiling, not a principle* — is conceded and does not block:
recall is monotone over every value d1 sampled, the whole remaining ANN loss is 3.0 pp, the cost is
bounded above, and a future Zvec that moves the bound either leaves recall on the table (a one-line
change) or fails loudly at query time. `_MAX_EF_SEARCH = 2048` is a C++ check in `hnsw_index.cc`,
declared in no Zvec `.pyi`, and **not guardable cheaply**: synthetic collections to 16k vectors
reach `completeness 1.0` yet still accept `ef=4096`, because Zvec enforces the bound only when it
descends the graph. Recorded, not built. One operator paragraph went into `docs/operations.md`; a
`troubleshooting.md` diagnostic deliberately did not, because `index_completeness`, `doc_count` and
`exact=True` are on no operator surface, so the entry would end with no action.

### Cost and quality across the three branches

| branch | product change | quality | cost | status |
| --- | --- | --- | --- | --- |
| `r0913b/compaction` (tip `7c1d0c50`) | 32 lines, 3 files; merge at close when this session flushed and ≥ 2 segments persist, merge after `reindex()` | ranking identical on locomo and month-train (overlap@12 equal, @100 within 0.02 pp); on the graph-backed week store, gold hit@12 +6.5 pp at ef 300 and **+16.5 pp at ef 2048** | close +0.45 s (21 MB) to +1.4 s (481 MB) per writing session, 0 s for a reading one; `zvec/` −45 to −57 % in its band, unbounded multi-session growth stopped | default on, no setting |
| `r0913b/atomic-keys` (tip `002f5931`) | 26 lines of `src/`, one strict-bool setting, recipe-aware, plus two pre-existing upgrade-path defects fixed | day hit@12 +3.0 pp, week −0.5 pp, week hit@36 +6.5 pp; week single-gold −3.3 pp and two per-type cliffs | index rows −73 to −74 %, store bytes −65 to −69 %, 4 → 1 embedding projections per clip | opt-in, never default |
| `r0913b/ann-completeness` (tip `ab30accf`) | 20 lines, one constant plus a clamp | week gold hit@12 0.235 → 0.400 as shipped and 0.565 compacted; month-train recall@100 0.706 → 0.940; flat stores byte-identical; a scoped search at the default limit stops raising | search p50 24.1 → 30.7 ms on the real graph store, +8 ms at 20k and +21 ms at 100k synthetic, RSS unchanged, 0 on the other 1,560 indexes | default on, correctness fix |

Gates, each with `PYTHONPATH` proven to select the tree under test: `ruff format --check` 272 files,
`ruff check` clean, `mypy` 215 source files, `git diff --check` clean, markdownlint-cli2 v0.23.0
55 files 0 errors and lychee 0.23.0 0 errors on every branch and on the merged head. `pytest -W error`:
2194 on `r0913b/compaction`, 2193 on `r0913b/atomic-keys`, 2189 on `r0913b/ann-completeness`, and
**2202 passed in 133 s on the merged head** — exactly the base 2,187 plus the 7 + 6 + 2 tests the
three branches add, with nothing lost to the merge.

## 6. Defects and measurement facts found

- **`index_quantization` adds bytes.** Zvec 0.7 writes the quantised structure beside the fp32
  vectors rather than replacing them: +53.9 % (fp16), +54.5 % (int8), +85.5 % (rabitq) on `zvec/`
  over ten locomo units. `docs/operations.md:137` is qualified on `r0913b/compaction`.
- **`reindex()` leaves the index worse than doing nothing.** It queues one outbox row per embedding
  as a crash checkpoint, rebuilds from SQLite, then re-drains those rows over the index it just
  built with the flush counter reset, producing `ceil(N/1024) + 1` segments; predicted and matched
  on all five stores (217 → 2, 689 → 2, 1,359 → 3, 24,271 → 25, 50,642 → 51). On month-train that is
  Zvec 480.8 → 1,168.2 MB (+143 %) and p50 32.5 → 254.5 ms (7.8×), in the operation
  `docs/troubleshooting.md` recommends for index repair. Fixed by merging after the replay.
- **`index_completeness` is not a health metric.** It is the share of documents sitting in a built
  HNSW graph: 0.0 means no graph exists and every query is an exhaustive scan, the state with
  perfect recall on this corpus, and `optimize()` raises it 0.7277 → 1.0000 while *lowering* raw ANN
  recall@100 0.530 → 0.483. A store can lose half its true neighbours with both `doc_count` and
  `index_completeness` reading healthy, and `doctor` reports neither, because it never opens a store.
- **The harness embeds the prompt wrapper, not the question:** on the text corpora the embedded
  string is `"Answer concisely using only the memories.\nQuestion: …\nAnswer:"`
  (`p0/text/dump_questions.py`), shared by every arm but not the bare question.
- **The r0912 stores are local schema v19 and HEAD reads v18**, so P0-quant ran against a read-only
  `git archive` export of `05567858`, whose `zvec_index.py` is identical to HEAD's; the 2026-09-11
  baseline stores used by P0-compact and P2-pre are v18 and ran on HEAD's own `src/`.
- **Only three indexes on disk have a graph**, of 1,563 scanned under `.benchmarks/`: mm-lifelong
  week-test, month-val and month-train. Everything else — locomo, longmemeval, memlens, m3 and
  personamem included — was measured by exhaustive scan. The discriminator used (a `.proxima` larger
  than the uniform 5,017,600-byte flat segment) is dimension-specific and should not be reused as a
  general test; `index_completeness > 0` is the direct signal and costs nothing.
- **ANN recall is insert-order sensitive.** The same 24,271 vectors written in the store's real
  `created_at` order rather than by embedding id cost 13 pp of recall@100 at ef 300 (0.9556 → 0.8244)
  and drive worst-query recall to 0.00: adjacent 30-second clips are near-duplicates, so the graph is
  built along a chain.
- **The shipped mm-lifelong-week retrieval baseline is an index-state artefact.** Its recorded
  top-36 holds 64.9 % of the exact dense top-12; the same store and queries rebuilt give hit@12
  0.610 / hit@36 0.710 against the published 0.415 / 0.490, and mm-lifelong month train and val have
  never been read with a correct index at all. Two figures must not be conflated, and `d1-ann.md`
  did: the 2026-09-11 run's *recorded* ranking scores 0.415, while replaying the same queries through
  today's code against the same index scores 0.235. On the merged head that store answers 0.400
  untouched and 0.565 after `optimize()` (§5c).
- **Dev and holdout are by question parity on the video corpora** (even `question_id` is dev) and by
  alternating conversation on locomo, frozen in each worker's script before any arm was read.
- **Four further defects were found by the two adversarial reviews and are recorded in §5**: an
  interrupted key-layout re-embed accepted as finished (silent, permanent, pre-existing); the CLI
  re-embedding an aggregate-only store at model cost; a spatially scoped search raising
  `IndexUnavailableError` at the default limit on every graph-backed store; and the off → on key
  round trip not being row-count-reversible (230 → 61 → 183). The first three are fixed on the
  branches merged here; the fourth is documented.

## 7. Residuals and next round

- **A size-conditional no-graph policy.** ef 2048 is Zvec's ceiling and still leaves about 5 pp of
  raw recall@100 and about 10 pp of the product top-12 metric on the merged week store. The only
  exact configuration is `is_linear=True`, measured at 17.15 ms p50 on 50,642 documents raw and
  39.4 ms through the grouped path on 24,271, so 5× the ef-2048 cost and growing linearly where ef
  does not. "Do not build a graph below N documents" is the principled version, bigger than this
  round should make, and accidentally what every store under the 64-flush bound already does.
- **doc2query into BM25 (E4).** Generated questions in the lexical index only, never in the vector
  payload and never in the returned text, with a declared per-record call budget and precision@k as
  the guard metric.
- **Prospective-memory trigger table (E6)**, exact predicates evaluated deterministically rather
  than by similarity; it needs an agent interaction log to measure fire-rate.
- **ACT-R prune policy (E5)** needs a real usage log; the cheap first check is whether its ordering
  differs from the current recency decay at all.
- **α ≈ 0.25 stays attached to aggregate-only.** If a future round wants the economy setting's
  −73 % rows on video while keeping atom keys, the partial-key discount halves the Language Content
  Recall cost and removes the week single-gold regression; it is a note on the E1b setting, not a
  lever, because 96.5 to 99.0 % of its per-question outcomes coincide with aggregate-only.
- **Owed.** A control-plane surface for index health — segment count, index bytes,
  `index_completeness` on `health()`/`doctor`, which needs two directory walks and no store open.
  Without it the "graph-backed store with low ANN recall" troubleshooting entry cannot end in an
  action, which is why the ef review declined to write one.

## 8. Reproduction

Pre-registration and audit trail: `autoresearch/orchestrator-260913-0040/PROTOCOL.md` (amendments 1
to 9 appended in place), `PROTOCOL.sha256`. `reports/p2pre-econ.md` carries a dated correction to its
own §5 embedding-count arithmetic; `reports/integration.md` records the merge.

| worker report | scripts | artefacts |
| --- | --- | --- |
| `reports/p0-video.md` | `p0/video/run.py` (`--selfcheck`), `e3_target.py`, `e2a_on_e1b.py` | `p0/video/results.json`, `per_question.npz`, `e3_target.json`, `e2a_on_e1b.json` |
| `reports/p0-text.md` | `p0/text/step0_verify.py`, `dump_questions.py`, `embed_queries.py`, `extract_vectors.py`, `common.py`, `step1_replay.py` … `step6_diag.py` | `p0/text/results_*.json`, `queries_{locomo,longmemeval}.npz`, `vectors_*.npy` |
| `reports/p0-quant.md` | `p0/quant/run_mode.py`, `compare.py`, `src-05567858/` export | `p0/quant/{none,none-rebuilt,fp16,int8,rabitq}.jsonl`, `*-cost.json`, `cmp-*.json`; stores under `.benchmarks/r0913b/quant/<mode>/` |
| `reports/p0-video2.md` | `p0/video2/extract.py`, `identify.py`, `embed_queries.py`, `run.py`, `validate.py` | `p0/video2/{week,day}.npz`, `results.json`, `validate.json`, `single_gold.json` |
| `reports/p0-fusion.md` | `p0/fusion/run_video.py` (`--selfcheck`), `run_text.py` | `p0/fusion/results_video.json`, `results_text.json`, `results.json` |
| `reports/p0-compact.md` | `p0/compact/compact_run.py`, `compare.py`, `exact_replay.py` | `p0/compact/<store>-<op>.json`, `summary.json`, `exact-<store>.json`; copies under `.benchmarks/r0913b/compact/` |
| `reports/p2pre-econ.md` | `p2pre/run_arm.py`, `compare.py` (`--selfcheck`), `diagnose_index.py` | `p2pre/{day,week}-{control,agg_only,control_shipped_index}.jsonl`, `results.json`, `diagnostics.json`; copies under `.benchmarks/r0913b/econ/` |
| `reports/p1-compaction.md` | branch `r0913b/compaction`, worktree `.benchmarks/worktrees/r0913b-compaction` | commits `53e18e05`, `2c37b9be`, `7dbdfb03` |
| `reports/p1-atomic-keys.md` | branch `r0913b/atomic-keys`, worktree `.benchmarks/worktrees/r0913b-atomic-keys` | commits `4125c5c5`, `8739bca9` |
| `reports/rev-compaction.md` | `p2/compact/p2_close.py` | `p2/compact/<store>-<arm>.json`, commits `dd7a8082`, `7c1d0c50`; copies under `.benchmarks/r0913b/p2-compact/` |
| `reports/rev-atomic-keys.md` | `rev-keys/harness.py`, `kill_mid.py`, `qprobe.py` | commits `072efc36`, `5796cf6a`, `b614bc71`, `002f5931`; two pristine v18 unit copies under `.benchmarks/r0913b/rev-keys/` |
| `reports/rev-ann.md` | `rev-ann/run_arm.py`, `compare.py` (`--selfcheck`), `boundary.py`, `widen.py`, `scaling.py`, `minigraph.py` | `rev-ann/results.json`, `scaling.json`, five `week-week-*.jsonl` and two `day-day-*.jsonl`, commit `ab30accf`; copies under `.benchmarks/r0913b/rev-ann/` |
| `reports/d1-ann.md` | `d1/vectors.py`, `probe.py`, `probe_store.py`, `diag_scores.py`, `product_sweep.py`, `zvec_direct.py`, `mb_schema.py`, `calibrate.py`, `confirm_fix.py` | `d1/probe-*.json`, `direct-*.json`, `product-sweep.json`, `confirm-fix.json`, commit `8ead911d`; copies under `.benchmarks/r0913b/d1/` |

Every worker used `/home/yons/thomas/MindBridge/.venv/bin/python` with `PYTHONPATH` selecting the
code version under test and `PYTHONDONTWRITEBYTECODE=1`, and each report carries its exact command
block. Store copies under `.benchmarks/r0913b/` hardlink the media CAS and can be deleted once the
JSON artefacts have been read.
