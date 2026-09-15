# Round r0914 pre-registered protocol (verbatim copy)

This is `autoresearch/orchestrator-260914/PROTOCOL.md`, the pre-registration and Amendments 1-14
of the 2026-09-14 memory-dynamics round, copied here byte for byte at sha256
`8577c7e1222fe602f8b05784025ba859e6892eef1dc7d5c5361c490d3f7bff4d`.

It is published in the documentation set because the round directory is excluded from the
repository by `.gitignore`, so without this copy no record of the protocol exists inside the
repository. The copy is enclosed in a fenced block so that **no character of the source is
altered** — not for formatting, not for lint. Everything above the fence is this note; the file
itself starts at the fence. Verify with:

```bash
sha256sum autoresearch/orchestrator-260914/PROTOCOL.md
```

The round it governs is written up in [the memory dynamics round
note](2026-09-14-memory-dynamics-round.md).

````markdown
# Round r0914 — memory dynamics & episodic structure (pre-registered)

Lead session worktree: `.claude/worktrees/mindbridge-memory-mechanism-c767be` @ cf6aec4a
(contains r0913b compaction + ann-completeness; NOT atomic-keys).
Brief: mechanism-level innovation in the memory system itself, zero answer-policy changes,
paired A/B on retrieval-only metrics, adversarial review, honest reporting. Reader/prompt/abstention
untouched in every arm.

## Ceilings this round attacks (from r0912/r0913/r0913b, not re-tested)
- C1 continuous perception (mm-lifelong): near-duplicate 30 s clips flood the index; hit@12 day 41.5 %;
  ef+compaction lifted week hit@12 0.235→0.565 (r0913b). No captions in stores. Vision naming = binding limit.
- C2 long-lived text: locomo complete@12 0.785; 26.4 % of missed multi-gold not in depth-100 = discovery ceiling.
- C3 memory has no use-dependent dynamics: no strength, rehearsal, reconsolidation, forgetting-in-ranking
  (to be confirmed by kernel map; amend if wrong).

## Hypotheses
**H-A Episodic boundaries by prediction error (event segmentation, Zacks/Baldassano; ART vigilance).**
Consecutive observations whose key cosine to the running episode ≥ τ belong to one episode; a drop below τ
opens a new episode. Retrieval matches keys (unchanged), then groups hits by episode and allots the grounding
budget per episode (top members by own score, cap m per episode) instead of returning 12 near-duplicate clips.
Zero model calls. Product form: durable episode intervals = the temporal-relation projection substrate.
- Phase 0 (offline, mm-lifelong day d1 npz + gold; week if artefacts exist): arms A0 flat top-12 clips vs
  H-A(τ ∈ {0.90, 0.93, 0.95, 0.97}, m ∈ {1,2,3}); metric = any-gold-in-grounded-12-clips (same clip budget), also
  all-gold@12, episodes/rows count, mean episode length.
- Kill: best (τ,m) < A0 + 2.0 pp on day, or dev-selected (τ,m) negative on the second split. Report the sweep in
  full regardless.

**H-B Use-dependent accessibility (ACT-R base-level activation; testing effect; reconsolidation).**
B1 strength: a record cited as evidence by a prior answer gains activation that raises its rank for later
queries (decaying in time). B2 reconsolidation keys: the query vector of a question whose answer cited record g
is stored as an additional retrieval key of g (Widrow cognitive memory / doc2query without LLM).
- Phase 0a (offline, `.benchmarks/r0912-ab/out/a0-locomo.jsonl`, `a0-longmemeval.jsonl`): gold reuse rate =
  share of questions whose gold ∩ (gold of other questions in the same conversation) ≠ ∅; ORACLE upper bound of
  B1 = leave-one-question-out rank promotion of records that are gold∧ranked≤12 for other questions; metric
  complete@12/recall@12 delta and the interference cost on questions with no shared gold.
- Kill 0a: reuse rate < 15 % or oracle upper bound < +2.0 pp complete@12 on locomo → H-B closed without building.
- Phase 0b (only if 0a survives): B2 oracle with real query vectors (leave-one-out query-key addition on the
  a0 store); same kill.
- Reward-hacking guard (binding): any Phase-1 measurement of H-B uses ONLY the product's own signal (records the
  reader cited), a seeded random question order (report mean over ≥3 orders), never the judge label; and must
  report the loss on never-rehearsed gold. A gain that vanishes under this guard is a benchmark artefact.

## Frozen rules
- Retrieval metrics at limit=100 pool, sliced at 12/36; paired W/L with sign test + bootstrap CI (cluster t-CI when
  ≤4 clusters). Noise floor between two A0 ingests: 0 on recall@12 (r0912).
- No arm may change reader prompt, abstention, answer policy, or budget.
- Cost rule: rows/bytes and embed calls per observation must not rise > 10 % unless the quality gain is ≥ +5 pp.
- Adversarial review by a separate agent with Bash before any verdict is written.

## Amendment 1 (2026-09-14, after kernel map, before any Phase-0 number was read)
Kernel facts (reports/kernel_map.md) that change the hypotheses:
- C3 corrected: a bounded confirmation bonus already exists and is always on — `1+0.05*log2(1+min(access_count,20))`,
  ceiling ≈1.22, fed by `reinforce_on_answer=True`; eval stores never exercise it (access_count stays 0). So H-B1 "strength"
  is not new code; Phase 1 of H-B1 = measure the EXISTING mechanism honestly (sequential questions, product citations only)
  and, if positive, variants (cap, weight, spacing). H-B2 (query-vector reconsolidation keys) remains new.
- Video = exactly one atom key per asset (no temporal sub-segmentation); episodes in H-A are groups of clip RECORDS.
- Nothing computes novelty/surprise; arousal/valence never enter scores; identity/place are hard filters only; temporal
  predicate is binary overlap; no `session` concept; benchmarks never populate ObservationContext.

New hypotheses registered (Phase 0 offline, same frozen rules):
**H-C Identity-conditioned accessibility (transactive memory / source monitoring).** Each observation is bound to who
produced it; a query that names one known person modulates ranking toward that person's records (soft, bounded), not a
hard filter. Zero model calls (name match against known identity names).
- Phase 0 (a0-locomo jsonl + store content): among questions naming exactly one speaker, P(gold record's speaker == named
  speaker); oracle rank promotion of same-speaker records within top-100 (sweep strengths); interference on questions
  where gold is the OTHER speaker. Kill: P < 0.7 or oracle complete@12 < +2.0 pp or interference loss > half the gain.
**H-D Surprise-weighted encoding (prediction-error salience; von Restorff; novelty-gated hippocampal encoding).** At
write time novelty = 1 − max cos to the preceding N records of the same stream (N ∈ {20, all}); stored as a per-record
salience prior that scales ranking (bounded). Zero model calls beyond the embedding already paid.
- Phase 0 (a0-locomo + a0-longmemeval stores, embeddings from SQLite): novelty distribution gold vs non-gold within each
  question's top-100 (paired), AUC; oracle re-rank with real dense scores if the embedder is reachable else rank-percentile
  promotion. Kill: AUC < 0.55 or oracle complete@12 < +2.0 pp on locomo or negative on longmemeval.
Both are ranking priors, not representation changes; they cannot move the depth-100 discovery ceiling and are recorded as such.

## Amendment 2 (2026-09-14, after Phase-0 verdicts of H-A and H-B)
- H-B closed (reports/p0-hb.md): oracle +0.36 pp, honest proxy −1.45 pp. H-A closed (reports/p0-ha.md): best registered arm
  +1.50 pp day, dev-selected arms negative on week; adjacent-clip cosine median 0.826/0.842 so τ≥0.90 ≈ singletons; only 12 % of
  pool-resident missed gold is crowded out by a same-episode clip. **C1 is re-diagnosed: mm-lifelong loss is pool discovery
  (53/117 day misses have no gold in depth-100), and gold is a run of consecutive clips (median 4, mean 12–13).**
- Consequence: ranking priors (H-C, H-D, still running) cannot touch the discovery ceiling on either corpus (locomo 89/296,
  mm-lifelong day 53/117). Before proposing any Phase-1 build, a discovery-ceiling autopsy is registered: classify every
  pool-absent gold by the cheapest route that would reach it — (T) temporal reference in the question resolvable against
  occurred_at, (L) lexical/transcript term shared with the question but lost by tokenization or fusion, (N) temporal neighbour of a
  pool-resident clip, (D) dense-reachable at depth ≤ 1000 (exhaustive rank), (P) perceptual — nothing in stored text or vector
  distinguishes it (needs a perception model), (A) annotation noise. Only a class with ≥ 20 % of the pool-absent gold AND a zero
  model-call route qualifies a Phase-1 hypothesis; otherwise the round reports the ceiling map as its result.

## Amendment 3 (2026-09-14, after H-C verdict; before any H-E number)
- H-C closed (reports/p0-hc.md): P(gold speaker = named) 0.95 but oracle +0.08 pp, interference ≥ gain; the adapter inlines
  "<Speaker> said:" so dense already carries identity; and the kernel already re-indexes speech documents with a bound name
  (bind_speaker_names). The embodied form (out-of-band identity + kinship/pronoun queries) is unmeasurable on current corpora.
**H-E Surprise-gated perception (novelty-gated encoding; prediction-error allocates costly processing).** For a continuous
stream, spend the vision-description call only on observations whose key is novel relative to their predecessors; the rest are
stored with the paid embedding only. Mechanism claim: novelty computed from the already-paid embedding selects the clips whose
captions carry retrieval value, so a fraction f of the perception budget retains most of the full-caption gain.
- Phase 0 (offline, zero model calls): mm-lifelong day stores WITHOUT captions (`.benchmarks/data/baseline-qwen38-27b-wemm9b-20260911`
  day unit) and WITH captions (`.benchmarks/data/ablation-mmlifelong-day-20260912`), matched per clip; replay proxy = p0-ha's
  4-role max-over-part cosine with the 200 stored query vectors (reproduce both endpoints' hit@12 first: no-caption 0.415, caption
  ≈ PR #186 ablation). Per-clip novelty from the NO-caption visual/aggregate key vs preceding window (N ∈ {5, 20}, plus running
  max over the whole video). Arms at budget f ∈ {0.10, 0.25, 0.50}: (a) novelty top-f, (b) 5 random top-f seeds, (c) uniform
  stride, (d) oracle top-f = clips that are gold for some question (upper bound, label-leaking, reported only as ceiling).
  Metrics: hit@12, hit@36, coverage@12, all-gold@12; vision calls = f·2831.
- Kill: at f = 0.25 novelty retains < 50 % of the full-caption gain in hit@12, or beats the random-selection mean by < 2.0 pp.
- Adversarial guard: report whether "novel" clips are simply longer-transcript / scene-change clips that any heuristic finds
  (compare with a transcript-length selector and a shot-change selector if derivable), and the week split if both stores exist.

## Amendment 4 (2026-09-14, after H-D verdict)
- H-D closed (reports/p0-hd.md): AUC 0.72 passes but is 71 % explained by character length (AUC 0.717 alone) and ~18–29 % by the
  adapter prefix; best locomo cell novall β=0.4 +1.74 pp [+0.80,+2.69] < 2.0 bar, same β longmemeval −8.33 pp; session-local
  novelty (true episodic surprise) is harmful (−3.05 pp). Lost gold are low-novelty restatements that ARE the answers.
- Standing verdict after four ranking-prior hypotheses (H-B, H-C, H-D, H-A-as-grouping): the ranking layer of the kernel is at its
  ceiling on these corpora; remaining loss = candidate discovery (locomo 89/296, day 53/117) and perception. Only the autopsy
  classes and H-E (perception-budget allocation) remain open in this round.

## Amendment 5 (2026-09-14, after autopsy + queryform; before any H-F number)
Autopsy (reports/autopsy.md) findings that bind the rest of the round:
- locomo: the budget pass already grounds the whole depth-100 pool (mean 100.0 rows, 19 903 chars) → reader-visible complete =
  0.9354; the 12-window metrics of r0912–r0914 never measured what the reader saw. 100 % of pool-absent locomo gold is dense-reachable
  at depth ≤ 1000; N (neighbour) route lift 1.76× but at equal rows flat widening beats it. Text retrieval is closed for this round.
- mm-lifelong day: a clip costs ~12 420 chars so the 24 000-char budget admits ONE clip, yet the shipped path grounds 12 clips as a
  budget-exempt mandatory prefix (149 040 chars = 6.2× budget) — product defect to report, not to exploit. occurred_at is NULL on all
  2 831 records of the baseline store; 31/53 pool-absent questions carry ORDINAL temporal expressions (first/after/before) for which the
  kernel has no predicate. N route on day = chance (lift 0.93×). Transcript key −3.5 pp (already known from r0913b E1b, not general).
- queryform (reports/queryform.md): harness embeds a wrapped prompt on 20/30 tasks; product ask() embeds the bare question; bare is
  −1 pp on locomo, pool discovery unchanged → benign harness artefact, hygiene fix only.

**H-F Temporal evidence accumulation for continuous streams (event = run of clips; each clip is a noisy view; accumulate).**
Score each clip by its own cosine plus a bounded share of the match mass of its temporal neighbours in the same source stream:
s'(c) = max(s(c), s(c) + λ·agg_{|Δ|≤w, Δ≠0} s(c+Δ) − λ·baseline) or the softmax-sum variant; no boundaries, no eviction. Grouping
(H-A) evicted gold; accumulation should lift gold runs whose members sit at ranks 13–420 as individually weak but jointly strong.
- Phase 0 (offline, zero model calls, exhaustive cosine over all clip keys so the pool can widen): day dev, week holdout; replay
  proxy from tools/ha_common.py (4-role max-over-part; reproduce 0.4150/0.6050 first). Sweep w ∈ {1,2,4,8}, agg ∈ {mean, max,
  top-2 mean}, λ ∈ {0.25, 0.5, 1.0}; also the per-video z-scored variant (subtract the video's mean cosine so that a uniformly
  query-like video does not win by mass alone). Metrics hit@12, hit@36, cov@12, all@12, gold-in-top-100 share. Paired W/L/T,
  bootstrap CI, McNemar.
- Kill: dev-selected (w, agg, λ) < +2.0 pp hit@12 on day, or negative on week. Report the full grid regardless.
- Adversarial guard: compare against the trivial "widen to top-36 then take 12 by video-run order" and against H-A grouping;
  report whether gains concentrate in cross-clip-occurrence questions and whether any question loses because a single-clip gold is
  drowned by a high-mass non-gold run.

## Amendment 6 (2026-09-14, after H-F verdict)
- H-F closed under the strict reading (reports/p0-hf.md): dev cell (w=8, mean, λ=1.0) day +4.50 pp [+0.5,+8.5] p=0.049 uncorrected
  over 80 cells; week plain +2.00 [−2.0,+6.0], week z −1.50 — day (one video) cannot select plain vs z. Guards fail: 0/23 gained
  questions gained gold from outside top-100 (gold-in-top-100 0.735→0.730); day gain sits in single-clip-gold questions (+15.6 pp)
  which are −3.3 pp on week; cross-split cell Spearman 0.238. Durable mechanistic result: additive one-sided accumulation beats every
  evicting grouping (H-A/MMR) on both splits; run-mass re-order control is −4.0 / −7.0 pp.
- Budget verification (Explore agent, cited in reports/autopsy-budget-verify.md): the limit window's budget exemption is the
  documented contract (plugins.py:131-135, docs/configuration.md:369, test_memory_api.py:2236) — a floor, not a defect; the gap is
  naming and the unbounded floor cost on video. HEAD's _mm_lifelong sets synthetic occurred_at (2000-01-01 + offset, 37ff2d36); the
  day baseline store predates it. recall_limit is run-global (default 20; the cited rounds used 12).

## Amendment 7 (2026-09-15, after H-E verdict — round closed)
- H-E closed (reports/p0-he.md). Registered artefact error: `ablation-mmlifelong-day-20260912` has 0 captions; the only captioned
  day store is inside `baseline-2-qwen38-27b-wemm2b-20260912`, live-written by the user's `mindbridge-bench eval --config
  baseline_2.yaml --resume` (pid 1365516) — snapshotted read-only via VACUUM INTO at 1 256 / 1 576 / 2 112 captions. No captioned
  week store exists. Control re-paired to the ablation (no-caption, same PR #186 write path) store.
- Full captions (74.6 % coverage) hit@12 0.4000 → 0.3350 (−6.50 pp [−12.0,−1.5], W/L 8/21, McNemar p=0.024); clause 1 vacuous,
  clause 2 fires (novelty − random = −0.00 pp; oracle also 0.4000). Mechanism: the caption's own text key is argmax for 0.28 % of
  pairs (deleting it changes nothing); the loss is the AGGREGATE key being rewritten; partial coverage biases the index (captioned
  clips 83.5 % of top-12; un-captioned gold −15.5 pp constant). Confound-free graft (caption as extra key, aggregate untouched):
  +1.00 pp ns at f=1, exactly 0 at f=0.10. Cost: ingest 49 → 182 min, +52.5 % keys per captioned clip.
- Round result = the ceiling map + six pre-registered kills + two write-path facts (transcript key, caption-in-aggregate) that
  point at the same design rule: derived text must not rewrite the perceptual aggregate key.

## Amendment 8 (2026-09-15, after independent adversarial review reports/review-round.md)
Accepted corrections (arithmetic of every report reproduced; meaning corrected):
- H-E magnitude withdrawn: at 99.4 % caption coverage full captions are −3.00 pp hit@12 [−8.5,+2.5] p=0.39 (hit@36 +3.00);
  the caption-as-separate-key graft is **+3.00 pp hit@12 [+1.0,+5.5] p=0.031, +3.50 hit@36, +1.07 cov@12, W/L 6/0** (one
  pre-specified contrast, faithful day proxy). Kill of H-E stands on clause 2 only (novelty − random +0.7…+1.2 pp; oracle < random).
- Transcript-key finding is not general (day +3.50 / week −4.00) and is unregistered; Amendment 7's "same design rule" withdrawn.
- H-F: the registered rule did NOT fire on the dev cell (week plain +2.00 ≥ 0); the strict reading in Amendment 6 was adopted after
  the numbers. Corrected count: five pre-registered kills (H-A, H-B, H-C, H-D, H-E) + one post-hoc no-build (H-F) on guards.
- Week replay proxy is a stronger ranker than the product (0.605 vs 0.415; overlap@12 0.630); conclusions resting on it alone are
  weak. Integrity: protocol lives in a gitignored dir with a hand-appended hash chain; pre-registration is attested by process only.
  Mitigation: PROTOCOL.md copied verbatim into docs/research/ and committed with the note.
Candidate for Phase 1 (registration pending reports/hg-feasibility.md):
**H-G Write-path key composition: derived enrichment as separate keys, perceptual aggregate untouched.** Claim: a vision caption
(and any derived text) must not be unioned into the object_part-0 aggregate embedding input; it keeps its own text key(s) and stays
in the lexical document. Zero additional model calls. Pre-registered acceptance (to be finalised with the feasibility numbers):
product-level paired A/B on mm-lifelong day via `search_with_trace` through the real ranking (not the proxy): hit@12 ≥ +2.0 pp
with CI excluding 0 or McNemar p < 0.05; holdout = a second captioned corpus (offline graft and/or product re-ingest); cost rule
unchanged. Kill if day product delta < +2.0 pp or holdout negative.

## Amendment 9 (2026-09-15, H-G registration; before any holdout number is read)
Feasibility (reports/hg-feasibility.md): both benchmark runs use the same embedder `tencent/WeMM-Embedding-2B:2048:messages-v1:l2-v1`
and index recipe, so cross-run grafts are valid; matched caption/no-caption store pairs exist for m3-bench-robot (100 stores) and
memlens 32k/64k/128k/256k (195 stores each); mm-lifelong week caption store lands ≈06:10 from the user's live eval. DescriptionCache
covers 2 813/2 831 day clips. Change = aggregate key (object_part 0) embeds only the caller-provided observation; `[visual description:`
/ `[facts:` sections keep their own keys and stay in the lexical document; transcripts untouched; recipe v12→v13; key count unchanged.
**H-G acceptance, pre-registered:**
- Dev (already read): day graft +3.00 pp hit@12 [+1.0,+5.5] p=0.031.
- Holdouts (offline graft, zero model calls, same tooling he_extract/he_mech): memlens-32k, memlens-256k, m3-bench-robot, and
  mm-lifelong week when available. Metric = the corpus's retrieval hit@12 / cov@12 against its gold evidence ids (if a corpus has no
  gold ids it is NOT a holdout — say so). Rule: pooled holdout hit@12 delta > 0 with paired bootstrap CI excluding 0, AND no single
  holdout < −1.0 pp with CI excluding 0. Report every split regardless.
- Transfer check: product re-ingest of the day unit with the patched kernel (cached captions, ~18 vision calls), then the 200
  questions through the REAL ranking (`search_with_trace`, limit 100) vs the existing captioned day store: hit@12 ≥ +1.0 pp with the
  same sign as the proxy; if the product delta is ≤ 0 the proxy result does not transfer and H-G is not adopted.
- Cost rule: embedding calls and rows per observation unchanged (verified by key count); ingest wall-clock within +5 %.
- Adversarial review of the patch and of the holdout numbers by a separate agent before the verdict is written.

## Amendment 10 (2026-09-15, H-G holdout + patch review read; transfer check pending)
- Holdout (reports/hg-holdout.md): m3-bench-robot has no gold ids → not a holdout; week caption store not yet written. memlens-32k /
  256k (n=173 each, nested): G−A hit@12 0.00 / +0.58 pp; pooled +0.29 pp CI95 [0.00, +0.87], W/L/T 1/0/345 → clause 1 (CI excludes 0)
  FAILS by the letter. Reading recorded, not re-read: the population is inert (A already 0.9884 / 0.9480; captions 11 % of records;
  dataset caption text already inside every captioned record; caption key argmax 0.52 %). Caption-in-aggregate is not harmful there
  (C−A +1.16 / +0.58 ns). The only holdout that exercises the mechanism (video aggregate = vector + ~7-char ASR) is week — deferred
  until the user's eval finishes writing it; H-G cannot be adopted as a default in this round regardless of the day transfer check.
- Patch review (reports/hg-review.md): fix-first. F1 caption with interior blank line leaks into the aggregate (0/34 564 real captions
  affected → read numbers uncontaminated); F2 caption-only record loses its redundant aggregate row (key count claim false);
  F3 opening any v12 store with the patched kernel re-embeds it in place → controls must be measured on copies with the original
  kernel (transfer agent warned); F7–F9 docstring/docs/migration-cost prose. Path parity add()==capture()+settle()==migration verified.

## Amendment 11 (2026-09-15, transfer check read)
- H-G transfer (reports/hg-transfer.md): product ranking on a reindexed copy of the fresh day store: NEW 0.415 / C 0.400 / A 0.405
  hit@12; NEW−C +1.50 pp [−3.50,+6.50] 13W/10L p=0.68 (hit@36 +2.00, cov@12 +0.14, all@12 −0.50); NEW−A +1.00 [−4.0,+6.0]. Same sign as
  the proxy on every hit@k; magnitude 2–4× smaller. Registered transfer rule met on the point estimate only; every CI crosses 0 →
  **H-G not adopted as default**; branch kept (tip 5f4e576c) pending the week holdout. Cost: ingest 41 min with cached captions
  (47 vision requests), vectors 10 014 vs 10 725 (−6.6 %, review F2: caption-only clips drop the redundant aggregate row).
- **Defect found, not H-G's**: a store ingested from scratch by either kernel of this branch lineage (631704fa and its parent 8b3483e1,
  i.e. the session branch with r0913b compaction/ef merges) has a Zvec index whose vectors are bound to wrong records: index vs
  exhaustive-cosine overlap 0.15@100, hit@12 0.255, score error 0.091 mean; delete `zvec/` + reopen → 0.994 / 0.415. Stores written
  by pre-branch kernels (26/40 unmerged segments) are 0.990/0.994. Suspect: incremental index write path + close-time segment merge
  (r0913b 53e18e05/7c1d0c50). Registered as a blocking defect: root-cause, regression test, fix on `r0914/index-binding-fix`.

## Amendment 12 (2026-09-15, H-G week holdout read — H-G closed)
- Week holdout (reports/hg-holdout-week.md, registered against Amendment 9, read at 7500bbe0…; Amendment 11 landed while it ran):
  the captioned week store finished 06:29 (6 266 clips, 6 258 captions = 99.87 % coverage; matched no-caption store
  `baseline-…-20260911`; both read from VACUUM INTO snapshots, never opened through `Memory`). Graft G−A hit@12 **−0.50 pp
  [−2.5, +1.0], W/L/T 1/2/197, McNemar 1.0**; hit@36 0.00; cov@12 +0.22 ns. Pooled over the three registered holdouts
  (memlens-32k, memlens-256k, week; clustered by question, n=546): **0.00 pp [−0.74, +0.73], W/L/T 2/2/542** → clause 1 fails
  (delta not > 0, CI includes 0); clause 2 holds (worst split −0.50, CI includes 0). **H-G fails its pre-registered holdout.**
  With the transfer clause met on its point estimate only (Amendment 11), the merge condition (holdout AND transfer) is not met:
  H-G is closed for this round, `r0914/aggregate-key-composition` stays unmerged (tip 5f4e576c).
- Mechanism on week: the caption key is argmax for 0.30 % of (query, captioned clip) pairs and 6.8 % of captioned top-12 clips
  (day 0.28 % / 2.1 %, memlens 0.52 % / 1.3 %); 75/200 top-12 sets change, gold moves on 3 questions (1 W / 2 L). C−A
  (confounded, as on day) −2.00 pp ns; product-level C−A from both runs' recorded rankings 0.00 pp hit@12 (W/L 33/33), cov@12
  −3.26 ns → caption-in-aggregate is not measurably harmful on week. Day's +3.00 pp stands as a single-split, single-video result
  that did not generalise. Week proxy caveat unchanged (replay 0.605 vs product 0.415; overlap@12 0.63 for A, 0.43 for C).
- Tooling: `hg_common.predict_roles` replays `stored_canonical_parts`, which does not cut `[speech identities:]` sections, so it
  under-counts keys by one on every speech-bearing record (5 473/6 266 on week; a first pass with it was discarded unread). Week
  roles are positional (`tools/hg_week.py`), asserted against the stored part count on every record and validated by cosine
  (video key A↔C min 0.991). memlens records carry no speech section; Amendment 10's numbers are unaffected (0 mismatches there).
- Chain note: this worker briefly appended a duplicate `7f52ed10… PROTOCOL.md` line to PROTOCOL.sha256 (a guard that stopped the
  amendment did not stop the hash line) and removed it two minutes later; the chain below line 13 is otherwise untouched.

## Amendment 13 (2026-09-15, index-binding defect root-caused; reports/index-binding-defect.md)
- Root cause: zvec 0.7.0 `Collection.optimize()` re-binds vectors to ids whenever a dead row (deleted doc or superseded upsert)
  exists — survivors written densely, ids keep pre-merge positions → each doc from the first dead row serves its neighbour's vector;
  scalar fields / memory_id / counts stay correct. Bisect: clean 599ae4b8 (1.000), broken from 53e18e05 (r0913b close-time merge,
  0.104) — exposure, not cause: 599ae4b8 with 70 deletes across the 64-flush bound also 0.107 → upstream affected wherever
  `ZvecIndex.optimize()` runs with a dead row. Smallest condition: 1000 records + one delete + close + reopen.
- Fix `r0914/index-binding-fix` @ 1337a01a: copy-compact via `_compact()` whenever `_dead_rows` may be set (delete; upsert whose doc
  count grew by less than the batch; open of a collection with persisted segments); native merge otherwise. Regression test fails
  before / passes after; 2197 tests. Prevents, does not repair: affected stores must `reindex()`. Detection = self-hit@1 over 200
  sampled stored vectors (healthy 1.000, damaged ≈0). Independent review of the fix pending before merge.
- Consequence for prior rounds: r0913b compaction/ef gains and any measurement on a store that merged with dead rows present must be
  re-verified with the self-hit check. H-G's transfer numbers were taken on a rebuilt (clean) index and stand.

## Amendment 14 (2026-09-15, round closed)
- Index-binding fix reviewed twice (reports/index-binding-fix-review.md): fix-first on 1337a01a (every writing session re-copied the
  collection; two triggers untested; docs silent on affected stores) → 763458e4 (clean marker, tests delete/replace/duplicate +
  marker lifecycle + debris sweep, docs/CHANGELOG) merge-ready with N1 (marker beside, not inside, the collection → stale marker on a
  restored dirty backup re-exposes the defect) → e82d5015 (marker at `zvec/.mindbridge-clean`, restored-backup test). 2 202 tests.
  Close-time merge cost: inherited unproven 2.0 s → once; steady state 0.40 s (baseline 0.45–0.54). **Merged into the session branch.**
- Final state: H-A/H-B/H-C/H-D/H-E killed at pre-registered Phase 0; H-F post-hoc no-build; H-G killed on pre-registered holdout
  (week/pooled 0.00), branch `r0914/aggregate-key-composition` @ 5f4e576c kept unmerged; defect fix merged. Research note
  docs/research/2026-09-14-memory-dynamics-round.md.
````
