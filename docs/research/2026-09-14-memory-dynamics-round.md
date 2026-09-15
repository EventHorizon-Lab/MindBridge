# Memory dynamics round r0914 — 2026-09-14

Pre-registered round on mechanism-level innovation inside the memory system itself. Round directory:
`autoresearch/orchestrator-260914/`, lead worktree
`.claude/worktrees/mindbridge-memory-mechanism-c767be` at `cf6aec4a` (carries r0913b compaction and
ann-completeness, not atomic-keys). Every number below is internal to this project's own retrieval
replays and frozen benchmark dumps; none is comparable to a published leaderboard row.

## 1. Brief and method

The brief was: mechanism-level innovation in the memory system, zero answer-policy changes, paired
A/B on retrieval-only metrics, adversarial review, honest reporting. Reader, prompt and abstention
were untouched in every arm.

Method rules, frozen before any number was read:

- **Pre-registration with a hash-pinned protocol.** Hypotheses, arms, metrics and kill rules were
  written into `PROTOCOL.md` before the corresponding measurement; each amendment was appended and
  the file re-hashed into `PROTOCOL.sha256`, so a report can name the digest it was registered
  against.
- **Phase 0 is offline and model-call-free, and it computes oracle upper bounds before any build.**
  Every hypothesis had to clear a numeric bar on frozen stores and frozen retrieval dumps, replayed
  with numpy, before a single line of product code could be proposed. The only model calls in the
  whole round were query embeddings against the same endpoint the frozen run used (H-D and the
  query-form ablation), cached to disk.
- **Metric reproduction as a gate.** No arm was read until the baseline metric was reproduced
  exactly: `recall@{12,36}` and `complete@{12,36}` with 0 mismatches on all 1,437 locomo and
  longmemeval rows, and mm-lifelong day A0 `hit@12` 0.4150 / `hit@36` 0.5800 / `cov@12` 0.1903
  bit-for-bit against the previous round.
- **Frozen statistical rules.** Retrieval metrics at a limit-100 pool sliced at 12/36; paired W/L
  with sign test and bootstrap CI (cluster t-CI when clusters are few); the noise floor between two
  A0 ingests is 0 on recall@12 (r0912). No arm may change the reader prompt, abstention, answer
  policy or budget. Cost rule: rows, bytes and embed calls per observation may not rise more than
  10 % unless the quality gain is at least +5 pp.
- **Adversarial review by a separate agent, with Bash, before any verdict was written.**
- **Honest-signal guard.** Binding for H-B and by extension for any use-dependent mechanism: a
  Phase-1 measurement may use only the product's own signal (the records the reader cited), a seeded
  random question order reported as a mean over at least three orders, never the judge label, and
  must report the loss on never-rehearsed gold. A gain that vanishes under this guard is a benchmark
  artefact.

Six hypotheses were registered and all six are closed in Phase 0: H-A, H-B, H-C and H-D in section
3, and H-F and H-E — registered later, in Amendments 5 and 3, and resolved last — in section 6. Five
are kills by their own pre-registered rules; H-F is a post-hoc decision not to build, and section 6
says so plainly.

After the round closed, an independent reviewer re-ran every `--selfcheck` and every primary script
from a scratch copy and re-derived three headline metrics with separate code: `p0_hb`, `p0_hc`,
`ha_run`, `hf_run`, `qf_run`, `autopsy_routes` and the whole `he_*` chain reproduce
**byte-identically**, and `p0_hd` differs only in four float-ULP entries and one live re-embed at
1e-5. The arithmetic of this round is sound. What the review changed is what the numbers are allowed
to mean, and those corrections are carried inline below rather than appended — the caption effect at
full coverage (section 6), the transcript key on the second split (section 4), H-F's verdict status
(section 6), the week proxy and the pre-registration integrity (section 7).

Source: `autoresearch/orchestrator-260914/PROTOCOL.md`.

## 2. What the kernel does by default

A read-only map of `HEAD` `cf6aec4a` was produced before the first hypothesis was scored, and it
corrected the round's own premise C3.

MindBridge at default settings is a hybrid dense plus lexical retriever over an append-only
bitemporal SQLite store with a Zvec ANN index. Everything that resembles memory science — formation,
consolidation, decay, forgetting — is model-gated (backend `None` by default) or config-gated
(`None` by default). Between write and read only `access_count` and `last_accessed_at` change.

**The one always-on dynamic.** A bounded confirmation bonus multiplies the score:
`1 + 0.05*log2(1 + min(access_count, 20))`, ceiling ≈ 1.22, fed by `reinforce_on_answer=True`.
`access_count` carries a CHECK of 0..20 and `last_accessed_at` is a monotone max, so inter-access
intervals are never recorded. Every eval store's `access_count` stayed zero, so no measurement this
project has published has ever exercised it.

**Gated off by default.** Decay (`decay_half_life_days=None`), the PRESSURE consolidation trigger
(`memory_budget_records=None`), formation (`if former is None: return`), `consolidate()` (raises
without a consolidator), `recall_planning=False`, and the transcriber, vision describer, face
analyzer, former and consolidator backends (all `None`; only the embedder is required).

**Score formula.** Dense relevance is `clamp01(1 - dist)` taken as a max over parts; the lexical
route is a Zvec RRF over stemmed and n-gram FTS, returned as `score/max_in_route`; fusion sets
`lexical_score = 0.75*lex` when coverage is at least 0.999 and 0 otherwise,
`base = max(dense, lexical_score)`, `relevance = bounded_scale(base, 1 + 0.3*coverage)`. The final
score is `bounded_scale(relevance, reinforcement)` times the temporal factor (RANK_CEILING 1.5 on
interval overlap, else 1.0), times retention (off) and `context.confidence`, gated at
`minimum_relevance = 0.10`. `RERANK_CANDIDATES = 100` is request-independent. In practice this is
dense-scored with lexical as candidate supply: 2 of more than 20,000 candidates ever reached full
coverage.

**Absent mechanisms.** No per-memory strength beyond `access_count` at most 20, no spacing, no
durability growth; nothing forgets by itself and there is no interference or competition; no
rehearsal; no reconsolidation, since records are immutable and retrieval never alters content or
keys; no schema, gist or prototypes; nothing computes novelty or surprise, and arousal and valence
are stored but never scored; the association second hop was built and refuted in an earlier round;
the only temporal predicate is interval overlap, with no before/after, no Allen relations and no
timeline API; there is no `session` concept anywhere; video is exactly one atom key per asset, with
no temporal sub-segmentation; identity and place are hard filters pushed into the index, with no
speaker or entity bonus; and no benchmark adapter populates `ObservationContext` or `place_id`, so
the identity and spatial axes are unexercised by any measurement.

Source: `autoresearch/orchestrator-260914/reports/kernel_map.md`.

## 3. Hypotheses and verdicts

### H-A — episodic boundaries by prediction error

**Mechanism.** Consecutive observations whose key cosine to the running episode is at least τ belong
to one episode; a drop below τ opens a new one. Retrieval matches keys unchanged, then groups hits
by episode and allots the grounding budget per episode (top members by own score, cap `m` per
episode) instead of returning 12 near-duplicate clips. Zero model calls. Two reference rules: `r1`
compares against the previous clip, `r2` against the running mean of the episode so far.

**Biological rationale.** Event segmentation theory holds that a spike in perceptual prediction
error marks an event boundary, and memory is organised around those boundaries.

**Pre-registered kill rule.** Best (τ, m) below A0 + 2.0 pp `hit@12` on mm-lifelong day, or the
dev-selected (τ, m) negative on the second split.

| clause | registered grid τ ∈ {0.90, 0.93, 0.95, 0.97} | with the assignment's τ = 0.85 |
| --- | --- | --- |
| best (τ, m) on day | **+1.50 pp** (r2 τ0.90 m1), CI95 [+0.00, +3.50], W/L/T 3/0/197, McNemar p = 0.250 | **+2.00 pp** (r2 τ0.85 m1), CI95 [−0.50, +4.50], W/L/T 5/1/194, p = 0.219 — equals, does not clear |
| dev-selected on the second split | r2 τ0.90 m3: day-dev +3.00 pp, day-holdout **−3.00 pp**, week **−1.00 pp** | r2 τ0.85 m1: week **−3.50 pp**, W/L 4/11 |

A0 is `hit@12` 0.4150 on day and 0.6050 on week; the clip budget is 12 clips in every arm.

**Why it failed mechanistically.** The median adjacent aggregate cosine is 0.8258 on day and 0.8416
on week, so the entire registered τ grid sits above it: at τ ≥ 0.93 more than 94 % of episodes are
singletons, and at τ = 0.97 the segmentation is the identity. Of the 288 missed gold clips that are
inside the depth-100 pool on day, only 35 (12.2 % at r2@0.85) are crowded out by a same-episode
clip; 88 % are genuinely far, with no grounded clip in their episode to trade against. The reason is
that the gold for a question is itself a run of consecutive clips (median 4 per question, mean 12.2
on day and 13.0 on week, max 511 / 237), so any rule that says "one clip per neighbourhood" evicts
gold before it evicts distractors. The exploratory sweep below the registered grid confirms it: as τ
falls, episodes lengthen, the de-duplication route's own bound rises from 4.5 to 32 pp on day, and
the realised delta falls monotonically to −10.5 pp (day) and −19.5 pp (week).

**The bar was mis-set relative to the corpus.** Inside the registered grid the de-duplication
route's own upper bound runs from +0.0 pp (r1@0.90) to +2.0 pp (r2@0.90) — the +2.0 pp bar equals
the mechanism's entire measurable ceiling, so the registered grid was unpassable by construction.
The verdict is unaffected, because the dev-selected arm is negative on the second split and the
exploratory sweep below the grid is monotonically worse; but this is not a mechanism tested at full
strength and beaten.

**Adversarial reading.** Largely diversity by another name. The same shape, sign and monotonicity as
the r0913b MMR arm E3, and the naive control — temporal de-duplication by clip index, containing no
embedding geometry at all — reproduces the same monotone collapse of `cov@12` and `all@12` (−3.7 to
−4.9 pp on day, −5.0 to −8.3 pp on week). The honest residue is that the premise behind C1 was
wrong: mm-lifelong's retrieval loss is not near-duplicate flooding but that the right 30 seconds of
video is not in the depth-100 pool at all — 117/200 day questions miss, and for 53 of them no gold
clip is anywhere in the pool. The ceiling over any reselection of that pool is `hit@100` 0.735 (day)
/ 0.850 (week).

Source: `autoresearch/orchestrator-260914/reports/p0-ha.md`.

### H-B — use-dependent accessibility

**Mechanism.** B1 strength: a record cited as evidence by a prior answer gains activation that
raises its rank for later queries, decaying in time. B2 reconsolidation keys: the query vector of a
question whose answer cited record `g` is stored as an additional retrieval key of `g`. Amendment 1
corrected B1 before it was measured — the bounded confirmation bonus already exists and is always
on, so Phase 1 would have been an honest measurement of an existing mechanism, not new code.

**Biological rationale.** ACT-R base-level activation and the testing effect: retrieving a memory is
itself a learning event that makes the memory more accessible later.

**Pre-registered kill rule.** Reuse rate below 15 %, or an oracle upper bound below +2.0 pp
`complete@12` on locomo, closes H-B without building.

| clause | measured | verdict |
| --- | --- | --- |
| reuse rate at least 15 % (locomo, per conversation) | 60.86 % (838 of 1,377 questions) | pass |
| reuse rate at least 15 % (longmemeval, registered unit) | 0.00 % — `unit_id == question_id`, one question per unit | fail, structurally |
| oracle upper bound at least +2.0 pp `complete@12` on locomo | **+0.36 pp** (promote = 6), boot CI [−0.36, +1.09], W/L/T 16/11/1350, sign p 0.44 | **fail** |
| honest variant retains at least half the oracle gain | **−1.45 pp** [−2.54, −0.36] against +0.36 pp, so 0 % | **fail** |

The strongest promotion, moving the whole rehearsal set to the front, is −36.53 pp. The rule-free
ceiling — any reordering that may only lift the rehearsal set and is charged no displacement cost —
is +5.88 pp on locomo, 81 rescuable questions of 1,377.

**Why it failed mechanically.** `R(q)` is 18.4 records wide inside a 100-record pool, and the useful
part of it is a fraction of one record per question, so selectivity, not strength, is the binding
constraint. Interference is intrinsic rather than a tuning artefact: the shared-gold gain and the
no-shared-gold loss stay within a factor of about one of each other at every strength (+1.19/−0.93
at d = 6, +2.03/−2.41 at d = 12, −10.74/−76.62 at front). And the reuse lives where it is useless:
the locomo reuse rate falls from 60.9 % to 29.1 % of questions once restricted to gold that is
actually missed at 12, and only 11.9 % of missed gold records serve two or more questions. Rehearsal
strengthens what is already retrieved.

**Adversarial reading.** The reuse is a benchmark-construction property, not a memory property: the
gold-citation Gini is 0.254 and the top 10 % of gold records carry 23.6 % of citations, a broad
shallow overlap consistent with a fixed pool of questions written from one conversation, not with a
few salient facts everyone asks about. The oracle leaks the label — `R(q)` is built from gold
membership, which the product never sees — and the honest proxy a real reader could rehearse turns
the sign negative, which is exactly the failure mode the reward-hacking guard was written to catch.
Headroom, not mechanism, caps the result: only 296 of 1,377 locomo questions are incomplete at 12,
and 89 of those have gold outside the depth-100 pool entirely.

Source: `autoresearch/orchestrator-260914/reports/p0-hb.md`.

### H-C — identity-conditioned accessibility

**Mechanism.** Each observation is bound to who produced it; a query that names one known person
modulates ranking toward that person's records — soft and bounded, not the hard filter `identity_id`
is today. Zero model calls, since it is a name match against known identity names.

**Biological rationale.** Transactive memory and source monitoring: who told you something is part
of the retrieval cue.

**Pre-registered kill rule.** P(gold speaker == named speaker) below 0.7, or an oracle `complete@12`
below +2.0 pp, or interference loss above half the gain.

| gate | threshold | measured | verdict |
| --- | --- | --- | --- |
| P(gold record's speaker == named speaker) | at least 0.70 | **0.9499** (1,632 of 1,718 gold records; 1,235 of 1,377 questions name exactly one speaker) | pass |
| oracle Δ`complete@12` | at least +2.0 pp | **+0.081 pp** (promote = 3, subset; +0.073 corpus), W/L/T 1/0/1234 | **fail** |
| interference at most half the gain | — | best arm 2 questions lost against 2 gained; promote = 12 is −0.405 pp, front −0.891 pp with Δ`complete@36` −2.11 pp (sign p 2.2e-4) | **fail** |

**Why it failed mechanistically.** The mean boost set is 90.1 of 100 ranked records — there is
almost nothing left to promote. The embedder already lifts the named speaker from a 50 % base rate
to 90.1 % of the depth-100 pool, 91.9 % of the top-36 and 92.8 % of the top-12, monotonically with
depth, because `"<Speaker> said:"` is literally in the embedded text of every record; and the kernel
already re-indexes speech documents with a bound name through `bind_speaker_names`. The residual
7 pp of top-12 mass is precisely where the other speaker's true gold lives: baseline `complete@12`
is 0.840 when the gold belongs to the named speaker (n = 1,158) and 0.630 when it belongs to the
other (n = 27), of which 17 are currently complete and are exactly what an identity prior puts at
risk.

**Adversarial reading.** Both halves of the signal are annotation artefacts. Question-side: LoCoMo
annotators name a speaker in 89.7 % of items and use a bare pronoun in 0.44 % (6 of 1,377), which is
the opposite of how a companion's user asks. Record-side: the benchmark adapter writes the speaker
name into the record text, so "who produced this" rides the embedding for free — a real ingest binds
the producer out of band through diarisation, where the name is not in the vector. The kill is
therefore recorded as corpus-scoped: this falsifies identity-conditioned accessibility *on a corpus
where the mechanism is already inlined into the representation*, not as a mechanism.

Source: `autoresearch/orchestrator-260914/reports/p0-hc.md`.

### H-D — surprise-weighted encoding

**Mechanism.** At write time, novelty = 1 − max cosine to the preceding N records of the same stream
(N ∈ {20, all}, plus a session-local variant), stored as a per-record salience prior that scales
ranking, bounded. Zero model calls beyond the embedding already paid.

**Biological rationale.** Prediction-error salience and novelty-gated hippocampal encoding — the von
Restorff effect, where the item unlike its neighbours is the one remembered.

**Pre-registered kill rule.** AUC below 0.55, or an oracle `complete@12` below +2.0 pp on locomo, or
negative on longmemeval.

| criterion | threshold | measured | verdict |
| --- | --- | --- | --- |
| AUC, novelty gold vs non-gold | at least 0.55 | locomo 0.7202 (nov20) / 0.7318 (novall) / 0.7092 (novsession); longmemeval 0.7181 / 0.6097 | **pass** |
| oracle Δ`complete@12`, locomo | at least +2.0 pp | **+1.74 pp** (novall β = 0.4), boot CI [+0.80, +2.69], cluster [+0.63, +2.94], W/L/T 35/11/1331, sign p 5.4e-4 | **fail** |
| longmemeval at that β | not negative | **−8.33 pp** (novall), −5.00 pp (nov20); only β = 0.1 is non-negative, where locomo yields +0.51/+0.73 pp | **fail** |

**Why it failed mechanistically.** The signal is real but is mostly not surprise. Raw character
length has AUC 0.717 for the same gold-vs-non-gold discrimination — statistically indistinguishable
from novelty's 0.721, with Spearman 0.418 — and holding length still inside quartiles the mean AUC
falls to 0.687, so roughly a third of the above-chance signal is length. A further slice of it is
the adapter's own scaffolding: the shared `[source_id]` and ISO-timestamp prefix is more
self-similar than the records it labels and eats about 18 % of the novelty range; re-embedding the
bodies with the prefix stripped drops AUC from 0.723 to 0.658, a 29 % loss of the above-chance
signal. The interference is diagnostic: the gold the prior evicts has mean novelty 0.2455 against
0.3296 for the gold that survives, and reading the casualties shows they are polite restatements
that repeat the previous turn's topic — and are also the answer the question wants.

**Two corrections to how this was reported.** `PROTOCOL.md` registers H-D's novelty definition, its
AUC gate and its `complete@12` bars, and registers **no β**: β appears in the protocol only in
Amendment 4, written after the numbers, and `novsession` is likewise an unregistered third variant.
`p0-he.md`'s sibling report labels β ∈ {0.1, 0.2, 0.4, 0.8} "pre-registered"; that label is wrong
and is not used here. The sweep is 10 β × 3 variants × 2 corpora. Because H-D is a kill on the
*best* cell the error is conservative and the verdict is unaffected. Second, the bar sits inside the
interval: +1.74 pp with a bootstrap CI of [+0.80, +2.69] against a +2.0 pp bar makes that clause a
coin flip on a point estimate, and the kill is really carried by the second clause — longmemeval
−8.33 pp, itself n = 60 at a 0.90 baseline where one question is 1.67 pp.

**Adversarial reading.** The variant ranking says what the signal actually is. `novall` — uniqueness
against the whole unit — is the best re-ranker, while `novsession`, novelty against the same
conversation session and the closest thing to genuine temporal surprise, is harmful at every β above
0.1 (−3.05 pp at β = 0.4, −7.48 pp at β = 0.8). What helps is not "this was surprising when it
happened" but "this record is specific and unlike the rest of the corpus" — a length-correlated,
IDF-shaped redundancy prior that dense retrieval already partly prices in.

Source: `autoresearch/orchestrator-260914/reports/p0-hd.md`.

### Standing verdict after four ranking priors

Recorded in Amendment 4. Rehearsal strength, identity conditioning, write-time novelty and episode
grouping are four independent ways of reordering the same depth-100 pool, and all four land at or
below +2 pp with intrinsic interference. The ranking layer of the kernel is at its ceiling on these
corpora. The remaining loss is candidate discovery — 89 of 296 incomplete locomo questions have gold
outside the pool, 53 of 117 day misses have no gold in the pool at all — and perception. None of
these priors could ever have moved that ceiling, and each was registered as such before it was run.
H-F, measured after this verdict was recorded, confirmed it from the one direction still untested: a
fifth prior that never demotes anything still gained nothing from outside the pool.

Source: `autoresearch/orchestrator-260914/PROTOCOL.md` Amendments 2–4.

## 4. The discovery-ceiling autopsy

Registered in Amendment 2 after H-A and H-B closed: classify every pool-absent gold by the cheapest
route that would reach it — (T) temporal reference resolvable against `occurred_at`, (L) lexical
term shared with the question but lost by tokenization or fusion, (N) temporal neighbour of a
pool-resident record, (D) dense-reachable at depth 1000 or less, (P) perceptual, needing a model,
(A) annotation noise. Only a class holding at least 20 % of the pool-absent gold **and** offering a
zero-model-call route qualifies a Phase-1 hypothesis; otherwise the round reports the ceiling map as
its result. Both counts reproduce exactly: day A0 `hit@12` 0.4150, 117/200 missed, 53 pool-absent
and 64 pool-resident outside the top-12; locomo 296 incomplete@12, 89 with gold outside depth-100.

**mm-lifelong day, 53 pool-absent questions.** The second assignment removes `N`, which fails its
own null on this corpus.

| class | as registered | share | null-corrected | share | qualifies? |
| --- | ---: | ---: | ---: | ---: | --- |
| T temporal | 0 | 0.0 % | 0 | 0.0 % | no — `occurred_at` is NULL on all 2,831 records |
| L lexical | 2 | 3.8 % | 2 | 3.8 % | no — below 20 % |
| N neighbour | 26 | 49.1 % | **0** | 0.0 % | **no — lift 0.93×, below chance** |
| D ≤ 300 | 8 | 15.1 % | 23 | 43.4 % | share yes, no route to the reader |
| D ≤ 1000 | 13 | 24.5 % | 20 | 37.7 % | share yes, no route to the reader |
| P perceptual | 4 | 7.5 % | 8 | 15.1 % | no — below 20 %, and needs a model |
| A annotation | 0 | 0.0 % | 0 | 0.0 % | no |

The `N` null: 27 of 53 pool-absent questions have gold within ±4 clips of a depth-100 pool member
against a chance expectation of 29.0, because a ±4 expansion of a 100-clip pool already touches
22.1 % of all 2,831 clips. At the only seed the reader actually has, the grounded top-12, it is 8 of
53 at lift 1.00.

**locomo, 89 questions with gold outside depth-100.**

| class | n | share | qualifies? |
| --- | ---: | ---: | --- |
| N temporal neighbour / previous-turn context | **73** | **82.0 %** | share yes, signal real at lift 1.76× — fails on budget |
| D ≤ 300 | 9 | 10.1 % | no under this assignment |
| L lexical | 5 | 5.6 % | no |
| T temporal | 1 | 1.1 % | no |
| D ≤ 1000 | 1 | 1.1 % | no |
| P reasoning | 0 | 0.0 % | — |
| A annotation | 0 | 0.0 % | — |

With `N` removed everything falls into `D`: D ≤ 300 is 53 (59.6 %) and D ≤ 1000 is 30 (33.7 %).

**The two depth numbers.** On day, 90 % of questions have *at least one* gold clip dense-reachable
at depth about **420** (p50 24, p75 114, p95 806, p99 1,632); but over all 2,430 (question, gold
clip) pairs, 90 % requires depth about **2,080**, which is 73 % of the entire corpus. On locomo the
equivalent statement is categorical: **100 % of the missing gold is dense-reachable at depth 1000 or
less** (gold-pair rank CDF 0.756 @12 · 0.865 @36 · 0.936 @100 · 0.982 @300 · 1.000 @1000). Nothing
on locomo is perceptually or inferentially unreachable.

**The budget finding — the two corpora fail for opposite reasons.** On locomo the reader already
sees the whole pool: the budget pass grounded a mean of 100.0 evidence rows at 19,903 chars of the
24,000-char budget, and complete over that window is **0.9354**, not the 0.7850 the `complete@12`
slice reports. The 12-window metrics of r0912–r0914 never measured what the reader saw.

Three qualifications, all from the independent review, which reproduced the number exactly (0.9354,
99.5 rows, 19,903 chars) and verified the cost model against `context.py:140` with 0 `memory_assets`
rows in the a0 units. It is a **simulation of the grounding pass applied to a retrieval dump**: the
a0 run called `search_with_trace`, never `ask()`, and no locomo `ask()` run was inspected for its
evidence count. "The budget admits the whole depth-100 pool" is a mean — it holds for **1,258 of
1,377 questions**, and 8.6 % get less than their full pool. And it is an **exposure** number, not an
answerability one: 0.9354 complete evidence beside this project's locomo answer accuracy says the
loss has moved to the reader, which this round did not measure and which reading 100 rows instead of
12 may itself worsen.

| arm | complete | rows | chars | Δ pp | note |
| --- | ---: | ---: | ---: | ---: | --- |
| shipped budget packing, flat score order | **0.9354** | 99.5 | 19,903 | — | what ships today |
| neighbour-packed ±1, same budget | 0.9325 | 118.5 | 23,958 | −0.29 | W/L 33/37 |
| neighbour-packed ±2, same budget | 0.9477 | 117.6 | 23,965 | +1.23 | W/L 45/28, below the +2.0 pp bar |
| pool-100 ±1, budget ignored | 0.9666 | 224.9 | 45,113 | +3.12 | 1.9× budget, over on 99 % of questions |
| pool-100 ±2, budget ignored | 0.9862 | 298.1 | 59,949 | +5.08 | 2.5× budget |
| oracle: the whole conversation | 1.0000 | 602.0 | 121,462 | +6.46 | 5× budget |

At equal row count neighbours are never better than more of the ranking: top-12 ±1 gives complete
0.8315 at 30.7 rows where a flat top-31 gives 0.8584; top-12 ±2 gives 0.8874 at 46.0 rows where a
flat top-46 gives 0.8896.

On video the same budget does the opposite. A grounded clip costs `len(content) + 12,000` chars,
median **12,420**, so the 24,000-char budget admits **one** clip — yet the shipped path grounds 12
as a mandatory `hits[:limit]` prefix worth **149,040 chars, 6.2× the budget**, and `_budgeted_hits`
therefore adds **exactly 0** extra clips (confirmed in the run: all 200 day samples and 199/200 week
samples carry exactly 12 evidence rows). Depth buys candidates and never exposure.

**That exemption is the documented, tested contract, not a defect.** A separate read-only
verification checked it before this was written up: `grounding_hits` keeps `hits[:limit]`
unconditionally and `_budgeted_hits` only ever appends, its `used` counter starting at the cost of
the guaranteed hits; the rationale is stated in `plugins.py:131-135` ("raises a floor; not a
ceiling"), documented at `docs/configuration.md:369`, and pinned by
`tests/unit/test_memory_api.py:2236` (budget 10, limit 3 grounds 3 hits) along with the media-charge
test beside it. `recall_set_budget_chars` is the setting that genuinely is a ceiling. So the
observation is about naming and about an unbounded floor: `evidence_budget_chars` reads as a cap
while it is a floor, and nothing bounds what that floor costs once a hit carries a 12,000-char video
charge. Two adjacent facts sharpen the size of it. `recall_limit` is run-global with a default of 20
— the rounds cited here used 12, and at 20 the video overshoot would be about 248,400 chars. And
`video_limit 8` is a generation-stanza cap enforced after grounding, dropping video assets past the
eighth while keeping their text, so 12 clips are still charged 12,000 each in the budget arithmetic
while 8 video parts are actually sent.

**`occurred_at` is NULL on all 2,831 records of the day baseline store**
(`SELECT count(*), count(occurred_at) FROM memory_records` returns `(2831, 0)`), although the corpus
has an exact 30-second timeline in `metadata_json.start_seconds`, which is neither indexed nor a
predicate. That is why class `T` is 0 rather than small: 0 of 53 pool-absent questions carry an
absolute temporal expression at all, while **31 of 53 carry ordinal ones** (`before`, `after`,
`first`, `during`), for which the kernel has no predicate — overlap only, no Allen relations.

The NULL is a property of that store, not of HEAD: `_mm_lifelong` now writes a synthetic
`occurred_at` of `2000-01-01 UTC + start_seconds` (commit `37ff2d36`, an ancestor of HEAD), and the
day baseline store predates it; `_locomo` has always used the turn's real timestamp. The ordinal
half of the gap is untouched by that fix — a populated timestamp still has no predicate that can
express an order.

**One finding outside the registered class list, and it does not survive as a recommendation.**
Dropping the transcript-text key from the max-over-parts fusion lifts day `hit@12` from 0.4150 to
0.4500 (+3.50 pp, CI95 [+0.5, +6.5], W/L 8/1, McNemar p = 0.0391) at zero model calls and zero new
rows, and `hit@100` does not move (0.7350), so it reorders the pool rather than widening it. The
autopsy never ran it on week. The independent review did, with the same artefacts and mechanics, and
it is **−4.00 pp there (W/L 9/17, p = 0.169)**. It is also a post-hoc pick among at least five
fusion variants on a 200-question corpus, where the Bonferroni-corrected p is about 0.20. So it is a
day-only, unregistered, sign-unstable observation that earns a registered A/B, not a shippable
one-line fix. It is also **not the same phenomenon as the caption result in section 6**: the
transcript key is a *separate* key, which under max-over-parts can only raise a clip's score, while
the caption finding is about the *aggregate* being rewritten. Amendment 7's closing claim that the
two "point at the same design rule" is wrong on its own terms and is withdrawn here.

**Verdict.** No class clears both halves of the bar. `D` clears the share on both corpora and has no
route to the reader; `N` clears the share on locomo and returns +1.23 pp inside the budget; `T`,
`L`, `P` and `A` do not clear the share. Per the amendment's own fallback clause, the round reports
the ceiling map as its result.

Source: `autoresearch/orchestrator-260914/reports/autopsy.md`.

## 5. The query-form artefact

`Memory.ask()` embeds and lexically searches exactly the string its caller passed; the only text the
kernel ever adds to a question — the recall note and the reference-time note — is applied after
ranking, to the generation input. There is no product defect.

The benchmark harness is another matter. `_free_text_prompt` wraps the question as
`"Answer concisely using only the memories.\nQuestion: …\nAnswer:"`, and the adapters pass it as a
single content atom, so the whole preamble is what gets embedded and lexically searched. Of the 30
registered tasks, **20 embed a prompt-wrapped query as their only key** and 10 put the bare question
first. This has been true of locomo, longmemeval and mm-lifelong since the harness's first commit
`6c348906` (2026-08-27); the fix, `_free_text_parts`, has existed since `359dafa5` (2026-08-31) and
was applied only to m3, egolife and egomem. The wrapper adds 60 characters, +108 % over locomo's
mean bare question (55.5 chars) and +71 % over longmemeval's (85.0).

It is benign. Stripping it back to the bare question is *worse* on locomo at every cutoff (recall@12
−1.02 pp [−1.95, −0.11], complete@36 −2.18 pp [−3.12, −1.31], the loss concentrated entirely in
category 4), a wash on longmemeval's 60 near-ceiling questions, and inside noise on mm-lifelong day
(`hit@12` −1.50 pp, CI [−6.50, +3.50]). On the question this round cares about it buys nothing: gold
inside the depth-100 pool moves −0.47 pp on locomo, −0.83 pp on longmemeval and −1.00 pp on day,
every one of them the wrong sign. The one mildly better form is a bare question under a `Question:`
prefix with no instruction sentence (locomo recall@12 +0.33 pp against wrapped and +1.35 pp against
bare; day `hit@12` +1.50 pp) — a query-side formatting effect of this embedder, not a memory
mechanism.

The disclosure is wider than this round, and belongs on the record as such: **every locomo,
longmemeval and mm-lifelong number this project has published since `6c348906` was measured with a
query 60 characters longer than the question**, so any comparison drawn against a published
leaderboard row inherits it.

Recommendation, hygiene only and outside this round's scope: migrate `_locomo`, `_longmemeval`,
`_mm_lifelong` and the other single-atom adapters onto `_free_text_parts` so the harness measures
the product's own query shape, at a cost of about −1 pp on locomo, and only alongside a re-baseline.

Source: `autoresearch/orchestrator-260914/reports/queryform.md`.

## 6. The two hypotheses that ran last

### H-F — temporal evidence accumulation for continuous streams

**Mechanism.** An event is a run of clips and each clip is a noisy view of it, so score each clip by
its own cosine plus a bounded share of the match mass of its temporal neighbours inside the same
source video: `s'(c) = max(s(c), s(c) + λ·agg(neighbours) − λ·baseline)`, with the baseline either
the question's global mean (plain) or the clip's own video mean (z). No boundaries and no eviction;
the outer `max` makes the term additive and one-sided, so no clip is ever demoted. A softmax-sum
family was swept beside it. Zero model calls, and scoring is exhaustive over every clip key so the
pool can widen.

**Biological rationale.** Evidence accumulation: a noisy sensory sample is weak alone but a run of
samples of the same event integrates into a decision.

**Pre-registered kill rule.** Dev-selected (w, agg, λ) below +2.0 pp `hit@12` on day, or negative on
week; report the full grid regardless.

| split | dev-selected cell (w = 8, agg = mean, λ = 1.0) | hit@12 | Δ pp | CI95 | W/L/T | McNemar p |
| --- | --- | ---: | ---: | --- | --- | ---: |
| day (dev) | plain ≡ z, identical arrays | 0.4600 | **+4.50** | [+0.5, +8.5] | 13/4/183 | **0.0490** |
| week (holdout) | plain baseline | 0.6250 | **+2.00** | [−2.0, +6.0] | 10/6/184 | 0.4545 |
| week (holdout) | z baseline | 0.5900 | **−1.50** | [−5.5, +2.5] | 6/9/185 | 0.6072 |

A0 is `hit@12` 0.4150 (day) and 0.6050 (week). **The registered kill rule did not fire.** It reads
"dev-selected (w, agg, λ) < +2.0 pp hit@12 on day, or negative on week"; the dev-selected tuple is
+4.50 pp on day and +2.00 pp on week under the plain baseline, and the `baseline ∈ {plain, z}` axis
is not part of the registered tuple. Amendment 6 introduced a stricter reading — a registered
variant the dev split cannot select is not a variant the dev split may silently drop — and used it
to turn the pass into a kill. That reading appears nowhere before the numbers, and day cannot
arbitrate between the two baselines anyway: day is one source video, so the per-video mean and the
global mean are identical by construction and all 36 (w, agg, λ) cells are byte-identical in plain
and z. **So H-F is not a pre-registered kill. It is a judgement not to build, resting on the guards
below**, and the round's tally is five pre-registered kills and one post-hoc no-build. The day p =
0.049 is in any case uncorrected over an 80-cell grid, where Bonferroni α would be 6.25e-4.

**Why it failed mechanistically.** The registered claim was that gold *runs* sitting at ranks 13–420
are individually weak but jointly strong. Every guard says otherwise. **0 of the 23 gained questions
gained a gold clip from outside A0's top-100** (13/13 on day, 10/10 on week from inside), and
gold-in-top-100 *falls* on both splits (0.735 → 0.730 day, 0.850 → 0.845 week): the arm is a pure
re-ordering of the existing pool and moves nothing from the p90-rank-420 tail into the window.
Bucketed by gold-run size, the day gain is carried by the **single-clip-gold** questions (n = 32,
+15.62 pp, 5 of the 13 wins) — which by definition have no run to accumulate, and which are
**−3.33 pp** on week. The cross-split Spearman over the 72 accumulation cells is 0.238; 6 cells
clear +2.0 pp on day, 22 on week, 2 on both, and the best week cell is −3.50 pp on day. The
registered loss mode is also real: both week single-clip-gold losers are Language Content Recall
questions whose one gold frame, at A0 rank 9–10, is displaced by a 4–5-clip non-gold run.

**Adversarial reading.** The trivial control the guard asked for — take A0's top-36, order its
maximal runs by summed run mass, emit until 12 — is **−4.00 pp on day and −7.00 pp on week**
(McNemar 0.0161), so the dev cell is not "prefer clips in high-mass runs". What survives is a
mechanism result rather than a lever: the same neighbourhood signal read with an **additive
one-sided** operator beats every **evicting** grouping rule on both splits (H-A's best day arm
+2.00 pp day / −3.50 pp week; H-F's dev cell +4.50 / +2.00), and the symmetric version of the
identical statistic — a plain box filter with no `max` protection — collapses to −8.0 / −12.0 /
−19.5 / −24.5 pp on day at w = 1/2/4/8 because it demotes isolated-but-correct clips. The softmax
family fails for the mirror reason: at T ∈ {0.05, 0.1} the log-sum-exp is numerically the window
max, so the top-12 fills with near-duplicate neighbours of one hit and it degrades monotonically in
w (day −5.0 → −20.0 pp). That sign lesson is the durable result, and it is the round's cleanest
internally-controlled mechanism finding: state it as *a contextual prior must be additive and
one-sided, and a neighbourhood statistic must never demote an isolated hit*. What does not survive
is any claim about the size or transferability of the gain, or the registered rationale itself. Both
splits of the clamp control are also worth naming: the box filter is −8.0 / −12.0 / −19.5 / −24.5 pp
on day and +0.5 / −4.5 / −10.0 / −16.5 pp on week.

Source: `autoresearch/orchestrator-260914/reports/p0-hf.md`.

### H-E — surprise-gated perception

**Mechanism.** For a continuous stream, spend the vision-description call only on observations whose
key is novel relative to their predecessors; the rest are stored with the paid embedding only. The
claim was that novelty computed from the already-paid embedding selects the clips whose captions
carry retrieval value, so a fraction `f` of the perception budget retains most of the full-caption
gain.

**Biological rationale.** Novelty-gated encoding: prediction error is what licenses the brain to
spend costly consolidation on an observation.

**Pre-registered kill rule.** At f = 0.25, novelty retains less than 50 % of the full-caption gain
in `hit@12`, or beats the random-selection mean by less than 2.0 pp.

**The registered artefacts were wrong, and the correction is part of the result.** Amendment 3 named
`ablation-mmlifelong-day-20260912` as the WITH-captions store; it has **0 rows in
`visual_descriptions`** and `product.vision: null`. A sweep of all 1,796 `state.sqlite3` files under
`.benchmarks/data` found exactly one captioned mm-lifelong day unit, inside
`baseline-2-qwen38-27b-wemm2b-20260912`, and it was **being written during the analysis** by the
user's own `mindbridge-bench eval --config baseline_2.yaml --resume` (pid 1365516) at about 15.5
clips/min. It was read three times with `VACUUM INTO` — a read transaction that never writes the
live store — at 1,256 / 1,576 / **2,112** captioned clips (44.4 / 55.7 / **74.6 %** coverage). The
control was re-paired from the protocol's 0.415 endpoint to the ablation store, because the ablation
and the caption store share the PR #186 write path and so isolate the visual description instead of
confounding it with the removed context key. No captioned week store exists, so the week guard is
reported as a gap, not a null.

| metric (control → captions, 74.6 % coverage, n = 200) | control | captions | Δ pp | CI95 | W/L/T | McNemar p |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| hit@12 | 0.4000 | 0.3350 | **−6.50** | [−12.0, −1.5] | 8/21/171 | **0.024** |
| cov@12 | 0.1814 | 0.1408 | −4.07 | [−6.97, −1.24] | 13/38/149 | — |
| all@12 | 0.0950 | 0.0600 | −3.50 | [−6.5, −1.0] | 1/8/191 | **0.039** |

**The −6.50 pp is a partial-coverage number, and the independent review re-ran it at full
coverage.** The caption store kept ingesting after the round's snapshots. The reviewer took a fourth
read-only snapshot at **2,815 of 2,831 captioned clips = 99.4 % coverage**, extracted it with the
round's unmodified `he_extract.py` and ran the round's unmodified mechanics:

| arm (day, n = 200, control = the ablation no-caption store) | 74.6 % (round) | **99.4 % (review)** |
| --- | ---: | ---: |
| full captions, hit@12 | **−6.50** [−12.0, −1.5] p = 0.024 | **−3.00** [−8.5, +2.5] **p = 0.39**, W/L 14/20 |
| full captions, hit@36 | +1.00 | **+3.00** |
| **graft: caption as an extra key, aggregate untouched, hit@12** | +1.00 [−1.0, +3.0] p = 0.625, W/L 3/1 | **+3.00 [+1.0, +5.5] p = 0.031, W/L 6/0** |
| graft, hit@36 | — | **+3.50 [+0.5, +6.5] p = 0.039** |
| graft, cov@12 | −0.10 | **+1.07 [+0.17, +2.15]** |

Two things follow and both are corrections to what this round said. First, **the
coverage-monotonicity defence is broken.** `p0-he.md` answered the "it is just a partial prefix"
objection by observing that the loss grew with coverage (−5.50 → −5.00 → −6.50 pp) and named a
full-coverage snapshot as the one remaining measurement; that measurement came back at −3.00 pp and
not significant. The headline effect halves and loses significance once the partial-coverage rank
bias is removed — captioned clips took 83.5 % of the top-12 while being 74.6 % of the corpus, and
96.8 % of clips newly promoted into a top-12 were captioned, with a constant **−15.5 pp** penalty
(49 losses, 0 wins) on questions whose gold was left un-captioned. Second, **the kill rule's first
clause is no longer vacuous**: there is a gain, +3.00 pp in the confound-free construction with CIs
excluding zero on three metrics. The statements "there is no caption gain to allocate" and "a
caption has no retrieval value" are withdrawn.

**The kill stands, on the second clause.** Re-running the graft budget sweep at 99.4 % coverage: f =
0.10 spans +0.5…+1.5 pp, f = 0.25 +1.0…+2.5 pp and f = 0.50 +1.5…+3.0 pp — but random tracks it (f =
0.25 random mean 0.413 against a best novelty of 0.425), **the label-leaking oracle is 0.410, below
random**, and the best novelty-minus-random margin is **+0.7…+1.2 pp** against a +2.0 pp bar. At the
round's own 74.6 % coverage the same margin was −0.00 pp (caption store) and +0.30 pp, one changed
question (graft), and at f = 0.10 in the graft construction every selector including the oracle
changed exactly zero of 200 questions. The allocation question is inert at every coverage, and the
mechanism claim — that novelty selects the clips whose captions carry retrieval value — is false.

**The mechanism attribution survives, as a mechanism and not as a magnitude.** The reviewer
decomposed the control-to-caption delta by grafting single key roles across the two stores, an
attack built specifically to break the attribution:

| arm (day, n = 200) | 74.6 % coverage | 99.4 % coverage |
| --- | ---: | ---: |
| full captions (caption in the aggregate and as a key) | −6.50 | −3.00 |
| **control's keys with only the aggregate replaced** | **−5.00** (W/L 7/17) | **−2.50** |
| full captions with the control's aggregate restored | **+0.50** | — |
| control's keys with only the caption store's transcript key swapped in | +1.00 | — |
| control's keys plus the caption key (the graft) | +1.00 | **+3.00** |

The aggregate rewrite carries about 77 % of the loss at 74.6 % coverage and restoring the control's
aggregate removes all of it. The round's own §5 reached the same place from the other side: the
caption's dedicated key is the max-scoring part for **0.28 %** of (query, captioned-clip) pairs and
2.1 % of grounded top-12 slots, against 71.3 % for the aggregate, and deleting it changes `hit@12`,
`cov@12` and `all@12` by exactly zero. The confound the round worried about — the caption store's
transcript key differing from the control's at median cosine 0.53 through inlined speaker UUIDs — is
worth +1.00 pp and is not the story. So: **where the derived text goes decides the sign**, and the
size of the penalty depends on how much of the corpus is enriched.

**Adversarial reading.** Novelty is below chance as a predictor of what matters: gold-ness AUC
0.447–0.476 at every window, on a unimodal band with 90 % of clips between 0.06 and 0.25 — the same
geometry that killed H-A, seen from the encoding side. The short-window variants are a shot-change
detector with extra steps (set overlap 0.780 with `novelty_vid_5`) and are the worst arms; the cheap
transcript-length selector, which needs no embedding at all, lands mid-table, and novelty never
beats it. The label-leaking oracle — spending the budget on exactly the clips that are gold — is
+0.00 pp at f = 0.25 and never above +1.5 pp at any f in the round's own runs, so there is no
headroom for any selector to capture — at 99.4 % coverage the oracle sits *below* random. The
"vision fills the silence" story is also unsupported: silent clips only and the longest-transcript
quartile are both −1.00 pp. Cost, measured from the stores' own timestamps rather than estimated:
captioning takes the day ingest from **49 min to 182 min** for 2,831 clips and adds +52.5 % stored
keys per captioned clip, breaching the frozen 10 % cost rule at f ≥ 0.20 by the caption's own key
alone. The separate 66 → 49 min improvement is the PR #186 write path and has nothing to do with
captions; the two must not be compressed into one figure.

Source: `autoresearch/orchestrator-260914/reports/p0-he.md`.

With H-E the round closed (Amendment 7). Its result is the ceiling map of section 4, **five
pre-registered kills and one post-hoc no-build** (H-F, below), and one write-path measurement that
survived independent re-running: enrichment placed in a record's aggregate key costs, and the same
text placed beside it as a separate key gains.

## 7. What this round establishes for future rounds

Do not re-test the following on these corpora.

- **Any rank-order prior over the depth-100 pool.** Four independent mechanisms — rehearsal
  strength, identity conditioning, write-time novelty, episode grouping — each returned at most
  +2 pp with interference of the same order, and each was bounded a priori by what the pool already
  contains.
- **Episode grouping, temporal de-duplication and MMR-shaped diversity on continuous video.** The
  gold for a question is a run of consecutive clips, so every diversity constraint evicts gold
  before distractors, monotonically in the strength of the constraint.
- **The C1 premise that near-duplicate 30-second clips flood the index.** False at the resolution
  the embedder sees: median adjacent cosine 0.826 (day) / 0.842 (week), and only 12 % of recoverable
  gold is crowded out by a same-episode clip.
- **Rehearsal on locomo-shaped corpora.** A label-leaking oracle caps at +5.88 pp, no realizable
  promotion rule exceeds +0.36 pp, and the honest proxy is negative.
- **Identity priors on any corpus whose adapter writes the speaker name into the indexed text.** The
  retriever has already spent that signal: a 50 % base rate becomes 92.8 % of the top-12.
- **Query-string reformulation as a discovery lever.** Every form tested moves gold-in-pool the
  wrong way on all three corpora.
- **Temporal accumulation over clip neighbourhoods (H-F).** The +4.50 pp day cell is a re-ordering
  of the existing pool: 0 of 23 gained questions gained gold from outside the top-100,
  gold-in-top-100 falls on both splits, the gain sits in the single-clip-gold bucket that has no run
  to accumulate (+15.62 pp day, −3.33 pp week), and the holdout is a null +2.00 pp [−2.0, +6.0].
  Re-test only with a corpus of several source videos on the dev split, so the plain and z baselines
  can be told apart.
- **Allocating a perception budget by novelty (H-E).** There is no gain to allocate: full captions
  are −6.50 pp, the confound-free graft is +1.00 pp ns, and at f = 0.10 every selector including the
  label-leaking oracle changes exactly zero of 200 questions. Novelty is below chance as a gold
  predictor (AUC 0.447–0.476) and cheaper selectors match it.

Open design directions the data actually supports.

- **Ordinal and sequence temporal predicates.** 31 of 53 pool-absent day questions carry `before`,
  `after`, `first` or `during`; the kernel's only predicate is interval overlap. Two separable gaps
  sit behind that number: the day store has `occurred_at` NULL on all 2,831 records while an exact
  30-second timeline exists unindexed in metadata, and even a populated timestamp has no predicate
  that can express an order. The autopsy cannot score this hypothesis — there is no temporal signal
  in the store to score — so it is an open direction, not a measured one.
- **An out-of-band identity test needs a corpus that does not exist offline today.** The
  requirements are explicit: a diarised ingest that binds the producer through `identities` /
  `speech_analyses` with the speaker string stripped from the indexed text; queries that refer to
  people the way users do, by role, kinship or pronoun, which needs an alias-to-identity step no
  benchmark exercises; and a measurement on scores rather than ranks, since a bounded multiplicative
  prior cannot be simulated on rank-only dumps. H-C is closed on evidence, not on principle.
- **Enrichment as a separate key, with the aggregate untouched — registered as H-G, tested, closed.**
  The round's one positive measurement was the day graft: at 99.4 % caption coverage, adding the
  caption as its own retrieval key and leaving the record's aggregate alone is **+3.00 pp `hit@12`
  [+1.0, +5.5] p = 0.031, W/L 6/0**, confound-free by construction (the control's own record, one
  extra key, the same speaker ids). Amendment 9 registered it as H-G with a pooled-holdout rule and a
  product-level transfer check; Amendments 10–12 record the outcome.
  *Holdouts* (offline graft, zero model calls): memlens-32k 0.00 pp and memlens-256k +0.58 pp are
  inert — captions sit on 11 % of records and the dataset's own caption is already in every such
  record's text. **mm-lifelong week, the only holdout that exercises the mechanism at day's coverage
  (99.87 % of 6,266 clips captioned), is −0.50 pp `hit@12` [−2.5, +1.0], W/L/T 1/2/197.** Pooled
  over the three holdouts, clustered by question (n = 546): **0.00 pp [−0.74, +0.73]** — the rule's
  first clause (positive with a CI excluding zero) fails and its second (no split below −1.0 pp)
  holds. On week the caption key re-orders 75 of 200 top-12 sets and moves gold on three questions,
  one in and two out; the product's own recorded rankings put the captioned week store at 0.00 pp
  `hit@12` against the no-caption store (W/L 33/33), so the day observation that a caption inside
  the aggregate hurts does not recur either.
  *Transfer*: a product re-ingest of the day unit under the patched kernel is **+1.50 pp `hit@12`
  against the captioned store [−3.5, +6.5], p = 0.68** — the registered bar met on the point estimate
  and on nothing stronger, and only on a reindexed copy: the check found that a store ingested from
  scratch on this branch lineage gets a Zvec index whose vectors are bound to the wrong records (0.15
  overlap with its own exhaustive ranking, 0.994 after a plain reindex), a blocking defect
  independent of H-G that is registered for its own root cause and fix.
  **H-G is closed for this round.** The merge condition — holdout and transfer both passing — is not
  met, and `r0914/aggregate-key-composition` stays unmerged. What survives: the day number as a
  single-split, single-video result; the finding on all three corpora that the caption key is the
  argmax part for only 0.3–0.5 % of (query, clip) pairs; and the cost fact that the graft is free
  where it is inert (−6.6 % embed keys on day, from the silent-clip dedup). Every week number here
  is a replay-proxy number (see the proxy caveat below) except the product-level C − A.
- **Exposure on video, if the floor is to be bounded.** The reader's window is fixed at 12 clips by
  a 12,000-char-per-asset charge, and the guaranteed `hits[:limit]` prefix is 6.2× the evidence
  budget while `_budgeted_hits` adds zero. That is the documented contract, so the open work is
  naming and a bound on what the floor may cost, not a fix to the grounding rule. Both registered
  attempts to route around it are now closed: H-E found nothing to allocate and H-F found nothing
  outside the pool.

**Limits.** Everything here is retrieval-only on three corpora: locomo (1,377 questions, 10
conversations), longmemeval (60 questions, one per unit, baseline `complete@12` 0.90, so one
question is 1.67 pp), and mm-lifelong day and week (200 questions each). The locomo and longmemeval
replays are dense-only proxies of a hybrid ranking (mean top-12 overlap 81.7 % / 85.8 % with the
product's stored order). No answer was scored by a judge in this round. The only product-code
change, H-G's key-composition patch, lives on an unmerged branch, and its one harness run was the
transfer check.

**The mm-lifelong week proxy is not the product, and it is stronger than the product.** Week replay
overlap@12 is **0.630** and replay `hit@12` is **0.605** against the product's recorded **0.415** —
19 pp better, because the shipped week path also runs a lexical route over Chinese prose that the
replay does not model. It is therefore a different ranker with different headroom and a different
failure structure, not a noisier copy. Two verdicts lean on it. H-A's decisive clause is
*dev-selected arm negative on week*, since its day clause is marginal (+1.50 pp registered, +2.00 pp
with the τ = 0.85 extension, "equals, does not exceed"); H-A survives anyway on the day mechanism,
where only 12 % of pool-resident missed gold is crowded out and the low-τ sweep is monotonically
negative. H-F's entire holdout is a week number in both directions — the +2.00 pp that would have
passed and the −1.50 pp that would have killed. Week and day must not be read as two independent
confirmations of anything, and no week number in this note is a product result. Day is faithful
(overlap@12 0.974, `hit@12` identical) but is a single source video at n = 200, which is exactly why
H-F's plain-versus-z baseline could not be arbitrated.

**Pre-registration here is attested by process, not by an independent record.** `PROTOCOL.md` and
every report live under `autoresearch/`, which `.gitignore` excludes from the repository, so no
independent record of any earlier protocol state exists. `PROTOCOL.sha256` was appended by hand —
three different path spellings across its lines, two identical consecutive digests, no timestamps,
no signatures — and a list of digests of the current file can only detect a state you can no longer
produce; it cannot detect an amendment being edited and re-hashed. The `receipts/` and
`review-receipts/` files carry no tool name, no arguments and no hash, and every line reads
`signed: false`, so they have no evidentiary value and are not cited as provenance. What does check
out is internal consistency: the read-time digests cited by the H-A, H-D, query-form, H-F and H-E
reports are all in the chain, in an order consistent with each hypothesis being registered before
its numbers were read, and no amendment edits a prior hypothesis's kill-rule text. To give the claim
something a reader can check, `PROTOCOL.md` is copied verbatim into this documentation set as [the
r0914 protocol](2026-09-14-memory-dynamics-protocol.md).

**Four numbers in the round's data files have no producer in `tools/`** and are noted wherever they
are used: `class_counts_null_corrected` and `klass_null_corrected` in the day autopsy, and
`class_counts_if_N_not_a_route` and `klass_if_N_is_not_a_route` in the locomo autopsy. Re-running
the autopsy scripts regenerates everything else byte-identically and silently drops these, and the
null-corrected class table in section 4 is the table the Amendment-2 verdict is written against. The
same applies to the H-F exploratory box-filter control and split files; the reviewer recomputed the
box filter independently and matched it exactly, so the numbers are right and only the provenance is
missing.

## 8. Reproduction

Round directory layout, all under `autoresearch/orchestrator-260914/`:

```text
PROTOCOL.md         pre-registration plus Amendments 1-12, appended in order, never rewritten
PROTOCOL.sha256     the digest chain, one line appended per state of PROTOCOL.md
reports/            kernel_map.md, p0-ha.md … p0-hf.md, autopsy.md,
                    autopsy-budget-verify.md, queryform.md, review-round.md,
                    hg-feasibility.md, hg-holdout.md, hg-review.md, hg-transfer.md,
                    hg-holdout-week.md
tools/              the analysis scripts, each with a --selfcheck mode
p0/                 the numeric outputs and cached query vectors the reports cite
receipts/           tool receipts for the round
review-receipts/    tool receipts for the adversarial reviews
```

Scripts, by hypothesis: `ha_common.py` / `ha_run.py` / `ha_explore.py` (H-A), `p0_hb.py` (H-B),
`p0_hc.py` (H-C), `p0_hd.py` (H-D), `he_extract.py` / `he_common.py` / `he_endpoints.py` /
`he_run.py` / `he_diag.py` / `he_mech.py` (H-E), `hf_common.py` / `hf_run.py` (H-F),
`hg_common.py` / `hg_run.py` / `hg_pool.py` / `hg_week.py` / `hg_transfer_search.py` /
`hg_transfer_score.py` (H-G),
`autopsy_common.py` / `autopsy_day.py` / `autopsy_locomo.py` / `autopsy_routes.py` /
`autopsy_examples.py` (the autopsy), `qf_run.py` / `qf_day.py` (query form). Each runs under
`/home/yons/thomas/MindBridge/.venv/bin/python` with `PYTHONPATH` pointing at this worktree's `src`
where product code is imported, and each carries a `--selfcheck` that must pass before its report's
numbers are read. Bootstraps are seeded (seed 42, 10,000 resamples); everything else is
deterministic.

`PROTOCOL.sha256` holds fourteen appended lines with thirteen distinct digests — the
pre-registration plus Amendments 1–12 — and the last line, `26d45f1d…`, is the digest of the file as
it stands. Reports name the digest they were registered against: `69086e1f…` for H-A's start, `c4b8299c…` for Amendment
1 as H-A finished, `c3240ee7…` for H-D and the query-form ablation, `3f715e72…` for H-F, and
`dbac7b7e…` for H-E, `fe193294…` for the H-G memlens holdout, `7500bbe0…` for the H-G week holdout
and transfer check, and `7f52ed10…` / `26d45f1d…` for Amendments 11 and 12 as written. That
ordering is internally consistent with each hypothesis being registered before its numbers were
read, and no amendment edits a prior hypothesis's kill-rule text — but read
the integrity paragraph in section 7 before treating the chain as an audit trail, because it is
hand-appended inside a directory the repository does not track. The protocol itself is reproduced
byte for byte, inside a fenced block so that nothing is reformatted, at [the r0914 pre-registered
protocol](2026-09-14-memory-dynamics-protocol.md); it is the only copy of it inside this repository.
