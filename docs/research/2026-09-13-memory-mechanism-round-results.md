# Memory-mechanism round r0912: results

Historical branch report, recovered from commit `05567858`. The tools and optional settings
described below belong to that experimental branch and are not supported by the current SDK.
This archive preserves the original measurements; it does not adopt the experimental code.

Results for the pre-registration in
[2026-09-12-memory-mechanism-preregistration.md](2026-09-12-memory-mechanism-preregistration.md).
Every arm listed there is reported here. Numbers are quoted from the run artefacts under
`.benchmarks/r0912-ab/` and are not recomputed.

## Summary

Hypothesised: retrieval keys embedded without their conversational context (H1) and the absence of
an associative second hop (H2) are what lose multi-evidence questions. Measured: retrieval-only
A/B on `locomo-refined` (1,377 questions, 10 conversations) and `longmemeval-s` (60 questions,
seed 42), at the product ranking depth 100 with post-hoc @12/@36 cutoffs.

- H1b (512-char context) — not adopted. complete@12 0.7850 → 0.8017 (+79/−56, p = 0.0579) on
  locomo-refined, 0.9000 → 0.8500 (+2/−5) on longmemeval-s. Fails rule 1 (neither corpus reaches
  p < 0.05, signs disagree), rule 2 (single-gold recall@12 −3.85 pp) and rule 3 (input characters
  3.02×), the last as Amendment 9 predicted.
- H1b-lite (128-char context) — not adopted. Rule 1 fails in the negative direction on
  locomo-refined (0.7850 → 0.7647, +42/−70, p = 0.0104); indistinguishable from A0 on
  longmemeval-s (+1/−1). The 512 → 128 cut removes the signal, not the cost (+32/−83, p = 0.0000).
- H2 (Rocchio second hop) — refuted and closed: 0.7850 → 0.7255 (+2/−84, p = 0.0000).
  H1b+H2 is worse than either alone (0.7335).
- H4 (episodic contiguity prior, registered after the locomo autopsy) — refuted on its
  longmemeval-s holdout: 0.9000 → 0.8333, recall@12 −4.3 pp, CI [−0.092, −0.006].
- Determinism (rule 4): A0 vs A0-dup deltas exactly 0.0000 on every stratum and both metrics.

All four candidate mechanisms are refuted or null under the frozen rules. The deliverables are the
tool (`mindbridge-bench retrieval-ab`, c98bc19a), the defect fixes (439241bb) with the
`contextual_turn_characters` knob and `index_queries` in the trace, the measured noise floor, and
the headroom map below. Contextual encoding stays `off`; H2 is being removed from the code.

## Protocol as executed

Each arm was ingested once into its own store through the public SDK and the harness adapters,
then queried at `limit=100` with `reference_at` fixed to `2026-09-12T00:00:00Z`, batch size 32,
embedder `tencent/WeMM-Embedding-2B` (declared and served dimension 2048, preflight hit on every
store). No generation model is in the loop. Statistics: unit-clustered paired bootstrap (10,000
resamples, seed 42) for recall, exact McNemar for complete; corpora never pooled.

Amendments, all recorded before the arm they govern was read:

1. H1a (date prefix only) is untestable here: both adapters already prefix each turn with the ISO
   event time and `[source_id: …]`, so a date adds no information. Dropped; only H1b measured.
   The kernel-level gap — time is never encoded for applications that do not put it in content —
   stands.
2. The harness never passed `ObservationContext`, so session identity had to come from the adapter
   passing `context=` on ingest, which is what an application would do. The tool does that.
3. `limit` changes the candidate set (temporal deepening and the widening loop both key on it), so
   a `limit=36` arm is not a slice of the product arm. Every arm runs at `limit=100`; @12 and @36
   are post-hoc cutoffs of one ranking.
4. The bootstrap resamples units (10 on locomo-refined), so its CI is coarse there; longmemeval-s
   is one unit per question. A per-question bootstrap is a secondary descriptive column.
5. `reference_at` fixed per row, because locomo-refined questions carry no reference time.
6. A0 ingested twice into fresh directories in separate processes, compared before any other arm.
7. `add_many` must let a record see predecessors inside its own batch; batch size is an arm
   parameter (32).
8. Noise floor measured from Amendment 6, as the bar an arm on a separate store must clear.
9. H1b at 512 characters cannot satisfy rule 3 on these corpora (a LoCoMo turn is 150–250
   characters, so the prefix alone is 3–6× the content). H1b-lite at 128 characters was registered
   as the arm that can. The capture→settle defect found in review is excluded from this
   measurement and fixed separately.

H4 was registered after the locomo-refined autopsy, with locomo-refined declared the development
set (three λ explored) and longmemeval-s the only confirmation, λ = 0.2 fixed before it was read.

Noise floor: two independent A0 ingests give recall@12 and complete@12 differences of exactly zero;
6 of 1,377 questions swap one candidate at rank 11/12 (ANN boundary ties, `exhaustive=false`) and
640 differ only in order inside the same candidate set. The stores are not bit-identical because
the remote WeMM embedder is batch-nondeterministic: ~9 % of document vectors differ between
ingests, cosine ≥ 0.99987, per-component |Δ| median 1.18e-3. Query embeddings are reproducible and
the ranking is deterministic, so exact ranked-ID identity is not required; the metric floor is 0.

## Results

### Determinism, locomo-refined, k=12 (A0 vs A0-dup)

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8466 | 0.8466 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0.7850 | 0.7850 | +0/-0 | 1.0000 |
| single-gold | 1035 | 0.9150 | 0.9150 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0.9150 | 0.9150 | +0/-0 | 1.0000 |
| multi-gold | 342 | 0.6397 | 0.6397 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0.3918 | 0.3918 | +0/-0 | 1.0000 |

### H1b, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8466 | 0.8551 | +0.0085 | [-0.0017, +0.0172] | [-0.0055, +0.0225] | 0.7850 | 0.8017 | +79/-56 | 0.0579 |
| single-gold | 1035 | 0.9150 | 0.9304 | +0.0155 | [+0.0020, +0.0276] | [+0.0000, +0.0309] | 0.9150 | 0.9304 | +41/-25 | 0.0640 |
| multi-gold | 342 | 0.6397 | 0.6272 | -0.0126 | [-0.0488, +0.0208] | [-0.0441, +0.0182] | 0.3918 | 0.4123 | +38/-31 | 0.4704 |

### H1b, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9564 | 0.9072 | -0.0492 | [-0.1056, +0.0008] | [-0.1056, +0.0008] | 0.9000 | 0.8500 | +2/-5 | 0.4531 |
| single-gold | 26 | 0.9615 | 0.9231 | -0.0385 | [-0.1154, +0.0000] | [-0.1154, +0.0000] | 0.9615 | 0.9231 | +0/-1 | 1.0000 |
| multi-gold | 34 | 0.9525 | 0.8951 | -0.0574 | [-0.1348, +0.0147] | [-0.1348, +0.0147] | 0.8529 | 0.7941 | +2/-4 | 0.6875 |

### H2, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8466 | 0.7877 | -0.0589 | [-0.0712, -0.0469] | [-0.0700, -0.0481] | 0.7850 | 0.7255 | +2/-84 | 0.0000 |
| single-gold | 1035 | 0.9150 | 0.8725 | -0.0425 | [-0.0538, -0.0316] | [-0.0551, -0.0309] | 0.9150 | 0.8725 | +0/-44 | 0.0000 |
| multi-gold | 342 | 0.6397 | 0.5311 | -0.1086 | [-0.1472, -0.0759] | [-0.1341, -0.0843] | 0.3918 | 0.2807 | +2/-40 | 0.0000 |

### H2, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9564 | 0.9314 | -0.0250 | [-0.0667, +0.0000] | [-0.0667, +0.0000] | 0.9000 | 0.8667 | +0/-2 | 0.5000 |
| single-gold | 26 | 0.9615 | 0.9231 | -0.0385 | [-0.1154, +0.0000] | [-0.1154, +0.0000] | 0.9615 | 0.9231 | +0/-1 | 1.0000 |
| multi-gold | 34 | 0.9525 | 0.9377 | -0.0147 | [-0.0441, +0.0000] | [-0.0441, +0.0000] | 0.8529 | 0.8235 | +0/-1 | 1.0000 |

### H1b+H2 vs A0, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8466 | 0.7814 | -0.0652 | [-0.0879, -0.0443] | [-0.0829, -0.0484] | 0.7850 | 0.7335 | +55/-126 | 0.0000 |
| single-gold | 1035 | 0.9150 | 0.8821 | -0.0329 | [-0.0484, -0.0205] | [-0.0512, -0.0145] | 0.9150 | 0.8821 | +31/-65 | 0.0007 |
| multi-gold | 342 | 0.6397 | 0.4765 | -0.1632 | [-0.2393, -0.0932] | [-0.2030, -0.1239] | 0.3918 | 0.2836 | +24/-61 | 0.0001 |

### H1b+H2 vs A0, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9564 | 0.8628 | -0.0936 | [-0.1658, -0.0283] | [-0.1658, -0.0283] | 0.9000 | 0.8000 | +2/-8 | 0.1094 |
| single-gold | 26 | 0.9615 | 0.8462 | -0.1154 | [-0.2692, +0.0000] | [-0.2692, +0.0000] | 0.9615 | 0.8462 | +0/-3 | 0.2500 |
| multi-gold | 34 | 0.9525 | 0.8755 | -0.0770 | [-0.1554, -0.0059] | [-0.1554, -0.0059] | 0.8529 | 0.7647 | +2/-5 | 0.4531 |

### H1b+H2 vs H1b, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8551 | 0.7814 | -0.0737 | [-0.0913, -0.0580] | [-0.0865, -0.0619] | 0.8017 | 0.7335 | +2/-96 | 0.0000 |
| single-gold | 1035 | 0.9304 | 0.8821 | -0.0483 | [-0.0639, -0.0358] | [-0.0618, -0.0357] | 0.9304 | 0.8821 | +0/-50 | 0.0000 |
| multi-gold | 342 | 0.6272 | 0.4765 | -0.1507 | [-0.1938, -0.1109] | [-0.1785, -0.1236] | 0.4123 | 0.2836 | +2/-46 | 0.0000 |

### H1b+H2 vs H1b, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9072 | 0.8628 | -0.0444 | [-0.1000, +0.0000] | [-0.1000, +0.0000] | 0.8500 | 0.8000 | +1/-4 | 0.3750 |
| single-gold | 26 | 0.9231 | 0.8462 | -0.0769 | [-0.1923, +0.0000] | [-0.1923, +0.0000] | 0.9231 | 0.8462 | +0/-2 | 0.5000 |
| multi-gold | 34 | 0.8951 | 0.8755 | -0.0196 | [-0.0637, +0.0196] | [-0.0637, +0.0196] | 0.7941 | 0.7647 | +1/-2 | 1.0000 |

### The `all` stratum at k=36

Rows quoted from the `cutoff36` compare files; the same tables carry the single-gold and
multi-gold strata.

| comparison | corpus | recall@36 base | recall@36 cand | delta | unit CI95 | complete base | complete cand | McNemar +/- | p |
| --- | --- | ---: | ---: | ---: | :--- | ---: | ---: | :--- | ---: |
| A0 vs H1b | locomo-refined | 0.9217 | 0.9240 | +0.0022 | [-0.0090, +0.0132] | 0.8715 | 0.8736 | +46/-43 | 0.8323 |
| A0 vs H1b | longmemeval-s | 0.9819 | 0.9903 | +0.0083 | [+0.0000, +0.0250] | 0.9500 | 0.9667 | +1/-0 | 1.0000 |
| A0 vs H2 | locomo-refined | 0.9217 | 0.8829 | -0.0388 | [-0.0456, -0.0316] | 0.8715 | 0.8243 | +6/-71 | 0.0000 |
| A0 vs H2 | longmemeval-s | 0.9819 | 0.9819 | +0.0000 | [+0.0000, +0.0000] | 0.9500 | 0.9500 | +0/-0 | 1.0000 |
| A0 vs H1b+H2 | locomo-refined | 0.9217 | 0.8823 | -0.0394 | [-0.0582, -0.0226] | 0.8715 | 0.8257 | +35/-98 | 0.0000 |
| A0 vs H1b+H2 | longmemeval-s | 0.9819 | 0.9542 | -0.0278 | [-0.0833, +0.0194] | 0.9500 | 0.9333 | +2/-3 | 1.0000 |
| H1b vs H1b+H2 | locomo-refined | 0.9240 | 0.8823 | -0.0417 | [-0.0517, -0.0315] | 0.8736 | 0.8257 | +7/-73 | 0.0000 |
| H1b vs H1b+H2 | longmemeval-s | 0.9903 | 0.9542 | -0.0361 | [-0.0917, +0.0056] | 0.9667 | 0.9333 | +1/-3 | 0.6250 |

### Ingest efficiency

From the arm manifests written by `mindbridge.benchmarks.retrieval_ab ingest`.

| arm | corpus | embed_calls | embed_inputs | embed_input_chars | wall seconds | bytes_on_disk |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A0 | locomo-refined | 189 | 5882 | 1,188,233 | 50.84 | 217,280,512 |
| A0-dup | locomo-refined | 189 | 5882 | 1,188,233 | 93.17 | 217,280,512 |
| H1b | locomo-refined | 189 | 5882 | 3,588,730 | 60.09 | 217,280,512 |
| H1b-lite | locomo-refined | 189 | 5882 | 2,126,216 | 55.52 | 217,280,512 |
| A0 | longmemeval-s | 954 | 41765 | 49,047,098 | 435.78 | 1,587,907,386 |
| H1b | longmemeval-s | 954 | 41765 | 69,007,064 | 494.22 | 1,577,601,850 |
| H1b-lite | longmemeval-s | 954 | 41765 | 55,518,882 | 481.62 | 1,587,907,386 |

Rule 3 for H1: embedding requests per record are unchanged (identical `embed_calls` and
`embed_inputs`); bytes on disk are identical on locomo-refined and −0.65 % on longmemeval-s, inside
the 1 % band; input characters are 3.02× on locomo-refined and 1.41× on longmemeval-s, so the 1.5×
bound fails. H1b-lite does not repair it: 1.79× on locomo-refined (1.13× on longmemeval-s), bytes
identical to A0 on both. Amendment 9's premise — that the shorter budget is the variant which can
satisfy rule 3 — is wrong on locomo-refined, where turns are short enough that even 128 characters
is comparable to the content prefixed.

### Latency and index queries

Quoted from the compare files (p50/p95 of `latency_ms` over all queried questions).

| arm | locomo-refined p50/p95 ms | longmemeval-s p50/p95 ms | index_queries per search |
| --- | --- | --- | --- |
| A0 | 64.4 / 85.4 | 77.5 / 95.6 | 2 on 1177/1377 and 55/60; 3–6 on the rest |
| A0-dup | 63.2 / 87.6 | — | same distribution as A0 |
| H1b | 62.4 / 87.1 | 76.0 / 101.6 | identical to A0 (write-path change only) |
| H1b-lite | 63.8 / 87.2 | 67.8 / 98.2 | identical to A0 (write-path change only) |
| H2 | 68.7 / 91.0 | 76.1 / 101.6 | exactly A0 + 1 |
| H1b+H2 | 69.4 / 92.5 | 75.7 / 103.8 | exactly A0 + 1 |

H2's p95 is 1.066× baseline on locomo-refined and 1.063× on longmemeval-s, inside its 1.20× bound:
the hop is cheap, it is simply wrong.

### End-to-end reference

One `mindbridge-bench eval` run of A0 on locomo-refined
(`.benchmarks/results/r0912-a0-locomo`): llm_judge 0.7779, CI95 [0.7571, 0.8053], n = 1,382,
21,878 tokens per question, abstentions 69/1,382. Single arm, no blind control: a reference point
for the next round's confirmation runs, not a result.

## H2 autopsy

H2 lost 84 questions' complete@12 on locomo-refined and gained 2; 93 gold records were pushed out
(median A0 rank of a displaced gold 7.0, Q1/Q3 5.0/10.0). All 646 displacers entering H2's top 12
were hop-only candidates absent from A0's depth-100 set, and none was within 0.95 cosine of a seed.

| quantity (pooled over the 84 losing questions) | n | median [Q1, Q3] |
| --- | ---: | :--- |
| cos(q, displaced gold) | 93 | 0.486 [0.463, 0.505] |
| cos(centroid, displaced gold) | 93 | 0.633 [0.608, 0.660] |
| cos(q, every record of the conversation) | 50664 | 0.342 [0.280, 0.390] |
| cos(centroid, every record of the conversation) | 50664 | 0.501 [0.436, 0.552] |

The centroid sits inside document space, so its cosine against an arbitrary record is inflated by
≈ 0.16 relative to the query's. No fixed damping factor can reconcile the two scales: the hop's
scores are not comparable with the first route's, and 0.9 damping only reorders ties.

The premise fails too. Gold partners are no closer to a found gold than an arbitrary adjacent turn
is: gold_i vs gold_j 0.587 [0.529, 0.650] against 0.602 [0.541, 0.661] for immediate previous/next
records and 0.475 [0.416, 0.532] for a random record of the same conversation. On an oracle seed
(≥ 1 gold at rank ≤ 3 and ≥ 1 gold outside the top 12: 120 questions, 186 pairs), ranking the whole
conversation by max cosine to the found gold puts the missed partner in its own top 12 for
38/186 = 20.4 %, and 47 of those partners are not in A0's depth-100 set at all. A perfect seed
recovers a fifth of what the hop is aimed at.

Offline counterfactual re-ranks of A0's depth-100 set (ranks 1–3 pinned) separate the two ideas.
Rescoring by cosine hurts (`pure cos(q, d)` re-rank: complete@12 0.7734, +25/−41); a flat
same-session bonus in rank space helps on this corpus (λ = 0.2: 0.8017, +48/−25, p = 0.0095;
λ = 0.3: 0.8032, +64/−39, p = 0.0176). That observation is what became H4.

## H4 holdout refutation

H4 was fitted on locomo-refined (complete@12 0.7850 → 0.8017, recall@12 +0.0137
CI [+0.0039, +0.0237], +48/−25, p = 0.0095 at λ = 0.2) and confirmed only on longmemeval-s, with
λ fixed beforehand.

| variant | stratum | n | complete@12 A0 | complete@12 H4 | recall@12 A0 | recall@12 H4 | recall delta | question CI95 | McNemar +/- | p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: |
| lambda=0.2 (primary) | all | 60 | 0.9000 | 0.8333 | 0.9564 | 0.9133 | -0.0431 | [-0.0917, -0.0056] | +1/-5 | 0.2188 |
| lambda=0.2 (primary) | single-gold | 26 | 0.9615 | 0.9231 | 0.9615 | 0.9231 | -0.0385 | [-0.1154, +0.0000] | +0/-1 | 1.0000 |
| lambda=0.2 (primary) | multi-gold | 34 | 0.8529 | 0.7647 | 0.9525 | 0.9059 | -0.0466 | [-0.1029, +0.0025] | +1/-4 | 0.3750 |
| lambda=0.3 (secondary) | all | 60 | 0.9000 | 0.8000 | 0.9564 | 0.8928 | -0.0636 | [-0.1153, -0.0203] | +1/-7 | 0.0703 |

Rule 1 fails (complete@12 down) and rule 2 fails (single-gold recall@12 −3.85 pp). The
development gain was a property of locomo-refined's few-session conversations: 26.0 % of its gold
pairs share a session against 11.6 % on longmemeval-s, where 0 % share one by source-id prefix.
H4 is closed.

## Where the headroom is

On locomo-refined, multi-gold complete@12 is 0.3918 and 933 gold records split as 562 (60.2 %) in
the top 12, 157 in ranks 13–36, 116 in 37–100 and 98 outside depth 100 — 26.4 % of missed
multi-gold records are never candidates. That fraction is unreachable by any re-ranking hypothesis
measured here; it is a discovery problem, and encoding is the only registered lever for it.

On longmemeval-s, A0 already places 103 of 109 gold records inside the top 12 and every gold inside
depth 100 (single-gold 25/26, multi-gold 78/83). Text retrieval on these corpora is near its
ceiling, and the two disagree about which prior helps — a development-set gain on one is not a
result.

The two context budgets give a dose–response in the wrong shape for a cheap fix: at 512 characters
the prefix at least carries the topic (complete@12 0.8017), at 128 it does not and the arm falls
below A0 (0.7647), +32/−83, p = 0.0000. A short prefix adds noise without carrying the topic, so
the budget cannot be tuned down to meet rule 3.

The loss is elsewhere, as the [2026-09-12 decomposition](2026-09-12-baseline-loss-decomposition.md)
measured: with every gold record in the window the reader still scores 0.831 (LoCoMo), 0.842
(LongMemEval) and 0.681 (ATM-main); on LoCoMo that is 13.3 % of the whole task lost inside the
reader against 21.5 % of questions with any retrieval shortfall. Reordering the same 12 records
chronologically moved the "wrong with all gold present" stratum 0.028 → 0.233, and widening
12 → 36 rows moved the near-miss stratum 0.331 → 0.613 at ~5× answer tokens. ATM-main refuses 14 %
of questions whose gold media is already in the prompt, which is a representation problem, not a
ranking one.

## H1b-lite

Pre-registered as Amendment 9: the same encoding as H1b with the context budget cut from 512 to
128 characters. Measured on the same corpora, against A0 and against H1b.

### H1b-lite vs A0, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8466 | 0.8252 | -0.0214 | [-0.0344, -0.0078] | [-0.0343, -0.0085] | 0.7850 | 0.7647 | +42/-70 | 0.0104 |
| single-gold | 1035 | 0.9150 | 0.9005 | -0.0145 | [-0.0289, -0.0019] | [-0.0290, +0.0000] | 0.9150 | 0.9005 | +21/-36 | 0.0627 |
| multi-gold | 342 | 0.6397 | 0.5974 | -0.0424 | [-0.0710, -0.0156] | [-0.0716, -0.0142] | 0.3918 | 0.3538 | +21/-34 | 0.1048 |

### H1b-lite vs A0, locomo-refined, k=36

| stratum | n | recall@36 base | recall@36 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.9217 | 0.9126 | -0.0092 | [-0.0191, +0.0005] | [-0.0188, +0.0005] | 0.8715 | 0.8613 | +34/-48 | 0.1507 |
| single-gold | 1035 | 0.9623 | 0.9585 | -0.0039 | [-0.0137, +0.0049] | [-0.0145, +0.0068] | 0.9623 | 0.9585 | +13/-17 | 0.5847 |
| multi-gold | 342 | 0.7988 | 0.7736 | -0.0252 | [-0.0492, +0.0003] | [-0.0483, -0.0026] | 0.5965 | 0.5673 | +21/-31 | 0.2116 |

### H1b-lite vs H1b, locomo-refined, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 1377 | 0.8551 | 0.8252 | -0.0299 | [-0.0394, -0.0183] | [-0.0428, -0.0169] | 0.8017 | 0.7647 | +32/-83 | 0.0000 |
| single-gold | 1035 | 0.9304 | 0.9005 | -0.0300 | [-0.0411, -0.0169] | [-0.0444, -0.0155] | 0.9304 | 0.9005 | +15/-46 | 0.0001 |
| multi-gold | 342 | 0.6272 | 0.5974 | -0.0298 | [-0.0625, -0.0016] | [-0.0570, -0.0030] | 0.4123 | 0.3538 | +17/-37 | 0.0091 |

### H1b-lite vs A0, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9564 | 0.9353 | -0.0211 | [-0.0550, +0.0056] | [-0.0550, +0.0056] | 0.9000 | 0.9000 | +1/-1 | 1.0000 |
| single-gold | 26 | 0.9615 | 0.9615 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 0.9615 | 0.9615 | +0/-0 | 1.0000 |
| multi-gold | 34 | 0.9525 | 0.9152 | -0.0373 | [-0.0971, +0.0118] | [-0.0971, +0.0118] | 0.8529 | 0.8529 | +1/-1 | 1.0000 |

### H1b-lite vs A0, longmemeval-s, k=36

| stratum | n | recall@36 base | recall@36 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9819 | 0.9861 | +0.0042 | [-0.0125, +0.0250] | [-0.0125, +0.0250] | 0.9500 | 0.9667 | +1/-0 | 1.0000 |
| single-gold | 26 | 1.0000 | 1.0000 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | 1.0000 | 1.0000 | +0/-0 | 1.0000 |
| multi-gold | 34 | 0.9681 | 0.9755 | +0.0074 | [-0.0221, +0.0441] | [-0.0221, +0.0441] | 0.9118 | 0.9412 | +1/-0 | 1.0000 |

### H1b-lite vs H1b, longmemeval-s, k=12

| stratum | n | recall@12 base | recall@12 cand | delta | unit CI95 | question CI95 (secondary) | complete base | complete cand | McNemar +/- | p |
| --- | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | :--- | ---: |
| all | 60 | 0.9072 | 0.9353 | +0.0281 | [-0.0156, +0.0778] | [-0.0156, +0.0778] | 0.8500 | 0.9000 | +4/-1 | 0.3750 |
| single-gold | 26 | 0.9231 | 0.9615 | +0.0385 | [+0.0000, +0.1154] | [+0.0000, +0.1154] | 0.9231 | 0.9615 | +1/-0 | 1.0000 |
| multi-gold | 34 | 0.8951 | 0.9152 | +0.0201 | [-0.0382, +0.0789] | [-0.0382, +0.0789] | 0.7941 | 0.8529 | +3/-1 | 0.6250 |

Verdict: H1b-lite fails rule 1 on locomo-refined in the negative direction — complete@12
0.7850 → 0.7647 (+42/−70, p = 0.0104), recall@12 CI [−0.0344, −0.0078] — and is indistinguishable
from A0 on longmemeval-s (+1/−1). Not adopted.

## Defects found and fixed

Two adversarial reviews (of the tool and of 00148fc1) produced 439241bb. None of it changes the
embedding input `add`/`add_many` builds at the default budget, or any hop score when every seed is
usable.

- F1 — the prefix was bounded by `occurred_at <= ?` alone, so a chunk captured before its session
  settled keyed turn one on turns two and three, and a formed record read its source's successors.
  Now cut at the newest row the record must follow — itself once stored, or `anchor_memory_ids`
  for a derived record (`kernel/contracts.py`, `kernel/ingestion.py`,
  `infrastructure/local/store/records.py`).
- F3 — `text_selector_map` keyed on the unprefixed `ModelInput` while `inputs()` emitted the
  prefixed one, so under any encoding but `off` a record stored no text selectors at all. The
  prefix is now built once through `Embedding.key_prefix` and keys both (`kernel/embedding.py`,
  `kernel/content.py`, `kernel/ingestion.py`).
- F4 — REST, MCP and SDK references for the hop fields 00148fc1 added and did not document
  (`docs/api/rest.md`, `docs/api/mcp.md`, `docs/api/python-sdk.md`).
- F6 — `known_metadata_upgrade` compared against literal recipes with no contextual segment, so the
  next recipe bump would have refused to open a `time+turns` store; both sides now come from one
  segment builder (`infrastructure/local/store/_schema.py`).
- F7 — an empty preceding turn ended `_bounded_turns` instead of being skipped, dropping every
  older turn behind it (`infrastructure/local/store/records.py`).
- F8 — the damping claim corrected wherever it appeared: damping loses every tie at equal cosine
  but is applied before the ranking signals, which can still lift a hop-only candidate.
- F9 — `RetrievalCandidateTrace.associative`, true for a candidate only the hop found, so a damped
  `dense_relevance` is attributable after the fact (`kernel/retrieval.py`, `types.py`).
- F12/F13 — a hop protected `associative_seed_hits` ranks even when fewer seeds were readable; it
  now protects only the ranks usable seeds earned (`kernel/ranking.py`).
- New knob `MemoryConfig.contextual_turn_characters` (default 512), named in the index recipe only
  when it can move vectors, so every recipe string in use today is byte-identical
  (`kernel/settings.py`, `memory.py`).

Known limitations, unfixed and recorded:

- F5 — re-embedding for an embedding-space migration re-derives the prefix from the live store, so
  a re-embed is not guaranteed to reproduce the original ingest's vectors under `time+turns`.
- The capture→settle prefix is now bounded by rowid, which is an ordering proxy, not a clock.
- Dedup collapses identical turns across sessions, so a repeated utterance carries the first
  session's context.

## Next-round candidates

Each needs the falsification test named; none is worth building without it.

- Contextual encoding for the discovery-limited stratum, at full length only — the 128-character
  budget is measured and dead. Run it solely on a corpus where gold is demonstrably outside depth
  100 (locomo-refined multi-gold: 26.4 %). Falsified if that share does not fall while
  complete@12 holds.
- Any future associative hop must re-score candidates against the query and use the centroid only
  as a recall route. Falsified if hop-only candidates admitted this way still displace gold at the
  rate H2 did (93 displaced golds over 84 questions).
- Evidence-sufficiency estimator: stop at 12 rows when 12 suffice, widen to 36 when they do not.
  Framed as cost, not accuracy — falsified if it cannot hold complete@k while cutting the ~5×
  answer-token cost of the 36-row window. It borders the answer layer, so it needs a separate
  guard against becoming a prompt experiment.
- Media representation: speech text in records is confirmed as a gain, vision captions are refuted.
  Falsified if a new representation does not move ATM-main's 14 % refusal-with-gold-present rate.
- Embedding cache keyed on content hash, to remove the remote embedder's ~9 % vector drift.
  Falsified if two ingests of the same corpus still produce differing vectors.

## Question set

The 60 longmemeval-s question IDs (seed 42, sampled before any run), as recorded in the arm
manifests:

```text
001be529 01493427 0bc8ad93 0e5e2d1a 0f05491a 10d9b85a 10e09553 129d1232 1a8a66a6 1da05512
1faac195 22d2cb42 28dc39ac 2bf43736 2e6d26dc 35a27287 38146c39 3b6f954b 3e321797 3fdac837
4fd1909e 58bf7951 5a4f22c0 60159905 6456829e_abs 66f24dbb 6b168ec8 71017277 71315a70 72e3ee87
75f70248 81507db6 8a137a7f 8e9d538c 8fb83627 94f70d80 993da5e2 9bbe84a2 a1cc6108 a82c026e
a96c20ee aae3761f bbf86515 bcbe585f c8f1aeed c960da58 ce6d2d27 dad224aa e3038f8c edced276
ef66a6e5 efc3f7c2 gpt4_1d4ab0c9 gpt4_2d58bcd6 gpt4_4cd9eba1 gpt4_59149c78 gpt4_5dcc0aab
gpt4_70e84552_abs gpt4_8279ba02 gpt4_93159ced
```
