# Memory-mechanism round r0912: pre-registration

Status: pre-registered before any implementation. Hypotheses, metrics, controls, and decision
rules below are frozen; results go in a separate note and must report every arm listed here.

## Why this round is different

Previous score movements came mostly from the answer layer (`best_effort` policy, 24k-character
evidence budget) or from fixing plumbing (score completion, speech text in records). The loss
decomposition of 2026-09-12 shows that on text benchmarks gold is almost never *absent* from the
36-candidate window (1.7–9.2%), but on multi-evidence questions ranking finds the turn that
shares words with the question and misses its partner. Widening the grounded window 12→36 rows
lifted that stratum 0.331→0.613 at ~5× answer tokens. That is a symptom of the memory
mechanism, not of the reader: records are encoded and ranked as if independent of the
conversation they came from.

Two mechanism facts (see kernel map): the embedding input is content text only — no event time,
no session, no neighbouring turns (`kernel/hydration.py`, `kernel/embedding.py`); and every
record's FP32 vector is already persisted in SQLite and read at query time by score completion
(`infrastructure/local/scoring.py`).

## Hypotheses

H1 — Contextual encoding (write path). Retrieval keys for a record should be embedded with the
context an agent will later need to find it: the event date as text and a bounded prefix of the
preceding turns from the same source (`ObservationContext.source_id`). Stored content, record
identity, dedup digest, and lexical document stay unchanged; only the embedding input changes.
Prediction: complete-gold coverage@12 rises on multi-evidence questions; single-gold recall does
not fall; embedding calls per record unchanged; storage unchanged.

H2 — Associative second hop (read path). After the first ranking, use the stored vectors of the
top-m hits to run one more dense route (Rocchio-style centroid, plus per-hit routes if cheap) and
fill ranks m+1..k from the union with damped scores. Zero model calls, one or two extra Zvec
queries. The original top-m stays fixed to bound query drift. Prediction: complete-gold
coverage@12 approaches the coverage the 36-row window buys, at 12-row evidence cost; p95 search
latency within +20%.

Pre-registered but deferred (needs a generation model at write time, so it is a cost trade-off):
H3 — raw + session-summary union consolidation as indexable derived records.

Already refuted, not retried: positional ±1 neighbour expansion, deeper single-route depth
100→200, caption formation on video, formation-derived records as a QA lever, prompt wording.

## Corpora and sample

- `locomo-refined`: all conversations; all questions with gold evidence IDs.
- `longmemeval-s`: a fixed random subset of 60 questions, seed 42, sampled before any run;
  question IDs recorded in the results note. Chosen because 62% of its questions cite >1 gold
  record.
- Media routes are out of scope for this round (representation ceiling is a different problem).

## Metrics (retrieval-only, no generation model in the loop)

Per question, from the true product ranking (`Memory.search`, not the harness `random` arm):

- recall@12 — share of gold records in the top 12.
- complete@12 — 1 if every gold record is in the top 12 (primary metric).
- recall@36 / complete@36 — to relate to the 36-row grounding window.
- Efficiency: search p50/p95 latency; ingest embedding requests and input characters; SQLite +
  Zvec bytes on disk; number of Zvec queries per search.

Statistics: paired bootstrap (10 000 resamples, seed 42) on the per-question difference for
recall@12; exact McNemar for complete@12; both corpora reported separately, never pooled.

## Arms

- A0 baseline: current master mechanism, `limit=12`.
- A0-wide control: baseline with `limit=36` (what the 24k budget effectively buys).
- H1a: date prefix only. H1b: date + previous 2 turns (≤512 chars) from the same source.
- H2: associative hop with m=3 fixed, fill to 12. H2 is evaluated on the same ingested store as
  A0 (query-time only), which also serves as a check that the store is identical.
- H1b+H2 combined, only if both pass individually.

## Decision rules (frozen)

Adopt an arm as default only if all hold:

1. complete@12 improves on both corpora with the McNemar p < 0.05 and the recall@12 bootstrap CI
   excludes zero on at least one corpus and does not exclude zero in the negative direction on
   the other.
2. Single-gold questions: recall@12 does not drop by more than 1 pp on either corpus.
3. H2: p95 search latency ≤ 1.20× baseline. H1: embedding requests per record unchanged and
   embedding input characters ≤ 1.5× baseline; bytes on disk unchanged within 1%.
4. An adversarial review of the measurement finds no leakage (gold IDs must never reach ranking),
   no metric redefinition after the fact, and byte-identical A0 results when the new code path is
   disabled.

An arm that moves recall but fails an end-to-end confirmation (one `mindbridge-bench eval` run per
arm on the same subset with the standard generation model) is reported as "retrieval gain, answer
null", as identity-anchored keys were, and stays off by default.

## Amendments (recorded before any arm was measured)

1. H1a is untestable on `locomo-refined` and `longmemeval-s`: their adapters already prefix each
   turn's content with the ISO event time and `[source_id: …]`, so a date prefix adds no
   information. H1a is dropped for this round; only H1b (preceding turns) is measured. The
   kernel-level gap (time never encoded for applications that do not put it in content) stands.
2. The harness never passes `ObservationContext`; session grouping for H1b must come from the
   adapter passing `context=` on ingest, which is what an application would do.
3. Withdrawn: `limit` does change the candidate set (temporal deepening and the widening loop
   both key on it), so a `limit=36` run is not a slice of the product arm. Every arm is measured
   at `limit=100`, the depth `ask()` uses, and @12/@36 are post-hoc cutoffs of that ranking.
4. The bootstrap resamples units (10 for locomo-refined), so its CI is coarse there; longmemeval-s
   is one unit per question. McNemar on complete@12 is per question on both. When a stratum has
   fewer than two units the clustered CI is reported as n/a; a per-question bootstrap is printed
   as a secondary, descriptive column only.
5. `reference_at` is fixed to `2026-09-12T00:00:00Z` for every arm and recorded per row, because
   locomo-refined questions carry no reference time and the temporal factor otherwise moves with
   the wall clock.
6. Determinism check for decision rule 4: A0 is ingested twice into fresh directories in separate
   processes and the two `query` outputs must be identical before any candidate arm is compared.
7. Batched ingestion (`add_many`) must let a record see the predecessors in its own batch; that
   is a write-path requirement for H1b, not a tool setting, because applications stream sessions
   in batches too. Batch size is recorded as an arm parameter.
8. Measured noise floor (Amendment 6 executed on locomo-refined, 1,377 questions): two
   independent A0 ingests give recall@12 and complete@12 differences of exactly zero; 6 questions
   swap one candidate at rank 11/12 (ANN boundary ties, `exhaustive=false`), 640 differ only in
   order within the same candidate set. Exact ranked-ID identity is therefore not required; an
   arm measured on a separate store (H1b) must beat this floor, and H2 is measured on the same
   store as A0, where no such noise exists.
9. Registered before any H1b result was read: H1b as implemented (512-character context budget)
   cannot satisfy decision rule 3 on these corpora, because a LoCoMo turn is 150–250 characters
   and the prefix alone is 3–6× that. H1b is still measured and reported, but it fails rule 3 by
   construction. A variant H1b-lite with a 128-character context budget is added as the arm that
   can satisfy rule 3. Both are ingested with `add_many`; the capture+settle path has a known
   defect (same-timestamp successors leak into the prefix) that is fixed separately and excluded
   from this measurement.

## Round-2 hypotheses registered after the locomo-refined autopsy (2026-09-13)

H2 is closed as refuted (locomo-refined complete@12 0.785→0.726, +2/−84). Its autopsy showed a
scale mismatch (a Rocchio centroid's cosine against any record is inflated by ≈0.16 relative to
the query's, so no fixed damping works) and that gold partners are not closer to found gold than
an arbitrary adjacent turn is (0.587 vs 0.602). Any future hop must re-score candidates against
the query and use the centroid only as a recall route.

H4 — Episodic contiguity prior (read path, zero extra queries). Within the existing depth-100
candidate set, a candidate that shares its session (`memory_semantics.source_id`) with any of the
top-3 ranked hits receives a flat rank-space bonus of λ = 0.2 (score = (100 − rank)/100 + λ);
ranks 1–3 stay fixed. Motivation: the temporal contiguity effect in human free recall; the
harness now passes session identity, so the kernel can use it. locomo-refined is the development
set for this hypothesis (three λ values were explored there: complete@12 +0.017, +48/−25,
p = 0.010 at λ = 0.2). **Only longmemeval-s (60 questions, seed 42) counts as confirmation**, with
λ fixed at 0.2 before it is examined; λ = 0.3 is reported as secondary only. Decision rule 1
applies on longmemeval-s alone; rule 2 (single-gold recall not down > 1 pp) applies on both.
Adjacent refuted idea: ±1 positional neighbour expansion added rows; H4 adds none.

Discovery ceiling recorded: 26.4% of locomo-refined multi-gold records that miss the top 12 are
not in the depth-100 set at all; no re-ranking hypothesis can reach them. H1 (encoding) is the
only registered lever for that stratum.

H4 holdout result (recorded 2026-09-13, λ fixed before examination): longmemeval-s complete@12
0.900→0.833 (+1/−5), recall@12 −4.3 pp with CI [−0.092, −0.006]; single-gold recall −3.85 pp.
Fails rules 1 and 2. H4 is refuted and closed. longmemeval-s A0 already places 103/109 gold
records inside the top 12 and every gold inside depth 100; the locomo-refined development gain
was a property of its few-session conversations (26% co-session gold pairs vs 11.6%).
