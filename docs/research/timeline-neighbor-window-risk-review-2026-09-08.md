# Timeline neighbor-window risk review

**Status:** architecture hypothesis only; not implemented or benchmark-validated.

**Question:** if `ObservationContext` gains an explicit `TimelineSpan`, how can an anchor retrieve
nearby records from the same sequence without increasing the existing context budget or weakening
causal and provenance guarantees?

The current product contract does not provide this relationship. `ObservationContext` has an
opaque `source_id` and a validity interval, while `Memory.add` and `StreamInput` accept timezone-aware
calendar `occurred_at`/`occurred_end` values. None of those fields is a typed cross-clip sequence and
relative-offset contract. M3's clip `start_seconds`/`end_seconds` values are benchmark-application
metadata and were not supplied as product occurrence intervals. Product code must not hard-code
those benchmark fields or infer adjacency from a `source_id` naming convention.

## Required contract

`TimelineSpan` needs a producer-assigned opaque sequence identity and a half-open position interval
`[start_seconds, end_seconds)`. A restart or epoch may be a separate field or encoded into the
producer-unique sequence identity; that public-contract choice remains open. Transport `stream_id`,
wall-clock proximity, `source_id`, place, and matching media type are insufficient sequence
identities. A camera restart, imported file, or a second robot must resolve to another sequence even
if its visible stream name is reused. The backend must reject invalid intervals and must not infer a
timeline from timestamps.

SQLite remains authoritative. It should index the producer sequence identity and position for a
bounded range query; if the public contract represents an epoch separately, that field belongs in
the same key. Zvec remains a rebuildable anchor-discovery index. Neighbor expansion should read raw
observation records only. A derived fact can expand through its explicit evidence IDs and then
through each raw source's timeline; derived-record adjacency would mix formation order with world
chronology.

## Safety invariants

1. **Same sequence is an equality constraint.** Every neighbor must match the producer-unique
   sequence identity, including its restart or epoch semantics. Missing or conflicting span metadata
   disables expansion for that anchor. No fallback may join by nearby time, place, identity,
   filename, or semantic similarity.
2. **Causal filtering precedes adjacency.** If the query declares a relative cutoff for this
   timeline, a record is eligible only when its complete `span.end_seconds <= relative_cutoff`; a
   segment crossing the cutoff is excluded because its content may contain the future. If a raw
   observation also has a real calendar occurrence range, the existing occurrence-time filter is
   applied independently. Knowledge-time scope remains a third, independent constraint. SQL must
   apply every applicable predicate before returning neighbors. A raw video without a real calendar
   time must not fabricate `occurred_at` from its relative span. With no declared relative cutoff,
   expansion follows existing visibility constraints but cannot claim an online causal prefix.
3. **Adjacency never means causality or support.** Render a relation such as `timeline_neighbor` with
   signed distance and interval. Do not call it a cause, prerequisite, corroborating witness, or
   evidence for the anchor's claim. Do not copy the anchor's relevance or confidence onto it or add
   its source to a derived fact's supporting-evidence set.
4. **Version lineage stays separate.** `supersedes_id`, validity intervals, and evidence IDs retain
   their current meanings. A nearby older observation must not revive an invalidated fact, while a
   later correction must not be hidden merely because chronological rendering puts it last.
5. **The budget is unchanged.** Anchors and neighbors compete inside the same `ContextBudget`,
   including media-item and character or token limits. Start with at most one predecessor and one
   successor per anchor and one global neighbor cap. A neighbor may be dropped; truncating a video
   record into a fictitious causal prefix is not allowed.

## Avoiding ordinary-query regressions

Expansion should be gated by measurable retrieval signals, not described as calibrated answer
uncertainty. Candidate signals include an explicit before, after, or sequence query, uncovered query
parts, disagreement between dense and lexical routes, a small score margin, or a retrieved visual
segment whose raw context is requested. A simple fact query with one high-margin, full-coverage
anchor should keep today's path and cost.

Neighbor admission needs its own low prior or marginal-coverage test; raw temporal distance cannot
replace semantic relevance. Context rendering should preserve anchor labels and group their neighbors
chronologically, so added context cannot masquerade as another top-ranked answer. Measure regressions
both when the feature fires and when it should remain dormant.

## Acceptance evidence before implementation can ship

- Property tests generate interleaved timelines, reused stream names, restarts, backfills, overlapping
  spans, corrections, and cutoff-crossing records; no returned neighbor may cross sequence identity,
  a declared relative cutoff, an applicable occurrence-time bound, or `known_at`.
- A generation-request canary proves that future and cross-stream content never reaches the model,
  rather than merely checking returned IDs.
- Every expansion records anchor ID, relation, signed distance, eligibility cutoff, budget cost, and
  drop reason. Raw evidence chronology and source or version lineage remain inspectable.
- Held-out ablations compare current retrieval with neighbor expansion under identical total context,
  embedding and model calls, and answer model. Report ordinary fact accuracy, source recall,
  future-leak violations, same-burst redundancy, latency, SQLite reads, media count, and context tokens.
- The experiment is rejected if gains require benchmark-specific window sizes, if ordinary fact
  queries regress outside the declared noise band, or if a late filter is needed to remove records
  that should never have been admitted.

Coverage, hierarchy, temporal windows, and expansion all have prior art. A possible contribution
would lie in a rebuildable embedded implementation that jointly enforces causal cutoffs, sequence
isolation, version lineage, and a fixed context budget without formation or query-time model calls.
