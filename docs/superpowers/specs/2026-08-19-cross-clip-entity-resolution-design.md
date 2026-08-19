# Cross-clip entity resolution

> Status: approved design, not yet implemented
> Date: 2026-08-19
> Origin: M3-Bench robot subset diagnosis — one 7.5-minute video produced 17 `person`
> entities for 3 real people

## Problem

Perception runs per clip. Each run names what it sees, and it names the same person
differently every time. One video (`living_room_22`, 15 clips) produced these `person`
entities:

```text
man in blue denim jacket          woman in pink sweater
man in denim jacket               woman in pink sweatshirt
man in black t-shirt              woman in pink hoodie
man in black t-shirt (denim jacket removed)   woman in pink sweatshirt and green skirt
person in black t-shirt           person in pink sweater
person in black shirt on sofa     person in reddish top on sofa
camera operator (first-person, unseen)        camera wearer
unseen camera operator (hand visible)         two people seated on living room sofa
```

Three real people. Seventeen nodes. Every fact about the man is split across six of them,
so nothing accumulates and no multi-hop question about a person can traverse.

### What already works, and why it is not enough

Two merge mechanisms exist:

1. `derive_observation_graph.py` derives `entity_id` from
   `(tenant_id, entity_type, casefold(canonical_name))`. Identical names already collapse
   across clips. The failure is lexical variation, not a missing merge.
2. When `identity_observations` accompany an observation, the entity is keyed by
   `identity_id` with `canonical_name = None` — a stable anonymous person across clips.
   This is the designed cross-clip identity path and it works, but it needs an edge
   identity signal. The caption path is contractually barred from carrying one
   (`M3PreparedClip` rejects `identity_observations` without `media_object`), and the only
   available released clustering for M3 collapses 152 of 153 observations in this very
   video into a single character, merging two visibly different people.

So the gap is narrow and specific: **decide identity semantically when no edge identity
signal exists, without ever merging two different entities.**

## Success criteria

Graph correctness first; benchmark score is explicitly not a gate. A preceding experiment
in this same investigation showed that optimising an M3 number can move the metric while
degrading behaviour, so the acceptance bar here is a labelled precision/recall measurement,
not an accuracy delta.

- **Merge precision = 1.0 is a hard gate.** A written `same_as` edge that joins two
  different real entities fails the change.
- Recall is reported, not gated. Under-merging is the acceptable failure.

## Approved decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Where | A fourth consolidation kind, beside episodes/claims/summaries | Same cadence, same control plane, no new operational surface |
| Effect | Additive `same_as` edge; `entity_id` is never rewritten | Matches the `supersedes`/`contradicts` precedent: reversible and auditable |
| Transitivity | None. Pairwise only, no connected components, no cluster id | A single misjudgement pollutes one pair instead of collapsing a cluster |
| Signal | Generator adjudication over the original AV, shortlisted by type/time | Appearance-text similarity is the wrong signal for identity (below) |
| Recall integration | Out of scope for this change | Traversal is where regressions land; it gets its own measured decision |

### Why not embedding similarity as the decision

Rejected as a *decision* signal, kept only as a shortlist ordering. The entities that must
merge are lexically far apart (`man in blue denim jacket` and
`man in black t-shirt (denim jacket removed)` are the same man after he takes the jacket
off), and entities that must not merge are lexically close (`woman in pink sweater` and
`person in pink hoodie` could be two people dressed alike). Text similarity over appearance
descriptions inverts the very cases this exists to handle.

### Why not feed prior entity names back into the perception prompt

It would prevent fragmentation at the source, but it forces the model to reuse a stale
description — the man really did remove the jacket, and pinning the old name makes the new
description wrong. It also grows the prompt without bound and repairs nothing already
written.

## Architecture

Every component mirrors an existing one; the shapes are copied deliberately.

| New | Mirrors |
| --- | --- |
| `application/entity_resolution.py` — pure domain: candidates plus adjudications produce the write set | `application/semantic_claims.py` |
| `application/consolidate_entities.py` — `ConsolidateEntities.run(EntityCandidateRequest) -> EntityCandidateResult` | `application/consolidate_claims.py` |
| `consolidate_tenant_entities` + `EntitySweepSummary` in `application/consolidation_sweep.py` | `consolidate_tenant_claims` in the same file |
| `infrastructure/_postgres_entity_resolution.py` — candidate page query and atomic write | `infrastructure/_postgres_claim_consolidation.py` |
| `application/pipelines/entities.py` — adjudication pipeline | `application/pipelines/claims.py` |
| `RESOLVE_ENTITIES_PROMPT` (`resolve_entities_v1`) in `prompts.py` | `consolidate_claims_v2` |
| `--entity-*` options in `consolidation_cli.py` | existing `--claim-*` options |
| Migration `0019_entity_resolution_edges.sql` | `relations_claim_consolidation_idx` |

No new table. The generic `relations` row
`(tenant_id, relation_id, source_type, source_id, relation_type, target_type, target_id, created_at)`
already carries this edge; the migration adds only a partial index for
`source_type = 'entity' AND target_type = 'entity'`.

## Data flow

1. **Page.** Cursor over `entities` by `entity_id` at one fixed `evaluated_at`, `page_size`
   per page, exactly like the claim sweep, including its "cursor did not advance" guard.
2. **Shortlist.** For each page entity, candidates are same tenant, same `entity_type`,
   `canonical_name IS NOT NULL`, with a mention within `maximum_gap_seconds`, capped at
   `candidate_limit` (default 8) and ordered by entity-embedding cosine. Entities with a
   null `canonical_name` are skipped on both sides: they are identity-backed and already
   stable, and guessing about them would undo a stronger signal.
3. **Canonicalise.** Emit each unordered pair once, as `entity_id_a < entity_id_b`. This
   halves the work and makes the edge naturally unique.
4. **Skip settled pairs.** A pair already carrying `same_as` or `not_same_as` in either
   direction is skipped unless `--entity-readjudicate` is set.
5. **Adjudicate.** For each surviving pair take up to `evidence_per_side` mentions per
   side, resolve them through the existing `resolve_evidence_media` and `evidence_parts`,
   and call the Generator with `RESOLVE_ENTITIES_PROMPT`. Structured output:
   `{"same_entity": bool, "confidence": float, "discriminating_cue": str}`.
6. **Write.** `same_entity` true and `confidence >= minimum_confidence` writes `same_as`;
   an explicit false writes `not_same_as`. Anything else writes nothing.
   `relation_id = derive_stable_id("relation", tenant_id, relation_type, a, b)`, so a
   re-run is a no-op and concurrent sweeps stay idempotent.

## Error handling

The load-bearing rule: **`not_same_as` means "inspected and judged different". It must
never mean "could not inspect".**

| Condition | Behaviour |
| --- | --- |
| Media missing or signed URL expired | Skip the pair, count it, write no edge |
| Generator returns invalid JSON | Existing `generate_json` retry-once; still invalid, skip and write no edge |
| `ModelUnavailableError` / `ModelRequestError` | Propagate so the sweep retries later; never record a negative from an infrastructure failure |
| Entity deleted between page read and write | The atomic write guard drops the pair |
| `maximum_pairs` reached | Stop, and log the count dropped — a bounded sweep must say what it did not look at |

## Cost

Pairs per sweep are bounded by `entities × candidate_limit / 2`. The observed video has 119
entities (77 object, 20 place, 17 person, 5 device), so an all-types sweep is up to 476
adjudications, each carrying media.

Two bounds, both defaults:

- **`--entity-types` defaults to `person`.** The finding is about people; whether
  `blue tissue box` and `tissue box` merge is worth much less at identical risk. This caps
  the observed video at 68 pairs.
- **`--entity-maximum-pairs`** hard-stops a sweep, with the dropped count logged.

The full option set on `mindbridge consolidate`, matching the `--claim-*` shape:
`--entity-page-size`, `--entity-maximum-gap-seconds`, `--entity-candidate-limit`,
`--entity-minimum-confidence`, `--entity-evidence-per-side`, `--entity-maximum-pairs`,
`--entity-types`, `--entity-readjudicate`.

## Testing

**Labelled measurement.** The 17 `person` entities of `living_room_22` are hand-labelled to
their real-world referents, giving the full truth table over C(17,2) = 136 pairs. The
labelling scheme needs a third value beside a person id: `two people seated on living room
sofa` is a collective that corresponds to no single person, so it takes `not-an-individual`
and joins no positive pair while the adjudicator must still refuse to merge it with anyone.
A snapshot of the entities, mentions, events and evidence spans is kept outside the
database, because the benchmark tenants are shared and can be cleared.

Reported: merge precision (gate: 1.0) and recall (reported).

**Unit,** with a fake adjudicator, in `tests/unit/application/test_entity_resolution.py`:

- non-transitivity — adjudicating A~B and B~C produces no A~C edge
- ordered-pair canonicalisation — one edge regardless of input order
- default-deny on each of the four failure conditions above
- `relation_id` stability across runs
- null `canonical_name` entities are never shortlisted

**Contract:** the prompt joins the existing fingerprint catalog test, which fails on any
text change without a version bump.

**Integration,** marked `pytest.mark.integration`: the candidate page query and the atomic
write against PostgreSQL.

## Out of scope

- Recall traversal of `same_as`. Written, unused by retrieval, decided separately.
- Merging non-person types by default.
- Any change to `entity_id` derivation or to the `identity_observations` path.
- Repairing already-written fragmented entities beyond adding edges between them.
