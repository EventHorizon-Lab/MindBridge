# Predicate stability: exact aliases and correction evidence

This report records the original investigation and its measurements. The current runtime uses
schema 20, accepts no older schema, and keeps predicate provenance without migration lineage.
The migration regressions below describe the evaluated implementation at
the time; current storage behavior is documented in [operations](../operations.md#changing-the-runtime-or-stored-representations).

## Decision

M03 is reproducible: formation hashes the normalized predicate literal into a claim's lineage.
Standing naming assertions already bind recognized subjects; that mechanism remains unchanged.
The [September writer evaluation](2026-09-08-memory-backend-evaluation.md) audited 27 inferred
STATE rows without confirming a contradictory
alias, so production incidence remains unknown. This justifies a bounded deterministic registry,
not unrestricted semantic merging.

Recent practice supports explicit schemas and conservative equivalence: Graphiti exposes
[custom edge types and pair mappings](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types),
while [SKOS](https://www.w3.org/TR/skos-reference/#mapping) distinguishes exact equivalence from
non-transitive close matches. These are design precedents, not evidence of MindBridge incidence
or of the correctness of an arbitrary alias.

Use immutable predicate identifiers, an explicitly versioned exact-alias registry, and a scalar
label guard. Preserve original predicates and persist the resolution method. Existing kind and
basis rules determine cardinality; an alias never changes those rules. Keep home city and current
location separate and leave unknown, ambiguous, and compound predicates literal. No model, prompt,
dependency, or public response field changes are needed.

Revision 1 declares three label slots: preferred drink, home city, and current location. Accepted
kinds are STATE, TRAIT, and RELATION. The value guard rejects absent, oversized, non-printable,
JSON-container, and common delimiter-separated values. It is a syntax check, not a model-independent
proof that free text denotes a beverage or a city. Ambiguous paraphrases such as `likes` remain
unregistered. Multiple preferences belong in RELATION; registry aliases never promote them to
functional STATE assertions. TRAIT's key carries the existing assertion-dependent cardinality
policy so an explicit statement can still override an inference in that slot.

The registry revision is provenance, not part of the hash: adding aliases must not rename an
existing canonical slot. Reclassifying stored literal predicates changes the stored
representation; deploying a larger alias list or rebuilding Zvec is insufficient. Occurrence
record IDs remain distinct from the scoped lineage key.

Alternatives considered: prompt-only naming cannot enforce stability across backends; fuzzy
matching can silently merge distinct slots. Neither is appropriate for authoritative correction.

## Evaluation contract

Resolver revision: `mindbridge-predicates-v1`; no model revision changes. Deterministic fixtures
cover alias groups and negative controls through the SDK. Pass criteria are zero extra lineages
within the declared alias groups, zero merges across negative controls, and full correction
propagation within each supported functional group. These are fixture metrics, not estimates of
production frequency or coverage of arbitrary model paraphrases. No real-service quality check
is implied by deterministic transport-free tests.

## Regression evidence

The public-SDK correction fixtures failed before the change for English, Chinese, and fullwidth
aliases, then passed with one lineage and the expected supersession. All three declared alias
groups (nine registered spellings) converge to one scoped key per group: fixture fragmentation
is zero extra lineages. All four SDK negative controls remain separate, and all three correction
cases propagate: zero false merges and 3/3 correction recall on those fixtures. These rates do not
measure production incidence, model extraction quality, or unknown paraphrase coverage.

Additional regressions cover compound values, bound identities and spatial frames, accumulating
relations, asserted traits overriding inferences, stable digest snapshots, schema upgrade
idempotence, custom keys, complete history preservation, rollback after an injected migration
failure, incomplete-schema rejection, and index rebuild with document embedding calls forbidden.
Three deletion cases verify that removing a post-upgrade correction restores legacy conflicts.
An independent review found that last replay failure; its test was observed failing before the
previous-key boundary fix. No model prompts or clients changed and no real-service check ran.

## Validation scope

The isolated predicate patch on baseline `b6b5ad28` passed the complete credential-free test
suite, lockfile verification, Ruff, and strict mypy. Whole-workspace checks overlapped concurrent
edits and did not establish a clean immutable snapshot. The migration fixtures above describe
the historical implementation; the current runtime accepts only schema 20. No model prompts or
clients changed, and no real-service quality check ran.
