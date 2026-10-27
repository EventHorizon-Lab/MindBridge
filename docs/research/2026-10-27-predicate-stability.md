# Predicate stability: investigation and implementation plan

This report records the original investigation and its measurements. The current runtime uses
schema 20, accepts no older schema, and keeps predicate provenance without migration lineage.
The upgrade procedure and migration regressions below describe the evaluated implementation at
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
existing canonical slot. A later registry change that reclassifies stored literal predicates
requires a new explicit SQLite migration and reviewed fixtures. Deploying a larger alias list
or rebuilding Zvec is insufficient. Occurrence record IDs remain distinct from the scoped
lineage key and are never rewritten by this upgrade.

Alternatives considered: prompt-only naming cannot enforce stability across backends; fuzzy
matching can silently merge distinct slots. Neither is appropriate for authoritative correction.

## Implementation plan

1. Add public-SDK regression fixtures for multilingual alias corrections, independent slots,
   accumulating relations, identity and spatial isolation, and raw predicate preservation.
   Observe the correction regression fail before implementing the pure resolver and claim key.
2. Persist registry provenance in schema 19. Transactionally upgrade validated schema-18 stores,
   changing only generated lineage keys whose original digest matches exactly. Preserve versions,
   record IDs, evidence, embeddings, custom lineage IDs, and operation history. Preserve historical
   competing versions as conflicts rather than inventing a retroactive correction order.
   Deletion replay retains the old independent-slot boundaries using the previous keys, so
   withdrawing a new correction restores the original conflict instead of choosing a winner.
3. Test upgrade idempotence, failure atomicity, and correction after reopening. Document the
   compatibility change and deterministic evaluation, then run every repository quality gate.

## Review focus

- Registry entries must never alias naming or consent policy predicates.
- Kind, basis/cardinality, subject binding, and spatial scope constrain convergence.
- Unknown and compound values must not be coerced into registered scalar claims.
- Upgrade must not reinterpret custom lineage keys or rewrite temporal evidence history.
- Missing derived indexes must rebuild from persisted vectors without changing semantic keys.

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

## Validation results

The shared checkout was being edited by other sessions during validation. An isolated copy of
baseline `b6b5ad28` plus only this predicate patch passed the complete `pytest -W error` suite:
**2,628 passed**, including the 34 new predicate regressions. Those 34 tests also passed as a
separate run. A broader run of relevant tests in the shared checkout passed 314 tests.

The isolated patch passed `uv lock --check --default-index https://pypi.org/simple`, Ruff's complete
format and lint gates, and mypy (255 source files). The existing dependency environment was reused
with `uv run --no-sync --frozen`. `git diff --check` passed in the shared checkout. The pinned
Markdown tool passed on all five documentation files touched by this patch. A later whole-tree
Markdown run reported unrelated list spacing and bare-URL errors in the concurrently created
`.superpowers/boundary-cleanup/` notes.

A frozen copy including the other sessions' changes completed with 2,858 passed, 12 opt-in
real-service tests skipped, and these six failures outside the predicate patch:

- `test_response_cache_namespace_changes_with_runner_recipe`
- `test_a_fully_cached_unit_completes_its_ingest_share_without_writing`
- `test_runner_reports_cached_progress_before_pending_answer_finishes`
- `test_failed_answer_keeps_its_trace_and_cached_sample_cannot_acquire_a_fresh_one`
- `test_global_help_lists_only_supported_runners_without_importing_them`
- `test_semantic_recall_capability_changes_the_actual_planner_system_instruction`

The first three are in `tests/unit/benchmarks/test_eval.py`; the next two are in
`test_eval_diagnostics.py` and `test_suite_cli.py` in that directory; the last is in
`tests/unit/test_openai_sdk.py`. These failures and the shared-tree formatting changes were left
to their owning work. Earlier live-tree runs were superseded because source files changed while
tests were running.

Mypy on that combined snapshot also reported two errors: the missing `SEMANTIC_QUERY_CAPABILITY`
export used by `test_openai_sdk.py`, and an incompatible `_plan` generator argument in
`test_memory_api.py`. Neither occurs in the isolated predicate patch.

The pinned full link check was attempted. Its final attempt reported a TLS timeout for the W3C
SKOS reference and a connection failure for the Uvicorn resource-limits documentation. Both source
links were retained; no host exclusion or weakened check was added. The earlier full link run
before those links were present passed with 815 successful links and eight redirects.
