# Memory backend validation

Date: 2026-09-08

This review compared the working tree with the frozen pre-change source in
`.benchmarks/research/2026-09-08-e2e-baseline-current/extracted/`. It independently checked the
deterministic evidence-closure implementation and its public `Memory.compile` path. This is a
structural correctness review; it is not evidence of benchmark or semantic-answer improvement.

## Reviewed invariants

- Every admitted derived hit recursively includes its declared evidence memories.
- Candidate-known conflicts in functional STATE and user-stated TRAIT lineages, and their supports,
  are admitted together or withheld. Multi-valued RELATION and model-inferred TRAIT assertions
  accumulate without being treated as automatic contradictions.
- Missing, cyclic, depth-bounded, deadline-unresolved, out-of-scope, type-excluded and
  consent-restrained dependencies cannot leave an affirmative anchor in the bundle.
- Item and rendered-character costs use the union of memory IDs, so a shared source is charged once.
- Empty feasible-closure results cannot fall back to the legacy flat selector.
- Hydrated dependencies have retrieval score zero and remain subject to the requested retrieval
  bounds.
- Untyped singleton compilation retains the frozen behavior covered by the compatibility tests.

The SDK regressions exercise evidence hydration outside the initially ranked hits, strict memory
type bounds, shared-source diamonds, cycles, deadline behavior, tight budgets, invalid conflict
peers and empty-closure fallback. The implementation treats every declared `evidence_id` as
conjunctively required. It does not claim semantic entailment, a minimum sufficient proof, an
independent confidence increase or corpus-wide contradiction discovery.

The final correction aligns compilation conflicts with functional write semantics. STATE lineages
and user-stated TRAIT lineages can carry one standing value, so candidate-known disagreements remain
atomic with their supports. RELATION and model-inferred TRAIT lineages accumulate: two children or
two inferred interests are not automatically contradictory. A public SDK regression verifies that
two supported `has_child` relations coexist and that a two-item budget can admit one relationship
with its own source instead of withholding both as a false conflict.

This functional user-stated-TRAIT rule predates the evidence-closure change: the frozen snapshot's
write path already retired overlapping same-lineage values. It is an expressivity limitation for
set-valued predicates, not a compiler regression. Model simultaneous languages, likes, or similar
facts as multivalued `RELATION` assertions; do not infer predicate cardinality from their text.

## Temporal identity review

`Memory.compile` now passes `RetrievalScope.valid_at` and `known_at` into named and provisional actor
lookups. A later naming assertion therefore does not decorate an earlier historical bundle. Evidence
projection changes retire the current `memory_versions` row and create a new row; they do not mutate
the historical row's visibility or confidence. The as-of predicate selects the version whose
`recorded_at <= known_at < retired_at`, and valid-time bounds match the normal scoped read.

The SDK regression covers an old observation, a later name, an unnamed historical actor, the named
current actor and current withdrawal. Withdrawal suppresses actor disclosure for the historical
compile under the existing present-consent policy; it does not suppress the raw observation. Actor
decoration may still reference a naming assertion outside the context hit budget, so this is not a
complete identity-proof guarantee.

## Quality gates

| Check | Result |
| --- | --- |
| `uv lock --check --default-index https://pypi.org/simple` | Pass |
| `uv run --frozen ruff format --check .` | Pass; 203 files already formatted |
| `uv run --frozen ruff check .` | Pass |
| `uv run --frozen mypy` | Pass; 155 source files |
| `uv run --frozen pytest -W error` | Pass; 1,800 tests in 189.35 seconds |
| `git diff --check` | Pass |
| Pinned repository link check from `CONTRIBUTING.md` | Pass; 500 links, 0 errors, 6 redirects |
| Pinned repository Markdown check from `CONTRIBUTING.md` | Pass; 47 files, 0 errors |

The original Markdown command searched tool-managed nested worktrees and their virtual environments,
producing 4,354 third-party errors. Excluding nested `.venv` directories reduced that to 202 errors
inside `.claude/worktrees`; excluding that tool-managed checkout root made the lint boundary match
the main repository. The pinned image and Markdown rules were unchanged.

## Source signoff

No actionable source-correctness defect remains in the reviewed evidence-closure and historical
actor-resolution changes. The compiler source is suitable for a frozen empirical comparison. Product
claims still depend on the separate evaluation artifact: lower wrong-answer rate is insufficient if
it comes only from lower answerable coverage, and greedy rank-order closure has no optimal-coverage
guarantee.

## Initial raw LoCoMo diagnostic

The first 138-question `locomo-refined` slice in
`.benchmarks/results/e2e-raw-snapshot-smoke-20260908/samples.partial.jsonl` compares two generation
interfaces on the frozen baseline source. It does not compare the baseline compiler with the new
compiler. The `compile` arm's mean deterministic token F1 is 0.5682 versus 0.3366 for `mindbridge`
ask, with 102 paired wins, 12 ties and 24 losses. On the 137 questions carrying gold source IDs,
compile retrieves every labelled source for 119 and at least one for 129; ask does so for 116 and
127. Their macro labelled-source recalls are 0.9057 and 0.8875. The small retrieval difference
cannot by itself explain the 0.2316 F1 difference.

Answer format is a major confound. Compile answers average 8.2 whitespace-delimited words with a
median of 3, while ask answers average 20.9 with a median of 16. Token F1 penalizes extra explanation,
so this slice primarily shows that the compile generator interface is terse. It is not an estimate
of the evidence-closure change. A valid algorithmic comparison must run baseline compile against
candidate compile with identical provider-ready generation requests.

Thirteen compile answers preserve relative phrases such as “last year,” “next month,” or
“yesterday”; nine illustrative temporal questions (`q0001`, `q0006`, `q0037`, `q0048`, `q0056`,
`q0057`, `q0060`, `q0064`, and `q0129`) include all labelled gold evidence but receive zero token F1
because the expected answer is an absolute date. Other all-gold generation failures include `q0051`
(generic family-hike activity instead of the labelled details). Conversely, ask substantially beats
compile on `q0048`, `q0057`, `q0060`, and `q0135`, largely because it resolves or restates the date
or object rather than returning the terse source phrase.

Clear labelled-source misses include `q0041` (both people's shared painting subject, zero of two),
`q0059` (child count, zero of two and an incorrect generated count), and `q0069` (later summer plan,
zero of one and a stale-plan answer). `q0055`, `q0114`, and `q0130` retrieve only part of their
labelled evidence and fail to supply the requested book, duration, or art type. Some zero-labelled
overlap answers are nevertheless correct (`q0090`, `q0118`) or plausible (`q0073`), so source IDs
are useful diagnostics rather than a complete semantic relevance oracle. Identity question `q0004`
also answers correctly from alternate evidence despite missing its one labelled source; this slice
does not establish an identity-attribution regression.

No LoCoMo model judge ran for these rows: `judge_model` is null, `judge_cached` is false, and metrics
contain only token F1 and BLEU-1. The `locomo_refined_judge_887091190789` scorer-protocol string names
the intended protocol; it is not proof of judge execution. These numbers must remain labelled as
deterministic proxy scores until the separately owned pinned judge pass completes.
