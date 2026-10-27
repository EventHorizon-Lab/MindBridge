# Bounded temporal lexical retrieval

Evaluated 2026-10-27 with CPython 3.12.14 and Zvec 0.8.0 on Linux x86-64.
This is deterministic retrieval evidence, not a real-model or production latency evaluation.

## Finding and decision

P01 contains two different claims. The 100/200 parent depth is a per-route window; multiple
query keys, memory types, and preferred/global routes intentionally contribute more candidates.
An aggregate cap at that same depth would discard their diversity. The
[October 26 accuracy-first evaluation](2026-10-26-accuracy-first-retrieval.md) also found
quality benefits from retaining a wider quantized pool. Keep those route budgets.

The temporal lexical workaround was corpus-dependent. It loaded every aggregate embedding ID
in a SQLite time window plus a searchable row count, then doubled global lexical top-k up to
that count. Its result-count-dependent threshold heuristic could stop short of useful evidence,
and it repeated work outside the ordinary scope-expansion ceiling. A corpus of 385 memories
already reproduced top-k 385 in all four quantization modes.

Replace that workaround with bounded native lexical retrieval inside the preferred time window,
alongside the existing global fallback. Delete the whole-window SQLite reader and expansion
loop. Keep SQLite hydration, hard filters, the scope-expansion ceiling, and FP32 rescoring.
No public signature, setting, dependency, schema, or model/prompt revision changes.

Alternatives rejected: a cap on global expansion loses rare-window discovery; streaming the
SQLite whitelist reduces its memory footprint but retains corpus-sized top-k work.

## Native compatibility evidence

The current dependency is Zvec 0.8.0. Upstream documents
[filtered full-text queries](https://zvec.org/en/docs/db/data-operations/query/) and lists
read-concurrency, filtering, and FTS-persistence fixes in its
[0.8 release](https://github.com/alibaba/zvec/releases/tag/v0.8.0).
Those notes alone do not prove the historical range-FTS instability is fixed.

A native probe with 2,000 documents and 20 time-window matches returned the exact expected
set in 120/120 searches: 20 sequential and 20 concurrent dense/FTS pairs in each of fresh,
optimized, and reopened collections. Durable tests additionally cover English, Chinese, mixed
scripts, half-open point/interval boundaries, absent times, type/place/identity filters,
zero-match filters, and repeated dense/FTS concurrency across the same three lifecycle phases.
These characterize the supported adapter usage, not every native query or platform.

## Paired retrieval evaluation

Fixture revision: `temporal-lexical-fixture-v1`; deterministic 64-dimensional orthogonal query
and document vectors from the test backend of that name. No remote model was configured or
called, and there is no returned provider model identity. Each baseline/final case allocated a
distinct temporary physical directory and used public SDK ingestion, index optimization,
reopening, and traced search. The baseline restored only the former temporal workaround over
the same surrounding working-copy code. The shared checkout contained unrelated concurrent
changes; this is not a comparison against the entire previous Git commit.

The corpus contains numbered `common witness` distractors dated 2026-10-27 and one longer
`common witness target extra` record dated 2024-06-01. The query is `common witness in 2024`,
with result limit 5, minimum relevance 0.4, and fixed reference 2026-10-27. The distractors
outrank the target globally; semantic vectors cannot rescue lexical discovery. Baseline and
final use NONE/INT8 and hybrid/lexical modes. Each case runs 12 searches, excluding the first
from timing. Acceptance requires bounded lexical top-k and target candidate discovery in every
final run. Discovery measures the traced lexical candidate, not necessarily a returned top-five
hit: pure lexical scores can tie at their ceiling.

| Corpus memories | Baseline maximum lexical top-k | Final NONE / INT8 top-k | Baseline target discoveries | Final target discoveries |
| --- | --- | --- | --- | --- |
| 385 | 385 | 100 / 200 | 48/48 | 48/48 |
| 2,049 | 2,049 | 100 / 200 | 0/48 | 48/48 |
| 8,193 | 8,193 | 100 / 200 | 0/48 | 48/48 |

In this fixture, the former global normalized score and admission heuristic lost the target
at larger corpus sizes even after reading the whole global result window. The new route
discovers those candidates without relying on the former admission heuristic. Existing relevance
gates and soft temporal weighting still apply.

Independent review reproduced a regression in the initial design: normalizing a preferred
window independently promoted `Alice coffee` to the same lexical strength as the global exact
match `Alice bought telescope in 2024`, making a one-result search abstain. The retained fix
preserves every known global score. For new single-field BM25 hits, the adapter carries raw
scores and their type-route identity, and the kernel applies that global route's normalizer.
It never substitutes a maximum from a different memory-type route.

For mixed-script RRF and zero-score fallbacks, the filtered rank cannot identify an unseen
candidate's global rank. Such candidates retain their preferred ordering below the corresponding
global tail. This is a bounded approximation: temporal weighting and the ambiguity margin can
still change final order or abstention for near ties. No exact global-rank equivalence is claimed.
Regressions cover the reported exact-versus-weak match, saturated windows, mixed scripts,
per-type normalization, absent native scores, and result-limit-independent discovery.

A separate default-threshold SDK smoke check used 384 distractors containing the entire query
`common witness in 2024` plus a numbered suffix, and one longer target with that same phrase
inside the requested year. Both NONE and INT8 returned the target in the first five results
with the unchanged 0.55 default. Unlike the main fixture, this checks complete lexical coverage.

At 8,193 memories, one paired NONE build measured hybrid p50 153.08 ms to 16.05 ms and lexical
p50 135.16 ms to 7.62 ms. INT8 hybrid measured 163.33 ms to 29.78 ms. These warm traced-query
observations came from a busy shared host, one build per arm, and eleven timed searches each;
they establish removal of corpus-sized result handling, not stable latency ratios or constant
query time. No process-memory measurement was taken.

## Reproduction and limits

The retained fixture and native compatibility checks run with:

```bash
uv run --frozen pytest -W error tests/unit/test_temporal_lexical_retrieval.py
```

The regression includes all four quantization modes, lexical mode without query embedding,
hard-window intersections, type and missing-place filters, global fallback, absent lexical
evidence without a time phrase, and stable result prefixes when widening is unnecessary.
The paired runner and frozen baseline are temporary investigation artifacts; the table records
their aggregate evidence. Larger paired counts can be reproduced with the same numbered corpus
and parameters above. Model-service answer quality, instruction-following over retrieved hostile
content, other platforms, and a production precision/recall distribution were not evaluated.
Existing synthetic temporal boosts can admit weak semantic evidence; this change does not
recalibrate that separate ranking policy.

## Validation snapshot

The final focused SDK/retrieval/read-path run passed 383 tests. A separate final native-index,
storage, and temporal run passed 208 tests; these runs overlap and must not be added together.
Strict mypy passed for the seven changed retrieval/adapter modules and new test files.
Lock consistency and `git diff --check` passed. The three documentation files passed pinned
Markdown lint. Full-checkout Ruff and mypy passed during intermediate snapshots, but subsequent
unrelated concurrent edits introduced formatting, lint, and benchmark-test typing failures;
the final shared checkout cannot be reported as passing those complete gates.

The complete final `uv run --frozen pytest -W error --tb=short` run reported 2,861 passed,
12 skipped, and five failures in other concurrently edited areas:

- `tests/unit/benchmarks/test_eval.py::test_a_fully_cached_unit_completes_its_ingest_share_without_writing`
- `tests/unit/benchmarks/test_eval.py::test_runner_reports_cached_progress_before_pending_answer_finishes`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_returned_metadata_cannot_replace_observed_reader_source_provenance`
- `tests/unit/benchmarks/test_suite_cli.py::test_global_help_lists_only_supported_runners_without_importing_them`
- `tests/unit/test_runtime_capacity.py::test_cancellation_during_terminal_cleanup_remains_cancellation`

No temporal retrieval test failed in that run. The twelve skipped cases require an explicit
formation-time model-service evaluation opt-in and credentials. The pinned whole-checkout link
check reported 880 successful links, eight redirects, one connection failure for the existing
Uvicorn server-behavior URL, and one timeout for the separate predicate report's W3C SKOS URL.
The final whole-checkout Markdown run also encountered six errors in another task's ignored
`.superpowers/boundary-cleanup/` plan/progress files; no exclusions were added.
