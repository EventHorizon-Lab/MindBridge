# Local runtime performance — 2026-10-27

Two local optimizations reduce vector materialization and unnecessary SQLite hydration. Model
requests, retrieval routes, candidate depths, score arithmetic, evidence selection, SQLite
durability, outbox ordering, and the on-disk schema are unchanged.

## Changes and invariants

Quantized retrieval streams the original little-endian FP32 embedding bytes from one SQLite WAL
snapshot. Multi-key scoring promotes buffer views directly into the same float64 matrix and runs
the existing normalization and dot products. Scalar scoring decodes one vector at a time instead
of materializing every component of a full batch. Batch size, parent reduction, preferred-part
tie breaking, and the release of the scoring snapshot remain unchanged. Unquantized vector reads
still return decoded tuples.

Record hydration reads media and semantic rows only for parent IDs surviving its authoritative
SQL predicates. Missing parents and parents excluded by place or identity cannot contribute a
result. This avoids reading their related rows while preserving the input order, duplicate IDs,
record visibility rules, media descriptors, and the same read transaction. Returned parent IDs
are decoded once and reused when constructing the hydrated records.

No model backend, prompt, model input, public setting, dependency, or public signature changed.

## Frozen SDK comparison

Environment: Linux x86-64, Intel Xeon Gold 6230R, CPython 3.12.14, NumPy 2.3.5, SQLite 3.53.1,
and Zvec 0.8.0. The host was shared with other development work.

The baseline froze the working copy before this patch, including existing uncommitted work. Its
HEAD was `b6b5ad28`; HEAD alone does not reconstruct that working copy. The candidate used that
same frozen copy with only the scoring, embedding-part stream, and retrieval type changes.
The [manifest](artifacts/local-runtime-20261027/manifest.json) records source hashes and fixtures.

Fixture `local-perf-fixture-v1` inserted 2,000 text records through `Memory.add_many`, with one
1,024-dimensional vector per record. Texts were
`Observation {i}: shared kitchen Alice archive garden. Unique token {i}.` A NumPy generator seeded
from the first eight little-endian SHA-256 bytes of each input produced normalized Gaussian
vectors. The backend cached these synthetic vectors; this measurement excludes actual inference.
No formation, vision, speech, or generation backend was configured.

The real INT8 Zvec index was optimized and closed once, then copied into a distinct physical
directory for every arm. All arms therefore searched the same native graph. Hybrid retrieval
used `limit=10`, zero relevance floor, and zero ambiguity margin. The short query was
`shared kitchen Alice`; the long query appended 180 copies of `"archive garden Tuesday office. "`
after `shared kitchen Alice` and a newline, producing four query keys with the current recipe.

Three pairs alternated arm order. Each arm performed five warmups and 31 timed searches per query,
then one traced search. The table gives medians of the three arm results, in milliseconds.

| Query | Baseline p50 | Updated p50 | Baseline p95 | Updated p95 |
| --- | --- | --- | --- | --- |
| Short | 43.256 | 39.949 | 50.008 | 48.560 |
| Long | 121.765 | 69.513 | 139.789 | 85.595 |

Every timed search returned the same IDs and exact scores across all arms. Complete traced-result
digests also matched across arms. The supported result is lower warm latency for these synthetic
INT8 workloads, particularly multi-key scoring. It does not establish production latency,
throughput, unquantized-search gains, or model quality. See the
[individual arm results](artifacts/local-runtime-20261027/sdk-results.json).

## Numerical and SQLite microbenchmarks

The numerical comparison included FP32 decoding in the baseline and buffer consumption in the
candidate. Seed 20261027 generated 128×1024 and 512×2048 document matrices, with one, four, or
eight query vectors. Fifteen pairs alternated execution order. Every result dictionary matched
exactly. A separate `tracemalloc` pass measured temporary allocation, excluding the existing
input buffers; these are allocation peaks, not process RSS.

For the 512×2048, four-query case, median time changed from 69.885 to 9.230 ms and traced peak
allocation from 19.24 to 6.70 MB. This includes a large reduction in Python float allocation;
it is a numerical-kernel result rather than an SDK speedup. The
[raw kernel results](artifacts/local-runtime-20261027/kernels.json) retain all six shapes.

The SQLite-only fixture `scoped-hydration-v1` held 1,000 image records, each with one media
descriptor and no embedding. It read metadata, not media files. Fifteen alternating pairs each
measured three reads per arm against the same store; returned records matched exactly.

| Selected parents | Baseline median ms | Updated median ms |
| --- | --- | --- |
| All 1,000 | 23.896 | 23.757 |
| Ten by place | 13.717 | 0.994 |
| None by place | 12.881 | 0.516 |

Unfiltered reads were effectively unchanged. Selective reads avoided work proportional to the
excluded parents. Native filter pushdown already prevents many such candidates from entering
normal retrieval, so these storage figures must not be presented as a general search speedup.
The [SQLite samples](artifacts/local-runtime-20261027/sql-results.json) retain every measurement.

Changing only matrix normalization to operate in place produced no clear improvement in the
small repeated probe and was not retained. No lower-precision arithmetic, smaller candidate pool,
weaker validation, or relaxed SQLite synchronization was adopted.

## Verification and limits

The new buffer tests failed on the preceding implementation and passed after the change. They
compare exact scores and winning parts across batches for one, three, four, and eight query
vectors, cover extreme finite FP32 scales, preferred ties, malformed buffers, NaN, infinity,
and zero vectors. The storage stream test also verifies the original FP32 bytes and decoded-vector
read behavior. Scoped-read regressions preserve media and ordering while ensuring rejected or
missing parents do not enter related-row queries. The final new-test run passed all 22 cases.

The broader storage/retrieval run passed 702 tests with warnings treated as errors in 144.63 s.
It covered all local infrastructure tests, `test_local_store.py`, `test_memory_search_index.py`,
`test_hybrid_score_completion.py`, `test_retrieval_concurrency.py`, `test_memory_api.py`, and
`test_data_subject_rights.py`. This includes ownership and directory isolation, authoritative
stale-ID filtering, missing-index recovery without re-embedding, exact-score completion, and
concurrent deletion with media leases.

Lock consistency, whole-workspace Ruff formatting and lint, strict mypy over all 283 source files,
and `git diff --check` passed. An initial full pytest attempt stopped with 13 collection errors
because `benchmarks/eval_judge.py` imported `_transport_errors` after concurrent model-adapter edits
removed it. The frozen baseline still contained the helper. The shared workspace subsequently
resolved that import independently. The complete pytest retry passed 3,054 tests with 12 opt-in
real-service formation evaluations skipped, in 455.51 s. Warnings were treated as errors.

Pinned Markdown lint passed all 369 Markdown files. The new report and research index passed
all 64 link checks. Whole-workspace pinned link validation checked 930 links twice, with zero
invalid-link errors, eight redirects, and one repeated timeout at the pre-existing W3C SKOS link
in `2026-10-27-predicate-stability.md`. No exclusion or unrelated link edit was introduced.

No real model-service quality evaluation was run: model inputs and retrieval policy were not
changed, and the synthetic equality checks do not establish natural-language quality.
