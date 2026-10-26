# Accuracy-first retrieval results

Evaluated 2026-10-26 on Linux x86-64 with CPython 3.14.2, NumPy 2.4.6 and Zvec 0.8.0.
The comparison used the working-copy snapshot preceding this optimization, including the
Zvec upgrade. No credentialed model service was called. The results concern synthetic
retrieval with known FP32 truth, not natural-language embedding or answer quality.
Temporary runners, raw outputs, source snapshots and implementation plans were removed
after evaluation; this page retains the final decisions and aggregate results.

## Shipped behavior

- FP16, INT8 and RaBitQ routes retain 200 candidate parents. SQLite visibility and scope
  filtering precede exact scoring from authoritative FP32 embeddings. The best stored part
  across all query keys determines relevance and ranking.
- Evidence excerpts use that exact winning part. Exact ties retain the existing match.
- NONE retains its 100-parent route and native-score semantics. Search effort remains
  ef=2048; grouped retrieval preserves diversity among parents with many embeddings.
- Cosine scoring streams bounded batches. Larger workloads with at least four query keys
  use NumPy; small workloads retain scalar scoring. Scaled normalization handles finite
  subnormal vectors and norms that would otherwise overflow.
- Ranked trace values are constructed and validated once with their final rank and rejection.

No public setting, dependency or on-disk schema was added. SQLite ownership, outbox ordering
and exclusive index maintenance remain unchanged. Current behavior is documented in the
[architecture](../architecture.md) and [Python SDK reference](../api/python-sdk.md).

## Quality evidence

Fixture revision was `mindbridge-synthetic-retrieval-fp32-v1`: 128 dimensions, 1,671 memories
and 1,696 document keys, using public SDK ingestion, optimization, reopening and traced search.
Seed 20261026 was development; 20261027 and 20261028 were held out. Four cases covered the
0.55 relevance threshold, near ties, multi-key parent diversity and type/time/place filters.
Each evaluation owned a distinct physical store. An independent scalar cosine oracle supplied
the expected ranking. Acceptance required no recall/NDCG regression or added admission errors.

| Mode | Baseline mean Recall@10 | Baseline mean NDCG@10 | Final mean Recall/NDCG |
| --- | --- | --- | --- |
| NONE | 1.000000 | 1.000000 | 1.000000 / 1.000000 |
| FP16 | 0.950000 | 0.964588 | 1.000000 / 1.000000 |
| INT8 | 0.741667 | 0.706581 | 1.000000 / 1.000000 |
| RABITQ | 0.775000 | 0.703706 | 1.000000 / 1.000000 |

All 36 quantized cases passed with no false or missed relevance admissions. Maximum returned
and trace score error was at most 2.22e-16. NONE rankings and returned scores were unchanged.
NDCG used ordinal gold grades 10 through 1 and gain `2**grade - 1`.

A separate native test compared both methods on the same live graph: about 7,800 vectors,
800 parents, three seeds, eight queries and five filters. Across 360 paired cases there was
no recall/NDCG regression; mean parent recall changed from 0.980333 to 0.983583 for FP16,
0.977583 to 0.981000 for INT8, and 0.977500 to 0.986000 for RaBitQ.

Discovery remains approximate. Exact scoring cannot recover a parent absent from every route.
These fixtures do not establish universal recall or real-service model quality.

## Speed and tradeoffs

Three independent NONE build pairs, each with 31 warm traced searches per case, measured
6.0–7.4% lower median paired p50. Scores and rankings were unchanged. Filter latency improved
in two of three pairs; p95 was unstable. This supports warm traced search only.

For INT8, one final paired build measured baseline p50 of 14.65–14.99 ms versus final p50
of 22.17–22.89 ms across the four cases: 7.5–7.9 ms additional latency. Three preceding
independent pairs showed similar ratios. Wider hydration and exact scoring retain this
accuracy-first cost; other modes and ordinary untraced search may differ.

A numerical microbenchmark with eleven alternating measurements per shape found 1.4–2.6
times faster scoring for four/eight query keys at 128×1024 and 512×2048 vector shapes.
Single-key work gained no speed; maximum numerical differences were 1.67e-16.
Kernel speed does not imply the same end-to-end improvement. The shared host and native
graph construction introduce timing variability.

## Validation

Before cleanup, the Python 3.14.2 suite passed all 2,596 tests with warnings treated as errors.
Lock consistency, Ruff formatting/lint, strict mypy and documentation checks passed.
Python 3.12 passed 43 focused scoring, quantized retrieval and evidence-excerpt tests;
Python 3.10 passed a numerical smoke check. Full-suite support on every interpreter and
real model-service quality were not evaluated. Durable regression tests retain coverage of
FP32 admission, candidate recovery, numerical edge cases and exact evidence selection.
