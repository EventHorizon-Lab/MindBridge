# Retrieval critical-section review — 2026-10-27

P02 is a real, workload-dependent source of write-lock blocking. Single-directory ownership and
SQLite's single writer remain appropriate for the embedded product. The problem is the additional
serialization of read-only scoring and ranking, after the recent
[accuracy-first change](2026-10-26-accuracy-first-retrieval.md) widened quantized candidate routes
and added authoritative FP32 completion. This review supports a bounded optimization, without a
claim of universal throughput improvement.

## Required invariants and decision

SQLite decides record existence and scope; Zvec supplies candidates. A deletion completed before
final authoritative hydration must exclude the record. A deletion overlapping an already hydrated
search may complete while that search returns its snapshot, as before. Media referenced by that
snapshot must remain available through the active operation. An operation does not give its caller
a permanent asset lease after returning.

The write lock therefore still covers final record hydration, provisional candidate-media leases,
and establishment of a scoring snapshot. FP32 computation, lexical coverage, relevance gating,
ambiguity checks, ranking, and trace construction then run outside that lock. Candidate media
leases end after ranking; selected media receive operation-long leases before that handoff.
Provisional leases do not validate file existence: a rejected candidate with missing media must
not fail an otherwise valid search. Selected assets retain the existing validation.

The stored embedding iterator now holds one WAL read transaction across SQL batches. Because
`BEGIN` is deferred, retrieval consumes its first row while still holding the write lock, then
prepends that row to the scoring stream. Deletion cannot change later batches. The stream closes
after scoring, before lexical work; failure unwinds both the reader and provisional media leases.
Vectors remain streamed in bounded scoring batches rather than copied into a request-sized array.

Keeping the old lock would remain reasonable for exclusively serial workloads. Copying every
candidate vector under the lock would increase transient memory with dimensions and part count.
Optimistic rehydration and reranking would introduce retries and starvation under mutation.
The existing WAL snapshot and asset reference counts provide the narrower solution. No public
contract, retrieval policy, model, dependency, or on-disk schema changes in this optimization.

## Practice and source checks

The SQLite documentation states that WAL readers retain a stable snapshot while writers commit;
the initial `SELECT`, not merely deferred `BEGIN`, establishes the view. This supports moving
the streamed computation outside the application lock. SQLite cannot protect external CAS files,
so media leases remain a separate requirement. See
[SQLite isolation](https://www.sqlite.org/isolation.html).

WAL readers can delay checkpoint progress, which is why the reader is released immediately after
scoring rather than held during ranking or answer generation. See
[WAL concurrency and checkpointing](https://www.sqlite.org/wal.html).
CPython's GIL also limits CPU-bound thread scaling; shortening an application lock alone does not
establish higher overall QPS. See
[Python threading](https://docs.python.org/3.14/library/threading.html).
These sources were checked on 2026-10-27. They inform the mechanism; local measurements determine
whether the optimization helps MindBridge.

## Measurements

Environment: Linux x86-64, CPython 3.12.14, NumPy 2.3.5, Zvec 0.8.0. The deterministic embedder
was `critical-section-fixture-v1:128`, using finite trigonometric vectors without remote services.
The fixture ingested 200 records through the public SDK, optimized the real INT8 index, and ran
hybrid traced search with `limit=10`, zero relevance floor, and zero ambiguity margin. Queries
were one text part or four text parts. Each evaluation owned its own temporary physical directory.

The paired comparison used the retrieval implementation at `b6b5ad28` and the optimized retrieval
class on the same live native graph and remaining working-tree implementation. Three pairs
alternated execution order. Each arm used three warmups, 21 serial measurements, and 16 searches
with four threads. Instrumentation measured the retrieval plane's cumulative write-lock hold time
per serial search. Every returned ID and score matched exactly between arms and threaded searches.
The host was shared with other development work; the table gives medians of the three arm results.

| Query | Lock ms, before → after | Serial p50 ms, before → after | Four-thread QPS, before → after |
| --- | --- | --- | --- |
| One text part | 8.344 → 3.690 | 18.173 → 17.959 | 47.516 → 49.588 |
| Four text parts | 10.685 → 3.777 | 21.606 → 23.056 | 36.850 → 39.286 |

Paired lock-time reductions were 55.5–59.7% and 59.3–64.7%, respectively. Serial latency and
throughput did not improve consistently in every pair; one multi-part pair lost throughput.
The supported benefit is reduced write-lock blocking, verified independently by forced
interleavings. These synthetic fixtures do not establish model quality, production tail latency,
or universal retrieval throughput. The temporary runner and raw output are local review artifacts.

## Validation

`tests/unit/test_retrieval_concurrency.py` forces deletions and another search during quantized
scoring, unquantized lexical-score completion, lexical coverage, and ranking. It checks unchanged
hits and traces for the in-flight snapshot, exclusion from later searches, media availability
during deletion, and cleanup after success or injected failure. It also checks SQL-batch snapshot
consistency, deletion before hydration, missing rejected media, and release of rejected media
before final hit hydration. A truncate checkpoint detects retained readers after operations end.

The original implementation failed the six initial progress cases by timing out behind the
write lock, and lost a deleted embedding from the second SQL batch. The corresponding optimized
tests passed.

An isolated export of `b6b5ad28` plus this patch passed the relevant SDK, concurrency, retrieval,
index, and asset tests, along with lock consistency, Ruff, strict mypy, Markdown, and link checks.
Whole-workspace checks overlapped concurrent edits and did not establish a clean immutable
snapshot. No real model-service quality evaluation ran; retrieval policy and model inputs
were unchanged.
