# Eval concurrency and resource reuse — 2026-10-27

The eval runner now lends its model-operation budget to the work that is actually running.
Ingestion, settlement, deliberation, and answer attempts share 28 slots; idle unit workers and
retry backoff reserve none. Answer concurrency and the SDK's per-directory limits still apply.
Caller latency and TTFT start before admission, while response latency starts after admission.

This change preserves model recipes, prompts, retrieval candidates, scoring, evidence budgets,
causal cutoffs, directory ownership, and the SQLite-to-outbox-to-Zvec durability order. It
changes scheduling and resource reuse, not the meaning of an answer or its evidence.

## Resource ownership

The initial pass capped each SQLite pool at four idle connections, while keeping its
32-connection active limit and the 128-connection process limit. Sixteen idle stores therefore
retained at most 64 handles. The third pass below redistributes that same idle budget by demand.
Previously, twelve readers visiting each open store could fill the process budget with
idle handles and reject readers in the eleventh store. Idle retention is bounded independently
of active admission; no connection is shared by simultaneous borrowers.

Index and speech child jobs acquire their directory's quota before submission to the shared
executor. Previously, one directory could occupy every shared thread with jobs waiting for its
own semaphore, preventing a second directory from using its free quota. A later submission
failure now also waits for already-submitted routes before allowing asset cleanup.

Each SDK attempt holds model admission until its synchronous work and cleanup finish, including
when its asyncio waiter is cancelled repeatedly. Backoff happens outside that admission.
Cancellation stops the next retry, and the runner drains every unit worker before returning or
closing the shared backend pool. This does not promise an immediate cancellation of a blocking
provider call; its configured timeout still applies.

Candidate-count diagnostics now index each source's earliest available end time once per unit.
Queries use binary search instead of rebuilding the source set for every sample and arm. Multiple
fragments from the same source still count once, and an item without an end time remains available
only to an uncut question.

## Reproducible comparison

The baseline is the immutable `3d801fbf` source exported into a separate directory. Both arms run
in fresh processes against the same locked dependencies, use separate temporary physical stores,
and call `run_loaded_task`, the engine used by the eval CLI. Product work goes through public
`Memory` and `AsyncMemory` methods. The baseline is selected with `PYTHONPATH`; the candidate uses
the working tree. No existing memory directory or response cache is reused.

The fixture is `concurrency-audit-fixture-v1:128`: deterministic normalized vectors from a SHAKE256
digest of the query's unique probe token. Each of sixteen units has 72 observations and sixteen
questions at two causal cutoffs, which admit 24 and 48 observations. The remaining observations
are future evidence. The answer backend reads the expected choice from the first hit and asserts
that every hit belongs to the same directory and is available at the question's cutoff.

The service-wait case adds a deterministic 500 ms answer delay and requests 128 concurrent
answers. This measures scheduling when model service time matters; it is not a real LLM quality
evaluation. A short-wait case retains the audit's 20 ms delay. Three paired measurements alternate
baseline/candidate order. Comparison requires identical digests of every question's answer, score,
ranked evidence IDs, and candidate count, with no ingestion, answer, or replay errors.

Raw runners and output are local review artifacts under
`.benchmarks/performance-optimization-20261027/`. The paired runner records its actual imported
source path, dependency versions, answer digest, peak answer concurrency, wall times, and p95
caller latency. No provider credentials or private memory content are used.

## Measurements

Environment: Linux x86-64, Xeon Gold 6230R with 104 logical CPUs, CPython 3.13.12, NumPy 2.4.6,
Zvec 0.8.0, Pydantic 2.13.4, and OpenAI SDK 3.6.0. No real provider was called. The host is shared
with other desktop work; the full test suite finished before these paired measurements began.
The table reports medians of three measurements per arm, with alternating order.

| Workload | Formal seconds, before → after | Total seconds, before → after | Answer p95 ms, before → after |
| --- | --- | --- | --- |
| 16 units, 128 requests, 500 ms answer wait | 20.397 → 17.843 | 27.632 → 26.113 | 5832 → 3581 |
| 16 units, 128 requests, 20 ms answer wait | 15.980 → 17.871 | 22.716 → 25.305 | 3153 → 2714 |
| 4 units, 8 requests, 20 ms answer wait | 4.122 → 4.216 | 5.872 → 6.097 | 664 → 747 |

Formal time includes store opening, ingestion, answering, and closing. Total adds the standalone
search replay, but excludes input acquisition, real judging, and report aggregation. In the
500 ms case, formal time fell 12.5%, total time fell 5.5%, and answer p95 fell 38.6%. Peak answering
concurrency rose from 12 to 28. All 3,456 paired answers were correct and their per-case answer,
score, evidence-ID, and candidate-count digests matched exactly across both versions.

The short-wait cases expose the tradeoff: the larger available window and fairer worker admission
do not make CPU work cheaper. Total time increased 11.4% at 16 units/128 requests and 3.8% at
4 units/8 requests. Search replay also uses the full requested window up to 28 instead of keeping
the old ingestion reservation. These changes fix capacity loss and cross-directory blocking;
they are not a universal throughput improvement. Keep answer concurrency at the workload's
measured saturation point rather than increasing it solely because more slots are available.

The forced INT8 snapshot workload now completes twelve concurrent searches in each of sixteen
open stores with no rejection, retaining 64 idle connections in total. The original implementation
rejected nine searches in the eleventh store and retained 128 idle connections. In the forced
child-worker schedule, an independent directory now completes before the busy directory is
released; previously it could not complete within the 300 ms observation window.

For the already-built source index, 1,000 candidate-count lookups over 10,000 observations fell
from 1.018 seconds to 0.00024 seconds in a separate probe, with identical counts. This excludes
one-time index construction and is not an end-to-end eval speedup claim.

## Validation boundaries

Regression tests force connection pressure across sixteen stores, child-pool contention across
two independent directories, mixed ingestion and answering at the 28-operation limit, partial
submission failure, repeated cancellation, cancellation during backoff, and closure of every
unit before runner exit. Existing native index, stale-ID hydration, missing-index recovery,
outbox failure, scope, and snapshot tests remain required.

The complete warning-as-error suite passed: 3,078 tests, with twelve opt-in real-model integration
cases skipped. Lock consistency, Ruff formatting/lint, and strict mypy passed. Markdown was
checked with markdownlint-cli2 0.23.0. Lychee 0.23.0 checked 927 links: 918 succeeded and eight
redirected; the existing W3C SKOS mapping URL timed out in this environment, and a separate curl
probe failed its TLS handshake. No exclusion or unrelated link edit was added. The pinned tools
ran natively because Docker was unavailable.

The audit's complete fixture matrix was repeated after the paired runs: 4,032 answers across
1/4/16 directories and requested concurrency 1/8/32/128 matched their serial answer and evidence-ID
baselines, with no errors or causal-boundary violations. Another 1,120 searches over 200- and
2,000-record stores at concurrency 1/4/8/16/28 matched serial IDs, order, and floating-point scores
exactly. Temporary stores were removed after each case. Source digests recorded before testing
and measurement matched the final implementation.

These fixtures test deterministic concurrent behavior. They do not establish semantic accuracy
for a real embedding model, generator, or judge, and they do not establish a universal optimal
concurrency. Local model inference safety locks, multimodal identity transactions, Zvec maintenance
locking, and the separate post-answer search replay remain in place.

## Follow-up: candidate hydration and statistical aggregation

A second optimization pass compares against a source snapshot taken after the changes above,
including dynamic admission and the four-idle-connection cap. The baseline source was copied
before editing, selected with `PYTHONPATH`, and recorded with the follow-up artifacts under
`.benchmarks/performance-followup-20261027/`. It is not the original `3d801fbf` baseline.

Worker-stack sampling of the 16-unit, 128-request short-wait fixture found candidate reads
waiting to open SQLite connections behind the shared file-descriptor lock, alongside waits for
the directory write lock and index maintenance. This sampling includes its own overhead and is
used to locate waits, not to quantify a speedup.

Candidate projection now fetches the same SQLite rows and returns the exclusive connection
before parsing dates or constructing metadata objects. It preserves requested order, repeated
IDs, and stale-ID filtering. A regression test blocks decoding and verifies that another reader
can reuse the warm connection. The four-idle cap, active admission, file-descriptor protection,
authoritative hydration, and index maintenance locks are unchanged.

Bootstrap resampling now precomputes cluster totals for integer-valued scores only when every
possible draw remains within the exact floating-point integer range. Noninteger scores and
large integer sums retain their original score summation order using an iterator instead of a
flattened tuple. Draws still use the same seeded random choices, sample weighting, and configured
count. Paired comparison reuses those draws for its confidence interval and probability of
improvement instead of running the same bootstrap twice. No result cache is retained across runs.

The isolated statistics fixture has 8,381 scores in 64 unequal clusters and 2,000 bootstrap draws.
Three alternating before/after measurements produced these medians:

| Calculation | Before seconds | After seconds | Speedup |
| --- | --- | --- | --- |
| Integer-valued score summary | 0.562 | 0.050 | 11.2× |
| Fractional score summary | 0.568 | 0.168 | 3.4× |
| Paired integer-valued comparison | 1.148 | 0.060 | 19.0× |

Every complete statistical output matched exactly across versions. Separate regression cases
compare every sorted draw with the previous algorithm over three seeds, unequal cluster sizes,
negative scores, fractional scores, and cancellation-prone values beyond the exact integer range.
These are calculation timings, not whole-command speedups; they exclude ingestion, model calls,
judging, and diagnostic search replay.

The public-SDK fixture was also repeated in three alternating pairs per workload. All 3,456
answers were correct, and per-case digests of answers, scores, evidence IDs, and candidate counts
matched the follow-up baseline. Every diagnostic replay succeeded. Median wall times were:

| Workload | Formal seconds, before → after | Total seconds, before → after | Answer p95 ms, before → after |
| --- | --- | --- | --- |
| 16 units, 128 requests, 20 ms answer wait | 18.877 → 18.692 | 27.144 → 26.906 | 2815 → 2770 |
| 16 units, 128 requests, 500 ms answer wait | 17.594 → 17.372 | 25.552 → 25.365 | 3163 → 3276 |
| 4 units, 8 requests, 20 ms answer wait | 4.349 → 4.461 | 6.099 → 6.224 | 744 → 745 |

These small, mixed changes do not establish a consistent whole-run throughput gain. The earlier
CPU-heavy regression remains open. Connection-open medians were 267 → 271, 259 → 244, and
57 → 56 respectively; shortening each borrow does not guarantee fewer opens under every schedule.
The connection change is retained to return an otherwise usable exclusive resource before pure
decoding work, as verified under controlled reader contention. The substantial measured gain in
this pass is statistical aggregation, which the public-SDK timing fixture does not execute.

The final warning-as-error suite passed 3,095 tests in 727.96 seconds, with the same twelve
opt-in real-model integration cases skipped. Seventeen new cases cover draw equivalence, paired
draw reuse, and exclusive connection reuse during blocked decoding. The paired-reuse and
connection-reuse tests both fail against the follow-up baseline. The native pressure fixture
then completed 1,120 searches over 200- and 2,000-record stores at concurrency 1/4/8/16/28:
every ID, rank, and floating-point score matched its serial reference, with no errors.

Lock consistency, Ruff formatting/lint, strict mypy, and whitespace checks passed. Markdown
passed the pinned linter; the link checker again reported only the existing W3C SKOS timeout
and eight redirects. Implementation digests remained unchanged from the validation snapshot.
Real provider, GPU, and semantic model-quality checks were not run in this pass.

## Third pass: reuse the idle SQLite budget across bursts

The next baseline is the complete source snapshot after the second pass. Only
`infrastructure/local/store/_connections.py` differs in the product implementation. Reproducible
runners, the baseline source archive, source digests, and raw measurements are local artifacts
under `.benchmarks/performance-retrieval-20261027/`.

Idle connections now share a 64-handle process budget while remaining in their own directory's
pool. Every checkout is exclusive. A directory's first four idle handles are protected from
eviction to fit the idle budget; unused budget can retain more handles for a hot directory.
Old surplus handles are closed when that budget fills. An active request can reclaim any idle
handle before reporting process overload. The limits remain 32 handles per directory and 128
per process, including opening handles and eviction cleanup. Schema initialization, failed
borrows, and secure-deletion connections still bypass retention.

One short metadata lock protects idle lists and eviction order. SQLite open, close, and query
work run outside it. Closing an evicted handle keeps its original directory owned until cleanup
finishes. Closing handles also count against that directory's admission limit. Returns that need
no physical cleanup release admission within the metadata lock. Forked processes create fresh
idle coordination rather than touching the inherited condition. The coordinator retains weak
owner references, and a handle is never lent to another directory.

The native comparison builds one 2,000-record store through the public SDK, optimizes and closes
it, then copies that closed image to a fresh temporary physical directory for each measurement.
Before and after therefore read the same persisted graph, IDs, vectors, and SQLite records.
Fresh processes use the same locked dependencies and `concurrency-audit-fixture-v1:128` embedder.
Each process runs 24 serial reference queries, followed by 112 searches at each concurrency.
Three pairs alternate before/after order. Every measured result must equal the serial reference
in ID, order, and floating-point score; the reference digests must also match across versions.

Final-source medians over those three pairs were:

| Concurrent searches | QPS, before → after | Search p95 ms, before → after | Connection opens during 112 searches, before → after |
| --- | --- | --- | --- |
| 1 | 30.820 → 29.944 | 34.571 → 35.174 | 0 → 0 |
| 8 | 39.145 → 38.526 | 246.002 → 241.084 | 51 → 7 |
| 28 | 37.049 → 41.261 | 823.161 → 771.223 | 86 → 20 |

At 28 concurrent searches, QPS rose 11.4% and p95 fell 6.3%. Warm retention grew to 28 handles
within the unchanged 64-idle process budget. At eight concurrent searches, opens fell 86.3%, but
QPS fell 1.6%; serial QPS fell 2.8%. The host is shared, and three pairs do not establish universal
throughput improvement. These 2,016 measured searches all matched exactly. This optimization
reduces connection churn without changing query vectors, candidates, ranking, model inputs,
durability settings, or measurement admission.

The eval fixture was repeated in three alternating pairs per workload against that same final
source. All 3,456 answers were correct and their per-case answer, score, evidence-ID, and
candidate-count digests matched. Diagnostic replay had no errors. Median times were:

| Workload | Formal seconds, before → after | Total seconds, before → after | Answer p95 ms, before → after |
| --- | --- | --- | --- |
| 16 units, 128 requests, 20 ms answer wait | 18.459 → 17.978 | 25.803 → 25.440 | 2769 → 2690 |
| 16 units, 128 requests, 500 ms answer wait | 16.969 → 16.912 | 24.436 → 24.254 | 3496 → 3119 |
| 4 units, 8 requests, 20 ms answer wait | 4.295 → 4.250 | 6.015 → 5.907 | 755 → 752 |

Total time fell 1.4%, 0.7%, and 1.8% respectively. These are modest fixture-specific changes;
the earlier CPU-heavy regression relative to the original first-pass baseline remains open.
Connection opens fell from 267 → 265, 254 → 242, and 55 → 51. The larger available idle pool
helps a busy directory, while many simultaneously busy directories still share the same total
budget. The native single-directory gain must not be described as an equal speedup of all eval
stages.

The complete warning-as-error suite passed 3,102 tests in 728.05 seconds, with twelve opt-in
real-model integration cases skipped. Seven new cases cover repeated burst reuse, surplus
eviction, directory isolation, reclaim before overload, refusal when every handle is active,
protection of another directory's idle reserve, blocked eviction cleanup, and fresh process
coordination. The blocked-close case also verifies that closing handles still count toward
directory admission and that another directory's warm reader progresses during cleanup.
Repeated burst reuse and reclamation both fail against the four-idle baseline; a separate
forced-close regression fails before the final admission correction.

After the full suite, another 1,120 searches over 200- and 2,000-record stores at concurrency
1/4/8/16/28 matched serial IDs, order, and scores exactly, with no errors. Twelve concurrent
INT8 snapshot searches then completed in each of sixteen open directories. All sixteen pressure
steps filled their readers, reported no errors, and retained at most 64 idle handles in total.
After every directory had been exercised, each retained four handles. The independent-directory
worker check also completed before the blocked directory was released.

Product source digests matched the frozen snapshots after measurements and validation. Lock
consistency, Ruff, strict mypy, whitespace, and pinned Markdown lint checks passed. The link
checker again found only the existing W3C SKOS timeout and eight redirects. Real provider, GPU,
and semantic model-quality evaluations were not run; the product change affects resource reuse.
