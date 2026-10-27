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

An isolated export of `b6b5ad28` plus only this eight-file patch passed 435 tests: the complete
`test_memory_api.py`, the 13 new concurrency cases, hybrid completion, quantized retrieval,
native search-index behavior, asset storage, and on-demand media. Lock consistency, Ruff format
and lint, strict mypy, pinned Markdown lint, and pinned link validation also passed for this
patch. Link validation checked 816 links with zero errors. In the shared working tree, an
additional focused run passed 99 tests before the final missing-selected-media case was added;
18 final concurrency, ownership, close, stale-ID, and rebuild checks also passed.

Full-workspace gates and their limitations are recorded in the task handoff because other
uncommitted work was changing concurrently. No real model-service quality evaluation was run;
retrieval policy and model inputs were not changed by this patch.

## Shared-workspace gate observations

The full shared-tree run finished with 2,793 passed, 22 failed, and 12 skipped in 691.70 seconds.
It spanned simultaneous edits to retrieval policy, temporal routing, model adapters, benchmark
instrumentation, and runtime capacity. This was not a frozen-source comparison, so these failures
are retained as observations rather than attributed to this patch. The isolated P02 export above
is the evidence for this change. Real-service formation evaluation was skipped without opt-in
credentials. The Jina failure was an import-time Torch `_utils` error.

The latest shared-tree mypy attempt also reported an argument-type error in
`tests/unit/test_memory_api.py:8053`. Shared-tree formatting and link checks observed other
incomplete edits, including execution-pool formatting and a not-yet-created semantic-recall
report. Those files were preserved.

Failed nodes from the full run:

- `tests/unit/benchmarks/test_eval.py::test_eval_config_reuses_the_declarative_memory_schema[override-model-0.0-7]`
- `tests/unit/benchmarks/test_eval.py::test_eval_config_reuses_the_declarative_memory_schema[gpt-5-mini-None-None]`
- `tests/unit/benchmarks/test_eval_arms.py::test_blind_arm_never_reads_memory_and_keeps_query_media`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_layers_use_observed_boundaries_instead_of_returned_hits`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_missing_observations_and_labels_never_become_zero_recall`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_unmapped_reader_ids_and_partial_excerpts_do_not_prove_coverage`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_conditional_scores_count_unknown_observations_separately`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_cache_preserves_original_trace_and_organization_without_inventing_requests`
- `tests/unit/benchmarks/test_eval_diagnostics.py::test_review_export_is_stratified_reproducible_and_has_no_synthetic_gold`
- `tests/unit/benchmarks/test_eval_metrics.py::test_blind_baseline_rejects_the_old_forced_guess_prompt`
- `tests/unit/benchmarks/test_eval_metrics.py::test_blind_baseline_rejects_a_different_answer_policy`
- `tests/unit/models/test_jina.py::test_loaded_jina_uses_pyav_metadata_and_unique_bounded_frames`
- `tests/unit/test_bitemporal_memory.py::test_equal_confidence_state_consolidation_still_revises[0.3]`
- `tests/unit/test_memory_api.py::test_generation_observer_sees_selected_evidence_and_cannot_change_the_answer`
- `tests/unit/test_memory_api.py::test_ask_observes_ranking_before_parallel_speech_failure`
- `tests/unit/test_memory_api.py::test_decay_reranks_softly_and_requires_explicit_reinforcement`
- `tests/unit/test_memory_api.py::test_a_window_bound_to_matched_records_reads_the_dates_their_text_states`
- `tests/unit/test_on_demand_media.py::test_diagnosis_text_ceiling_includes_system_prompt_before_http`
- `tests/unit/test_temporal_lexical_retrieval.py::test_temporal_lexical_routes_preserve_hard_filters_and_global_fallback[hybrid]`
- `tests/unit/test_temporal_lexical_retrieval.py::test_temporal_lexical_routes_preserve_hard_filters_and_global_fallback[lexical]`
- `tests/unit/test_top1_ambiguity.py::test_lexical_ties_are_ambiguous_even_with_full_query_coverage[Alice-lexical]`
- `tests/unit/test_top1_ambiguity.py::test_lexical_ties_are_ambiguous_even_with_full_query_coverage[Alice bakery-lexical]`
