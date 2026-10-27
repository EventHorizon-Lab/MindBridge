# Benchmark answer policy and context budgets

## Finding and decision

E01 is reproducible: baseline v1 system prompts force guessing, while `Memory.ask` uses the
task's `strict` or `best_effort` policy. This confounds answer accuracy, particularly on
LongMemEval abstention and Mem-Gallery answer refusal. Baseline refusals also went uncounted.

Keep the product contract. Resolve the policy once per task for every grounded answer arm;
reuse the product reader prompts for full-context and compile baselines. Keep blind as a
prior-knowledge control, with the same refusal policy and an explicit prohibition on inventing
user facts. An aligned strict blind control may correctly refuse most personal questions.

Changing only the scorer would hide the request mismatch. Making the product always guess
would remove a measured capability. Neither addresses the cause.

Budgets describe distinct experiments: full-context is bounded raw text, compile is a compiled
bundle, and ask is retrieved evidence. Count full-context separators within its character cap;
record each cap's scope, and never call these character limits token-equivalent. An arm
comparison remains an end-to-end comparison, not a causal estimate of memory alone.

## Implementation plan

- [x] Reproduce policy propagation, refusal accounting, separator overflow, and stale budget
  cache reuse with failing tests.
- [x] Pass task policies to all baseline readers, reuse grounded prompts and answer decoding,
  and retain policy in sample and task provenance.
- [x] Version baseline prompts, include the full-context cap in cache identity, reject external
  blind controls with a different prompt or policy, and document comparison limits.
- [x] Run focused regressions, repository quality gates, and a bounded reader-quality check
  where an available real model permits it. Record unavailable checks explicitly.

## Evidence and validation

### Why this needs a protocol correction

The quantity of interest is downstream success given a memory system. A difference in scores
also depends on the reader, refusal policy, available evidence, representation, budget, and
judge. Holding the reader model fixed while forcing one arm to guess does not isolate memory.
On unanswerable items this can favor the product; on answerable items where retrieval is weak,
the same mismatch can favor the guessing baseline. Its direction is not uniformly pro-product.

The [official LongMemEval protocol](https://github.com/xiaowu0162/LongMemEval) includes abstention
as one of five abilities. It identifies 30 abstention questions out of 500 and excludes those
questions from retrieval scoring because they have no answer location. Its S histories consume
roughly 115k Llama-3 tokens, so this harness's default 24k-character full-context arm is a
bounded stuffing control, not a full-history oracle.

Recent [RAW-RAG evaluation profiles](https://github.com/Litmeb/RAW-RAG/blob/main/scripts/paper/README.md)
publish reader, judge, window size, overlap, retrieval depths, and separate 470-question
non-abstention reports that reuse existing LongMemEval answers. This supports reporting the
answerable slice separately while retaining the full official result. It does not establish
how much MindBridge's score changes, or justify deleting refusal items from its overall score.

### Implemented boundary

Baseline prompt protocols are now `mindbridge_blind_v2` and `mindbridge_full_context_v2`.
Full-context and compile use the same base strict/best-effort prompts and decoder as the
product adapter. The task-policy override reaches every grounded baseline. Dataset-specific
refusal sentences and the product marker both count in baseline refusal diagnostics. Random
retrieval and the product's host-assistant compile surface still request no ask policy.

Full-context counts the two-character separator before admitting the next whole item. Its
cap and both baseline prompt versions participate in response-cache identity. External blind
controls with a different prompt or task policy are rejected. Product arm provenance also
records actual per-task compiler caps for host-assistant compile tasks.

The scorer is unchanged. A recognized refusal is not automatically a correct answer: Mem-Gallery
AR expects `Not mentioned.`, and the product's generic unknown sentence can still receive zero
under F1. This correction does not translate failed format compliance into a passing score.

No SDK, REST, MCP, dependency, or on-disk product schema change is needed. The change belongs
to the benchmark harness. Old forced-guess scores must retain their v1 identity and be rerun
before a policy-aligned comparison.

### Real-model check

Fixture `mindbridge-baseline-policy-quality-v1` compares v1 and v2 on six synthetic cases,
with one deterministic request per prompt and case. Passing requires a sole expected answer
word for grounded/public questions, and an explicit abstention marker or declared refusal for
missing information. The injected instruction must not replace the grounded answer.

| Fixture | Old v1 | Aligned v2 |
| --- | --- | --- |
| Grounded toolbox color | Pass | Pass |
| Missing private launch code, with unrelated evidence | Fail | Pass |
| Instruction embedded in stored evidence | Pass | Pass |
| Gallery-style declared refusal | Pass | Pass |
| Blind private launch code | Fail | Pass |
| Blind public capital question | Pass | Pass |

Reader: local Ollama `qwen2.5:7b`, Q4_K_M, model digest
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
All returned model identities were `qwen2.5:7b`. Controls: CPU only, 8 threads,
temperature 0, seed 7, 2048-token context, 96 output tokens. All requests ended normally.
The [safe report](artifacts/e01-answer-policy-20261027.json) retains grades, model identity,
prompt hashes, and token counts; generated answer bodies are not retained.

Reproduce with the checked-in synthetic fixtures:

```bash
uv run --frozen python benchmarks/answer_policy_quality.py \
  --output /tmp/mindbridge-answer-policy-quality.json
```

The configured Qwen3.8 service's `/v1/models` request timed out after 10 seconds. No deployed
reader check, complete LongMemEval/Mem-Gallery rerun, or real-model best-effort comparison was
completed. The local check establishes narrow strict-policy behavior, not deployed-reader
quality, exact Mem-Gallery formatting compliance, or statistically reliable dataset uplift.
The gallery-style case passed under both prompts: policy conflict does not imply every reader
will follow the forced-guess instruction on every item.

### Repository validation

Focused tests reproduce each new failure before its fix, including product compile-budget
provenance. Independent read-only review found no critical or important E01 defect and identified
the compiler-budget provenance issue, which now has a regression test.

Validation was performed in a shared, dirty working tree with other changes actively arriving;
the complete run is not evidence of a frozen patch. No unrelated changes were reverted.

| Check | Observed outcome |
| --- | --- |
| Complete eval, answer-arm, and metric test files | 330 passed in 26.17 seconds on the final focused run |
| Policy, surface, and full-context cache namespace checks | 4 passed |
| `uv lock --check --default-index https://pypi.org/simple` | Passed |
| Ruff format and lint on the seven changed Python files | Passed |
| Full Ruff format/lint | Not green in the shared tree; the final check reported concurrent `test_eval_diagnostics.py` formatting/import ordering |
| `uv run --frozen mypy` | Passed, 274 source files at that snapshot |
| Pinned Markdown lint | Changed documentation passed; full run passed at 177 files, then a later repeat found six errors in another task's `.superpowers/boundary-cleanup/` scratch Markdown |
| Pinned Lychee link check | No broken links; one timeout on the unrelated W3C SKOS link in `2026-10-27-predicate-stability.md` |
| `git diff --check` | Passed |
| `uv run --frozen pytest -W error` | 2829 passed, 12 skipped, 7 failed; 713.44 seconds |

The complete run's failures were inspected individually. None points to the E01 policy,
separator, or cache-cap change:

- `test_response_cache_namespace_changes_with_runner_recipe`: concurrent diagnostics schema
  changed to 20 while the loaded test still expected 19.
- `test_a_fully_cached_unit_completes_its_ingest_share_without_writing`: concurrent cached-answer
  diagnostics added `generation_evidence`, absent on the loaded test double.
- `test_runner_reports_cached_progress_before_pending_answer_finishes`: its background cached
  path hit the same missing field, then its progress wait timed out.
- `test_global_help_lists_only_supported_runners_without_importing_them`: concurrent
  `review-samples` runner was absent from the loaded expected command set.
- `test_state_revision_uses_grounded_confidence_before_retiring[True]`: bitemporal confidence
  assertion, in a product module this change does not modify.
- `test_semantic_recall_can_deepen_the_original_query_for_structured_anchors`: semantic recall
  depth assertion, in a product module this change does not modify.
- `test_async_closed_memory_keeps_public_error_contract`: concurrent execution-capacity code
  surfaced a closed executor's `RuntimeError`.

These observations are not a claim that the repository passes all quality gates. They also do
not establish current failures after other contributors finish their changes. A subsequent
focused rerun confirmed that all 330 eval/arm/metric tests pass, including the three cached-answer
and schema failures above. The separate global-help assertion still failed on `review-samples`.
