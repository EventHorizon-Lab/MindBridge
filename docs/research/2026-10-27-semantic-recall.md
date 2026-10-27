# Controlled semantic recall — 2026-10-27

## Finding and decision

R01 is confirmed in the checkout based on `b6b5ad28`. `_RecallReads.similar` discarded
`query` and returned `context.ranked[:k]`. Without `evidence_budget_chars`, that ranking
contained at most `min(100, limit * 3)` results. The default character budget is 24,000,
which already requests 100 results; the smaller depth is conditional, not the default.
Larger `k`, different query strings and replanning reused the same pool. Furthermore,
`_grounded_recall` ignored a similarity-only program's ranked result. Executing a new
query without changing grounding would therefore still fail to deliver its evidence.

This is a limitation of the existing reuse policy, not evidence that all planning is
ineffective. Predicate, temporal, neighbor and entity reads can add evidence; completeness,
selectivity and grounding protections remain useful. The
[historical recall evaluation](2026-09-10-recall-programs-round.md) and the point-plan
regression describe previous decisions. They are not new experiments in this audit.
Likewise, [FP32 rescoring](2026-10-26-accuracy-first-retrieval.md) improves candidates
already discovered and cannot recover a parent absent from every route.

The repair is a disabled-by-default experiment, `recall_query_expansion`. It makes planned
queries executable through the existing retrieval plane and ensures their new evidence can
reach the reader, while retaining a comparable unchanged default arm. No dependency,
on-disk schema, endpoint or MCP tool schema is added. The Python constructor and
`MemoryConfig` gain the documented optional boolean.

## Research context

Three original papers were retrieved and read in full through OpenResearch on 2026-10-27.
These are external task-specific findings, not MindBridge quality measurements:

- [EviQE](https://www.alphaxiv.org/pdf/2609.14875) separates complementary query portfolios
  from evidence selection. With fixed retriever, generator and document budgets, relevance
  selection outperforms pooling alone; further expansion rounds can degrade results.
  The authors write: “Additional retrieval-generation rounds provide little benefit once
  strong conditioning evidence has been selected and can reduce effectiveness.”
- [Beyond One-Shot Expansion](https://www.alphaxiv.org/pdf/2609.07050) evaluates sequential
  evidence exploration and coverage-aware selection on three multi-hop datasets. It reports
  retrieval coverage and downstream QA separately. Its ablations support testing missing-hop
  acquisition; its full system also includes facets and evidence-conditioned probes, which
  this patch does not implement.
- [Faithful Query Resolution](https://www.alphaxiv.org/pdf/2610.15034) reports a ranking/recall
  tradeoff when combining faithful and alternate queries. Better retrieval scores do not
  establish better conversational utility, and the interactive runs include other changes.

These findings justify a bounded experiment. They do not justify making rewriting the default
or importing a paper's fusion method into a system whose own historical fusion test lost recall.

## Implemented controls

- Planning must also be enabled. Default planning continues to reuse the original ranking.
- One reader cache survives all replanning rounds of one `ask`; independent calls share none.
  Exact query strings, after trimming their ends, identify cache entries.
- At most two additional searches run per `ask`, including any original-query deepening.
  Each requests 100 ranked results once, so increasing `k` reuses the cached prefix. Failed
  embedding calls consume a slot and are cached. Storage/index failures retain their errors.
- Rewrites are text inputs to the existing retrieval plane. The original scope, memory type,
  reference clock and inferred temporal intent remain authoritative. Original multimodal
  preparation is reused only when deepening the original query.
- Existing grounding is retained. Each executed query can append at most `limit` novel rows;
  character and media caps can admit fewer. Query-local scores never displace or reorder the
  original evidence. This also applies to plans containing exhaustive reads.
- Semantic additions are explicitly described as incomplete rankings. They cannot license a
  total over the predicate's set. No provider failure body, query or evidence content is added
  to ordinary telemetry.
- Failed or quota-blocked semantic anchors mark dependent neighbor/time reads incomplete,
  including transitive dependencies. An unread anchor cannot license a complete empty set.
  The OpenAI adapter changes its system-level operator definition only when the kernel
  advertises the experimental capability; the default planner system prompt is unchanged.

Packing charges existing evidence first and applies the smaller of the configured evidence
and set character budgets to additions. Thus a saturated original budget may admit no semantic
rows. The mandatory original window remains a soft-budget floor. This preserves existing
grounding rather than asserting that expanded evidence is always more valuable.
Per-query selection takes the first `limit` novel rows before packing. A large selected row
can therefore be rejected without backfilling from lower ranks of that query. The selection
and budget counters distinguish this remaining limitation; this experiment does not replace
the evidence selector or promise optimal packing.

## Reproducible quality evidence

The versioned fixture is `semantic-recall-v1` in
[`tests/benchmarks/test_semantic_recall.py`](../../tests/benchmarks/test_semantic_recall.py).
Embedding model/space are `semantic-recall-v1` / `semantic-recall-v1:4`; the deterministic
reader is `semantic-recall-reader-v1`. The planner supplies a fixed, oracle rewrite. There is
no learned model, provider revision, credentialed call or real-service check in these results.
The local run used Linux x86-64, CPython 3.12.14, NumPy 2.3.5 and Zvec 0.8.0.

The paired native-index test ingests two supporting records and 230 distractors through the
public SDK into a fresh physical directory for every arm. The second support lies outside
both the original NONE route and the wider INT8 route, verified with `search_with_trace`.
The answer uses `limit=1`, no character widening, a fixed reference clock, and no reinforcement.
Seeds 20261027, 20261028 and 20261029 vary ingestion order; both NONE and INT8 are tested.
Acceptance requires keeping the original support and delivering both supports to the reader
with expansion, without extra searches in the control arm.

| Measurement | Reuse control | Controlled expansion |
| --- | --- | --- |
| Full support in original route candidates | 0/6 | 0/6 |
| Full support delivered to reader | 0/6 | 6/6 |
| Deterministic reader success | 0/6 | 6/6 |
| Query embedding calls per ask | 1 | 2 |

Four additional native-index controls separate complete delivery, character-budget rejection,
a reader deliberately ignoring delivered evidence, and absent support. Stored instructions
appear as ordinary distractor data; the fixed models do not establish real prompt-injection
resistance. All ten native-index tests passed with warnings treated as errors:

```bash
uv run --frozen --no-sync pytest -W error tests/benchmarks/test_semantic_recall.py -q
```

Unit regressions cover default parity, delivery beyond the original pool, query caching and
the shared ceiling across replans, type/place/temporal constraints, failed query fallback,
point and exhaustive-plan row caps, same-query depth for structured anchors, and safe counts
of candidates, selected rows and budget losses. Existing store ownership, stale-ID hydration
and missing-index recovery tests exercise the shared storage/retrieval boundary.

## Diagnosing the remaining losses

| Boundary | Evidence to inspect | Interpretation |
| --- | --- | --- |
| Discovery to ranked result | Public retrieval trace and `query_new_candidates` | Missing support here needs another route/query or better candidate discovery; rescoring alone cannot recover an absent candidate |
| Ranked result to grounding | `query_selected`, `query_grounded`, `query_budget_dropped`, `query_omitted` | Separate per-query selection limits from character/media packing; more candidates alone may not reach the reader |
| Grounding to answer | Actual provider input diagnostics and task-specific answer scores | Delivered support can still be omitted by provider packing or ignored/misused by the reader |

Query execution/failure/limit counts are cumulative per ask; candidate and grounding counts
describe one round. They count records rather than gold evidence and cannot prove correctness
without labelled fixtures. Provider-side packing happens later.

## Validation status

The final focused run passed 30 semantic-recall, planner-request and native-index tests with
warnings as errors. Strict mypy passed for the eleven touched product/test modules; their Ruff
lint and formatting checks passed. The seven touched documentation pages passed pinned
Markdownlint. Lock consistency and `git diff --check` passed.

The shared-workspace full run recorded 2,861 passes, 12 opt-in real-service skips and five
failures while unrelated changes were being written concurrently. All five subsequently
passed in a fresh targeted run:

- `test_a_fully_cached_unit_completes_its_ingest_share_without_writing`
- `test_runner_reports_cached_progress_before_pending_answer_finishes`
- `test_returned_metadata_cannot_replace_observed_reader_source_provenance`
- `test_global_help_lists_only_supported_runners_without_importing_them`
- `test_cancellation_during_terminal_cleanup_remains_cancellation`

The full suite was not repeated after those concurrent changes. Later whole-workspace Ruff
lint and strict mypy passed. The last formatting check still reported an unrelated blank-line
change in `benchmarks/eval_judge.py`; whole-workspace Markdownlint also observed errors in
ignored agent plans. The final full Lychee run checked 891 links: 882 succeeded, eight redirected and one
timed out at the W3C SKOS mapping link in the predicate-stability report. None of this page's
links failed. These results establish the focused scope and record the incomplete global gate;
they do not assert a clean immutable snapshot of the shared workspace.

## Promotion gate and limits

The mechanism tests pass; real natural-language quality is unmeasured. Replanning still sees
counts, dates and failure reason rather than the preceding evidence text. This is not an
implementation of adaptive entity discovery or a guarantee of complete multi-hop retrieval.

Before enabling this for a workload, pin the actual embedding, planner, reader and judge
snapshots, retain safe returned model/request identities, and compare physically isolated
paired arms on development and held-out data. Measure complete-support recall before packing,
after packing and after provider encoding, plus answer accuracy, groundedness, abstention,
query count, tokens and p50/p95 latency. Include point questions, multi-hop questions,
insufficient evidence and stored adversarial instructions. Require a positive held-out answer
gain with a paired confidence interval excluding zero, no guard-task regression, and an agreed
cost ceiling. No such real-service evaluation was run in this change; the switch stays off.
