# Grounded retrieval research closeout

Snapshot: 2026-08-31; protocol audit: 2026-09-01. Product revision:
`36447989a9634c0aab2106d5adb182c5ed437e57`.

## Decision

Keep the product at the current revision. This round found no additional candidate that satisfies
the project's lexicographic objective of stronger, then faster, then cheaper. Rejected experiments
remain research artifacts and did not change tracked product code, schemas, or public APIs.

The 2026-09-01 audit withdrew the prior M3 payload and source-bound answer-quality conclusions. The
conservative no-merge decision remains correct, but the invalidated numbers are not evidence for or
against those candidates until a corrected no-retry replay is run.

No full EgoLifeQA evaluation was run. Existing EgoLife evidence is limited to previously frozen
subsets and is not a full-dataset claim.

## Candidate outcomes

| Candidate | Stronger | Faster | Cheaper | Decision |
| --- | --- | --- | --- | --- |
| Stronger grounded-abstention prompt | Not measurable without gold evidence pointers | Not run | Not run | Rejected before cloud evaluation to avoid reward hacking |
| Remove private `memory_id` from generation payload | Prior four-task macro and M3 score withdrawn: replay omitted product `created_at` fallback | Not established | Prior token result is diagnostic only | No product change; corrected cloud replay required |
| Replace `memory_id` with request-local `evidence_index` | Prior M3 score withdrawn for the same payload defect | Not established | Prior token result is diagnostic only | No product change; corrected no-retry replay required |
| Store extracted facts as separate memories | Provenance-folded retrieval improved, but raw Hit@1/5/10/20 became zero through derived-record crowding | Regressed | Added extraction and index cost | Rejected |
| Bind grounded fact keys to original sources at K20 | Two-holdout retrieval macro Hit@20 and Recall@20 both +6.25 pp; prior answer judge withdrawn | Holdout search p50 regressed slightly | Larger index; prior generation-token result withdrawn | Retrieval-qualified representation only |
| Source-bound keys at K8 | Holdout retrieval macro missed its +5 pp gate by 0.3125 pp; prior answer judge withdrawn | Prior ask/TTFT result withdrawn | Prior generation-token result withdrawn | Not promoted |
| K12/K16 answer ladder | Not run | Not run | Zero cloud calls | Cancelled because holdout rank data had already exposed both cutoffs |
| Query-embedding cache or executor rewrite | No representative quality-safe speed candidate | Repeated-query caching would measure an artificial workload | Added lifecycle/memory cost | Skipped |

The source-bound representation is the only retrieval direction that generalized across the three
frozen LoCoMo units. It attaches grounded fact and exact-quote embeddings to the original source ID,
so search returns only original memories. Productization is deferred until a new, independently
preregistered experiment covers durable SQLite provenance and rebuilds, corrected answer quality,
cross-modal tasks, latency, and token cost.

## Protocol audit and repair

- M3 replay payloads now delegate to the current product serializer, preserving its mutually
  exclusive `occurred_at`/`created_at` behavior. Frozen offline validation passes for both affected
  runners, but no replacement cloud result was produced.
- The score replay now explicitly adds `hit.score` to its control arm; its compact arm is the exact
  current product payload.
- Every artifact claiming zero provider retries now constructs its OpenAI client with
  `max_retries=0`.
- Candidate-pool resume rewrites its journal with only groups containing exactly one row for each
  frozen candidate; partial and duplicate groups are rerun.
- Answer caches use schema 2 and bind full query content, logical SQLite plus assets for both
  stores, limits/seed, endpoint, grounding prompt, and serializer source. The K8 override uses the
  same identity builder.
- Fact extraction resumes reject changes to schema, task, source, model, prompt, fact limit,
  concurrency, blindness, or parse-retry policy before appending.

Superseded generation artifacts are retained byte-for-byte for auditability: the M3 memory-ID,
M3 Q01 repetition, evidence-index, source-bound K20 answer, and source-bound K8 answer results.
They must not be used as acceptance evidence.

## Source-bound evidence

| Unit | Baseline Hit@20 | Candidate Hit@20 | Baseline Recall@20 | Candidate Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| `conv-26` dev | 40.625% | 50.000% | 36.719% | 46.875% |
| `conv-30` holdout | 37.500% | 46.875% | 35.938% | 45.312% |
| `conv-41` holdout | 34.375% | 37.500% | 26.562% | 29.688% |

Extraction was query- and answer-blind. Every accepted fact carried an exact source substring and a
locally recomputed character span. Across the two holdouts, 1,471 grounded keys were added to 1,032
source records, no derived record reached Top20 or generation, and retrieval returned 20 unique
original sources per question. The K20 holdout gate passed; the combined low-K experiment was
not promoted because frozen K8 missed its +5 pp macro threshold by 0.3125 pp. Cloud-derived answer,
latency, and token values from the old answer caches are superseded.

## Speed finding

A separate 216-search audit over fixed LoCoMo, Mem-Gallery, and ATM text queries produced zero
errors and 100% stable ordered Top20 IDs across three warm rounds. Jina query embedding was the
largest stable p50 stage; ATM's p95 instead varied in Zvec and SQLite. A separate WeMM process shared
the GPU, so these measurements are diagnostic and not isolated throughput claims. No speed-only
product change was justified.

## Verification boundary

The pinned command

```bash
uv run --frozen python autoresearch/orchestrator-260830-2214/verify.py --suite dev
```

currently reports `52`, not the required `0`. Its archived evidence has a quality macro delta of
-1.118 pp and two hard regressions: M3 Robot `0.298851 < 0.310345` and the existing EgoLifeQA subset
`0.040000 < 0.120000`; speed evidence is also incomplete. The EgoLifeQA value is not from a full run,
and no full run was started to resolve it. Consequently this round makes no five-task SOTA,
universally faster, or universally cheaper claim.

Raw benchmark-derived outputs and local SQLite/Zvec clones stay untracked to avoid committing
dataset content and roughly 20 GB of runtime state. `round-grounded-retrieval.sha256` records the
digests of retained evidence, including the explicitly superseded artifacts.
