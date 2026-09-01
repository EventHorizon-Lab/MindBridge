# Source-bound answer-quality paired gate

> Protocol audit 2026-09-01: this run used the SDK's implicit provider retries while recording zero
> retries, and its cache identity did not bind store, query text, grounding prompt, or serializer.
> All answer, latency, and token conclusions below are superseded pending a schema-2 cache replay.
> Retrieval-only results are reported separately and are unaffected.

Status: **accepted for further research, not yet accepted for productization**.

## Frozen protocol

- Task/unit: LoCoMo-Refined `conv-26`, the deterministic lowest-SHA unit among the
  first three dataset units.
- Questions: the same frozen 32 lowest-SHA question IDs used by the retrieval gate;
  query-set SHA-256 is
  `4212fa33dec954e8f4552fbb1f0a076cb10d7791a1f8cd6140d753231c70bd50`.
- Public API: `Memory.ask(..., limit=20)` for both arms.
- Order: fixed baseline/candidate alternation from question-ID SHA parity.
- Concurrency: 1. Generation and judge provider retries: 0. Generation best-of: 1.
- Generation: `qwen3.8-flash`, temperature 0, no max-token override,
  `enable_thinking=false`.
- Scoring: the repository's pinned `official_scorers_v2` LoCoMo prompt and parser,
  with the same `qwen3.8-flash` and no-think configuration for both arms.
- Gold answers were loaded only after all 64 paired answers had been generated.
- Every generation boundary asserted that all 20 inputs were byte-identical original
  source memories. Derived fact text and derived records sent to generation: 0.

The configured judge is not LoCoMo-Refined's publication-comparable official
`qwen3-14b` judge. The result is therefore an internal paired experiment, not a
claim about an official LoCoMo score or SOTA.

## Results

| Metric | Baseline | Source-bound keys | Delta |
| --- | ---: | ---: | ---: |
| LLM judge (primary) | 0.34375 | 0.37500 | +0.03125 |
| Token F1 | 0.17969 | 0.23325 | +0.05356 |
| BLEU-1 | 0.13583 | 0.17755 | +0.04172 |
| Evidence coverage@20 | 0.36719 | 0.46875 | +0.10156 |
| Evidence Hit@20 | 0.40625 | 0.50000 | +0.09375 |
| Generation tokens | 111,083 | 114,719 | +3,636 (+3.27%) |
| Errors | 0 | 0 | 0 |

Paired LLM-judge outcomes were 5 gains, 4 losses, and 23 ties. Token-F1 had 9
gains, 6 losses, and 17 ties. BLEU-1 had 6 gains, 8 losses, and 18 ties; its mean
rose because the positive changes were larger. Coverage had 7 gains, 3 losses,
and 22 ties.

The pre-registered primary gate passed because primary quality did not decline and
the candidate introduced no errors. The token-cost objective did not pass: the
candidate used 3.27% more generation tokens, largely because improved retrieval
produced more substantive answers.

## Latency observations

| Metric | Baseline | Source-bound keys | Relative delta |
| --- | ---: | ---: | ---: |
| Ask mean | 3205.2 ms | 2200.0 ms | -31.4% |
| Ask p50 | 1115.6 ms | 1145.3 ms | +2.7% |
| Ask p95 | 15678.5 ms | 8835.7 ms | -43.6% |
| Generation TTFT mean | 2275.8 ms | 1846.1 ms | -18.9% |
| Generation TTFT p50 | 810.9 ms | 827.8 ms | +2.1% |
| Generation TTFT p95 | 12328.3 ms | 8589.4 ms | -30.3% |

These raw values include provider tail latency and a 26-second first baseline call.
The p50 values slightly regress, so this experiment does not establish a speed
improvement. No post-hoc warmup exclusion was applied.

## Integrity and productization boundary

- Before and after reopening, both stores contained exactly 419 authoritative
  memories. Baseline embeddings remained 419; candidate embeddings remained 1,140
  (419 source keys plus 721 query-blind grounded fact keys).
- Both stores retained the same embedding-space ID and current `context-keys-v9`
  recipe. Reopening caused no recipe migration and no loss of fact keys.
- The candidate returned and generated from original source memories only. Facts
  were never separate search results or generation context records.

The answer-quality gate supports continuing the source-bound-key direction. It
does not justify the runtime monkeypatch as a product implementation. A product
design must durably persist each fact, exact quote, character span, prompt digest,
and parent source binding in SQLite so index rebuilds reproduce the same keys while
SQLite remains authoritative. It also needs cross-unit validation (at least the
frozen first three LoCoMo units) before a tracked implementation. Gallery and
EgoLife were not run.
