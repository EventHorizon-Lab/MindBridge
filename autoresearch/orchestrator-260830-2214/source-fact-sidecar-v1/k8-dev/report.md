# Conv-26 source-bound K8 answer gate

> Protocol audit 2026-09-01: the answer cache did not bind immutable store, query, prompt, and
> serializer inputs. The quality, latency, and token values below are superseded and must not be used
> as acceptance evidence. The independent K8 retrieval holdout gate remains valid.

Verdict: **REJECTED**

| Metric | source-only K20 | source-bound K8 |
| --- | ---: | ---: |
| LLM judge | 0.343750 | 0.312500 |
| Token F1 | 0.191294 | 0.222685 |
| BLEU-1 | 0.149817 | 0.173633 |
| Evidence coverage@arm limit | 0.367188 | 0.390625 |
| Evidence Hit@arm limit | 0.406250 | 0.406250 |
| Generation tokens | 111015 | 48721 |
| Ask p50 ms | 1248.44206252601 | 909.7751690133009 |
| Ask p95 ms | 20926.8325479934 | 3717.81695401296 |
| TTFT p50 ms | 868.2493390224408 | 643.1526164815295 |
| TTFT p95 ms | 13729.03219301952 | 2921.645174967125 |
| Errors | 0 | 0 |

Generation was fixed at qwen3.8-flash, temperature 0, seed 1234, no-think, no retries, and best-of 1. Question order is the frozen lowest-SHA 32; arm order alternates by question-ID SHA parity. A fail-closed guard verified that every generation hit used an original source record ID and exact original source content.

Acceptance checks: primary_not_lower=fail, errors_zero=pass, generation_tokens_lower=pass, ask_p50_lower=pass, ttft_p50_lower=pass
