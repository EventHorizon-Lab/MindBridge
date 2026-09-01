# Generation `memory_id` removal decision

> Protocol audit 2026-09-01: the M3 payload omitted the product serializer's `created_at` fallback,
> and the Q01 repetition did not enforce its recorded zero-retry protocol. The M3 values, four-task
> macro, and rejection rationale below are superseded pending a corrected cloud replay. The product
> field remains unchanged by default; this historical report is not acceptance evidence.

Date: 2026-08-31. Decision: **reject the current removal candidate**.

## Contract and mechanism

`memory_id` is private generation-prompt data, not the source of `AnswerResult.hits` or search
results. Removing it from `_hit_payload` would therefore preserve structured public hit IDs, while
preventing the answer model from copying the opaque internal ID. The grounded system prompt already
directs source-ID questions to `metadata`.

The mechanism is real:

- ATM emitted a full 64-hex internal ID in four with-ID answers and none without the field. Its
  three source-ID list gains were stable in all three follow-up paired repetitions.
- One LoCoMo with-ID answer copied five real retrieved-ID prefixes; without the field it used valid
  `metadata.source_id` values.
- M3 Q14 copied one full internal ID with the field and none without it.

## Frozen-payload results

Every comparison fixed the archived ordered hits, read-only store, system prompt, question, media,
model settings, and scorer. The only generation-request difference was the presence of each hit's
`memory_id` field. No EgoLife full run was performed.

| Task | Questions | With ID | Without ID | Change | Generation tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| ATM-SGM | 102 | 27.4510% accuracy | 29.0850% | +1.6340 pp | 2,324,217 -> 2,201,044 (-5.30%) |
| LoCoMo-Refined | 32 SHA-fixed | 34.3750% surrogate judge | 34.3750% | 0 | 120,569 -> 81,785 (-32.17%) |
| Mem-Gallery | 32 SHA-fixed | 55.2911% official F1 | 57.0465% | +1.7554 pp | 366,677 -> 328,068 (-10.53%) |
| M3 Robot bedroom | 15 | 60.0000% surrogate accuracy | 53.3333% | **-6.6667 pp** | 868,697 -> 852,212 (-1.90%) |

The first three task gates passed. LoCoMo deterministic token-F1 changed by -0.214 pp and BLEU-1
by -0.482 pp, both within the preregistered 0.5 pp per-task tolerance, and manual review found no
semantic degradation. Gallery F1, exact match, BLEU, and surrogate judge all improved.

M3 failed the stronger-first gate: one loss, no gains, and fourteen ties. The only score flip was
Q01, where the with-ID arm answered `taller coat rack` and the without-ID arm abstained. The archived
baseline also abstained, so provider variance was plausible rather than a clean causal loss.

## Preregistered M3 Q01 resolution

Q01 was therefore repeated exactly five times, sequentially, with orders
`AB/BA/AB/BA/AB`. The repetition count and acceptance condition were fixed before results were
observed; there was no retry or best-of selection.

- With-ID official scores: `[0, 0, 0, 1, 0]`, sum `1`.
- Without-ID official scores: `[0, 0, 0, 0, 0]`, sum `0`.
- Paired wins/losses/ties for removal: `0/1/4`.
- Acceptance required without-ID sum >= with-ID sum and losses <= wins. Both conditions failed.

Across the four main strict runs, macro primary change was about **-0.819 pp**. Weighted generation
tokens fell from 3,680,160 to 3,463,109 (-5.90%), but token and identifier-leak gains cannot override
the failed M3 quality gate under the project's `stronger > faster > cheaper` priority.

## Recommendation

Do not merge the current deletion. Restore the product payload and keep structured public IDs
unchanged. A subsequent general candidate may test replacing the 64-hex value with a short,
non-sensitive request-local `evidence_index`: this could retain an evidence anchor while removing
opaque-ID leakage and most token cost. It must pass the same frozen ATM, LoCoMo, Gallery, and M3
gates before adoption; no benchmark-specific routing is acceptable.

## Artifacts

- `memory-id-atm-closed-ab.json`
- `memory-id-atm-flips-3rep.json`
- `memory-id-locomo-sha32-ab.json`
- `memory-id-gallery-sha32-ab.json`
- `memory-id-m3-bedroom-n15-frozen-ab.json`
- `memory-id-m3-q01-5rep.json`
