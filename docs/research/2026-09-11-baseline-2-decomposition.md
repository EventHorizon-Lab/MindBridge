# Baseline-2 decomposition — 2026-09-11

Run: `mindbridge-bench eval --config .benchmarks/baseline_2.yaml`, run id `baseline-20260911`, code
`origin/mindbridge-v03` at `112ef9a6`. Answerer and judge `qwen3.8-27b` (non-thinking, no
`temperature` set), embedder `tencent/WeMM-Embedding-2B` (2048-d), speech FunASR, `recall_limit 12`,
`answer_policy` per task (strict everywhere except `m3-bench-robot`), no `vision:` slot. Every number
below is read from `results/baseline-20260911/samples.partial.jsonl` (3,638 samples: locomo-refined
1,382, m3-bench-robot 1,276, memlens 4×195, mm-lifelong-day-test 200) or from paired re-runs of the
exact stored reader request; nothing is comparable with a published leaderboard row.

## What the score is made of

Reported: locomo-refined 0.7135, memlens-32k 0.4769, -64k 0.4923, -128k 0.4359, -256k 0.4205,
mm-lifelong-day-test 0.0525.

| Task | Abstained | Gold in the 12 grounded hits | Abstained *with* gold in the window |
| --- | --- | --- | --- |
| locomo-refined (1,382) | 159 (11.5 %) | 1,253 (90.7 %); 70 more within the 36-candidate pool; 54 nowhere | 88 (55 % of abstentions) |
| memlens, per window (195) | 93–108 (48–55 %) | phrase-answer questions: 36–48 labelable, gold in window for 60–75 % | 8–15 per window (≈25 % of labelable) |
| mm-lifelong-day-test (200) | 132 (66 %) | ref@300 0.10; 0 visual descriptions in the store | — |
| m3-bench-robot (1,276) | 868 (68 %) | — | — |

Gold labels: LoCoMo `evidence_ids` are exact; MemLens has none, so a turn is "gold" when it contains
the reference answer string (phrase questions) or a `$` amount for the item (sum questions).

Where the LoCoMo points go, by count: 159 abstentions (88 with gold shown to the reader), ≈137
committed wrong answers with gold shown (12 % of answered questions under the same judge, see
control below), 124 questions whose gold is outside the 12-hit window (26 have a gold neighbour at
±1 turn, 38 at ±2, 60 are plain misses).

MemLens by question type (64k window; the other windows look the same): `answer_refusal` 22/22
refused, which is correct; `temporal_reasoning` 23/48 abstained; `information_extraction` 26/61;
`multi_session_reasoning` 13/35; `knowledge_update` 9/29. The sum questions ("how much total have I
spent on X") have gold sets of 1–23 turns; the 12-hit window holds 0–3 of them and the 36-candidate
pool 1–5, so their committed answers are systematically low ($560 for $590, $175 for $240) and the
refusals are genuine. `knowledge_update` answers usually live only in the image ("that al pastor taco
looks fantastic" is the assistant's reply; the user turn says "a taco filling like that" plus a photo
whose release caption is `image content`).

## Reader re-runs on the stored requests

The exact request each refusal was built from (same 12 memories read back from the unit store,
same system prompt, same `extra_body`) was sent again to the same endpoint. Correctness is the
same model judging prediction against the references; abstention is the product marker.
`fail` = the 70 LoCoMo refusals with gold in the window whose unit store was still readable;
`ctrl` = 136 random answered LoCoMo questions with gold in the window; `nogold` = 118 LoCoMo
questions with gold outside the window; `mlrefusal` = the 88 MemLens must-refuse questions;
`mlabst` = 200 MemLens refusals on other types (text-only hits, images not re-attached).

| Arm | fail (70): abstain / correct / wrong | ctrl (136) | nogold (118) | mlrefusal (88) refused | mlabst (200) refused |
| --- | --- | --- | --- | --- | --- |
| stored prompt, provider-default temperature | 49 / 12 / 9 | 1 / 119 / 16 | 55 / 35 / 28 | 87 | 173 |
| stored prompt, `temperature 0` | 56 / 11 / 3 | 1 / 117 / 18 | — | — | — |
| evidence as plain numbered text instead of JSON | 58 / 7 / 5 | — | — | — | — |
| "enough when a hit states or implies the answer, even if names, dates or wording differ" | 21 / 32 / 17 | 0 / 119 / 17 | 37 / 40 / 41 | 87 | 154 |
| same, without the "names differ" licence | 51 / 14 / 5 | 2 / — / — | 54 / 35 / 29 | 87 | 166 |

Readings:

- Refusal with the evidence in front of the reader reproduces 70 % of the time; the rest is sampling
  noise from the unset temperature. `temperature 0` makes refusal *more* frequent (it is the modal
  token), so it is a reproducibility knob, not a quality lever.
- The JSON rendering is not the problem; plain text refuses more.
- The only wording that cut over-refusal did it by licensing attribution mismatches ("Melanie's
  hand-painted bowl" answered from Caroline's bowl; "Jean and John" answered from Gina and Jon).
  That is LoCoMo question noise, not memory quality, and on gold-absent questions it turned 18
  refusals into 5 right and 13 wrong answers. Removing that clause removes the whole effect. The
  strict reader is therefore not fixable by prompt wording without teaching it to guess; the
  existing `best_effort` policy is the honest version of the same trade and was already measured
  ([recall programs round](2026-09-10-recall-programs-round.md): memlens-32k holdout +0.20,
  locomo +0.039, m3-robot +0.124).
- The MemLens must-refuse questions stay refused under every arm (87/88), so the 22 per window
  scored as correct refusals are not at risk from the reader.

## Retrieval

- LoCoMo: gold rank median 0; 4 % plain misses are semantic gaps ("What are Caroline's plans for
  the summer?" against "Researching adoption agencies — it's been a dream…"), not a defect. The
  question's time phrase is a rank boost, never a filter (`_temporal_factor`), and the harness
  already supplies the unit's latest memory time as the reference clock for undated questions, so
  the "last weekend"-style questions are not mis-anchored.
- Adjacent-turn expansion (±1 in corpus order, the `neighbor_memories` read already in the store)
  would put gold into the window for 26 more LoCoMo questions (1.9 %), ±2 for 64 (4.6 %), at 2–3×
  the grounding tokens. Not built; the ceiling is written down here so nobody re-derives it.
- Set-shaped MemLens questions need exhaustive retrieval, not a wider window: even the 36-candidate
  pool holds ≤5 of 8–23 gold turns. The `recall_planning` set plan exists and is off by default for
  the reasons the 09-10 round recorded.

## Media

`visual_descriptions` is empty in every store of this run: 24 image assets per MemLens unit and all
2,831 MM-Lifelong clips were embedded natively but never described, because `baseline_2.yaml`
declares no `vision:` slot (`Memory` skips derivation when `vision_describer is None`). For
MM-Lifelong that is the root cause already recorded on 2026-09-11 (0 captions, 66 % refusals, no
event time); PR #186 supplies the committed-answer policy and the event time, the slot supplies
the text. For MemLens the 09-10 round measured raw pixels alone (−0.017, n.s.) but never
descriptions; that measurement is below.

### memlens-32k dev slice (first 60 questions), strict, same code and seed

Both arms ran on `112ef9a6` with `baseline_2.yaml` settings, `unit_concurrency 2`; the vision arm
adds `vision: {provider: openai, model: qwen3.8-27b, modalities: [image]}` on the same endpoint. Every
image in the slice received a description (≈13 per unit; e.g. "A Costco Gasoline station canopy with
three cars parked underneath…"), and the strict control reproduces the 09-10 round's 0.300 exactly.

| Arm | Score (60) | Abstained | W/L/T vs strict | multi_session (30) | knowledge_update (23) | temporal (7) |
| --- | --- | --- | --- | --- | --- | --- |
| strict, no `vision:` | 0.3000 | 27 | — | 0.367 | 0.304 | 0.000 |
| strict + image descriptions | 0.2833 | 23 | 1 / 2 / 57 | 0.333 | 0.304 | 0.000 |

The ten questions that changed: four refusals became committed answers and all four were wrong
("Milton stainless steel water bottle" for the copper bottle, "Rose Milk Bath" for the bath bomb,
8 guitars for 5); one refusal became a correct "barbell"; two correct answers flipped to wrong
("woven carry-all" for the straw tote, "Yes" for the moka pot); two wrong answers became refusals.
Descriptions make the reader trust whichever image the ranking brought, and on this slice that is as
often a look-alike as the gold. Together with the 09-10 pixel result (−0.017) this closes the
"MemLens loses because the images are opaque" hypothesis for this reader: the loss is retrieval
selecting the wrong image among near-duplicates, which descriptions do not fix and can worsen.

## What this run does and does not say about the gap

- MemLens: the 09-10 round's literature check puts text-only memory agents at 30–33 and Sonnet 4.5
  full context at 36.5 on the 32k split; this run's 0.42–0.49 is above both, with the caveat that
  the judge here is the answering model, not the official `qwen3-235b-judge`.
- LoCoMo 0.71 under a qwen judge is not comparable with the 0.75–0.92 rows published under a
  GPT-4o-mini judge; the same-judge holdout in the 09-10 round put strict at 0.70 and best-effort
  at 0.74.
- MM-Lifelong 0.05 against ReMA 16.75 is a real gap with three known, addressable causes (policy,
  event time, descriptions); none of them is a retrieval or reader defect found in this run.

## Recommendations

1. Keep `strict` as the product default; make the benchmark policy an explicit choice in the run
   config (`answer_policy: best_effort` for the tasks whose protocol gives no credit for refusal),
   as the 09-10 round did, instead of letting the table decide silently.
2. Set `temperature: 0` and a `seed` on `generation` in measurement configs; 30 % of the refusals
   here are sampling noise and the ±3-point run-to-run drift the 09-10 round saw has the same
   source. Expect slightly more refusals, not fewer.
3. Configure `vision:` (image and video) for MM-Lifelong, where the clips carry no text at all; do
   not expect it to move MemLens, where the measured effect is nil.
4. Merge PR #186 before the next MM-Lifelong run.
5. Do not tune the strict prompt against LoCoMo: the only wording that helps does so by accepting
   the dataset's attribution noise.

Artifacts: scratch scripts `recon.py` (rebuilds a reader request from a unit store),
`exp_reader.py`, `judge.py`, `adjacency.py`, `locomo_temporal.py` under the session scratchpad;
per-arm outputs `exp_<slice>_<arm>.judged.jsonl`.
