# Baseline loss decomposition — 2026-09-12

Source run: `.benchmarks/results/baseline-qwen38-27b-wemm9b-20260911` (`mindbridge-bench eval
--config .benchmarks/baseline.yaml`, runner at `7d2dd30`, answerer and proxy judge `qwen3.8-27b`,
embedder `tencent/WeMM-Embedding-2B`, `recall_limit 12`, `answer_policy strict`). Every number here
is internal to that harness and judge; none is comparable to a published leaderboard row. Where an
arm was replayed, "replay" means the same reader, the same reconstructed evidence window and the
same judge protocol, run outside the harness on a fixed subset of the baseline's own questions.

## The question

The headline table says LoCoMo 0.717, LongMemEval 0.754, ATM-main 0.572, MemLens 0.43–0.49, with a
32.5 % abstention rate across the run. Before touching anything, the loss was split by *where* it
happens on the only path the product has -- retrieve, rank, read -- using the gold evidence IDs the
three labelled tasks carry.

## Where the points go

For each labelled question the grounded window (the 12 records the reader saw) was compared with
the release's gold evidence IDs, and the 36-record candidate list with the same.

| Task (labelled n) | All gold in window | Some gold in window | Gold only in ranks 13–36 | Gold not in 36 |
| --- | --- | --- | --- | --- |
| LoCoMo-Refined (1,377) | 78.5 % — score 0.831, refused 4.6 % | 12.4 % — 0.404, refused 22.8 % | 5.2 % — 0.099, refused 54.9 % | 3.9 % — 0.259, refused 50.0 % |
| LongMemEval-S (479) | 84.3 % — 0.842, refused 5.9 % | 12.1 % — 0.310, refused 32.8 % | 1.9 % — 0.111, refused 55.6 % | 1.7 % — 0.125, refused 87.5 % |
| ATM-Bench main (1,013) | 79.2 % — 0.681, refused 14.1 % | 7.2 % — 0.306, refused 20.5 % | 4.4 % — 0.111, refused 46.7 % | 9.2 % — 0.065, refused 58.1 % |

Two readings follow directly.

1. **The reader loses more than retrieval does.** With every gold record in the window, the reader
   still scores 0.83 / 0.84 / 0.68: on LoCoMo that is 133 questions wrong and 50 refused out of
   1,081 -- 13.3 % of the whole task -- against 21.5 % of questions with any retrieval shortfall,
   most of which score above zero anyway. ATM-main refuses 14 % of questions whose gold media is
   in the prompt.
2. **"Refused with gold present" is mostly a coverage problem in disguise.** 89 LoCoMo refusals had
   at least one gold turn in the window, but only 50 had all of them. A quarter of LoCoMo questions
   and 62 % of LongMemEval questions cite more than one gold record; the ranking finds the turn
   that shares words with the question and misses its partner (Caroline asks "how long have you
   been into art?" -- retrieved; Melanie's reply "Seven years now" -- not retrieved). Refusing there
   is correct behaviour on the evidence the reader had.

### What the refusals are

Replaying the 124 LoCoMo/LongMemEval questions that were refused with gold in the window, with the
baseline prompt, reproduced 110 refusals (88.7 %), so the reconstruction is faithful. Of the
refusals with *all* gold present, the questions fall into: counting over several turns ("how many
times has Melanie gone to the beach"), dates that exist only in `occurred_at` ("last night" on a
turn dated 14 August), and dataset noise (a question dated December 2023 whose turn is December
2022). None is a missing-evidence case; all are reader inference.

### What the wrong answers are

A sample of the 133 LoCoMo and 48 LongMemEval answers that were wrong with all gold present
splits into three kinds: over-inclusive lists ("hard work, determination, dedication, motivation
and working together" for a gold of "hard work and determination" -- LoCoMo-Refined's judge marks
extra distinct facts WRONG), sequence errors (stale value on a knowledge update, wrong "most
recent", wrong duration between two dated turns, wrong event order), and calendar arithmetic. The
second kind dominates LongMemEval, where 69 % of questions are temporal, knowledge-update or
multi-session.

## Levers, measured by replay before any code changed

Each row replays fixed subsets of the baseline's own questions through the same reader and judge.
`A` = refused with gold in window (124), `B` = LongMemEval abstention-class questions (30, the
guard: refusing is correct), `C` = answered correctly with all gold (80–100), `D` = answered wrong
with all gold (180), `E` = gold missing from the 12-window but present in the 36 candidates or in
an adjacent turn (160).

| Lever | A | B (guard) | C (guard) | D | E |
| --- | --- | --- | --- | --- | --- |
| Baseline prompt, rank order, 12 rows | 0.032, refused 88.7 % | 0.867, refused 86.7 % | 0.963 | 0.028 | 0.331 |
| Reworded abstention clause ("refuse only when no hit bears on the question") | 0.177, refused 62.1 % | 0.800, refused 80.0 % | 0.975 | — | — |
| Answer-precision sentence ("give the specific fact … do not add related facts") | 0.008, refused 100 % | 0.900 | 0.838, refused 15 % | 0.122, refused 46 % | — |
| Chronological evidence order, same prompt | 0.065, refused 83.9 % | 0.833 | 0.963 | **0.233**, refused 6.1 % | — |
| Precision sentence + chronological | 0.016 | 0.967 | 0.637, refused 34 % | 0.133, refused 54 % | — |
| 24 rows by rank | — | — | 0.960 | — | 0.525 |
| 12 rows + the turn before and after each | — | — | 0.970 | — | 0.431 |
| 36 rows by rank | — | — | **0.990** | — | **0.613** |

What this says:

- **Any sentence that constrains the answer's form makes this reader refuse.** The precision
  sentence turned 46 % of previously answered questions into refusals and cost 0.13 on the
  correct set; the 2026-09-10 round measured the same for a "shortest complete answer" sentence.
  The abstention marker is an escape hatch whose use tracks the reader's uncertainty about
  satisfying the prompt, not the sufficiency of the evidence. Prompt wording is therefore not a
  lever for calibration, in either direction: the reworded clause converted 33 refusals into
  18 correct and 15 wrong answers and lost two of the thirty guard questions.
- **Order is information.** Handing the same 12 records to the reader oldest first, with nothing
  else changed, recovered a fifth of the answers that were wrong with all gold present and left
  the correct set unchanged; the guard moved by one question.
- **The window is too small for this reader, and count is the wrong unit.** 36 short dialogue turns
  scored *higher* than 12 on the questions it already got right and nearly doubled the score on
  the questions whose gold sat at ranks 13–36. A LoCoMo turn is ~210 characters; a LongMemEval turn
  ~1,000; a MemLens assistant turn ~1,100. A fixed `limit` of 12 records is 2.5 k characters on one
  corpus and 13 k on another. The product already has the right unit, `evidence_budget_chars`,
  which keeps the `limit` hits and admits more ranked evidence while it fits; the baseline left it
  unset.
- **Adjacent turns are not the fix.** Adding the neighbouring turn on each side (12 → ~31 rows)
  helped less than the same number of rows taken from the ranking.

## Harness confirmation

Dev slices, each arm a separate `mindbridge-bench eval --config` run with its own response cache,
paired per question against a same-code control run in the same session (`ctrl`). LoCoMo dev is
conversations 1–3 (346 questions), the holdout conversations 4–7 (592); LongMemEval is the first
120 units (70 single-session-user, 50 multi-session); MemLens-32k the first 50 units. W/L/T counts
questions whose score rose, fell, or held. Two same-code control replicates on LoCoMo dev differed by
+0.015 (7 / 2 / 337), which is the noise floor for these rows.

| Arm | LoCoMo dev (346) | LoCoMo holdout (592) | LongMemEval (120) | MemLens-32k (50) |
| --- | --- | --- | --- | --- |
| control | 0.7514, refused 27 | 0.6909, refused 70 | 0.7417, refused 15 | 0.3600, refused 19 |
| chronological order, 12 rows | −0.015 [−0.037, +0.014], 16/21/309, refused 33 | — | −0.042 [−0.083, 0.000], 1/6/113 | −0.080 [−0.16, −0.02], 0/4/46 |
| `evidence_budget_chars` 8 000 (~38 / 12.5 / 12 rows) | +0.055 [+0.014, +0.066], 26/7/313, refused 13 | +0.063 [+0.057, +0.069], 53/16/523, refused 32 | +0.025 [0.000, +0.058], 3/0/117 | −0.020 [−0.08, +0.04], 1/2/47 |
| `evidence_budget_chars` 24 000 (~100 / 26 / 17 rows) | **+0.090 [+0.042, +0.110]**, 41/10/295, refused 5 | **+0.076 [+0.054, +0.098]**, 61/16/515, refused 24 | **+0.067 [+0.017, +0.117]**, 9/1/110, refused 12 | −0.020 [−0.10, +0.06], 2/3/45 |
| chronological + 8 000 | +0.055, 34/15/297 | — | −0.033 [−0.075, +0.008], 2/6/112 | −0.100, 0/5/45 |

Cost on LoCoMo dev: answer tokens per question 3,066 (control) → 6,475 (8 000) → 14,649 (24 000);
mean `ask` latency 362 ms → 457 ms → 743 ms. LongMemEval's per-question tokens are dominated by
ingest embedding and did not move measurably.

What held and what did not:

- **Chronological order is rejected.** The replay gain on the wrong-with-gold subset was real, but
  in the harness it raised refusals on LoCoMo (27 → 33), lost 0.04 on LongMemEval multi-session
  questions and 0.08 on MemLens, and on knowledge-update questions it made the reader pick the
  *older* value ("woven carry-all" for a gold of "straw tote" that rank order had answered). The
  reader anchors on the first record it is shown; with rank order that is the best match, with
  time order it is the oldest. The code change was reverted; only the diagnosis stays.
- **The evidence budget is adopted, at 24 000 characters.** It is positive with the interval
  excluding zero on LoCoMo dev, LoCoMo holdout and LongMemEval, neutral on MemLens, and it cuts
  refusals wherever it adds rows. 8 000 bought most of the LoCoMo gain at half the tokens but is a
  no-op on corpora whose dozen records already cost more than that, which is exactly where a record
  count misleads most; a caller who wants the cheaper window lowers the budget. The earlier note
  in `plugins.py` that a budget was a measured null (limit 20 → 56 rows, +0.003) was measured on a
  different reader and from a window already twice this baseline's.
- **Not reward hacking, and why.** Nothing about scoring, prompts, refusal wording or task
  protocol changed. The reader is handed more of the evidence the store already ranked, and the
  abstention-class questions kept their refusals (LongMemEval abstention questions were not in the
  slice; the replay guard set moved by at most one question under every evidence-window lever).
  The same change costs tokens and latency, which the note above states rather than hides.
- **Not measured here.** ATM-Bench and Mem-Gallery were not re-run. On a media corpus the budget
  admits up to twelve more images or two more videos per question at the text-equivalent charges,
  and each grounded media row pays face and speech recognition before the answer plus the reader's
  media tokens; the next full baseline is where that cost and any gain get measured.

## Other findings from the run

- The `product.settings` block of the effective-config artifact (`config.yaml`) reports
  `reinforce_on_answer: true`, but the harness pins it to `false` for every product arm (the
  stores' `access_count` columns are all zero). The artifact echoes the parsed file, not the
  settings the run used.
- PersonaMem-v3 took 25,349 s -- 58 % of the 44,090 s run -- for 15,298 questions of which 3,355
  carry no score in `samples.jsonl` and whose families are personalization tasks (chatbot
  response, agentic, proactive) that always expect a reply: `proactive_actions` refused 83.8 %,
  `agentic` 42.3 %, `chatbot_response` 46.0 %. `ask()` is a closed-book reader over memories; the
  benchmark measures an assistant that combines memory with general knowledge, which is what
  `compile()` exists to feed. The same mismatch shows on LongMemEval `single-session-preference`
  (0.333, refused 30 %).
- MemLens carries `answer_session_ids` in the release JSON that the adapter does not surface, so
  its retrieval quality is reported as unmeasurable although a session-level gold exists.
- ATM-Bench-Hard is 31 questions; one question is 3.2 points and its retrieval recall@10 is 0.35,
  so it is a retrieval problem (recall program work from the previous round), not a reader one.
