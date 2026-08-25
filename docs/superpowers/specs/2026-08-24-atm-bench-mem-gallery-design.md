# ATM-Bench and Mem-Gallery benchmark support

> Status: approved design, not yet implemented
> Date: 2026-08-24
> Origin: request to support <https://github.com/JingbiaoMei/ATM-Bench> and
> <https://github.com/YuanchenBei/Mem-Gallery>

## Problem

MindBridge replays nine official long-memory benchmarks through thin adapters. Neither of
the two newest multimodal memory benchmarks is among them, and both test capabilities the
current nine do not:

- **ATM-Bench** is a four-year single-person archive of images, videos, and email, queried
  by referential questions ("the steel bridge in Porto"). No existing adapter covers a
  multi-source personal archive, and none covers email at all.
- **Mem-Gallery** is twenty multi-session multimodal dialogues where 487 of 1,711 questions
  carry a query image. No existing adapter exercises image-as-query recall, and none
  exercises abstention, conflict detection, and knowledge-update resolution as scored task
  types.

## What the two releases actually contain

Counted from the pinned releases, not from either README.

### ATM-Bench — `Jingbiao/ATM-Bench` @ `78e826dc07e97466b2f54443831ef9a83ab8b27c`

| Item | Value |
| --- | --- |
| Total release | 3.33 GB, 8,597 files |
| `data/atm-bench/atm-bench.json` | 1,013 questions: 360 `number`, 514 `open_end`, 139 `list_recall` |
| `data/atm-bench/atm-bench-hard.json` | 31 questions: 12 `list_recall`, 13 `open_end`, 6 `number` |
| `data/atm-bench/niah/*.json` | fixed evidence pools over the hard split at k=25/50/100/200 |
| `data/raw_memory/image/` | 3,759 `.jpg`, 1.2 GB |
| `data/raw_memory/video/` | 533 `.mp4`, 1.9 GB |
| `data/raw_memory/email/emails.json` | 6,742 emails, 5.0 MB |
| `data/processed_memory/{image,video}_batch_results.json` | 29 MB, the official SGM text, covering all 4,292 media items |

Downloaded and checked: every evidence ID in both QA splits resolves — 1,140 image, 81
video, 430 email references, zero unresolved. Image and video stems do not collide, so one
stem names exactly one file. Those 430 references resolve to 362 distinct emails — 354 in the main split, 13 in the
hard one — so 5.4% of the 6,742 emails are ever cited and the rest are the archive's
distractor mass, ingested all the same. Media spans 2022-04-30 to 2025-06-25 and
email 2022-01-01 to 2025-07-22 — closer to three and a half years than the advertised four.

Question schema: `id`, `question`, `answer`, `notes`, `evidence_ids`, `qtype`. NIAH pool
files add `niah_evidence_ids`. Evidence IDs are either a media filename stem
(`20250223_130249`, 1,046 references) or an email ID (`email202411160004`, 411
references); a question cites 1 to 10 of them, median 1.

Two facts worth pinning because both are easy to assume wrongly:

- **The hard split is disjoint from the main split.** Intersection of the two ID sets is 0.
  Total question count across both files is 1,044, not 1,013.
- **`notes` is empty in every one of the 1,013 main-split rows, but not the hard split's.** 10 of
  the hard split's 31 rows carry substantive annotator notes explaining why the question is hard
  (confounding evidence, a renamed hotel, an arithmetic breakdown); the field is not universally
  unused, only unused where most questions live.

### Mem-Gallery — `Ethan-Bei/Mem-Gallery` @ `af912daba984e896e253016b7c7e334ef92c2a6f`

| Item | Value |
| --- | --- |
| Total release | 0.55 GB, 1,515 files |
| `data/dialog/*.json` | 20 topics, one persona each |
| Sessions / rounds | 240 sessions, 3,962 rounds |
| Images in dialogue | 1,003 |
| Questions | 1,711, of which 487 carry a `question_image` |

Per-topic file: `character_profile` (name, persona summary, traits, conversation style),
`multi_session_dialogues` (`session_id` `D1`…, `date`, `dialogues` of
`round`/`user`/`assistant` with optional `image_id`/`input_image`/`image_caption`), and
`human-annotated QAs` (`point`, `question`, `answer`, `session_id`, `clue`, optional
`question_image`/`image_caption`).

Nine `point` types, counts matching the paper exactly: TTL 337, VS 306, FR 219, MR 206,
AR 184, VR 174, TR 123, CD 81, KR 81. `clue` is a list of round IDs (`D8:2`) — the
official retrieval ground truth.

## What MindBridge already has

- `RecallQuery.media_object_ids` accepts up to eight stored media objects as the query
  itself, so Mem-Gallery's 487 image-carrying questions need no new capability.
- `MediaObjectInput.media_object_id` is caller-assigned and comes back on every
  `EvidenceView` a recall returns. Setting it to the official ID — an ATM media stem, a
  Mem-Gallery `image_id` — makes the mapping from returned evidence to dataset evidence
  free, with no side table to keep consistent.
- `ObserveRequest` requires at least one media object, so a text-only unit is a `remember`
  write by construction, not by choice.
- `RecallMode.ENUMERATE` scans a complete filter scope rather than truncating, which is
  what ATM `list_recall` and Mem-Gallery `VS` ask for.
- `official_score` reads the benchmark name out of the run manifest, so recording an
  official scorer's verdict beside a new benchmark's predictions needs no change to it.
- Both releases are plain JSON, so neither adapter needs an optional extra. Only
  Video-MME's parquet does.

## Success criteria

1. `mindbridge-bench datasets` parses both releases and pins their digests, reproducing the
   official counts above (1,013 / 31 / 1,711) as a regression gate.
2. `mindbridge-bench atm` and `mindbridge-bench mem-gallery` each produce official-shaped
   predictions plus a manifest that fixes source, deployment, prompt, and adapter identity.
3. One subset run of each completes end to end against the local deployment, and the run
   artifact proves the write path executed — non-zero ingested evidence, non-zero recalled
   memory IDs. A run that answers from an empty store is a failed run, not a low score.
4. ATM runs both media arms at least once, so a later score difference can be attributed.
5. All five quality gates pass, plus the pinned Markdown and link checks.

### What was actually demonstrated

Criteria 1, 2 and 5 are met. Criteria 3 and 4 are **not**, and this section records that
rather than letting the list above read as a report of success.

- **Met for the ATM SGM arm.** One subset run completed end to end against a live
  deployment: 4,292 schema-guided blocks and 6,742 emails written with zero ingest
  failures, and all 19 questions in the run returned twenty non-empty memory IDs.
- **Not met for the ATM raw arm.** It ingested real media through perception — six items
  succeeded, verified in the store — but produced no predictions file, so criterion 4's
  "both arms at least once" is unmet and no arm-to-arm comparison exists yet.
- **Not met for Mem-Gallery.** No run has completed. Twenty of its 89 staged images were
  ingested, and 128 memory records landed, but the recall path never ran to completion.
  **Image-as-query recall — the most novel capability this work adds, and the reason
  `RecallQuery.media_object_ids` is cited above — has therefore never executed end to end.**

Both shortfalls were environmental rather than defects in this code: the shared worker fleet
the runs depended on was torn down and replaced mid-run, and the retry afterwards saturated a
single-process embedding service. The staging, tenants and commands are all in place, so
completing them is a matter of machine time rather than more work here.

The runs that did happen were worth more than their artifacts: they surfaced four defects
that no unit test could have — a recall mode that refuses at archive scale and took the whole
run down with it, a retrieval diagnostic dead across 37.6% of one split, artifacts unable to
express a partial ingest failure, and a staging contract the release makes unsatisfiable for
any multi-topic run. All four are fixed.

## Approved decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Module layout | `atm_bench.py` + `atm_bench_runner.py` + `atm_cli.py`; `mem_gallery.py` + `mem_gallery_runner.py` + `mem_gallery_cli.py` | Exactly the trio every other benchmark uses; `m3_bench.py` + `m3_cli.py` is the naming precedent |
| CLI names | `atm`, `mem-gallery` | Short, matches `m3` / `mm-lifelong` |
| Answer-quality metrics | None computed in the runner | Both benchmarks judge with an LLM under their own normalizers; a second implementation would diverge silently and is exactly the reward-hacking surface `benchmarks-sota.md` §5 forbids |
| Retrieval metrics | Computed in the runner | Recall@k over `evidence_ids` / `clue` needs MindBridge's internal retrieved IDs, which the official scorers cannot see |
| ATM media arm | `--media-source {raw,sgm}`, default `raw` | Mirrors ATM's own flag; separates a perception deficit from a retrieval deficit |
| ATM tenancy | One tenant per run, ingested once, answering all questions | The archive is one person's four years; per-question isolation would misrepresent the benchmark |
| Mem-Gallery tenancy | One tenant per topic, 20 tenants per full run | Twenty independent personas; a shared store would leak one persona's memory into another's questions |
| Mem-Gallery ingest granularity | One observation per round, keyed by its `round` ID | `clue` recall is only computable if a returned memory maps back to a round ID |
| Media staging | Operator step outside the wheel, consumed as a prepared-media manifest | `MediaObjectInput` takes a URI the store already holds and the SDK has no upload call; `benchmarks/` may only use the public SDK, so it cannot reach the S3 adapter, and adding boto3 to the `benchmarks` extra to re-implement `mc mirror` is not worth it |
| Official query wordings | Reproduced in `benchmarks/prompts.py` with versions and fingerprints | Where every other benchmark's official wording lives; the product answer policy must not see them |

### Why the runner emits predictions instead of scores

ATM scores `number` by exact match after its own date/time/number normalization,
`list_recall` by Jaccard, `open_end` by a `gpt-5-mini` judge. Mem-Gallery scores EM, token
F1, BLEU, BERTScore, and a five-level judge, all under a normalizer that deliberately
protects `IMG_001`-style underscores. Reimplementing either normalizer would produce
numbers that look official and are not. The runner therefore writes predictions the
official scorers accept, and `mindbridge-bench score` records their verdict beside the run
with the scorer command and digest. This also keeps two whole judge implementations out of
the diff.

### Why ATM keeps both media arms

ATM's own finding is that SGM beats raw once distractors fill the haystack — every "w/o
SGM" agent run lands far below its SGM run. But MindBridge's `raw` arm is not ATM's raw
arm: ATM's raw arm stuffs images into the answerer's context, while MindBridge's raw arm
sends media through its own perception and stores structured memory. MindBridge's raw arm
is therefore closer to *building its own SGM* than to ATM's raw baseline, and the
comparison to write down is raw-arm MindBridge against the SGM column. The `sgm` arm eats
the official `batch_results` text directly, which isolates retrieval from perception: if
raw scores below sgm, perception is the deficit; if both score alike, retrieval is.

## Architecture

Six new modules, five touched registries, four new test modules.

```text
src/mindbridge/benchmarks/
  atm_bench.py            contracts + loaders: questions, NIAH pools, emails, SGM records
  atm_bench_runner.py     ingest archive once, answer each question, retrieval diagnostics
  atm_cli.py              argument surface, manifest, artifacts
  mem_gallery.py          contracts + loader for one topic file, and for a directory of them
  mem_gallery_runner.py   per-topic ingest, per-question answer incl. image-as-query
  mem_gallery_cli.py      argument surface, manifest, artifacts
  cli.py                  + two RUNNERS rows
  dataset_smoke.py        + three summaries, min_length 13 -> 16, three new arguments
  prompts.py              + official wordings for both benchmarks
docs/benchmarking.md      + download, staging, and smoke commands
docs/benchmarks-sota.md   + sections 3.10 and 3.11 with the published baselines
tests/unit/benchmarks/    test_atm_runner.py, test_atm_cli.py,
                          test_mem_gallery_runner.py, test_mem_gallery_cli.py
tests/unit/benchmarks/test_dataset_adapters.py    + adapter schema cases
tests/contracts/test_prompt_catalog.py            + new fingerprints
```

### ATM-Bench contracts

`AtmBenchQuestion(question_id, question, reference_answer, qtype, evidence_ids,
niah_evidence_ids=())` where `qtype` is `Literal["number", "list_recall", "open_end"]`.
`load_atm_bench(path)` refuses an empty release, a duplicate ID, and a question with no
evidence. `load_atm_niah_pool(path)` additionally refuses a pool that does not contain
every `evidence_ids` entry, which is the property the pool exists to have.

`AtmEmail(email_id, occurred_at, summary, body)` over the release's `id`, `timestamp`,
`short_summary`, `detail`.

`AtmSgmRecord(media_id, media_kind, occurred_at, location_name, city, caption,
short_caption, ocr_text, tags, entities)` — the subset of the release's 17 image and 25
video fields that the published SGM schema names, plus `city` because the geocoded place is
what a referential question ("my visit to Porto") actually keys on. `entities` is present on
3,757 of 3,759 images but only 190 of 533 videos, and `device` is empty on every video, so
both are optional in the contract.

### One clock, and why

The release carries two clocks, measured across all 4,292 items:

- An image's `timestamp` is naive and agrees with its filename stem to within five seconds.
- A video's `timestamp` is timezone-aware and true UTC: its offset from the stem is exactly
  0 for January, February, March, November and December captures and −60 minutes for April
  through October — the UK's GMT/BST boundary — with −120 minute and +60 minute clusters
  where the owner was travelling. The stem is the camera's local wall clock at the place of
  capture.
- Every one of the 6,742 email timestamps is naive.

The adapter therefore takes **the filename stem as the capture time for all media, read as
UTC**, and reads email timestamps the same way. Every memory then sits on one self-consistent
local wall clock, which is also the clock the questions are phrased in. Two alternatives are
rejected: using the video `timestamp` field would put video an hour off image for half the
year, and inferring a per-item zone from the record's GPS would add a timezone layer to a
benchmark adapter. The consequence to keep in mind is that stored times are local, not UTC,
so this archive must not be joined against an external UTC timeline.

The same rule keeps the two media arms comparable: the `raw` arm's capture time comes from
the prepared manifest and the `sgm` arm's from the record, and both must be the stem or the
arms would differ by an hour for reasons that have nothing to do with memory quality.

Evidence-kind classification lives in the adapter as one function over an ID string, so
runner and metrics cannot disagree about whether `email202411160004` is email.

### Mem-Gallery contracts

`MemGalleryTopic(topic, profile, sessions, questions)` with
`MemGallerySession(session_id, occurred_on, rounds)`,
`MemGalleryRound(round_id, user, assistant, image_id=None, image_path=None,
image_caption=None)`, and `MemGalleryQuestion(question_id, point, question,
reference_answer, session_ids, clue_round_ids, question_image_path=None,
question_image_caption=None)`.

`point` is a `Literal` over the nine official codes, so an unknown code fails the load
rather than being silently scored as an unknown category. Question IDs are not in the
release, so the adapter derives `<topic>:<index>` from release order and pins it — the same
approach `mm_lifelong` takes with question indices.

`load_mem_gallery_topic(path)` loads one file; `load_mem_gallery(directory)` loads all
twenty in sorted filename order.

## Data flow

### ATM-Bench

1. Operator stages `data/raw_memory/{image,video}` into the object store once and writes a
   prepared-media manifest mapping each media stem to its URI, digest, size, and capture
   time. `atm_cli` validates that every stem cited by the selected questions is present.
2. `raw` arm: each image and video is observed through `ingest_media`, one media object per
   observation, with `media_object_id` set to the dataset stem — so MindBridge's own
   perception writes the memory and returned evidence names its source stem directly.
   Batching eight images into one observation is allowed by the contract and rejected here:
   an archive's images minutes or months apart would share one `occurred_at`-to-`ended_at`
   span and be described as one scene.
   `sgm` arm: each `batch_results` record is written with `remember` as text, carrying the
   official schema fields.
3. Emails are always written with `remember`, in both arms — there is no email media. The
   runner keeps the email-ID-to-memory-ID map itself; `remember` has no external-ID field.
4. Each question is answered with `recall`, `ENUMERATE` for `list_recall` and `ANSWER`
   otherwise. The result records the prediction, the recalled memory IDs, the evidence
   media object IDs, retrieval recall against `evidence_ids`, and the ingest failure count.

### Mem-Gallery

1. Operator stages `data/image/<topic>` once, with `media_object_id` set to the official
   `image_id` (`D2:IMG_003`) — the exact token a `VS` answer has to name. That ID is unique
   per topic, not per release: `D1:IMG_001` alone names a different picture in all twenty
   topics, so the staging contract requires uniqueness only within one topic's images.
   Question images are staged the same way, keyed by their relative path.
2. Per topic: one tenant, sessions in release order. A round with no image is one
   `remember` write; a round with an image is one `ingest_media` observation carrying the
   round's text, so the image and the words that surround it land in the same memory. The
   runner keeps the round-ID-to-memory-ID map, which is what makes `clue` recall
   computable.
3. Per question: the official wording for its `point` is applied, the query carries the
   staged question image when the row has one, and the answer is recorded with recalled
   memory IDs mapped back to round IDs for `clue` recall.

## Error handling

- A missing prepared-media entry, an unknown `point`, an unparseable timestamp, or a NIAH
  pool that omits a gold evidence ID fails the run before any request is sent. A benchmark
  run that starts and dies halfway costs hours; every check that can be static is static.
- An ingest failure is counted per unit and carried into the result, never swallowed. A run
  with a non-zero failure count is still a run — the count is what makes it interpretable.
- `asyncio.gather` calls pass `return_exceptions=True` and reduce the outcomes explicitly.
  A bare gather has already discarded a page of paid work once in this repository.

## Cost

Both corpora are downloaded at their pinned revisions: 3.2 GB for ATM-Bench, 530 MB for
Mem-Gallery. Staging is a one-off upload of those bytes.

A full ATM run's `raw` arm is 4,292 media observations plus 6,742 email `remember` calls. The
`sgm` arm is not cheaper by call count just because it skips perception: writing every
`batch_results` record as text is 6,042 `remember` calls, not 4,292, because 1,063 of the 4,292
blocks exceed the 2,048-character summary limit and split into multiple chunks (one into as many
as 6) — worth knowing before assuming the `sgm` arm is cheap. The same 6,742 email writes happen
in both arms, and email blocks never chunk (longest measured at 1,355 characters). Neither full
run is planned as part of this change; each subset run targets under two hours of wall clock:

- ATM subset: the questions selected plus the media they cite, both arms.
- Mem-Gallery subset: one topic — the worked example's `Baking_Dessert_Daily_Life_Skill`, 15
  sessions, 262 rounds, 57 images.

Full runs are follow-up work with their own budget: the raw arm's 533 videos are the
expensive item, and prior measurements in this repository put video perception in the tens
of minutes per clip when the generator is slow.

## Testing

- Adapter schema tests with inline fixtures in `test_dataset_adapters.py`, one per refusal
  the loader promises.
- Runner tests against the existing fake-deployment helper, asserting tenancy, ingest
  ordering, the media-arm switch, image-as-query recall, and the retrieval-recall
  arithmetic.
- Every new test is mutation-checked: break the implementation, confirm the test goes red.
  A test whose assertion never depended on the code under test has passed review here
  before.
- `mindbridge-bench datasets` against the pinned releases is the count regression gate;
  the official counts are asserted there, not in unit tests, because they are properties of
  the release rather than of the code.
- Prompt fingerprints for the new official wordings are added to the contract test.

## Out of scope

- Any implementation of either official judge, and any answer-quality number produced
  outside the official scorers.
- ATM's NIAH sweep as a run mode. The pools are loaded and validated so a later change can
  use them, but a fixed-evidence-pool run measures an answerer, not a memory system.
- Full-corpus runs and leaderboard submissions.
- A staging command inside the wheel. If three benchmarks later need one, that is when it
  earns its dependency.
