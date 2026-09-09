# Historical out-of-scope EgoLife diagnostics

Status: historical appendix; excluded from MindBridge product acceptance and future optimization.

The fixed Ego100 raw-causal comparison and later read-only diagnostics below were executed during
the September 8 research. They were distinct from a pre-existing EgoLife v5-speech job that the
study left undisturbed and whose state and outcome are not reported here. The project owner later
removed EgoLifeQA from scope because its objective does not match MindBridge's intended use. The
negative results, costs, and failed hypotheses remain recorded to avoid hiding unfavorable evidence.
They are not release gates, optimization targets, or evidence that the implemented repair
generalized to product video memory.

## Historical causal result

After these diagnostics were completed, the project owner removed EgoLifeQA from the product
acceptance scope because its objective does not match MindBridge's intended use. The artifacts and
negative result remain here as a failure-analysis appendix: they still expose useful problems in
causal evaluation, raw-video representation, and evidence budgeting. They are not a release gate,
an optimization target, or evidence that the implemented repair generalized to the product's video
workload. No further EgoLife provider call or evaluation was made after that scope correction.

The fixed 100-question exposed EgoLife characterization is a negative result. The initial tree
answers 20 questions correctly and score completion answers 16, for means of 0.200 and 0.160. Nine
questions improve, 13 decline, and 78 are unchanged; the paired difference is -0.040 and the
descriptive question-bootstrap interval is [-0.130, 0.050]. Each arm has one failed generation
attempt whose usage is unavailable and 99 completed, usage-reported calls. Their known generation
totals are 1,745,424 and 1,745,631 tokens, respectively. EgoLife uses deterministic scoring and no
judge calls.

This is not evidence of a retrieval regression. Among the 37 questions whose ordered grounded
sources change, each arm answers nine correctly, with six improvements and six declines. Among the
63 with unchanged ordered grounding, the observed result moves from 11 correct to seven, accounting
for the entire net difference. The strata are descriptive and do not identify a causal reader
effect because the live endpoint and reader-request controls below still vary.

The paired structural checks pass: both arms use the same 100 questions, raw source corpus, causal
cutoffs, configuration, and exact cached query-embedding request multiset with zero misses. Ordered
top-12 source IDs change on 43 questions and ordered grounded source IDs change on 37. The recorded
generation-request hash multisets differ, but the proxy did not retain request bodies or a durable
question-to-hash binding. A later local reconstruction of the first roster entry with equal
grounding found that the two 13,891,135-byte requests differ only in an automatically inserted
wall-clock reference time; replacing that line makes the bodies byte-identical, including all eight
video data URLs. This proves a sufficient mechanism for hash differences in the reconstruction, not
the contents of unavailable historical bodies. Combined with a live generation endpoint and the
full-index-then-delete limitation, the historical evidence cannot attribute the answer difference
to score completion. The result still rejects any claim that the ATM gain already generalizes to
raw video memory.

The fixed-denominator aggregate is recorded in
`.benchmarks/research/2026-09-08-evaluation-audit/paired-qa-v1/egolife100-causal-fixed-planned-denominator-quality-v3.json`
(SHA-256 `8dffdbf0821b8309a15660877fb61bda25c0543af3d3175f45315dc1dc83d580`).

A separate read-only store audit gives one reason score completion may transfer poorly to this raw
video setting. In the first three source-order records, durable text is only a benchmark
`[source_id: ...]` marker; the video asset has a generic name and no transcript, speech segment, or
visual description. Zvec indexes lexical content on aggregate part zero, so the marker is the only
lexical evidence for those sampled records. This verifies weak route input in a narrow sample; it
does not establish that a question matched the marker or that this caused the negative result.
MindBridge already has optional ingest-time visual description; the matched configuration leaves
it disabled and also disables speech indexing. Enabling it adds one logical visual-description
request per write batch with pending visual assets; the SDK may make additional provider attempts
when a response has an invalid format. It is therefore a separate formation-and-cost experiment
whose outputs must be frozen into one attested corpus before comparing retrieval arms. The
audit is `.benchmarks/research/2026-09-08-evaluation-audit/ego-raw-lexical-semantics-v1/README.md`
(SHA-256 `da329dac9ae5aab6794adc7fbf52a13e0b5ee4b398d315c9148599225e384bc6`).
The request reconstruction is recorded in
`.benchmarks/research/2026-09-08-evaluation-audit/ego-request-body-diagnostic-v1/receipt.json`
(SHA-256 `faf2ea1e024afbadd92277c463b7ccd1cd1baed18cc7159cd4a0559d211cb834`).
The reconstruction established a general evaluation rule: paired runs should freeze the query's
evaluation reference time when a task does not supply one. The benchmark CLI now provides
`--fallback-reference-at` for that case, filling only missing question clocks while preserving
dataset and corpus dates. This is a reader-request control; it must not be written into observation
`occurred_at` or knowledge-visibility fields. It does not authorize another EgoLife evaluation.

## Historical evaluated alternatives

An exposed, fixed source-order 40-question EgoLife diagnostic separates temporal coverage from
answer correctness. Of 29 failures, 11 have no target slice in the recorded retrieval pool, one has
only one of two target slices, six contain the targets in the pool but not the top 12, and 11 expose
all targets in the top 12 but still answer incorrectly. Under the eight-video input cap, nine failures
still expose every target slice and fail. A radius-two neighbor oracle raises the target-coverage
ceiling only to 18/40 and charges neither displaced evidence nor reader errors. This supports a
historical risk review, not adding `TimelineSpan` now or authorizing another EgoLife experiment.
The required sequence, relative-cutoff, provenance, and fixed-budget gates are recorded in the
[timeline neighbor-window risk review](timeline-neighbor-window-risk-review-2026-09-08.md). The
diagnostic artifact is
`.benchmarks/research/2026-09-08-evaluation-audit/ego-exposed-temporal-diagnostic-v1/README.md`
(SHA-256 `dd23a5af8bd5997925a44bf71c8191e97ba2adbd9f4dd961ece62e2354e4dc5b`).

A subsequent fixed-budget facility-location selector was also rejected. It keeps the top-ranked
anchor and greedily uses public `SearchHit.score` plus persisted video-part cosine coverage to pick
eight clips from each actual 36-hit public pool. Over the fixed exposed Ego100 roster, mean temporal
target-slice recall falls from 0.1183 to 0.0817 and complete coverage falls from 11 to seven
questions; three rows improve, seven decline, and 90 are unchanged. The fixed decision rule required
both metrics to improve. No generation provider was called and the selector was not integrated.
This is an oracle temporal-overlap diagnostic over a bounded, non-exhaustive pool; it does not
measure semantic sufficiency or answer quality. Its offline vector projection is specific to the
verified one-video raw records and does not generalize to multi-asset memories without durable part
provenance. The aggregate receipt is
`.benchmarks/research/2026-09-08-evaluation-audit/video-evidence-coverage-v1/ego100-coverage-diagnostic-v1/aggregate.json`
(SHA-256 `afca2ebac54181e5388d374fc09d1e02270b16bada170899c08c205f2504aa91`),
and the audit receipt is SHA-256
`f0cf8166152678aa02a7f959caad3e63d4b032cc803f505dda1c22f5f6c40e88`.

An exhaustive exact-cosine diagnostic then measures how much of the miss is bounded ANN candidate
generation. It uses the fixed Ego100 roster, the exact cached query vectors, and filters the 6,264
raw parents separately by each question's cutoff without a provider call. The public pool fully
covers 11 questions at eight and 19 at 36. Exact all-part MaxSim covers 13, 25, and 34 at ranks 8,
36, and 100; aggregate-part scoring covers 13, 31, and 47; video-only scoring covers 15, 31, and 45.
No row is selected after the result. This is real headroom, but exact scanning alone remains
insufficient: at 100, aggregate and video-only scoring still place 54 and 55 of 114 target slices
below the cutoff, respectively. The
result identifies bounded candidate-generation headroom while also showing the importance of
per-clip semantics and part provenance; it does not show that a video lacks the answer or that visual
captioning will fix it. The diagnostic is an offline, exposed temporal oracle rather than a
public-SDK quality result;
its source-ID tie rule is diagnostic-only. The aggregate is
`.benchmarks/research/2026-09-08-evaluation-audit/exact-cosine-visibility-v1/result/aggregate.json`
(SHA-256 `6e0bfe72eb0be439b414bc50773e80f6c627a998c493bfa1b94f4e524e1fa269`),
with receipt SHA-256 `bfb2ff1eb8a0ab295c1e961d4fa2e94d1db3360a306ca796b158bbce66c9a7b1`.
