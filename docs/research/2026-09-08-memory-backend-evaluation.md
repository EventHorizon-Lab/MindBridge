# Memory backend evaluation — 2026-09-08

This report records the frozen diagnostic evaluation of the evidence-closed context compiler. The
For the original compiler-v3 phase, the product baseline is the exact initial dirty-tree snapshot in
`.benchmarks/research/2026-09-08-e2e-baseline-current/extracted/src`; its archive SHA-256 is
`34744596f185f6dc9280dea0b0f21a385a309c429f632f940e468bf6ed9ddff0`. That phase's candidate is
compiler final v3 in
`.benchmarks/research/2026-09-08-e2e-candidate-final-v3/extracted/src`; its archive SHA-256 is
`0c9abfc81cb292d89f2de162a1bc6a7fdc17bcd0292783a16f3346f9b060aa69`. Every subprocess put the
selected frozen source first on `PYTHONPATH` and recorded the imported module paths and hashes.

These runs are diagnostic subsets, not official leaderboard submissions or evidence of a general
ranking. Latency from overlapping recovery and evaluation is operational telemetry only. The
experiments changed no product configuration while a run was active.

The work began on 2026-09-08 and acceptance artifacts continued into 2026-09-09 in
Asia/Shanghai. Source-access dates in the research ledger remain the dates on which those sources
were actually read. The final compiler-v3 experiment and the later schema-17 writer experiment are
separately frozen phases; results from one are not relabelled as results from the other.

## Phase and result index

| Phase | Frozen product or current tree | Main result and boundary |
| --- | --- | --- |
| Compiler-v3 closed-store pair | Baseline archive `34744596…`; compiler candidate archive `0c9abfc8…` | LoCoMo declared-evidence closure improved from 2/138 to 138/138 bundles, while the common cached judge tied 97/138 to 97/138. This is a structural gain without demonstrated semantic gain. |
| Schema-17 fresh writer | Pre-witness snapshot versus clean archive `11655a14…` | Held-out LoCoMo conv-30 moved 52/72 to 54/72 with 5/7 discordance and no source-recall gain. One conversation is not independent replication. The retrospective Persona2 run exposed future sources on 106/149 questions and is not a causal score; its 2,930 clauses were all singleton. |
| Schema-18 excerpt experiment | Frozen archive `db8ccb82…`, default excerpt delivery off | Raw causal-prefix capture, 298 fresh answers and task-dependent scoring completed with zero future-source delivery. Results are mixed and one baseline judge row remains an explicit error; fresh formed writers are still running. Selector persistence and exact span containment do not prove semantic sufficiency. |
| Current main after the frozen experiments | Schema-18 plus the corrected Persona ranking scorer and post-v3 general-consolidation clause change | Current full validation is 1,876/1,876 tests on CPython 3.12.11. New general `CONSOLIDATE` operations treat one cited set as an AND clause and separate operations as OR alternatives. No large-model result in this report used that post-v3 consolidation change. |

No phase establishes a state-of-the-art result, a population-level semantic improvement, or a
faster, cheaper backend. The tables below are diagnostic evidence with the exact prompt, media,
time-firewall, and cost qualifications stated for each run.

## Controls and recovery

Dataset selection was fixed before scoring: release order, offset 0, seed 42, and the recorded
dataset and evaluation hashes determine each roster. One physical `data_dir` was used by one owner.
Paired replays locked a closed source, made a physical clone per arm, and verified the source did not
change. The paired driver captured the complete provider-ready HTTP request bytes, not just rendered
context. It also used one durable exact-request embedding cache: the baseline populated it and the
candidate received byte-identical successful embedding response bytes. No errors were cached.

The completed generic paired replays passed each question's `reference_at` and used the SDK's default
`scope=None`; they did not pin historical `valid_at` or `known_at`. Some original replay row metadata
incorrectly serialized both scope axes as `reference_at`, although the actual SDK calls supplied no
scope. The preserved provider bytes, observed results, and equality conclusions are unaffected; a
separate correction ledger records the provenance defect rather than rewriting those artifacts.
That ledger is
`.benchmarks/results/e2e-paired-replay-scope-provenance-correction-20260908/correction.json`
(SHA-256 `181c1064de150b123fa7eb365ef8d4da92ae1405c5b9336568d4833cf105b05a`).

The interrupted formed LoCoMo store was recovered through the public SDK rather than rebuilt. Its
closed state contains 419 distinct sources, 419 completed formation runs, 1,248 records, 1,248
embeddings, 837 evidence links, and empty capture and search-index queues. SQLite integrity passed,
the WAL was checkpointed, and an exclusive-owner check succeeded. The immutable archive is
`.benchmarks/research/2026-09-08-e2e-recovery-v1/closed-formed-419.tar.gz` with SHA-256
`aac4b8d23c5d872526ede3152604ebc8247ba382a19d84f980bd5f41fdd73b04`.

Two 64-source formation batches and the final 35-source batch exceeded the fixed 2,048-token output
cap. Raw observations had already committed, so the recovery driver journalled the failure and
settled the durable queue through the public SDK. Those queued rows formed one source at a time.
Initial `add_many` can expose several observations to one former request, but truncation and
bisection mean this run does not establish stable neighboring-turn context or one model call per 64
sources. It compares compiler behavior on the same recovered formed store; it is not an optimal
formation-cost measurement.

## LoCoMo scoring protocol

The first strict non-streaming judge attempt was preserved separately. Gateway failures motivated an
SSE transport collector, and repeated schema-invalid output motivated a separately labelled JSON
object variant. That uncached pass is retained as reliability history. A byte audit later found that
one identical formed request for q0089 received opposite labels, while another identical request for
q0005 produced one label and one schema error. Its apparent one-answer candidate edge is therefore
not a causal result.

A two-question feasibility probe then added the standard OpenAI strict JSON schema
`{label: enum(CORRECT, WRONG)}`. Both a normal case and previously malformed q0029 returned a valid
eight-token object. The primary pass used that schema and one durable response cache shared by all
six arms. Its key covers the exact serialized HTTP method, URL, body, and scorer protocol. Only the
first successfully parsed response is cached; failed attempts remain in the journal and can never be
selected by label. The rubric, parser, model (`Qwen3.8-27B`), temperature, roster, and deterministic
local-score rules remained fixed. SSE adds only `stream: true` and `include_usage: true`.

The scorer locally resolves deterministic empty-prediction and exact-reference cases when
`judge_plan` returns `None`. Version 4 repaired 23 such LightMem rows atomically without model calls;
it did not reinterpret model outputs. The frozen scorer SHA-256 is
`8cd8d35a2c6e1d10df21961b8e79018a9914b144c65d3f9eef6af744073852a7`.

The LoCoMo rubric rejects some relative-to-absolute date conversions and extra list items, while the
grounded answer prompt asks for resolved dates. Both instructions were preserved. Judge results can
therefore reflect answer-format compatibility as well as retrieved memory quality. The final wire
also contains a top-level `enable_thinking: false`; the generation configuration's nested
`chat_template_kwargs.enable_thinking: false` was not merged into it. The endpoint did not report
reasoning-token counts, so no cause is assigned to failures from older protocol variants.

## LoCoMo results

The common strict-schema pass completed all 828 rows with no scorer errors. It made 552 successful
network judge calls and served 345 exact-wire cache hits. These are per-reference plan calls, so they
exceed the 828 answer rows when one question has multiple references. The primary scores are:

| Arm | Correct / 138 | Accuracy |
| --- | ---: | ---: |
| Raw baseline `Memory.ask` | 104 | 75.36% |
| Raw baseline `Memory.compile` | 96 | 69.57% |
| Formed baseline compiler | 97 | 70.29% |
| Formed final-v3 compiler | 97 | 70.29% |
| LightMem feasibility arm | 97 | 70.29% |
| StructMem feasibility arm | 94 | 68.12% |

The raw track compares answer interfaces on the baseline product, not compiler versions. Across 138
questions, 87 were correct for both, 17 only for `ask`, 9 only for `compile`, and 25 wrong for both.
Compile had higher deterministic token F1 (0.5682 versus 0.3366) but lower common-judge accuracy. The
F1 gain is largely sensitive to terse outputs and is not evidence that compile answers are better.

The answer interfaces had different evidence limits in these frozen runs:

| Interface | Retrieval / context limit | Answer-generation limit |
| --- | --- | --- |
| `Memory.ask` | At most 20 ranked hits (`recall_limit=20`) supplied through the product's grounded-answer path | Qwen provider default output limit; no `max_tokens` field; at most 8 input videos |
| `Memory.compile` | `ContextBudget(max_items=24, max_chars=16000)`, with media priced inside the character budget | The same Qwen provider default output limit and video cap through the benchmark generator |

The prompts also differ: `ask` uses the product's grounded-answer prompt, while compile sends the
rendered bundle through the benchmark's full-context wrapper. Consequently, ask/compile token and
score differences are interface diagnostics rather than a budget-matched retrieval comparison. The
provider-ready request captures, where available, are the authoritative prompt and media record.

The formed track is the algorithm comparison. Baseline and candidate compiled the same closed
419-source store with `ContextBudget(max_items=24, max_chars=16000)`. The 138 full provider requests
changed between arms, so all 276 answers were freshly generated. Both generation arms completed
138/138 with no retry or error. Candidate embedding requests were 139/139 cache hits and made zero
network calls.

The primary formed result is exactly tied: 93 correct for both, 4 only for baseline, 4 only for
candidate, and 37 wrong for both. Final v3 therefore has zero net semantic gain on this conversation,
despite changing all 138 provider requests. The six-arm summary is
`.benchmarks/results/e2e-locomo-common-six-arm-json-schema-cached-20260908-v1/summary.json`. LightMem
and StructMem used the same Qwen model, grounded system instruction, and top-24/16,000-character
answer wrapper as the MindBridge arms. Their upstream-native memory formatting remained different,
and both builds explicitly disabled upstream local pre-compression and topic segmentation. These are
matched feasibility variants rather than canonical reproductions of every upstream module. The
MindBridge compile and formed captures preserve complete provider request bodies with system-prompt
SHA-256 `7220dd72c8a79386679f0b3390e6d62e9d5333d703869eb6f3d362c6bee99af4`.
The external answer runner source hardcodes the same prompt and controls, but its answer rows retain
only a question-plus-context request hash rather than the final provider body. Original raw-ask
provider requests are absent, so its historical final wire cannot be reconstructed. The common judge
wire, rather than every generation wire, is exact across all six arms. The
LightMem build summary covers only the recovered 192-source suffix (15 extraction calls, 55,043 LLM
tokens, and 420 embedding calls), so it is not a full-build cost total; StructMem's summary covers all
419 sources (31 extraction calls, 208,938 LLM tokens, and 1,213 embedding calls). Construction cost
must not be ranked from the suffix-only LightMem accounting.

Structural closure changed materially. Baseline bundles delivered 2,032 derived records; 766 had all
declared evidence present and 1,266 lacked at least one declared link, for 1,468 missing links and
136/138 incomplete bundles. Candidate bundles delivered 1,394 derived records, all with their
declared evidence, and closed 138/138 bundles. Both arms delivered 3,312 total items and had no empty
contexts. Rendered characters increased from 918,084 to 1,106,757 (20.55%). All active evidence edges
in this store directly target observations, so direct and recursive closure counts coincide. These
are structural checks only; they do not establish semantic entailment. No bundle in either arm
rendered an `evidence_unavailable` unknown, so missing baseline links must not be described as
explicitly withheld records.

## Fresh writer validation

The first complete held-out writer comparison used the next LoCoMo-refined unit in release order
after the diagnosed conversation: `conv-30`, with 369 sources and 72 questions. Both arms used 47
fixed ingestion batches of eight sources except the final one, Qwen formation with temperature 0,
seed 42 and an 8,192-token limit, WeMM embeddings through a cohort-local exact-response cache, and
the same 24-item/16,000-character answer budget. The pre-witness arm used the accepted formation/SSE
source before schema 17. The witness arm used the bytecode-free schema-17 v3 archive
`11655a1494403c8845ba8a6fcbafa7a825c77f97f914dc705187ec85c729538e`. Both stores were built fresh
from the identical source schedule; neither reused the formed conv-26 store.

The pre-witness store closed with 369 formation runs, 820 total records and 456 flat evidence links.
The witness store closed with 369 formation runs, 900 total records, 531 derived records, 542 flat
links, 540 support clauses and 542 clause members. Of those clauses, 538 were single-source and two
were real two-source conjunctions; no clause cited all eight batch inputs. Two records had
alternative sufficient clauses, seven inputs produced no persisted derived output, and both durable
queues were empty. The two joint clauses described agreement with another speaker's sentiment and
shared anticipation for an event. This shows that the new representation was exercised; it does not
show that the inferred claims were semantically correct.

A root-agent read of those two natural-discourse clauses found their local context bindings faithful:
“Agreed!” linked Gina's enjoyment-of-friends sentiment to Jon's agreement, and “Can't wait too!”
linked tomorrow's opening to Gina's shared anticipation. This is root-assistant adjudication of two
clauses, not independent human gold or semantic validation of all 531 derived records. The independent
artifact-control review receipt is
`.benchmarks/research/2026-09-09-conv30-witness-acceptance-review-v1/receipt.json` (SHA-256
`686cf9c296f5ea68036e792a765908db433f748deabf0fd98e2b518fc4b71381`).

Candidate construction made 94 logical embedding requests: 47 exact cache hits and 47 network
calls. The transport journal preserved request, response and chunk bytes but omitted terminal token
usage, so formation-token cost is unavailable and was not reconstructed. The stores preserve exact
observation content and observation vectors across arms. Derived counts rose from 451 to 531. Derived
IDs had no intersection because the ID recipe includes the changed recipe/prompt fingerprint; that
fact does not show zero semantic overlap. Exact normalized derived-content multiset overlap was 159.
The cache keys exact whole embedding requests. Independently formed batch payloads differ, so this
does not prove byte-identical vectors for those 159 coincident strings when they occur inside
different multi-object requests.

All 72 query embeddings for the candidate were served from successful baseline cache entries, and
all 72 provider-ready answer requests changed. Both arms therefore generated fresh answers. Under
the same strict-schema cached LoCoMo judge, pre-witness scored 52/72 (72.22%) and schema-17 witness
scored 54/72 (75.00%): 47 both correct, seven witness-only, five pre-witness-only and 13 both wrong.
The exact two-sided McNemar p-value is 0.7744. These 72 questions are correlated within one
conversation, so the p-value is a question-level paired diagnostic rather than an independent
cross-conversation population estimate. Token F1 fell from 0.5381 to 0.5345 and BLEU-1 from 0.4806
to 0.4761. This is a descriptive two-answer gain, not reliable evidence of a general semantic
improvement.

Gold evidence is annotated at exact turn granularity for this unit. Of 96 annotated source turns,
the pre-witness bundles delivered 72 and witness bundles delivered 71. Both had full gold recall on
56/72 questions and at least one gold turn on 64/72. Both arms were structurally closed on all 72
bundles with no missing declared evidence, unknown evidence or empty context. The pre-witness arm
delivered 612 derived and 1,116 observation records; the witness arm delivered 569 derived and 1,159
observations. This non-improvement in source recall and the nonsignificant task score prevent a broad
semantic claim.

A separate eight-case real-Qwen probe tested hand-constructed cross-input and control cases. Both
arms answered all eight questions correctly. Only two of six target claims actually resolved a
cross-input identity or pet claim; their persisted attribution changed from 0/2 exact joint support
to 2/2 exact `AND {source 0, source 1}`, while resolved-target coverage stayed 2/6. The other four
targets preserved atomic statements and remained answerable from full context. Controls did not
overcite target sources, and affect was not promoted to a trait. A withdrawal probe then removed one
member of each joint pair: the witness arm mechanically withdrew both derived claims while the
pre-witness arm withdrew neither. Both answerers nevertheless asserted the facts from remaining
pronoun-only observations and abstained on 0/2. These receipts validate structural attribution and
withdrawal behavior; they do not establish semantic truth or improved refusal.

The conv-30 score, closure and exact-turn recall artifacts are under
`.benchmarks/results/e2e-writer-validation-locomo-conv30-score-json-schema-cached-v1` and
`.benchmarks/results/e2e-writer-validation-locomo-conv30-paired-schema17-v1`. Store closure and
formation comparison receipts are under
`.benchmarks/research/2026-09-08-locomo-writer-validation-v1`. A bounded audit of all 27 visible
model-inferred `STATE` rows found no confirmed contradictory predicate alias; one spelling alias had
compatible multivalued objects. This does not justify automatic slot merging without declared
cardinality and object/event identity. The audit is
`.benchmarks/research/2026-09-09-conv30-state-alias-audit-v1`.

Matched LightMem and StructMem feasibility variants were also rebuilt from all 369 conv-30 sources.
The LightMem variant produced 691 entries with 23 construction calls and 88,850 Qwen tokens; the
StructMem variant produced 964 entries with 46 construction calls and 161,112 Qwen tokens. Their
native retrieval initially embedded the bare question, whereas MindBridge embedded the full frozen
benchmark query wrapper. Before those v1 scores were inspected, a v2 sensitivity run was frozen to
reuse the exact successful MindBridge query-embedding response bytes: 72 cache hits and zero network
calls per external arm. The stores, top-24/16,000-character ceiling, forced-answer system-prompt
SHA-256 `7220dd72c8a79386679f0b3390e6d62e9d5333d703869eb6f3d362c6bee99af4`,
generator and strict cached judge were held fixed.

The aligned-query v2 scores were LightMem 44/72 (61.11%) and StructMem 49/72 (68.06%). Their native
bare-question v1 scores, preserved as a query-format sensitivity diagnostic, were 39/72 and 46/72.
A reciprocal bare-query sensitivity run scored the pre-witness and witness MindBridge stores 52/72
and 53/72, compared with 52/72 and 54/72 under the wrapped query. The query string affects both
retrieval and the compiled text, so these differences cannot be attributed to embedding alone.
The final external user message nevertheless contained the bare source question, while MindBridge's
final user message repeated the full benchmark wrapper, including its concise grounded-answer
instructions. Retrieval-query bytes were aligned for the wrapped condition, but final-question
bytes were not aligned across systems. The four-backend table is therefore a feasibility and
query-format sensitivity result rather than a causal comparison of backend quality. The two
MindBridge arms remain paired within each query condition because their final messages match.

The common 24-item/16,000-character ceilings do not make read cost equal. Across all 72 wrapped
queries, pre-witness and witness MindBridge rendered 557,182 and 562,152 characters because they
included evidence closure and headers; aligned LightMem and StructMem rendered 149,986 and 160,082
characters using different semantic units. Observed generation usage likewise records 298,953
prompt tokens for the wrapped witness arm versus 71,043 for aligned LightMem and 72,262 for aligned
StructMem; no provider rate card was frozen, so no monetary comparison is reported. The aligned
external results use complete source histories but remain matched feasibility variants with
upstream-native memory formatting and
disabled local compression/topic modules. They are one conversation's correlated questions, not
canonical-paper reproductions, cost-matched arms or evidence of a population ranking. The complete
four-backend/two-query receipt is
`.benchmarks/results/e2e-writer-validation-locomo-conv30-query-sensitivity-receipt-v3/receipt.json`
(SHA-256 `faaacb99b780196c08edc5855d0b5faf6a1e2314b6d71bac7f454d287067915d`). Exact character and wire
accounting is in
`.benchmarks/results/e2e-writer-validation-locomo-conv30-prompt-read-cost-v2/analysis.json`
(SHA-256 `180f260bde09c82dbbbec7becd5f58eb5b59b2dff1b3d3d7e1ce28df6aec37ec`). The artifacts are under
`.benchmarks/results/e2e-writer-validation-locomo-conv30-competitors-aligned-query-v2` and its
`-score-json-schema-cached-v1` companion.

## Other fixed diagnostics

| Dataset | Completed coverage | Current interpretation |
| --- | --- | --- |
| MemLens-32K | 32 questions, 64 baseline answers | `ask` and `compile` each scored 10/32. A new baseline/candidate-v3 replay found all 32 provider requests byte-identical with shared query embeddings, proving request parity for that new pair only. The original scored request bytes were absent, zero old scores were transferred, and the replay sent no generation calls. A bounded four-case localization found three excluded lower-ranked short turns and one retained `$225` negative control; it does not prove that a later overlapping answer chunk was lost. |
| ATM-Bench-Hard | 31 questions and three complete answer artifacts on one shared 11,034-record history | Generation used protected closed-store clones without corpus ingest or re-embedding. The original compile score of 0.0645 is superseded because its wire omitted selected native media. Corrected compile v3 sent 178 images and 1 video, completed 31/31 with no generation or scoring errors, and scored 0.0968: 3 full-credit and 28 zero-credit answers. Raw `ask` scored 0.2307: 4 full, 6 partial, and 21 zero. The corrected full-credit cross-tab is 3 both, 1 raw-only, 0 compile-only, and 27 neither. This remains a diagnostic set rather than a leaderboard estimate. |
| LongMemEval-S | 16 histories, 32 answers | Both arms completed with zero generation errors under one frozen evaluation hash. A first judge variant was invalid because truncated reasoning contained the substring `yes`, which the declared parser accepted. A separately frozen variant disabled thinking in the provider's nested control, required a bare `yes` or `no` before applying the unchanged parser, and completed all 32 rows without error. Both arms scored 13/16 (81.25%): 12 both correct, 1 only `ask`, 1 only compile, and 2 both wrong. |
| PersonaMem-v3 | 185 questions, 370 generated and scored answers | The fixed roster completed with zero generation errors: questions 36–184 contributed 298 rows and the verified questions 0–35 prefix contributed 72, all under one dataset and evaluation hash. Per arm, 94 rows used the dataset's rubric judge and 91 resolved locally; 131 carry the declared aggregate headline and 54 task rows intentionally do not. All 188 judge calls succeeded on their first attempt. The Qwen proxy `personamem_score` mean was 0.2548 for `ask` and 0.4119 for compile over the same 131 rows. The upstream protocol declares `gpt-5.5`. A later time-firewall audit also found that this generic run preloaded the full persona before answering; these are retrospective full-corpus diagnostics, not official causal or leaderboard scores. |
| Mem-Gallery | 57 questions; 57 original `ask` and 57 corrected compile answers | The original compile artifact supplied only text and remains a superseded diagnostic. Two intermediate media adapters are preserved but unscored: v1 left images ambiguously unbound, while v2 duplicated full memory labels beyond the declared text budget. Final v3 used compact source-bound labels with 5.52% marker overhead, sent all 8 query and 186 selected evidence images, and completed 57/57 without error. Its deterministic local scores were F1 0.6422, BLEU 0.2148, and exact match 0.4737; no LLM judge was called. |
| M3-Bench Robot | 15 questions from `bedroom_01` | Native 30-second video, no audio/ASR: `ask` 1/15, compile 4/15, blind 2/15. One later fixed-clock baseline generation differs from the old run, so the old 4/15 is not relabelled as a candidate result. |

The MemLens and ATM request-parity replays in this table compare the initial schema-16 archive
`34744596…` with the schema-16 compiler-v3 archive `0c9abfc8…`. They used raw stores,
`scope=None`, a 24-item/16,000-character budget, and the then-current text-only answer adapter. ATM
selected assets but sent no provider media. These replays made no answer calls and transferred no
old scores; they do not establish parity for schema 17 or 18, partial excerpts, or the corrected
native-media adapter. The versioned clarification is
`.benchmarks/research/2026-09-09-paired-replay-driver-final-v1/baseline-comparisons-clarification.v2.json`
(SHA-256 `c8ffb0b4258541f1e708bda813ba69ea6951dc3f58511ac9ed6133b64cc3d5ff`).

The original ATM artifact also records a bounded three-case diagnosis. One list-recall question gave
the raw arm 6/8 annotated sources and the compile arm 5/8; raw returned six correct evidence IDs,
while compile emitted an internal memory hash. One numeric question gave both arms 1/2 annotated
sources; raw nevertheless returned the reference amount while compile selected a different amount,
so it does not isolate retrieval from synthesis. In one open-ended question, raw had source IDs for 3/3
annotated sources and answered the shop and dates, while compile received 1/3 and said the facts were
absent. The compile model did not receive selected native images in any of these rows, so the examples
combine source selection, media delivery, synthesis, and answer-interface failures. They do not
establish a corpus-wide ranking or embedding defect.

The completed Mem-Gallery text-only artifact reports local lexical diagnostics: `ask` F1 0.4961,
BLEU 0.1420, exact match 0.2982; compile F1 0.6194, BLEU 0.1924, exact match 0.4211. The corpus has
185 text atoms and 31 native image atoms. SQLite contains 154 text records, 31 image records, 31
image assets, and 31 record-asset links. Eight questions contain a native image. The `ask` answer
model received those 8 query images and 283 repeated grounded-hit image assets across the 57 rows.
The old compile answer generator received zero native images even though compilation used the 8 query
images and selected 186 repeated image-bearing assets. Its scores are therefore not a fair
multimodal-arm comparison. Final v3 binds every image to its asset, source, and memory identity next
to the image while keeping the compiled context inside its declared budget; its local scores above
replace only the old compile diagnostic. PersonaMem contains 504 text memories and 185 text questions
with no media.

A separate Gallery blind control sent only the eight native query images across 57 questions. It had
no store, ingest, embedding, retrieved context, or evidence media, and scored F1 0.3218, BLEU 0.0758,
and exact match 0.2281. It bounds what the query alone produced but is not a causal baseline for the
corrected compile arm. The corrected arm's 0.4737 exact match and the blind control's 0.2281 are
consistent with useful retrieved memory, but the provider runs were not a randomized paired
intervention and the gap is not an effect estimate for the compiler change.

ATM-Hard contains 6,742 text memory atoms and 4,292 native-file atoms; its 31 questions contain 62
text prompt parts and no query media. Frozen compile captures selected 179 image assets, yet their
provider bodies contained zero media parts. The original raw result cache did not retain provider
request bytes, so its exact sent-media count cannot be recovered from that artifact. Corrected
compile v3 bound and sent 178 selected images and one selected video; its binding markers added
10.22% to the declared compiled-context character total without duplicating memory content.
MemLens-32K
contains 3,746 text memories and 32 text questions; its paired captures contain no media. The frozen
LoCoMo conversation, raw and formed, also declares and sends no media. LongMemEval-S contains 8,177
text memories and 16 text questions, and PersonaMem is text-only as stated above.

PersonaMem's older retrospective 185-question artifact reports 43 ranking rows per arm, including
six lifecycle rows without a per-row aggregate headline. Its `ndcg_graded@5` values of 0.2392 for
`ask` and 0.3128 for compile are the frozen target-only discount diagnostic, despite that historical
name; they do not implement the current upstream `+2/+1/-2` gain formula. Recall@5 was 0.4496 and
0.5426, and MRR was 0.2325 and 0.2844. Across the 131 paired rows with a primary score, `ask` was
higher on 14, compile on 58, and 59 tied. The artifact uses the configured Qwen judge rather than
the upstream-declared model and remains a retrospective proxy; its generated answers were not
rerun.

## Causal and attribution boundary

Two comparisons in this work isolate an intended implementation difference. The frozen LoCoMo pair
uses one closed formed store, shared query embeddings, fresh provider requests, and the same generator
and cached judge to compare compiler v3 with its baseline. It found complete structural closure but no
net semantic-score change. The fresh conv-30 writer pair used identical source schedules and common
embedding responses to compare pre-witness formation with schema-17 witness persistence. It exercised
joint clauses and improved the descriptive judge count by two while exact-turn source recall fell by
one; neither change establishes a general semantic gain.

The completed Persona2 writer pair used fresh stores, the same 2,575-source schedule, shared
source/query embedding responses, and the same 149-question roster. It did not preserve the
benchmark's availability intervention: all source memories were formed before all questions, then
`compile(reference_at=..., scope=None)` queried each full store. `reference_at` controls relative
time and decay; `valid_at` is world-valid time and `known_at` is transaction time. None is the
official source-interaction cutoff. Persona2 is therefore a retrospective formation and retrieval
diagnostic rather than a third causal comparison.

The other ask/compile results are broader SDK and harness diagnostics. They compare answer surfaces,
media delivery, or independent construction methods and should not be attributed to compiler v3 or
schema 17. Gallery's corrected media adapter repairs a harness defect; it is not a product-algorithm
result. LightMem and StructMem are independently built feasibility arms rather than tightly paired
backend comparisons.

Attribution is based on the exact initial dirty working tree, whose archive SHA-256 is
`34744596f185f6dc9280dea0b0f21a385a309c429f632f940e468bf6ed9ddff0`, rather than Git status.
The phase inventory is
`.benchmarks/research/2026-09-08-e2e-recovery-v1/change-attribution.json`. Its accepted product
comparison target is the clean bytecode-free schema-17 v3 archive
`11655a1494403c8845ba8a6fcbafa7a825c77f97f914dc705187ec85c729538e`, with source manifest
`11baaa66f0ca3fbffb7c752016df788c6ac2f9fd882d95c0a19dc2e84bc5780c`. In particular,
`src/mindbridge/benchmarks/eval_journal.py` existed in that initial snapshot and is still byte-identical
(SHA-256 `b861a7ec09880745b3f181e277ed584b9075bf985976c45147154b27099e9629`);
its untracked status does not make it a change from this work. The inventory remains a phase snapshot
for final product bytes rather than an attribution of individual overlapping lines to an agent.

The held-out Persona2 plan contains 149 questions across 21 task types. It has no dedicated affect
inference/calibration or biometric/cross-modal identity questions. Eighteen rows concern restraint
around sensitive events, and 17 use simulated voice or relationship-profile rubrics; neither validates
sensor affect or real-person identity. The no-gold answer roster and complete execution contract are
`.benchmarks/research/2026-09-08-personamem-writer-validation-v1/answer-roster.jsonl` and
`.benchmarks/research/2026-09-08-personamem-writer-validation-v1/answer-plan.json`. The writer report
will keep generation failures, unsupported proposals, persisted joint clauses, all-input citation
inflation, output-resolution coverage, and support closure separate from task scores.

Both Persona2 source stores are now closed. Each contains all 2,575 scheduled observations and 2,575
completed formation runs with empty formation and search-index queues. The pre-witness store has 5,506
records and 5,897 embeddings; the schema-17 store has 5,504 records, 5,895 embeddings and 2,930 durable
clause versions. All 2,930 candidate clauses are singleton clauses; one record has two alternative
singletons, but no clause has multiple members. This cohort therefore does not exercise joint-input
AND persistence, and later QA differences cannot be assigned to that mechanism. The model emitted 89
pre-witness and 45 candidate policy rejections with the reason that an affect formation must name a
source modality. Proposal response bodies and a proposal denominator were not retained, so the counts
are not interpreted as a safety improvement.

Both answer arms completed 149/149 generations with zero errors or retries, and the task-dependent
scorer completed all 298 rows with no error. Per arm, 78 questions used the declared Qwen rubric proxy
and 71 resolved locally; 127 judge responses came from the network and 29 were exact-request cache
hits. Ninety-three provider wires changed: among the 77 pairs with a declared task score, schema 17
was higher on 13, pre-witness on 11 and 53 tied. Fifty-six wires were byte-identical: among 46 scored
pairs, each arm was higher once and 44 tied. Sixteen identical wires still produced different
predictions, so provider nondeterminism remains visible. PersonaMem declares different primary metrics
across its 21 task types; no cross-task mean or headline is reported. The corrected task-stratified
summary is
`.benchmarks/results/e2e-writer-validation-personamem-persona2-paired-schema17-v1/paired-analysis-summary.v2.json`
(SHA-256 `3802fbaf9f14821821bb8b10d6318aaf689a5d6f3ba3747d7f1971a64d3f0123`).
Its originally frozen ranking column used the same target-only diagnostic. An offline recomputation
from the preserved 45 ranking slates applied the current upstream formula without new generation:
hidden recommendation was 0.02446 for both arms over six questions; personalized recommendation
was 0.18372 versus 0.11436 over 27; and at-AI directive follow-up was 0.81675 versus 0.83246 over 12.
Lifecycle remains diagnostic because the checked upstream contract does not declare this formula
for that task. The version audit is
`.benchmarks/research/2026-09-09-personamem-ranking-metric-version-audit-v1/receipt.json`; older
artifacts and their mislabeled key remain preserved.

These scores cannot be read as official causal PersonaMem-v3 results. The release specifies one
time-gated view for every task: source events at or after each `T_test` are unavailable. A read-only
audit found future-source delivery on 106/149 questions in both arms. The pre-witness bundles
delivered 437 future raw record instances and 81 derived instances with a future raw ancestor; the
schema-17 bundles delivered 423 and 66, respectively. All listed contexts reached the final provider
wire. Public `search` exposes an `occurred_until` filter, but public `compile` and `ask` do not, so a
historical `RetrievalScope` cannot repair a fully ingested store. The correction requires fresh
public-SDK stores ingested in causal prefixes before each cutoff. The frozen 116-cutoff schedule has
367 batches capped at eight sources and is recorded with the per-question audit in
`.benchmarks/research/2026-09-09-personamem-persona2-future-availability-audit-v1/receipt.json`
(SHA-256 `0b6ff3e0857d496f3855cda809384a061a54789e24aca154df74444e4e24bccd`).

A separate raw-only storage control ingested the same 2,575 observations under the excerpt prototype
without formation, compile, answers or model-network calls. All 2,966 embedding rows and all
non-transactional raw fields matched its schema-17 control. The new store persisted selectors for
391 non-primary embedding text keys across 139 raw memories; those selectors referenced 643
contiguous span pieces, comprising 391 body pieces and 252 context pieces. Hash, containment and
input-key checks all passed. This validates selector persistence on this corpus, not excerpt
sufficiency or QA. The closed receipt is
`.benchmarks/research/2026-09-09-personamem-raw-only-excerpt-preregister-v1/closed/excerpt-schema18-v2-persona2-manifest.json`
(SHA-256 `6f26948bb560b0f7dcd44372d5b2ce37f703263f44d11ffa6e3989af666268b9`), bound to the independent
prototype review receipt SHA-256
`264a9f8abf46f580e564d543825507e531bfc3dd52d259e3f3f19019c4f1f0b5`.

The causal raw comparison then used two fresh public-SDK stores built from the frozen 116-cutoff
schedule. Each question ran against an isolated clone containing exactly the source prefix whose
official interaction end was strictly earlier than its `T_test`; all 149 baseline and 149 excerpt
captures had zero future raw records and zero derived records with a future ancestor. The schema-18
default-off capture matched the schema-17 baseline on all 149 provider wires. Enabling partial raw
excerpts changed 73 wires and left 76 byte-identical. All 298 fresh generations completed on their
first attempt with `finish_reason=stop`, terminal usage, and no error. The baseline used 711,251
total generation tokens and the partial-excerpt arm used 710,046; provider usage is operational
accounting, not a product-cost result.

The corrected current upstream ranking formula gives these task-specific local results; it does not
define a cross-task average:

| PersonaMem ranking task | Questions | Raw baseline `ndcg_at_5` | Partial excerpts `ndcg_at_5` |
| --- | ---: | ---: | ---: |
| At-AI directive follow-up | 12 | 0.87743 | 0.85779 |
| Hidden persona recommendation | 6 | 0.16884 | 0.11985 |
| Personalized recommendation | 27 | 0.18122 | 0.19220 |

The task-dependent rubric scorer completed 297/298 rows. One baseline sensitive-event row, q0014,
exhausted three attempts with `ReadError`; its score remains null in the fixed denominator, with no
imputation, exclusion, or later retry. The judge transport recorded 114 network successes, 41
exact-request cache successes, and three failed attempts; every successful response ended with
`finish_reason=stop`. Across the 73 changed-wire pairs, partial excerpts scored higher on 12,
baseline on 11, 41 tied, and nine were unavailable under heterogeneous task scalars. Across the 76
equal-wire pairs, partial excerpts scored higher on one, baseline on none, 57 tied, and 18 were
unavailable. These direction counts are not a cross-task quality mean.

Of the 76 identical provider wires, 17 produced different answers, so generation nondeterminism
remains directly visible. The partial arm delivered 1,047 full sources and 105 partial sources,
versus 1,201 full sources in baseline; it did not add full-source coverage and reduced prompt input by only 1,309 tokens
across the run (0.187%). The frozen qualitative review traced all 17 equal-wire answer differences
and the first three release-order excerpt cases back to their raw parents and selectors. It found
exact byte containment, while also showing that containment alone does not establish speaker,
applicability, qualification, or withdrawal semantics. Partial delivery therefore remains opt-in
and cannot be described as a semantic gain. The frozen review is
`.benchmarks/research/2026-09-09-personamem-causal-raw-root-adjudication-v1.md` (SHA-256
`2c20a28f05d058ffa5567b79a278e5cb3ac111909d66ca78471b7157d7ad5dcd`). The answer verification is
`.benchmarks/research/2026-09-09-personamem-persona2-causal-prefix-v1/causal-answer-verification.v1.json`
(SHA-256 `4bfb53276e6aa68cfaacff59780bfe5dabff47fcdb349f409dfd5e31e961c0c0`),
and the offline summary is
`.benchmarks/research/2026-09-09-personamem-persona2-causal-prefix-v1/causal-answer-offline-summary.v1.json`
(SHA-256 `dbdef0a9384db6cf50b16507458ccc5c86abdff8a5a90cbcebb261a186b92a04`).
The scored paired-analysis receipt is
`.benchmarks/results/e2e-personamem-persona2-raw-causal-prefix-v1/paired-analysis-v1/receipt.v1.json`
(SHA-256 `ea2d00284a0ea998218bcef0ef64a6961654f0cca1c75004a9f597c342b0c000`).
A root-agent read of the fixed six directional cases, all 12 sensitive-event rows, and the actual
role/source context for q0018 and q0041 found no supported privacy or affect improvement. In q0018,
the judge's claim that braking and home preferences lacked support conflicted with text present in
the context, while an assistant statement and negative home engagement still did not establish a
positive user preference. In q0041, both arms received the wedding-fund fact; the judge penalized
only baseline's extra wedding detail, and both arms also received an explicit private/forgotten
house-savings statement. The preserved scores were not rejudged. This bounded assistant review is
`.benchmarks/research/2026-09-09-personamem-causal-raw-scored-root-adjudication-v1.md` (SHA-256
`9451a288cead1b328e6ae116385ff6b3c401451ef422e021978e146170ade9ac`).

### Fresh formed causal-prefix pair

A separate in-progress pair rebuilds formed memory under the same causal firewall. It compares the
pre-witness archive `fd70d57c…` with the frozen schema-18 archive `db8ccb82…`; partial excerpts stay
disabled in the candidate. Both arms use the same 2,575-source release-order schedule, 116 cutoffs,
367 batches capped at eight observations, Qwen formation configuration, and exact source/query
embedding responses. Every question is compiled from a separate clone of the store closed at its
cutoff. The source cache records 367 logical source requests; baseline populated the 334 misses
after 33 existing hits, while candidate reuses the same successful response bytes without source
embedding network calls. This controls raw and query embeddings, but independently generated
derived text can still produce distinct embedding requests.

The original long writers stopped only after their public `add_many` calls had durably completed
scheduled batches 125 and 179. In each case, a stable derived projection changed while its identity
and version history remained durable, and the first runner's overly strict audit treated that valid
versioned update as removal. Independent receipts verified exact raw and formation-run coverage,
empty queues, SQLite integrity, and the retained old/new evidence before a versioned recovery runner
continued at batches 126 and 180 without replaying either committed batch. The recovery runner's
boundary audit is exactly idempotent: an immediate restart accepts only the complete canonical row,
does not append a duplicate, and rejects a changed row. Its independent receipt is
`.benchmarks/research/2026-09-09-personamem-causal-formed-resume-v3-independent-review/receipt.json`
(SHA-256 `93923c330ab9894e4b36c9e864345a4b658f99865f7358688e9a95e329af5b48`).

Final closure uses an external audit because a scheduled multi-source response that fails envelope
validation may still leave all raw observations committed and then be recovered through public
singleton `settle` calls. The audit reports fully qualified scheduled batches and singleton
recoveries separately, validates every actual request roster and response, replays declared record
projection updates, checks the complete runtime/owner chain, and requires every delivered derived
branch to terminate in an eligible raw observation. Passing this structural audit will not turn a
singleton recovery into evidence that the former saw joint context. The accepted finalizer review is
`.benchmarks/research/2026-09-09-personamem-causal-formed-finalizer-v3-independent-review/receipt.json`
(SHA-256 `d2c39d487147e5e913247b1ee448ae40e4c0ccfb4e04089fc4a8d9b2c2169a7b`).
Fresh formed answers and task-specific scores remain pending until both writers and this audit close;
no retrospective Persona2 score is transferred into the pair.

The candidate close manifest is
`.benchmarks/research/2026-09-08-personamem-writer-validation-v1/closed/schema17-witness-persona2-manifest.json`
(SHA-256 `adcacf11fbfe8e6dba4876b03c7fc7eb009357973d40824dab0ddec2b5766094`);
its immutable store archive is SHA-256
`f79cba596903d37fb339533af5a466808a74c66e98bbcaa63c27f27ab49fb6b9`. The corresponding
pre-witness manifest is SHA-256
`a8c7c3d067cd3e7d50203a486088fe605d6dbe06005b21f1226091e7af64463a`, and its archive is
SHA-256 `29ee49ea95988c6f40dc6544766b94da18225f0bd04a463a6cb410ab4333b462`.
An exact reconciliation found no SQLite-only or Zvec-only embedding IDs. Zvec's reported completeness
was 5,861/5,895 for the candidate and 5,863/5,897 for the baseline: 34 documents in each arm remained
in Flat-searchable segments rather than the configured HNSW index. All 34 were visible through both
ordinary and explicit-linear self-vector probes; this does not establish arbitrary-query recall.
Opening Zvec 0.7.0 in read-only mode changed bytes in five live `.proxima` files per arm, so paired
evaluation clones are extracted from the immutable close archives. The audit and mutation ledger are
under `.benchmarks/research/2026-09-09-personamem-zvec-completeness-audit-v1`.

A bounded affect-benchmark feasibility audit found no cached MemEmo or A-MBER release artifact.
MemEmo's primary paper describes EIE, EMU, and EQA but does not publish a linked executable dataset,
fixed unit roster, or complete evaluator contract. The official A-MBER repository at commit
`2910ad1a113ffea4f6b72a849302fd58994aa27e` publishes generation and judging machinery but no
generated scenario release or `all_units.json`; creating one would evaluate a newly generated
benchmark rather than a frozen release unit. MemEmo mixes emotion classification/calibration with
historical memory tasks, while A-MBER uses text-described affect cues and does not test sensor
classifier calibration. The read-only audit and primary URLs are frozen in
`.benchmarks/research/2026-09-08-affect-benchmark-feasibility-v1.json` (SHA-256
`c3e94ad5f55a37fb23afc1bb117a06c276c37d158fefcfe6bf875d3865135dd9`). No substitute dataset was
generated.

## Validation boundary

The frozen compiler-v3 snapshot passed the locked dependency check, Ruff format and lint, mypy, all
1,800 tests, `git diff --check`, the pinned Markdown check, and the pinned link check. The later
formation-SSE integration passed 1,811 tests under its own phase gate. The paired replay driver and
benchmark-only recovery workers have separate focused checks and durable manifests. All fixed
diagnostic coverage described above is complete. Schema-17 witness-clause work remains outside the
frozen compiler-v3 benchmark. Its clean v3 snapshot passed all 1,840 tests with warnings as errors
on CPython 3.12.11 under Linux 6.8.0-138-generic x86_64 with glibc 2.35,
plus the lock, Ruff, mypy, diff, Markdown and link gates. A separate default-scope migration replay
preserved all 138/138 conv-26 provider request bytes and hit state, with zero candidate embedding
network calls. The conv-30 writer comparison is complete. The retrospective Persona2 pair is also
complete. Its raw causal-prefix correction and fresh scoring are complete with no transferred
scores; the fresh formed causal-prefix pair remains in progress and is reported separately from the
raw result. After the post-v3 consolidation change, corrected Persona ranking scorer, and final
benchmark-only paired-replay typing correction, the current tree passed the lock check, Ruff format
and lint, mypy, `git diff --check`, and all 1,876 tests with warnings as errors on CPython 3.12.11.
The source-bound receipt is
`.benchmarks/research/2026-09-09-final-source-gate-v1/receipt.json` (SHA-256
`23148a06b32e087f36c036af89d73fb01a3b0d9942c12bea2fa3bc0d71a5c987`).
