# Grounded recall quality evidence

These changes promote general retrieval delivery, chronology, native-media transport, optional
query vision/concise answers, self-evidence rejection, and evaluation correctness. They do not
promote external maps, GPS input augmentation, query-image neighbor expansion, automatic state
classification, additional OCR/rerankers, or local quantized planner substitutions.

## Completed dataset evidence

All runs below used the public SDK, independent physical data directories, original questions,
fixed references and official task scoring. The remote models were the user-hosted aliases
`Qwen3.8-27B` and `tencent/WeMM-Embedding-9B`; checkpoint precision/revision was not disclosed.
Seed 0 does not guarantee deterministic remote output.

| Completed run | Questions | Primary score |
| --- | ---: | ---: |
| ATM raw v4 | 1013 | QS 0.536872248 |
| ATM raw v6 | 1013 | QS 0.594972500 |
| Mem-Gallery v4 | 1711 | F1 0.794585112 |

ATM number/list/open scores changed from 0.60833/0.63203/0.46109 to
0.75000/0.65257/0.47082. Complete gold source/asset delivery rose from 826 to 869 questions.
The combination changed product logic, caller numeric instructions and budgets; this is not a
single-factor budget ablation. V6 also supplied native capture metadata, which is excluded from
this promotion. These historical scores are evidence for the selected design, not a claim that
the newly extracted patch has repeated the full dataset scores.

Dataset evaluation SHA256 values:

- ATM: `8bebb9bdb602ee15eb9c1927290c72a7e1239a5b08c2e6aa366b53c25992af08`.
- Gallery: `9ac83cc5643c607e781049d7d898b3dfc02d5c705236089414ca19715a53bd33`.

Frozen source aggregate SHA256 values:

- V4: `50eb3e32f8c1abe9a8e6c7d62789ff2a5db01f00c76de843823555d4fcfa2498`.
- V6: `ab1385dd49c3abfd7b1b0456d2f0be30e5ad94821edf2b2b38925380b1881fc0`.

The committed example profiles use 64k characters/set100 for ATM and 32k/set60 for Gallery,
K12, native WeMM4096/hybrid, query vision, bounded recall rounds and concise non-thinking Qwen.
Their endpoints are replaceable localhost examples. Run with `--gen-kwargs temperature=0.7`;
the harness owns the generation temperature and seed. No credentials are committed.

## Promotion acceptance

Deterministic public-SDK fixtures cover off-ranking bridge recovery, shared route budgets,
strict replan delivery, same-time sequence order, transient query descriptions, failure fallback,
original media integrity, derived-view labels, self-dependent proposals, and local scoring after
judge failures. They establish mechanics, not natural-language accuracy.

The real-service answering gate uses fixture `mindbridge-reader-quality-v2`, four cases with three
repeats for each of `default` and `concise`: grounded color, grounded weekday, insufficient
evidence and an instruction embedded in stored content. Acceptance requires all cases to pass;
service errors remain failures. It uses the remote Qwen alias, temperature0.7/seed0/top-p0.8/
top-k20/presence1.5, non-thinking, max4096 and no transport retries. The gate does not evaluate
vision/planning schema support or establish full benchmark gains.

| Revision | Answer style | Passed / calls |
| --- | --- | ---: |
| Initial extracted patch | default | 9 / 12 |
| Initial extracted patch | concise | 12 / 12 |
| Cloud base `0513c7d2` | default | 11 / 12 |
| Promoted patch | default | 12 / 12 |
| Promoted patch | concise | 12 / 12 |

The initial default failures were over-abstentions on the stored-instruction fixture, not
execution of the injected marker. The base also over-abstained once. Before the final check,
ordinary text payloads were restored to their previous shape, restricting source-modality
fields to media evidence. An offline HTTP transport comparison then confirmed byte-identical
default request bodies for all four fixtures against the cloud base. No default answering
prompt was changed. This comparison does not establish wire identity outside these fixtures.

The final patch passed the acceptance criterion for both styles. All checks returned the model
alias `Qwen3.8-27B`, with no service errors and no emitted injection marker. Earlier failed checks
remain in the [aggregate evidence](artifacts/grounded-recall-20261029/reader-quality.json), alongside
the fixture hash, source hash, sampling settings and request-body hashes. Remote stochastic
variation remains a limitation; this small suite is not a statistical accuracy estimate.

No new complete dataset evaluation or live vision/planning/formation quality check is claimed for
this extracted patch. The historical dataset results and deterministic regression fixtures cover
different scopes and must not be read as a fresh full-benchmark result for this commit.

## Submission validation

The extracted patch passed the locked dependency check, Ruff formatting/lint, mypy, and
`git diff --check`. `pytest -W error` completed with 3294 passed and 12 skipped in 780.73 seconds.
The skipped formation-time checks require explicit real-service opt-in; no live formation
quality claim is made here. Both example configurations passed task listing and zero-network
configuration validation.

Pinned markdownlint-cli2 0.23.0 reported zero errors across 90 Markdown files. Pinned Lychee
0.23.0 checked 939 links: 930 OK, eight redirects, and one timeout on the pre-existing W3C SKOS
mapping reference in `2026-10-27-predicate-stability.md`. Direct TLS checks against both resolved
addresses also failed. This external link gate remains incomplete; no host exclusion or source
replacement was added to hide it. The tools ran as local binaries with the documented CI
arguments because Docker was unavailable. A process-local address pin was needed for the
existing Geofabrik link; TLS verification remained enabled.
