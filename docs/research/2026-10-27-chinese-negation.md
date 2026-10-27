# Chinese negation weighting

Final decision: lexical coverage measures observed term overlap using ordinary IDF. Remove
the uncalibrated sixfold negation multiplier for both words and bigrams; retain negative
terms and contraction normalization. The prefix rule below was rejected. Its measurements
are historical evidence, followed by the final repair, acceptance checks, and tradeoffs.

## Finding and rejected prefix rule

R02 is reproducible on baseline `b6b5ad280404216d04f2f4bd972a90966e0c377b`. Hybrid
coverage multiplies a negation term's IDF by six, but its Chinese cues were single
characters while unspaced Chinese emits only adjacent-character bigrams. For example,
`不喜欢` produces `不喜` and `喜欢`; neither matched the old negation term set.
The English `not` path was reachable. This is a weighting defect, not evidence that
Chinese or multilingual candidate retrieval is generally broken: the bigrams still match,
and complete lexical coverage remains reachable.

The first proposed repair retained the stemmed and bigram routes and applied the sixfold
weight to a bigram beginning with `不`, `没`, `未`, `无`, `非`, or `别`, while English
cues retained whole-term matching. Requiring the whole bigram meant `不喝` could not
satisfy a query's `不喜`. This repaired representation compatibility without changing
tokenization or candidate discovery, but its ranking policy failed the re-review below.

Recent practice supports retaining topical retrieval while testing polarity separately.
The [Zvec n-gram release](https://zvec.org/en/blog/2026-08-25-zvec-release/)
describes overlapping character sequences for ambiguous word boundaries.
[E-SENS, pp. 1–2](https://www.alphaxiv.org/pdf/2608.30130?page=1) studies a
"recall–violation trade-off" when reranking negative-constraint queries. The local
decision is narrower: repair an existing lexical signal and verify opposite-polarity
candidates, without adopting that paper's query rewriting or penalty method.

## Initial paired evaluation of the rejected rule

Evaluated on 2026-10-27 with CPython 3.12.14 and Zvec 0.8.0. The fixture revision is
`mindbridge-chinese-negation-v1`. Its six cue pairs and three subject forms are also
replayed in the final comparison's legacy diagnostics. The current acceptance tests live in
[`test_negation_ranking.py`](../../tests/unit/test_negation_ranking.py).
The embedding model and revision are the deterministic `synthetic-negation-v1`, space
`synthetic-negation-v1:2`; no external model service or answer generator was called.

Each public-SDK case ingests three records through real SQLite, outbox, and Zvec paths
into a fresh physical directory. The desired record preserves the queried negation;
one distractor removes it, and another negates a different clause. Distractors share
additional time words with the query and have cosine 0.72 against the desired record's
0.70. Six cue phrases are crossed with Chinese, mixed Latin/Chinese, and fullwidth
Latin/Chinese subjects, giving 18 cases. The fixed vectors isolate the ranking policy;
they are not measurements of a natural-language embedding model.

Acceptance requires the desired record to rank first, all three records to remain
retrievable, scores to stay strictly between zero and one, and all candidates to retain
their lexical route. Direct coverage tests additionally require sixfold weighting for
each cue, exact coverage of one, no single-character query terms, no cross-boundary
matches, and unchanged English whole-word/contraction handling.

| Check | Baseline | Rejected prefix rule |
| --- | --- | --- |
| Desired record ranks first, public SDK | 0/18 | 18/18 |
| Chinese coverage-weight regressions | 0/6 | 6/6 |
| English and token-boundary controls | 6/6 | 6/6 |

The baseline used an isolated archive of the pinned commit plus the new tests; the fixed
copy added only the ranking change. This avoids unrelated concurrent working-tree edits.
The focused command was `uv run --frozen pytest -W error tests/unit/test_negation_ranking.py`:
baseline 24 failed / 6 passed; fixed 30 passed. Verification reused the installed environment
with `UV_NO_SYNC=1`, and the isolated source was selected through `PYTHONPATH`.

## Adversarial re-review

A follow-up on 2026-10-27 tested whether the reachable rule is also a sound ranking
signal. Public-SDK probes used the same isolated product snapshot, real SQLite/Zvec,
and fresh physical directories. Only the coverage function was swapped between the
pinned baseline and the proposed rule. The deterministic model revision was
`r02-negative-control-v1`, space `r02-negative-control-v1:2`. Each desired record had
cosine 0.72 and each distractor 0.70. All three cases reversed a previously correct order:

| Query | Desired record | Distractor ranked first after the change |
| --- | --- | --- |
| 非常安静的酒店 | 安静的酒店，适合休息 | 非常热闹的酒店，适合聚会 |
| 特别喜欢红茶 | 喜欢红茶 | 特别喜欢绿茶 |
| 不喜欢喝浓咖啡 | 不爱喝浓咖啡 | 不喜欢喝茶 |

For the hotel case, baseline scores were 0.768601 for the desired record and 0.736860
for the distractor. Proposed-rule scores were 0.746340 and 0.761200 respectively.
`非常` is not a negation cue here. In the second case, the tokenizer emits `别喜`
across a word boundary inside `特别喜欢`; its initial character does not make it
negative. The third case uses genuine negation, but overweights an exact surface
phrase about a different object relative to a paraphrase about the queried object.

A separate admission control reused the hotel records with both cosines fixed at 0.45
and explicitly set `minimum_relevance=0.55`. The baseline returned no hits; the
proposed rule admitted only the noisy-hotel distractor, with score 0.562200. Candidate
discovery being unchanged therefore does not imply unchanged returned recall or admission.
The rule also weights `非常` in the Japanese query `非常口の場所`: it has no language
guard, so its ranking effects are not restricted to Chinese.

These are controlled counterexamples, not estimates of real-world regression frequency.
They invalidate the stronger acceptance claim. A subsequent solution needs evidence that
its polarity signal is bound to the relevant relation and object, neutral behavior when
the signal is uncertain, and paired positive/negative, paraphrase, scope, and multilingual
controls. Fixing token/weight compatibility alone does not establish those properties.

## Final repair: keep statistical evidence separate from polarity inference

The root policy error was assigning semantic authority to a term's spelling. A token does
not identify which relation or object a negation applies to; neither an English word nor a
Chinese bigram establishes that relation. Restoring a dead multiplier therefore made its
unsupported inference active. Exception lists for ordinary compounds would only address
the examples already seen.

The final implementation deletes `_NEGATION_WEIGHT`, `_NEGATION_TERMS`, and
`_NEGATION_CHARACTERS`. Every retained query term uses the same smoothed IDF:
`log((candidate_count + 1) / (document_frequency + 1)) + 1`. Coverage is the sum of
matched weights divided by the sum of all query-term weights. Equal-frequency terms receive
equal weight in every script; their spelling does not create a second relevance policy.

Negation is preserved as literal evidence. A candidate missing the query's negative word or
bigram has incomplete coverage. A complete negative phrase still reaches coverage one and
the existing full-coverage lift. English contractions still contribute normalized `not`.
The tokenizer, model clients, candidate routes, score bounds, index recipe, and storage
contract are unchanged by this repair. No parser, language dictionary, exception list,
additional model call, or dependency was introduced.

The acceptance fixture revision is `mindbridge-negation-evidence-v2`, with deterministic
model `synthetic-negation-v2`, space `synthetic-negation-v2:2`. Tests cover six Chinese
cues, equal-frequency English and Chinese terms, token boundaries, contractions, 18 literal
Chinese/mixed/fullwidth cases, three literal English cases, six adverse Chinese/English/
Japanese cases, and the explicit 0.55 admission control. Desired adverse candidates have
cosine 0.72 against distractors at 0.70; literal cases use equal cosines of 0.70. All stores
are real SQLite/Zvec and each SDK case has its own physical directory. No real model service
was called.

The older 18 multi-clause cases are diagnostics, not a promise of semantic understanding.
Their desired record omits the Tuesday evidence while the other record has a stronger fixed
dense score. Pure term overlap cannot establish which clause decides the intended relation;
the final policy deliberately does not assert that a negative-looking fragment outranks all
other evidence. Their original inputs and directional outcomes remain part of the comparison.
This is a disclosed loss of the rejected heuristic's forced preference, not proof that the
desired record is semantically worse. Complete grammatical polarity and scope remain outside
the coverage function's contract.

## Final paired evidence and validation

The [machine-readable comparison](artifacts/negation-evidence-20261027.json) retains
per-case results, the historical diagnostic inputs and scores, source hashes, Python/Zvec
versions, and model identities. Each arm used the pinned baseline plus only its coverage
policy; the same v2 test source ran in all three isolated archives.

| Check | Original policy | Rejected prefix rule | Final pure IDF |
| --- | --- | --- | --- |
| Literal negative evidence first, SDK | 21/21 | 21/21 | 21/21 |
| Adverse object/paraphrase/ordinary-word/Japanese controls, SDK | 4/6 | 0/6 | 6/6 |
| No accidental 0.55 admission, SDK | 1/1 | 0/1 | 1/1 |
| Equal-frequency and token invariants | 15/21 | 5/21 | 21/21 |
| All v2 acceptance tests | 41/49 | 26/49 | 49/49 |
| Old Chinese multi-clause desired record first, diagnostic | 0/18 | 18/18 | 0/18 |
| All old Chinese diagnostic records remain returned | 18/18 | 18/18 | 18/18 |
| English negative paraphrase desired record first, diagnostic | 2/2 | 2/2 | 0/2 |

The English diagnostic loss is material and was added during independent review. For
`Mira did not purchase green tea`, the desired record is `Mira did not buy green tea`;
the opposite record is `Mira did purchase green tea`. With desired cosine 0.70 and opposite
cosine 0.72, the original policy ranked the desired record first at 0.780147 against
0.748825. Pure IDF ranks the opposite record first at 0.783683 against 0.768232. The
parallel `Mira never purchases red tea` / `Mira never buys red tea` case shows the same
change. All records remain returned. These are genuine useful preferences of the old
English heuristic that this repair gives up; full-coverage tests cannot establish that
partial-coverage negation quality is preserved.

Acceptance here means removing an unsupported semantic multiplier while preserving literal
negative evidence, not improving every polarity-sensitive ranking. The measured losses
preclude a blanket claim of better natural-language negation understanding or unchanged
multilingual quality. Reliable scope/object compatibility still depends on semantic
evidence and task-specific evaluation of the configured models.

Commands were `uv run --frozen pytest -W error tests/unit/test_negation_ranking.py`
for each arm and `uv run --frozen pytest -W error` for the final isolated source. Final
results were 49 focused tests and all 2,643 complete-suite tests passing. The current
shared working tree also passed the 49 focused tests plus the two existing English
negation/strong-semantic-evidence tests. Lock consistency, Ruff, strict mypy, Markdown,
link checking, and `git diff --check` passed. Verification reused the installed
environment with `UV_NO_SYNC=1` and selected each isolated source with `PYTHONPATH`.
An independent review found no product-code blocker and required disclosure of the
English partial-coverage loss; that loss is retained above and in the comparison artifact.
No real model service was evaluated.

## Historical validation of the rejected rule

The isolated patch passed lock consistency, Ruff formatting/lint, strict mypy, and the
30 focused tests. The same focused tests also passed in the concurrent working tree.
Pinned Markdown lint checked 76 files with zero errors; pinned link checking checked
817 links with zero errors. An independent read-only review found no blocking issue.

The complete isolated command, `uv run --frozen pytest -W error`, finished with 2,623
passed and one failure. `test_loaded_jina_uses_pyav_metadata_and_unique_bounded_frames`
failed while importing PyTorch because `libcudnn.so.9` is absent, before any retrieval
code ran. A focused rerun reproduced the same environment failure. The full suite was
therefore not green.

## Limits of the rejected rule

This remains a lexical cue heuristic, not grammatical negation detection. Shared
characters in ordinary compounds such as `非常` and `别墅` can receive the same weight;
double negation, negation scope, paraphrases, and an isolated single-character negator
are not resolved. An unrelated matching bigram elsewhere in a long record can still
contribute coverage. No global claim about multilingual recall, live-model quality, or
downstream answers follows from these controlled fixtures. Real-service evaluations
and a broader Chinese polarity corpus were not run.
