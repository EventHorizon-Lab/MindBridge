# Offline semantic support audit

Formation validates declared provenance, activity, modality and spatial frames. Those checks do
not establish that the evidence entails the proposed sentence. An independent review can measure
this boundary before introducing a semantic verifier into high-risk formation or consolidation.

## Run the scorer

```bash
uv run --frozen python -m mindbridge.benchmarks.semantic_support \
  --input benchmarks/semantic_support.example.json \
  --output /tmp/mindbridge-semantic-support.json
```

This offline module reads annotations and observed outcomes. It does not generate judgments,
contact providers, open a memory store, or change admission. The aggregate JSON contains no claim
or source IDs. Invalid input returns exit code 2 without printing the offending values.

The example corresponds to the scripted public-SDK counterexample in
`tests/benchmarks/test_semantic_support.py`: the source is
`Lin drank tea and did not drink coffee.` and the atomic proposals are `Lin drank tea.`,
`Lin drank coffee.`, and `Lin owns a bicycle.` Current formation accepts all three with valid
provenance. Deleting their sole source removes them. Its unsupported rate is therefore 2/3 and its
withdrawal residual rate is zero. These intentionally adversarial scripted proposals are a
boundary demonstration, not an estimate of model error frequency. No verifier was run; its
classification, localization accuracy and costs remain `null`.

## Prepare independent annotations

Each input describes one arm and one fixed evidence snapshot. Preserve a private sidecar with
the source content, atomic claim text, reviewed evidence clauses, timestamps, counterevidence,
configured model snapshots, returned model identities and annotation decisions. Put its immutable
version or digest in `fixture_version`, the formation recipe in `producer_revision`, and the
verifier's pinned model and prompt recipe in `reviewer_revision`. The example uses a scripted
recipe, not a real model revision.

Split compound proposals into independently scored atomic claims before review. Record every
scheduled claim, including failures; do not discard rejected or failed cases. Gold labels must be
annotated independently of the verifier under test:

- `entailment`: the supplied, applicable evidence supports the entire atomic assertion, including
  its subject, negation, time, conditions and exceptions.
- `contradiction`: the applicable evidence contradicts that assertion.
- `unknown`: support is insufficient or unresolved; absence is not contradiction.

Review contradictory or retracted historical evidence in the same snapshot. A verifier that sees
only a selected positive witness cannot establish that omitted history contains no rebuttal.
Review each AND clause as a whole and its OR alternatives separately; do not reinterpret the union
of all citations as one independent corroboration. The scorer consumes these judgments and cannot
establish their correctness, evidence completeness, or whether text is truly atomic.

Stratify memory kinds and evidence bases as well. A model-inferred trait or affect assessment may
be a useful hypothesis without being logically entailed. Score literal factual assertions and
inductive hypotheses separately before choosing an admission policy. An `unknown` verdict alone
is neither contradiction nor authorization to withdraw an existing memory.

Each row requires `claim_id`, `gold`, and `admitted`. `admitted` is the observed durable admission
outcome, independent of `predicted`, the optional three-way verifier judgment. A failed attempt
uses `admitted: null` and a safe `error_code`; it is never a successful rejection. A verifier
prediction or predicted location requires `reviewer_revision`. Duplicate claim IDs, coercible
booleans, unknown fields and invalid coordinates fail validation.

## Locations and lifecycle outcomes

`gold_locations` is an array of complete acceptable location sets: each inner set is AND, and
the alternatives are OR. `predicted_locations` is the verifier's returned set; `null` means not
evaluated, while `[]` means it tried and found no location. Exact matching requires one complete
gold alternative, including every source and asset ID. A valid substring alone is not semantic
support. Coordinates are half-open intervals:

| Kind | Coordinate unit | Required identity |
| --- | --- | --- |
| `text` | Unicode code points in the frozen source text | `source_id`; omit `asset_id` |
| `frame` | One zero-based original-media frame, `[i, i+1)`; a still image uses `[0, 1)` | `source_id` and `asset_id` |
| `audio` | Original-media milliseconds, `[start, end)` | `source_id` and `asset_id` |

For example, an audio segment is
`{"source_id":"s1","kind":"audio","asset_id":"a1","start":1200,"end":1800}`.
These coordinates are review artifacts, not new product citations. Bounds, transcript alignment
and actual perceptual support must be checked against the original media independently.

An optional `lifecycle` describes an initially admitted claim after an operation. Use
`kind: "withdrawal"` for an explicit dependency withdrawal, correction, rollback or deletion,
and `kind: "soft_forgetting"` for cognitive forgetting. `expected_absent` is the reviewed
visibility obligation after accounting for surviving OR alternatives. `visible_after` records
ordinary current recall eligibility, not mere persistence of an auditable record; `null` means
the observation was not collected. The SDK regression uses `get()` plus `context.visible` and
record absence after deletion. Natural-language retraction requires its own semantic annotation;
the scorer never translates it into a host operation.

## Read the joint report

| Metric | Numerator / denominator |
| --- | --- |
| `unsupported_assertion_rate` | Admitted contradiction or unknown claims / all admitted claims |
| `supported_retention_rate` | Admitted entailed claims / all gold entailed claims, including failed attempts |
| `classification_accuracy` | Correct verifier judgments / supplied judgments |
| `contradiction_recall` | Contradictions labeled contradiction / all gold contradictions; missing judgments are misses once any judgments exist |
| `localization_exact_match_rate` | Complete correct sets / claims with gold locations; missing predictions are misses once any locations were evaluated |
| `withdrawal_residual_rate` | Still-visible claims expected absent / observed withdrawal cases expected absent |
| `withdrawal_preservation_rate` | Still-visible claims expected present / observed withdrawal cases expected present |

Coverage fields accompany admission, classification, localization and both withdrawal rates.
Zero eligible cases produce `null`, and absent verifier evaluation is unmeasured. Soft forgetting
is counted separately and excluded from withdrawal rates. Preservation and retention prevent
"reject everything" or "delete everything" from appearing successful.

`write_cost` and `verification_cost` optionally report measured run totals: `latency_ms`,
`requests`, `input_tokens`, and `output_tokens`. Count batched requests once and include retries;
unknown measurements remain `null`. Do not infer billed costs from wire sizes or fixture counts.

Compare equivalent baseline and candidate snapshots using the same claim population, gold labels,
counterevidence and budgets. Run storage behavior through the public SDK with a fresh physical
directory per case. Stratify identity, temporal qualifiers, prompt injection and each media kind.
Only consider a high-risk admission verifier after a held-out paired evaluation reduces unsupported
claims without unacceptable supported-claim loss, preserves withdrawal obligations, and meets the
application's predeclared latency and token budget. This scorer establishes measurement machinery;
it does not establish a verifier's benefit or authorize a new default.
