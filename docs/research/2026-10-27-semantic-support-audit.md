# Formation semantic support audit — 2026-10-27

## Decision

The reported boundary exists. It is a semantic-quality risk above working provenance mechanisms,
not a missing evidence chain. Add an offline measurement path now; do not add an unconditional
serial LLM verification call or promote a new admission policy without measured benefit.

This review started from `b6b5ad28` after the kernel/storage refactor. Other concurrent workspace
changes are outside this patch. No product prompt, model, retrieval policy, public SDK contract,
disk schema or default is changed here. The new benchmark leaf scores independently reviewed
atomic claims; it neither infers semantics nor writes to the runtime.

## Code and practice evidence

- `kernel/formation.py::_proposal_sources` validates batch/private-history membership, primary
  citation and active evidence. `formation_refusal` checks host policy authority, affect modality
  and observation frame/anchor. `_history_identity_refusal` requires a standing cited naming
  witness for a name known in history. None compares the source text with the proposed assertion.
- `_grounded_formation_pairs` detects competing structured state values within a source and
  overlapping lineage. It does not detect a proposal contradicting ordinary source prose.
- `_commit_targets` rechecks visibility inside SQLite and retains joint evidence clauses. The
  authoritative commit and outbox/index ordering remain unchanged.
- Existing formation regressions cover conjunctive evidence, alternative singleton support,
  withdrawal and missing-index rebuild without re-embedding. These are real guarantees.
- The [September capacity analysis](2026-09-13-memory-capacity-results.md) found missing necessary
  identity citations and structured-output failures. It explicitly separated semantic review from
  an unsuitable original score and did not treat failed writes as safe rejections.
- The [September end-to-end experiment](2026-09-14-end-to-end-results.md) delivered two wrong named
  facts after natural-language name withdrawal. The selected positive provenance and proof were
  complete but omitted retraction evidence. Its frozen protocol used research hooks and is not
  a fresh measurement of this checkout. The current
  [formation contract](../configuration.md#automatic-memory-formation) checks history and
  commit-time activity without proving general entailment.

The new public-SDK counterexample uses the source
`Lin drank tea and did not drink coffee.` A scripted former proposes three events:
`Lin drank tea.`, `Lin drank coffee.`, and `Lin owns a bicycle.` All three are admitted with the
same legitimate evidence ID. The independent labels are entailment, contradiction and unknown.
Soft forgetting the source preserves the derived claims' standing support; deleting the sole
source removes all three. Thus semantic admission and withdrawal are separate properties.
This is a deterministic adversarial witness of the boundary, not an empirical hallucination rate.

## Recent original research

Retrieval used alphaXiv discovery with a 2024-01-01 lower bound and recency priority. An initial
semantic search was off-topic; one focused keyword follow-up supplied the selected papers.
Full texts were read with `orx paper --full` and checked against the original arXiv HTML versions.
The evidence below motivates evaluation design; none measures MindBridge or audio verification.

- [ContraVis, September 2026](https://arxiv.org/html/2609.27014v1) separates graph-based evidence
  neighborhoods from contradiction review by an analyst. Its four-contract comparison uses
  injected contradictions and evidence-span matching. It reports a retrieval/recall benefit, but
  also extra candidates and explicitly uncalibrated model confidence. Its formative study has
  three experts from one firm. This supports preserving structural context, checking locations,
  and measuring false positives rather than treating a model verdict as proof. Its own caution
  is: "the results do not support generalizable conclusions about usability or effectiveness."
- [Linguistic Features for Interpretable Textual Entailment, September 2026](https://arxiv.org/html/2609.24932v1)
  evaluates both three-way SICK and binary SICK-CE. The paper explains that SICK is controlled and
  avoids complex named entities and idioms; binary results omit neutrality. This supports keeping
  unknown separate and testing MindBridge's identity, time and condition cases independently.
  It does not justify choosing this classifier for memory admission.
- [Video-Language Entailment for Video Similarity Estimation, October 2026](https://arxiv.org/html/2610.24786v1)
  uses concept-conditioned captions and video-caption entailment for video-pair similarity. The
  authors report additional forward passes and warn that sparse sampling can miss relevant
  moments in long videos. Their explicit limit is: "VICE is not designed for long videos".
  This motivates separate original-frame localization and cost reporting; it is not evidence
  that text entailment or caption matching verifies arbitrary image/video/audio claims.

## Implemented evaluation boundary

The [offline audit guide](../semantic-support-audit.md) defines the JSON input and denominators.
The scorer separates independently annotated gold labels, verifier predictions and actual durable
admission. It reports unsupported assertions, supported-claim retention, classification and
contradiction recall, exact source/asset location sets, withdrawal residue/preservation and
measured costs. Missing observations have explicit coverage; failures cannot become safe refusals.
Gold location sets retain AND requirements and OR alternatives. Soft forgetting is excluded from
withdrawal rates. Unknown costs remain unknown, including batched request usage.

The versioned example `semantic-boundary-text-v1` mirrors the scripted SDK case:

| Quantity | Result | Interpretation |
| --- | --- | --- |
| Admitted atomic claims | 3/3 | Scripted counterexample, not natural model behavior |
| Unsupported admitted claims | 2/3 | One contradiction and one unknown |
| Entailed-claim retention | 1/1 | Positive control |
| Residual after sole-source deletion | 0/3 | Structural withdrawal works |
| Verifier quality, perceptual localization, model cost | Not measured | No real verifier or model call |

Validation fixtures include a wrong source ID, incomplete joint locations, valid alternative
sets, all three coordinate kinds, missing predictions, admission errors, surviving OR obligations,
soft forgetting, partial usage accounting and invalid inputs. The media rows are annotation and
metric tests; no pixels or audio are interpreted. The SDK witness uses real SQLite/Zvec through
public imports with a fresh temporary directory and constant four-dimensional fixture embeddings.

## Promotion criteria and limits

Before adding a high-risk formation or consolidation verifier, freeze the claim decomposition,
source/counterevidence snapshot, configured and returned model identities, prompt/fixture revisions,
pass criteria and cost budget. Compare paired baseline and candidate arms on held-out supported,
contradicted and unknown claims. Include qualifiers, ambiguous identities, natural-language
retractions, injected instructions, explicit withdrawal and media localization. Count every failed
attempt and retry. Gate on unsupported reduction together with supported retention, complete
lifecycle obligations and the predeclared budget; rejecting every proposal is not a benefit.
Separate literal assertions from model-inferred traits and affect hypotheses: strict entailment
may reject legitimate induction. Unknown support is not contradiction or a host withdrawal command.

No real-service experiment, trained entailment classifier, multimodal perception benchmark or
production admission verifier was run or added. The general semantic gap remains explicit.
The present change removes the measurement blind spot and gives future verifier proposals a
reproducible evaluation boundary without increasing ordinary ingestion model calls.

## Validation performed

An isolated archive of `b6b5ad28` with only this patch applied passed all **2,610 tests** with
`pytest -W error` on Python 3.12.14. The 16 new tests and the existing formation tests also passed
together (44 tests) in the shared workspace. The isolated snapshot passed lockfile verification,
Ruff format/check and strict mypy (253 source files). The pinned Markdown and link checks passed;
link redirects are informational. Snapshot tests used the existing interpreter with an explicit
`PYTHONPATH` pointing to the archived source, rather than changing the shared editable install.
These checks establish local behavior and metric calculation, not real-model verification quality
or compatibility of unrelated concurrent workspace changes.
