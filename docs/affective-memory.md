# Affective memory direction

This page defines MindBridge's direction for affective context: what an affect record means, what
exists at this release, and the gates a richer capability must pass. It does not claim that every
behavior below is implemented. [Memory types, time, and decay](memory-types-time-and-decay.md)
owns the typed-record contract, [context compilation](context-compilation.md) owns the bundle,
[plugin architecture](plugin-architecture.md) owns the admission rule, and the API references own
current contracts.

## Positioning

The product goal is not to recognize a person's true emotion. Emotion is not a measurable field
that a better model eventually reads correctly. The goal is narrower and achievable: preserve
affective evidence from text, speech, vision, and the person's own statements; track it over time;
and keep it traceably associated with people, events, objects, and later feedback.

An affect record is therefore a sourced, timed, confidence-bearing hypothesis with a perspective.
It is never an unconditional fact about how somebody felt.

Four results set that boundary:

- Core affect is dimensional and continuous, so valence and arousal carry change and comparison
  better than a discrete label
  ([Russell 2003](https://pubmed.ncbi.nlm.nih.gov/12529060/)).
- Facial and vocal expressions are not diagnostic of emotional state; the same configuration
  occurs across states and cultures
  ([Barrett et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31313636/)).
- Self-report, partner report, and external annotation disagree systematically on the same
  interaction, so there is no single ground truth to store
  ([K-EmoCon](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479607/)).
- Text and speech agree mainly on valence, and the agreement varies by content genre, so fusing
  modalities into one label discards the disagreement that carries information
  ([Lindevelt, Verberne, and Broekens, EACL 2026 Findings](https://aclanthology.org/2026.findings-eacl.136/)).

A memory system that stores a label as truth inherits every one of these errors permanently. A
memory system that stores evidence with its perspective can be corrected.

## Four layers

Affect is not one thing at one time scale. MindBridge separates four, and a change at one layer
does not overwrite another.

| Layer | What it is | Representation |
| --- | --- | --- |
| Affective cue | A raw observation containing affective information: the words used, the audio, the frames | An ordinary `OBSERVATION` record; the media stays authoritative |
| Affective state | A situated estimate of how somebody was at one time | `AFFECT`, with valence, arousal, cue modality, confidence, and basis |
| Emotional event | What the affect was about, in the same capture | An `EVENT` record plus the evidence association |
| Long-horizon disposition | A pattern that holds across sessions and contexts | `TRAIT`, produced by the consolidation loop, not by one observation |

Three statements about the same moment are three different things:

- "the voice rose" is an observation;
- "the user may be anxious" is model inference;
- "the user says they were nervous, not angry" is a user statement.

None of the three overwrites the other two. The observation stays as evidence, the inference stays
as a version with its own basis and confidence, and the user statement becomes a new, higher-
authority version in the same lineage. This is the general lifecycle rule in
[Context OS direction](context-os.md), applied to affect.

## What exists today

Everything in this section is current behavior at this release. Each row names the reference that
owns it.

| Capability | State | Owner |
| --- | --- | --- |
| `MemoryKind` carries `AFFECT`, `EVENT`, `TRAIT`, and `RESPONSE_POLICY` | Implemented | [Python SDK](api/python-sdk.md) |
| `EvidenceBasis` carries `observation`, `user_statement`, `model_inference`, and `response_feedback` | Implemented | [Python SDK](api/python-sdk.md) |
| `FormationProposal` carries `valence` in [-1, 1], `arousal` in [0, 1], one `cue_modality`, a validity interval, `confidence`, and `subject` | Implemented | [Python SDK](api/python-sdk.md) |
| `AFFECT` is an episodic record, so it ranks and decays as an episode | Implemented | [Memory types, time, and decay](memory-types-time-and-decay.md) |
| SQLite stores affect fields authoritatively; Zvec stays a rebuildable projection | Implemented | [Architecture](architecture.md) |
| `compile()` gives affect its own bundle section, separate from `traits` | Implemented | [Context compilation](context-compilation.md) |

Four properties of the current implementation matter more than the field list.

**One atomic cue modality, and it must be real.** A proposal's `cue_modality` is a single atomic
modality; `omni` is rejected. The kernel additionally refuses an `AFFECT` proposal whose cue
modality is absent from the source observation's own modalities, so a text-only observation cannot
yield a claim about tone of voice.

**The only affect producer is a prompt.** The bundled OpenAI-compatible `FormationBackend` asks a
model for typed proposals, and affect is one of the kinds it may propose. There is no prosody
analysis, no facial-expression analysis, and no acoustic affect model anywhere in `src/`. Today's
affect quality is the quality of one text-shaped inference, which is exactly why the roadmap below
starts with per-modality estimates rather than with fusion.

**Cues do not reconcile against each other.** Conflict detection and lineage supersession cover
`state`, `relation`, and `trait`. `AFFECT` is deliberately excluded. Two cues from different
modalities at the same moment are two separate records, not a contradiction to resolve: a situated
cue is a report about one channel at one time, not a standing assertion about the world. Keeping
them apart is what preserves the disagreement the EACL result says is informative.

**Traits need independent evidence.** A model-inferred `TRAIT` stays hidden from active retrieval
until two independent evidence groups support the same normalized claim; a trusted user
statement is visible at once. A group is a capture (the observation context's `source_id`),
falling back to the observation record when no capture was named, and a derived record inherits
the groups of its own sources. So several cues extracted from one capture are one group, not
two, and a single talkative turn cannot promote a disposition on its own. [Memory types, time, and decay](memory-types-time-and-decay.md) owns that
rule.

**Affect is associated with events through existing evidence links.** There is no affect-specific
edge. An `EVENT` and an `AFFECT` formed from the same committed observation share that source, and
the `memory_evidence` rows record it. This round exposes that association in the compiled bundle:
an affect entry reports `event_ids` beside its own `context.evidence_ids`, and renders its basis,
confidence, cue modality, valence, and arousal so a reader can judge the estimate instead of
reading a bare label.
[Context compilation](context-compilation.md) owns the bundle contract.

That association is co-occurrence within one capture. It is not an attributed cause. A cue and an
event that share a source were observed together; nothing in the current implementation establishes
that the event produced the affect.

## Perspective mapping

Affect research distinguishes who is making the claim. MindBridge maps two of the three
perspectives onto existing values and records the third as an open gap.

| Perspective | MindBridge representation |
| --- | --- |
| Self-reported: the person says how they felt | `EvidenceBasis.USER_STATEMENT` |
| Inferred: a model estimates from cues | `EvidenceBasis.MODEL_INFERENCE` |
| Observed: another person judges how somebody seemed | No distinct basis today |

The observed perspective is a real gap, not an oversight to close by adding an enum member.
`EvidenceBasis` is a public contract, and a new member changes storage, every transport schema, and
every consumer's exhaustiveness. It is admitted when a caller exists that can produce third-party
judgements and a product path consumes them, and not before.

## Design rules

These are the affect-specific readings of rules the rest of the documentation already states.

**Per-modality estimates stay separate.** Store one record per cue channel. Late fusion is a
compilation-time reading over those records; it is never stored as a truth.

**A user statement does not delete an inference.** It forms a new, higher-authority version. The
inference stays visible in history, because a system that erases its own wrong guesses cannot be
evaluated or corrected.

**Abstention is a valid output.** A backend that cannot distinguish states on the evidence it was
given proposes nothing. An emitted label with no discriminating power is worse than silence: it
costs storage, it enters recall, and it looks like knowledge.

**Discrete labels for expression, dimensions for change.** A discrete label may describe the cue as
observed. Valence and arousal are what make two moments comparable and a trajectory measurable.

**Cause must link to an event record.** Evidence identifiers are provenance, not causation. A claim
that something caused an affect state is an assertion about the world and needs its own record with
its own basis.

**Mood is a long-interval `AFFECT`, not a new kind.** The validity interval already expresses the
difference between a momentary cue and a day-long mood.

**Rate, duration, and change points are derived.** They are computed from the affect time series on
read. They are not new kinds and not new stored fields.

**Affect is never a primary retrieval key.** Ranking is driven by relevance, person, time, and
task. Affect may inform a bounded rerank at most, and retrieval must preserve emotional diversity
in its results. A store that preferentially returns memories matching the current mood builds a
mood-congruent feedback loop, which is a product defect and not a feature.

**`RESPONSE_POLICY` comes only from explicit feedback.** Inferred affect is not consent to change
how the system behaves toward somebody.

**Arousal-weighted consolidation priority stays capped.** High arousal may raise the priority of
deliberation, bounded, so that a single intense episode cannot dominate the slow loop. If such a
weight ever reaches ranking, it must be a `MemoryConfig` flag that defaults off, because a benchmark
run has to be reproducible.

## Event boundaries

Affect change is one boundary signal for segmenting experience into events. It is not the only one
and not the strongest one. Turn and scene end, actor change, place change, topic change, and
activity change are all boundary evidence, consistent with Event Segmentation Theory. A segmenter
that cuts only on affect change produces events that no other query can find.

## Roadmap with gates

Each phase has an exit criterion. A phase that widens the public surface must satisfy the
[admission rule](plugin-architecture.md#admission-rule) in full: a concrete implementation in
`src/`, a reachable caller on a product path, declared modalities and typed output, documented
provenance and privacy and failure behavior, and product-path tests.

### Phase 0: make what exists legible

No public-contract widening beyond exposing fields that are already stored. Affect entries in the
compiled bundle report their basis, confidence, cue modality, valence, arousal, and the event and
source identifiers that already exist as evidence rows. The independence rule for inferred traits
is tightened so cues from one capture count once.

**Exit:** contract tests pin the exposed fields and the independence rule, and a companion-dialogue
evaluation shows the bundle changes downstream answers.

### Phase 1: one real optional affect adapter

One local formation adapter that actually looks at the signal, reusing `FormationBackend` rather
than introducing a protocol. It must emit per-modality outputs, report input quality, be calibrated,
abstain when it cannot discriminate, and record model and recipe provenance. Heavy dependencies stay
outside the core install, in the narrowest optional extra.

A new `AffectBackend` protocol is admitted only if `FormationBackend` provably cannot express that
contract. "Provably" means a written contract that the existing protocol cannot carry, not a
preference for a dedicated name.

**Exit:** the adapter ships in `src/`, is reachable through configuration, and its calibration and
abstention behavior are measured against the text-only baseline.

### Phase 2: richer affect structure

Per-modality estimate, quality, and disagreement fields; an explicit `perspective`; and typed
`about` and `triggered_by` links. Every one of these waits for Phase 0 and Phase 1 evidence that
the current representation loses something measurable. Existing evidence edges are reused first. A
narrow `affect_links` table is added only when traversal is a measured problem, not because a graph
shape reads more naturally.

**Exit:** a measurement showing the current representation loses information the new fields recover.

### Phase 3: dynamics

Per-identity baseline, deviation from it, change points, recovery time, and per-topic trajectory,
all derived from the stored series. `TRAIT` promotion requires cross-session, cross-context
independent evidence, and the person can confirm, correct, roll back, or forget any promoted trait.

**Exit:** change-point detection and trait promotion measured against the false-promotion rate
below, with the user controls implemented rather than planned.

## Required measurements

No MindBridge affect benchmark exists yet. Adding one is part of Phase 1, not a later concern:
without it, every claim in this section is an opinion.

| Axis | Metrics |
| --- | --- |
| Perception | CCC and MAE on valence and arousal; Macro-F1 on discrete labels |
| Calibration | ECE, Brier score, selective risk against abstention rate |
| Association | Emotion-cause pair F1, target accuracy, evidence citation accuracy |
| Dynamics | Change-point F1 and latency, false alarm rate, false trait promotion rate |
| Memory and interaction | Recall@5 and MRR on affect-conditioned queries, affect QA accuracy, factual regression against the text baseline, user correction rate, human ratings of appropriateness and creepiness |

Ablations run in this order, so that each addition is credited only with what it adds:

1. Text-only baseline.
2. Raw multimodal input, no affect typing.
3. A single fused affect label.
4. Per-modality calibrated affect.
5. Event and cause association.
6. Dynamics and trait promotion.

Candidate datasets are MME-Emotion, SemEval-2024 Task 3 (emotion-cause pairs in conversation),
VoiceLongMemEval (arXiv 2609.00570), and a consented MindBridge companion set. External results
guide hypotheses; only reproducible MindBridge runs select defaults, as
[benchmarking](benchmarking.md) requires.

Speech emotion confidence needs explicit calibration before it can gate anything: reported
confidences from SER models are poorly calibrated (Chou et al., Interspeech 2023). A confidence
field that is not calibrated is a number, not a probability, and must not drive visibility.

## Safety boundary

Affect inference is among the most sensitive things this product can do. The boundary is a
precondition for the roadmap, not a later compliance pass.

- Consent is per modality, and the recording state is visible to the person being recorded. Consent
  to transcription is not consent to vocal affect analysis.
- Local processing is the default. A remote call that carries affective media or derived affect is
  an explicit, disclosed deployment choice, consistent with
  [design principles](design-principles.md).
- Retention and deletion are separate for raw media, derived affect, and biometric
  representations. Deleting derived affect must not require deleting the source, and deleting the
  source must recompute or remove what depended on it.
- Prohibited uses: clinical or psychiatric diagnosis, workplace or education evaluation, individual
  risk scoring, and engagement manipulation.

The regulatory floor is explicit.
[EU AI Act Article 5(1)(f)](https://artificialintelligenceact.eu/article/5/) prohibits inferring
emotions of natural persons in the workplace and in education institutions, except for medical or
safety purposes. Annex III classifies other emotion-recognition uses as high-risk, and Article 50
imposes transparency obligations on systems that perform emotion recognition. A deployment that
cannot say which of these it is in should not enable affect formation.

## Known hazards for implementers

Four properties of the current tree will silently defeat a naive acoustic-affect implementation.

**The bundled speech adapter strips special tokens.** `FunASRTranscriber` removes every `<|...|>`
tag from transcribed text. A SenseVoice-style model that signals emotion as `<|HAPPY|>` would have
those tags removed with no error and no warning. An acoustic-affect adapter must not route its
output through that filter; it needs its own typed return path.

**Formation sees text only by default.** The `formation` slot's `modalities` defaults to text, and
observations outside the declared set are skipped by design. Acoustic affect therefore requires
declaring the audio modality in configuration; otherwise the audio never reaches the former and the
absence looks like a model that found nothing.

**A badly grounded affect proposal is expensive.** The kernel refuses an `AFFECT` proposal whose
cue modality is not present in its source, and that refusal currently fails the whole formation
pass rather than dropping the one proposal. A model that occasionally claims a vocal cue on a
text-only observation therefore costs the other proposals in that batch too. Dropping the one
proposal and counting the drop is the intended behaviour; until it lands, keep formation prompts
conservative about cue modality.

**Affect never becomes a standing assertion by accident.** Because `AFFECT` is outside the conflict
kinds, an adapter cannot rely on supersession to clean up its own history. Two disagreeing cues stay
as two records. Anything that needs a single current answer belongs in `STATE` or `TRAIT`, with the
evidence requirements those kinds carry.

## What MindBridge will not build

There will be no second "emotional right brain" store beside SQLite, no storage of a label as a
fact, and no graph database or background service to hold affect links. Affect uses the same
records, the same evidence edges, the same versioning, and the same authority model as everything
else; if it cannot be expressed there, the representation is wrong and a parallel store would only
hide that. [VoiceMem](https://arxiv.org/abs/2608.26005) is the source of three ideas taken up here:
splitting momentary state from long-horizon disposition, cross-linking affect with people, topics,
and events, and using streaming cues to drive speculative retrieval. Three of its implementation
choices are declined: triggering only on negative affect, falling back to a VAD-quadrant label with
no semantic content, and upserting affect state without versions. Each would cost exactly the
auditability that makes an affect hypothesis correctable.
