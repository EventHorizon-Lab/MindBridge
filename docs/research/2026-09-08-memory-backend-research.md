# Memory backend research for MindBridge

**Research date:** 2026-09-08 (Asia/Shanghai)

**Scope:** primary-source review for MindBridge's P0 backend/emotion/identity, P1 agentic/omni,
and P2 stream/embodied priorities. All URLs below were checked live on the research date.

**Evidence convention:** “reported” means a paper or project author's own result. “Reproduced” is
reserved for a result supported by an independent execution or an explicit same-setting reproduction
inside another primary artifact. A public repository or script makes a claim *reproducible in
principle*; it does not establish that an independent party reproduced it.

## Decision

MindBridge should use an **authoritative event/evidence ledger with multiple derived read models**.
The normal durable write is an immutable observation or event cell; identity links, affect estimates,
semantic facts, summaries, graph edges, and procedural skills are versioned derivations that retain
source pointers. An authorized retention or erasure operation can still physically remove that
evidence. Retrieval should be planned per query and operating regime, then compile a bounded,
attributed evidence packet. Consolidation may remove details from the normal recall tier, but it must
not silently destroy source evidence or derivation lineage.

This is preferable to selecting one “memory substrate.” The strongest recent controlled evidence
reports that substrate rankings reverse between factual QA and sequential action: broad retrieval
helps QA, while excessive retrieval distracts an acting agent; graph construction can improve
relational QA but adds high write and query cost; refinement memories can be better for action but
lose factual detail ([Harness the Memory](https://arxiv.org/abs/2608.15008)). Its code was only
promised “upon acceptance” in v1, so the numbers are author-reported, but the result is consistent
with the distinct workload shapes that MindBridge already targets.

The concrete architecture is:

1. **Evidence plane:** canonical observations and event anchors, with content identity, valid/event
   time, transaction time, modality/source pointers, speaker/face observations, and model/recipe
   provenance. SQLite remains authoritative.
2. **Derived state plane:** replaceable projections for event fields, entities, identity hypotheses,
   affect episodes, preferences/traits, semantic relations, summaries, and skills. Every item has
   evidence edges, confidence, derivation identity, validity interval, and supersession status.
3. **Retrieval-control plane:** query classification and an explicit plan select dense, lexical,
   temporal, identity, graph expansion, summary, or procedural routes. The compiler budgets and
   orders evidence and exposes omissions and conflicts.
4. **Evolution plane:** background consolidation proposes new projections and routing statistics.
   Feedback may alter edge utility and retrieval policy. It cannot rewrite observations, silently
   merge people, or promote an inferred emotion or trait to ground truth.

This design adopts EM²Mem's align-then-retrieve insight, StructMem's raw-plus-abstraction hierarchy,
Zep/Graphiti's bitemporal invalidation, Mem0's explicit add/update/delete/no-op reconciliation, and
MemGPT/Letta's model-directed memory paging without inheriting any one system's full stack.

It is **not a novelty claim**. In particular, MemIR already formalizes typed evidence/cue/claim atoms,
provenance closure and a claim-centered fact interface; Agent Zero Memory already combines a
provenanced event timeline, entity graph, curated documentary memory, agentic routing and citation
locks. Any future MindBridge research claim must name and test a narrower difference.

## Source-access and claim-status ledger

| Starting source | Live status on 2026-09-08 | What can safely be concluded |
| --- | --- | --- |
| [Mnemoverse Library](https://mnemoverse.com/docs/library/) | Accessible; secondary/vendor-authored library | Useful discovery map and unusually candid benchmark commentary. It is not independent evidence for Mnemoverse product claims. |
| [LightMem repository](https://github.com/zjunlp/LightMem) | Accessible; public source and reproduction scripts | Code and instructions exist for LightMem, StructMem and FluxMem. “Coming soon” still appears for the PyPI package. Repository result tables remain author artifacts until independently run. |
| [EM²Mem arXiv v1](https://arxiv.org/abs/2609.00551v1) | Accessible; submitted 2026-09-01; paper says accepted to EMNLP 2026 Findings | Full paper is primary evidence of the method and author-reported experiments. The abstract says code “will be integrated” into LightMem; no independent reproduction was found in this pass. |
| [StructMem arXiv](https://arxiv.org/abs/2604.21748) | Accessible; submitted 2026-04-23; paper says accepted to ACL 2026 | Full method, prompts/results and repository code are available. Results are author-reported. |
| [StructMem repository note](https://github.com/zjunlp/LightMem/blob/main/StructMem.md) | Accessible despite GitHub's transient “Uh oh” shell; raw document is present | Documents event extraction, time-windowed cross-event summary storage, and LoCoMo scripts. It is implementation documentation, not independent validation. |
| [FluxMem repository note](https://github.com/zjunlp/LightMem/blob/main/FluxMem.md) | Accessible despite the same GitHub shell issue | Documents a heterogeneous evolving graph and PEMS stopping rule. The supplied note's paper link/status could not be established as a separately accessible peer-reviewed primary paper in this pass; treat it as a repository proposal. |
| [TeleAI-UAGI Awesome Agent Memory](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) | Accessible; large curated bibliography | Discovery index only. It lists 2026 multimodal work including VoiceMem, video FluxMem, Visual Agentic Memory, Omni-SimpleMem, HERMES, EventMemAgent and M2A. Entries and venue labels require primary-source checking. |
| [AgentMemoryWorld Awesome Agent Memory](https://github.com/AgentMemoryWorld/Awesome-Agent-Memory) | Accessible; claims a TMLR 2026 survey and links year files | Discovery index only. Its live page reported 1,224 papers (911 in 2026), which makes exhaustive title-level verification out of scope and also signals substantial automated-list noise risk. |
| [mnemoverse awesome-agent-memory](https://github.com/mnemoverse/awesome-agent-memory) | Accessible; five-commit curated list | Discovery index with factual one-line descriptions, not a validation corpus. Vendor owns both this list and the Mnemoverse library. |
| [MemIR](https://arxiv.org/abs/2605.25869) | Accessible arXiv v1, submitted 2026-05-25 | Direct close prior art for typed evidence/cues/claims and provenance-closed, claim-centered retrieval. Author-reported LoCoMo and BEAM-100K results; no independent reproduction established here. |
| [Agent Zero Memory](https://arxiv.org/abs/2608.29606) | Accessible arXiv v1, submitted 2026-08-30 | Direct close prior art for three parallel stores, citation-locked reads and agentic retrieval. 95.60 LongMemEval and 93.60 LoCoMo are single-run author claims, not independently reproduced results. |

The two supplied 2026 arXiv identifiers do exist; neither is a nonexistent or future placeholder.
The date matters: EM²Mem v1 was only seven days old during this review. FluxMem's repository prose
must not be cited as a peer-reviewed result. Similarly, [Harness the Memory](https://arxiv.org/abs/2608.15008)
is a recent arXiv v1 whose code is not yet linked, so it informs experiments and architecture but does
not close reproduction risk.

## Algorithm comparison

| System | Write / evolution mechanism | Retrieval mechanism | Evidence and time | Main cost or failure mode | MindBridge use |
| --- | --- | --- | --- | --- | --- |
| [LightMem](https://arxiv.org/abs/2510.18866) | Extracts and updates asynchronously/offline; compresses conversations with LLMLingua-style machinery | Dense retrieval over compact extracted memory and summaries | Timestamped conversation facts; limited identity/provenance semantics in the published core | Extraction/compression can omit evidence; comparisons depend strongly on answer/judge stack | Cost baseline and deferred-update pattern |
| [StructMem](https://arxiv.org/abs/2604.21748) | Extracts factual and relational perspectives per event; periodically clusters buffered events with semantically similar historical events and synthesizes separate summaries | Retrieves detailed memories plus cross-event abstraction | Same-timestamp entries reconstruct an event; raw episodic memories are retained beside summaries | LLM extraction/synthesis is lossy; evaluated only on four LoCoMo QA categories | Event projection and evidence-preserving consolidation |
| [EM²Mem](https://arxiv.org/abs/2609.00551v1) | Splits video into event anchors; binds caption, transcript, keyframes, fields and multi-scale temporal summaries; builds episodic and semantic graphs | Event-cell retrieval, lightweight graph expansion, LLM selector, query-specific evidence view | Explicit anchor and source grounding for derived semantic facts | Offline multimodal parsing cost; fixed segment boundaries; new and unreproduced; QA rather than live robotics | Strongest pattern for P1 omni and P2 embodied event binding |
| [FluxMem note](https://github.com/zjunlp/LightMem/blob/main/FluxMem.md) | Semantic, episodic and procedural nodes; temporary step edges; failure-driven expand/prune/reshape; offline skill distillation until PEMS stabilizes | Shared typed graph / current subgraph | Ground/distill edges express derivation, but durability and rollback semantics are underspecified in the note | Self-referential feedback can amplify bad attribution; PEMS validity and independent results unestablished | Experimental utility edges and skill projection, behind audit controls |
| [Mem0](https://arxiv.org/abs/2504.19413) | LLM extracts facts, compares similar memories, selects add/update/delete/no-op; optional entity-relation graph | Dense search, with graph augmentation in Mem0g | Timestamps exist; update process seeks temporal consistency | LLM makes destructive semantic decisions; graph extraction adds calls; paper's LoCoMo claims are author-reported | Operation vocabulary, but map delete/update to reversible proposals |
| [Zep / Graphiti](https://arxiv.org/abs/2501.13956) | Incrementally creates episodic/entity/fact graph; LLM detects contradictions; invalidates overlapping facts; dynamically extends communities | Hybrid semantic, keyword and graph search; community summaries | Four timestamps distinguish system creation/expiry from real-world valid/invalid time; episodes provide source provenance | Graph and LLM construction cost; “newest wins” transaction policy is unsafe for authority-sensitive identity | Bitemporal relations, contradiction closure, episode provenance |
| [MemGPT](https://arxiv.org/abs/2310.08560) / [Letta](https://github.com/letta-ai/letta) | Agent pages between bounded main context and external recall/archival memory through tool calls and interrupts | Agent-directed search and memory movement | Transparent memory blocks in modern Letta; paper focuses on virtual context rather than evidence lineage | Agent can waste calls or corrupt its own working/core memory; it is a full agent runtime | Retrieval planning and context paging, without coupling MindBridge to an agent framework |
| [MemIR](https://arxiv.org/abs/2605.25869) | Compiles page/span evidence, handle/time/pivot cues, claims and retrieval views; only supported claims are truth-bearing | BM25 + dense routes, RRF, projection to claims, cross-encoder rerank, LLM bundle selector | Formal support relation and complete claim association set (“provenance closure”) | Multiple extraction/read models; closure may be large; claims are author-extracted and experiments are author-reported | Mandatory prior art for any evidence-closed compiler |
| [Agent Zero Memory](https://arxiv.org/abs/2608.29606) | Builds event timeline, entity-event graph and three-level curated HDM over indexed raw sources | Intent gate, source router, three concurrent tool-using hybrid searches, integration | Provenanced items and citation lock: readers cite only opened evidence | High query latency/model use; HDM trust comes from curated notes; benchmark results are single-run and cross-system recipes may differ | Prior art for routed multi-store memory and read-time citation discipline |
| [Hindsight](https://aclanthology.org/2026.acl-demo.27/) | Retain/recall/reflect; facts and beliefs with confidence and evolving opinions | Parallel vector, keyword, graph and temporal filtering | Temporal entity graph and fact/belief separation | Demo paper's quality figures remain authors' results | Prior art for evolving belief state and affect-adjacent confidence |

### Event binding is the best common denominator

EM²Mem's central move is to make an event anchor a shared address, rather than memory content. A
cell binds visual caption, transcript/dialogue, representative keyframes, action/object/topic/scene/
entity fields, and multi-scale temporal views. Episodic graph edges connect concrete events;
semantic edges summarize recurring habits, preferences and relationships but point back to events.
This directly fits an embodied companion: “who appeared,” “what was said,” “what facial or vocal cue
was observed,” and “what happened next” can remain one event without collapsing every modality into
text.

For MindBridge, event boundaries should be adaptive and nested rather than fixed 30-second clips.
Speech turns, visual scene changes, interaction boundaries, explicit application markers, and maximum
duration can close an event. The durable anchor should record uncertainty and permit later boundary
views; otherwise a wrong segmentation becomes a permanent retrieval error.

### Structure should remain a projection

StructMem reports 76.82 overall LoCoMo judge accuracy with 1.937 million construction tokens and
1,056 API calls, compared with Mem0g's 68.44 overall, 35.825 million tokens and 53,514 calls in its
reported setup.
Its useful idea is not the headline score. It batches recent events, retrieves semantically similar
historical events, restores each seed's full same-time context, and writes a separate cross-event
summary while retaining episodic detail. Its authors attribute graph cost to four cascading LLM
operations per event and growing deduplication overhead.

MindBridge should therefore avoid a graph database as the primary truth. Store typed nodes and edges
in SQLite tables as rebuildable projections initially. Each edge needs `source_memory_ids`, producer
and recipe identity, confidence, valid interval, transaction interval, and status. This makes graph
repair a normal correction or rollback operation and lets dense/lexical retrieval continue if graph
formation is disabled or stale.

### Identity is an evidence-resolution problem

Most general memory systems reviewed here treat a named entity as identity, which is inadequate for
biometric enrollment. VoiceMem is a relevant exception: it performs streaming speaker identification
and can retain voiceprints, acoustic embeddings or raw waveforms, but its paper does not specify the
consent, merge/split, calibrated false-link and erasure contract MindBridge needs. Entity extraction
(“Alice”) and vector similarity are not identity proof. MindBridge should keep:

- an observation layer (face track, voice segment, asserted name, co-occurrence, device/session
  context), each with modality-specific quality and provenance;
- a hypothesis layer linking observations to an identity with independent support counts and
  calibrated confidence;
- a claim layer for names, relationships and preferences, with speaker/asserter and validity time;
- explicit human-authorized enrollment, merge, split, rename, unlink and erasure operations.

Graphiti's bitemporal distinction is particularly useful: “Yon's favorite changed on Monday” and
“the system learned that on Thursday” are different times. Its default policy of prioritizing newer
transactions when invalidating contradictions should not decide identity or affective truth. Source
authority, independent corroboration, and explicit correction must outrank arrival time.

### Emotion must stay episodic, uncertain, and context-bound

Affect models produce observations, not durable facts about a person. Persist the cue (prosody,
facial expression, words, posture), event context, model/revision, score distribution, and consent
scope. A derived affect episode may aggregate cues over a short interval. Longer-term mood or trait
state requires repeated evidence, decay, counterevidence, and a visible inference label. Retrieval
should distinguish “the user said they were sad,” “the model estimated sadness,” and “the agent
responded as though sadness were present.” This prevents a mistaken classifier output from hardening
into identity.

The response outcome is valuable feedback, but it should tune retrieval utility or a response-policy
projection, not retroactively relabel the user's emotion. This is the safe version of FluxMem's
failure-driven rewiring: update an edge's observed usefulness, preserve the original evidence and the
decision trace, and allow rollback.

### P0/P2 close work: affect, voice and lifelong companionship

[VoiceMem](https://arxiv.org/abs/2608.26005) is the strongest direct comparator for MindBridge's
emotion-plus-stream path. It separates an informational schema/entity graph from a persona graph with
intrinsic disposition nodes and entity-bound affect nodes, performs short-horizon affect attribution
per turn and long-horizon consolidation per session, and overlaps ASR, matching, speaker identity,
embedding and retrieval with the VAD window. The [public repository](https://github.com/xzf-thu/VoiceMem)
includes audio/face-oriented components. This validates the architectural relevance of separating
situational affect from persistent persona and beginning speculative retrieval before turn end. It
does not establish that model-estimated traits are calibrated identity facts; MindBridge's evidence,
consent and rollback boundaries remain material differences. The current arXiv abstract and rendered
v1 HTML disagree on the persona aggregate gain (+4.29 versus +1.89), so neither figure is treated as
stable here. The paper's 134 ms retrieval result and other gains remain author-reported.

[MemEmo](https://arxiv.org/abs/2602.23944) evaluates emotional extraction, updating and memory QA;
[A-MBER](https://arxiv.org/abs/2604.07017) tests current-affect inference with historical evidence,
including modality degradation and insufficient-evidence cases; and
[LifeSide](https://arxiv.org/abs/2606.04660) supplies large simulated Memory-Emotion-Environment loops
covering user understanding, privacy control and companionship. They are valuable, complementary
evaluation instruments, but synthetic dialogue labels do not validate real sensor calibration or
biometric identity behavior.

[Omni-SimpleMem](https://arxiv.org/abs/2604.01007) uses novelty gates for frames, audio and text;
multimodal atomic units with summary, embedding, raw-content pointer, timestamp, modality and links;
hot/cold storage; hybrid dense/sparse/graph retrieval; and budgeted progressive expansion from summary
to full/raw content. These closely parallel MindBridge's lazy assets and compiled context. Its roughly
50 autoresearch iterations used the same two benchmark harnesses as the optimization signal, making a
strict untouched test split and complete experiment ledger essential before interpreting gains.
The arXiv title/version surfaces have varied between “OmniMem” and “Omni-SimpleMem”; this report cites
the current v2 title and identifier rather than assuming the name was stable. Its reported 5.81
queries/second uses eight workers, while the table's single-worker OmniMem and SimpleMem figures are
1.05 and 1.68 respectively, so the headline is not a matched-concurrency backend speedup. Mem-Gallery
top-k and budgets also vary by gold question category and the appendix supplies category-specific
formatting. That is valid only when such category labels are part of the declared inference interface;
MindBridge must not feed hidden benchmark labels into its product router. The reported improvement
mixes bug fixes, prompts, architecture and model changes and therefore does not isolate retrieval's
causal contribution.

## Retrieval planning and context compilation

A single top-k API is insufficient. Plan from query intent and requested answer guarantees:

| Query class | First route | Conditional expansion | Stop condition |
| --- | --- | --- | --- |
| Direct fact / preference | lexical + dense over active semantic claims | source events if confidence/conflict is weak | one supported active answer or explicit disagreement |
| Temporal / “what changed?” | valid-time and transaction-time filters over events/claims | predecessor/successor and supersession edges | timeline covers referenced interval |
| Identity / “who?” | identity index and scoped claims | corroborating face/voice/name observations | threshold plus policy/consent gate; abstain otherwise |
| Emotional / interpersonal | recent affect episodes and explicit user statements | causal/interaction neighbors and response outcomes | bounded interval with uncertainty retained |
| Multi-hop explanation | dense/lexical seeds | bounded graph expansion, then hydrate source events | evidence coverage or hop/token budget |
| Agent action / procedure | high-utility procedural projection | one or two matched success/failure episodes | minimal action-critical packet |
| Omni / embodied scene | event-cell retrieval over modality fields | adjacent events, shared entities, selected source media | event-level evidence coverage within media/token budget |

The compiler should return records plus a trace: plan, candidates by route, score components, graph
expansions, stale IDs dropped, conflicts, time filters, selected evidence, omitted evidence and budget.
It should use deterministic ordering and stable formatting where possible to improve auditability and
prompt-cache reuse. An LLM selector can operate inside a hard candidate and token budget, but it
should not be the only path to an answer.

### Close prior art and the defensible research delta

[MemIR](https://arxiv.org/abs/2605.25869) is the closest published method to an evidence-closed
compiler. It defines page and verbatim span atoms; handle, time and pivot cues; truth-bearing claim
atoms; and retrieval views. Every non-page derived atom must have a non-empty support set. Sparse and
dense hits are projected into their associated claims, and Equation 3 defines each bundle's
“provenance closure” as retrieved evidence plus the claim's complete association set. A cross-encoder
reranks bundles, an LLM selects at most a configured number, and a normalized fact interface instructs
the answer model to abstain on insufficient evidence. Broad typed provenance, closure, normalized
fact records, and abstention are therefore prior art and must not be presented as MindBridge firsts.

[Agent Zero Memory](https://arxiv.org/abs/2608.29606) is equally close at system level. It indexes all
raw sources lexically and densely, distils a merged event timeline, builds an entity-event graph, and
keeps curated facts in a three-level Hierarchical Documentary Memory. An intent gate and source router
launch three concurrent agentic searches. Its citation lock permits a reader to cite only evidence it
opened. The authors report 95.60% LongMemEval and 93.60% four-category LoCoMo, but explicitly describe
single runs under official LLM judges. Those numbers do not establish independent superiority across
different vendors' configurations.

[xmemory](https://arxiv.org/html/2604.27906v1) stages schema-grounded extraction as
object, field and value decisions with local validation. Its fields and unknown values come from a
declared workload rather than open-ended schema-free memory. The evaluations are deliberately
structured and vendor-run, and the authors say their comparisons are not controlled component
ablations. The reported 3.15× token saving applies symbolic assumed write/read costs rather than
measured benchmark billing. It is relevant prior art for typed slots, but it neither reproduces
MindBridge nor shows that free-form predicate aliases are reconciled safely.

[AetnaMem](https://huggingface.co/blog/telcom/aetnamem) is close prior art for auditable local
provenance: SQLite persistence, mandatory source-linked derivations, deterministic keyed
supersession, per-subject hash chains and an independent standard-library verifier. Unkeyed
contradictions remain unresolved, and externally anchored heads are still required to detect history
replacement. Its 33/33 result is the authors' own MemoryStackBench gate suite, not QA or leaderboard
evidence. Origin classification and approval authentication remain host responsibilities. This
supports independent artifact verification without requiring an action gateway or implicit user
scope in MindBridge.

A potentially defensible MindBridge delta is narrower: **deterministic atomic context selection under
a hard budget, subject to temporal, identity and consent constraints, that keeps candidate-bounded
conflicting claims together and charges a shared source memory once, with no extra query-time model**.
This remains a hypothesis until directly compared with MemIR's cross-encoder-plus-LLM selector and
Agent Zero's agentic searches. The current v1 implementation does not prove semantic entailment,
minimum-sufficient evidence, independent corroboration, or globally complete counterevidence.

Alternative and joint support are classical provenance problems, not a MindBridge invention.
[Green, Karvounarakis and Tannen](https://web.cs.ucdavis.edu/~green/papers/pods07.pdf) show that flat
why-provenance loses how inputs combine and use provenance polynomials to preserve those
distinctions. [de Kleer's ATMS](https://www.dekleer.org/Publications/An%20Assumption-Based%20TMS.pdf)
tracks assumption environments and justifications. The essential example is
`support(claim) = (A AND B) OR C`: removing A leaves C as the valid justification; it must never turn
`A AND B` into B. These clauses express logical support, not calibrated truth probability.
MindBridge's narrower engineering contribution is their transactional integration with batch
formation, typed memory, correction, deletion, rollback, and bounded compilation, rather than the
Boolean structure itself.

“Complete provenance closure” also cannot be blindly required. In an unbounded corpus, expanding all
associations can exceed every context budget; broad or noisy claims can cause closure explosion; a
single source may support many claims and be repeatedly charged; contradictory claims need joint
selection; and stale or unauthorized evidence may be ineligible even if structurally associated.
The frozen compiler-v3 selector treats every `evidence_id` as conjunctively required. It roots
closure expansion in the finite ranked
candidate window and batch-hydrates referenced sources outside that window under bounded traversal
and identical scope. It admits each anchor and its eligible sources atomically, and expands
candidate-bounded functional-lineage conflicts so both representatives must fit. Functional here
matches the storage contract: STATE assertions and user-stated TRAIT assertions have one standing
value per lineage. Accumulating RELATION values and model-inferred TRAIT values are not automatically
contradictory. A source outside the requested
spatial, identity, valid/known-time, type, confidence, freshness or consent bounds makes the derived
anchor unavailable. Shared source *memory records* are charged once across admitted closures; media
attached separately to distinct records can still be conservatively charged more than once.

This strict behavior prevents an answer from rendering a derived claim without every declared source,
but it can suppress a valid claim when any one redundant source is stale, unauthorized, or too large.
The later schema-17/18 work represents OR-of-AND support clauses durably, but that persistence work
is outside the frozen compiler-v3 benchmark. Current main also records the complete cited set from
one new general `CONSOLIDATE` operation as one AND clause, while separate operations remain OR
alternatives; that post-v3 change did not participate in the large-model runs. The compiler still
includes the conservative union of surviving support groups; choosing one sufficient group under
budget while preferring independent groups remains deferred. If the eligible bounded packet is
insufficient, the compiler should abstain and expose the missing requirement. This is bounded
declared-support coverage, not an assertion that the entire corpus has been closed.

[Sufficient Context](https://arxiv.org/abs/2411.06037) reinforces the last point: its authors report
that standard QA datasets contain many insufficient-context cases, that models still answer some of
them correctly from parametric knowledge, and that models often hallucinate rather than abstain.
Consequently, answer accuracy alone cannot validate an evidence compiler. Evaluate context
sufficiency, grounded entailment and selective-answer coverage separately.

The “Harness the Memory” top-k ablation reports why planning matters. Increasing retrieval breadth
helped LoCoMo-style QA but harmed ALFWorld sequential decisions as model attention shifted from the
current observation toward retrieved text. MindBridge should measure a route's marginal utility,
not globally tune one `top_k`.

## Lossy compression and memory evolution

Compression has three distinct meanings and must be measured separately:

1. **Representation compression:** captions, transcripts, embeddings and structured fields derived
   from media. The original asset/source pointer and derivation recipe survive.
2. **Semantic consolidation:** summaries, traits, routines and skills synthesized across events.
   Evidence coverage and contradiction status survive; the derived item can be superseded.
3. **Recall suppression:** detailed events leave the default retrieval tier because of age, low
   utility or successful consolidation. They remain directly addressable and auditable until an
   authorized retention or erasure operation physically removes them.

Measure compression ratio alongside evidence recall, unsupported-claim rate, contradiction
preservation, identity-link error, affect calibration, storage growth, write/read/management tokens,
model calls, energy where available, and p50/p95 latency. A smaller prompt is not a win if the omitted
evidence makes correction or attribution impossible.

[Retain or Consolidate?](https://arxiv.org/html/2607.17545v1) provides a useful bounded warning:
under tight context budgets consolidation can improve coverage, while replacing raw evidence can
lose answer-bearing detail. Its strongest independent full-history result was a fixed policy; the
authors report that their pre-generation router did not beat it. The controlled main experiment gives
every action oracle evidence and exposes benchmark query type, so it does not establish a deployable
end-to-end memory router. MindBridge should therefore avoid oracle-category routing and naive
replacement summaries, and test consolidation with raw evidence, retrieval, and total budget in the
loop.

Evolution should be proposal-driven: observe outcomes; propose reinforce/suppress/consolidate/
supersede/split operations; validate source and policy constraints; commit the operation log and
SQLite projection; update the rebuildable index through the durable outbox. Feedback is scoped to a
task and consumer because a memory useful to casual conversation may be harmful in a safety-critical
action plan.

Three recent primary sources sharpen the validation boundary:

- [TrustMem](https://arxiv.org/html/2606.25161v1) uses a frozen LLM transition verifier for coverage,
  preservation, and faithfulness over touched entries, then transition-ranked GRPO to learn updates.
  This is semantic supervision and training; MindBridge's dependency and closure checks are
  structural and must not be described as equivalent.
- [When Not to Write Memory](https://arxiv.org/html/2607.02579v1) studies dependency-aware promotion,
  scope, and counterevidence. Its external 133-row adjudication rejected every automatic promotion,
  including 11 gate positives, while a simple review-all-correlated policy remained competitive.
  Repetition is not independent confidence, and rejecting everything is not useful success.
- [When Stale Constraints Go Unchecked](https://arxiv.org/html/2608.25553v3) shows that linked
  provenance does not ensure the current source is inspected. Its forced-critical intervention uses
  an oracle path and unsolicited-first delivery; its target-blind rule has bounded toy-store support.
  Neither establishes a deployable general scheduler.

## Benchmark evidence and leakage controls

### What recent harness work adds

- [Harness the Memory](https://arxiv.org/abs/2608.15008) compares 11 implemented substrates in seven
  families under three backbones and four suites, with 26 performance/efficiency metrics. It reports
  that 62% of 52 surveyed system-benchmark uses concentrate on LoCoMo/LongMemEval, only 21% report an
  efficiency metric, and 81% use GPT-family backbones alone. The authors exclude production Mem0,
  Zep and MemGPT from the controlled table because their auxiliary-model budgets differ materially.
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493) and its
  [official repository](https://github.com/xiaowu0162/LongMemEval-V2) provide 451 hand-curated
  questions over multimodal web-agent histories of 100–498 trajectories and 25M–115M tokens. It
  tests static state, dynamic state, workflows, local gotchas and premise awareness. It is much
  closer to P1 agentic memory than chat QA, though still web/enterprise rather than robotics.
- [MemoryArena](https://arxiv.org/abs/2602.16313) uses 766 interdependent subtasks in memory-agent-
  environment loops. Its main value is causal dependence across sessions: earlier actions and
  feedback change what later action should be taken.
- [MemoHarness](https://arxiv.org/abs/2607.14159) treats context assembly, tool use, memory,
  orchestration, decoding and output handling as a searched harness configuration. Reference outputs
  are permitted in search-time scoring but excluded at test time. This is a useful design pattern and
  a leakage boundary that must be logged.
- [PersonaMem](https://openreview.net/forum?id=6ox8XZGOqP) evaluates evolving user profiles over
  multi-session interaction. It is relevant to personalization but does not validate face/voice
  identity or emotion inference.
- For P1/P2, use route-specific multimodal suites: EM²Mem evaluates EgoLifeQA, Ego-R1 Bench and
  Video-MME(L); LongMemEval-V2 supplies multimodal agent trajectories. No reviewed benchmark jointly
  validates live streaming, embodied identity resolution, affect calibration, erasure, provenance
  and crash recovery. MindBridge needs dedicated contract/behavior cases for those properties.

### Required anti-leakage protocol

Freeze and publish a run manifest containing dataset artifact hash and license, exact subset and
question categories, train/dev/search/test split, memory-bank construction inputs, prompts, model and
revision, embedder/reranker, top-k and token budget, concurrency, cache state, judge and rubric,
random seeds, code commit, and failures/retries. Save per-question retrieved IDs, compiled context,
answer and judge output.

Enforce these boundaries:

- Gold answers, gold evidence IDs, labels, and timestamps inferred from them may enter scoring only.
  They cannot enter extraction, indexing, query rewriting, retrieval, consolidation, routing, or
  stopping. A benchmark's public `T_test` is different: it defines the official source-availability
  firewall and normal query reference time, so a preregistered runner may use it to ingest the
  strictly earlier source prefix. It must not derive a cutoff or route from the answer or gold
  evidence.
- If a harness searches configurations using labeled cases, those cases and near-duplicates cannot
  appear in final evaluation. Record search budget and all attempted configurations.
- Do not compare retrieval recall with end-to-end answer accuracy, or a four-category LoCoMo subset
  with all five categories. Report denominators and exclusions.
- Separate ingestion, management, retrieval, answer-generation and judge costs. Cached embeddings or
  prebuilt graphs must be declared and amortized over a stated query volume.
- Report judge robustness with at least a deterministic metric where applicable, blinded/manual
  audit samples, and judge-model/prompt sensitivity. StructMem's own appendix shows material score
  changes between GPT-4o-mini, Qwen2.5-32B-Instruct and DeepSeek-V3.2 judges.
- Prevent self-preference by avoiding the evaluated answer model as the only judge. Randomize answer
  ordering for pairwise judgments and retain raw verdicts.
- Track dataset contamination as an unresolved risk when training data are undisclosed. A high score
  is not evidence that retrieval caused the answer; ablate memory, shuffle retrieved evidence, and
  measure evidence-grounded answer deltas.
- Require clean-room reruns or independently verified artifacts before describing a result as
  reproduced. Public scripts and author-committed JSON are lower levels of evidence.

The supplied Mnemoverse library itself documents an earlier oracle leak in its “Building Memory That
Scales” article and highlights judge-prompt variance. That candor is useful, but the site is a vendor
source. The benchmark protocol above relies on inspectable artifacts and primary benchmark contracts,
not its comparative rankings.

## Priority-ordered implementation implications

### P0: backend, emotion, identity

1. Preserve the existing event/evidence kernel: it already carries valid/recorded time, evidence
   edges, basis/confidence, identity context and operation lineage, with one physical `data_dir` per
   instance. Close audited gaps incrementally rather than replacing this contract: validate
   validated schema-18 support-clause persistence, evaluate bounded proof-group selection beyond
   the conservative union before changing it, refine asserter/producer semantics where absent, and
   add direct evidence-sufficiency diagnostics.
2. Keep the existing identity and affect projections conservative. Extend tests for mistaken-link rollback,
   conflicting names, cross-modal corroboration, uncertain/absent identity, emotion disagreement,
   consent withdrawal, export and physical erasure.
3. Preserve SQLite authority and Zvec rebuildability. Every derived index mutation follows the
   durable outbox; stale IDs are filtered through SQLite hydration.
4. Add evidence coverage and unsupported-derivation metrics before optimizing QA accuracy.

### P1: agentic and omni

1. Introduce a typed internal retrieval plan and retain the existing simple public operation. Route
   by query/task regime, with bounded iterative retrieval and full trace output.
2. Build event cells over ordered arbitrary modalities. Keep media-native pointers and selected
   keyframes/audio spans available for verification; text summaries are projections.
3. Add procedural/response-policy projections from successful and failed episodes. Use outcome
   feedback to adjust task-scoped utility, with minimum support, decay and rollback.
4. Evaluate on both dialogue QA and action loops. LongMemEval-V2/MemoryArena-style tasks should sit
   beside LoCoMo/LongMemEval, never replace them or be merged into one headline score.

### P2: stream and embodied

1. Close adaptive event windows from stream signals and commit only final observations; speculative
   prefetch never becomes evidence. Measure late packets, reordered timestamps, dropped media,
   crash/restart and concurrent readers.
2. Build multi-scale temporal views asynchronously with explicit freshness. Query-time results state
   whether consolidation or modality enrichment is pending.
3. Add embodied scenarios where identical semantics occur with different people, rooms, objects,
   times and affect cues. Evaluate identity precision/abstention and event-level evidence recall,
   along with action success, latency, resource use and recovery.

## Falsifiable experiments before adopting complex graph evolution

1. **Event-cell ablation:** compare flat chunks, transcript-only events, multimodal event cells and
   event cells plus bounded graph expansion under identical embeddings, answer model and budget.
2. **Raw-plus-summary ablation:** compare raw retrieval, destructive summaries, and separate
   evidence-linked summaries on temporal/multi-hop accuracy, evidence recall and correction tasks.
3. **Regime router:** train/tune only on a development split to choose factual, temporal, identity,
   affect, procedural and omni routes; compare with a single global hybrid retriever on held-out QA
   and action tasks.
4. **Feedback safety:** inject wrong success/failure attribution and measure whether utility-edge
   updates can be detected, bounded and rolled back without changing source memories.
5. **Identity stress:** test lookalike voices/faces, renamed identities, contradictory assertions and
   cross-modal false matches; report false-link rate and abstention, not just recall.
6. **Streaming amortization:** report construction cost, freshness lag and break-even query count for
   multi-scale summaries and graphs. EM²Mem reports large query-time savings, but offline construction
   must be counted for a companion with many events and few repeated questions.

Adopt graph evolution only if it improves evidence-grounded outcomes after total write, management,
read and recovery cost is counted, while preserving rollback and identity safety. Otherwise,
event-local structured fields plus lexical/dense/temporal retrieval are the stronger backend.

## Acceptance matrix for deterministic evidence closure

| Property | Required observation |
| --- | --- |
| Closure | For every admitted anchor `a`, every recursively declared and eligible evidence memory in `E*(a)` appears in the bundle. |
| Conflict | Every contradictory representative of a functional STATE or user-stated TRAIT lineage known inside the ranked candidate window, and its own support, is admitted atomically or the conflicting group is withheld. Multi-valued RELATION and model-inferred TRAIT assertions accumulate. This does not claim corpus-wide conflict discovery. |
| Budget | Item, character and media limits are computed over the union of memory IDs. Reused source memories have zero marginal item/text cost; duplicate derivations do not raise confidence. |
| Eligibility | Every closure member independently satisfies spatial/identity scope, valid/known time, requested memory type, confidence, freshness and consent rules. Dependencies carry zero retrieval score because they were hydrated as support rather than ranked for the query. |
| Failure | Missing, cyclic or node-bounded support produces a truthful `evidence_unavailable` diagnostic and no dangling affirmative anchor. A passed latency deadline skips optional enrichment and conflict detection visibly; it must not imply those checks succeeded. |
| Compatibility | Untyped singleton input preserves the frozen legacy output byte-for-byte. |
| Epistemic limit | An evidence link is a declared provenance relation. Closure does not prove semantic entailment, independent corroboration, minimum sufficiency or correctness. |
| Evaluation | Report answerable coverage, wrong answers, correct refusals, context size and source coverage together. A lower wrong-answer rate obtained only by refusing more questions is not an improvement. |

The frozen compiler-v3 rule can be stated precisely. For each eligible anchor `a`, construct `C(a)` as
the least finite fixed point of its declared evidence dependencies plus conflicting representatives
of functional lineages visible in the bounded candidate window. Missing, ineligible, cyclic or bounded
dependencies make that anchor infeasible. Traverse anchors in retrieval rank order and admit
`C(a)` only when its union with the selected memory IDs fits the item, rendered-character and media
limits. A memory ID already selected therefore has zero marginal materialization cost. The union of
dependency-closed sets remains closed, but this greedy order has no optimal-coverage guarantee.
Confidence remains the stored value; projections do not manufacture corroboration.

Stop or roll back the selection change if it reduces broad-QA performance beyond measured run noise,
or if source-closure growth dominates practical context budgets without compensating grounded-quality
gains. Preserve the diagnostics in either case; do not turn a safety tradeoff into a state-of-the-art
claim.

The implementation now passes the compile request's `known_at` and `valid_at` scope into named and
provisional actor resolution, preventing a later name from decorating an earlier historical bundle.
This is a separate temporal-identity fix, not an evidence-closure novelty claim. Current consent
still restrains actor disclosure in historical compilation, while the underlying raw observation
remains eligible under the existing policy. The naming assertion used for actor decoration can sit
outside the hit budget under the existing contract, so the bundle still does not claim a complete
identity proof.

## Primary references

- Chen, Y., et al. (2026). [EM²Mem: Event-Centric Multimodal Memory for Large Language
  Models](https://arxiv.org/abs/2609.00551v1). arXiv v1; paper reports EMNLP 2026 Findings acceptance.
- Xu, B., et al. (2026). [StructMem: Structured Memory for Long-Horizon Behavior in
  LLMs](https://arxiv.org/abs/2604.21748). arXiv v1; paper reports ACL 2026 acceptance.
- Huang, W.-C., et al. (2026). [Harness the Memory: A Holistic Evaluation of Memory Substrates in
  Memory Agents](https://arxiv.org/abs/2608.15008). arXiv v1; code unavailable at review time.
- Wu, D., et al. (2026). [LongMemEval-V2](https://arxiv.org/abs/2605.12493) and
  [official code](https://github.com/xiaowu0162/LongMemEval-V2).
- [MemoryArena](https://arxiv.org/abs/2602.16313) (2026), official paper.
- [MemoHarness](https://arxiv.org/abs/2607.14159) (2026), official paper.
- Chhikara, P., et al. (2025). [Mem0: Building Production-Ready AI Agents with Scalable Long-Term
  Memory](https://arxiv.org/abs/2504.19413) and [official repository](https://github.com/mem0ai/mem0).
- Rasmussen, P., et al. (2025). [Zep: A Temporal Knowledge Graph Architecture for Agent
  Memory](https://arxiv.org/abs/2501.13956) and [Graphiti source](https://github.com/getzep/graphiti).
- Packer, C., et al. (2023/2024). [MemGPT: Towards LLMs as Operating
  Systems](https://arxiv.org/abs/2310.08560) and [Letta source](https://github.com/letta-ai/letta).
- Wu, D., et al. (2025). [LongMemEval](https://arxiv.org/abs/2410.10813) and
  [official repository](https://github.com/xiaowu0162/LongMemEval).
- Maharana, A., et al. (2024). [LoCoMo](https://arxiv.org/abs/2402.17753) and
  [official repository](https://github.com/snap-research/locomo).
- Salemi, A., et al. (2025). [PersonaMem](https://openreview.net/forum?id=6ox8XZGOqP).
- Jin, Z., et al. (2026). [Mitigating Provenance-Role Collapse in Long-Term Agents via Typed Memory
  Representation (MemIR)](https://arxiv.org/abs/2605.25869).
- Wu, M., & Zhu, P. (2026). [Agent Zero Memory: Provenance-Aware Long-Term Memory for LLM
  Agents](https://arxiv.org/abs/2608.29606).
- Latimer, C., et al. (2026). [Hindsight: Structured Agent Memory that Retains, Recalls, and
  Reflects](https://aclanthology.org/2026.acl-demo.27/). ACL 2026 System Demonstrations.
- Joren, H., et al. (2024). [Sufficient Context: A New Lens on Retrieval Augmented Generation
  Systems](https://arxiv.org/abs/2411.06037).
- Xie, Z., et al. (2026). [VoiceMem: Streaming Dual-Brain Memory for Real-Time
  Interaction](https://arxiv.org/abs/2608.26005) and [official code](https://github.com/xzf-thu/VoiceMem).
- [MemEmo](https://arxiv.org/abs/2602.23944), [A-MBER](https://arxiv.org/abs/2604.07017), and
  [LifeSide](https://arxiv.org/abs/2606.04660) (2026), affective and companion-memory benchmarks.
- Liu, J., et al. (2026). [Omni-SimpleMem / OmniMem](https://arxiv.org/abs/2604.01007) and
  [official code](https://github.com/aiming-lab/SimpleMem).
