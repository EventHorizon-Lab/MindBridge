# Competitive memory-system review

This review records the evidence used to evolve MindBridge's embodied-memory contract. It is dated
2026-09-01. Paper claims and implementation claims are kept separate: a published architecture is
not treated as production code, and a repository snapshot is not treated as a benchmark result.

## Scope and method

The first tier was reviewed paper-first and then source-first. The second and third tiers were used
to test whether the resulting design generalizes beyond one architecture.

| System | Primary evidence | Source snapshot inspected |
| --- | --- | --- |
| ABot-AgentOS | [paper](https://arxiv.org/abs/2607.10350); the paper-linked repository returned 404 on the review date | A previously indexed release notice said code/resources were being prepared; no implementation maturity claim is possible |
| M3-Agent | [paper](https://arxiv.org/abs/2508.09736), [official repository](https://github.com/ByteDance-Seed/m3-agent) | [`0e3e419`](https://github.com/ByteDance-Seed/m3-agent/commit/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c) |
| VoiceMem | [paper](https://arxiv.org/abs/2608.26005), [official repository](https://github.com/xzf-thu/VoiceMem) | [`f99569e`](https://github.com/xzf-thu/VoiceMem/commit/f99569ed0718543c44da44ae833aeb5c3659cdfc) |
| eMEM | [paper](https://arxiv.org/abs/2606.03374), [official repository](https://github.com/Automatika-Robotics/eMEM) | [`82e3da6`](https://github.com/Automatika-Robotics/eMEM/commit/82e3da61cf710c4379e0cd7bf7a6a21710caaa96) |
| MIRIX | [paper](https://arxiv.org/abs/2507.07957), [official repository](https://github.com/Mirix-AI/MIRIX) | [`8cb06a6`](https://github.com/Mirix-AI/MIRIX/commit/8cb06a62bbb7c478beb33dd4f2815696a72df482) |
| Graphiti/Zep | [paper](https://arxiv.org/abs/2501.13956), [official repository](https://github.com/getzep/graphiti) | [`8b61fce`](https://github.com/getzep/graphiti/commit/8b61fce9f003cc3a05e246f6201f8b782dfe6546) |
| Mem0 | [paper](https://arxiv.org/abs/2504.19413), [official repository](https://github.com/mem0ai/mem0) | [`71fba8d`](https://github.com/mem0ai/mem0/commit/71fba8d46436f88569d600f81a55208c38ad30b5) |
| TeleMem | [paper](https://arxiv.org/abs/2601.06037), [official repository](https://github.com/TeleAI-UAGI/TeleMem) | [`24630b1`](https://github.com/TeleAI-UAGI/TeleMem/commit/24630b107ffab670952358954cab92f5a914d4af) |

The review traced ingestion, conflict resolution, persistence, retrieval, deletion, and streaming
code paths. Published scores are useful orientation, but they are not directly comparable because
datasets, model endpoints, retrieval budgets, hardware, and evaluators differ. MindBridge must
reproduce any quality claim through its public SDK before using it to select a default.

## Executive assessment

The largest pre-iteration gap was not another vector index. It was the absence of a trustworthy
memory-formation plane between raw multimodal observations and retrieval. Entity, event, state,
source, confidence, validity, conflict, and replacement semantics existed only as free metadata, so
the kernel could neither validate nor maintain them.

The implemented direction is deliberately smaller than a graph database or a multi-agent memory
service:

```text
immutable observation
        |
        v
optional FormationBackend ----> typed, source-grounded proposals
        |                                  |
        |                         validation and trust gates
        v                                  v
SQLite evidence + bitemporal semantic versions (authoritative)
        |
durable outbox
        v
Zvec retrieval projection (derived and rebuildable)
```

This closes the semantic, temporal, basic metric-spatial, generalized snapshot-streaming, and
evidence-grounded affect/personality gaps without adding a database service, worker queue, global
plugin registry, or product-specific API fork.

## Capability comparison

The table describes demonstrated paper or inspected-source behavior, not marketing breadth.

| Capability | Strong references | MindBridge after this iteration | Remaining ceiling |
| --- | --- | --- | --- |
| Automatic formation | ABot typed source-grounded graph; VoiceMem dual-brain extraction; Mem0 fact extraction; MIRIX specialized agents | Optional `FormationBackend`; typed entity, event, state, relation, affect, trait, and response-policy proposals; deterministic IDs and evidence links | Formation quality needs benchmarked local and remote adapters; no autonomous rule evolution |
| Conflict and forgetting | Graphiti relation invalidation; ABot supersession; VoiceMem left-brain operations | Overlapping state assertions are retired and split by valid time; evidence deletion recomputes or removes derived projections; expiry is query-visible, not destructive | No background consolidation scheduler or learned forgetting policy |
| Temporal model | Graphiti `valid_at`, `invalid_at`, `created_at`, `expired_at` | Separate valid time and recorded/retired transaction time; `valid_at` and `known_at` retrieval; historical corrections and A→B→A state evolution | No temporal relation planner or natural-language interval extraction beyond the former/parser |
| Spatial memory | eMEM R-tree radius/nearest; ABot typed places and spatial edges | Typed Cartesian frame, observer/subject anchor, position, normalized quaternion, uncertainty, and exact same-frame radius search | No frame transform tree, topological map, occupancy, trajectory, or geodesic query |
| Live multimodal input | VoiceMem VAD/ASR partial/audio stream; M3 online video/audio clips | Associated updates drive speculative retrieval; `AsyncAudioStream` accepts canonical speech packets; `AsyncVisionStream` accepts encoded frames, visual partials, and scene boundaries; exact finals persist | Device SDKs still normalize provider deltas and pixels; measured-sensor packet protocols remain adapter work |
| Emotion and personality | VoiceMem right brain; M3 semantic personality/emotion; MIRIX core memory | Affect retains cue modality, valence, arousal, confidence, time, and evidence; inferred traits stay hidden until two independent sources support them | No companion-policy evaluator, calibrated multimodal fusion model, or user-facing profile editor |
| Multi-hop structure | ABot and Graphiti typed graphs; Mem0 platform entity graph | Typed lineage and source/evidence links in SQLite; semantic and lexical retrieval | No graph expansion; add only after measured top-k evidence displacement |
| Embedded durability | eMEM SQLite plus derived indices | SQLite is authoritative; formation, evidence, versions, embeddings, and outbox enqueue commit atomically; Zvec is rebuildable | Device-level crash and power-loss campaigns remain to be published |
| Developer experience | Mem0 short add/search loop; TeleMem Mem0-compatible surface | Existing `add`, `search`, and `ask` remain; context and scope are optional across Python, REST, MCP, and CLI | Generated client examples and a compatibility/versioning guide would improve adoption |

## First-tier findings

### ABot-AgentOS

ABot is the strongest architectural target for embodied memory. Its Universal Multi-modal Graph
Memory represents source-grounded entities, events, places, sessions, evidence, temporal
relations, spatial context, and provenance as typed nodes and edges. Its write path selectively
forms memory, merges duplicates, and supersedes stale state with temporal edges rather than erasing
the original source. Its failure-driven self-evolution loop diagnoses an error, proposes a bounded
JSON asset, compiles it, and promotes it only after regression gates on later splits.

The distinction matters: self-evolution is not simply increasing the score of frequently retrieved
records. It changes retrieval or memory behavior only through gated, replayable assets. MindBridge
adopts source grounding, typed semantics, supersession, and regression-gated development. It does
not adopt ABot's planner, skill runtime, verifier, edge/cloud control plane, or generated evolution
assets because those belong to an AgentOS rather than an embedded memory kernel. The paper-linked
repository was unavailable on the review date and does not provide an implementation that can be
audited for durability or API maturity, so those remain paper-level claims.

### M3-Agent

M3-Agent processes online visual and auditory clips into episodic and semantic memory. Its source
implementation maintains entity-centric text, image, and audio graph nodes in Python dictionaries
and serializes them with pickle. Similar semantic memories reinforce graph edges; conflicting
attributes strengthen or weaken edges, and retrieval can vote by accumulated edge weight.

That design is compact and works for the paper's online-video agent, but frequency is not truth.
Repeated evidence from one sensor segment, one speaker, or one extraction retry can dominate a
rarer correction. The inspected source has a single timestamp/weight-oriented graph rather than
bitemporal assertions, metric coordinate frames, or transactional evidence retraction. MindBridge
therefore groups trait evidence by independent observation source, stores confidence on each
evidence edge, and treats state replacement as a validity/transaction-time operation rather than a
frequency vote.

### VoiceMem

VoiceMem is the most concrete streaming and companion-memory reference. Its stream consumes PCM,
external ASR partials, VAD state, and sound-only environment windows; speculative work is cancelled
on barge-in. Its left brain uses hierarchical informational memory, while its right brain separates
situational emotional evidence from longer-lived persona claims.

Two inspected details set important safety requirements for MindBridge:

- VoiceMem's final confirmation can reuse a partial-query result when the final transcript adds
  trailing words. MindBridge instead requires the prefetch revision to match the exact final
  snapshot and searches again when it does not.
- The inspected trait store starts extracted traits at fixed confidence and stores a single
  extraction immediately. Similar evidence can then merge without proving source independence.
  MindBridge keeps a model-inferred trait invisible until two independent observation sources
  support it; an explicit trusted user statement is visible immediately.

VoiceMem still leads MindBridge in ready-made speech capture, duplex-SLM integration, trained
persona models, and published real-time evaluation. MindBridge's contribution here is a
modality-neutral stream and evidence contract that also accepts visual and omni snapshots, rather
than a second speech-only execution plane.

## Second-tier findings

### eMEM

eMEM uses SQLite for structured storage, HNSW for semantic search, and an R-tree for exact spatial
candidate lookup. Its observations, episodes, gists, entities, and tiered consolidation provide a
strong embodied baseline, and its agent-facing tools expose radius, nearest, concept-to-location,
and cross-layer recall.

The inspected `ObservationNode` carries a coordinate vector, timestamp, source, confidence, layer,
and tier. Radius/nearest search is genuinely metric, but the model does not identify the coordinate
frame or whether coordinates locate observer versus subject; it has no orientation quaternion or
position uncertainty. Entity updates overwrite latest coordinates and aggregate confidence rather
than preserving transaction-time state history. MindBridge adopts the metric core while requiring
same frame and anchor and including pose uncertainty. Cross-frame transforms and R-tree scaling
remain future measured work.

eMEM also updates several derived indices around SQLite writes. MindBridge retains its stricter
consistency rule: SQLite and its durable index outbox commit first, and an operation is acknowledged
only after the Zvec flush succeeds.

### MIRIX

MIRIX divides memory into Core, Episodic, Semantic, Procedural, Resource, and Knowledge Vault stores
and coordinates them with specialized agents and a meta-controller. Its screen-capture product and
ScreenshotVQA results demonstrate the value of continuous visual experience and aggressive
compression.

The inspected episodic model has occurrence time, actor, event type, summary, and details, but no
validity interval or assertion history. Semantic updates delete an old record and insert a new one
through separate manager calls. Core-memory blocks are bounded text values without fact-level
source, confidence, and validity. Its Docker deployment introduces PostgreSQL/pgvector and Redis.
MindBridge borrows typed cognitive roles and multimodal evidence but keeps one embedded consistency
core. It does not add six physical stores or an eight-agent write path.

## Third-tier findings

### Graphiti/Zep

Graphiti is the strongest temporal/provenance reference. Entity edges retain episode provenance,
`valid_at`, `invalid_at`, `created_at`, and `expired_at`. Extraction identifies temporal bounds and
contradictory facts; a newer contradictory edge can invalidate an older edge at the new edge's
valid time. Current drivers include Neo4j, FalkorDB, Kuzu, and Neptune.

MindBridge adopts the separation between world validity and system knowledge time, but constrains
automatic replacement to an exact typed lineage: kind, normalized subject, predicate, coordinate
frame, and spatial anchor. This avoids broad semantic invalidation and allows a historical
correction to split the old interval into before/after carry-forward versions. Conflicts from one
explicit SQLite batch remain visible as independent evidence instead of being silently ordered;
wall-clock equality alone does not define a batch.

Graphiti remains ahead for typed multi-hop graph expansion and custom entity/edge schemas.
MindBridge remains lighter to deploy and stronger on raw-media durability, exact same-directory
ownership, and a SQLite-first rebuildable-index contract.

### Mem0

Mem0 remains the developer-experience reference: the API has short add/search verbs, provider
configuration, history, filters, and a broad integration ecosystem. The current open-source Python
snapshot defaults to an LLM, embeddings, local Qdrant, and a separate SQLite history store. Its V3
ingestion path performs one additive extraction call, hash de-duplicates exact text, inserts vector
records, and separately links extracted entities to memory IDs. The source docstring still
describes ADD/UPDATE/DELETE behavior while the inspected default path is additive-only, showing why
versioned implementation evidence matters.

Mem0 Platform's current Graph Memory automatically extracts entity nodes and connects memories by
co-occurrence. The [official documentation](https://github.com/mem0ai/mem0/blob/main/docs/platform/features/graph-memory.mdx)
states that it boosts normal retrieval scores and does not expose typed relationship edges. The
open-source configuration documentation says graph memory is not part of either OSS SDK.

MindBridge keeps Mem0's low-friction verbs but rejects implicit user/run scoping: one physical
`data_dir` is one memory domain. Its typed semantic versions are in the same authoritative SQLite
transaction as evidence and the search outbox rather than a separate audit log around a vector
store.

### TeleMem

TeleMem extends a Mem0-compatible interface with narrative extraction and a buffered
retrieve-cluster-consolidate pipeline. Its video module extracts frames/captions into a separate
vector store and uses ReAct-style question answering. This improves batching and narrative
compression, but it is not one live multimodal observation contract. In the inspected buffer path,
consolidated summaries are added without an obvious matching deletion of every old clustered
record, so storage-level replacement semantics require further verification.

MindBridge does not add asynchronous consolidation yet. Deterministic per-source formation and
evidence-aware retirement are safer until a benchmark demonstrates that buffered clustering is
worth its latency, model calls, and crash-recovery complexity.

## Implemented architecture and contracts

### Formation and adaptive forgetting

`FormationBackend` is an optional, runtime-checkable model protocol. It receives committed
`FormationInput` observations and returns source-aligned tuples of `FormationProposal`. The kernel,
not the model, owns IDs, evidence edges, source binding, modality checks, validity, transaction
time, conflict rules, and storage.

The built-in OpenAI adapter performs one batched strict-JSON generation request. It does not send
stable memory IDs, CAS IDs, exact spatial values, or local source identifiers to the remote model.
It also cannot label its own inference as an observed or user-stated fact. A trusted custom former
may emit `USER_STATEMENT` only when its own adapter can prove that basis from the input protocol.

Formation is idempotent by source memory plus durable formation recipe. Derived records, evidence,
embeddings, the completion marker, and outbox operations commit in one SQLite transaction. If the
former fails, the raw observation remains durable and the call reports the model error; a retry
reuses the deterministic source record and can complete formation. An unsupported modality is not
marked complete, so reopening with a broader adapter can form the existing observation.

Forgetting is evidence-aware and non-destructive by default:

- expired or superseded semantic versions leave active retrieval but remain auditable;
- deleting one source recomputes confidence and visibility from remaining independent evidence;
- deleting the final source removes the unsupported derived projection;
- raw observations are never destroyed merely because a model changed its mind;
- ordinary recency decay continues to affect ranking only and never changes truth.

### Bitemporal state evolution

`MemoryContext` exposes `valid_from`/`valid_until` for world time and
`recorded_at`/`retired_at` for knowledge time. `RetrievalScope(valid_at=..., known_at=...)` selects
the assertion that was valid in the world and known to the system at those instants. Evidence edges
and their confidence/visibility projection use one monotonic transaction instant, avoiding mixed
historical snapshots on repeated device clocks.

State and explicit trait lineages use half-open validity intervals. A later overlapping assertion
retires the old transaction version and carries non-overlapping before/after segments forward. This
supports A→B→A, bounded historical backfill, and correction without destroying the prior knowledge
view. Proposals formed in the same SQLite batch may remain conflicting evidence; arbitrary tuple
order and equal wall-clock values do not manufacture truth.

### Spatial and embodied state

`SpatialContext` contains a caller-selected Cartesian `frame_id`, an `OBSERVER` or `SUBJECT`
anchor, x/y/z metres, optional normalized XYZW orientation, and optional positional uncertainty.
Quaternions are normalized and canonicalized so `q` and `-q` have one representation.

`RetrievalScope(near=..., radius_m=...)` performs an authoritative SQLite distance check only among
records with the same frame and anchor. Uncertainty expands the intersection test conservatively.
No implicit frame transform is attempted; returning a numerically nearby point from a different
map would be worse than returning no result.

### Generalized capture streaming

`StreamEvent` is a three-state adapter boundary:

- `UPDATE` carries the complete current immutable `ContentInput` snapshot and may trigger
  speculative retrieval;
- `FINAL` carries the exact completed content or `StreamInput`, persists it once, and yields a
  `StreamCommit`;
- `CANCEL` discards the turn and writes nothing; EOF has the same no-write behavior for an
  unfinished turn;
- `stream_id` isolates interleaved prefetch state and is returned on the commit.

`AsyncCaptureStream` remains modality-neutral. `AsyncAudioStream` now maps canonical PCM chunks,
VAD state, complete ASR hypotheses, and acoustic boundaries into those states while keeping one WAV
buffer per `stream_id`. Native-audio embedders receive audio directly; text-only embedders receive
external ASR hypotheses or the configured final transcription fallback. Camera and omni adapters
may instead use `AsyncVisionStream`, which retains the latest encoded frame per scene and routes an
explicit or plugin-produced visual description to text-only embedders. Provider-specific deltas and
device SDK values do not enter the kernel.

Retrieval failure does not erase a final observation: the commit contains the durable record and a
stable retrieval error. Consumer cancellation before commit writes nothing. Once the final write
starts, cancellation waits for the irrevocable commit and then propagates; deterministic source
identity makes a retry idempotent.

### Affect and personality memory

`MemoryKind.AFFECT` stores situated evidence with source modality, optional valence/arousal,
confidence, valid time, and evidence IDs. Text, voice, face, posture, and other cues remain
separately attributable; a model cannot claim an audio cue when the source has no audio.

`MemoryKind.TRAIT` is a long-horizon hypothesis, not an emotion label. Model-inferred traits remain
stored but hidden from active retrieval until two independent source observations support the same
typed claim. Repeated extraction from one `source_id` counts once, using only that source's maximum
confidence. Independent confidence combines with a noisy-OR projection. Deleting evidence can hide
or remove the trait again. An explicit trusted user statement is immediately visible and follows
state-like correction semantics.

## Public API impact

The common path remains `Memory(...).add(...)`, `search(...)`, and `ask(...)`. All additions are
optional.

| Surface | Add observation semantics | Retrieve world/knowledge/spatial scope | Streaming capture |
| --- | --- | --- | --- |
| Python | `context=ObservationContext(...)` | `scope=RetrievalScope(...)` | `AsyncCaptureStream.consume(...)`; `AsyncAudioStream.consume(...)`; `AsyncVisionStream.consume(...)` |
| REST | optional `context` object | optional `scope` object | clients send finalized observations; no streaming route |
| MCP | optional `context` on `add_memory` | optional `scope` on search/ask | adapters call existing tools after finality |
| CLI | `--context JSON/@PATH/-`; per-item JSONL context | `--scope JSON/@PATH/-` | finite finalized JSONL remains `add-stream` |

No new MCP tool was added: the six-tool vocabulary remains stable. Typed context is application
data about evidence, not an account, tenant, request, or benchmark scope.

## Remaining priorities

### P0: prove memory quality and safety

1. Add formation benchmarks for extraction precision, source attribution, conflict handling,
   bitemporal state QA, trait false-positive rate, and evidence-retraction correctness.
2. Measure final-snapshot recall, correction latency, write amplification, SQLite/Zvec recovery,
   and p50/p95/p99 on representative edge and server devices.
3. Calibrate former confidence and the two-source trait gate on companion scenarios. Keep the gate
   conservative until false personalization is measured.

### P1: complete adapter and embodied coverage

1. Benchmark the canonical audio and vision adapters under barge-in, scene churn, and concurrent
   streams; add an omni correlation adapter only when one durable cross-modal final is required.
2. Add a local multimodal formation adapter so privacy-sensitive companion deployments need not
   call a remote model.
3. Add explicit coordinate transforms only with a real robotics integration and covariance tests.
   Add topological relations and object trajectories only when exact metric retrieval is
   insufficient.
4. Add companion-policy evaluation that distinguishes factual recall, affect attribution, trait
   calibration, tone adaptation, and correction recovery.

### P2: add structure only after a measured miss

1. Prototype an additive entity/relation projection if gold evidence is commonly present at large K
   but displaced at product K. Do not introduce a graph database before that gate passes.
2. Consider background consolidation or ABot-style gated evolution only with replayable assets,
   held-out regression checks, rollback, and explicit privacy review.
3. Add device-specific runtimes only after profiling identifies a model/runtime bottleneck; semantic
   and durability contracts must remain identical across RK3588, Jetson, OpenVINO, Apple silicon,
   workstation GPU, and server GPU targets.

This work intentionally does not make version 0.2 the unique public or installable MindBridge
surface. It improves the existing developer-facing contracts without changing the repository's
release-selection policy.
