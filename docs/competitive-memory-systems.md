# Competitive memory-system review

This review records the external evidence behind MindBridge's embodied-memory decisions. The
review date is 2026-09-01. It is an architecture input, not a product ranking or a benchmark.

## How to read the evidence

Four evidence levels are kept separate:

1. A paper supports a published design or evaluation claim.
2. A pinned repository snapshot supports an implementation observation at that commit.
3. MindBridge source and tests support statements about its current behavior.
4. A proposed feature remains deferred until a named measurement or integration requires it.

A paper is not evidence of production maturity. A source snapshot is not a benchmark result, and
it says nothing about later revisions. Published scores are not compared across systems because
datasets, model endpoints, retrieval budgets, hardware, and evaluators differ. A quality claim must
be reproduced through MindBridge's public SDK before it can select a default; see
[benchmarking](benchmarking.md).

## Reviewed evidence

The first three systems were reviewed paper-first and then source-first. The remaining systems
tested whether the resulting decisions generalized across temporal graphs, embedded stores,
multi-agent memory, and consolidation pipelines.

| System | Primary evidence | Source snapshot inspected |
| --- | --- | --- |
| ABot-AgentOS | [Paper](https://arxiv.org/abs/2607.10350); the paper-linked repository returned 404 on the review date | A previously indexed release notice said code and resources were being prepared; no implementation-maturity claim is possible |
| M3-Agent | [Paper](https://arxiv.org/abs/2508.09736), [official repository](https://github.com/ByteDance-Seed/m3-agent) | [`0e3e419`](https://github.com/ByteDance-Seed/m3-agent/commit/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c) |
| VoiceMem | [Paper](https://arxiv.org/abs/2608.26005), [official repository](https://github.com/xzf-thu/VoiceMem) | [`f99569e`](https://github.com/xzf-thu/VoiceMem/commit/f99569ed0718543c44da44ae833aeb5c3659cdfc) |
| eMEM | [Paper](https://arxiv.org/abs/2606.03374), [official repository](https://github.com/Automatika-Robotics/eMEM) | [`82e3da6`](https://github.com/Automatika-Robotics/eMEM/commit/82e3da61cf710c4379e0cd7bf7a6a21710caaa96) |
| MIRIX | [Paper](https://arxiv.org/abs/2507.07957), [official repository](https://github.com/Mirix-AI/MIRIX) | [`8cb06a6`](https://github.com/Mirix-AI/MIRIX/commit/8cb06a62bbb7c478beb33dd4f2815696a72df482) |
| Graphiti/Zep | [Paper](https://arxiv.org/abs/2501.13956), [official repository](https://github.com/getzep/graphiti) | [`8b61fce`](https://github.com/getzep/graphiti/commit/8b61fce9f003cc3a05e246f6201f8b782dfe6546) |
| Mem0 | [Paper](https://arxiv.org/abs/2504.19413), [official repository](https://github.com/mem0ai/mem0) | [`71fba8d`](https://github.com/mem0ai/mem0/commit/71fba8d46436f88569d600f81a55208c38ad30b5) |
| TeleMem | [Paper](https://arxiv.org/abs/2601.06037), [official repository](https://github.com/TeleAI-UAGI/TeleMem) | [`24630b1`](https://github.com/TeleAI-UAGI/TeleMem/commit/24630b107ffab670952358954cab92f5a914d4af) |

The source review traced ingestion, conflict resolution, persistence, retrieval, deletion, and
stream finality where each repository exposed those paths.

## Capability audit and decisions

The main gap was not another vector index. It was a trustworthy formation boundary between raw
multimodal observations and retrievable typed memory. The smallest design that closed that gap
kept one embedded consistency core:

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

The audit describes demonstrated paper or inspected-source behavior, not marketing breadth.

| Capability | Strong references | MindBridge contract | Remaining ceiling |
| --- | --- | --- | --- |
| Automatic formation | ABot typed source-grounded graph; VoiceMem dual-brain extraction; Mem0 fact extraction; MIRIX specialized agents | Optional `FormationBackend`; typed entity, event, state, relation, affect, trait, and response-policy proposals; deterministic IDs and evidence links | Formation quality needs benchmarked local and remote adapters; no autonomous rule evolution |
| Conflict and forgetting | Graphiti relation invalidation; ABot supersession; VoiceMem left-brain operations | Overlapping state assertions are retired and split by valid time; evidence deletion recomputes or removes derived projections; expiry is query-visible, not destructive | No background consolidation scheduler or learned forgetting policy |
| Temporal model | Graphiti `valid_at`, `invalid_at`, `created_at`, `expired_at` | Separate valid time and recorded/retired transaction time; `valid_at` and `known_at` retrieval; historical corrections and A→B→A state evolution | No temporal relation planner or natural-language interval extraction beyond the former/parser |
| Spatial memory | eMEM R-tree radius/nearest; ABot typed places and spatial edges | Typed Cartesian frame, observer/subject anchor, position, normalized quaternion, uncertainty, and exact same-frame radius search | No frame transform tree, topological map, occupancy, trajectory, or geodesic query |
| Live multimodal input | VoiceMem VAD/ASR partial/audio stream; M3 online video/audio clips | Associated updates drive speculative retrieval; `AsyncAudioStream` accepts canonical speech packets; `AsyncVisionStream` accepts encoded frames, visual partials, and scene boundaries; exact finals persist | Device SDKs still normalize provider deltas and pixels; measured-sensor packet protocols remain adapter work |
| Emotion and personality | VoiceMem right brain; M3 semantic personality/emotion; MIRIX core memory | Affect retains cue modality, valence, arousal, confidence, time, and evidence; inferred traits stay hidden until two independent sources support them | No companion-policy evaluator, calibrated multimodal fusion model, or user-facing profile editor |
| Multi-hop structure | ABot and Graphiti typed graphs; Mem0 Platform entity graph | Typed lineage and source/evidence links in SQLite; semantic and lexical retrieval | No graph expansion; add only after measured top-k evidence displacement |
| Embedded durability | eMEM SQLite plus derived indices | SQLite is authoritative; formation, evidence, versions, embeddings, and outbox enqueue commit atomically; Zvec is rebuildable | Device-level crash and power-loss campaigns remain to be published |
| Developer experience | Mem0 short add/search loop; TeleMem Mem0-compatible surface | Existing `add`, `search`, and `ask` remain; context and scope are optional across Python, REST, MCP, and CLI | Generated client examples and a compatibility/versioning guide would improve adoption |

## Findings by system

Each subsection distinguishes the inspected external behavior from the MindBridge decision it
informed.

### ABot-AgentOS

The paper's Universal Multi-modal Graph Memory represents source-grounded entities, events, places,
sessions, evidence, temporal relations, spatial context, and provenance as typed nodes and edges.
Its described write path forms memory selectively, merges duplicates, and supersedes stale state
with temporal edges instead of erasing the source. The failure-driven self-evolution loop proposes
a bounded JSON asset and promotes it only after regression gates on later splits.

That distinction matters: self-evolution changes behavior through gated, replayable assets; it is
not a larger score for frequently retrieved records. MindBridge adopts source grounding, typed
semantics, supersession, and regression-gated development. It does not adopt ABot's planner, skill
runtime, verifier, edge/cloud control plane, or generated evolution assets because those belong to
an AgentOS rather than an embedded memory kernel.

The paper-linked repository was unavailable on the review date. Durability, API maturity, and the
runtime behavior described by the paper therefore remain unaudited paper-level claims.

### M3-Agent

M3-Agent processes online visual and auditory clips into episodic and semantic memory. At the
inspected commit, entity-centric text, image, and audio graph nodes live in Python dictionaries and
are serialized with pickle. Similar semantic memories reinforce graph edges; conflicting
attributes strengthen or weaken edges, and retrieval can vote by accumulated edge weight.

That design is compact for the paper's online-video agent, but repeated evidence from one sensor
segment, speaker, or extraction retry can outweigh a rarer correction. The inspected graph is
timestamp- and weight-oriented rather than bitemporal, metric-frame-aware, or transactionally
retractable. MindBridge therefore groups support by independent observation source and handles
state replacement through valid and transaction time instead of a frequency vote.

### VoiceMem

VoiceMem is the most concrete streaming and companion-memory reference in this set. Its stream
consumes PCM, external ASR partials, VAD state, and sound-only environment windows, and cancels
speculative work on barge-in. Its left brain uses hierarchical informational memory; its right
brain separates situated emotional evidence from longer-lived persona claims.

Two inspected behaviors informed stricter MindBridge boundaries:

- Final confirmation can reuse a partial-query result when the final transcript merely adds
  trailing words. MindBridge reuses speculative retrieval only when its revision matches the exact
  final snapshot.
- Extracted traits begin at a fixed confidence and one extraction is stored immediately; later
  similar evidence can merge without proving source independence. MindBridge hides a model-inferred
  trait until two independent observation sources support it. A trusted explicit user statement is
  visible immediately.

VoiceMem has ready-made speech capture, duplex-SLM integration, trained persona models, and a
published real-time evaluation that MindBridge does not reproduce here. MindBridge takes the
streaming and evidence lessons while keeping one modality-neutral snapshot contract for text,
audio, visual, and omni adapters.

### eMEM

At the inspected commit, eMEM combines SQLite structured storage, HNSW semantic search, and an
R-tree for exact spatial candidates. Its observations, episodes, gists, entities, and tiered
consolidation provide an embodied baseline, and its agent tools expose radius, nearest,
concept-to-location, and cross-layer recall.

`ObservationNode` carries a coordinate vector, timestamp, source, confidence, layer, and tier.
Radius and nearest queries are metric, but the inspected model does not identify a coordinate frame
or whether the coordinate locates observer versus subject; it has no orientation quaternion or
position uncertainty. Entity updates overwrite latest coordinates and aggregate confidence rather
than preserving transaction-time state history.

MindBridge adopts the metric core but requires frame and anchor equality and includes pose
uncertainty. eMEM also updates multiple derived indices around SQLite writes. MindBridge instead
commits SQLite and its durable index outbox first, then acknowledges an operation only after the
Zvec flush succeeds.

### MIRIX

MIRIX divides memory into Core, Episodic, Semantic, Procedural, Resource, and Knowledge Vault stores
coordinated by specialized agents and a meta-controller. Its screen-capture product and reported
ScreenshotVQA results support the value of continuous visual experience and aggressive
compression within its own evaluation.

At the inspected commit, the episodic model has occurrence time, actor, event type, summary, and
details, but no validity interval or assertion history. Semantic updates delete an old record and
insert a new one through separate manager calls. Core-memory blocks are bounded text values without
fact-level source, confidence, and validity. The documented Docker path introduces
PostgreSQL/pgvector and Redis.

MindBridge borrows typed cognitive roles and multimodal evidence, but keeps one embedded
consistency core. It does not add six physical stores or an eight-agent write path.

### Graphiti/Zep

Graphiti is the strongest temporal and provenance reference in this set. At the inspected commit,
entity edges retain episode provenance plus `valid_at`, `invalid_at`, `created_at`, and
`expired_at`. Extraction identifies temporal bounds and contradictory facts; a newer contradictory
edge can invalidate an older edge at the new edge's valid time. The inspected drivers include
Neo4j, FalkorDB, Kuzu, and Neptune.

MindBridge adopts separate world-validity and system-knowledge time, but limits automatic
replacement to an exact typed lineage: kind, normalized subject, predicate, coordinate frame, and
spatial anchor. A historical correction can split an old interval into before and after versions.
Conflicts formed in one explicit SQLite batch remain independent evidence; tuple order or equal
wall-clock values do not manufacture truth.

Graphiti exposes typed multi-hop expansion and custom entity or edge schemas that MindBridge does
not. MindBridge keeps raw-media durability, exact same-directory ownership, and the
SQLite-first/rebuildable-index contract instead of introducing a graph service before retrieval
measurements justify one.

### Mem0

Mem0 is the developer-experience reference in this set: its API provides short add and search
verbs, provider configuration, history, filters, and a broad integration surface. At the inspected
open-source commit, the default Python path combines an LLM and embeddings with local Qdrant and a
separate SQLite history store. V3 ingestion performs one additive extraction call, hash
de-duplicates exact text, inserts vector records, and separately links extracted entities to memory
IDs. A source docstring still describes ADD/UPDATE/DELETE behavior while the inspected default path
is additive-only, demonstrating why implementation claims need a pinned revision.

Mem0 Platform's Graph Memory automatically extracts entity nodes and links memories by
co-occurrence. Its
[official documentation](https://github.com/mem0ai/mem0/blob/main/docs/platform/features/graph-memory.mdx)
says that graph context boosts normal retrieval scores and does not expose typed relationship
edges. The inspected open-source configuration documentation says graph memory is not part of
either OSS SDK.

MindBridge keeps the low-friction verbs but rejects implicit user or run scoping. One physical
`data_dir` is one memory domain. Typed versions, evidence, embeddings, and the search outbox share
the authoritative SQLite consistency boundary rather than surrounding a vector store with a
separate history log.

### TeleMem

TeleMem extends a Mem0-compatible interface with narrative extraction and a buffered
retrieve-cluster-consolidate pipeline. Its video module extracts frames and captions into a
separate vector store and uses ReAct-style question answering. This provides batching and narrative
compression, but not one live multimodal observation contract.

In the inspected buffer path, consolidated summaries are added without an obvious matching
deletion of every old clustered record. Storage-level replacement semantics therefore require
further verification. MindBridge does not add asynchronous consolidation yet; deterministic
per-source formation and evidence-aware retirement avoid its model-call, latency, and crash
recovery costs until a benchmark demonstrates the tradeoff is worthwhile.

## MindBridge contract resulting from the review

The detailed behavior belongs to task-focused pages rather than this comparison. Current source
and tests establish these boundaries:

| Decision area | Current contract | Canonical documentation |
| --- | --- | --- |
| Storage authority | SQLite owns records, embeddings, metadata, typed evidence and versions, and the durable search-index outbox. Zvec is derived and rebuildable. SQLite commits before Zvec changes; outbox acknowledgement follows a successful flush. | [Write consistency](architecture.md#write-consistency) |
| Isolation | One physical `data_dir` has one live `Memory` owner. Separate directories can run concurrently. Metadata is payload, not an access-control or isolation boundary. | [Ownership and concurrency](architecture.md#ownership-and-concurrency) |
| Formation and evidence | `FormationBackend` proposes source-aligned typed records. The kernel validates and commits them; deleting evidence recomputes visibility or removes an unsupported projection without rewriting the raw observation. | [Write consistency](architecture.md#write-consistency), [backend protocols](api/python-sdk.md#backend-protocols) |
| Temporal and spatial retrieval | `RetrievalScope` supports world-valid `valid_at`, knowledge-time `known_at`, and exact same-frame `near`/`radius_m` filtering. | [Memory operations](api/python-sdk.md#memory-operations) |
| Stream finality | `UPDATE` is speculative, `FINAL` persists once, and `CANCEL` or unfinished EOF writes nothing. Audio and vision adapters normalize provider events into the same boundary. | [Async capture streams](api/python-sdk.md#async-capture-streams) |
| Affect and traits | Affect keeps cue modality and evidence. A model-inferred trait needs two independent source groups; a trusted explicit user statement is visible immediately. | [Write consistency](architecture.md#write-consistency), [public values](api/python-sdk.md#public-values) |
| Public surfaces | Supported Python values are imported from `mindbridge`; REST has eight routes under `/v1`; MCP publishes fourteen tools covering core memory, reinforcement, embodied analysis, and identity operations. Optional context does not introduce account, user, request, or benchmark scope. | [Python SDK](api/python-sdk.md), [REST](api/rest.md), [MCP](api/mcp.md) |

## Deferred work and evidence gates

- Benchmark formation precision, source attribution, conflict handling, bitemporal QA, trait false
  positives, evidence retraction, and final-snapshot recall before changing defaults.
- Add a graph projection only when relevant evidence is repeatedly present at large K but displaced
  at product K. That result would justify a projection, not automatically a graph database.
- Add background consolidation or ABot-style evolution only with replayable assets, held-out
  regressions, rollback, crash-recovery tests, and privacy review.
- Add coordinate transforms only for a concrete robotics integration with frame and covariance
  tests. Add topological relations or trajectories only when exact metric retrieval is insufficient.
- Add a local multimodal former and device-specific runtimes only after privacy requirements or
  profiling identify a real model/runtime bottleneck. Storage and durability contracts remain the
  same across devices.
- Evaluate companion behavior separately across factual recall, affect attribution, trait
  calibration, tone adaptation, and correction recovery before claiming personalization quality.

This evidence does not establish cross-system benchmark superiority, justify restoring logical
scoping identifiers, or require a database service, worker queue, graph database, or alternate
product API.
