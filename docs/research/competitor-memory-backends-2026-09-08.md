# Competitor memory backends: evidence and transfer decisions

**Research date:** 2026-09-08

**Scope:** primary papers, official repositories, and official documentation. Repository claims are
pinned to the inspected commit. Benchmark values below are author-reported; they are not treated as
MindBridge baselines unless reproduced under the same data, answer model, prompt, and budget.

This note extends the 2026-09-07 [backend audit](memory-backend-audit-2026-09-07.md) and
[source ledger](memory-backend-sources-2026-09-07.md). It focuses on gaps relevant to MindBridge's
embedded contract: one physical `data_dir` per instance, SQLite as the authoritative record and
embedding store, Zvec as a rebuildable index, and local WeMM embeddings.

## Decisions first

| Design | Verified strength | Lifecycle cost or evidence gap | Transfer decision |
| --- | --- | --- | --- |
| Graphiti/Zep | Episode-to-fact provenance, valid/invalid intervals, ingestion and event time kept separately, hybrid retrieval | Content-dependent multi-LLM write path plus graph database; Zep's production engine is not the OSS engine | Preserve raw evidence and dual time in SQLite; do not copy the graph service or extraction tax |
| Hindsight | Semantic, keyword, graph, and temporal candidates are fused; raw chunks can remain recallable; explicit token budgets | Default structured retain/consolidation is LLM-heavy and PostgreSQL-centric; zero-LLM chunks mode loses extracted entities and time | Reuse multi-arm admission and raw evidence, implemented on existing SQLite/Zvec data |
| Supermemory | Public contract exposes raw document chunks plus derived memories, current/history versions, soft forget, and hybrid search | Managed indexing and graph algorithms are absent from the public repository, so backend claims cannot be audited | Copy the contract ideas only; do not cite the managed service as algorithmic evidence |
| A-MEM | Links and evolves structured notes during memory formation | One or more LLM decisions on later writes, Chroma top-k at query, no bi-temporal/source-version model | Treat tags/context as optional derived views; never mutate away authoritative evidence |
| Letta/MemGPT | Clear hot context, immutable recall log, cold archive, and version history | Agent-controlled paging consumes model calls/tokens; current dreaming uses background agents and Git-backed MemFS | Use the tiering concept and delayed activation semantics, not Git or agent-controlled retrieval |
| Mem0 | Simple raw-memory baseline exists; mutation history exists | Default extraction/update and v3 flows use LLMs; OSS hybrid rerank can remain gated by the semantic pool | Retain the raw baseline and independent candidate admission already present in MindBridge |
| MemOS | Raw `general_text` path exists; paper covers pluggable memory forms | Rich tree/graph path adds LLM, Qdrant, Neo4j, reranking, and service components | Keep local raw-mode parity; reject service topology without measured need |
| M3-Agent | Face/voice continuity and causal `before_clip` cutoff address embodied identity and leakage | Heavy video/identity preprocessing, query expansion, and up to five model-driven retrieval rounds | Add identity and causal metadata only when available; keep retrieval bounded and deterministic |
| WorldMM | Multiscale episodic, evolving semantic, and raw visual memories are complementary; visual feature and timestamp routes both matter | Captions/triples at several scales, LLM consolidation, graph search, cross-scale rerank, and up to five agent rounds; no numeric formation-cost report | Reuse stored multimodal vectors and raw pointers; route and expand by query need instead of building LLM graphs |
| xMemory | Direct prior art for intact hierarchy, greedy coverage/representative selection, and uncertainty-gated expansion | Formation hierarchy and query-time uncertainty checks use LLMs; evaluation omits adversarial LoCoMo and uses lexical metrics | Anti-redundancy must act before final top-k, but coverage selection itself is not a novelty claim |
| RRM | Separates procedural retrieval experience from current-video factual evidence; reuses experience as query guidance | Evaluation adapts after each 64-question batch; failed-task extraction receives the reference answer | Keep process experience distinct from factual evidence; do not import benchmark-feedback adaptation into independent QA |
| NS-Mem | Combines episodic, semantic, and rule layers with deterministic symbolic queries | Structured-knowledge generation and rule maintenance add formation work | Treat deterministic rules as derived views only after their cost and provenance are measured |
| StreamMeCo | Compresses connected memory graphs and adds time-decayed retrieval | Its reported compression and speed apply to its graph pipeline, not MindBridge's SQLite/Zvec path | Measure topology-aware pruning separately before considering transfer |
| PMMC | Compiles and validates bounded evidence-access programs at write time, then re-executes them over visible raw evidence | Paid compilation is amortized; its main evaluation excludes refusal and conflict subsets | Keep programs separate from answers and charge write plus query cost; do not claim broad coverage |

The transfer column records design choices and candidate directions, not features added by this
patch. This round implements hybrid score completion, visual-description corrective retry, and a
reproducible benchmark fallback reference clock. Existing capabilities such as optional ingest-time
visual description remain existing capabilities. Validity/predecessor fields, new temporal routes,
hierarchical expansion, and other transfers in this table were not implemented or validated here.
A later benchmark-integrity fix assigns opaque source-order IDs in the default LongMemEval adapter;
it does not add a retrieval feature or supply a LongMemEval quality result.

A later, independent preprint, [*Harness the Memory*](https://arxiv.org/html/2608.15008v1),
reinforces the evaluation method rather than any specific transfer above. Its authors compare 11
memory substrates under three reader backbones and four QA or agent tasks, and report that the
preferred substrate reverses across task regimes. They also account for write, read, retrieval, and
management costs separately. Those are author-reported v1 results; the paper says code will be
released upon acceptance, and we did not reproduce them. Its sequential-action findings therefore
do not explain the M3 QA result in this study.

[RRM](https://arxiv.org/html/2607.28156v1) builds on M3-Agent's entity graph and stores retrieval
strategies distilled from earlier successful and failed trajectories; answer generation still uses
facts newly retrieved from the current video. The authors report gains on their evaluated video
benchmarks. Their protocol finalizes one 64-question batch before revealing references, then uses
those references when extracting corrective experience from failed tasks. Questions from one video
stay within one batch. This is delayed-feedback test-time adaptation, so its reported numbers are
not comparable to our independent, label-free M3 QA runs. We did not reproduce RRM, and this patch
does not add reflective experience memory.

[NS-Mem](https://arxiv.org/html/2603.15280v1), with an
[official repository](https://github.com/T-Lab/NS-Mem), combines episodic, semantic, and logical-rule
memory with deterministic symbolic query functions; its structured-knowledge generator also updates
neural representations and rules. [StreamMeCo](https://aclanthology.org/2026.findings-acl.647/) prunes
connected and isolated graph nodes differently and adds time-decayed retrieval; the authors report
70% graph compression and 1.87x retrieval speed in their setup. Neither result was reproduced here,
and neither establishes a drop-in optimization for MindBridge's embedded index.

[PMMC](https://arxiv.org/html/2608.00962v1) moves Questioner, Planner, and Doubter work to
consolidation, stores verified access programs without provisional answers, and re-executes them
against currently visible raw evidence; failed routing falls back to multimodal RAG. Its lifecycle
must charge compilation and query execution together. Its primary generative evaluation excludes
90 MemLens refusal questions and 81 conflict plus 184 refusal questions from Mem-Gallery, so the
reported gains do not establish broad memory-task SOTA. We did not reproduce or implement PMMC;
similarly named compiler work elsewhere in the shared tree is not attributed to this patch.

Two findings apply directly to the current hybrid score-completion change. MindBridge's scalar
`max(dense, full-coverage lexical)` fusion makes a missing dense value consequential for a partial
lexical candidate; rank-fusion designs such as RRF do not universally require every arm's feature for
every candidate. Late filtering still cannot recover a candidate omitted upstream. In Hindsight's
specific PostgreSQL implementation, fetching more already-sorted returned rows and then discarding
them does not enlarge fixed ANN exploration. That observation does not prove that changing Zvec
route depth or graph-search controls is useless: deeper routes can alter parent grouping and the
explored result set. Completing dense evidence from persisted embeddings for MindBridge's small
admitted lexical set is the narrower hypothesis evaluated here.

## Evidence ledger

### Graphiti/Zep

**Primary sources.** Official repository at commit
[`b943c9e`](https://github.com/getzep/graphiti/tree/b943c9e8486cdc7fe6cb2f4cfe151ae53f0a884d),
inspected 2026-09-08; in particular
[`graphiti.py`](https://github.com/getzep/graphiti/blob/b943c9e8486cdc7fe6cb2f4cfe151ae53f0a884d/graphiti_core/graphiti.py),
[`edges.py`](https://github.com/getzep/graphiti/blob/b943c9e8486cdc7fe6cb2f4cfe151ae53f0a884d/graphiti_core/edges.py), and
[`search_config_recipes.py`](https://github.com/getzep/graphiti/blob/b943c9e8486cdc7fe6cb2f4cfe151ae53f0a884d/graphiti_core/search/search_config_recipes.py).

**Verified.** Graphiti stores episodes as provenance for derived entity facts. Facts carry
`valid_at`, `invalid_at`, and expiration state, while episode `created_at` records ingestion time.
Its saga summarizer deliberately uses two watermarks: wall-clock ingestion time selects newly added
episodes, while the maximum episode `valid_at` preserves chronology. This prevents a newly ingested
backfill with an old event time from being skipped. The standard edge search recipe combines BM25
and cosine retrieval with reciprocal-rank fusion. `add_episode` performs entity extraction,
resolution/deduplication, edge extraction, duplicate/contradiction resolution, temporal parsing, and
optional attributes/summaries; the number of model calls grows with extracted content. The project
recommends sequential episode addition per group and a background queue for production ingestion.

**Boundary.** The README states that Graphiti is the OSS temporal-graph core and that Zep's
production graph engine is proprietary. Zep production performance is therefore not evidence for
this inspected code. Graph database dependencies and logical `group_id` scopes also conflict with
MindBridge's embedded physical-isolation contract.

**Transfer.** Store event time, knowledge/ingestion time, validity interval, predecessor, and source
record IDs in authoritative rows. Derived summaries or facts should be independently rebuildable and
must never replace source records. This gives the useful temporal semantics without a mandatory LLM
or graph database on each write.

### Hindsight

**Primary sources.** [ACL 2026 demo paper](https://aclanthology.org/2026.acl-demo.27/),
[arXiv paper](https://arxiv.org/abs/2512.12818), and official repository at
[`9e6d9e7`](https://github.com/vectorize-io/hindsight/tree/9e6d9e76cc52e918b24440355759991722a758a2),
inspected 2026-09-08. Retrieval details are visible in
[`retrieval.py`](https://github.com/vectorize-io/hindsight/blob/9e6d9e76cc52e918b24440355759991722a758a2/hindsight-api-slim/hindsight_api/engine/search/retrieval.py)
and defaults in
[`config.py`](https://github.com/vectorize-io/hindsight/blob/9e6d9e76cc52e918b24440355759991722a758a2/hindsight-api-slim/hindsight_api/config.py).

**Verified.** TEMPR retrieves semantic, keyword, graph, and temporal evidence and fuses candidates.
The current implementation retains raw chunks by default and gives them a separate token budget.
Semantic and BM25 subqueries use per-fact-type limits and partial HNSW indexes; graph seeds reuse the
semantic results rather than issuing a duplicate ANN query. A current source comment documents why,
for this implementation, changing SQL `LIMIT k` to `LIMIT 5k` and then retaining the first `k`
cannot improve approximate-neighbor quality: PostgreSQL already returns those rows ordered, while
the fixed HNSW exploration setting is controlled elsewhere. The current code's zero-LLM `chunks`
extraction mode stores verbatim chunks but does not infer entities or temporal structure. Default
concise extraction, observations, and auto-consolidation do use model work.

**Author-reported result.** The paper reports 83.6 on LongMemEval and 83.2 on LoCoMo for its 20B
configuration, with higher results for a larger proprietary reader. These values mix memory design,
reader, prompts, and service stack and are not comparable with MindBridge runs.

**Transfer.** Preserve candidates admitted independently by dense, lexical, temporal, and metadata
routes, then calculate comparable features on the bounded union. Keep raw source evidence beside
derived facts. Do not import PostgreSQL, bank scopes, or default consolidation into the embedded
runtime.

### Supermemory

**Primary sources.** Official repository at
[`5258cb7`](https://github.com/supermemoryai/supermemory/tree/5258cb74c895c5297fbfff594935741aeb14a915),
inspected 2026-09-08: [processing model](https://github.com/supermemoryai/supermemory/blob/5258cb74c895c5297fbfff594935741aeb14a915/apps/docs/concepts/how-it-works.mdx),
[graph-memory contract](https://github.com/supermemoryai/supermemory/blob/5258cb74c895c5297fbfff594935741aeb14a915/apps/docs/concepts/graph-memory.mdx), and
[search contract](https://github.com/supermemoryai/supermemory/blob/5258cb74c895c5297fbfff594935741aeb14a915/apps/docs/recall/search.mdx).

**Verified public contract.** Ingestion exposes queued, extraction, chunking, embedding, and indexing
stages followed by a separate memory-extraction or "dreaming" phase. Search can return graph-derived
memories and raw document chunks together. Updates create a new version while preserving the old one
with `isLatest=false`; forgetting is reversible/soft. Reranking and query rewriting are optional;
the docs state that rewrite generates several searches and merges them, and that rerank adds about
100 ms. Search is scoped by `containerTag`.

**Unverified.** The managed graph construction, contradiction handling, retrieval implementation,
and benchmark claims are not present in the public repository. The latency statements are vendor
self-reports, not reproducible measurements. They should not be used to justify an algorithm choice.

**Transfer.** Return source chunks and current derived memories in one evidence package; keep older
versions queryable for chronology and audit. Implement that contract with SQLite lineage and derived
Zvec entries rather than an opaque hosted graph.

### A-MEM

**Primary sources.** [NeurIPS 2025 paper](https://arxiv.org/abs/2502.12110) and official repository at
[`ceffb86`](https://github.com/agiresearch/A-mem/tree/ceffb860f0712bbae97b184d440df62bc910ca8d),
inspected 2026-09-08, especially
[`memory_system.py`](https://github.com/agiresearch/A-mem/blob/ceffb860f0712bbae97b184d440df62bc910ca8d/agentic_memory/memory_system.py)
and [`retrievers.py`](https://github.com/agiresearch/A-mem/blob/ceffb860f0712bbae97b184d440df62bc910ca8d/agentic_memory/retrievers.py).

**Verified.** A-MEM models a Zettelkasten note with content, context, keywords, tags, and links. On
later additions, it retrieves nearby notes and asks an LLM whether to link/evolve them; periodic
consolidation recreates the vector collection. The default search is Chroma vector top-k. Agentic
search appends linked neighbors into the same result budget rather than performing a scored graph
traversal. The inspected implementation keeps note objects in memory and resets its Chroma
collection at initialization; it does not provide MindBridge-grade durability, source lineage, or
bi-temporal history.

**Author-reported result.** Paper quality gains across six reader models are self-reported and include
LLM-based memory formation. They do not isolate the value of links from the write-time model cost.

**Transfer.** Optional tags and contextual descriptions can be derived asynchronously, but source
facts remain immutable and the system must work when no formation model is configured.

### Letta/MemGPT

**Primary sources.** Original [MemGPT paper](https://arxiv.org/abs/2310.08560) and current official
[Letta Code repository](https://github.com/letta-ai/letta-code/tree/2f0fb7c12c6973be7d52d9c7d3bf0bf4d9120cb8),
inspected 2026-09-08, including its
[MemFS prompt contract](https://github.com/letta-ai/letta-code/blob/2f0fb7c12c6973be7d52d9c7d3bf0bf4d9120cb8/src/agent/prompts/letta_local_memfs.md).

**Verified.** MemGPT frames memory as virtual-context paging: small in-context state, searchable
recall history, and archival storage, with the model deciding when to move or retrieve information.
Current Letta Code uses a Git-backed memory filesystem, immutable message history, bounded always-on
memory blocks, and external files discovered through paths. Memory-file changes affect a later
prompt compilation. "Dreaming" delegates consolidation to background agents.

**Boundary and transfer.** This is an agent runtime and memory UX, not evidence of a cheap retrieval
backend. The useful ideas are explicit hot/warm/cold tiers, immutable history, versioning, and delayed
activation. Git, background agents, and model-controlled paging add dependencies and token cost that
do not fit the default embedded backend.

### Mem0 and MemOS

These were source-audited in the 2026-09-07 ledger. The pinned evidence remains the official Mem0
repository at
[`dae67f7`](https://github.com/mem0ai/mem0/tree/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3)
and MemOS at
[`78a372a`](https://github.com/MemTensor/MemOS/tree/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad).
Mem0's current
[`how-it-works` documentation](https://github.com/mem0ai/mem0/blob/main/docs/core-concepts/how-it-works.mdx),
checked 2026-09-08, explicitly says that the OSS package does not include graph memory. Older Mem0
documentation and platform descriptions that present vector, key-value, and graph stores together
must not be projected onto the current OSS backend.

**Verified summary.** Mem0 has a raw no-inference path, but default extraction/update uses a model;
its history records mutations rather than a complete event-time truth model. Some OSS hybrid/entity
flows rerank or expand an already semantic-gated pool. MemOS `general_text` can store raw messages
and use vector retrieval; rich tree/graph modes add LLM construction, Qdrant, Neo4j, BM25, and
reranking. Published and hosted-service benchmark claims must be labeled author-reported and do not
describe these cheap defaults.

**Transfer.** Keep the raw no-LLM path as a quality and cost baseline. MindBridge already admits dense
and full-coverage lexical candidates independently; the current opportunity is to finish missing
features on that bounded union, preserve lineage, and measure end-to-end value rather than add more
formation machinery.

### M3-Agent and robot episodic memory

**Primary sources.** [ICLR 2026 paper](https://openreview.net/forum?id=PMz29A7Muq) and official
repository at
[`0e3e419`](https://github.com/bytedance-seed/m3-agent/tree/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c),
inspected 2026-09-08.

**Verified.** The reference preprocessing uses 30-second clips at 5 fps, caps sampled face/audio/image
embeddings, performs face and voice identity processing, and generates episodic/semantic text memory.
Retrieval embeds expanded query variants, searches graph nodes, aggregates node scores to clips, and
can run up to five model-driven retrieval steps; the default retrieved clip count is two per round.
Robot annotations expose `before_clip`, making the causal cutoff explicit. Visual and voice features
mainly support identity resolution; inference evidence is centered on clip text memories.

**Boundary.** Identity clustering, video memory generation, combinatorial query expansion, and
iterative planning make lifecycle cost large and variable. Repo and paper revisions also report
different headline deltas; neither should be compared to MindBridge without reproducing the exact
version and answer model.

**Transfer.** Honor a strict observation cutoff before retrieval and retain stable optional identity
IDs across clips. Use raw clip/frame pointers as evidence so a caption error is recoverable. Avoid
random query-subset selection and open-ended planner loops in the default path.

### WorldMM

**Primary sources.** [CVPR 2026 paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Yeo_WorldMM_Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning_CVPR_2026_paper.html),
[arXiv v2](https://arxiv.org/html/2512.02425v2), inspected 2026-09-08. A repository URL found during
search no longer resolved, so no implementation claim below relies on it.

**Verified architecture.** WorldMM builds episodic graphs at several non-overlapping timescales. For
EgoLifeQA these are 30 seconds, 3 minutes, 10 minutes, and 1 hour. It captions segments and converts
them to entity-action-entity triples. A semantic graph is updated by embedding matching followed by
an LLM decision to remove conflicting/outdated triples or add/revise facts. Visual memory keeps
fixed-segment multimodal embeddings and timestamp-addressable frames. At query time an agent chooses
episodic, semantic, or visual retrieval; episodic search spans every timescale and is reranked across
scales, while visual access supports both feature similarity and direct timestamp lookup. The
reference configuration uses GPT-5-mini for construction and permits up to five retrieval rounds.

**Author-reported results and ablations.** WorldMM-GPT reports 69.5 mean accuracy across five video
benchmarks and 65.6 on EgoLifeQA. Its Table 2 reports averages of 64.9 for episodic only, 44.9 for
visual only, 66.8 for episodic+semantic, 66.9 for episodic+visual, and 69.5 for all three. In the 8B
component ablation, fixed-timescale and embedding episodic retrieval score 51.8 and 52.0 on EgoLifeQA,
versus 56.4 for the full system; no semantic consolidation scores 53.0; feature-only and
timestamp-only visual retrieval score 53.6 and 51.8. Five retrieval steps improve EgoLifeQA by a
reported 9.3% over one step. Nearby scale choices score 64.8--65.6, suggesting the hierarchy matters
more than exact interval tuning.

**Cost boundary.** These results bundle captioning at the finest scale, captions/triples at coarser
scales, semantic extraction and LLM consolidation, graph storage/search, cross-scale model reranking,
and as many as five query-agent calls. The paper plots query latency but publishes no numeric
formation-token, GPU-time, storage-amplification, or lifecycle break-even table. It explicitly lists
captioning, triple extraction, and consolidation as preprocessing limitations. Semantic consolidation
removes stale/conflicting triples; the described method does not establish preserved version lineage.

**Transfer.** Persist WeMM segment/frame embeddings with timestamp and source pointers, then open raw
visual evidence only when the query or an earlier temporal hit needs it. Represent multiple temporal
scales as cheap derived windows over the same records rather than repeatedly captioning and
triplifying every scale. Preserve old facts and their validity ranges instead of deleting them.

WorldMM and xMemory also explain a failure in ordinary top-k. Repetitions around one event form a
dense burst, so several query parts and modalities can still select near-duplicate clips. A late
filter can remove duplicates but cannot admit the distant event or temporal prerequisite that was
excluded upstream. Diversity, temporal/source coverage, or missing-score completion must therefore
influence candidate admission or bounded-union scoring before the final top-k.

### xMemory

**Primary source.** [arXiv:2602.02007v1](https://arxiv.org/html/2602.02007v1), published 2026-02-02
and inspected 2026-09-08.

**Verified architecture.** xMemory preserves a four-level mapping from contiguous original messages
to episodes, reusable semantics, and themes: each message block maps to one episode, an episode can
produce several semantics, and each semantic belongs to one theme. A sparsity/coherence objective
guides theme attach, split, and merge; theme and semantic levels have bounded k-nearest-neighbor
graphs. Retrieval first uses greedy representative selection that trades query relevance against
weighted graph-neighborhood coverage and stops on a coverage or size budget. It then admits intact
linked episodes and optionally raw messages only when they reduce a reader model's predictive
uncertainty.

**Author-reported results.** On the non-adversarial LoCoMo categories with Qwen3-8B, xMemory reports
43.98 token-F1 and 4,711 total tokens/query, versus naive RAG at 36.42 and 6,613. The reported token
count includes retrieval, answer, and auxiliary model calls. With GPT-5 nano it reports 50.00 F1 and
6,581 tokens/query. The benchmark omits LoCoMo's adversarial subset and uses token-F1/BLEU rather than
the LoCoMo-refined judge, so the values are not directly comparable to MindBridge's target suite.

**Novelty boundary and transfer.** Intact hierarchy, coverage-oriented greedy representative
selection, and uncertainty-gated expansion are explicit prior art. MindBridge should not claim
novelty for coverage selection or multi-level expansion alone. The transferable problem statement is
strong: fixed similarity top-k collapses into correlated regions, while post-hoc pruning can remove
temporally linked prerequisites. A distinct embedded design can reuse persisted raw/multimodal
vectors, compute route geometry without a formation LLM or graph service, and expand with explicit
chronology/source/version closure. Its evaluation must charge every auxiliary call and verify source
evidence rather than optimize only answer-token containment.

## Contribution boundary and architecture hypotheses

The implemented bounded score completion is a correctness repair inside MindBridge's existing
scalar hybrid fusion. It supplies persisted dense evidence that a lexical-admitted candidate was
missing. It adds no retrieval arm, learned ranker, RRF/MMR variant, graph, formation model, or new
novelty claim. Its benchmark result can support the value of this repair under the frozen protocol;
it cannot establish a generally novel retrieval algorithm or broad SOTA by itself.

The strongest next hypotheses are not implemented or validated:

1. **Query-adaptive multiscale evidence closure without formation LLMs.** WorldMM's ablations show
   complementary episodic, semantic, and visual routes, while xMemory shows value from intact linked
   context. MindBridge could derive bounded temporal windows and neighbors from persisted timestamps,
   raw pointers, and WeMM vectors, escalating scales only when route disagreement or ambiguity calls
   for it. The hypothesis is that this recovers distant events and prerequisites with much less
   lifecycle cost than captioning/triplifying every scale. Coverage, hierarchy, and expansion are
   prior art; any possible novelty would be in a rebuildable low-cost implementation combined with
   explicit causal and version-lineage constraints.
2. **Chronology-aware representative admission.** xMemory supplies direct prior art for greedy
   neighborhood coverage, and Graphiti supplies dual-time/version semantics. A MindBridge-specific
   hypothesis is to select representatives over source, event-window, modality, and version lineage,
   then include required predecessors within one fixed candidate/context budget. This may reduce
   burst redundancy while preserving causal evidence, but needs adversarial temporal and provenance
   tests before any novelty or quality claim.
3. **Expansion from measurable retrieval signals.** xMemory uses reader-model uncertainty, which adds
   query calls. MindBridge could test dense/lexical disagreement, query-part coverage, score margin,
   and, only where provenance can be reconstructed, cross-modal agreement as retrieval trigger signals
   for opening a wider time window or raw visual evidence. Current stored embedding rows carry a part
   ordinal but no independent asset/chunk provenance field, so reliable modality routing would require
   a proved reconstruction or a separately justified product contract. These signals are not calibrated
   answer uncertainty. This is only a cost-quality hypothesis until evaluated on held-out tasks with
   every extra embedding, model call, token, and byte charged.

## Implications for evaluation

1. **Charge the lifecycle.** Report formation model calls/tokens, embedding calls, SQLite and Zvec
   bytes, p50/p95 write time, cold rebuild time, query calls/tokens, p50/p95 retrieval time, and final
   context tokens. WorldMM and structured competitors hide large costs if only answer accuracy and
   query latency are counted.
2. **Separate admission from ranking.** Measure source-record recall at the candidate-union boundary,
   after feature completion, and after final top-k. This detects unrecoverable early gating.
3. **Measure redundancy and closure.** Track unique event/time-window coverage, same-burst duplicate
   rate, predecessor/version closure, and raw-source availability. High answer score with unsupported
   or chronologically invalid evidence is a failure.
4. **Use causal cutoffs.** ATM-Bench and M3-Bench Robot cases must exclude records observed after the
   question's cutoff. Apply the cutoff before ANN/lexical admission, not only during answer
   formatting. Historical EgoLifeQA diagnostics exposed the same evaluation risk, but that dataset
   has since been removed from MindBridge's product-acceptance scope.
5. **Ablate mechanisms, not knobs.** Compare raw dense, current hybrid union, bounded dense-score
   completion, temporal/source diversity, cheap multiscale windows, and raw-visual escalation under the
   same candidate/context/call budget. Do not tune benchmark-specific weights without a held-out rule.
6. **Keep a blind baseline.** Benchmark the configured reader without memory and reject changes that
   improve one public aggregate while degrading source recall, calibration, cost, or the priority
   benchmark family.

## Immediate review checklist for bounded hybrid score completion

- The lexical-only candidate set is bounded before stored embeddings are read; no full-store scan or
  embedding/model call is introduced.
- Query and stored vectors use the same normalization and cosine-to-score mapping as Zvec, including
  zero norm and dimension validation.
- Every eligible multimodal part is considered using the product's documented part aggregation; a
  convenient first part must not substitute for the record score.
- All time, type, and known-at filters apply before completion, so scoring cannot resurrect an
  inadmissible record.
- Dense-only behavior and candidates already carrying an ANN score remain unchanged.
- Missing/stale derived Zvec IDs still hydrate through SQLite and are dropped according to the
  existing contract.
- Malformed authoritative embedding blobs take the existing corruption/error path and never become a
  plausible zero score.
- Tests distinguish "missing dense evidence" from a valid low or zero dense score and include a
  lexical-only record whose persisted dense similarity changes its fused order.
