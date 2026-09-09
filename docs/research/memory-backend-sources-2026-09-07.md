# Memory competitor source ledger (2026-09-07)

This is a source-first audit, not a leaderboard. “Observed” means the claim follows from the
pinned source path or a committed artifact. “Author-reported” means the artifact was not
independently rerun here. Costs count calls made by the memory layer; answer and judge calls are
listed separately when relevant. Transfer recommendations assume MindBridge keeps SQLite
authoritative, Zvec derived/rebuildable, one physical directory per instance, and no logical
scope identifier.

## Pinned official sources

| Project | Official repository | Pinned commit | Commit date |
| --- | --- | --- | --- |
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | [`dae67f7`](https://github.com/mem0ai/mem0/commit/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3) | 2026-09-04 |
| MemOS | [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | [`78a372a`](https://github.com/MemTensor/MemOS/commit/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad) | 2026-09-03 |
| OmniMemEval | [MemTensor/OmniMemEval](https://github.com/MemTensor/OmniMemEval) | [`0b1ea8d`](https://github.com/MemTensor/OmniMemEval/commit/0b1ea8d28aa2d3e03ac4a6aee17b3006a131da7d) | 2026-07-24 |
| EverOS | [EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) | [`8754365`](https://github.com/EverMind-AI/EverOS/commit/8754365c76daa2f13521fcd29a53044bba083403) | 2026-09-07 |
| EverMemOS (separate related project) | [NetMindAI-Open/EverMemOS](https://github.com/NetMindAI-Open/EverMemOS) | [`806ad05`](https://github.com/NetMindAI-Open/EverMemOS/commit/806ad0555a09245ec90ad936e848bee9c64c6a49) | 2026-02-10 |
| TencentDB-Agent-Memory | [TencentCloud/TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory) | [`2ee2239`](https://github.com/TencentCloud/TencentDB-Agent-Memory/commit/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94), `feat/server_team` checkout | 2026-09-03 |
| MemPalace | [MemPalace/mempalace](https://github.com/MemPalace/mempalace) | [`d9f0590`](https://github.com/MemPalace/mempalace/commit/d9f059076c866fa6f29195679d75712436986024) | 2026-09-01 |
| VoiceMem | [xzf-thu/VoiceMem](https://github.com/xzf-thu/VoiceMem) | [`a450911`](https://github.com/xzf-thu/VoiceMem/commit/a450911fc8cbb44c46d810aace2f3288bad287e4) | 2026-09-05 |
| RE-call | [GiulioDER/RE-call](https://github.com/GiulioDER/RE-call) | [`4b009bf`](https://github.com/GiulioDER/RE-call/commit/4b009bfba3416db80a10c00ca6f1e086c3f378a2) | 2026-09-06 |
| InvMem implementation | [wenxiaof345-ctrl/vanilla-rag-memory](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory) | [`31ab7bf`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/commit/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca) | 2026-08-11 |
| ReFind | [imlrz/ReFind](https://github.com/imlrz/ReFind) | [`a80175c`](https://github.com/imlrz/ReFind/commit/a80175ca0eeb52a938d7cab7a602bc780de8a577) | 2026-08-14 |
| ActiveMemoryIndex | [linxuhao/ActiveMemoryIndex](https://github.com/linxuhao/ActiveMemoryIndex) | [`2e6d76c`](https://github.com/linxuhao/ActiveMemoryIndex/commit/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69) | 2026-08-26 |
| ChronoHybridMem | [Tin11Mn/chrono-hybrid-mem](https://github.com/Tin11Mn/chrono-hybrid-mem) | [`f6c1f98`](https://github.com/Tin11Mn/chrono-hybrid-mem/commit/f6c1f98663025a28cf5ab8d1ed08ea62484b3972) | 2026-09-04 |

An earlier inspection used `smaugho/mempalace`; that is a fork and is excluded from the conclusions
below. EverOS and EverMemOS are distinct systems and are evaluated separately.

## Core systems

### Mem0 OSS

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| The default OSS stack uses an OpenAI-compatible LLM and embeddings plus Qdrant. SQLite stores mutation history; it is not the authoritative memory/vector store. | [`mem0/configs/base.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/configs/base.py), [`mem0/memory/main.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/memory/main.py) | High, source path. Its `user_id`/`agent_id`/`run_id` contract conflicts with MindBridge physical-directory isolation. |
| Default v3 inference writes only `ADD` operations. It performs one LLM extraction call, embeds the proposed memories, retrieves existing memories for deduplication, and inserts non-MD5 duplicates. | [`mem0/configs/prompts.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/configs/prompts.py), [`mem0/memory/main.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/memory/main.py) | High. At least one LLM call per inferred batch plus embedding/search/write. |
| Search embeds the query, performs semantic overfetch, then combines semantic, BM25 and entity signals. However only semantic hits enter the candidate set; lexical/entity-only records cannot be rescued. NER/entity lookup can fan out to eight entities × 500 hits. | [`mem0/memory/main.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/memory/main.py), [`mem0/utils/scoring.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/utils/scoring.py) | High. This is a recall ceiling and potentially large read amplification. No LLM reranker is in the default path. |
| OSS has no automatic forgetting in the inspected path. Timestamp arguments are rejected there; `reference_date`/decay described for the platform are not evidence of OSS behavior. | [`mem0/memory/main.py`](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/mem0/memory/main.py) | High for this commit; managed service behavior was not inspected. |
| Published paper scores and managed “v3” claims do not validate this pinned OSS default: system versions and private optimizations differ. | [Mem0 paper](https://arxiv.org/abs/2504.19413), [repository README](https://github.com/mem0ai/mem0/blob/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3/README.md) | High on mismatch, unverified on claimed managed gains. |

**Transfer judgment.** Batch embedding and exact-content deduplication are cheap candidates. A true
lexical/vector candidate **union** followed by deterministic fusion is worth testing. Do not port
the semantic-first gate or entity fanout. Compare raw storage against extraction under total
lifecycle cost, because an ingest LLM call is paid before any query exists.

### MemOS OSS and OmniMemEval

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| The ordinary `general_text` default stores every raw message and embedding in Qdrant; search is pure vector retrieval. An extraction method exists, but `MOS.add` does not invoke it on this default path. | [`src/memos/mem_os/utils/default_config.py`](https://github.com/MemTensor/MemOS/blob/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad/src/memos/mem_os/utils/default_config.py), [`src/memos/mem_os/core.py`](https://github.com/MemTensor/MemOS/blob/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad/src/memos/mem_os/core.py), [`src/memos/memories/textual/general.py`](https://github.com/MemTensor/MemOS/blob/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad/src/memos/memories/textual/general.py) | High. The more elaborate mechanisms are not the default SDK path. |
| Tree memory adds LLM construction, Neo4j, vector/BM25/graph retrieval and reranking. Scheduler, activation/KV memory and reorganization are configurable but off in the general default. | [`src/memos/mem_os/utils/default_config.py`](https://github.com/MemTensor/MemOS/blob/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad/src/memos/mem_os/utils/default_config.py) | High for wiring; gains need a matched ablation. |
| The 2025 paper reports LoCoMo 75.8 and LongMemEval 77.8 for an older `MemOS-1031`, chosen using a best validation configuration. It gives no held-out component ablation at an equal final context budget. KV memory evidence is TTFT with a prebuilt cache, not answer quality. | [MemOS paper](https://arxiv.org/abs/2507.03724) | High as paper reading; author-reported, not independently reproduced. |
| Current MemOS cloud-service numbers in OmniMemEval do not establish pinned OSS quality. The harness uses a cloud endpoint and different answer/judge models and settings; LoCoMo category 5 is excluded. | [`configs`](https://github.com/MemTensor/OmniMemEval/tree/0b1ea8d28aa2d3e03ac4a6aee17b3006a131da7d/configs), [`scripts/locomo`](https://github.com/MemTensor/OmniMemEval/tree/0b1ea8d28aa2d3e03ac4a6aee17b3006a131da7d/scripts/locomo) | High on protocol mismatch; cloud internals unobserved. |

**Transfer judgment.** Do not import Neo4j or KV/scheduler complexity until a loss-location
diagnostic shows raw candidate recall is the bottleneck and an equal-budget ablation is positive.
The raw-message default is itself evidence that a strong raw baseline must be retained.

### EverOS (the user-named project)

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Markdown is the user-visible source of truth. SQLite holds internal state and a durable cascade queue; LanceDB stores derived dense/sparse indexes. | [`docs/how-memory-works.md`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/docs/how-memory-works.md) | High. This authority/derived split resembles MindBridge, though Markdown is authoritative there. |
| Default hybrid search retrieves sparse+dense episode candidates, fuses with RRF, applies calibrated linear ranking, then expands each episode into child atomic facts and globally competes them under `top_k`. No LLM reranker is enabled by default. | [`src/everos/memory/search/manager.py`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/src/everos/memory/search/manager.py), [`src/everos/memory/search/hierarchy.py`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/src/everos/memory/search/hierarchy.py), [`src/everos/memory/search/dto.py`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/src/everos/memory/search/dto.py) | High. Query cost: one embedding plus local sparse/dense retrieval and expansion. |
| Write creates a cell/episode with an LLM boundary call; atomic facts add asynchronous LLM calls per owner/episode. Profile extraction is on by default, reflection and foresight are off; foresight has no read consumer in this commit. Reads are eventually consistent with the cascade. | [`src/everos/memory/extract/pipeline/user_memory.py`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/src/everos/memory/extract/pipeline/user_memory.py), [`src/everos/memory/strategies/extract_atomic_facts.py`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/src/everos/memory/strategies/extract_atomic_facts.py), [`docs/how-memory-works.md`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/docs/how-memory-works.md) | High. Write cost is variable with episode fanout. |
| The checked benchmark is agentic LoCoMo, top-10, GPT-4.1-mini answer and GPT-4o-mini judge repeated three times; category 5 is excluded. README 93.3 is a sample report and no per-question result artifact was found in this checkout. | [`benchmarks/config.toml`](https://github.com/EverMind-AI/EverOS/blob/8754365c76daa2f13521fcd29a53044bba083403/benchmarks/config.toml), [`benchmarks`](https://github.com/EverMind-AI/EverOS/tree/8754365c76daa2f13521fcd29a53044bba083403/benchmarks) | Medium-high. Author-reported; not independent and not a category-complete LoCoMo score. |

**Transfer judgment.** The durable cascade/outbox and parent-to-source expansion are relevant,
but MindBridge already has an authoritative SQLite/outbox rule. Test global child/parent competition
under a fixed token budget. Avoid eventual visibility for the public write/read contract and avoid
logical scopes.

### EverMemOS (additional, not EverOS)

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| The system builds memcells, extracted facts, deterministic clusters/scenes, profile and foresight, with agentic/hybrid retrieval. The published evaluation disables foresight/profile but keeps clustering. | [`src`](https://github.com/NetMindAI-Open/EverMemOS/tree/806ad0555a09245ec90ad936e848bee9c64c6a49/src), [`evaluation adapter config`](https://github.com/NetMindAI-Open/EverMemOS/blob/806ad0555a09245ec90ad936e848bee9c64c6a49/evaluation/src/adapters/evermemos/config.py) | High for wiring. |
| Multimodal LoCoMo input is converted from images to BLIP captions before memory processing; this is not native image retrieval or adaptive image-token spending. | [`evaluation`](https://github.com/NetMindAI-Open/EverMemOS/tree/806ad0555a09245ec90ad936e848bee9c64c6a49/evaluation) | High for harness path. |
| Paper ablations report scene and memcell gains, but final token budgets differ materially among systems and there is no equal-budget ablation. Its reported LongMemEval 83 uses about 2.8k context tokens versus MemOS 77.8 at about 1.4k. | [EverMemOS paper](https://arxiv.org/abs/2601.02163) | Author-reported; causal gain unverified. |
| The paper reports large memory-layer call volume: on LoCoMo, add 7,056 calls/9.42M tokens and search 2,017 calls/4.45M tokens, before 1,540 answer calls/5.82M tokens. Runtime requires MongoDB, Elasticsearch, Milvus and Redis. | [EverMemOS paper](https://arxiv.org/abs/2601.02163), [`requirements`](https://github.com/NetMindAI-Open/EverMemOS/tree/806ad0555a09245ec90ad936e848bee9c64c6a49) | High as disclosed author measurements. Poor fit for an embedded runtime. |

**Transfer judgment.** Scene/memcell ideas need equal-budget, raw-baseline tests. The service stack
and per-memory LLM fanout violate “省” unless a measured query-volume break-even offsets ingestion
and maintenance: `C_total = C_ingest + N_query*C_query + C_maintenance`.

### TencentDB-Agent-Memory

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| In the inspected `feat/server_team` commit, L0 retains raw conversation, L1 stores LLM-extracted atomic memory, while L2/L3 consolidation is asynchronous. L1 extraction costs one model call and candidate-dependent deduplication can cost an additional call. | [`l1-extractor.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/core/record/l1-extractor.ts), [`l1-dedup.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/core/record/l1-dedup.ts), [`l1-writer.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/core/record/l1-writer.ts) | High for this commit/path. Variable 1–2+ write LLM calls, plus embeddings if configured. |
| In this MemoryCore path, zero-config embedding is `none`, so retrieval falls back to SQLite FTS5. With embeddings, recall takes FTS and vector candidates and fuses their **union** using RRF. | [`config.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/config.ts), [`memory-search.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/core/tools/memory-search.ts), [`auto-recall.ts`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src/core/hooks/auto-recall.ts) | High for this commit. Candidate union avoids the Mem0/MemPalace semantic-gate failure. Current web README recommends a multi-service/team form, so this is not a claim about every current deployment path. |
| TTL is optional/off. The inspected persistence uses JSONL plus a vector index; repair/cleaning can treat the vector side as truth, so it lacks MindBridge’s SQLite-authoritative transaction/outbox invariant. | [`MemoryCore/src`](https://github.com/TencentCloud/TencentDB-Agent-Memory/tree/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94/MemoryCore/src) | Medium-high; exact crash behavior was not fault-injected. |
| The persona “48→76” statement is not backed by an executable, pinned evaluation harness in the inspected tree. | [repository](https://github.com/TencentCloud/TencentDB-Agent-Memory/tree/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94) | High on absence in this commit; claimed gain unverified. |

**Transfer judgment.** Candidate union+RRF, provenance and stable prompt prefixes merit direct
tests. Reimplement them on SQLite plus the existing Zvec outbox rather than adopting its
consistency model. L2/L3 should stay off until incremental value pays for background calls.

### MemPalace official

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Product storage is verbatim by default, with ChromaDB as the default backend. Its general extractor is a separate rule-based tool, not default LLM rewriting. | [`README.md`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/README.md), [`mempalace/config.py`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/mempalace/config.py), [`mempalace/general_extractor.py`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/mempalace/general_extractor.py) | High. Zero memory-layer LLM calls on default write/read. |
| `search_memories` defaults to `candidate_strategy="vector"`: it retrieves roughly `n_results*4` semantic candidates, then hybrid BM25/vector scoring reranks only that pool. Lexical-only records can enter only through the non-default `union` strategy. | [`mempalace/searcher.py`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/mempalace/searcher.py) | High. Default product has the same semantic candidate ceiling as Mem0, although a union implementation exists. |
| Its headline raw LongMemEval result is a bespoke benchmark: each of 500 questions creates a fresh Chroma collection from that question’s ~session haystack, embeds only user turns, retrieves 50 and scores whether any labelled session is in top 5. The script does not call product `search_memories`. | [`benchmarks/longmemeval_bench.py`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/benchmarks/longmemeval_bench.py) | High. It validates raw Chroma retrieval on per-question corpora, not the default product pipeline or QA. |
| Committed artifacts contain 500 rows with raw R@5=0.966/R@10=0.982 and a 450-row v4 “held-out” artifact with R@5=0.9844/R@10=0.9978. The repository explicitly says these are retrieval recall, and admits the final 100% was tuned on three misses. | [`raw artifact`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/benchmarks/results_mempal_raw_session_20260414_1629.jsonl), [`held-out artifact`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/benchmarks/results_mempal_hybrid_v4_held_out_session_20260414_1634.jsonl), [`BENCHMARKS.md`](https://github.com/MemPalace/mempalace/blob/d9f059076c866fa6f29195679d75712436986024/benchmarks/BENCHMARKS.md) | Artifact arithmetic independently checked. Not answer accuracy; “held-out” is author-defined and the dataset is public/exposed. |

**Transfer judgment.** The strongest lesson is a raw, zero-LLM baseline. Test union mode rather
than the shipped semantic-only candidate gate. Do not treat per-question retrieval R@5 as evidence
of production full-corpus QA, and do not port hand-written benchmark-specific phrase boosts before
blind evaluation.

## Bounded source checks requested for the second wave

### VoiceMem

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Streaming speculative retrieval is real: once partial ASR text crosses a character threshold, `_kick` starts/cancels an async task; `_speculate` performs classification and search in a thread, and the final turn reuses the prefetched result even if trailing ASR text changed. | [`voicemem/stream.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/voicemem/stream.py) | High. This hides latency through overlap; it does not reduce retrieval compute and stale partial-query results are an explicit trade-off. |
| Default text search routes through slot/entity graph narrowing, relative-date query expansion, semantic Mem0 retrieval, lexical/time bonuses, and optional right-brain context. Candidate filtering may fetch up to 10,000 Mem0 hits and filter client-side. Cold archive is an explicit batch method, not automatic. | [`voicemem/orchestrator.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/voicemem/orchestrator.py), [`voicemem/leftbrain/brain.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/voicemem/leftbrain/brain.py), [`voicemem/leftbrain/mem0_backend_store.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/voicemem/leftbrain/mem0_backend_store.py) | High. “Local/fast” still carries potentially large scan/fetch amplification. |
| The public LoCoMo evaluator defaults to `left_brain_single`, synchronously ingests with `async_facts=False`, calls `vm.search` top-5, then GPT-4o-mini answers and a permissive GPT-4o-mini judge scores 152 questions. It does not exercise audio streaming/prefetch or the right brain. | [`evaluation/run.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/evaluation/run.py), [`evaluation/datasets/locomo.py`](https://github.com/xzf-thu/VoiceMem/blob/a450911fc8cbb44c46d810aace2f3288bad287e4/evaluation/datasets/locomo.py) | High. Therefore 91.2% and 134 ms are separate author-reported experiments, not one jointly validated end-to-end result. |

**Transfer judgment.** Speculative prefetch is useful for voice latency only if measured from
partial audio arrival to final answer and charged for cancelled/repeated searches. Relative-time
normalization is worth testing. The dual-brain/graph stack is not validated by its default LoCoMo
harness.

### RE-call

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Default retrieval embeds once, runs pgvector dense and PostgreSQL FTS candidates, fuses the union with RRF, optionally reranks the entire fused pool, then reports dense-score gap and staleness. The core path has no LLM call. | [`recall/retriever.py`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/recall/retriever.py), [`recall/trust.py`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/recall/trust.py) | High. PostgreSQL/pgvector and tenant/generation scopes do not fit MindBridge’s embedded/no-logical-scope contract. |
| Calibration is offline statistics over labelled answerable/unanswerable top-cosine samples: q05/q95-style boundary, logistic confidence mapping, AUC/separability interval and minimum class size. It is not an LLM calibration call. Optional entailment costs one judge pass per admitted hit. | [`recall/calibration.py`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/recall/calibration.py), [`recall/calibration_v2.py`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/recall/calibration_v2.py), [`recall/trust.py`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/recall/trust.py) | High. Requires representative labels per corpus/embedding generation. |
| The project itself reports the strongest counterexample: near-miss abstention fails (LongMemEval false-abstain 0.481; tested signals AUC ≤0.753), while far-gap abstention works. Its LongMemEval per-question-haystack hit@5 0.970 falls to 0.366 on one merged 19,195-session index. | [`docs/EVIDENCE.md`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/docs/EVIDENCE.md), [`results/FINDINGS.md`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/results/FINDINGS.md) | Author-reported artifacts, unusually explicit negative results. Do not generalize a fixed threshold to near misses or new corpora. |
| Its ATM-Bench recall claim is answer-model independent, but QS used DeepSeek V4 Pro while published baselines used Qwen3-VL-8B; the repository discloses that QS is not model matched. | [`docs/ATM_BENCH.md`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/docs/ATM_BENCH.md), [`docs/EVIDENCE.md`](https://github.com/GiulioDER/RE-call/blob/4b009bfba3416db80a10c00ca6f1e086c3f378a2/docs/EVIDENCE.md) | High on protocol; author artifact not rerun. |

**Transfer judgment.** Preserve provenance, explicit supersession and fail-closed calibration
artifacts, but treat abstention as a separately validated capability. A portable version would
bind a threshold to MindBridge’s embedding/index fingerprint in SQLite. It must abstain only where
held-out separability supports it; it should not add a per-hit LLM judge by default.

### InvMem / vanilla-rag-memory

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Default API stores raw, timestamped chunks and uses hybrid dense+BM25 union/RRF, optional result-neighbor expansion, and no LLM. Temporal enrichment and cross-encoder reranking are opt-in. | [`src/vanilla_rag/memory_store.py`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/blob/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca/src/vanilla_rag/memory_store.py), [`src/vanilla_rag/api.py`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/blob/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca/src/vanilla_rag/api.py) | High. |
| Search loads every user row/vector into a NumPy matrix and computes exact dot products; BM25 is also over all rows. This is a sound small-corpus baseline but O(Nd) dense compute and O(N) memory per query, not an indexed production design. | [`SQLiteMemoryStore.search`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/blob/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca/src/vanilla_rag/memory_store.py) | High. |
| Local benchmark defaults to top-100 and measures answer substring/evidence presence in returned context. The repository states it cannot reproduce the private official suite, locked answer model or evaluator. | [`scripts/local_benchmark.py`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/blob/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca/scripts/local_benchmark.py), [`docs/competition-contract.md`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/blob/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca/docs/competition-contract.md) | High. Not an end-to-end or official score. |

**Transfer judgment.** This is the most relevant boring control: raw dense, raw BM25, union RRF,
fixed total token budget. MindBridge already has a scalable derived index, so use this as an
algorithmic arm rather than copying its full-scan implementation.

### ReFind

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Default search is a four-action GPT-4o-mini ReAct loop. It repeatedly proposes keywords/date ranges, runs BM25 top-5, excludes prior chunks, takes notes and returns raw evidence with ±2 neighboring chunks. Each action is one LLM completion; the usual workflow can consume roughly four planning calls. | [`app/retriever.py`](https://github.com/imlrz/ReFind/blob/a80175ca0eeb52a938d7cab7a602bc780de8a577/app/retriever.py), [`app/config.py`](https://github.com/imlrz/ReFind/blob/a80175ca0eeb52a938d7cab7a602bc780de8a577/app/config.py), [`app/bm25.py`](https://github.com/imlrz/ReFind/blob/a80175ca0eeb52a938d7cab7a602bc780de8a577/app/bm25.py) | High. Add is deterministic/no-LLM. Query cost is high and variable. |
| BM25 combines conversation-chunk rank and aggregate session rank with RRF; the competition adapter omits the paper’s second answer-stage model and lets the shared platform answer. | [`app/bm25.py`](https://github.com/imlrz/ReFind/blob/a80175ca0eeb52a938d7cab7a602bc780de8a577/app/bm25.py), [`docs/METHOD_CARD.md`](https://github.com/imlrz/ReFind/blob/a80175ca0eeb52a938d7cab7a602bc780de8a577/docs/METHOD_CARD.md) | High. |
| The paper’s LongMemEval evidence is sample-limited and expensive: S=50/M=15 questions, five runs, about 69.8–99.2k input tokens/task and about five LLM calls. It does not establish cheap full-set SOTA. | [ReFind paper](https://arxiv.org/abs/2608.12888) | High as paper protocol reading; author-reported outcomes. |

**Transfer judgment.** Date-aware iterative lexical search is a useful diagnostic/oracle arm.
Production adoption should be gated by a cheap first-pass failure signal, capped in rounds/tokens,
and compared against deterministic query variants; otherwise it violates “快/省”.

### ActiveMemoryIndex

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Default write stores verbatim turns plus GPT-4o-mini extracted facts when a key is present; default query also asks the model for a user-voice recall rewrite. Retrieval is an in-process exact vector scan. It returns raw-first and expands selected raw turns to ±1 neighbors under the same top-k slots. | [`app/config.py`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/app/config.py), [`app/main.py`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/app/main.py), [`app/store.py`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/app/store.py) | High. Default costs at least extraction calls on add and one query-rewrite call/search when configured. |
| Agentic second search, event dating and chrono ordering are off. The source comments disclose that a second round changed results but did not improve complete multi-hop evidence; utterance-time numbering was rejected. | [`app/config.py`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/app/config.py) | High; useful negative evidence. |
| Its local harness retrieves top-100. It shows answer accuracy rises with returned characters; category 5 is excluded, public LoCoMo differs from `locomo_refined`, judge is undisclosed/different, and fact provenance receives more generous retrieval credit than raw turns. | [`bench/README.md`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/bench/README.md), [`bench/run_bench.py`](https://github.com/linxuhao/ActiveMemoryIndex/blob/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69/bench/run_bench.py) | High. Parent+7.8-type gains without equal tokens are not causal evidence; article itself reports a 5.4× context increase and equal-budget regression. |

**Transfer judgment.** Neighbor/source expansion should compete under one token budget and retain
raw provenance. Do not count a derived fact and its source as independent evidence. Avoid query
LLM calls until failure-bucket diagnostics show deterministic retrieval cannot recover the target.

### ChronoHybridMem

| Claim | Primary evidence | Confidence / boundary |
| --- | --- | --- |
| Default production construction enables structured query planning and evidence-need retrieval, but graph, adjacent-turn expansion, set-aware rerank, temporal bonus, session fusion and most experimental channels are off. Search may therefore use model query planning/ranking, but the advertised graph stack is not default. | [`app/main.py`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/app/main.py), [`app/storage.py`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/app/storage.py) | High. |
| The code keeps multiple lexical/dense/context/entity/graph channels with RRF and a model rerank pool capped at 30, but many knobs are explicitly controlled ablations. | [`app/storage.py`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/app/storage.py) | High. Complexity and calls depend heavily on flags. |
| The repository records negative graph/gate evidence: global graph regressed on a 20-question slice with only 0.72% relation coverage; four selective gate signals failed, including self-reported LLM confidence, which changed the prompt and reduced Hit@1. | [`docs/P5_SELECTIVE_GATE_EXPERIMENT.md`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/docs/P5_SELECTIVE_GATE_EXPERIMENT.md), [`docs/GRAPH_SELECTIVE_EXPERIMENT.md`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/docs/GRAPH_SELECTIVE_EXPERIMENT.md) | High as author artifacts; supports rejection rather than benefit. |
| Its positive “NEW” result is evidence retrieval on 1,976 LoCoMo queries with a local Qwen3-4B proxy, not official QA. Hit@10 improved 0.7601→0.7809, while Hit@1 CI includes zero; efficiency was not measured. | [`docs/CHRONOHYBRIDMEM_RESULTS_FOR_PAPER.md`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/docs/CHRONOHYBRIDMEM_RESULTS_FOR_PAPER.md), [`scripts/evaluate_locomo_retrieval.py`](https://github.com/Tin11Mn/chrono-hybrid-mem/blob/f6c1f98663025a28cf5ab8d1ed08ea62484b3972/scripts/evaluate_locomo_retrieval.py) | Author-reported, reproducible harness; no equal official answer-model result. |

**Transfer judgment.** Evidence-need queries can be a bounded second-pass experiment. Graph and
confidence gating should remain rejected until coverage and paired, equal-budget gains reverse the
current negative results. The useful reusable artifact is its failure taxonomy and negative-result
discipline, not its accumulated feature surface.

## Cross-system candidates and falsifiers

| Candidate mechanism | Why it may help MindBridge | Cheapest decisive test | Disconfirming result |
| --- | --- | --- | --- |
| Raw dense + raw BM25 union/RRF | Rescues exact names, IDs and rare terms without LLM calls; compatible with SQLite/Zvec. | Same candidate/token budget: dense, BM25, union/RRF across exposed dev set, then untouched clusters. Report candidate recall and QA. | Union adds latency/tokens but no paired QA or gold-evidence gain, or displaces dense hits. |
| Source-aware parent/child expansion | Derived facts can recover their raw source and neighboring evidence; preserves provenance. | Fixed token budget A/B/C packing, deduplicate by final source footprint, score official QA plus gold evidence. | “Completeness” rises while QA is flat/down, or gains disappear at equal tokens. |
| Relative/dual time normalization | Helps temporal queries where relative phrases do not align with normalized dates. | Freeze parser rules; compare retrieval and QA by temporal subtype with malformed/missing time cases. | Non-temporal regression, stale-state errors, or no temporal lift after equal budget. |
| Bounded query decomposition/second pass | May recover multi-hop/channel misses. | Trigger only on predeclared failure proxy; cap one extra round; compare deterministic and LLM rewrites including all calls/tokens. | Trigger AUC is weak, fixed-depth is as good, or query cost has no break-even. |
| Calibrated abstention/supersession | Prevents unsupported or stale answers when evidence is clearly absent/obsolete. | Per-index fingerprinted labelled calibration; report answerable false-abstain and unanswerable false-confirm CIs separately. | Near-miss distributions overlap or threshold fails on new cluster/index generation. |
| Voice speculative prefetch | Hides search behind ongoing speech. | End-to-end audio timing with cancelled searches and partial/final query drift counted. | No user-visible latency reduction, increased wrong retrieval, or material wasted compute. |
| Cheap multimodal index then selective raw-media escalation | May retain quality while paying image/audio tokens only when needed. | Caption-only vs raw-image/audio vs gated escalation, fixed total multimodal token/I/O budget. | Gate misses necessary media or spends near raw-media cost without QA gain. |

Across all candidates, use `C_total = C_ingest + N_query*C_query + C_maintenance` and state the
break-even query count before calling an approach cheaper. Retrieval recall, source completeness
and latency are separate outcomes; none substitutes for matched-model answer quality. Strong raw,
no-memory, oracle-diagnostic and random controls are needed to prevent reward hacking.
