# MindBridge memory backend 深度审计与研究路线

日期：2026-09-07。代码审计基于 MindBridge `c62a1c41de442a8b07daa9108c12851148a4f128`；
竞品源码固定到下文列出的 commit。本文把证据分为四级：**源码执行路径**、**本次实测**、
**作者自报**和**待验证假设**。只有前两类能证明当前代码或本次环境中的事实；作者排名、论文表格
和设计推测不会被改写成 MindBridge 的已验证收益。

## 结论

MindBridge 已有成熟的工程底座：SQLite 是记录、embedding、元数据和 durable outbox 的
权威来源；Zvec 是可重建索引；raw 先提交，formation 后派生；搜索可组合 dense、lexical、
类型、双时态、地点和身份条件；音频、图像、情绪、衰减、纠错、遗忘、回滚与上下文编译也都有
真实代码路径。它不是一个“只有向量库”的 memory backend。

问题主要在能力之间的连接与收益证据。`ask` 没有消费 `compile` 的预算化计划；formation 写下
`evidence_ids`，主检索却不回查来源；自适应检索主要是候选窗口加深，没有依据证据缺口、模态缺口
或答案失败类型决定是否二次检索；产品也没有通用的来源顺序/邻接契约。已有 benchmark 多数对应
旧 commit、旧 9B/4096 embedding 或非官方 judge，部分有运行错误或不完整 controls。因此当前
不能称 SOTA，也没有证据证明现有全部能力合起来同时“更强、更快、更省”。

竞品研究给出的最稳妥结论也不是继续堆图或 LLM 抽取。MemPalace、InvMem、RE-call 等项目反复
显示 raw memory 是必须保留的强基线；ABot 与 ChronoHybridMem 的负结果说明图可能降低质量；
ActiveMemoryIndex 的父块收益在等字符预算后反转；ReFind、EverMemOS 的高调用量说明复杂检索
可能把成本转移到 query 或 ingest。近期最高优先级应是：建立强 raw 对照与真实 token 预算，定位
“未召回、进入候选后被挤出、已 grounding 但模型未使用、冲突/时序表达错误”中的实际损失位置，
再做有界的来源、时间和模态恢复。

## 当前能力与五个连接边界

| 能力 | 当前执行路径与默认状态 | 审计判断 |
| --- | --- | --- |
| 持久化与恢复 | [`Memory.add`](../../src/mindbridge/memory.py)、SQLite 事务 → outbox → Zvec flush → 精确 ack；索引缺失时由 SQLite embedding 重建 | 工程闭环完整，是后续算法实验应复用的底座 |
| Dense 与 lexical hybrid | aggregate 加最多七个 retrieval key；Zvec dense/FTS 候选，经 coverage/rank/gate | 默认接线；召回导向，但额外 key、窗口和延迟没有广泛收益消融 |
| 类型、双时态、地点、身份 | `RetrievalScope(valid_at, known_at, ...)` 经索引下推和 SQLite hydration | 能表达资格边界；不能由单元测试推导广泛 QA 增益 |
| 原生多模态 | 内容寻址媒体、可选 ASR、视觉描述、face/speaker/affect | 后端配置后可用；视觉描述会增加写入成本，现有结果未隔离其贡献 |
| Capture → settle | `capture` 先做快速持久化，`settle` 再完成可搜索记录 | 主机显式选择；capture 成功不代表 search 已可见，漏 settle 会形成产品层断点 |
| Memory role | 基础 `MemoryType` 为 semantic、episodic、procedural；formation 还可派生 affect、trait、response policy 等类型 | 类型可写可编译；formation 默认关，`ask` 又不调用编译器，所以尚无端到端角色收益证据 |
| Typed formation | raw 写入后由模型生成 observation/entity/event/state/relation 等派生项并保存来源 ID | 默认关闭；一条写入模型调用起步；没有有效 A/B 证明净收益 |
| 衰减与强化 | answer 默认强化访问；decay 默认关闭 | 影响排序，不负责 relevance admission；没有跨任务质量证明 |
| Consolidation/纠错/遗忘/回滚 | 主机显式触发、验证 operation log、可逆控制 | 审计能力强；不是自动学习闭环，成本与 QA 收益未证实 |
| `compile` | 按角色与字符预算选择 section | 显式 API；当前 `ask` 不调用它 |

五个已确认边界如下。

1. `ask` 走 `_search_prepared → _grounding_hits → generation`，不经过 `compile`。二者有不同
   语义，不能盲目合并，但 search/compile/ask 最终没有共享同一份预算证据计划。
2. formation、vision 与 consolidation 都是 opt-in。历史 `r0905-*` 配置中 formation 和
   consolidation 为 `null`，所以那些分数没有验证这些算法。
3. provenance 在写侧真实存在，普通 read path 不读取它。`compile` 仅做 affect 到同源
   event ID 的轻量关联，并不 hydrate event 内容，也不能证明因果。
4. 当前 adaptive depth 主要在 scope 过滤后候选不足时扩窗，另有时间 lexical deepening；它
   不是依赖、答案缺口、模态覆盖或预算效用驱动的恢复。
5. 产品 metadata 是应用数据，没有 session/turn/ordinal/adjacency 的公共语义。benchmark 的
   `D2:x` 等来源标记只属于评测，任何产品算法都不能据此解析邻接或先后。

这意味着“算法存在”与“能力有效”必须分开回答。双时态和 provenance 提供了正确性基础；图式
formation、衰减、情绪和编译器则需要通过公共 SDK 的端到端实验，才能证明它们让答案更好而非只让
内部结构更丰富。完整源码审计和单元验证记录保存在本地研究收据
`.benchmarks/research/2026-09-07-memory-audit/code-audit.md`；其中 24 个 search-index、13 个
interaction、233 个 memory API、154 个 control-plane/context 测试均通过，但这些是回归保护，
不是 SOTA 证据。

## 竞品实现、成本与可迁移结论

下表使用官方仓库，不把 README 排名当独立验证。EverOS 与 EverMemOS 是两个项目；MemPalace
使用官方 `MemPalace/mempalace`，不是社区 fork。

| 系统（固定 commit） | 默认写入与检索 | 成本/评测边界 | 对 MindBridge 的取舍 |
| --- | --- | --- | --- |
| [Mem0 `dae67f7`](https://github.com/mem0ai/mem0/tree/dae67f74f5cc7bf138c7d7d6f9cec5ce4b4373b3) | OSS 默认 OpenAI LLM/embedding + Qdrant；v3 写入一次 LLM 抽取且仅 ADD。搜索先 semantic overfetch，再加 BM25/entity 分；lexical/entity-only 文档不能进入候选 | 实体最多 8×500 fanout；无默认 reranker/自动遗忘。论文与 managed v3 不能证明此 OSS commit | 试 batch embedding、精确去重、真正 dense/BM25 候选 union；拒绝 semantic gate 与实体大 fanout |
| [MemOS `78a372a`](https://github.com/MemTensor/MemOS/tree/78a372a4fc853a24d2a78efa3b4bbbd27ab9f7ad) | 普通 `general_text` 实际是 raw message → Qdrant → pure vector；tree 才引入 LLM、Neo4j、BM25、图与 rerank | 论文是旧 `MemOS-1031`，无等预算组件消融；当前 OmniMemEval 使用 MemOS 云服务评测路径，非 OSS 默认 | raw 默认本身是强基线证据；不引 Neo4j/KV/scheduler，除非损失诊断支持且等预算为正 |
| [EverOS `8754365`](https://github.com/EverMind-AI/EverOS/tree/8754365c76daa2f13521fcd29a53044bba083403) | Markdown truth、SQLite 状态/队列、Lance 派生索引；dense+sparse RRF 后 episode→atomic facts 全局竞争 | episode/atomic fact/profile 带多个写入 LLM；read 最终一致。LoCoMo top-10 排除 category 5，93.3 是样例报告 | durable cascade 与来源展开值得比较；MindBridge 已有更严格 SQLite/outbox，不能换成最终一致或 logical scope |
| [EverMemOS `806ad05`](https://github.com/NetMindAI-Open/EverMemOS/tree/806ad0555a09245ec90ad936e848bee9c64c6a49) | memcell/fact/scene/cluster/profile/foresight 与 agentic retrieval | 论文披露 LoCoMo add 7,056 calls/9.42M tokens，search 2,017/4.45M；依赖 Mongo/ES/Milvus/Redis；图片先变 BLIP caption | 场景与 memcell 必须做等 token raw 对照；服务栈和调用量不适合 embedded “省”目标 |
| [TencentDB-Agent-Memory `2ee2239`](https://github.com/TencentCloud/TencentDB-Agent-Memory/tree/2ee22397f6091b8cd3ea847bc1edb04d3bec0c94) | 本次固定的 `feat/server_team` MemoryCore 路径为 L0 raw、L1 一次抽取并可能再一次 dedup LLM、L2/L3 后台；无 embedding 时 FTS5，有 embedding 时 dense/FTS union RRF | TTL 默认关；该路径一致性弱于 SQLite authority；persona 48→76 未找到固定 harness。当前 web README 推荐多 service/team 形态，不能把此 commit 的 zero-config 概括为所有部署 | 移植 union RRF、provenance、稳定 prompt prefix；在现有事务/outbox 上重写 |
| [MemPalace `d9f0590`](https://github.com/MemPalace/mempalace/tree/d9f059076c866fa6f29195679d75712436986024) | 默认 verbatim + Chroma；`search_memories` 默认 vector candidate，BM25 只重排语义池，非默认 `union` 才能救 lexical-only | LongMemEval raw 96.6% 是每题重建 Chroma、只存 user turns、取 50 后算 retrieval R@5；不调用产品 search，也不是 QA。450 题 v4 为 98.44% R@5，但公开数据已暴露 | 最重要的是 raw/zero-LLM 对照与 union；不能把 per-question R@5 当生产全库答案能力 |
| [VoiceMem `a450911`](https://github.com/xzf-thu/VoiceMem/tree/a450911fc8cbb44c46d810aace2f3288bad287e4) | 音频 partial ASR 达阈值后异步取消/重启 classify+search，句末复用预取；文本路径有 slot/entity narrowing、时间扩写、semantic+lexical/time bonus | LoCoMo evaluator 实际 `left_brain_single`、同步 `vm.search` top-5、152 问，没跑 stream/right brain。论文 91.2/430 tokens/134 ms 是分离的质量/上下文/延迟测量 | voice prefetch 可测试，但要计取消搜索和 partial/final 漂移；不能把重叠隐藏延迟写成计算减少 |
| [RE-call `4b009bf`](https://github.com/GiulioDER/RE-call/tree/4b009bfba3416db80a10c00ca6f1e086c3f378a2) | pgvector dense + PostgreSQL FTS union RRF，校准 threshold/confidence 与显式 supersession；核心无 LLM，entailment 是逐 hit 可选调用 | 自己证伪 near-miss：LongMemEval false-abstain .481、信号 AUC≤.753；每题约49 session 的 hit@5 .970 在合并19,195 session 后降至 .366 | 可在 SQLite 绑定 embedding/index fingerprint 做校准与 supersession；近失配不可套固定阈值，且其 tenant/generation scope 不移植 |
| [InvMem `31ab7bf`](https://github.com/wenxiaof345-ctrl/vanilla-rag-memory/tree/31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca) | raw + timestamp，dense/BM25 union RRF，邻接窗口；默认无 LLM | 每 query 加载全用户向量做 NumPy 点积并全量 BM25，是 O(Nd)/O(N) 小库实现；local benchmark top-100 只算答案/证据是否在 context | 作为 raw dense/BM25/RRF 控制臂，不复制全扫基础设施 |
| [ReFind `a80175c`](https://github.com/imlrz/ReFind/tree/a80175ca0eeb52a938d7cab7a602bc780de8a577) | 默认 GPT-4o-mini 四步 ReAct：关键词/日期搜索、BM25 top-5、排除已见、记笔记；返回 ±2 context | 论文 LongMemEval 仅 S=50/M=15、五轮，约5 calls、70–99k tokens、41–42s/query；不是全500或低成本 | 作为 date-aware 二次检索/oracle；生产只允许预注册触发器、一次上限和完整计费 |
| [ActiveMemoryIndex `2e6d76c`](https://github.com/linxuhao/ActiveMemoryIndex/tree/2e6d76cae4454c2f9c722d0c927aaf0825f1cd69) | raw + LLM facts；query LLM recall rewrite；exact-vector scan；raw-first 与 ±1 neighbor 默认开 | 自己报告父块 .711 vs .633 使用 5.4× context，等 chars 仅 .572；±1 .6333→.6802 也多 8% chars。local LoCoMo 排除 category5，fact provenance 评分更宽 | 试同预算 source/neighbor 包；derived 与 raw 不能算独立佐证 |
| [ChronoHybridMem `f6c1f98`](https://github.com/Tin11Mn/chrono-hybrid-mem/tree/f6c1f98663025a28cf5ab8d1ed08ea62484b3972) | 默认 structured query plan + evidence-need retrieval；graph、adjacent、set-aware、temporal 等多数关闭 | global graph 在20题退化；P2 set cover 在35 synthetic Hit@1 1→.8571；四种选择门控均失败。正向 1,976 问只是 local Qwen3-4B evidence retrieval，非官方 QA | 借鉴失败分类与有界 need query；图与 confidence gate 保持关闭，直到覆盖率和配对等预算结果反转 |
| [ABot-AgentOS 论文](https://arxiv.org/html/2607.10350v1) | typed 时空实体图，hybrid seeds → bounded typed expansion，再做 source compression 与 split-wise evolution | 代码尚未发布，只能核论文。Table 4 的 Mem-Gallery LLM judge 同 Qwen3.6-Plus 下 RawRAG 90.2 > ABot Static 88.6；92.6 是 Qwen/GPT full-context 参考，不是 self-evolution 分数。该对照也未固定等 token，故只说明图不必然更优。OpenEQA 24 帧 direct GPT-5.4 74.1 > static 59.9。Table 2 LoCoMo Static 87.5 > 其复现 Mem0 85.6，但 judge 协议不是 LoCoMo-refined | 说明图收益依赖任务；typed expansion/evolution 只是待验证假设，先要求 raw 与静态图同模型、同 token 的消融 |

四篇微信文章分别映射到 [InvMem 文章](https://mp.weixin.qq.com/s/c1PZpb2O8IwPi_bHjI5fnA)、
[ReFind 文章](https://mp.weixin.qq.com/s/xG-qFWgkEYBOrxfA_Ezuvg)、
[ActiveMemoryIndex 文章](https://mp.weixin.qq.com/s/UHokeVam5eDop8Fq4JdLzQ) 和
[ChronoHybridMem 文章](https://mp.weixin.qq.com/s/zSbIjyDeJBNCzWean_vpZQ)。本文结论以其官方源码和
论文为准；文章中的分数只有在对应执行入口、样本、预算与 judge 一致时才被采用。

还有三条重要反证。ABot 论文的 Mem-Gallery LLM judge 同 Qwen3.6-Plus 下 RawRAG 90.2 高于
ABot Static 88.6；同表 92.6 是 Qwen/GPT full-context 参考，不能归因 self-evolution。OpenEQA
24 帧中 GPT-5.4 direct 74.1 高于 static 59.9。另一方面，其 LoCoMo Table 2 Static 87.5 高于
复现 Mem0 85.6，但不是 LoCoMo-refined judge 协议。代码尚未发布，故只能视为
[任务依赖的作者消融](https://arxiv.org/html/2607.10350v1)。[InMind](https://arxiv.org/html/2607.24368v1)
的 125 个 implicit-association 任务显示“记住”不等于“会用”：memory agents 最大 14.4，oracle
84，但不同 baseline 不能证明通用算法。[V-Mem](https://arxiv.org/html/2608.01543v1) 的模态路由
和同轮保留值得测试，不过 caption 是数据已提供，不是免费生成；其约 10,086 answer input tokens
高于 A-Mem 959，且 Qwen2.5-VL-7B .509 低于 Omni .534，说明结果依赖模型与预算。

## 历史 benchmark 信任审计与本次 19 题 smoke

历史结果不能当当前基线：多数运行在 `6774d1b...`，使用 WeMM-Embedding-9B/4096；当前 HEAD
为 `c62a1c4...`，本次服务为 `tencent/WeMM-Embedding-2B`/2048。旧 LongMemEval 350 和全部
Mem-Gallery 已暴露，不能再叫 holdout；PersonaMem 旧论文/旧数据也不能代替 PersonaMem-v3。
历史 PersonaMem、EgoLife 结果甚至低于 blind，若只展示成功任务会形成 reward hacking。

| 历史 artifact | 范围与旧结果 | 关键限制 |
| --- | --- | --- |
| LoCoMo dev | 4 units/525题，本地 judge .760 | 仅四个 cluster，非官方 judge |
| LME dev / 旧“holdout” | 150题 .718；350题 .804 | 各有 answer errors；350 已暴露，不能再作最终集 |
| Gallery dev | 6 topics/436题，deterministic F1 .696 | 两个 answer errors，仅六个 cluster |
| MemLens 32k dev | 60题，本地 judge .200 | controls 不完整 |
| M3 Robot dev | 25视频/314题，本地 judge .283 | 无 gold retrieval IDs；另一次 candidate .471 不是有效同 commit A/B |
| PersonaMem-v3 dev | 8 users/1,253题，本地 .337，blind .424 | product 低于 blind；两臂各两个错误 |
| EgoLife dev | 1 unit/150题，accuracy .288，blind .340 | product 低于 blind且有25个 product errors |

本次冻结 smoke 预先按来源顺序选 19 题，使用公共 SDK、物理目录隔离、无 harness response-cache
命中、0 ingest/answer/retrieval failure；provider KV/prefix cache 不可观测。它只比较当前
raw-hybrid MindBridge 与无信息 blind，formation 关闭；
因此能证明 memory 提供了信息，不能证明当前算法胜过 strong raw，更不能外推竞品或 SOTA。

| 任务 | MindBridge | Blind | 口径 |
| --- | ---: | ---: | --- |
| LoCoMo-refined，8题 | .875 | .125 | 本地 Qwen judge；非官方 |
| LongMemEval-S，3题 | .6667 | 0 | 本地 Qwen accuracy；非官方 |
| Mem-Gallery，8题 | .7298 | .4702 | deterministic F1；另有本地 judge .875 vs .625，不与 F1 混合 |

`controls_complete=true` 只表示这次选择的 blind control 完成，没有运行 random 或 oracle。小样本
p95 只是观察值，不是 SLA。5090 是 32GB 客户端机器，VLM/embedding 推理由远端服务完成；本地
1.075 Wh 大多是显卡基线功耗，不能声称为 Qwen 推理能耗。2B/2048 与旧 9B/4096 结果也不能把
差异直接归因于算法。

逐题审计说明下一步不能只扩大 top-k。LoCoMo `conv-26#q0003` 的 gold 已 rank 2 且进入 12 条
grounding，答案包含正确 adoption agencies，却混入 counseling/career，被当前 judge 判 0；这是
答案精度和 judge 敏感性。LME `51a45a95` 的 gold turn 不含商店名，紧邻 rank 2 grounding 中
出现两次 Target，模型仍拒答；这是 gold span 不足与答案/abstention 损失。Gallery 第 8 题两条
gold 都 grounding，但同日、offset 都为 0，检索相关性顺序与来源顺序相反，最终把 after 答成
before；这是 chronology 表达/排序或推理损失。三者不能被统一归因于召回不足。

本次收据固定了 `results.jsonl` SHA-256
`1e976c1ebf9147191f8efcd676d570dbef20596d736432d264869b3e60f867db` 与
`samples.jsonl` SHA-256
`421cf19e42f5eaa71af2ef76e158381dc18bbe962e73692a3f0cccdb219d3b2f`；完整本地记录在
`.benchmarks/research/2026-09-07-memory-audit/AUDIT.md`。

本次成本必须分账，不能把远端模型、客户端资源和证据 token 混为一个数字：

| 项目 | 本次观察值 | 解释边界 |
| --- | ---: | --- |
| ingest | 2,180 records，36 successful batches | LoCoMo 7、LME 26、Gallery 3；各任务 batch p50 分别 860、1,751、4,155 ms，n 很小 |
| embedding | 55 requests / 760,120 input tokens | 含 ingest 与 query；Gallery 有 108,329 unattributed mixed-media input，图像 token 未知 |
| MindBridge generation | 19 calls / 77,207 input / 369 output | 上下文还含 prompt/多模态序列化，不是纯 evidence token |
| Blind generation | 19 calls / 2,653 input / 231 output | 三任务 input 为 719、275、1,659；只证明无 memory 上下文对照完成 |
| judge | 39 calls / 22,014 input / 1,101 output | `conv-26#q0005` 有两个可接受 reference，故比预估多一次；非官方 Qwen judge |
| LLM 合计 | 77 calls | generation 共 38 calls；judge 39 calls |
| wall time | 86.52 s | 包含完整 invocation；不等于稳定服务吞吐或远端模型 latency |

Answer end-to-end latency 含 request admission、本地 retrieval 和远端 generation。以下 p50/p95
只来自 8、3、8 道题，是本次观测值，不是 SLA 或稳定服务间比较：

| Task | MindBridge p50 / p95（ms） | Blind p50 / p95（ms） | 每臂 n |
| --- | ---: | ---: | ---: |
| LoCoMo-refined | 3,210 / 5,053 | 794 / 1,000 | 8 |
| LongMemEval-S | 958 / 1,199 | 139 / 580 | 3 |
| Mem-Gallery | 5,540 / 13,006 | 1,206 / 1,505 | 8 |

## 创新假设：来源约束的预算化证据组装

差异化方向暂称“来源约束的预算化证据组装”。它利用现有 SQLite 权威 raw、双时态与派生
`evidence_ids`，使有损派生表示可以回查、按需恢复，并让 search/compile/ask 最终消费同一份
证据计划。它不是新建 graph database，也不增加隐式 session/account scope。

候选池固定后比较三种 selector：A 按 rank 装入完整单条；B 把 derived 与全部可读的一跳来源
作为不可拆包；C 先按最终 source-ID footprint 分组去重，保留最高 rank 可行 anchor，再按
`sum(1/(60+unique_rank))/新增 token` 贪心加入尚未覆盖的独立 footprint group；`unique_rank`
是在完全相同最终可读一跳 source-ID footprint 去重后重新分配的秩，共享来源只收费一次。预算必须包含 ID、
日期、labels、记录边界并使用 Qwen 服务的真实 tokenizer。summary 与 raw 不算两份独立佐证，
重复 summary 不增加权重；上游 top-k 被重复项挤占仍是单独的 upstream failure。该方法只是
utility surrogate，没有 submodular 或最优保证。

安全边界先于质量。原型规则要求只通过公共 `get` hydrate，并把 forgotten、deleted、evicted、时间
无效来源记为 unresolved，不能从 store 内部偷偷恢复；其中 evicted 与 `known_at` 历史读取尚未实测。
当前 evidence link 没绑定 source version，公共 raw 又没有 content update，所以不能伪称已复现串版；
所有原型包必须标 `source_version_unverified`。未来需要通用
provenance/sequence contract，不能利用 benchmark 的 `D2:x` 字段。

已完成的 synthetic 一跳实验只验证 forgotten、deleted、`valid_at` 过期三种来源可见性及整包预算，
不证明 QA 提升。在 598-token evidence-package
预算下，A/B/C 最终序列化为 578/487/487 tokens；599-token 长来源整包被跳过，forgotten、deleted、
`valid_at` 过期来源分别成为 unreadable/missing/unreadable。`known_at` 历史 source/version closure
未测，因此这不是完整时态安全证明，也没有形成产品 API。

真实开发诊断固定
`conv-26 session_1[:12]`、四个预先问题、top-64、2,048-token、temperature 0、128 answer tokens；
mixed A/B/C 共用一个 public search snapshot，raw-only 使用独立 12-raw formation-off store。真实
Qwen formation 12 次得到 13 个 derived，独立 HTTP answer harness 运行 16 次，不是产品 `ask`，
也不用 LLM judge。

首轮实现按单记录 token 相加，后来改为完整序列化包计费，导致 3/16 个上下文选择改变。因此旧
16 个答案被验收排除；这是实验 selector 的计费错误，不是产品 bug。随后用同 snapshot、同 prompt、
0 formation 做了修正后的 16 次 answer：每个最终 evidence package 均断言不超过 2,048 tokens；实际
evidence 30,564 tokens，服务报告 prompt 31,460、completion 141，共 31,601 tokens。2,048 只是
evidence-package 上限，system/question wrapper 另计，不能写成 total-prompt 上限。

修正收据中，两个可回答开发问题四臂均正确，没有 B/C 超过 raw-only/A 的例子；另外两个问题的
制定过程看见了 `D1:14/D1:16`，但 fixture 只 ingest 到 `D1:12`，所需事实实际不在任一 store。
它们是两个 invalid fixture rows，
不能丢掉后宣称 2/2，也不能解释为 retrieval/image failure。本实验因此不给算法质量推荐，候选停止
晋级，不再换题追求正结果。严格 replay 的 404 个 tokenizer cache miss 是离线组合枚举成本，进一步
说明当前 selector 不能直接上线或宣称更快。修正结果 SHA-256 为
`4199b0d69c5dd21a160f7b013ca808afc24e51a683d365d607c333f504dba167`；runner 最终 hash 见验证节。

后续假设按损失类型推进，不预先承诺收益：来源闭包过长时比较 raw、summary 和有 offset/version
可追溯 span 的多表示选择；多模态先检索便宜索引，只在缺失视觉证据时提升原图/时间片分辨率，
把 image token、I/O 与 description 全部计费；只在代理指标证明缺口时做至多一次二次检索，并在
未见 cluster 验证触发校准，绝不把 LLM 自信当真值；固定小预算下测试重要状态的隐式关联，同时
量化过度个性化与陈旧状态伤害。

## 分层迭代与反 reward-hacking 门槛

每个实验一次只改变一个因素。预注册 arms 至少包括 raw-dense、raw-BM25、union-RRF、formation
on/off、A/B/C、固定/自适应 depth、caption/raw-image、时间版本与拒答。先在已暴露 development
cluster 上用每能力固定小 slice 做机械诊断；1–2 天内完成小 split 消融只是合理节奏，不是收益或
SOTA 时间保证。只有配对收益且无安全回归的候选才进入完整评测，昂贵视频任务最后运行。

八套 benchmark 各自承担不同能力，不能各取 post-hoc 最佳结果拼成一个 SOTA：

1. [LoCoMo-refined](https://github.com/mem-eval-suite/LoCoMo_refined/tree/887091190789e8d6760e70b9edd696539923dc4f)：多跳、时间、跨 session 对话；官方 judge 是 `Qwen/Qwen3-14B`。
2. [LongMemEval-S](https://huggingface.co/datasets/xiaowu0162/longmemeval/tree/2ec2a557f339b6c0369619b1ed5793734cc87533)：500 QA 的更新、跨 session 与拒答；其中 30 道 abstention 问题，official `gpt-4o-2024-08-06`；不与 V2 混报。
3. [Mem-Gallery](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery/tree/af912daba984e896e253016b7c7e334ef92c2a6f)：文本+图像与顺序；deterministic F1 和 judge 分开。
4. [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS/tree/afa101a1907cc37db40b50d649547964387b96b7)：32–256k 视觉密集；memory-agent 固定 195 题，不能与全 789 混报。
5. [PersonaMem-v3](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3/tree/7b00a090b35b7293e6efeeb19494207f32b5a9ee)：推荐、主动性和过度个性化；旧 PersonaMem 结果无效。
6. [ATM-Bench](https://huggingface.co/datasets/Jingbiao/ATM-Bench/tree/78e826dc07e97466b2f54443831ef9a83ab8b27c)：跨应用四年长期记忆和时间查询；不同 judge/媒体处理的 leaderboard 不直接比较。
7. [M3-Bench Robot](https://github.com/ByteDance-Seed/m3-agent/tree/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c)：机器人视频、身份与空间多模态。
8. [EgoLifeQA](https://huggingface.co/datasets/lmms-lab/EgoLife/tree/143fb319be7aa5ae210c936bf4f0f3a86092afb0)：长视频生活情节与因果截止。

最终门槛是：冻结数据、模型、commit、prompt 与 hashes；按 conversation/user/video 划分开发、验证
和最终集，已暴露数据永不转正；official judge 加独立 judge/人工抽查；全部计划题进入分母，错误
按零做敏感性分析；同时运行 blind、strong raw、oracle（只诊断）与 random control；物理隔离并
遵守 causal cutoff；按 cluster 做 paired bootstrap CI，并用至少三个 seed 评估模型噪声。上下文
报告真实文本 token、图像/audio、写/读/维护调用，冷热 cache、P50/P95 均报告。storage/search
1e4/1e5 microbenchmark 每例独立临时目录，只有纯存储测量可直接调用 local adapter；所有行为评测
走公共 SDK。

“更省”按生命周期成本判断：

```text
C_total = C_ingest + N_query * C_query + C_maintenance
```

只有给出新增写成本、每读节省与 break-even query count，才可声称摊销更省。retrieval R@5、来源
完整度、QA、延迟和成本分别报告 Pareto；任一普遍能力回归不能靠提高权重或混合指标隐藏。达到这些
门槛后，才能讨论某个任务上的 SOTA，不能宣称一个跨不同指标的单一“全域 SOTA”。

## 可复现范围与当前限制

本轮完成源码执行路径审计、历史 artifact 信任审计、19 题冻结 smoke、竞品官方源码/论文入口
核验和来源约束 selector 的结构诊断；没有修改产品代码、默认值、依赖、schema 或公共 API。完整
竞品 claim-source ledger 已复制为可跟踪的
[源码与证据附录](memory-backend-sources-2026-09-07.md)，包括每个入口的 commit permalink、调用复杂度
和反证方法；本地 scratch 另保留完整调查过程。当前没有公平运行所有竞品，也没有 official-judge
的完整未见集结果，因此不做竞品排名。

冻结协议的可复核 manifest 如下。选择均为指定 unit 中官方 source order 的首批 question ID，不按
label、category、answer 或 retrieval 结果筛选；retrieval `recall_limit=12`、candidate limit 36；
unit/request/judge concurrency 均为 2。embedding 固定 `tencent/WeMM-Embedding-2B`、messages
格式、2048 维、0 retry、90 秒 timeout；generation/judge 固定 `Qwen3.8-27B`、temperature 0、
`do_sample=false`、seed 0、thinking off、0 retry、90 秒 timeout。协议在
`2026-09-07T17:22:44+08:00` 冻结，`selection_sha256` 为
`1dc54b0dbf1f4861a462f97c9397f1f869e649db3ca588aa8f3f24ba4c596eff`，运行 arms 为
`[mindbridge, blind]`。

| Task | Dataset / full evaluation / filtered evaluation SHA-256 | Unit、question 与 ingest |
| --- | --- | --- |
| LoCoMo-refined | `1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17` / `7033db68062e83eac67762d211810fcad470c509a37d94cee6596b669e0e2d3a` / `24bc77d13b84645a4ad24021dd7d512ff38bb7233be645d23e468f375f794ee3` | `conv-26#q0000..q0007`，419 records |
| LongMemEval-S | `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894` / `dcc3815b9d7e7c2c280545dfb45ad86f9a344130b066f51fc06d616e7d1925f7` / `a8277c45dd7f44766571dbf6af4141f56fb5f197c3211826c6d70063707c6ceb` | `e47becba`/550、`118b2229`/497、`51a45a95`/529，合计1,576 records |
| Mem-Gallery | `fcd47af2b493cd9a7856cb77a622291b5aa9c6dad12b1f3553d8f569e2c5f6b8` / `938f5785fd76afb50ec6f2e2c20ce573c371c290bc37082322946db51a679623` / `bddc0fdce411f783f398402f1510593eabbcf27e011f3c7e5ca097f38c154794` | `AI_Robotics_Automation_Future_Tech:1..8`，185 records / 31 images |

原型 runner 最终 SHA-256 为
`12c129f6e528f7487f978cf8a6cf7f1bed77063d7efd23c35e99e9051fe814e7`；synthetic runner 为
`cd444238dd44ea76c69fcb345713f9a3f5c03674cbefc9f95e974a41c8469e10`。这些 scratch runner
不属于产品发布物；hash 只用于复核本轮研究收据。

本轮实际验证如下：

- `uv lock --check --default-index https://pypi.org/simple` 通过，解析 178 个 packages；
- Ruff format/check 通过，185 files already formatted，lint 无错误；
- `mypy` 首次与其他门禁并发运行时无输出、exit 139；立即隔离复跑后通过，146 source files
  无问题。两次结果均保留，不能把首次进程崩溃写成类型错误或静默删掉；
- 默认 `pytest -W error` 通过，1,713 tests passed in 160.53 秒，没有额外启用外部 endpoint；
- 两份研究文档、README 索引的定向 markdownlint 通过，3 files、0 errors；固定 lychee 命令
  检查 410 links，406 OK、0 errors、4 redirects；
- 两份文档的 68 个唯一 GitHub fixed-commit `blob/tree` 链接均以本地 clone 的
  `git cat-file -t commit:path` 复核存在且对象类型匹配；
- `git diff --check` 通过。

仓库规定的原样 markdownlint glob 在这个长期工作区还扫描了
`.claude/worktrees/*/.venv` 中的第三方包 Markdown，因 4,442 个仓库外问题退出 1；不能把它记为
通过，也不应修改依赖文档。本轮新增的 sources 表格分隔符问题已修复，排除该工作区目录后的
全仓受控文档检查和上述三份变更文档定向检查均为 0 errors。

下一次有效决策点不是增加来源组装功能；A/B/C 已按停止规则退出。应先完成 raw dense/BM25/union
的同预算 loss-location 对照，再根据未召回、上下文挤占、模型未用和时序/冲突桶选择下一项实验；
负结果与成本同样进入报告。
