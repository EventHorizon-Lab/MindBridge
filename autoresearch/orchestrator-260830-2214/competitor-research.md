# MindBridge Memory Backend 竞品研究

> 研究截止：2026-08-31；目标优先级：更强 > 更快 > 更省。
> 本文只使用论文、官方仓库与官方文档。仓库实现按固定提交审阅，避免把后续 README、论文概念图或托管产品能力误当成可复现的开源实现。

## 0. 结论先行

MindBridge 不应复制任何一家竞品的完整架构。当前最有价值、且符合
[`design-principles.md`](../../docs/design-principles.md) 的组合是：

1. **保留原始证据，新增可重建的来源可追踪派生层。** 从 ABot、Zep/Graphiti、M3-Agent
   吸收事件、实体、地点、关系、状态变化和证据出处，但原始文本与媒体仍由现有 SQLite/CAS
   权威保存；派生事实不得覆盖或删除原始证据。
2. **在 SQLite 内做实体一致性与双时间事实，而不是引入图数据库。** M3-Agent 的消融结果最有力地支持
   face/voice/character 等价关系；Zep 最值得迁移的是 `valid time` 与 `transaction time` 的区分。两者都可由
   SQLite 的窄表、索引和现有 outbox 语义实现，Neo4j/FalkorDB/PostgreSQL/Redis/Kafka 均无必要。
3. **检索采用“现有单次混合检索优先，按需有界扩展”。** 先走 MindBridge 已有的 dense + lexical +
   temporal + identity 路径；只有实体、多跳、状态变化或空间问题才进行一跳或固定预算的邻域扩展。ABot
   的固定深度/证据 token 预算比 M3-Agent、MIRIX、TeleMem 的默认多轮 Agent 检索更适合低 TTFT 产品。
4. **空间记忆是最明确的 embodied 缺口，但应作为窄、可选能力验证后引入。** eMEM 的坐标、层、时间联合
   查询有现实价值；其点坐标模型仍过弱。候选设计至少应包含坐标系、2D/3D 位姿、有效时间区间、观测不确定度
   与来源证据，且先在 M3-Bench Robot / EgoLifeQA / ATM-Bench 上证明收益。
5. **写入侧优先 ADD-only / supersede，不采用 LLM 破坏性 UPDATE/DELETE。** 新版 Mem0 的方向正确；
   TeleMem 论文和旧 Mem0 的聚类后删除/更新容易丢证据。允许派生结论过期、被取代或重建，不允许把原始记录
   当作模型可随意改写的“事实数据库”。
6. **目前没有竞品给出可直接当作 MindBridge SOTA 目标的统一数字。** 各论文更换 writer、answerer、judge、
   帧预算、题目子集和硬件；有的排除 adversarial，有的使用私有训练数据，有的只报告托管闭源路径。
   因此只能迁移机制假设，再通过 MindBridge 公共 SDK、固定默认栈和非重叠 holdout 验证。

最应避免的做法是：六路 Memory Manager 扇出、默认 ReAct 多轮检索、依赖外部图数据库/队列、删除原始截图或
视频换取“存储节省”、把 benchmark split 的失败规则编译成运行时提示、以及把 `user_id`/metadata 当隔离边界。

## 1. 证据口径与 MindBridge 现状边界

本文使用三种标签：

- **实现事实**：能在官方仓库固定提交中定位到数据结构、调用路径或依赖。
- **论文主张**：论文描述或作者实验结果；不自动代表官方仓库已实现，也不代表在 MindBridge 栈上可复现。
- **不可比/弱证据**：模型、judge、数据切分、帧预算、训练数据、硬件或产品形态不同，或缺少测量方法。

研究的官方材料及固定提交：

| 系统 | 论文 | 官方实现证据 |
| --- | --- | --- |
| ABot-AgentOS | [arXiv:2607.10350](https://arxiv.org/abs/2607.10350) | [官方仓库](https://github.com/amap-cvlab/ABot-AgentOS) 已上线项目说明，但 README 明确表示代码与资源仍在准备发布，因此只有论文证据 |
| M3-Agent | [arXiv:2508.09736](https://arxiv.org/abs/2508.09736) | [ByteDance-Seed/m3-agent @ `0e3e419`](https://github.com/bytedance-seed/m3-agent/tree/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c) |
| eMEM | [arXiv:2606.03374](https://arxiv.org/abs/2606.03374) | [Automatika-Robotics/eMEM @ `82e3da6`](https://github.com/Automatika-Robotics/eMEM/tree/82e3da61cf710c4379e0cd7bf7a6a21710caaa96) |
| MIRIX | [arXiv:2507.07957](https://arxiv.org/abs/2507.07957) | [Mirix-AI/MIRIX @ `8cb06a6`](https://github.com/Mirix-AI/MIRIX/tree/8cb06a62bbb7c478beb33dd4f2815696a72df482) |
| Zep/Graphiti | [arXiv:2501.13956](https://arxiv.org/abs/2501.13956) | [getzep/graphiti @ `8b61fce`](https://github.com/getzep/graphiti/tree/8b61fce9f003cc3a05e246f6201f8b782dfe6546) |
| Mem0 | [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) | [mem0ai/mem0 @ `19cb89a`](https://github.com/mem0ai/mem0/tree/19cb89aff472325c707f64b2f34ae6afdbf7faf7) |
| TeleMem | [arXiv:2601.06037](https://arxiv.org/abs/2601.06037) | [TeleAI-UAGI/telemem @ `1a69eee`](https://github.com/TeleAI-UAGI/telemem/tree/1a69eee42890637960a138456c97ac57caaca3b6) |

这里的 eMEM 指 Automatika Robotics 的 **embodied spatiotemporal memory**，不是同名的纯对话
EMem 项目。

对 MindBridge 的建议必须从已实现边界出发，而不是重复建设。当前产品已经有：

- 一个公共 `Memory` 面向 text/image/video/audio/omni；原始媒体进入内容寻址 CAS；
- SQLite 权威记录与 embedding，Zvec 是可重建投影，SQLite commit → outbox → Zvec flush → ack；
- dense 与 lexical 并行检索、SQLite 最终 hydration、时间区间过滤/评分、多向量记录；
- face/voice 的稳定 identity 合并；`semantic`、`episodic`、`procedural` 三类公共语义；
- 公共 SDK、REST、MCP、CLI 同一执行面，且一物理 `data_dir` 是一实例/记忆域。

因此，“再加一个向量库”“再定义 semantic/episodic”“再做一次 BM25 + embedding”都不是创新。真实缺口主要在
**来源可追踪的派生事实、实体关系、多跳/状态变化检索，以及显式空间语义**。

## 2. 后端机制总览

| 系统 | 真实存储/索引 | 写入与维护 | 检索/推理 | 多模态、时空、实体 | 成本与延迟证据 |
| --- | --- | --- | --- | --- | --- |
| ABot-AgentOS | 论文只给 typed evidence graph；未披露 DB/索引库，仓库暂无实现源码 | 多源 adapter → selective writer → 去重/合并；旧状态以 supersede 边保留 | semantic + lexical + metadata seeds；固定深度、证据 token 预算的 typed-edge 扩展 | 文本、视觉、视频、空间、任务 trace；实体/地点/session/event/provenance | 无存储、索引、p95、TTFT 或 token 表；只能视为架构主张 |
| M3-Agent | Python `dict` 图对象，NumPy/sklearn 全量 cosine；最终 pickle | 每 30 秒视频片段生成 episodic + semantic；face/voice 聚类和跨模态等价；重复关系加权 | 控制模型循环生成 query；`search_node`/clip max-sim；最多多轮 | 原生视频、音频、face、voice、character、clip time；无空间坐标 | 训练使用 16×80GB GPU；未报告 memory backend latency/token/durability |
| eMEM | SQLite + hnswlib + SQLite R-tree + FTS5 BM25/RRF | 5 条或 2 秒 flush；episode gist、DBSCAN 空间聚合、entity 周期抽取；原始 observation 可被归档 | semantic/spatial/temporal/gist/entity 工具；统一 `recall` | 主要存后感知文本 + 3D 点 + layer/time；不保存完整传感器证据 | 有 A5000 随机向量微基准，但非端到端；固定 HNSW 参数在 100k recall@10 仅 0.06 |
| MIRIX | PostgreSQL + pgvector/pg_bm25；Redis 可选缓存；内存队列默认、Kafka 可选 | Meta Agent 路由到最多六个专用 manager；可 auto-dream 合并/冲突处理 | 每类先取候选，chat agent 可继续主动搜索；多 Agent chaining | text/image/voice/screenshot；六类记忆；时间事件与敏感 vault | 论文没有统一 token/TTFT；多 Agent 与多类 top-k 天然放大调用数和上下文 |
| Zep/Graphiti | 论文用 Neo4j/Lucene；OSS 支持 Neo4j、FalkorDB、Neptune+OpenSearch，Kuzu 已弃用 | episode 无损写入；LLM 抽实体/事实、entity resolution、冲突 invalidation、community 更新 | vector + full-text + BFS；RRF/MMR/node-distance/cross-encoder 可组合 | 文本/JSON episode；entity/fact/community；双时间事实；无原生媒体证据 | 论文 LongMemEval 总响应快于 full context，但含网络与不同部署；写入 LLM 成本较重 |
| Mem0 | 旧论文：vector store + SQLite history，graph 版 Neo4j；当前 OSS：可插拔 vector store + 独立 entity collection | 旧论文每个 fact 执行 ADD/UPDATE/DELETE/NOOP；当前实现改为单次 ADD-only + hash/事实抽取 | 当前 OSS semantic + BM25 + entity boost；平台另有闭源 native graph/temporal fusion | OSS 实际 vision 是图片转文本；不开启 vision 时图片被静默丢弃；无原生音视频记忆 | 旧论文 p95 可参考但协议过时；新 92.5/94.4 是闭源平台、top_200、约 7k token/query |
| TeleMem | 当前 OSS 文本路径继承 Mem0，FAISS + JSON；视频另用 frames/captions/NanoVectorDB JSON | 16 线程抽取；buffer=64；相似度聚类后 LLM 融合；视频切片 caption | 文本单次向量检索/可选 rerank；视频最多 15 轮 ReAct、caption 检索与回看帧 | 视频→帧→caption；subject registry；没有独立音频/跨视频实体/空间 | 论文称 token -43%、2.1×，但未给测量表；当前仓库承认旧结果尚未满足其新评测 charter |

## 3. 各系统深挖

### 3.1 ABot-AgentOS：最完整的“证据图”概念，最薄弱的实现证据

**论文主张。** ABot 把输入统一为 source container、evidence unit、entity、place、session、semantic
event 等 typed nodes；边覆盖 temporal order、containment、observation、participation、location、identity
continuity、spatial、interaction 与 provenance。节点携带 schema version、source reference、time reference、
evidence summary、confidence、extractor model/version 等。这一点比只存 caption 或 fact 的系统更符合 embodied
memory，因为答案可以回溯到原始观察和提取器版本。

写入 adapter 处理 dialogue、egocentric video、visual observations、spatial context 与 task traces。Selective
writer 重点保留身份、物体位置/状态变化、承诺、偏好、异常、时间事实及验证证据。维护阶段只在 provenance、
时间和 identity 兼容时合并近重复项，并用 supersession 表达旧状态失效，而非删除历史。

检索先做 semantic、lexical、时间/source/modality/place filter 和 node type 打分，再从 seeds 沿 typed edges
有界扩展；最终按 evidence-token budget 序列化紧凑子图。它还记录检索 trace，并要求证据不足时显式回答不确定。
这是七个系统里最接近 MindBridge “记忆语义 + 可审计证据”的设计。

**实现事实。** 无法确认。论文没有披露 SQLite/Neo4j、向量索引、事务、恢复、崩溃一致性或具体延迟路径；论文给出的
官方仓库在 2026-08-31 已上线项目说明，但 README 明确表示代码与资源仍在准备发布。因此不能把
“typed graph”“private-by-default edge/cloud routing”当作已经可复用的生产 backend。

**benchmark 解释。** Static 版论文报告 LoCoMo 87.5、Mem-Gallery 88.6、OpenEQA 59.9、NExT-QA
76.5、EgoLifeQA 65.4；self-evolution 分别有小幅到中等提升。但：

- writer/answerer/judge 随 benchmark 改变：Qwen3.6-Plus、GPT-5.4、Qwen3.5-Flash 混用；部分行同一模型同时作
  answerer 与 judge；
- OpenEQA 的 24-frame direct VQA 达 73.0/74.1，明显高于 graph memory 59.9；Mem-Gallery full-context
  为 92.6，高于 88.6；NExT-QA Qwen direct QA 为 81.9，高于 76.5；故不能概括为“全模态 SOTA”；
- self-evolution 虽声明 splitwise no-leakage，但本质是读取失败 trace、将诊断编译成受限 DSL evo-assets，再用于后续
  split。它适合作为研究假设生成器，不应在无真实线上反馈时成为产品默认，也不能把 benchmark 标签规律编译进产品。

**可迁移结论。** 采用其 source-grounded schema、置信度、提取器 provenance、supersede 与预算化邻域扩展；不采用
未知存储实现、cloud routing 或 benchmark 自演化规则。最小实现应是 SQLite 中的可重建派生表，而不是“Graph
Backend”公共插件。

### 3.2 M3-Agent：跨模态身份最有价值，backend 本身不是生产系统

**实现事实。** 官方代码的 [`VideoGraph`](https://github.com/bytedance-seed/m3-agent/blob/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c/mmagent/videograph.py)
是进程内 Python 对象：节点和双向边存 `dict`，文本、face、voice embeddings 存 list，检索用 sklearn/NumPy
对所有候选计算 cosine，最终用 pickle 落盘。它没有 ANN、事务日志、增量 checkpoint、并发写入或崩溃恢复，因而
不能作为 MindBridge 存储参照。

视频以 30 秒 clip 处理。InsightFace `buffalo_l` 在抽帧上检测/聚类 face；ERes2NetV2 提取 voice embedding，
过短音段被丢弃。face 与 voice 分别用 0.3/0.6 阈值匹配已有节点，每个实体仅保留有限 embedding exemplar；
semantic/episodic 文本用 `text-embedding-3-large`。生成的语义内容显式引用 `<face_i>`/`<voice_i>`；LLM 产生的
equivalence relation 经 union-find 变成共同 `<character_i>`。代码中的默认 query top-k 是 2，检索阈值 0.5；
clip score 是其 memory entry 的最大相似度。对应实现见
[`retrieve.py`](https://github.com/bytedance-seed/m3-agent/blob/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c/mmagent/retrieve.py)
和 [`processing_config.json`](https://github.com/bytedance-seed/m3-agent/blob/0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c/configs/processing_config.json)。

控制模型会反复判断证据是否足够，不足则产生新的 embedding query。论文训练 Qwen2.5-Omni-7B memorizer 和
Qwen3 control model，并用 RL 优化；训练使用 16×80GB GPU。它是一个特定长视频 QA agent，而不是可插拔通用
Memory Backend。

**论文主张与最可信信号。** M3-Bench Robot/Web/VideoMME-long 得分为 30.7/48.9/61.8。绝对分数受私有
训练集、RL control model 和视频预处理影响，不能与 MindBridge 默认 qwen3.8-flash 直接比较。更有迁移价值的是同一
系统内消融：去掉跨模态 identity equivalence 后三项降到 19.5/39.7/52.1；去掉 semantic memory 降到
13.6/29.7/48.7。这强烈支持“稳定身份 + 来源一致的语义派生记忆”，但不证明频次加权或无限多轮搜索是最佳方案。

**风险。** M3 对重复语义的处理会强化节点/边权，冲突时倾向频率多数。家庭机器人里，重复的 ASR/VLM 错误也会被
强化；随机抽取 exemplar 还损害可复现性。MindBridge 应以置信度、来源多样性、时间和提取器版本共同投票，并保留
反例，不使用简单频次真理。

### 3.3 eMEM：最接近嵌入式时空后端，但持久性次序不合格

**实现事实。** eMEM 是单进程嵌入式后端：SQLite 存 observation、episode、gist、entity、edge 与 HNSW ID
mapping；hnswlib 做统一 semantic ANN；SQLite R-tree 做 3D 空间查询；当前代码另有 FTS5 BM25，并用 RRF 与
HNSW 融合。默认 HNSW `M=16`、`efConstruction=200`、`efSearch=50`。配置和实现见
[`config.py`](https://github.com/Automatika-Robotics/eMEM/blob/82e3da61cf710c4379e0cd7bf7a6a21710caaa96/emem/config.py)
及 [`store.py`](https://github.com/Automatika-Robotics/eMEM/blob/82e3da61cf710c4379e0cd7bf7a6a21710caaa96/emem/store.py)。

输入 observation 主要是**后感知文本 + 3D 点 + timestamp + user-defined layer**。内存 deque 满 5 条或经过
2 秒就 flush；episode 结束后按时间间隔生成 gist，原始短期 observation 后续可被归档；非 episode 观察使用
DBSCAN 做空间聚合。entity 默认每 10 次 flush 或 60 秒抽取，名字相似度和 5 米距离共同参与 merge。API 暴露
semantic、spatial、temporal、gist、entity、body status、locate 与组合 `recall`。

**关键持久性问题。** 当前 `add_observations_batch` 在 SQLite commit 前就修改 HNSW/R-tree，HNSW 文件只在
`close()` 时 `save_index()`，而 ID mapping 已写入 SQLite。进程崩溃可造成 SQLite mapping、内存 HNSW 和磁盘
HNSW 三者分叉。这与 MindBridge “SQLite 先提交、投影 flush 成功后才 ack、投影可从 SQLite 重建”的硬规则冲突。
因此只能迁移时空语义，不能复制其写入顺序。

**benchmark 解释。** eMEM 的系统微基准使用 A5000 和随机 unit embeddings，不包含 embedder/VLM/LLM：
1k/10k/100k 的 ingest throughput 约 2558/947/561 obs/s，但固定 `efSearch=50` 时 recall@10 从 0.89 降到
0.34/0.06。这反而说明参数未随规模校准，不能把吞吐数字当作高质量检索速度。eMEM-Bench 是作者自建的 20 个
ProcTHOR houses、988 probes；报告 overall 80.8，消融支持 hybrid、gists 与 entities，但没有同协议独立产品
基线，也没有真实机器人长期漂移。

**可迁移结论。** 候选 `SpatialObservation`/空间投影应显式携带 coordinate frame、position/pose、uncertainty、
valid interval 与 source memory ID；先用 SQLite R-tree（若运行时可用）或小规模精确查询验证，不能立即增加
hnswlib/外部 spatial DB。gist 只能是派生摘要，不能像 eMEM 一样丢原始 observation/text/embedding。

### 3.4 MIRIX：记忆分类有启发，六 Agent 扇出和服务栈不适合嵌入式产品

**论文主张。** MIRIX 把记忆拆为 Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault，
Meta Memory Manager 决定一次写入路由到 0–6 个 manager。检索先生成 topic，再从每类取 top-10（最坏 60 条），
chat agent 仍可继续定向搜索。屏幕端每 1.5 秒抓图，极相似帧丢弃，约 20 个独特截图触发一次更新。

**实现事实。** 当前官方架构明确以 PostgreSQL 为主存，使用 pgvector/pg_bm25 与 rapidfuzz；Redis Stack 是可选
缓存，队列默认可用进程内实现、Kafka 是可选部署。写入经过 MetaAgent chaining 与多个 LLM sub-agent，读取分发到
不同 managers。见
[`docs/ARCHITECTURE.md`](https://github.com/Mirix-AI/MIRIX/blob/8cb06a62bbb7c478beb33dd4f2815696a72df482/docs/ARCHITECTURE.md)
和 [`docker-compose.yml`](https://github.com/Mirix-AI/MIRIX/blob/8cb06a62bbb7c478beb33dd4f2815696a72df482/docker-compose.yml)。
这套部署与 MindBridge 的嵌入式、无外部 DB/worker queue 原则不兼容；即便将 Kafka 换成进程内 queue，六类 agent
chaining 的调用和 failure surface 仍然很大。

**benchmark 解释。** ScreenshotVQA 只有 3 名参与者、87 个问题。MIRIX 得分 0.595，SQLite 记忆约
15.89MB；SigLIP 基线保留 15.07GB 原始截图，Gemini 基线保留 236.7MB resize 图。这里比较的是“删除原图后的
派生文本”与“保留证据的原媒体”，不能证明等价质量下节省 99.9%，也违反 MindBridge 证据保留。LoCoMo 实验排除
adversarial，使用 GPT-4.1-mini，MIRIX overall 85.38，仍低于 full context 87.52；它不能与 LoCoMo-refined
或 ABot 包含 adversarial 的 87.5 直接排序。

**可迁移结论。** `semantic/episodic/procedural` MindBridge 已有；Resource 是现有 CAS 媒体，Core 是应用 prompt
state，Vault 是 secret-management/ACL 问题，不应扩张成核心 MemoryType。可借鉴的是批次内将一句输入抽成多个
有 provenance 的派生类别，以及“只在任务需要时路由”，而非始终启动六个 agent。

### 3.5 Zep/Graphiti：双时间与 provenance 最强，外部图数据库和重写成本不可迁移

**论文主张与实现事实。** Zep 的图分三层：episodic nodes 保存原始 message/text/JSON；semantic entities/facts
是派生层；communities 是聚类摘要。episode 与 derived fact/entity 之间保留双向 provenance。关系事实具有：

- transaction time：`created_at` / `expired_at`，表示系统何时知道或撤销；
- valid time：`valid_at` / `invalid_at`，表示事实在现实世界何时成立。

当新事实冲突时，旧关系被失效而非抹除。当前 Graphiti 源码仍显式保存这些字段，见
[`edges.py`](https://github.com/getzep/graphiti/blob/8b61fce9f003cc3a05e246f6201f8b782dfe6546/graphiti_core/edges.py)。

写入会让 LLM 抽 entity/fact 并反思遗漏；entity name embedding + full-text 先找候选，再由 LLM resolution；fact
去重被限制在相同 entity pair，避免全局昂贵比较。读取组合 cosine、BM25、BFS，并可选 RRF、MMR、mention
frequency、node distance 或 cross-encoder。community 通过动态 label propagation 更新，降低全图重算，但仍有
显著 LLM 写入与陈旧性成本。

论文实现使用 Neo4j/Lucene；当前 OSS 支持 Neo4j、FalkorDB、Amazon Neptune + OpenSearch，Kuzu 已标记为
deprecated；FalkorDB Lite 仍引入专门图/Redis 运行时。见
[`README`](https://github.com/getzep/graphiti/blob/8b61fce9f003cc3a05e246f6201f8b782dfe6546/README.md)
与 [`pyproject.toml`](https://github.com/getzep/graphiti/blob/8b61fce9f003cc3a05e246f6201f8b782dfe6546/pyproject.toml)。
MindBridge 没有理由为邻接表和一跳扩展引入这些系统。

**benchmark 解释。** 论文 LongMemEval 在 GPT-4o-mini 上报告 Zep 63.8、full context 55.4，total response
latency 3.20s vs 31.3s；GPT-4o 为 71.2 vs 60.2、2.58s vs 28.9s，平均 context 约 1.6k vs 115k。
方向上说明压缩检索比全上下文省，但测试让 Zep 通过网络访问 AWS，而 baseline 路径不同；指标也是完整生成耗时，
不是 search p95/TTFT。因此不能用 3.2s 作为 MindBridge 后端性能目标。DMR 94.8 仅略高于 full-context 94.4，
论文自己也指出该 benchmark 已接近饱和。

**可迁移结论。** 先实现小型 `derived_entities`、`derived_facts`、`fact_evidence`、`fact_relations` 表，事实具有
valid/transaction interval 与 extractor provenance；通过 source memory FK 恢复。community summary 暂缓，除非
长程 benchmark 明确证明一跳事实图仍不够。

### 3.6 Mem0：API 值得参考，但论文、当前 OSS 与托管平台是三个不同系统

**旧论文路径。** 2025 论文的 writer 输入当前 user-assistant pair、异步 conversation summary 和最近 10 条消息；
LLM 抽 salient facts。每个 fact 先向量检索 top-10，再由 LLM 选择 ADD/UPDATE/DELETE/NOOP。graph variant
抽 entity/relationship triplets 并使用 Neo4j。这个逐 fact 检索 + 第二次决策调用质量可控但写入昂贵，并允许模型
破坏性改写。

**当前 OSS 实现事实。** 当前仓库已改为 `ADDITIVE_EXTRACTION_PROMPT`：单次抽取只 ADD，关联旧 memory ID，
不在默认写入中 UPDATE/DELETE；见
[`memory/main.py`](https://github.com/mem0ai/mem0/blob/19cb89aff472325c707f64b2f34ae6afdbf7faf7/mem0/memory/main.py)
与 [`configs/prompts.py`](https://github.com/mem0ai/mem0/blob/19cb89aff472325c707f64b2f34ae6afdbf7faf7/mem0/configs/prompts.py)。
OSS v3 可做 semantic + BM25 + entity matching，entity 进入平行 collection；外部 graph drivers 已从 OSS
移除，native graph 是托管 Platform 专属。官方迁移文档明确区分这两条路径：
[`OSS v2→v3`](https://github.com/mem0ai/mem0/blob/19cb89aff472325c707f64b2f34ae6afdbf7faf7/docs/migration/oss-v2-to-v3.mdx)。

当前多模态代码实际只识别图片消息并先生成 description，再把文本送入 memory extraction；它不是原生视觉 embedding
或 evidence retrieval。若 `enable_vision=False`，官方文档明确说明图片 turn 会被静默丢弃：
[`multimodal-support.mdx`](https://github.com/mem0ai/mem0/blob/19cb89aff472325c707f64b2f34ae6afdbf7faf7/docs/open-source/features/multimodal-support.mdx)。
这恰好违反 MindBridge “unsupported media must never be discarded silently”。代码和详细文档没有证明独立 audio/video
memory，尽管 overview 有宽泛营销表述。

**benchmark 版本漂移。** 旧论文 LoCoMo 排除 adversarial：Mem0 66.88、graph 68.44、full context 72.9；
search p95 约 0.200s/0.657s、平均检索 context 1764/3616 tokens。该算法已不是当前默认。当前 README 的
LoCoMo 92.5、LongMemEval 94.4、BEAM 1M/10M 64.1/48.6 明确来自**包含 proprietary optimizations 的托管
平台**，单次 top_200，约 6.7k–7.0k tokens/query；OSS 用户不会得到相同结果。因而既不能用新分数评价 OSS，也
不能拿旧 p95 评价新平台。

**可迁移结论。** ADD-only + related-memory link、hash dedup before embedding、并行多信号 fusion 值得验证；但
MindBridge 不复制隐式 provider construction、`user_id` 逻辑域、metadata 隔离、平行 entity vector collection 或
图片转文本后丢原媒体。

### 3.7 TeleMem：批写思路有工程价值，论文 DAG 尚未出现在开源产品

**论文主张。** TeleMem 将 profile/event/entity/object 组织为严格按 effective timestamp 前进的 DAG。新节点先从
历史向量 top-k 找 parent，再去除存在替代路径的冗余边，形成“minimal causal skeleton”。读取时从 embedding seeds
向祖先闭包展开，按时间线性化；过大时可限制深度/相关度/token budget。离线 writer 是 turn summarization →
retrieval alignment → global clustering → cluster-level LLM consolidation，并为每项选择 add/delete/update/no-op；在线
版省略 global clustering，局部 Insert/ReInsert，删除节点用 tombstone 并重接 children。视频侧另有 ReAct 工具。

**实现事实与论文断层。** 当前 OSS 仓库没有 DAG、closure、Insert/ReInsert、parent/edge 或 causal skeleton
实现。文本类 [`TeleMemory`](https://github.com/TeleAI-UAGI/telemem/blob/1a69eee42890637960a138456c97ac57caaca3b6/telemem/mem0.py)
继承 Mem0 私有方法：每条消息做 LLM summary 和相关 memory 检索；`add_batch` 用 16 个线程并行，按 scope 放进
`buffer_size=64` 的内存 buffer；flush 时以 0.95 阈值做贪心、近 O(n²) embedding clustering，然后每个 cluster
再调用 LLM 融合并只 ADD。README 所称持久层是 FAISS + JSON；这不是论文的持久 DAG，也没有 outbox/可恢复
双写协议。

视频是另一条 API：frame extraction → 每 10 秒左右 caption → NanoVectorDB JSON。查询可 `global browse`、
caption search、frame inspect，默认公共 `search_mm` 最多 15 轮；prompt 还要求得到答案后用 frame inspection 再
确认。每个 `output_dir` 当前强制恰有一个 captions 和一个 VDB 文件，不是统一、可增长的跨视频 memory domain；
也没有独立音频、跨视频 identity 或空间位姿。

**benchmark 解释。** 论文在 ZH-4O（28 个真实 session、1068 道选择题）用统一 Qwen3-8B/Qwen3-Embedding-8B
重做 baseline，TeleMem 86.33、full context 84.92、Mem0 70.20。这是有价值的内部同栈对照，但单数据集不足以
说明通用 SOTA。LoCoMo 中 TeleMem 61.49，低于 full context 70.71，多跳仅 20.56，说明祖先 closure/压缩
不是普遍收益。论文摘要声称 token -43%、2.1× speedup，却没有给对应表、统计口径、硬件或 ingestion/query
分解，无法审计。更重要的是，官方当前
[`evaluation charter`](https://github.com/TeleAI-UAGI/telemem/blob/1a69eee42890637960a138456c97ac57caaca3b6/docs/evaluation.md)
承认 README 的 ZH-4O 数字早于其 multi-seed、CI、full-context/grep、token/latency 新规范，补跑尚待完成。

**可迁移结论。** 可测试“批次内先检索对齐，再一次聚类/融合”是否降低写入 token；但应将新事实与来源保留为
ADD-only 派生记录。不要复制内存 buffer 的非持久语义、FAISS/JSON 双写、默认 15 轮视频 ReAct 或尚未开源的
DAG 假设。

## 4. Benchmark 可比性审计

| 声称 | 为什么不能直接与 MindBridge 排名 | 可保留的信号 |
| --- | --- | --- |
| ABot “多 benchmark SOTA” | benchmark 间换模型/同模型 judge；帧预算不同；多项 direct/full-context 更高；无成本/延迟 | typed evidence + bounded graph expansion 值得独立验证 |
| M3-Agent 领先强 prompting agent | 私有训练视频、SFT+RL control、16×80GB GPU；不是同一 memory backend/default VLM | identity equivalence 与 semantic-memory 消融幅度大 |
| eMEM overall 80.8 / 高吞吐 | 自建 simulated benchmark；系统微基准是随机 embedding 且 HNSW recall 随规模崩落 | spatial+temporal+layer 联合语义有效 |
| MIRIX LoCoMo 85.38 / 99.9% 存储节省 | 排除 adversarial；ScreenshotVQA 仅 87 问；删除原图与保留原图比磁盘 | typed extraction 和截图去近重复可作为派生优化 |
| Zep 10× total latency 改进 | 压缩检索与 115k full context 的总生成时延比较；部署/网络不同，不是 search p95 | 双时间和 provenance 支持 temporal QA |
| Mem0 92.5 / 94.4 | 新数字是 proprietary managed platform，top_200、约 7k token；当前 OSS 不同 | 单次 ADD-only、多信号并行是合理方向 |
| TeleMem token -43%、2.1× | 论文没有测量表/硬件/分项；当前 OSS 没实现论文 DAG；官方承认旧结果未满足新 charter | 同模型 ZH-4O 对照支持批写/结构化写入假设，但需复现 |

MindBridge 的比较必须固定：数据集 revision、official split、qwen3.8-flash no-think、Jina Omni Small、
FunASR-Nano-2512、输入 route、帧/音频采样、同一 answer/judge prompt、5090 32GB、缓存政策、候选数、token
预算、p50/p95/p99、TTFT、存储增长和至少一个非重叠 holdout。任何竞品数字只应形成假设，不得进入验收阈值。

## 5. 可迁移假设：按“更强 > 更快 > 更省”排序

### P0：更强

#### H1. Source-grounded derived memory（最高优先级）

在现有 immutable source memory 上派生：

- `event`：谁、做了什么、何时、何地、结果；
- `entity`：canonical identity、alias、face/voice identity link；
- `fact/relation`：subject、predicate、object/value；
- `place/spatial observation`：coordinate frame、pose/position、uncertainty；
- 每项均保存 `source_memory_id`、asset/time span、extractor ID/version、confidence、created/valid interval。

派生层必须可删除并从 SQLite/CAS 重建；原始 record、media、transcript 和 embeddings 不因 consolidation 被删除。
这是 ABot provenance、Zep episodic grounding、M3 identity、eMEM spatial 的最小交集。

#### H2. Bitemporal supersession，而非事实覆盖

关系拥有：现实有效区间 `valid_from/valid_to`，以及系统观察区间 `recorded_at/expired_at`。新事实可以
`supersedes` 旧事实，旧证据仍可回答“以前是什么”。时间不确定时存区间/粒度而非伪精确 timestamp。首先只支持
少数关系类型和状态变化，避免通用 ontology。

#### H3. Entity-aware bounded expansion

检索保持现有 dense+lexical+temporal+identity 为第一阶段。只有以下条件触发邻域扩展：

- query 提及已解析 entity 或 identity；
- temporal/state-change intent；
- spatial/where intent；
- 多跳候选边际不足或 answerer 明确返回证据不足。

扩展最多 1 跳（实验阶段可比较 2 跳）、固定节点数和 evidence token budget；按来源多样性和有效时间去重，最后
仍从 SQLite hydration。默认不启动 ReAct；如单次检索低置信，最多允许一次 query rewrite/follow-up，并单独计入 TTFT、
总时延和 token。

#### H4. Cross-modal identity evidence voting

MindBridge 已有 face/voice identity，下一步不是重做识别器，而是让派生 event/fact 绑定稳定 identity，并测试跨模态
合并的答案收益。合并评分应综合独立 source 数、face/voice confidence、时间连续性和反证；exemplar 选择必须
确定性且有上限，不能像 M3 随机抽样，也不能让重复模型错误无限强化。

#### H5. Optional spatial capability

空间必须是真语义，不塞进 metadata 假装可查询。候选内部协议：`frame_id`、`position`、可选 quaternion/heading、
`observed_at/end`、`uncertainty`、`source_memory_id`。先用 SQLite 精确/R-tree 投影；无数据时整个能力不加载、不增
调用、不改变 API。若 M3 Robot/EgoLife/ATM 的 where/object-state 子集无稳定提升则删除，不保留 speculative abstraction。

#### H6. Selective, batched derivation

将短窗口内的多条 source memories 一次送入 qwen3.8-flash，抽取多个 ADD-only derived items；先做内容 hash 和
现有近邻对齐，再对真正冲突/重复的 cluster 调一次 consolidation。应同时测试：逐条、固定小批次、session/end-of-
episode 三种策略。选择依据是跨 benchmark 质量，不能因为批写更省就接受漏记。

### P1：更快

1. **Query-gated graph lane。** 绝大多数单跳事实保持现有快速路径；entity/spatial/temporal lane 与 dense/lexical
   并行，只在有 signal 时启动。避免 MIRIX 六类全 fan-out。
2. **Write-time precomputation。** entity aliases、relation adjacency、time buckets 和 compact evidence snippets 在写入/
   maintenance 阶段生成，query 不临时调用 LLM resolution。
3. **有界候选与 early stop。** 如果 top hits 具有高 calibrated strength、identity/time 一致且足以回答，就不 rewrite；
   低置信才一次补检。对 search、ask TTFT、complete latency 分开测。
4. **批 embedding/批 SQLite transaction。** 复用现有 provider batch 和 outbox，派生项与 source FK 可同事务提交；
   不引入 TeleMem 的易失内存 buffer，也不让 graph projection 先于 SQLite 改动。
5. **5090 用于本地可测瓶颈，不用于复制大规模 RL。** Jina/FunASR/face 可并发、显存驻留和批处理；云端
   qwen3.8-flash 只做真正需要的生成。M3 的 16×80GB 训练不是产品前置条件。

### P2：更省

1. **Evidence budgeter。** 每个 source/derived group 只序列化最短可验证 snippet、时间、identity 与 asset reference；
   限制单一 long record/多向量记录占满上下文，报告 answer input/output tokens。
2. **ADD-only 避免第二次逐 fact CRUD 判决。** 如需 consolidation，对 batch cluster 调一次；先由 hash/exact/embedding
   过滤显然重复。对比 Mem0 旧双调用与新版单调用。
3. **多层内容按需展开。** 默认返回 derived fact/event；answerer 需要细节时才展开 source transcript/frame span。原始
   媒体仍保留，不把“删除证据”记作节省。
4. **不建设 community summary。** 在一跳/双时间图没有证明不足前，Graphiti community 与周期 LLM refresh 是无
   证据成本。
5. **不把 top_200/7k token 当先进检索。** 候选数与 token 必须通过质量曲线选择，目标是更少 token 下不降质量，
   而非用巨大上下文买分。

## 6. 推荐实验顺序与停止门槛

以下是研究假设，不是要求立即扩公共 API。

### E1：来源可追踪的 event/fact 派生层

- 开发子集：LoCoMo-refined + Mem-Gallery，先用足够覆盖 single-hop、temporal、multi-hop、conflict、refusal 的分层
  子集；派生规则只看通用 schema，不读取 dataset/category 名称。
- 对照：当前 incumbent；ADD-only derived facts；再加 bitemporal supersession。
- 必报：分类型 accuracy/F1、evidence recall/precision、错误归因、ingestion p50/p95、search/ask p50/p95、TTFT、
  qwen input/output tokens、SQLite/Zvec/CAS bytes。
- 保留条件：跨两个 text/image-text benchmark 质量稳定提升；任何原始证据丢失、媒体 route 退化、holdout 回落或
  durability 破坏直接淘汰。小于 judge 噪声的提升不算胜利。

### E2：entity-aware 一跳扩展

- 数据：M3-Bench Robot 身份题、EgoLifeQA entity/relation 子集，加 Mem-Gallery 多实体题。
- 对照：现有 identity scoring；绑定 derived event；再加一跳 adjacency；最后才比较一次低置信 query rewrite。
- 保留条件：身份/多实体题显著提升，普通单跳 p95/TTFT 不回退；rewrite 只在触发样本计费并有正净收益。

### E3：空间能力

- 数据：ATM-Bench raw media、M3 Robot object-location/state、EgoLife where/temporal-spatial；需先确认官方标签能评估
  坐标/相对位置而非仅 caption。
- 对照：文本 caption；结构化 3D/2D observation；结构化 observation + 一跳 event/entity link。
- 停止条件：若坐标在这些公开任务不能增加 evidence recall/答案质量，或输入 adapter 需要 benchmark 私有字段，立即
  删除，不以“未来机器人可能有用”保留。

### E4：批量派生与上下文预算

- 批次候选：1、4、8、episode-end；evidence budget 曲线而非单点 top-k。
- 质量先过 E1/E2 门槛，再优化 write throughput、query latency、tokens。
- 与逐条两阶段 CRUD、无派生当前路径比较；报告 cache hit/miss，禁止把缓存响应计入 cold-path SOTA。

### 扩大样本的节奏

1. 先做分层小样本用于快速淘汰明显无效机制；
2. 候选连续两轮在不同 category 改善后扩大至中等样本；
3. 只有中等样本无质量回归，才运行五 benchmark 公共 SDK dev suite；
4. 参数冻结后运行非重叠 holdout；holdout 失败不得回看标签调参后仍称该 holdout；
5. 最终再做并发、长时增长、崩溃恢复和磁盘增长，防止只优化静态 QA。

## 7. 明确不做

- 不引入 Neo4j、FalkorDB、PostgreSQL、Redis、Kafka、Qdrant 或第二个权威 vector store；
- 不复制 M3 的 pickle graph、eMEM 的 HNSW-before-SQLite 写序或 TeleMem 的 FAISS/JSON 双写；
- 不新增 Core/Resource/Vault 等公共 MemoryType，除非出现独立产品需求和完整契约；
- 不默认六 Agent 写入、每类 top-k fan-out、15 轮视频 ReAct 或无限 query rewrite；
- 不归档/删除原始 frame、audio、video、transcript 来换取磁盘指标；
- 不将 metadata、`user_id`、benchmark case ID 当执行或隔离语义；
- 不用 benchmark-specific prompts、题型关键词路由、split 标签、答案缓存污染或 judge 同源偏好提高分数；
- 不因为论文自称 SOTA 就改默认值；所有默认改变必须在 MindBridge 固定栈、公共 SDK 和 holdout 上成立。

## 8. 后续 autoresearch 的建议优先序

本轮已经冻结五 benchmark 的配对协议，并将候选限制为无新增依赖的检索、媒体采样与证据预算改进。竞品研究不支持
在同一候选里再塞入派生图、空间层或 query rewrite。后续主循环应逐项推进：

1. 先完成当前冻结候选的质量、token、latency/TTFT 和 evidence trace 验证；
2. 下一独立候选只做 **source-grounded ADD-only derived event/fact + provenance**，不同时加 graph、spatial、
   query rewrite；
3. 若 LoCoMo-refined/Mem-Gallery 跨任务提升，再加 **bitemporal supersession**；
4. 若 M3/Ego identity 子集仍有缺口，再加 **entity-aware 一跳扩展**；
5. 只有 embodied 子集证明确有坐标缺口，才引入 **optional spatial projection**；
6. 每个质量候选冻结通过后，才优化批写、early stop、budgeter 和 GPU 并发。

这个顺序能够逐项归因，遵守“更强第一、更快第二、更省第三”，也避免把一次大规模 graph rewrite 的 benchmark
波动误认成真实能力提升。
