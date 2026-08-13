# MindBridge RTX 5090 整体效果与生命周期验证

> 运行日期：2026-08-13
>
> 运行编号：LoCoMo `5090-clean-006`；多模态 `5090-clean-007`
>
> 验证边界：一台 RTX 5090 同时模拟机器人端与云端；本轮验证功能、质量和数据闭环，延迟、功耗、温度与 Jetson TensorRT 性能暂不作为验收门禁。
>
> 可复现清单：[`benchmark-5090-clean-007.json`](../benchmarks/manifests/benchmark-5090-clean-007.json)

## 1. 结论

MindBridge 的 MaaS 主链路已经能够真实运行：端侧原始视频进入近期记忆与 Outbox，云端完成对象存储、视听理解、事件/实体/Claim 构建、Embedding、混合召回和证据回答；反馈能够强化或版本化纠错；生命周期能够降冷并在访问时回热；显式遗忘能够删除云端媒体、派生记录、索引、端侧近期记忆和本地身份模板；Episode、Claim 与 Summary consolidation 也已通过真实 Omni/VLM 调用完成提交。

这不等于 MindBridge 的最终目标已经完成：LoCoMo 的官方 token-F1 已超过当前公开 T-Mem 结果，但 Judge 模型和重复次数不同，不能据此宣布整体 SOTA；三个多模态 Benchmark 的完整公开题集已经评估，但公开输入条件分别是发布 caption、transcript 或 memory graph，而不是完全相同的原始视听管线，因此只能用于定位记忆层差距，不能冒充与论文榜单严格可比的 SOTA 复现。真正未完成的是原始视听全量重放、迭代检索/经验记忆、目标 Jetson 标定和严格官方 Judge 复跑。

## 2. 运行配置与可比性

| 项目 | 本轮配置 |
| --- | --- |
| 云/端模拟硬件 | NVIDIA GeForce RTX 5090，32 GB VRAM |
| API 与存储 | 生产 FastAPI/Python SDK、PostgreSQL + pgvector、S3-compatible MinIO |
| 回答与证据核验 | `qwen3.8-max`，仅通过异步 OpenAI SDK 调用，`reasoning_effort=low` |
| 文本 Embedding | `jinaai/jina-embeddings-v5-text-small-retrieval@6856e76bb72982e58de0620458a4e8b3614da340`，SentenceTransformers + CUDA |
| 跨模态 Embedding | `jinaai/jina-embeddings-v5-omni-small-retrieval`；原始视频垂直切片由 GPU Worker 实测 |
| 模型训练 | 无微调、无 Benchmark 专用分类头；学习只发生在记忆、检索、反馈和生命周期状态 |
| 隔离 | 每个数据集样本/视频使用独立 tenant，`run_id` 写入 tenant 和 sidecar manifest |
| 因果性 | 只摄入提问时间之前完整结束的片段；M3 Robot 严格遵循 `before_clip` 边界 |

数据与代码固定为：LoCoMo `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`；EgoLife 数据 `143fb319be7aa5ae210c936bf4f0f3a86092afb0`、Evaluator `7a97157908757cc898c26835b718653055ecc5f5`；SuperMemory-VQA 数据 `1d228e0f10049a8a84c458dded2aa25b1e21ce8f`、源码 `8123980820ffa23a3452faa6bd8ce5dff0f03164`；M3-Agent 标注 `0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`、媒体 release `2672152eee36b25ccb38fdbc3b72135347adbb63`。

四套评测都走生产 `remember`/`recall`，没有读取答案或 evidence label。公开媒体限制决定了本轮输入：EgoLifeQA 使用 6,388 条官方发布的视听 caption；SuperMemory-VQA 使用 6,157 条对齐 transcript；M3-Bench 使用 M3-Agent 发布的 56,833 个 30 秒 memory-graph caption。另用真实 EgoLife 视频完成原始视听垂直切片，以验证上述文本派生输入没有掩盖产品媒体路径是否能工作。

## 3. 四个 Benchmark 结果

| Benchmark | 覆盖 | MindBridge | 当前公开可审计强基线 | 数值差 | 是否可直接声称 SOTA |
| --- | ---: | ---: | ---: | ---: | --- |
| LoCoMo token-F1 | 10 conversations / 1,986 QA | 53.09% | T-Mem 51.96% | +1.13 pp | 该指标超过；整体声明仍需官方 Judge 与三次重复 |
| LoCoMo LLM Judge | 1,540 non-adversarial QA | 81.43% | T-Mem 80.26% | +1.17 pp | 否；本轮 Judge 为 Qwen，论文为 GPT-4o-mini 且三次平均 |
| EgoLifeQA | 500 Jake QA | 61.20% | Human Visual-Audio Caption + EgoRAG 45.5% | +15.70 pp | 否；回答模型、检索和 Top-K 不同；可部署 Gemini 基线为 36.9% |
| SuperMemory-VQA Ans-F1 | 4,853 QA | 67.41% | Gemini-3-Flash + Video-RAG 83.9% | -16.49 pp | 否；本轮无原始帧/OCR/object 通道 |
| SuperMemory-VQA QA-Acc | 4,853 QA | 58.69% | 61.0% | -2.31 pp | 否；同上 |
| SuperMemory-VQA QA-MRR | 4,853 QA | 72.65% | 76.0% | -3.35 pp | 否；同上 |
| M3-Bench Robot semantic judge | 100 videos / 1,276 QA | 30.02% | RRM 39.6% | -9.58 pp | 否；Judge、输入表示和在线反馈协议不同 |
| M3-Bench Web semantic judge | 920 videos / 3,214 QA | 58.18% | RRM 54.4% | +3.78 pp | 否；Judge、输入表示、重复次数和在线反馈协议不同 |

公开对照来自 [T-Mem](https://arxiv.org/html/2606.15405)、[EgoLife](https://arxiv.org/html/2503.03803v3)、[SuperMemory-VQA](https://arxiv.org/html/2606.00825v1) 与 [RRM](https://arxiv.org/html/2607.28156v1)。数值差仅用于指示优化方向；`是否可直接声称 SOTA` 一列才是发布口径。

这里刻意使用“可审计强基线”而不是把所有公开数字混成一张榜。截至 2026-08-13，LoCoMo 还有
[Synthius-Mem 94.37%](https://arxiv.org/abs/2604.11563)（1,813 题）、Backboard 90.1% 的产品报告，
以及 [TrueMemory/EverMemOS 94.5%](https://github.com/buildingjoshbetter/TrueMemory/blob/main/benchmarks/locomo/BENCHMARK_RESULTS.md)
（1,540 题）；后者明确说明使用更宽松的自定义 Judge、绝对分数不能与论文严格结果直接比较。
相对本轮 `81.43%`，三项最高公开数字的表面差分别是 `+12.94 / +8.67 / +13.07 pp`；这些差值
混合了题集和 Judge 协议，只用于说明“尚不能宣布整体 LoCoMo SOTA”，不能据此决定代码优化。
因此本报告以题数、Prompt、Judge 和三次重复均披露的 T-Mem 官方路径作为主锚点，同时保留这些
更高公开 claim，避免把“最高数字”误写成“同协议 SOTA”。EgoLife 也同时区分可部署模型 captioner
的 36.9% 与人工 Visual-Audio Caption 的 45.5% oracle 参考；本轮使用发布 caption，不能忽略这项
输入优势。

### 3.1 LoCoMo：检索覆盖决定上限

官方非对抗 token-F1 为 `0.5309`，比 T-Mem 的 `0.5196` 高 `0.0113`。全部五类的综合分数为 `0.5934`，标注 evidence coverage 为 `0.7440`。按 evidence 命中程度分层后，完整命中的 1,128 题 F1 为 `0.6260`，部分命中的 194 题为 `0.3675`，完全未命中的 214 题只有 `0.1755`；因此下一阶段的最高收益点不是换回答模型，而是让目标证据可达。

使用 Mem0 相同 Judge Prompt、但将 Judge 替换为 `qwen3.8-max` 的诊断结果为 `81.43%`。按官方类别
映射：Single-hop `85.97%`、Multi-hop `72.70%`、Temporal `83.80%`、Open-domain `59.38%`。
与 T-Mem 同序的 `85.97% / 69.15% / 82.55% / 55.21%` 只能作非严格诊断，数值差分别为
`-0.00 / +3.55 / +1.25 / +4.17 pp`。token-F1 切片则是 Multi-hop `40.70%`、Temporal
`52.68%`、Open-domain `35.47%`、Single-hop `59.41%`；需要优先提高开放域和跨证据问题的答案
完整性，而不是基于错误编号映射去专项修改时间逻辑。

115 个非对抗 abstention 在 Judge 下全部错误；完整 evidence 命中的 Judge accuracy 为 `92.55%`，部分命中 `67.53%`，未命中 `35.05%`。446 个 adversarial 问题的严格拒答分数为 `80.94%`，其中 361 题明确拒答。两组结果共同支持先优化查询计划和 associative cue，并把“缺证据”与“证据存在但答案不完整”分开校准，而不是盲目增加上下文。

### 3.2 EgoLifeQA：需要把跨日层级与身份真正接入召回

500 个唯一问题全部完成，准确率为 `61.20%`，所有问题都召回了候选记忆。44 题因证据不足被
answerer 拒答，按精确选项协议全部计错；其余 456 题准确率为 `67.11%`。五类分别为 EntityLog
`53.60%`、EventRecall `65.08%`、HabitInsight `65.57%`、RelationMap `56.00%`、TaskMaster
`74.60%`。需要音频的题反而达到 `68.32%`，高于不需要音频的 `56.38%`，说明发布 caption 中的
Audio transcript 已经有效进入召回；需要姓名的题为 `59.12%`，低于其余题的 `64.84%`，端侧身份
profile 还没有进入这次全量 released-caption 评测。

165 个 `last time` 问题只有 `56.36%`，其余问题为 `63.58%`；Day 1 为 `70.59%`，Day 3/6/7
下降到 `51.76% / 56.58% / 52.63%`。这把优化目标收窄为跨日分层、最近一次事件排序和人物
ambient context，而不是笼统地“换更大模型”。数值相对 Gemini+EgoButler 高 `24.30 pp`，相对论文
人工 Visual-Audio Caption oracle 的 `45.5%` 仍高 `15.70 pp`；由于本轮直接使用 6,388 条发布
caption，并由 `qwen3.8-max` 在 Top-20 记忆上回答，两项都不是同协议 SOTA 声明。

论文在同一 Jake 500 QA quick-eval 中使用 EgoButler：先生成 30 秒视听 caption，再做日/小时层级检索，最终统一由 GPT-4o 回答。MindBridge 当前已保持严格时间因果性，但本轮直接消费发布 caption；这能评估长时记忆与问答，不能评估自身 captioner 的全量质量。后续应优先加入日→小时→片段 coarse-to-fine 查询、人物 profile ambient context，以及 `needs_audio`、`needs_name`、`asks_last_time` 三类失败切片的定向重读。

### 3.3 SuperMemory-VQA：拒答校准和视觉证据是两个独立问题

10 个 participant、4,853 个唯一问题全部完成，Recall 均返回候选记忆，回答和首选项解析也没有
失败。Ans-F1 为 `67.41%`，比 Video-RAG 低 `16.49 pp`；QA-Acc 为 `58.69%`，低 `2.31 pp`；
QA-MRR 为 `72.65%`，低 `3.35 pp`。5 题的模型输出只有一个选项而不是请求的四项完整排序；
Evaluator 对未输出的候选不补猜名次，正确候选若被省略则该题 reciprocal rank 为 0。

核心矛盾不是误答太多，而是过度拒答：3,385 个可回答问题中，1,628 个被选为
“This question can not be answered”，answerability recall 只有 `51.91%`；1,468 个不可回答问题中
只有 71 个被误判为可回答，所以 precision 高达 `96.12%`。可回答题 QA-Acc 为 `42.87%`，不可回答
题为 `95.16%`。因此简单提高拒答阈值只会让 Ans-F1 更差；必须先补足证据，再分别校准
answerability 和 option ranking。

六类 QA-Acc 为 conversational memory `93.24%`、intent recall `85.84%`、in-context retrieval
`52.39%`、timeline reconstruction `46.67%`、visual recall `39.06%`、object location memory
`37.51%`。对应的 answerability recall 在 object/visual 上只有 `13.87%/16.07%`，而
conversational/intent 为 `92.99%/95.76%`。transcript 路径已经证明对话和意图记忆有效，主要差距
被精确定位到 transcript 无法承载的物体位置、纯视觉状态，以及需要跨片段排列的时间线。

论文最强 Video-RAG 同时检索 ASR、OCR、物体检测，并向 VLM 提供相关时间片的 32 帧。本轮只有发布 transcript，所以 conversational memory 是有效诊断，物体位置、视觉回忆和时间线则天然缺少一半证据。Ans-F1 与 QA-Acc 必须分开优化：前者需要 answerability calibration；后者需要原始帧/OCR/object 检索和四个候选项的稳定全排序，不能用更激进的拒答掩盖视觉召回失败。

### 3.4 M3-Bench：最缺的不是更大的静态 Top-K，而是反思式查询

Robot 的 `qwen3.8-max` 语义 Judge 为 `30.02%`，所有 1,276 个 Judge 输出均有效。全部题都有候选记忆，
但 552 题被 answerer 拒答且全部判错；其余 724 题的正确率为 `52.90%`。按类型看，Person
Understanding `38.32%` 最好，General Knowledge Extraction `21.10%` 最弱。

Web 的 3,214 个 Judge 输出同样全部有效，语义准确率为 `58.18%`；它在数值上比 RRM 高
`3.78 pp`，但本轮只有一次运行，Judge 是 Qwen，输入是发布的 memory-graph caption，而且没有
RRM 使用的跨 mini-batch 延迟真值反馈，因此不能据此声明 SOTA。679 个拒答全部判错；其余 2,535
题的准确率为 `73.77%`。按类型为 Multi-Evidence `55.89%`、Multi-Hop `55.14%`、Cross-Modal
`50.94%`、Person Understanding `62.34%`、General Knowledge `59.05%`。对照 RRM 同序类别的
`50.0% / 37.0% / 50.6% / 62.0% / 61.0%`，唯一数值落后的是 General Knowledge `-1.95 pp`；
其中论文的 Multi-Event 只作为本轮 Multi-Evidence 的最近对照，标签并不完全相同；这些仍只是
非严格切片诊断。

Web 首轮并行批次只产出 885/920 个视频，35 个因上游结构化输出失败而未产出完整分片；随后
选择性重跑其中 23 个并保留成功结果，剩余 12 个才用修正后的 runner 完成。因此最终结果由
908 个 `m3_production_api_v6` 分片和 12 个 `v7` 分片组成，且前 23 次选择性重试可能向上偏置
分数；`58.18%` 只能标记为全覆盖诊断结果，不能作为单 revision 的严格榜单结果。`v7` 已在 SDK
重试耗尽后把 `model_request_failed/model_output_invalid` 明确计作该题错误并继续，最终 3,214
题中有 10 题走了该异常路径。作为完整性敏感性分析，如果把这 35 个重跑视频的 133 题全部保守
计错，准确率为 `55.94%`，相对 RRM 仍只有不可严格比较的 `+1.54 pp`。一个可复现失败样本的
衣服颜色问题检索到 20 条、共 33,109 字符的候选，其中混入无关的暴力案件描述，随后 provider
在生成前拒绝请求。这说明需要提高 context precision、按证据充分性补查，并对最终 evidence pack
去除无关内容，而不是无上限增加 Top-K。

RRM 的消融给出直接路线：M3-Agent 基线 Robot/Web 为 `30.5%/48.6%`；加入 Online Query Reflection 后为 `34.9%/50.2%`，再加入 Reflective Experience Memory 为 `38.0%/53.0%`，完整 lifecycle 后为 `39.6%/54.4%`。它使用跨 mini-batch 的延迟真值反馈，MindBridge 本轮没有消费任何标签，因此不是同协议竞争；但“先诊断缺失证据→生成补充 query→只把新检索事实交给 answerer”完全可以成为无标签产品能力。

## 4. MaaS 全流程实测

| 能力 | 真实运行结果 | 验收结论 |
| --- | --- | --- |
| 原始视听写入 | 30 秒 EgoLife 原始视频经端侧 Outbox、S3、Worker 与 Omni/VLM 产生 13 条 evidence-grounded memory | 通过；5090 用时 508.74 秒，本轮不做延迟门禁 |
| 端侧近期记忆 | 同一 Evidence 在同步前使用本地 `file://`，同步后可由云端召回；删除后本地记录消失 | 通过 |
| 纠错 | 旧 memory 被版本化 supersede，Recall 返回“inside the green tool drawer” | 通过 |
| 强化 | useful feedback 后状态进入 `strengthened` | 通过 |
| 冷却 | 未使用旧 memory 经 lifecycle sweep 进入 `cold` | 通过 |
| 回热 | Recall 命中 cold memory 后原子恢复 `active`，`useful_access_count=1` | 通过 |
| Consolidation | 首次幂等调用已提交 1 Episode；工件记录的重放新增 0 Episode、5 个 semantic Claim、4 个 Summary；最终 7 Event、1 Episode、26 verified Claim、38 Memory | 通过 |
| 显式遗忘 | 云端 GET 返回 `410 memory_deleted`；S3、本地媒体、近期记忆、身份模板和关联证据均删除；tombstone 已应用 | 通过 |
| 多租户与可恢复性 | Bearer allowlist + PostgreSQL 强制 RLS；重复写使用稳定 idempotency key | 通过 |
| 并发 Recall | 修复访问计数更新的锁顺序后，8 个重叠并发访问回归测试通过；完整测试套件 316 项通过 | 通过 |

本轮 consolidation 首次真实调用暴露了模型偶发 singleton、混合 Claim type 和非 JSON 结构。三个 adapter 现在都只进行一次受限 JSON-mode 重试；Claim 还在同一入口执行 source ID、claim type 与关系方向的语义验证。没有引入新的 provider abstraction 或自研 HTTP 层。

端侧删除回归还覆盖了一个真实一致性缺口：只删除 face/voice template 会留下
`edge_face_voice_evidence`，使已删除的人脸仍可能通过旧关联解析为声纹身份。删除 Inbox 现在在同一
SQLite 事务内移除源 Observation 的关联，以及任一端模板已不存在的悬空关联；测试先建立可解析的
face↔voice 绑定，再应用 tombstone，并确认只能由仍存在的声纹模板独立解析。

“自遗忘”在当前产品语义中是可逆的强度衰减与自动降冷，不是后台擅自物理删除用户经历：旧且无用的
memory 会进入 `cold`，命中后可以回热；不可逆删除仍要求显式 `forget`，并留下不含内容的 tombstone
防止离线设备复活数据。两条路径本轮都已验证。若产品目标改为自动硬删除，还需要用户同意、保留期
和误删恢复策略，当前不能把它写成已完成能力。

## 5. 端侧身份实测

本轮不再只使用合成 embedding：

- 人脸使用 InsightFace `buffalo_l`、ONNX Runtime GPU，在两段真实 EgoLife 视频中得到 14 个可用人脸、512 维 embedding。同场最近样本 cosine `0.734013`，远样本 `0.022282`；以仅供该样本功能验证的中点 `0.378147`，SQLite 加密模板能重复匹配并区分远样本。
- 声纹使用 3D-Speaker 官方 `iic/speech_eres2netv2_sv_zh-cn_16k-common@v1.0.1`、192 维 embedding。Jake/Jake cosine `0.811870`，Jake/另一说话人最大 `0.315321`；诊断阈值 `0.563595` 下，3 个模板在加密 SQLite 中正确匹配与分离。
- face↔voice 的有界 ASD 证据、跨 Observation 累积、双向互为最佳、margin、撤销和 tombstone 已通过实现与测试；本轮没有运行真实 ASD 模型，所以不能声称完整人物一致性管线已通过真实视频验收。

这些阈值只证明功能可分，不是 FAR/EER、跨日身份准确率或 Jetson 阈值。下一次身份验收必须使用带真值的机器人 replay 报告 TAR@FAR、false-link、跨日 IDF1 和撤销延迟。

## 6. 优化优先级

### P0：先补证据可达性

1. 把 EgoLife、SuperMemory 和 M3 的原始 video/audio、OCR、ASR、identity 与发布文本组成同一 EvidenceSpan；文本只作为派生索引，answerer 最终重看原始媒体。
2. 为 temporal/last-time 查询增加显式时间解析、发生时间排序和相邻 Episode 扩展；依据是 EgoLife `last time` 比其余问题低 `7.22 pp`，而非 LoCoMo 的错误类别映射。
3. 在 Recall 内加入最多两轮 Search→Evidence sufficiency→补充 query，而不是一次性扩大 Top-K。补充 query 必须可追踪、去重、限次且只影响检索。

### P1：让记忆在写入时为未来查询做好准备

1. 复用现有 Claim/Summary/Embedding 表达，生成 T-Mem 式 Entity、Bridge、Scene、Horizon 多视图 cue；cue 只决定如何找到证据，不进入答案事实通道。
2. 增加 query-level experience memory：记录“哪类查询缺了什么证据、哪个补充 query 成功”，按复用反馈强化、合并或衰减；不保存历史答案并且不依赖 benchmark 标签。
3. 将人物 persona 作为受证据支持的 ambient context，不与 Episode 竞争固定 Top-K；错误或过期属性继续走版本化 Claim。
4. SuperMemory 将 answerability 与 option ranking 分成两个受约束输出，分别校准，不用一个 confidence 同时承担拒答和四选一排序。

### P2：完成产品验收而不是继续堆框架

1. 用同一机器人 replay 在 Jina v5 Omni Small、Text Small、Nano 与必要候选间做 Evidence Recall@K bake-off；没有数据证明前不增加专用向量库或 reranker 服务。
2. 在 Orin Nano/NX/AGX 上复跑 capture、identity、outbox、断网恢复与 tombstone，记录 FPS/RTF、显存、功耗、温度、丢帧和队列增长。
3. 补真实 LR-ASD/Sortformer 或 3D-Speaker diarization 闭环；历史跨设备 identity alias 只有产品确实要求时再加入。
4. 用新 `run_id`、单一 `m3_production_api_v7` 和不选择性重跑的失败计分重跑 M3 Web，并按论文
   Judge 做三次重复；当前混合分片只保留为诊断基线。

## 7. 当前目标状态

| 目标 | 状态 |
| --- | --- |
| 可部署 MaaS 软件垂直链路 | 已完成 5090 功能验证 |
| 纠错、强化、冷热、回热、显式遗忘 | 已完成真实闭环验证 |
| 自学习、自进化 | 反馈、版本化 Claim、Consolidation 与生命周期已完成；跨任务检索经验学习尚未完成 |
| 端侧近期记忆、身份加密与删除 | 已完成；真实人脸/声纹模型功能 smoke 通过 |
| 四套 Benchmark 完整公开题集 | 已完成本轮可获得输入的评估 |
| LoCoMo SOTA | token-F1 超过公开强基线；严格 Judge SOTA 尚未确认 |
| 三套多模态 SOTA | 未完成；本轮是 memory-layer 诊断，原始 AV 与严格同协议复现仍缺失 |
| Jetson 部署验收 | 未开始；按用户要求暂缓延迟与硬件指标 |

因此，MindBridge 已经从“架构与单测”进入“可运行、可量化、能暴露真实瓶颈”的阶段，但最终 SOTA 和 Jetson 产品验收仍是明确未完成项。
