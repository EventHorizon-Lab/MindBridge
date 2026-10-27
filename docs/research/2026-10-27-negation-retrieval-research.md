# 否定敏感检索的论文与开源方案评估

核验日期：2026-10-27。本文评估中文与多语言检索中的否定、排除、作用域和对象绑定，
用于判断 MindBridge 下一步应验证哪种语义能力。文中的模型分数来自注明的原论文，
不是本仓库重新运行模型得到的结果；不同论文、候选池和指标之间不能直接排名。

**建议：保留当前通用召回与纯 IDF 基线，优先离线验证有界的语义重排；对显式排除查询，
再验证查询分解；对需要稳定判断“谁的什么属性被否定”的能力，引入中文主体与作用域标注。**
删除字符特权只修复了评分职责，不能替代这些语义能力。没有核验到一个可直接接入、
对一般中文记忆查询已经证明无回退的通用方案。

## 先区分三个任务

| 任务 | 示例 | 应评估的能力 |
| --- | --- | --- |
| 寻找否定事实的证据 | 找到“小林不喜欢咖啡”的记录 | 主体、关系、对象及极性的联合匹配 |
| 排除某个结果或主题 | 推荐饮料，但不要含咖啡因的 | 区分允许对象与排除条件，验证实际属性 |
| 核查有争议的事实 | 小林现在究竟喜不喜欢咖啡 | 保留正反证据，处理来源、时间与不足证据 |

这是本文对 MindBridge 的任务划分，不是新的已实现 API。它决定了哪些论文适用：
对“找不喜欢咖啡的记录”，咖啡必须出现在正确证据里；扣除所有与“咖啡”相似的文档，
可能正好惩罚目标。对事实核查，矛盾记录也可能是必须交给回答器的证据。

结构化布尔排除已经成熟。例如 Elasticsearch 的
[`filter` / `must_not`](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query)
按确定的字段条件过滤。但从自然语言中正确抽取条件是另一项任务，
文本里出现某词不能直接等价为对象具有该属性。缺少某个属性记录也不能当成它为假。

## 最强的共同证据来自否定专项评测

[NevIR](https://www.alphaxiv.org/pdf/2305.07614) 使用 2,556 组最小差异文档对，
为每对文档构造两个方向相反的查询。其 paired accuracy 要求两个方向都排对，
独立随机排序的基准是 25%，不是单个二选一问题的 50%。它测量局部区分能力，
不等价于大候选池的 Recall@K 或最终回答准确率。

SIGIR 2025 的独立复现给出了有选型价值的同协议结果：
[Reproducing NevIR，Table 1](https://www.alphaxiv.org/pdf/2502.13506?page=6)。

| 论文中的模型 | 架构 | NevIR paired accuracy |
| --- | --- | --- |
| text-embedding-3-large | bi-encoder | 22.6% |
| bge-reranker-v2-m3 | cross-encoder | 43.5% |
| jina-reranker-v2-base-multilingual | cross-encoder | 65.2% |
| RankGPT o3-mini | listwise LLM reranker | 77.3% |

这些是该论文当时的模型和输入设置，不能当作当前同名权重的复测结果，也不是中文成绩。
它们支持的结论是：联合读取查询与文档更值得验证，但“cross-encoder”这个类别本身
不保证充分理解否定；同属多语言重排器的结果仍可相差很大。

更关键的是跨数据集迁移。下图保留原论文的坐标和图例：纵轴是 NevIR，横轴是 ExcluIR，
颜色表示微调数据，形状表示模型。单独沿纵轴改善，不保证同时沿横轴改善。

![不同微调数据在 NevIR 和 ExcluIR 上的表现](artifacts/negation-retrieval-research/figures/nevir-generalization.png "原论文 Figure 4。单集否定微调不能直接证明跨任务迁移。[PDF 第 8 页](https://www.alphaxiv.org/pdf/2502.13506?page=8)")

该复现还发现，只按 NevIR 选 checkpoint 会损害一般检索；联合观察 NevIR 和 MS MARCO
可缓解问题。[研究的代码](https://github.com/thijmennijdam/NevIR-reproducibility)
包含训练与跨数据集评估流程。选型必须同时测否定区分与一般检索，不能只优化前者。

## 路线一：高召回后接语义重排

这是我对 MindBridge 最优先的实验建议。候选发现仍使用现有 dense + lexical 路径，
对有界候选联合编码 query 与 evidence，再决定哪些记录满足当前任务。
重排器能阅读关系、对象和上下文，而纯词面覆盖只观察到 term 是否出现。

需要比较的现成候选如下。参数量来自官方模型元数据，完整 revision 保存在
[模型快照](artifacts/negation-retrieval-research/model-pins.json)。

| 候选 | 规模与许可 | 值得验证的理由 | 不能据此承诺的能力 |
| --- | --- | --- | --- |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 约 568M，Apache-2.0 | 成熟的多语言 query-document 编码接口 | 中文否定正确率或无普通检索回退 |
| [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 约 596M，Apache-2.0 | 支持指令与 100+ 语言，适合试验不同证据任务 | 通用多语言分数不能证明否定专项能力 |
| [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) | 约 278M，CC-BY-NC-4.0 | 上述独立 NevIR 研究有较强实证信号 | 该英文结果不能外推中文；许可与前两者不同 |
| [mDeBERTa MNLI/XNLI](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | 约 279M，MIT | 可作 entailment / contradiction / neutral 的诊断基线 | 通用 NLI 不等于经过验证的否定检索器 |

NLI 需要把问题转换为明确的待核查命题，再将记录作为 premise；不能把任意问句直接
当作待蕴含的事实。对于核查任务，contradiction 应标记为反向证据，而非自动删除。
这是应用设计要求，尚不是 MindBridge 的现有重排能力。

Qwen 官方模型卡支持自定义指令；Jina 官方实现支持超过单段长度时的滑动窗口。
这为任务适配提供接口，但实际的中文可靠性、文档截断风险、CPU/GPU 延迟与内存，
仍需要使用相同候选池和输入预算评估。不要把 sigmoid 后的重排分数当成已校准的事实
正确概率，也不要直接与 cosine 做 `max` 来决定准入。

## 路线二：中文否定主体与作用域抽取

这条路线最直接对应最初的错误。ACL 2015 的中文研究已明确提出 cue detection 与
scope resolution，并在标注原则 G6 写道：

> A negative or speculative character or word may not be a cue.

即字符或词的出现不足以确定它是线索，需要结合上下文、主体和作用域。
[CNeSp 原论文，第 3 页](https://aclanthology.org/P15-1064.pdf#page=3)
覆盖科学文献、评论和财经文本，并讨论双重否定、主语省略等中文现象。

更接近当前需求的是 [SSENE / NegComment](https://www.alphaxiv.org/pdf/2404.09830)。
其任务是联合抽取 `<subject, cue, scope>`，而非只找否定字。下图中蓝色表示主体，
橙色表示线索，绿色表示作用域；主体与被否定的属性可以相隔很远。

![中文否定主体线索与作用域的联合绑定](artifacts/negation-retrieval-research/figures/chinese-negation-binding.png "SSENE 原论文 Figure 1，保留原图与颜色。[PDF 第 2 页](https://www.alphaxiv.org/pdf/2404.09830?page=2)")

论文用七类美团评论构造 7,049 条 NegComment 句子，按三元组完全匹配计分，
SSENE 的 F1 为 75.78%。这证明存在更贴近中文对象绑定问题的训练任务，
但不是 75.78% 的检索准确率，也不足以支持无条件硬过滤。
[作者仓库](https://github.com/Easonsi/SSENE) 确实包含数据、模型与训练代码；
其公开环境依赖较旧，包括 PyTorch 1.13.1、Transformers 4.26.1 和 HanLP。

对 MindBridge，可借鉴的是标注与监督方式：先把极性绑定到明确主体、关系和对象，
再讨论排序或结构化查询。论文三元组不会自动提供规范化谓词、事件时间、来源可靠性，
这些需要额外建模与验证；模型输出也不能未经语义核验就成为持久事实。

两套标注还存在需要先解决的定义差异：CNeSp 的 G1 要求 cue 包含在 scope 内，
而 NegComment 要求三元组各元素为连续、互不重叠的 span。直接拼接标签会形成不一致
监督。因此，复用的是任务思想；若共同训练，必须先明确统一的 span 定义和转换规则，
再验证转换是否保留原句含义。

## 路线三：对显式排除做查询分解与额外评分

[ExcluIR](https://www.alphaxiv.org/pdf/2404.17288) 及
[公开数据入口](https://github.com/zwh-sdu/ExcluIR) 专门评估“要什么，同时不要什么”。
它与寻找含否定陈述的记录相关，但不是同一任务；不能用一个数据集的改善替代另一个。

[DEO](https://www.alphaxiv.org/pdf/2603.09185) 将查询拆成正、负意图，优化查询向量，
拉近正意图、远离负意图，同时约束不要偏离原查询。文档向量可以保持不变。
其 BGE-M3 实验中，NegConstraint 的 nDCG@10 从 0.7250 到 0.7946；
NevIR paired accuracy 则从 0.2668 到 0.2928，后一项仍远未解决问题。
[代码](https://github.com/taegyeong-lee/DEO-negation-aware-retrieval)
包含查询分解、向量优化和评估入口。没有模型微调不等于没有推理成本：
需要生成子查询、额外 embedding 和向量优化；未见中文否定检索切片。

[E-SENS](https://www.alphaxiv.org/pdf/2608.30130) 保留原查询，生成排除侧的 trap query，
用 `S(d)=s(d,q)-beta*s(d,q_trap)` 重新排序。在其 ExcluIR、
text-embedding-3-small 设置中，beta=0.2 时 R@5 保持 0.9377，
trap V@5 从 0.8789 降至 0.6753。这是该设置内的排除收益，不是中文事实否定能力。
论文承认文档级惩罚会误伤只是顺带提到排除对象的正确记录。
正文共享固定 beta 扫描与 limitations 中验证集调 beta 的表述存在差异，
因此不能把它当作已经校准好的部署参数或确定的小幅召回优势。

对 MindBridge，这条路线只应作为明确的 exclusion 分支评估。查询分解错了，
后面的向量优化或扣分会把错误放大；简单出现“不、没、无”不能作为启动条件。

## 路线四：学习兼容性或显式逻辑结构

[NS-IR](https://www.alphaxiv.org/pdf/2505.22299) 将自然语言查询和候选转换为一阶逻辑
表示，再结合语义相似度与逻辑对齐。它直接处理自然语言相似却违反条件的问题。
[公开仓库](https://github.com/xgl-git/NS-IR-main) 有模型、运行脚本和 NegConstraint 数据，
但论文明确承认 NL-FOL 转换的额外 API 开销与有限候选带来的召回限制。
显式写出逻辑符号不代表翻译一定正确，也不等于完整系统具有逻辑完备性保证。

[CoDeR](https://www.alphaxiv.org/pdf/2606.13204) 把 topical relevance 与 constraint
compatibility 分开训练和组合，这个职责划分值得借鉴。论文也说明训练基于英文极性资源，
新语言需要适配。**截至核验日，作者指向的 [CoDeR 仓库](https://github.com/NICE-HKU/CoDeR)
为空，GitHub contents API 返回 repository is empty；不能视为可直接复现的现成组件。**

[EXCISE](https://www.alphaxiv.org/pdf/2608.05497) 展示另一种有价值的设计：先学习是否
存在排除意图及其目标 span，置信度不足时保留原排名。它给冻结的 ColBERT 加入小型
检测与适配模块，而非依赖否定字触发。论文同时明确：其诊断与修复针对 MaxSim，
不直接适用于单向量或生成式评分，实验也只覆盖英文。MindBridge 的 Zvec 多 key／
父记录检索不是 ColBERT 的 token 级 MaxSim，不能直接套用其算法或阈值。

这些方向适合后续研究，不宜为修一个词表问题立即改造存储或部署新服务。

## 规则系统在什么情况下有价值

[negspaCy](https://github.com/jenojp/negspacy) 是可访问、许可明确的 NegEx 实现。
它在实体 span 周围区分 preceding / following negation、pseudo-negation 和
termination，而不是给任意匹配字符固定加权。README 列出的内置配置为英文及临床变体，
没有核验到开箱即用的通用中文配置。

在实体类型、句式和领域稳定时，这类规则可能是低开销选择；对 MindBridge 的开放域、
混合语言和多分句记忆，维护规则与消歧成本会升高。它更适合作为专项对照，
不能凭一组中文否定词就宣布迁移完成。

## 对 MindBridge 的实验优先级

以下是基于论文与仓库边界推导的建议，尚未实现或验证其收益。

2026 年 10 月的预印本
[Preference Is Not Position](https://arxiv.org/html/2610.26260v1)
补充了一个直接相关的验收陷阱：在固定、且同时召回正反文档的候选池中，局部偏好率
从 52.30% 提高到 56.62%，正确文档的 MRR 却从 0.5764 降到 0.3055。
同一设置中的 top-20 文本重排同时改善了偏好和位置。其 Qwen3-14B-AWQ 加融合分数
打破平分的完整流程达到 77.45% 单查询严格偏好与 0.7509 MRR。该模型不是上文的
Qwen3-Reranker-0.6B；这些也不是给定文档对的 NevIR 双方向准确率，不能与前表横比。
结果仅限英文、单个固定检索配置和已覆盖文档对；论文还披露了平分、回退与确认集复用。

对本项目的推论是：验证重排时必须固定候选发现，同时报告候选覆盖、正确文档的完整
列表位置、正反区分和成本。仅修复已知的两条记录排序，仍可能让二者一起沉到列表尾部。
这一原始研究与 2025 年复现中的跨数据集退化证据共同约束后续验收，不能用单一否定
分数替代实际返回质量。

1. **先冻结候选发现，离线比较重排器。** 在相同候选、相同证据长度下，
   比较现有排序、BGE、Qwen 和许可条件适用的 Jina 对照。扫描有界候选宽度，
   同时记录候选中是否已含目标；目标没有被召回，重排器无法恢复。
   先人工核对本地诊断的目标标签：旧多分句样例中，预设目标没有星期二证据，不能把
   原来的强制偏好直接当作语义 gold。应明确它是在找否定陈述、回答星期二的问题，
   还是需要报告证据不足；两个英文否定改写损失则单独保留为关系匹配对照。
2. **先证明任务能力，再加入运行时 slot。** 当前 `models/base.py` 没有公共 reranking
   protocol，[插件架构](../plugin-architecture.md#kernel-and-plugin-boundary) 也把它列为
   需要先证明窄合同的能力。语义推理应放在模型适配器，检索面负责有界调用、验证和回退，
   不应塞进 `lexical_relevance` 或让 `Memory` facade 决策。
   同时分开审计排序与准入：重新排列候选的模型分数，不能未经校准就代替现有相关性
   门槛。应在独立验证集上测错误接纳、错误拒绝与置信度不足时的中性回退。
3. **为中文补对象与作用域标注。** 复用 NegComment/CNeSp 的任务思想，
   再补日常记忆、时间变化、主语省略、无否定句和伪线索负例。抽取训练集不能只有
   含否定的句子；只在中文评论上训练也不能宣称跨语言迁移。
4. **显式排除单独做消融。** 比较原查询、正向改写、原查询加 trap penalty，
   验证到底是更好理解查询，还是仅仅减少目标对象的出现。不要把文档提及某对象
   等价为违反排除条件。
5. **谨慎进入硬过滤或持久事实。** 只有字段含义、对象绑定、证据来源和缺失值语义
   都明确时，才使用确定性条件。unknown 不等于 negative；矛盾证据在核查任务中应保留。

## 验收应覆盖什么

现有 [49 项受控测试与诊断损失](2026-10-27-chinese-negation.md) 应继续保留，
特别是“非常”“特别”、对象改写、英文否定改写及阈值准入。新增语义能力必须针对实际
损失验证，不能只把旧启发式移除后通过的测试再跑一遍。

建议增加以下独立切片，并按模板、实体和关系隔离开发集与留出集：

| 切片 | 应区分的现象 |
| --- | --- |
| 普通词与伪线索 | 非常、特别、别墅、无锡、不但、不仅 |
| 主体与对象 | 不是小林喜欢咖啡／小林不喜欢咖啡／小林喜欢的不是咖啡 |
| 多分句与改写 | 不喜欢红茶但喜欢咖啡；不爱喝与不喜欢喝 |
| 量词与双重否定 | 不全是／全不是；并非不喜欢／不喜欢 |
| 事实状态与时间 | 以前不喝、现在喝；一次没有购买与长期不喜欢 |
| 证据不足与冲突 | 没有相关记录／明确否定记录／正反记录同时存在 |
| 跨语言及混合文本 | 中文、英文、日文及实体混排，分别报告 |
| 长文与不可信内容 | 线索跨窗口、主体远距离、记录内伪装的排序指令 |

局部报告 NevIR paired accuracy；完整候选池同时报告 Recall@K、MRR/nDCG、错误证据
进入 top-K 的比例、普通查询的变化和端到端延迟。不同论文对 violation 的定义不同，
先固定本地定义，不混用数字。再检查 `ask()` 的有据回答、拒答与矛盾处理，
不能从重排指标直接推断回答质量。

[否定分类研究](https://www.alphaxiv.org/pdf/2507.22337) 也指出其生成数据的 pairwise
分类设置不能直接代表大语料排序。图中的跨数据集迁移结果进一步说明，应该用多任务
约束选择 checkpoint，而非只追求一个否定分数。新模型应钉住 revision、输入预算与推理
参数；置信度和准入门槛只能在独立验证集上校准。

## 证据与复现记录

本次核验读取了原论文的方法、实验或局限段落，检查了作者仓库、官方模型卡和当前
MindBridge 的模型／检索边界。没有下载模型权重、执行作者训练代码或新增模型服务调用，
因此这里给出的是实验优先级，而不是已证明适用于 MindBridge 的模型选型结论。

- [论文来源](artifacts/negation-retrieval-research/papers.json) 记录 12 篇原始论文。
- [仓库 commit 与状态](artifacts/negation-retrieval-research/repository-pins.json) 区分
  可访问代码、空仓库和已检测到的许可元数据；公开仓库不等于许可条件已经明确。
- [模型 revision 与许可](artifacts/negation-retrieval-research/model-pins.json) 是核验时的
  快照，不能自动对应到旧论文使用的 checkpoint。
- [原始 PDF 摘要校验值](artifacts/negation-retrieval-research/pdf-sources.json) 与
  [图像提取脚本](artifacts/negation-retrieval-research/figures/extract_figures.py) 保留
  原图页码、裁剪范围与 SHA-256。图像只从原始 PDF 渲染和裁剪，没有重绘数据。
- [本轮独立核验](artifacts/negation-retrieval-research/independent-verification.json)
  记录检索范围、主要方法与局限段落、实际源码检查、空仓库状态和两张原图的重提取核对。
