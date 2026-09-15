# MindBridge 的记忆机制：从证据保存到可撤销的长期学习

## 结论边界

“记得更多”“回答更多”和“更有理由相信”是三个不同目标。MindBridge 应优化第三个
目标约束下的前两个目标：在相同资源与观察条件下，恢复更多相关经历，形成更多有
来源的稳定知识，并在证据失效时撤回依赖它的判断。单个问答 benchmark 的 accuracy
不能独立证明这些能力。

本轮研究选择原始证据约束的巩固作为第一个可实现、可证伪的机制。它改变证据如何
积累为长期记忆，涉及可见性、置信度、递归依赖和撤销，不改变回答提示或评判器。
它是针对 MindBridge 契约的算法组合实验。AND/OR provenance、来源监控和独立见证
均有先行研究；不能将组合实现称作已经证明全球首创的算法，也不能凭结构模拟宣称
突破真实多模态记忆能力的天花板。

## 从产品目标导出的记忆标准

对于陪伴机器人，一次语音与一次表情可能共同说明一个当下状态；只有多个相互独立
的经历，才可能支持长期倾向。如果对同一片段生成十份摘要就形成稳定人格判断，系统
虽然“记住”了更多文字，却没有获得更多事实。相反，如果每次联合感知都永远只能算
一份总支持，长期相处也无法积累知识。

以下目标分别对应战略中的五个层次，不能被一个回答指标替代。

| 层次 | 记忆问题 | 直接测量 |
| --- | --- | --- |
| Evidence Kernel | 原始观察与衍生判断是否可区分；撤销是否传播 | 完整见证率、来源错配率、撤销后残留率 |
| identity-aware omni formation | 谁在何时产生了哪些共同观察 | 跨批形成一致性、错误人物绑定率、缺失模态下的证据损失 |
| temporal-spatial relation projection | 当前状态、历史状态和同时冲突能否并存 | 状态转移正确性、时间泄漏率、错误覆盖率 |
| Agentic Slow Memory | 经历是否形成可修订的长期知识 | 合法巩固率、重复证据膨胀率、干扰曲线、恢复与反转成本 |
| Context Compiler | 在预算内交付多少有依据的可用信息 | 完整证据覆盖、错误肯定、预算超限、未满足证据义务 |

“更强”的基本比较应固定输入历史、模型与计算预算，测量这些能力的变化；再让多个
下游 Agent 在同一输出 Context 上完成任务。否则，改善可能来自更强的阅读器、更多
token 或更宽松的拒答，而非记忆。

## 现有代码中的三个机制缺口

**形成依赖写入批次。** `kernel/formation.py` 只把未完成形成的当前输入交给 former，
并要求引证位于该批次。当前跨见证测试在同一批次内联结两条观察。连续 `add()`、
`add_many()` 和逐条 `settle()` 因而提供不同的见证集合。这是未来情景重放实验的直接
切入点；但添加历史窗口会改变模型输入，必须与相同预算的时间窗口和语义检索对照。

**等价关系依赖谓词字符串。** 形成的 lineage 将主体、谓词与相关时空信息编码进
标识。不同谓词表达同一关系时，冲突检测可能落在不同 lineage 中。不能靠数据集专用
同义词表补洞；应研究有证据、可撤销的关系等价提案，同时测试住所与工作地点等
不可合并的负例。此方向会影响世界状态契约，本轮未同时修改。

**联合评估积累不足，多根摘要可能丢失来源重叠。** 默认 `_evidence_summary` 将全部
多源 clauses 取最大置信度，作为一个保守评估；单源 clauses 则按 capture group 合并。
`SOURCE_GROUP_QUERY` 在一个派生来源包含多个 groups 时使用该记录自身 ID。这导致两个
不重叠的联合情景不能形成第二份支持，而两个共享部分祖先的摘要可能被视为不同来源。
本轮只改变这个巩固机制，以避免同时改变输入、检索和回答而无法归因。

另有独立的调度可靠性问题：contradiction candidates 在过滤已审议工作前截断 lineage
窗口，可能使后面的待审议冲突长期得不到机会。它应单独修复，不能把这种修复包装成
记忆理论创新。

## 脑科学启发及其适用边界

Baddeley 的 episodic buffer 描述一个容量有限、能够结合不同信息来源的工作记忆
成分。对 MindBridge 的启发是保留“这些信号属于同一经历”的关系；它没有给出
机器人视频、声音和文本的可直接采用的绑定算法。[^buffer]

Complementary Learning Systems 将快速获取具体经历与较慢学习共享结构分开。工程上
可据此保留原始经历，并把抽象的改变放在显式慢循环中；不能将每次摘要都当成新的
经验。Tse 等关于 schema 的实验说明已有结构会影响新信息巩固，但实验对象和任务
不等于开放世界机器人，不支持直接照搬巩固时长或学习率。[^cls] [^schema]

来源监控研究讨论记忆、知识和信念的来源归因以及错误归因。本轮借鉴的是一个可检验
问题：系统是因为遇到了新的经历而加强判断，还是因为重读自己的产物而加强判断？
软件中的显式 evidence ID 比人的主观归因更便于审计，但前提仍是采集方正确表示
共同来源，模型如实申报推导依赖。[^source]

Hopfield 关联记忆提供从不完整线索恢复模式的机制，现代形式与 attention 存在数学
联系；论文中的容量结果依赖其表示与模型条件。它不自动解决人物混淆、证据来源或
状态过期。把检索向量再做一次 attention，不能据此宣称拥有新的长期记忆算法。[^hopfield]

Hawkins 与 Ahmad 的序列记忆工作讨论上下文相关的序列预测。对具身记忆更有意义的
候选问题是：同一观察在不同人物、地点和前序事件之后，是否应触发不同预测；预测
误差如何创建新情景边界。这与本轮固定证据巩固不同，需要独立的时间序列实验。[^htm]

数据库 provenance 已经用代数表示替代来源与联合来源，并讨论递归推导。因此本轮
保留 AND/OR 依赖不是新发明。具体工程探索在于把来源证明、共享中间推导、可见性
门槛和可撤销的双时间投影放入同一条 MindBridge 记忆生命周期。[^provenance]

离线预计算也不是天然的机制创新。Sleep-time Compute 研究把一部分计算移到查询前，
证明某些任务可交换查询时与空闲时的计算成本；这不能证明任意预先生成的摘要都更
可信，更不能省略预计算开销。[^sleep]

## 本轮算法

设一个原始 capture 为 `g`。每条派生记忆有若干替代评估，每个评估需要其列出的全部
来源。用 `+` 表示 OR，`*` 表示 AND：

```text
X = A * B + C
T = X + D
```

每条完整证明保留 capture 标识与经过的中间派生记忆标识。判断 `T` 的两条证明是否
独立时，只移除共同的目标 `T`；如果它们仍共享 capture 或中间记忆，就不能互相佐证。
例如 `X = A + B; Y = X; Z = X; T = Y + Z` 仍共享 `X`，不能为了取得两票而分别选
`A` 与 `B`。同一顶层评估内部的多个可选证明也只能算一次。

但不能将所有替代来源提前合并。例如 `X = A; T = X + B` 已有两份不重叠见证；增加
`X = B` 这个替代来源后，原来通过 `A` 的证明仍然存在。将 `X` 扁平化成 `{A,B}` 会
错误地令 `T` 失去支持。保留证明分支能够避免这一类非单调错误。

由于现有 inferred trait/name 门槛只需要两组，本轮不求任意规模最大集合打包：

```text
score(proof) = min(沿途评估与原始观察的 confidence)
support(T)  = 0、1 或 2（存在两份合法不相交评估时为 2）
score(T)    = max(单证明分数,
                  所有合法证明对的 1 - (1 - p) * (1 - q))
```

支持数和分数分别取最大值。一个联合评估置信度为 0.99，两个不相交的弱评估各为 0.1
时，可以报告支持数 2、分数 0.99；这两个量对应不同见证。分数不是经过校准的真实性
概率；同一模型的系统性偏差以及未申报的共同上下文并未被此计算消除。

每个投影最多读取 256 个节点（含目标）、接纳 4,096 个 clause-member 行（查询可多读
一个哨兵行来检测截断）；计算最多保留每节点 64 条证明，预算为 4,096 个
clause/join/pair 步骤。缺失或截断不能构造虚假证明；
达到上限时只使用仍完整的见证，并报告截断。它是可验证的下界，不是无限容量证明。
整个依赖闭包的更新成本仍随受影响的记忆数增长。

本轮没有新增公开的证明选择 schema。Context Compiler 仍按已有引证闭包规则编译，
所以“获得合法巩固支持”不等于“在任意紧预算或循环图中都能交付 Context”。下一步
若要让编译器选择最小完整证明，应单独比较证据完整性与预算成本，不能顺便放宽拒绝。

## 实验为什么没有测回答策略

A 使用旧投影；B 使用候选投影。各 case 的原始观察、capture IDs、巩固提案、模型
边界和 ContextBudget 相同。它们通过公开 `Memory` SDK 和真实 SQLite/Zvec 生命周期
执行，使用独立物理目录。A2 再运行一次 A，以检查非预期状态共享和结果噪声。
A/A2 是最终代码中关闭实验开关的 legacy 路径，B 则开启该开关；不是在两个不同
checkout 中运行完整旧版与新版。基线提交记录开发起点，运行源码哈希记录实际比较
的共同代码版本；全量既有测试约束 legacy 路径的行为回归。

固定提案是控制变量，不是一个智能模型：它消除了 formation 模型差异，使可见性和
错误佐证的变化能归因于证据机制。输入中不会附带正确可见性标签；期望只由实验评分
读取。常量 embedder 避免把检索排序质量混入证据图测试。编译预算为每臂相同的
256,000 字符、64 items；这是充分容纳小图的结构检查，不是紧预算效率证明。

日志中的 `same_inputs` 核对巩固调用次数、输入记录数和内容字符数，不表示序列化的
`MemoryRecord` 逐字节相同：独立物理库中的记录 ID 不同，投影上下文也会随策略改变。
固定提案生成器不根据这些上下文调整提案，因此可以隔离证据计算，却不能估计真实
形成模型受到新投影反馈后的行为。嵌入条数另外逐对比较；本实验没有外部模型调用。

五种载体为 text、image、audio、video、omni。媒体字节只用于检查资产身份、持久化
和证据流转，不经过感知模型；通过它们只能说明机制处理这些载体时一致，不能宣称
真实图像、语音或视频理解提升。

C 去掉来源约束，只计算不同顶层评估；D 将 OR 祖先做 union。两者是离线消融，
不冒充产品策略。它们分别检验“只多计票”和“过度保守地合并来源”是否足够。
开发集 7 个图结构，保留集 6 个由独立审查提出的结构。不同 seed 只改变名字和原始
写入次序；不同载体和 seed 的变体相关，不能伪装成独立自然样本来缩窄置信区间。
保留结构在正式运行前做过一次文本 smoke test；之后的身份生命周期修复没有更改这些
结构或期望值。正式重跑属于固定挑战集的回归验证，不是最终代码从未见过的盲测。

复现实验：

```bash
uv run --frozen python -m mindbridge.benchmarks.corroboration \
  --output autoresearch/corroboration-reproduction --split all --replicates 3
```

命令要求一个不存在的输出目录，先落盘 case manifest 和代码哈希，再逐条记录结果。
没有按表现删掉 case 的步骤。原始结果位于本地 `autoresearch/`，按仓库规则不进 Git。
正式结果和审查结论如下。

## 正式结果与接受决定

修正载体夹具后，完整运行 **585 次**：13 个结构 × 5 种载体 × 3 次变体 × A/A2/B。
每臂 195 次，未丢弃任何 case。以下先按 13 个结构报告，以免将相关变体冒充大样本。
同一结构的载体与顺序变体结果一致。

| 机制 | 合法学习：6 个正例 | 错误佐证：7 个负例，越低越好 | 可见性判断正确 |
| --- | --- | --- | --- |
| A：旧机制 | 4/6（66.7%） | 3/7（42.9%） | 8/13（61.5%） |
| B：完整且不相交的证明 | 6/6（100%） | 0/7 | 13/13（100%） |
| C：离线取消来源约束 | 6/6（100%） | 5/7（71.4%） | 8/13（61.5%） |
| D：离线合并 OR 祖先 | 2/6（33.3%） | 0/7 | 9/13（69.2%） |

B 相对 A 的 195 对变体中，75 对改善、0 对退化、120 对持平，来自 5 个结构改善。
A2 与 A 的全部可见性、置信度、编译选择和撤销结果一致。开发集的正确率由 5/7 到
7/7；独立审查提出的保留结构由 3/6 到 6/6。Context Compiler 对目标的选择正确率
同样由 120/195 到 195/195，585 次运行均遵守相同预算。

两个源删除结构中，A 正确处理 1/2，B 为 2/2；展开变体后分别为 15/30 与 30/30。
操作 rollback 的传播与失败原子性由单独的 SDK 回归测试覆盖。
其中有一个结构在删除后仍保有支持路径，但剩余路径共享原始来源：不能因为派生记录
尚未删除，就继续把它当成两份独立证据。

C 表明“多计票”不足：它在重复 capture、重叠摘要和共享中间节点上制造五类错误。
D 虽然没有虚假佐证，却在 OR 替代、交叉组合、剩余来源和绕行路径上漏掉四类合法
学习。候选同时需要保留完整替代证明和检查中间推导，不能只选择其中一个约束。

为进一步分离“原始 capture 去重”与“共享中间推导”的作用，又补做了**事后解释性
消融 E**，不纳入预先规定的接受条件。它用独立穷举校验器保留 AND、OR、capture
约束及同一顶层评估只能算一次，只移除中间节点的重叠检查。相同 13 个固定结构中，
E 合法学习为 6/6，却产生 2/7 错误佐证，正确率为 11/13。仅
`shared_intermediate` 与 `shared_deep_intermediate` 改变：支持数从 1 变成 2，分数
从 0.6 变成 0.84。这定位了共享摘要的自我佐证风险，但仍是离线代数归因，不是新增
产品实验臂。脚本、逐例结果和哈希保留在本地
`autoresearch/corroboration_intermediate_ablation.py` 与同名 `.json`。

| 实测资源，中位数 | A | B |
| --- | --- | --- |
| 写入阶段，含打开实例 | 438.22 ms | 445.68 ms |
| 编译报告耗时 | 10 ms | 9 ms |
| 整个 case，含关闭与可能的撤销 | 587.52 ms | 591.29 ms |
| 关闭后的目录大小 | 5,908,982 bytes | 5,916,964 bytes |
| 固定巩固调用 / 嵌入 items | 4 / 8 | 4 / 8 |

逐对写入耗时比的中位数为 **1.0111**；A2/A 为 1.0009。各对的巩固调用次数、输入
记录数、内容字符数与嵌入条数相同，外部模型调用为零。实验在全量测试结束后单独
运行，但没有隔离操作系统背景负载；这些小图中位数不支持显著加速或大图成本承诺。
身份名称变化带来的语音重嵌入由专项测试覆盖，没有纳入此固定图实验的性能分布。

本轮接受 B 为**默认关闭的实验机制**：通过预先规定的结构学习、错误佐证与撤销
门槛，并保留 A 的既有行为。没有据此接受“自然长期记忆全面提升”的命题，也不
将 100% 结构正确率称为真实多模态 benchmark 的 100% 准确率。

## 独立证伪与审查

独立校验器另写递归证明树穷举逻辑，对 1,000 个 seeded DAG 比较原图、排列与重复
评估、增加 OR 替代三个版本，共 3,000 次调用，没有找到支持数或分数不一致。新增
OR 使 95 个图的支持数提高、138 个图的分数提高，没有下降。seed 为 `260912731`。
此验证使用充足计算预算，最大目标证明数为 15，专门验证小图代数；不测试默认上限、
SQLite 生命周期或自然语言质量。脚本及结果保留在本地
`autoresearch/corroboration_oracle.py` 与 `autoresearch/corroboration_oracle.json`。

对抗审查并非只有结论确认。第一轮发现旧身份预计算会把证据不足的名称写入语音，
且祖先新增替代支持后，身份名称与索引文本可能没有随可见性更新。修复采用候选策略
的事务内准确投影；嵌入失败会撤回证据、名称、文本和 outbox 变更。第二轮进一步
发现两个人物共用一份语音时，删除路径会重复提交同一记录，导致删除失败。这些是
记忆生命周期的正确性缺陷，不能藏在平均准确率之后。

删除修复先在可回滚预览中取得准确名称，再一次性重建所有受影响人物共用的语音，
并与删除一起提交。扩展测试先复现重复 ID 错误，再验证嵌入失败保留证据与两个人名、
重试成功后同时撤销两个人名。第三轮对这个修复的独立审查与复测没有剩余发现。
这些审查使用独立上下文，未使用不同模型；它们也不构成整个仓库没有其他缺陷的保证。

首次五载体正式运行在 36 条结果后中止：omni 夹具把同一字节序列声明为图片与音频，
内容哈希相同但不可变资产元数据冲突。保留了全部部分结果与 `failure.json`，并用
公开 SDK 测试复现失败。修复只为测试字节添加模态前缀，未改变产品、证据图或期望
标签；复测通过后，在新的 `final-r2` 目录从头运行全部实验臂，没有跳过 omni case。

## 代码与验证记录

纯计算位于 `src/mindbridge/kernel/corroboration.py`；SQLite 有界读取位于
`infrastructure/local/store/_corroboration.py`。Formation 与 ControlPlane 负责把受影响
的身份语音重建接回生命周期，store 负责同一事务中的一致性。没有改变回答提示、
引入额外服务、添加依赖或修改 SQLite schema。新增配置默认为关闭，并按物理目录
固定 `evidence.projection_recipe`；旧版本程序不理解这个标记，不能用来打开实验库。

最终代码（含修正后的夹具测试）上的 `uv run --frozen pytest -W error` 结果为
**2,251 passed**，耗时 372.50 秒。`uv lock --check --default-index https://pypi.org/simple`、Ruff 格式检查、
Ruff 静态检查、mypy（222 个源文件）与 `git diff --check` 均通过。运行环境是 Linux
x86_64、Python 3.13.12；未在本机重新执行 Python 3.10–3.14 的完整版本矩阵。
`CONTRIBUTING.md` 固定的 Markdown 检查（57 个文件）与链接检查（593 项）均为
零错误；链接检查报告 8 个重定向。

正式实验的基线提交为 `599ae4b827a49ba0a86f972bdef0db2ffb29dd43`，产品和 harness
Python 源码的集合哈希为
`264e179819e09ecd43c701355bd193335bb8aaa5df98153783972b7e052793f7`，case manifest
哈希为 `ab1cb4349a2722b6f565c7f3abb8bb936471926453cb6ade363c0d3dd0a42960`。
实验结果、运行 manifest 与迭代记录分别保留在本地
`autoresearch/corroboration-260912-final-r2/` 和 `autoresearch/corroboration-iterations.tsv`。

## 其余研究线索与下一步门槛

以下是本轮审阅的指定资料目录。它们是发现问题和定位原始研究的线索，技术结论以上述
原论文为依据，不把厂商文章当作 MindBridge 的性能证据。

| 指定资料 | 对后续实验的用途 |
| --- | --- |
| [Multimodal integration](https://mnemoverse.com/docs/research/memory/architectures/multimodal-memory-integration) | 检查一个情景的跨模态见证是否被错误拆票 |
| [Self-organizing memory](https://mnemoverse.com/docs/research/memory/architectures/self-organizing-memory-systems) | 设计适应性情景边界；与固定窗口作同预算比较 |
| [Working memory](https://mnemoverse.com/docs/research/memory-science/working-memory) | 区分有限工作集、长期记录与证据义务 |
| [HTM](https://mnemoverse.com/docs/research/memory-science/jeff-hawkins-hierarchical-temporal-memory) | 研究前序上下文与预测误差的事件分段 |
| [Schema formation](https://mnemoverse.com/docs/research/memory-science/schema-formation) | 将泛化收益与错误覆盖旧经历的损失一起测量 |
| [Cognitive memory](https://mnemoverse.com/docs/research/memory-science/bernard-widrow-cognitive-memory) | 关联恢复的历史线索；不据此宣称神经实现等价 |
| [Hopfield](https://mnemoverse.com/docs/research/memory-science/hopfield-associative-memory) | 将部分线索恢复与干扰/错误补全一起测量 |
| [Episodic and semantic](https://mnemoverse.com/docs/research/memory/cognitive-models/episodic-semantic-memory) | 检查具体经历如何变成仍可追溯的共享知识 |
| [Computing stack](https://mnemoverse.com/docs/library/memory-across-the-computing-stack) | 避免把缓存、向量索引和长期认知记忆混为一谈 |
| [Transactive memory](https://mnemoverse.com/docs/research/memory-science/transactive-memory) | 后续研究“谁知道什么”；不把共享数据库当成信任证明 |
| [Agent consolidation](https://mnemoverse.com/docs/library/agent-memory-consolidation) | 明确慢循环成本与未经新观察支持的自我强化 |

后续有三个可独立淘汰的实验，而不是一次改完所有层次的架构重写。

1. **因果情景见证重放。** 对写入新观察恢复历史见证，比较固定时间窗口、语义邻居和
   自适应边界。主指标是跨批一致性与错误人物绑定；检验乱序到达、同名人物、地点切换
   和丢失模态。未来信息不能进入当时的形成过程。
2. **可撤销关系等价学习。** 从多个经历提出关系槽位对齐，同时保留原谓词与拒绝记录。
   用同义改写、反义关系、位置/归属混淆和真实状态改变测量 false merge 与 conflict recall。
3. **由外部反证驱动的慢循环。** 用新的观察、显式纠正与检索失败触发重审，保留旧版本；
   自己生成的总结不能充当新的观测奖励。测量长期漂移、干扰随历史增长的曲线和撤销成本。

从本轮走向“广泛 work”仍需冻结真实长期交互测试集，至少覆盖不同模型、真实模态、
身份切换和多种应用，并同时报告 formation 的来源申报错误。只有在这些条件下仍保留
合法学习增益且不提高错误肯定，才有理由提升默认策略。现有结构实验不能代替这一步。

## 原始资料

[^buffer]: Alan Baddeley. [The episodic buffer: a new component of working memory?](https://pubmed.ncbi.nlm.nih.gov/11058819/). Trends in Cognitive Sciences, 2000.
[^cls]: Dharshan Kumaran, Demis Hassabis, James L. McClelland. [What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated](https://stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf). Trends in Cognitive Sciences, 2016.
[^schema]: Dorothy Tse et al. [Schemas and memory consolidation](https://pubmed.ncbi.nlm.nih.gov/17412951/). Science, 2007. 本轮核对可访问摘要，不外推其动物实验时长。
[^source]: Marcia K. Johnson, Shahin Hashtroudi, D. Stephen Lindsay. [Source monitoring](https://pubmed.ncbi.nlm.nih.gov/8346328/). Psychological Bulletin, 1993.
[^hopfield]: Hubert Ramsauer et al. [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217). arXiv 2020, revised 2021.
[^htm]: Jeff Hawkins, Subutai Ahmad. [Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex](https://arxiv.org/abs/1511.00083). 2015 preprint; Frontiers in Neural Circuits, 2016.
[^provenance]: Todd J. Green, Grigoris Karvounarakis, Val Tannen. [Provenance Semirings](https://web.cs.ucdavis.edu/~green/papers/pods07.pdf). PODS, 2007.
[^sleep]: Kevin Lin et al. [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171). 2025.
