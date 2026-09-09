# 记忆后端本轮结论 — 2026-09-09

本轮实现的是一个范围明确的工程组合：持久化 `support = (A AND B) OR C` 形式的证据组，
在更正、删除、回滚和有预算的上下文编译中同步撤销失效分支；schema 18 另提供默认关闭的
原文片段交付，公共 Python 编译结果可选紧凑 ID 展示，而 REST/MCP 保持原有渲染。证据子句、
typed span 和紧凑局部 ID 都有公开先例，
不是 MindBridge 首创。这里可验证的增量，是它们在嵌入式事务存储、公共 SDK、REST/MCP、
迁移和编译闭包中的具体连接。

当前主树还把一次普通 `CONSOLIDATE` 操作引用的来源视为一个 AND 子句，不同操作形成 OR
备选；历史数据不猜测回填，`REINFORCE` 仍是独立 singleton。该变更发生在冻结的大模型实验
之后。本轮最终源码门禁在 CPython 3.12.11 上通过 1,876/1,876 项测试及 lock、Ruff、mypy
和 diff 检查；这不是 Python 3.10–3.14 的运行矩阵证明。

| 新鲜因果实验 | 结果 | 可以下的结论 |
| --- | --- | --- |
| LoCoMo conv-30，pre-witness → schema 17 | 52/72 → 54/72；5/7 discordance；精确来源召回 72/96 → 71/96 | 单个对话上的描述性变化，不显著，也不是跨对话复现。 |
| Persona2 causal-prefix formed，pre-witness full → schema 18 full | 官方 `ndcg_at_5`：`at_ai_directive_followup` 0.84246→0.87374；`hidden_persona_recommendation` 0.16884→0.16884；`personalized_recommendation` 0.17863→0.18483。98 个 changed-wire 问题中方向为 19 高、11 低、53 平、15 无可比标量 | 结果混合；该 cohort 的 2,886 个 active clauses 全是 singleton，不能把差异归因于 AND witness。 |
| Schema 18 full → compact | prompt tokens 712,466 → 626,878（-12.0%）；方向 14 高、15 低、94 平、26 无可比标量；12 个 sensitive-event 的主指标均值 0.74583→0.66667 | 已证明展示成本下降；compact 不是默认开启，未证明无损质量或更快、更便宜的后端。 |
| Raw baseline → partial excerpts | 73/149 个请求变化；任务结果有升有降；一行 judge 错误保留 | 原文 containment 成立，但不能自动证明说话人、指代、适用条件或撤回语义；功能保持默认关闭。 |

LoCoMo conv-26 的冻结 compiler-v3 对比把声明证据闭包从 2/138 提升到 138/138，但共同缓存
judge 的答案分数仍是 97/138 对 97/138。两个手工撤回探针中，后端机械地从 0/2 提升到
2/2 正确撤除 unsupported derived claim；两臂最终回答却都 0/2 拒答，因为仍看到仅有代词的
原始观察。存储撤回保证不能等同于任意 agent 必然拒答。

本轮没有达到或证明 SOTA，也没有证明总体语义、隐私、情感理解、身份识别或成本优势。
Persona2 的定性复核还发现相同请求可产生不同答案和相互矛盾的 judge 理由。下一阶段最主要
的瓶颈不是继续扩大 top-k，而是对说话人、指代对象、事实适用范围、时间可用性和撤回条件做
可验证的语义约束；模型声明的 witness 和字节级原文片段都不能替代这一层。

Fresh formed 的 447 行评分全部完成且无 scorer error。pre-witness 与 schema-18 full 的 51 个
同字节请求中有 13 个答案不同；异质指标方向各有 1 个看似胜负，因此不计作后端效果。28 个
gap-repair 问题还使实际生成顺序偏离冻结 roster；QID 集合和三臂顺序一致，评分前仅按 exact
QID 离线对齐，未重生成或改答案。schema-18 构造的 all-attempt formation tokens 比 baseline
高 4.926%，所以 compact 的读取节省也不是总体成本优势。

完整协议、任务级结果、失败日志和 artifact hash 见
[评测报告](2026-09-08-memory-backend-evaluation.md)，设计边界和公开先例见
[研究报告](2026-09-08-memory-backend-research.md)。
