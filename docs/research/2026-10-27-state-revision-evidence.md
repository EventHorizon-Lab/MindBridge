# STATE 修订的依据强度审查 — 2026-10-27

## 结论与复现

审查基线为 `b6b5ad28`。风险成立，值得修复：这是原有 last-write-wins 策略的缺口，
并非违反旧契约的回归，也不是物理数据丢失。`write_memory_context()` 在可见 STATE
进入同一 lineage 后，按有效时间重叠调用 `_retire_displaced_lineage()`，不比较依据
或置信度。删除后的 `rebuild_reconciled_lineage()` 也按事务顺序无条件裁剪旧区间。

公开 SDK 固定用例先形成自 1 月起有效的 tea / `0.99`，再形成自 2 月起有效的
coffee / `0.1`。旧实现的 3 月视图只剩 coffee；tea 的历史版本和来源仍然存在，
但事后过滤低置信度无法恢复已退休的 3 月 tea 区间。原始观察标为 `USER_STATEMENT`
也不改变派生 proposal 的 `MODEL_INFERENCE` 依据。OpenAI adapter 没有把原始依据
复制到派生 proposal，这一点应当保持。

## 近期原始证据及适用范围

- [PreCoMem，arXiv:2610.29467v1，2026-10-21](https://www.alphaxiv.org/pdf/2610.29467)：
  将稳定支持、含糊线索、可靠矛盾分别交给 MAINTAIN、PROFILE、CORRECT；
  原文称 “PROFILE buffers ambiguous observations as low-confidence hypotheses”。
  这一点支持给弱假设保留空间，而不立即替换事实。本文没有复现其模型实验，也不把其
  阈值和效果直接迁移到 MindBridge。
- [When to Remember, When to Abstain，arXiv:2610.07100v1，2026-10-05](https://www.alphaxiv.org/pdf/2610.07100)：
  在合成人设冷启动提取中比较全局与类别相关的保留策略，发现不同类别的来源支持存在差异。
  这是写入边界应检查支持的证据，不足以为所有 STATE 确定一个通用 confidence 门槛。
- [When Evidence Changes，arXiv:2610.03902v1，2026-10-02](https://www.alphaxiv.org/pdf/2610.03902)：
  显式区分来源撤销、替换和对照，并同时观察修复后正确性与有效信息保留。
  该设置支持把撤销回放列为本次验收条件；其医疗记录、小模型及完整记忆输入设置不等于
  MindBridge 的检索性能评估。
- [Mem0 的 confidence 实践说明，更新于 2026-09-03](https://mem0.ai/library/agent-memory/confidence-score-in-ml)：
  区分支持、部分冲突和明确反证，提醒未经校准的分数不能当作正确概率。
  这是作者方的工程指导，不是对本仓库的独立效果验证。

上述资料支持改进写入策略的方向；以下具体规则是本项目的保守工程选择。

## 实施规则与取舍

STATE 的退休资格统一由 `_state_can_supersede()` 判断：

1. 比较派生 proposal 自身的依据：`USER_STATEMENT`、`RESPONSE_FEEDBACK` 属于显式
   断言；它们优先于 `MODEL_INFERENCE`。两种显式依据同级，原始观察不能替派生结论升级。
2. 同级时，新 confidence 必须不低于旧版本的当前证据 confidence。
   仅容忍 noisy-OR 计算引入的浮点舍入差异；这不是新的置信度阈值。
3. 不够强的候选保留来源、置信度及有效区间，作为未决冲突与可靠状态共存。
   新证据不必被拒绝或删除，旧状态也不因候选到达较晚而退出视图。
4. 等强度及更强的修订保留已有事务排序、区间切分和历史查询行为。
   同一 SQLite 批次的独立断言仍保留冲突，不能用插入顺序决定胜者。
5. 删除来源或派生结论时，回放使用同一规则及剩余证据的聚合 confidence。
   失去部分证据但尚存的 STATE 及其间接依赖也纳入回放，不能仅处理整条删除的记录。
   被 `CORRECT` 明确撤回的结论保持撤回；后续显式重新断言或 rollback 可以恢复参与资格。
6. 启用独立证据投影时，写入前用已有的有界证明计算评估待提交的 STATE assessment。
   它替换同一来源集合的旧 assessment，不能先用 proposal 高分撤下旧状态，再落成低支持分。

只提高查询阈值不能保护已经退休的区间。直接拒绝弱候选则丢掉后续证据可检查的替代
解释。本次选择保留冲突；没有引入事实/假设新类型、额外模型调用或未经评估的固定阈值。

`search()` 可以读取两个值及各自的置信度。`compile()` 仍遵循已有的保守冲突契约：
如果 `ContextBudget.min_confidence` 排除一个已知竞争值，就报告竞争证据不可用，
不会默默把另一个值单独当成确定事实。这避免把查询过滤当成冲突裁决。
明确撤回错误状态仍使用 `CORRECT`，再提交替代结论。

## 验证证据与限制

固定输入位于 `tests/unit/test_bitemporal_memory.py`，全部驱动公开 SDK，实际经过 SQLite、
Zvec、证据投影与双时态读取。生成模型为脚本后端 `drink-test` / `drink-test:v1`；
嵌入为固定 `tiny-test:4:l2-v1`。没有调用外部模型，也没有修改模型、prompt 或提取 recipe。

| 固定用例 | 通过条件 | 原策略 |
| --- | --- | --- |
| 推断 `0.99` → 推断 `0.1` | tea 与低置信度 coffee 共存 | 失败：tea 退出 |
| 用户陈述 `0.6` → 推断 `0.99` | 保留显式 tea | 失败 |
| 响应反馈 `0.6` → 推断 `0.99` | 保留显式 tea | 失败 |
| 用户陈述 `0.99` → 用户陈述 `0.1` | 不用更弱的同级证据撤下 tea | 失败 |
| 推断 `0.99` → 用户陈述 `0.6` | 新显式 coffee 修订推断 | 通过 |
| 推断 `0.99` → 响应反馈 `0.6` | 新显式 coffee 修订推断 | 通过 |
| 推断 `0.9` → 推断 `0.9` | 等强度 coffee 修订 | 通过 |
| 推断 `0.6` → 推断 `0.9` | 更强 coffee 修订 | 通过 |

修改前八项规则用例有四项失败；删除后重放弱候选的用例也失败。额外回归检查覆盖：
区间前后及 `known_at`、低置信度编译时的保守输出、弱候选的空 `superseded` 日志和 rollback、
`0.3` 等分值经 noisy-OR 舍入后的正常修订、当前聚合 `0.96` 而非首版本 `0.8` 的比较，
以及删除一份独立支持后重算存活 STATE。
复核另补充了独立证据模式的来源置信度下界、间接 RELATION 支持变化、`CORRECT` 后的回放
和 rollback、同一来源重新断言时替换旧 assessment 等可复现回归，修复后双时态文件共
31 项测试通过。

扩大回归集运行了以下命令，结果为 **486 passed**：

```bash
uv run --frozen --no-sync pytest -W error \
  tests/unit/test_bitemporal_memory.py \
  tests/unit/test_memory_control_plane.py \
  tests/unit/test_context_compiler.py \
  tests/unit/test_independent_evidence.py \
  tests/unit/test_corroboration.py \
  tests/unit/test_local_store.py \
  tests/unit/test_recovery.py \
  tests/unit/test_memory_search_index.py -q
```

本项五个 Python 文件的 Ruff 格式、lint 和严格 Mypy 检查通过；`uv lock --check`
通过，`git diff --check` 通过。全库 pinned Markdown lint 通过，本项文档及研究索引的
pinned Lychee 检查为 **64 OK**。共享环境中另有依赖同步，测试使用已安装环境的
`--no-sync`；全库门禁结果还受同时进行的其他修改影响，不能由上述局部通过推导。

全库 `pytest -W error` 首轮为 **8 failed, 2819 passed, 12 skipped**。失败为：

- `tests/unit/benchmarks/test_eval.py::test_response_cache_namespace_changes_with_runner_recipe`
  （缓存 schema 版本预期未对齐）。
- `tests/unit/benchmarks/test_eval.py::test_a_fully_cached_unit_completes_its_ingest_share_without_writing`
  （缓存 fixture 缺少新增字段）。
- `tests/unit/benchmarks/test_eval.py::test_runner_reports_cached_progress_before_pending_answer_finishes`
  （等待超时）。
- `tests/unit/test_openai_contracts.py::test_answer_context_rejection_preserves_evidence_and_does_not_retry`
  的四个参数用例（错误分类不一致）。
- `tests/unit/test_openai_sdk.py::test_embedding_retries_a_context_length_rejection_as_ordered_stills[400]`
  （上下文长度错误分类）。

这些文件在本项处理期间由并行任务修改，本项没有编辑它们。随后按上述函数重跑对应
测试，当前参数集合共 **18 passed**。这不等于在停止并行修改后的固定工作树上获得完整
套件全绿。12 项跳过来自未启用真实服务的 formation 时间质量评估；不作为模型质量证据。

末次全库 Mypy 为 **275 个文件通过**，Markdown lint 通过。全库 Ruff 格式检查曾在
370 个文件上通过；末次 lint 在并行新增的 `tests/unit/benchmarks/test_eval_diagnostics.py`
报导入排序错误，本项五个文件均通过。全库链接复查仍有两处外部网络失败：
`docs/operations.md` 的 Uvicorn resource-limits 页面连接失败，以及另一份 predicate
研究记录的 W3C SKOS mapping 链接超时；未添加忽略规则来掩盖失败。

这些用例验证确定性修订策略，不证明真实模型能正确识别事实、标注依据或给出校准分数。
没有运行外部 formation/answer 服务，也没有新的真实对话、LoCoMo 或 LongMemEval 成绩。
同依据的错误高分推断仍可能替换旧值，需另用真实质量评估检验模型提取。
强化证据本身不重排状态断言；新的修订及删除回放使用更新后的支持。

这是 STATE 行为契约的收紧；Python/REST/MCP 字段、依赖及 on-disk schema 均未因本项
修改而新增。打开已有库不会自动全量回放；本规则适用于后续写入和实际触发的 lineage 回放，
不声称静默修复所有历史误修订。
