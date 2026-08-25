# MindBridge 已接入 Benchmark 的 SOTA 基线

> 状态：目标基线，不代表 MindBridge 已取得任何一项分数
>
> 更新日期：2026-08-25
>
> 范围：`src/mindbridge/benchmarks/` 中已有官方适配器的十二个 Benchmark
>
> 口径：**学术榜**指论文/官方 leaderboard 上可复现的公开结果；**工业榜**指厂商自测并对外发布的产品分数

## 1. 结论

十二个 Benchmark 里只有 **LoCoMo-Refined** 和 **Video-MME** 存在真正意义上的工业榜；其余十个是纯
学术榜，最强系统全部来自论文（Video-MME-v2 的榜由作者收邮件维护，尚不构成工业榜）。这决定了
MindBridge 的超越策略在两条赛道上完全不同：

| Benchmark | 官方指标 | 学术 SOTA | 工业 SOTA | MindBridge 需越过的线 |
| --- | --- | --- | --- | --- |
| LoCoMo-Refined | 4 类 LLM-judge（Qwen3-14B refined prompt） | 无（发布方未收录论文系统） | MemoraX AI 82.65 | 官方 judge 下 > 82.65 |
| Video-MME | MCQ Accuracy | video-SALMONN 2+ 81.6（含字幕） | Qwen3.8 Max 90.4（含字幕） | 记忆赛道先超 M3-Agent 的 long 61.8 |
| Video-MME-v2 | 组内非线性 Rating | Gemini-3-Pro 49.4（含字幕） | 无（榜由作者收邮件维护） | 先站上榜，Rating > 39.1 才进前三 |
| EgoLifeQA | 四选一 Accuracy | EgoGraph 45.8 | 无 | > 45.8（公开集 = A1_JAKE 500 题） |
| EgoTempo | 开放式 LLM-judge Acc | GPT-4o 42.0（人类 63.2） | 无 | > 42.0，目标逼近 63.2 |
| M3-Bench | Accuracy（GPT-4o judge） | NS-Mem 34.7 robot / 53.6 web | M3-Agent 30.7 / 48.9（字节 Seed） | > 34.7 robot、> 53.6 web |
| MemLens | LLM-judge Acc | Qwen3.5-122B 58.68（直读） | 无 | agent 赛道 > 32.82，真赢要 > 63.59 |
| MM-Lifelong | Acc + Ref@300 | ReMA 18.62 / 18.82 / 16.75 | 无 | > 18.8，Ref@300 > 16.4 |
| SuperMemory-VQA | QA-Acc / Ans-F1 | Gemini-3-Flash+Video-RAG 61.0 | 无（OSU × Meta Reality Labs 合作构建） | QA-Acc > 61.0 且 Ans-F1 > 83.9 |
| EgoMemReason | Accuracy | Gemini-3-Flash 39.6 | 无 | > 39.6，且必须提交官方 leaderboard |
| ATM-Bench | QS + Recall@10（三类题分别精确匹配 / Jaccard / LLM-judge） | MemPalace 56.8 main / MemoryOS 13.7 hard（Memexa 68.0 / 47.9 换了 judge） | 无 | 先与 SGM 列同口径，再设线 |
| Mem-Gallery | F1 + 五级 LLM-judge | MuRAG 0.6966 F1 / 0.8229 judge | 无 | 先与官方 backbone 同口径，再设线 |

最后两行的"需越过的线"是口径而不是数字，这是刻意的：3.10 / 3.11 两张表各自绑死在
`Qwen3-VL-8B-Instruct-FP8` + `gpt-5-mini` 和 `Qwen2.5-VL-7B` 一套固定口径上，而 MindBridge 走的是
自己的写路径和自己的答题模型，从没跑过同一套。在对齐口径之前填一个"> X"，等于先替读者认定两边
数字可比——那恰好是 3.10 / 3.11 正文明确否认的事。这两项 MindBridge 都还没有已发布的分数。

三个可以立刻看出的结构性机会：

1. **记忆系统在多模态长时记忆上普遍打不过"直接把上下文塞进模型"**。MemLens 上最好的记忆
   agent 是 32.82，同一批题直读 LVLM 是 63.59；EgoMemReason 上最好的 agentic framework 是
   34.0，纯 MLLM 是 39.6；ATM-Bench 的 hard split 把差距拉到全文最大——同 judge 下记忆系统最高
   只有 13.7，通用编程 agent 是 58.8 / 58.4，SGM Oracle 上限 60.5（按 3.10 的说明，agent 与记忆
   系统是两种测法，这里只能读数量级，不能读名次）。这正是 MindBridge 的正面战场：证明结构化
   记忆不是有损压缩。

   这条规律有边界，Mem-Gallery 就是现成的反例：官方表里两行 `Full memory`（0.3625 / 0.3354
   F1）明显低于检索式的 MuRAG（0.6966）——照官方给的行名读，把全部记忆整个塞给模型在这一项
   反而垫底。所以"记忆打不过长上下文"要按体裁分开讲，多会话对话上检索本身就是净收益。
2. **证据定位仍是空档，而且现在有两项在计分**。MM-Lifelong 上 ReMA 的 Ref@300 是 15.46，GPT-5
   是 0.44——差两个数量级；ATM-Bench 则把 Recall@10 直接列成官方指标，同一批系统从 main 的
   23.3–79.1 掉到 hard 的 7.0–44.7。MindBridge 的证据优先架构（原始视听跨度是最终证据）天然
   产出可定位区间，这两项都在正面考它。
3. **LoCoMo 的旧工业分数已经作废**。92.5/94.7 这类数字来自各家自定的 judge、backbone 和 4 类
   题面。LoCoMo-Refined 用统一的官方 judge 重打了同一批系统预测，EverMemOS、MemOS、MemPalace、
   Mem0 分别掉了 22.07、17.30、15.78、15.56 个百分点——差距本来就在 judge 里，不在系统里。

## 2. 评测口径警告

在报任何分数之前，这四条必须先落到 runner 里，否则数字无法被外部采信。

**原始 LoCoMo 的答案键和 judge 都不可用，这正是换成 LoCoMo-Refined 的原因。** 一份公开审计发现
1,540 题里有 99 题（6.4%）答案键错误，且官方 gpt-4o-mini judge 会接受最多 63% 的"故意写错"答案
——具体失效模式是：答案找对了会话、但没有任何细节时也判对，这恰好奖励弱检索。
（[审计](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer)、[代码](https://github.com/dial481/locomo-audit)）
[LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) 由 5 名标注者修订了 1,382 题
中的 337 题（措辞歧义、主客体反转、时间与原对话不一致），并换上一套要求"包含而不矛盾、完整而不
越界"的 judge：在 300 条人工标注上，与人类一致率从 43.67% 提到 86.33%。

**旧的"统一口径重跑"数字同样过期。** 那批数字（Dakera 88.2、Zep 修正 75.1、Letta Filesystem
74.0、Continua 74.4、Mem0 68.5）统一的是 answer model 和 response prompt，judge 仍是原始那个。
现在唯一有意义的统一口径是 LoCoMo-Refined 自带的 `Qwen/Qwen3-14B` + refined prompt。
（历史参考：[Continua 归一化重跑](https://blog.continua.ai/p/the-locomo-fair-fight)、[Dakera 对照表](https://dakera.ai/benchmark/)）

**Video-MME v1 已经饱和。** 官方 leaderboard 最后一次更新是 2025-09-28，前沿模型在含字幕设定下已
到 90.4；作者自己发布的 Video-MME-v2 用组内非线性打分把 Gemini-3-Pro 打到 49.4（人类 90.7），
明确指出 v1 的逐题准确率高估了真实能力。MindBridge 报 Video-MME 时应报 **long 子集 + 无字幕 +
记忆赛道**，而不是总分。

**EgoTempo 存在开放式与选择题两套口径。** 官方是开放式生成 + Gemini-1.5-Pro judge（GPT-4o
42.0，人类 63.2）；TGPO 那条 45.2 是把题目改造成选择题后的结果，两者不可混报。

## 3. 逐 Benchmark 详表

### 3.1 LoCoMo-Refined — 长程对话记忆（文本）

官方发布 1,382 题 / 10 段对话 / 4 类（category 1–4，按 `manifest.json` 分别是 213 / 299 / 68 /
802 题）。原始 LoCoMo 的 adversarial（category 5）在这一版里被整体删除，所以"4 类还是 5 类"这个
历史上最大的不可比来源已经不存在；521 题带图。

唯一有意义的榜是发布方用官方 judge（`Qwen/Qwen3-14B` + refined prompt，temperature 0，关思考
模式）重打同一批系统预测得到的：

| 名次 | 系统 | LoCoMo-Refined | 相对原始 judge 的跌幅 |
| --- | --- | --- | --- |
| 1 | MemoraX AI | 82.65 | 未公布 |
| 2 | MemOS | 63.60 | 17.30 |
| 3 | MemPalace | 58.68 | 15.78 |
| 4 | EverMemOS | 58.25 | 22.07 |
| 5 | Mem0 | 48.91 | 15.56 |

跌幅那一列才是重点：同一批预测、同一批题，只换 judge 就掉 15–22 个百分点。任何引用原始 LoCoMo
9x 分数的比较都应视为无效。

**MindBridge 目标**：官方 judge 下超过 82.65，并公开 answer backbone、judge 模型与 `--llm-judge`
用的是 `refined` 还是 `original`。用 `original` judge 报出来的数字不计入。

### 3.2 Video-MME — 通用视频理解

900 段视频 / 2,700 题，短中长三档，含字幕与不含字幕两套设定。

工业榜（含字幕总分，[BenchLM 2026-08-15 快照](https://benchlm.ai/benchmarks/videoMmeWithSub)）：

| 名次 | 模型 | 厂商 | 含字幕 |
| --- | --- | --- | --- |
| 1 | Qwen3.8 Max | 阿里 | 90.4 |
| 2 | Qwen3.7 Plus | 阿里 | 88.0 |
| 3 | Qwen3.6-27B / MiMo-V2.5 | 阿里 / 小米 | 87.7 |

学术榜（[官方 leaderboard](https://video-mme.github.io/home_page.html)，止于 2025-09-28）：

| 名次 | 方法 | 无字幕 | 含字幕 | Long 含字幕 |
| --- | --- | --- | --- | --- |
| 1 | video-SALMONN 2+（清华 × 字节） | 79.7 | 81.6 | 76.4 |
| 2 | Gemini 1.5 Pro | 75.0 | 81.3 | 77.4 |
| 3 | AdaReTaKe（哈工大 × 华为） | 73.5 | 79.6 | 76.4 |

记忆赛道（VideoMME-long，取自 M3-Agent 论文）：M3-Agent 61.8 > Gemini-GPT4o-Hybrid 56.5 >
Gemini-Agent 55.1。

**MindBridge 目标**：不参与"整段视频塞进上下文"的总分竞赛。先在 long 子集的记忆赛道超过 61.8，
再论证在无字幕设定下摄取一次、多次问答的成本优势。饱和度参见
[Video-MME-v2](https://arxiv.org/html/2604.05015v1)。

### 3.2.1 Video-MME-v2 — 组内非线性视频理解

800 段视频 / 3,200 题，A–H 八选一，**没有短中长三档**；四题一组，按组打分。分 Level 1/2/3 三层
认知维度，另有 `second_head`（10 个能力轴，论文雷达图用的就是它）和 `third_head`（33 个细分类）。

官方同时给两个数，[榜单](https://video-mme-v2.netlify.app/#leaderboard)按前者排：

| 名次 | 模型 | Non-Lin Rating（含字幕） | Avg Acc（含字幕） | Rating / Acc |
| --- | --- | --- | --- | --- |
| 1 | Gemini-3-Pro | 49.4 | 66.1 | 75% |
| 2 | Gemini-3-Flash | 42.5 | 61.1 | 70% |
| 3 | Qwen3.5-397B-A17B-Think (512) | 39.1 | 55.9 | 70% |
| 4 | MiMo-v2-Omni | 38.6 | 56.1 | 69% |

**Rating / Acc 这一列才是这个 benchmark 的产物**：它衡量"同一组相关问题能不能一起答对"。
Gemini-3-Pro 约 75%，InternVL3-5-241B-A28B-Instruct 约 56%，LLaVA-Video-7B 只有约 40%——逐题准确率
接近的两个系统，组内稳定性可以差一倍。这正是 v1 饱和到 90.4 却和真实体验脱节的原因。

**对 MindBridge 的意义**：这是十二个 benchmark 里唯一直接惩罚"部分检索"的口径。已有的端到端评测发现
部分检索在聚合题上比不检索更差，而 v1 的逐题准确率看不出这件事——组内非线性打分会把它变成明确的扣分。

**MindBridge 目标**：先在无字幕设定下产出一份完整的 800 组 Rating（哪怕很低），确认写入路径能覆盖
800 段视频；再对着 `second_head` 雷达定位薄弱轴。不要只报 `accuracy.overall`——那正是 v2 想淘汰的数字。

**成本提醒**：媒体 97.8 GiB / 40 个 zip，压缩包和解压后各占一份，规划磁盘时按两倍算。

### 3.3 EgoLifeQA — 一周第一人称生活记忆

六人共居一周、300 小时第一人称视频，500 道公开四选一题，五个类别。

学术榜（平均 Accuracy）：

| 名次 | 方法 | 平均 | 来源 |
| --- | --- | --- | --- |
| 1 | [EgoGraph](https://arxiv.org/html/2602.23709) | 45.8 | 时序知识图谱 |
| 2 | [EgoSelf](https://arxiv.org/pdf/2604.19564) | 40.6 | 个性化记忆 agent |
| 3 | LightRAG | 39.2 | EgoGraph 论文复现 |

对照：Gemini-1.5-Pro 36.9、GPT-4o 36.2、EgoGPT 32.2。EgoGraph 分项 TaskMaster 60.3 最高，
RelationMap 35.2 最低——关系推理是公认短板。

工业榜：无。

**MindBridge 目标**：平均 > 45.8，并把 RelationMap 单项拉过 35.2（身份原型 + 图镜像正对着这一项）。

> 口径核实（2026-08-17）：`lmms-lab/EgoLife` 的 32,817 个文件里只有一个 EgoLifeQA 标注文件
> `EgoLifeQA_A1_JAKE.json`，含 **500 题**，分布为 EventRecall 126、EntityLog 125、
> RelationMap 125、TaskMaster 63、HabitInsight 61。论文所说的"500 public QA pairs"就是这一个
> 文件，另外五名参与者的题目未公开发布。因此 MindBridge 现有的单文件跑法**已经是可比全集**，
> 不需要补下载。真正缺的是分类别口径：五类题量不等（63 到 126），macro 平均与题量加权平均能差
> 一个多点，所以 `EgoLifeMetrics` 现在同时给出加权 `accuracy` 与五类 `categories`，两种聚合约定
> 都能对上。

### 3.4 EgoTempo — 第一人称时序理解

500 道开放式问答，平均 45 秒片段，10 类时序推理，官方用 Gemini-1.5-Pro 作 judge。

学术榜（[官方开放式口径](https://arxiv.org/html/2503.13646v1) Table 3）：

| 名次 | 模型 | Accuracy |
| --- | --- | --- |
| 1 | GPT-4o（32 帧） | 42.0 |
| 2 | Gemini-Flash（1 FPS） | 39.1 |
| 3 | Qwen2-VL-72B | 28.4 |

人类 63.2，与最强模型差 21 个点。

选择题改造口径（[TGPO](https://arxiv.org/html/2603.27184v1) Table 7，**与上表不可混报**）：
Qwen2.5-VL-3B + TGPO 45.2 > EgoVLM GRPO(3B) 40.8 > GPT-4o 40.1。

工业榜：无。

**MindBridge 目标**：开放式口径 > 42.0。这条 benchmark 的片段只有 45 秒，考的是时序结构而非长
时记忆，因此它主要验证 `segment` 与时间戳契约的正确性，不是记忆规模。

### 3.5 M3-Bench — 长视频智能体记忆

robot 子集 100 段机器人视角视频 / 1,276 题；web 子集 920 段 / 3,214 题。GPT-4o 作 judge，与人工
一致率 96%；人类在 robot 上 90.7。

学术榜（[NS-Mem](https://arxiv.org/pdf/2603.15280) Table 1）：

| 名次 | 方法 | robot | web |
| --- | --- | --- | --- |
| 1 | NS-Mem（神经符号记忆） | 34.7 | 53.6 |
| 2 | M3-Agent（字节 Seed） | 30.7 | 48.9 |
| 3 | MA-LMM 24.4 / GPT-4o Socratic 28.7 | 24.4 | 28.7 |

NS-Mem 分项（robot）：MD 36.2、MH 31.5、CM 33.8、HU 45.7、GK 26.4。General Knowledge 是全场
最低项，两个系统都不到 27。

工业榜：M3-Agent 出自字节跳动 Seed，是这批里唯一的工业实验室系统，但以论文形式发布，仍按学术口
径计分。

**MindBridge 目标**：robot > 34.7、web > 53.6。GK 一项是明确突破口——它考的是跨事件沉淀出的世界
知识，正是 consolidation 与 semantic fact 的职责。

### 3.6 MemLens — 多模态长上下文记忆

789 题 / 五种记忆能力（信息抽取、多会话推理、时序推理、知识更新、拒答），四档上下文
32K–256K；记忆 agent 在固定的 195 题 canonical 子集上评测。judge 为
Qwen3-VL-235B-A22B-Instruct（GPT-5.4-mini 复判 κ=0.93）。

学术榜 · 直读 LVLM（全量 789 题最佳档）：

| 名次 | 模型 | Accuracy |
| --- | --- | --- |
| 1 | Qwen3.5-122B | 58.68（32K） |
| 2 | Kimi-K2.5 | 54.88 |
| 3 | Gemini-3.1-Pro | 54.10（128K 仍有 51.99，衰减最小） |

学术榜 · 记忆 agent（195 题子集，32K）：

| 名次 | 系统 | Accuracy |
| --- | --- | --- |
| 1 | MemAgent-7B | 32.82 |
| 2 | Mem0 | 31.79 |
| 3 | MemOS | 30.26 |

多模态记忆 agent 更低：M3-Agent 19.49、M3C 18.46、M2A 15.38。同一批 195 题上直读
Qwen3.5-122B 是 **63.59**。论文自己指出这里混入了输入适配的信息损失（文本 agent 只看 BLIP-2
caption），但结论仍然成立：**现有记忆管线相对直读丢失了视觉证据的保真度**。

工业榜：无。

**MindBridge 目标**：agent 赛道超过 32.82 只是入场；真正要证明的是超过 63.59 的直读上限——即
"结构化记忆 + 回原始证据"优于"把原图塞进上下文"。MindBridge 的证据优先设计（记忆保留可回源的
`image_uri`/`video_uri`，而不是只存 caption）恰好绕开了论文指出的那个损失点。适配器和 CLI 已经
支持 195 子集（`load_memlens_agent_subset`，`--agent-subset-index`），四档上下文都要跑。

### 3.7 MM-Lifelong — 百小时级终身流理解

105.6 小时连续直播流，day/week/month 三档；指标为 GPT-5 judge 的三档打分 {0, 0.5, 1} 与基于分桶
时序 IoU 的 Ref@300 证据定位。

学术榜（[ReMA 论文](https://arxiv.org/html/2603.05484) Table 4）：

| 名次 | 方法 | Val@Month Acc / Ref@300 | Test@Week | Test@Day |
| --- | --- | --- | --- | --- |
| 1 | ReMA（递归多模态 agent） | 18.62 / 15.46 | 18.82 / 16.37 | 16.75 / 11.51 |
| 2 | GPT-5 | 14.87 / 0.44 | 15.00 / 0.92 | 15.25 / 0.53 |
| 3 | Qwen3-VL-235B | 14.33 / 0.06 | 15.63 / 0.80 | 12.44 / 0.79 |

人类：Month 80.4、Week 95.6、Day 99.2。这是十二个 Benchmark 里人机差距最大的一个。

工业榜：无。

**MindBridge 目标**：Acc > 18.8，Ref@300 > 16.4。**Ref@300 是这里最容易拉开差距的指标**——端到端
模型基本是 0，说明"答得出但指不出证据"是普遍状态，而 MindBridge 每条记忆都带
`occurred_at`/`ended_at` 跨度。

> 工程缺口：`unofficial_reference_at_n()` 是 in-repo 诊断，分桶边界未对齐官方 scorer。对外公布的
> 数字必须由官方 scorer 跑 `pred` 行产生。
>
> **覆盖率规则（必须在第一次跑之前遵守，校验器无法强制）。** prepared timeline 的
> `require_ordered_non_overlapping_segments` 只检查排序与不重叠，`run_mm_lifelong` 只检查引用区间的
> 结尾落在 timeline 之内——**都不检查覆盖率**。因此"只准备每道题 `temporal_certificate` 附近的片段"
> 能拿到近乎完美的 Ref@300 并通过全部校验，这属于 §5 禁止的数据集专用旁路。
>
> 规则：prepared timeline **必须覆盖该 split 官方 `total_intervals` 的连续全长**，不得按题目位置
> 裁剪、抽样或加密。唯一允许的缺口是官方发布本身缺失的区间，且必须在 run 说明里逐一列出。
> 上报时必须同时给出 `segment_count` / `media_segment_count` / `caption_segment_count`，读者用它们
> 除以官方区间总数即可验算覆盖率——这三个数字是这条规则唯一的外部可核验凭据。

### 3.8 SuperMemory-VQA — AI 眼镜长时记忆问答

52.9 小时 AI 眼镜记录（RGB + 音频转写 + 眼动 + IMU + SLAM 轨迹），4,853 题，六类真实记忆任务，
含"无法回答"选项。OSU AIoT-MLSys Lab 与 Meta Reality Labs 合作构建，NeurIPS 2026 D&B 在审。

学术榜（按 QA-Acc 排序，[论文](https://arxiv.org/html/2606.00825v1)）：

| 名次 | 系统 | QA-Acc | Ans-F1 |
| --- | --- | --- | --- |
| 1 | Gemini-3-Flash + Video-RAG | 61.0 | 83.9 |
| 2 | Gemini-3.1-Pro + Video-RAG | 53.2 | 67.4 |
| 3 | GPT-5.4 + Video-RAG | 52.3 | 78.3 |

Video-RAG 全面优于 EgoButler。论文指出：83.9 的可答性判别配上 61.0 的答对率，说明瓶颈不在"知道
自己能不能答"，而在"取到精确的多模态证据并从干扰项中区分正解"。

工业榜：无对外产品分数（Meta Reality Labs 参与构建但未发布自家系统成绩）。

**MindBridge 目标**：QA-Acc > 61.0 且 Ans-F1 > 83.9 同时成立。runner 已实现 `qa_accuracy` /
`answerability_f1` / MRR 三项，与官方口径一致；拒答语义（`SUPERMEMORY_UNANSWERABLE_CHOICE`）
对应 MindBridge 自己的"无支撑则弃答"契约，这一项不应靠猜答案换分。

### 3.9 EgoMemReason — 周级第一人称记忆推理

500 题 / 三种记忆类型 / 六项核心挑战，平均每题 5.1 段证据、25.9 小时回溯。公开发布不含答案键，
需提交 [官方 leaderboard](https://huggingface.co/spaces/Ted412/EgoMemReason)。

学术榜（[官方页面](https://egomemreason.github.io/)，共 17 个系统）：

| 名次 | 系统 | Accuracy | 类型 |
| --- | --- | --- | --- |
| 1 | Gemini-3-Flash | 39.6 | 通用 MLLM |
| 2 | Gemini-3.1-Pro | 37.4 | 通用 MLLM |
| 3 | Qwen-3-VL-32B | 36.8 | 通用 MLLM |

最强 agentic 框架是 AVP 34.0，其次 WorldMM 30.6、Ego-R1 25.8、SiLVR 22.4——**四个 agentic 框架
全部落后于前三名通用 MLLM**。GPT-5 只有 27.8。论文结论：证据跨度越长性能衰减越明显。

工业榜：无。

**MindBridge 目标**：> 39.6，同时成为第一个超过通用 MLLM 的 agentic 记忆系统。这是九项里"记忆
系统 vs 大模型"叙事最干净的一项：题目本身要求跨周回溯，而榜单显示现有记忆框架反而更差。

### 3.10 ATM-Bench — 长期个性化指代记忆问答

官方发布 3,759 张图片、533 段视频、6,742 封邮件组成的个人档案，`main` split 1,013 题、`hard` split
31 题两者不相交（`hard` 不是 `main` 的子集）——但两个 split 的证据不是同样不相交：6,742 封邮件里，
430 次证据引用只对应 362 封不同邮件，占全档案 5.4%：`main` 引用 354 封、`hard` 引用 13 封，其中 5
封两个 split 都引用过，其余全是照样要摄入的干扰量。以下学术榜答题
模型为 `Qwen3-VL-8B-Instruct-FP8`，judge 为 `gpt-5-mini`（[官方论文](https://arxiv.org/abs/2603.01990)）：

| System | ATM-Bench QS | ATM-Bench Recall@10 | ATM-Bench-Hard QS | Hard Recall@10 |
| --- | --- | --- | --- | --- |
| Memexa (DeepSeek-V4-flash judge, not gpt-5-mini) | 68.0 | 79.1 | 47.9 | 44.7 |
| MemPalace | 56.8 | 76.4 | 9.7 | 28.3 |
| ATM-RAG (paper's own) | 51.0 | 68.7 | 8.4 | 28.8 |
| MemoryOS | 47.2 | 59.2 | 13.7 | 32.7 |
| A-Mem | 44.8 | 66.4 | 9.9 | 31.7 |
| mem0 | 43.5 | 61.9 | 9.2 | 23.7 |
| HippoRAG2 | 42.9 | 66.4 | 9.4 | 31.9 |
| SimpleMem | 27.3 | 23.3 | 3.2 | 7.0 |

通用编程 agent 在 31 题的 hard split 上反而更高——GPT-5.6 Sol（medium）58.8%、Claude Opus 5
（xhigh）58.4%；同一 split 的 SGM Oracle 上限是 60.5（MiniMax-M3）。这两处对比都不能与上表直接
并列：Memexa 一行用 DeepSeek-V4-flash 当 judge，不是表头统一的 gpt-5-mini；通用编程 agent 与专门
的记忆系统也是两种不同的测法，不是同一维度的名次。

表中数字对应官方 leaderboard 的 **SGM**（schema-guided memory，把结构化图文摘要交给答题模型）
赛道，不是 **Raw**（把原始图像直接放进答题模型上下文）赛道——两条赛道官方分别发布，本文只转载
前者。这个区分决定了 MindBridge 自己该对齐哪一列：`--media-source raw` 与 ATM 的 Raw 赛道同名，
但语义不同——`raw` arm 的原始媒体先经过 MindBridge 自己的感知与结构化记忆写入，答题模型看到的是
检索回的证据而不是像素本身，因此 MindBridge 的 `raw` 与 `sgm` 两条 arm 都只能对齐 ATM 的 SGM
列，不能对齐 Raw 列。

工业榜：无——表中每一行都出自论文自己搭的同一套受控评测（同一 answerer、同一 judge），不是
某厂商自测自发的产品分数；mem0 等系统虽有商业身份，也是被论文拿去跑分，不是自己对外公布成绩，
按本文档 3.5 节 M3-Agent 的先例，这种情形仍计入学术榜。

**评测口径**：MindBridge 走自己的生产写路径（`raw` arm 经 `observe` 触发感知，`sgm` arm 经
`remember` 写入官方文本），不是把题面或图像直接塞进答题模型的上下文。引用本表数字作对照时必须
同时点名 MindBridge 一侧实际使用的答题模型与 judge，否则两个数字不可比；MindBridge 目前在
ATM-Bench 上没有已发布的分数。

### 3.11 Mem-Gallery — 多模态长期对话记忆

官方发布 20 个主题、240 段多会话对话、3,962 轮对话、1,003 张对话图片、1,711 道问题（其中 487
题带查询图片）。以下学术榜 backbone 为 `Qwen2.5-VL-7B`，共对比 13 个记忆系统
（[官方论文](https://arxiv.org/abs/2601.03515)）：

| System | F1 | LLM judge |
| --- | --- | --- |
| MuRAG (best multimodal) | 0.6966 | 0.8229 |
| UniversalRAG | 0.6827 | 0.8016 |
| A-Mem (best textual) | 0.6228 | 0.7431 |
| Full memory (text) | 0.3625 | — |
| Full memory (multimodal) | 0.3354 | — |

分任务上还有几个值得单独记录的高点：MuRAG 在 `VS` 任务上以 0.8818 F1 领先、在 `TTL` 任务上以
0.8177 F1 领先；FIFO 在 `AR` 任务上拿到 1.0000 F1。`AR` 考的是"信息不在对话里时应当拒答"，一个
敢自由弃答的系统能在这一项直接封顶，所以 `AR` 的分数必须与其余八个任务分开看，不能并入同一个
平均分。

工业榜：无——13 个系统全部是论文在同一套受控实验（同一 backbone、同一评测流程）里对比出来
的结果，不是任何厂商自测自发的产品分数。

**评测口径**：MindBridge 走自己的生产写路径（对话经 `remember` 写入、图片经 `observe` 触发感知，
问题经 `recall` 取证据后再作答），不是把对话原文或图片直接塞进答题模型的上下文。引用本表数字作
对照时必须同时点名 MindBridge 一侧实际使用的答题模型与 judge，否则两个数字不可比；MindBridge
目前在 Mem-Gallery 上没有已发布的分数。

## 4. 达成 SOTA 前必须补齐的工程项

### 4.1 已落地

**外部 scorer 绑定（LoCoMo-Refined / MM-Lifelong / EgoMemReason 共用）。** 这三个 benchmark 的
分数都由 MindBridge 之外的程序产出：LoCoMo-Refined 用 `mem-eval-suite/LoCoMo_refined`，
MM-Lifelong 用其发布的 scorer，EgoMemReason 用留出答案键的 leaderboard。run manifest 在 scorer
运行之前就写完了，结构上不可能装下结果，所以结果落在一个独立的 `*.score.json` sidecar 里：

```bash
uv run mindbridge-bench score \
  --predictions runs/locomo-refined/predictions.jsonl \
  --manifest runs/locomo-refined/predictions.jsonl.manifest.json \
  --scorer-output runs/locomo-refined/official-scorer-summary.json \
  --scorer-repository mem-eval-suite/LoCoMo_refined \
  --scorer-command "./scripts/run_eval.sh --metrics llm f1 bleu --llm-judge refined" \
  --judge-model Qwen/Qwen3-14B \
  --answer-backbone qwen3.8-max \
  --scored-question-count 1382 \
  --metric llm=82.65
```

sidecar 会重新计算 predictions 的 sha256，与 manifest 里的不一致就拒绝写入——因此一个公布出去的
分数必然绑定到真实跑出来的那份预测，而不是"据说"。judge 模型与 answer backbone 是必填语义位，
正是 LoCoMo 各家不可比的根因；`run_eval.sh` 同时接受 `refined` 与 `original` 两套 judge，用哪一套
必须写进 `--scorer-command`。

**LoCoMo-Refined 分类别题量。** manifest 保留 `category_question_counts`。adversarial 已被删除，
所以它不再区分两套协议，但 1,382 题里有 802 题是 category 4，子集跑法很容易把题型分布跑歪而不
自知。

**EgoLifeQA 五类指标。** `EgoLifeMetrics` 新增 `categories`，与论文表格逐列对齐；预测行新增
`subject_id`，标明证据来自哪位佩戴者的流。

**Video-MME long 子集与字幕披露。** CLI 新增 `--duration`（可重复，scope 到 short/medium/long）与
**必填**的 `--transcript-source {none,asr,official_subtitles}`；`VideoMMEMetrics.by_duration` 一次
跑完即给出三档单元格。声明 `none` 却在 prepared media 里带 transcript 会被直接拒绝——这正是
"在无字幕设定下偷用字幕轨"这条禁令的执行点，反过来声明 `asr` 却没有任何 transcript 也会被拒。

### 4.2 仍是操作项，不需要改代码

**MemLens 四档全跑。** 适配器与 CLI 已完整支持（`load_memlens_agent_subset`、
`--agent-subset-index`），32K/64K/128K/256K 是四次独立调用、四份独立上报，没有可聚合的合并口径。

**EgoMemReason 提交 leaderboard。** `egomem_cli` 输出的 `[{example_id, predicted_answer}]` 与官方
要求的提交格式完全一致，直接上传即可；回收到的准确率用上面的 sidecar 记录。注意官方 2026-07-14 的
`e406eb1` 做过一次选项字母重排，README 钉的 `7e58150` 是其之后的 head，已包含重排。

## 5. 反 reward hacking 约束

[技术架构文档](technical-architecture.md) 第 2.2 节第 10 条已经规定：Benchmark 服务于 Harness，
分数必须来自 MindBridge 的整体能力，不得来自题面识别、judge 迎合或数据集专用旁路。在本文列出的
数字面前，这条约束有几个具体落点：

- 原始 LoCoMo 的 judge 会接受"找对会话但零细节"的答案，**刻意生成模糊回答**能提分。
  LoCoMo-Refined 的 judge 反过来同时惩罚遗漏和无依据的补充，所以模糊化和堆细节两个方向都会掉分
  ——但两者仍然都属于迎合 judge，禁止。
- SuperMemory-VQA 仍有拒答选项。**按题型统计规律调节拒答率**能提分，禁止；拒答必须由"证据不足"
  这一条真实判定触发。LoCoMo-Refined 没有 adversarial 类，空回答就是一次纯粹的失分，manifest 的
  `unanswered_question_count` 就是用来看这件事的。
- EgoLifeQA / EgoMemReason 是四选一。**利用选项文本长度或分布先验**属于题面识别，禁止。
- Video-MME 的字幕设定。**在声称无字幕的设定下偷用字幕轨**，禁止。

任何一项 SOTA 声明都必须附带：官方数据来源、工件 hash、judge 模型与版本、backbone 模型、上下文
档位，以及可复现的 manifest（沿用 `benchmarks/manifests/` 的既有格式）。
