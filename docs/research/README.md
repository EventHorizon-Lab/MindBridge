# Research archive

- [Grounded recall quality evidence](2026-10-29-grounded-recall-quality.md) — evidence and limits
  for the promoted retrieval, media, answer-style, and evaluation changes.

These reports and specifications preserve dated evidence, protocols, and design decisions.
They are not current API instructions or a fresh competitor survey. “Current”, paths, commands,
schemas, and model names inside a record refer to its own frozen snapshot. Use the
[documentation index](../README.md) for this checkout's product behavior.

A result is usable only within its stated code, dataset, model, judge, hardware, and sample limits.
Local `.benchmarks/`, `autoresearch/`, and worktree artifacts cited by a report may not ship with
the repository. An archived command may require those artifacts or an older checkout.

## Reports and protocols

| Record | Scope |
| --- | --- |
| [Mem-Gallery 评估一致性与失分归因 — 2026-10-27](2026-10-27-mem-gallery-evaluation-audit.md) | Complete fixed-answer parity against upstream, scoring corrections, protocol mismatches, causal attribution limits, and reward-hacking boundaries |
| [ATM raw main 与 Mem-Gallery：最佳配置与能力突破方向 — 2026-10-29](2026-10-29-atm-gallery-baseline.md) | Task-specific parameters, verified full-run limits, memory formation failures, and genuine grounding/perception improvements |
| [FunASR main adoption — 2026-10-28](2026-10-28-funasr-main.md) | Pinned upstream runtime, speaker-overlap regression, thread/placement controls, timestamp compatibility and model-check boundaries |
| [ATM raw main 与 Mem-Gallery：能力修复与真实模型运行 — 2026-10-28](2026-10-28-atm-gallery-capability-run.md) | Bounded multimodal retrieval, cross-round evidence retention, reader controls, verified media transport and ongoing full runs |
| [Eval concurrency and resource reuse — 2026-10-27](2026-10-27-eval-concurrency.md) | Dynamic model admission, idle connection reclamation, child-worker fairness, cancellation, and deterministic paired SDK evaluation |
| [Local runtime performance — 2026-10-27](2026-10-27-local-runtime-performance.md) | FP32 buffer scoring, scoped SQLite hydration, exact-result checks, and frozen SDK latency comparisons |
| [否定敏感检索的论文与开源方案评估 — 2026-10-27](2026-10-27-negation-retrieval-research.md) | Cross-encoder evidence, Chinese subject/scope extraction, exclusion-query methods, implementation status, and MindBridge evaluation priorities |
| [质量损失分层审查 — 2026-10-27](2026-10-27-quality-diagnostics.md) | Five observed question boundaries, trace correlation, source coverage limits and stratified human annotation |
| [Top1 ambiguity — 2026-10-27](2026-10-27-top1-ambiguity.md) | R03 reproduction, removal of the lexical route exemption, three-policy risk–coverage probes and correct-candidate losses |
| [Retrieval critical-section review — 2026-10-27](2026-10-27-retrieval-critical-section.md) | P02 invariants, WAL snapshots, asset lease handoff, forced deletion interleavings, and paired lock-time measurements |
| [Bounded temporal lexical retrieval — 2026-10-27](2026-10-27-temporal-lexical-retrieval.md) | Corpus-independent route depth, native time-filtered FTS compatibility, and paired synthetic discovery/cost evidence |
| [STATE 修订的依据强度审查 — 2026-10-27](2026-10-27-state-revision-evidence.md) | Reproduced weak-state supersession, evidence-aware revision contract, deterministic fixtures, and real-model evaluation limits |
| [Benchmark answer policy and budgets — 2026-10-27](2026-10-27-benchmark-answer-policy.md) | E01 policy confound, aligned readers, refusal accounting, budget/cache corrections, and validation limits |
| [Controlled semantic recall — 2026-10-27](2026-10-27-semantic-recall.md) | R01 root cause, bounded opt-in query execution, native-index loss decomposition and limits on quality claims |
| [Evidence selection and provider input boundaries — 2026-10-27](2026-10-27-answer-input-boundaries.md) | R04 review, retained soft evidence packing, complete message-text byte accounting, provider context failure contract, and validation limits |
| [Formation semantic support audit — 2026-10-27](2026-10-27-semantic-support-audit.md) | Declared provenance versus semantic support, SDK counterexample, recent research, offline metrics and admission-policy promotion criteria |
| [Negation and lexical evidence — 2026-10-27](2026-10-27-chinese-negation.md) | R02 root cause, pure IDF coverage, paired public-SDK fixtures, and disclosed ranking tradeoffs |
| [Accuracy-first retrieval — 2026-10-26](2026-10-26-accuracy-first-retrieval.md) | Public SDK retrieval quality, quantization score calibration, candidate breadth and numerical scoring measurements |
| [MindBridge 在 ATM-Bench 与 Mem-Gallery 上的竞争策略 — 2026-10-26](2026-10-26-atm-mem-gallery-sota-strategy.md) | Current-source review, code audit, protocol differences, and proposed discriminating experiments; no new model results |
| [Context OS competitor brief — 2026-09-03](2026-09-03-context-os-competitor-brief.md) | Dated source review for fast capture, the memory control plane, and context compilation |
| [Zvec 0.8 adoption — 2026-10-26](2026-10-26-zvec-08.md) | Dependency and maintenance measurements, cosine rotation regression, multilingual coverage, and native API limits |
| [Compact context presentation prototype](compact-context-prototype.md) | Frozen prototype and conv30 measurement; current contract lives in the SDK reference |
| [Memory backend evaluation — 2026-09-08](2026-09-08-memory-backend-evaluation.md) | Dated analysis or results; retain the stated evidence limits |
| [Memory backend research for MindBridge](2026-09-08-memory-backend-research.md) | Dated analysis or results; retain the stated evidence limits |
| [Memory backend validation](2026-09-08-memory-backend-validation.md) | Dated analysis or results; retain the stated evidence limits |
| [MindBridge Memory Backend 增量研究：证据闭包、竞争事实与情感来源](2026-09-09-companion-evidence-obligations-zh.md) | Dated analysis or results; retain the stated evidence limits |
| [Evidence excerpt experiment](2026-09-09-evidence-excerpt-experiment.md) | Dated analysis or results; retain the stated evidence limits |
| [记忆后端本轮结论 — 2026-09-09](2026-09-09-memory-backend-conclusions-zh.md) | Dated analysis or results; retain the stated evidence limits |
| [Recall programs round — 2026-09-10/11](2026-09-10-recall-programs-round.md) | Dated analysis or results; retain the stated evidence limits |
| [Baseline-2 decomposition — 2026-09-11](2026-09-11-baseline-2-decomposition.md) | Dated analysis or results; retain the stated evidence limits |
| [Baseline loss decomposition — 2026-09-12](2026-09-12-baseline-loss-decomposition.md) | Dated analysis or results; retain the stated evidence limits |
| [原始证据约束的巩固实验协议](2026-09-12-corroboration-protocol.md) | Experiment protocol; not an implementation guarantee |
| [MindBridge 的记忆机制：从证据保存到可撤销的长期学习](2026-09-12-memory-mechanisms.md) | Dated analysis or results; retain the stated evidence limits |
| [第五轮协议：历史身份恢复与显式证明交付](2026-09-13-identity-proof-protocol.md) | Experiment protocol; not an implementation guarantee |
| [形成前历史身份见证：真实模型实验](2026-09-13-identity-witness-results.md) | Dated analysis or results; retain the stated evidence limits |
| [第四轮协议：真实形成与实际 Context 交付](2026-09-13-live-context-protocol.md) | Experiment protocol; not an implementation guarantee |
| [第四轮：真实形成与 Context 交付的端到端验证](2026-09-13-live-context-results.md) | Dated analysis or results; retain the stated evidence limits |
| [第二轮：记忆容量、来源依赖与形成连续性](2026-09-13-memory-capacity-protocol.md) | Experiment protocol; not an implementation guarantee |
| [第二轮记忆实验：容量边界、来源申报与形成连续性](2026-09-13-memory-capacity-results.md) | Dated analysis or results; retain the stated evidence limits |
| [选定证明交付：显式协议实验](2026-09-13-proof-envelope-results.md) | Dated analysis or results; retain the stated evidence limits |
| [第三轮协议：按义务搜索完整证据](2026-09-13-witness-search-protocol.md) | Experiment protocol; not an implementation guarantee |
| [第三轮：按义务搜索证明 DAG](2026-09-13-witness-search-results.md) | Dated analysis or results; retain the stated evidence limits |
| [决定性端到端实验：冻结协议](2026-09-14-end-to-end-protocol.md) | Experiment protocol; not an implementation guarantee |
| [决定性端到端验证：当前组合 NO_GO](2026-09-14-end-to-end-results.md) | Dated analysis or results; retain the stated evidence limits |
| [Round r0914 pre-registered protocol (verbatim copy)](2026-09-14-memory-dynamics-protocol.md) | Frozen protocol; preserve verbatim content and its recorded digest |
| [Memory dynamics round r0914 — 2026-09-14](2026-09-14-memory-dynamics-round.md) | Dated analysis or results; retain the stated evidence limits |
| [Competitor memory backends: evidence and transfer decisions](competitor-memory-backends-2026-09-08.md) | Dated analysis or results; retain the stated evidence limits |
| [Historical out-of-scope EgoLife diagnostics](historical-egolife-diagnostics-2026-09-08.md) | Dated analysis or results; retain the stated evidence limits |
| [Completing hybrid retrieval scores from durable memory](hybrid-score-completion-2026-09-08.md) | Dated analysis or results; retain the stated evidence limits |
| [MindBridge memory backend 深度审计与研究路线](memory-backend-audit-2026-09-07.md) | Dated analysis or results; retain the stated evidence limits |
| [Memory competitor source ledger (2026-09-07)](memory-backend-sources-2026-09-07.md) | Dated analysis or results; retain the stated evidence limits |
| [Memory optimization: design and validation record](memory-optimization-2026-09-07.md) | Dated analysis or results; retain the stated evidence limits |
| [Timeline neighbor-window risk review](timeline-neighbor-window-risk-review-2026-09-08.md) | Dated analysis or results; retain the stated evidence limits |

## Current implementation and experiment boundaries

- [Architecture](../architecture.md) owns the current kernel and storage boundaries;
  [operations](../operations.md) owns index recovery and store compatibility requirements.
- [EmbeddingGemma 2 through vLLM](../configuration.md#embeddinggemma-2-through-vllm) and its
  [deployment guide](../deployment.md#embeddinggemma-2-model-server) own the remote model recipe.
- [Independent evidence](../memory-types-time-and-decay.md#experimental-independent-evidence)
  remains opt-in and store-pinned. Structural success is not proof of improved natural recall.
- [Exact excerpts](../context-compilation.md#exact-raw-excerpts) are an explicit compilation option;
  a partial source does not satisfy a full-record evidence dependency.
- [Compact presentation](../api/python-sdk.md#memory-operations) is an opt-in Python formatter. Its
  frozen measurement does not establish downstream quality for other models or prompts.
- [Recall policies](../configuration.md#local-memory-settings) document the shipped planning and
  expansion settings; historical proposals may describe different budgets.
- [Benchmarking](../benchmarking.md) owns the current task catalog and scorer protocols. Historical
  ICM scores require the documented protocol correction before comparison.

The `2026-09-14-memory-dynamics-protocol.md` file is a verbatim copy with a recorded SHA-256.
Its body is intentionally preserved. Third-party scorer notices are also preserved in
[NOTICE.md](../../src/mindbridge/benchmarks/_official/NOTICE.md).

For a new experiment, publish a new dated record with reproducibility pins. Do not change an old
score, acceptance decision, or evidence limit to make it look like a measurement of newer code.
