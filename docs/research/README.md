# Research archive

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
| [Next experiment for raw-video memory](video-memory-next-experiment-2026-09-08.md) | Dated analysis or results; retain the stated evidence limits |

## Design records

Approval of a specification describes a past implementation decision. The current SDK and
configuration references own the shipped signatures, defaults, and transport coverage.

| Record | Kind |
| --- | --- |
| [Context OS competitor brief: fast capture, memory control plane, context compiler](../superpowers/research/2026-09-03-context-os-competitor-brief.md) | Dated source review |
| [Context OS round 1: fast capture, memory control plane, context compiler](../superpowers/specs/2026-09-03-context-os-round-1-design.md) | Historical specification or acceptance scenario |
| [Round 2 acceptance scenario: the household companion](../superpowers/specs/2026-09-03-household-companion-scenario.md) | Historical specification or acceptance scenario |
| [Context OS round 2: identity as governed knowledge](../superpowers/specs/2026-09-03-identity-governance-design.md) | Historical specification or acceptance scenario |
| [Recall programs, identity-anchored keys, and completeness-aware grounding](../superpowers/specs/2026-09-10-recall-programs-design.md) | Historical specification or acceptance scenario |

## Current implementation and experiment boundaries

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
