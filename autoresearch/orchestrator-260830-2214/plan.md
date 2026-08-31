# MindBridge autoresearch configuration

- Archetype: `optimize-metric`
- Mode: bounded orchestration loop
- Direction: `lower_is_better`
- Iterations: 30 inner iterations, 50 orchestration cycles maximum
- Terminal choice: `stop-at-verified`
- Verify: `uv run --frozen python autoresearch/orchestrator-260830-2214/verify.py --suite dev`
- Expected output: `0`
- Guard: `autoresearch/orchestrator-260830-2214/guard.sh`
- Frozen product candidate: `e806cf3b1dd49a96fc9c477847b855f42c68c55b`
- Frozen baseline: `ba4bcced90b916bf28265576320639b8c1a0218a`
- Locked namespaces: `baseline-ba4bcced90b9-locked-v2` and
  `current-e806cf3b1dd4-locked-v3`
- Offline identity registry SHA-256:
  `481d3c40d3ca3a060fc7cfbab72ab4abaa1f5819d05132a5c540ebee3ad54ccc`
- Provenance wrapper SHA-256:
  `370179d9e12a21f1ed37e7bc8a7bb60e1c6137d02bde86f09ff243a08e19be32`

The predicate is lexicographic. Incomplete or invalid runs and quality regressions block latency
and token improvements. Provider credentials are process-only and must not be written here.

## Locked quality and cost suite

The final `dev` manifest name is retained for the skill's fixed stop command, but it points only to
the following locked samples. Membership was fixed before either candidate or baseline locked
score was inspected:

| Task | Locked selection | Expected questions |
| --- | --- | ---: |
| LoCoMo-Refined | `--offset 3 --limit 3` | 455 |
| ATM-Bench | official `atm-bench-hard` raw-media split | 31 |
| Mem-Gallery | `--offset 5 --limit 5` | 407 |
| M3-Bench Robot | loader-sorted offsets `5,14,20,45,67,73,85` (`bedroom_06`, `gym_03`, `kitchen_05`, `living_room_07`, `meeting_room_05`, `office_05`, `study_09`) | 87 |
| EgoLifeQA | `--offset 50 --limit 50` | 50 |

M3 and Mem-Gallery are unit-disjoint from their fast development slices, and ATM hard is disjoint
from ATM main. Earlier exploration evaluated all LoCoMo conversations, so its locked selection is
paired regression evidence rather than an untouched holdout. EgoLifeQA exposes one physical memory
unit only: questions 50–99 are sample-disjoint but not an independent-unit holdout. This suite is
internal paired evidence, not an official leaderboard or broad SOTA claim. Every run uses
the public SDK, seed `20260830`, no response cache, a fresh physical data directory, and logged
samples. The verifier pins dataset/evaluation/input identities and ordered sample IDs, recomputes
scores from samples, and checks run provenance against the revisions above. Every formal attempt
uses the committed wrapper on GPU `GPU-6fa4d834-e033-e492-66f5-9d2f3792c4dd`; a pre-spawn ledger
makes a failed or interrupted output path non-reusable. Each quality task is a two-entry
baseline/current provenance chain.

The first EgoLifeQA baseline attempt used automatic batches up to 32 and exhausted host memory
before a provenance file could be finalized. Its attempt ledger, log, and partial store are retained
outside the formal namespace. The sole retry keeps the locked samples, seed, models, and prompts
unchanged, while both paired sides use batch size 1 and request/judge concurrency 4 to bound peak
memory. The failed partial output is never admitted to a manifest.

Quality requires no task to lose more than 0.5 percentage points and at least +1 percentage point
macro gain. Cost requires no task to use over 2% more generation tokens and at least 5% geometric
mean reduction.

## Locked performance suite

Single cloud runs are not accepted as speed evidence. In particular, the earlier ATM baseline had
one 236-second provider outlier and is excluded from any acceleration claim. After quality is
frozen, run three fresh, uncached, `--predict-only` baseline/candidate pairs in AB/BA/AB order on a
fixed five-route, 95-question sentinel. It is materialized from release order without inspecting
answers or predictions:

- the first 20 released questions of LoCoMo conversation `conv-42`;
- ATM hard (31 questions);
- the first 20 released questions of Mem-Gallery topic
  `Dog_Behavior_Research_Academic_Life`;
- M3 Robot `bedroom_06` (14 questions);
- EgoLifeQA `--offset 50 --limit 10`.

Use one request and one unit at a time so per-question latency has no omitted semaphore queue, with
identical hardware, batch sizes, and model configuration. Judge latency is absent.
For each task, compare the median of the three paired ratios. The task-level sample p95 may not
regress by more than 5%; run-level mean `mindbridge.ask` and generation TTFT must also remain within
5%. At least one of the geometric mean ask or TTFT ratios must improve by 5%. The report must retain
the limitation that a shared cloud provider prevents causal attribution of small timing changes.
Each task has a six-entry `baseline-r1,current-r1,current-r2,baseline-r2,baseline-r3,current-r3`
provenance chain, implementing the pre-registered AB/BA/AB schedule without discardable retries.
