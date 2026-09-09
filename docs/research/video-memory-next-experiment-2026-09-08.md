# Next experiment for raw-video memory

Status: historical EgoLife exact-cosine and representative-selection diagnostics completed before
the project owner removed EgoLifeQA from product acceptance; temporal closure is a proposal; no new
product architecture is implemented.

## Branch status

- **Candidate generation:** the exposed full-corpus exact-cosine diagnostic is complete. It finds
  headroom over the bounded public pool and also shows exact ranking alone is insufficient.
- **Evidence selection:** the fixed budget-eight facility-location diagnostic is complete and fails
  its predeclared coverage gate, so it is rejected for product integration.
- **Semantic formation:** the five-clip public-SDK visual-description feasibility pilot and the
  separately authorized 20-store formation-cost run are complete. The latter retains all 1,433
  fixed records, including 26 records for which optional visual description produced no usable
  caption. The paired five-cell captioned-corpus QA is also complete; it does not establish a
  caption-formation or score-completion benefit on M3.
- **Temporal closure:** the `TimelineSpan` contract remains a design risk review. The previously
  drafted EgoLife same-sequence radius experiment was withdrawn with the dataset scope change and
  was not implemented.

## Evidence boundary

EgoLifeQA is no longer a product acceptance or optimization target. The fixed results below are
retained only as historical failure analysis and do not authorize more EgoLife calls or a product
change. The active video validation target is M3-Bench Robot.

The score-completion change succeeds on one static textual ATM corpus but does not demonstrate
generalization in the fixed causal EgoLife characterization: the observed mean moves from 0.200 to
0.160 under controls that do not support causal attribution. This excludes claims of broad SOTA and
points to a video-specific diagnosis before another product change.

Three bottlenecks must be separated. First, a target clip can be semantically absent from the
candidate pool because the persisted observations or their embeddings do not express the queried
event. Completing a score cannot create that observation. Second, a relevant anchor can imply
nearby evidence in the same recording sequence, but temporal proximity alone is neither semantic
relevance nor causality. Third, even correct candidates can be displaced by the eight-video answer
budget or fail in the reader. In the exposed EgoLife 40-question diagnostic, 11 of 29 failures have
no target slice in the recorded pool, while nine failures still expose all target slices under the
eight-video cap. A free radius-two neighbor oracle raises only a target-coverage ceiling and does
not account for displaced evidence or reader errors.

This is concrete in a narrow raw-store audit: the first three source-order video records expose only
their benchmark source-ID marker to lexical search and have no transcript or visual description.
The audit does not connect that fact to any question outcome, but it makes semantic observation
quality an independent variable rather than a synonym for retrieval depth. MindBridge already
offers optional ingest-time visual description; the matched run disables it. A later formation arm
would need to caption each clip once, freeze those nondeterministic outputs in an attested corpus,
and charge its model calls separately before either retrieval implementation sees the corpus.

## Architecture hypothesis

Test a provenance-grounded multiscale evidence planner. SQLite remains authoritative and Zvec
remains derived. Each experimental embedding part must be traceable to its parent source, part
kind, source offset, producer-unique sequence, and relative media span. A declared relative query
cutoff, actual calendar occurrence interval, and `known_at` visibility remain independent filters.
No cross-sequence neighbor is eligible, and no causal-prefix claim is made when the query declares
no relative cutoff.

The planner begins with current hybrid anchors. A measurable route-disagreement or score-margin
signal may trigger bounded expansion within the same sequence and eligible time range. It then
selects at most eight sources under the existing evidence budget, trading off query relevance,
query-part coverage, redundancy, and chronological continuity while retaining source lineage.
Coverage, hierarchy, representative selection, and uncertainty-gated expansion all have prior
art; the hypothesis specific to MindBridge is that their low-cost reconstruction from persisted
multimodal vectors can work with explicit causal and version constraints, without an LLM formation
graph or new service.

## Minimal falsification experiment

An offline exact-cosine scan has now measured one component. For each question it filters the 6,264
raw parents by that question's cutoff, rather than treating all 6,264 as eligible. Complete temporal
coverage is:

| Fixed condition | Top 8 | Top 36 | Top 100 |
| --- | ---: | ---: | ---: |
| Public ranking/pool | 11/100 | 19/100 | Not measured |
| Parent MaxSim over all parts | 13/100 | 25/100 | 34/100 |
| Aggregate part 0 | 13/100 | 31/100 | 47/100 |
| Video-only part 2 | 15/100 | 31/100 | 45/100 |

No single row is selected after observing the result. Even at 100, aggregate and video-only scoring
leave 54 and 55 of 114 target slices below the cutoff, respectively. This exposed temporal oracle
shows bounded candidate-generation headroom and also shows that exact ranking alone is insufficient;
it does not establish semantic sufficiency or answer quality. The aggregate is
`.benchmarks/research/2026-09-08-evaluation-audit/exact-cosine-visibility-v1/result/aggregate.json`
(SHA-256 `6e0bfe72eb0be439b414bc50773e80f6c627a998c493bfa1b94f4e524e1fa269`).

The earlier draft proposed a radius-one/radius-two closure factorial on this EgoLife roster. That
experiment is cancelled because EgoLifeQA is no longer a product target; it is neither an active
plan nor authorization for more calls. Its falsification logic remains useful for any future task
whose source truth and product relevance are established in advance: failure of exact scoring to
surface targets would weaken a bounded-ANN remedy while remaining consistent with a representation
gap, and any temporal expansion would have to improve coverage before the fixed media cap without
crossing sequence or cutoff boundaries. No such temporal-closure result is claimed here. The fixed
facility-location selector has already failed its historical exposed coverage gate and is rejected.
