"""Task metrics: accuracy, retrieval quality, controls, breakdowns, and baseline comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import (
    Mapping,
    Sequence,
)
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    pass
from mindbridge import MemoryConfig, MindBridgeConfig
from mindbridge.benchmarks.eval_adapters import LoadedTask
from mindbridge.benchmarks.eval_arms import (
    BLIND_PROMPT_VERSION,
    _Arm,
)
from mindbridge.benchmarks.eval_config import (
    _RESULTS_FILE,
    _SAMPLES_FILE,
    DEFAULT_ARM,
    RETRIEVAL_CANDIDATE_LIMIT,
    _Arguments,
)
from mindbridge.benchmarks.eval_results import (
    EVAL_SCHEMA_VERSION,
    SampleResult,
)
from mindbridge.benchmarks.eval_statistics import (
    ScoredValue,
    paired_comparison,
    percentile,
    summarize,
)
from mindbridge.benchmarks.official_scorers import (
    GOLD_SOURCE_KEYS,
    gold_source_groups,
    judge_model_is_official,
    metric_is_official,
    official_judge_model,
    scorer_protocol,
    task_family,
    task_primary_metric,
)

# Measured on this harness: a per-benchmark difference under three points is inside the run to
# run noise band, whose per-question standard deviation is about seventeen points.
NOISE_FLOOR = 0.03


_UNRESOLVED_EVIDENCE_KEY = "unresolved_evidence_ids"


_RECALL_CUTOFFS = (1, 5, 10, 20)


_MANDATORY_CONTROLS = ("random_ranker", "blind", "recall_at_20")


# The question-metadata fields each benchmark family groups its per-question
# scores by, keyed by family so that one catalog task cannot drift from its
# siblings. A family absent here reports no breakdown at all, which is
# invisible in a results document, so `tests/unit/benchmarks/test_eval.py`
# pins this table against the metadata the adapters actually emit.
_BREAKDOWN_FIELDS: Mapping[str, tuple[str, ...]] = {
    "locomo-refined": ("category",),
    "m3-bench": ("question_types",),
    "video-mme-v2": ("group_type", "level", "second_head", "third_head"),
    "worldmemarena": ("question_type", "question_type_abbrev", "difficulty"),
    "egotempo": ("question_type",),
    "memlens": ("question_type", "question_subtype"),
    "mm-lifelong": ("question_type",),
    "supermemory-vqa": ("skill",),
    "atm-bench": ("qtype",),
    "mem-gallery": ("point",),
    "longmemeval": ("question_type",),
    "es-memeval": ("capability",),
    "clbench": ("context_category", "sub_category"),
    "beam": ("category", "difficulty"),
    "personamem-v3": ("task_family", "task_type"),
    "openeqa": ("category",),
}


def _answer_retrieval_candidate_limit(
    recall_limit: int,
    memory_config: MindBridgeConfig | None,
) -> int:
    """Mirror the ranked window ``Memory.ask`` requests from the retrieval kernel."""
    return (
        RETRIEVAL_CANDIDATE_LIMIT
        if _evidence_budget_chars(memory_config) is not None
        else min(RETRIEVAL_CANDIDATE_LIMIT, recall_limit * 3)
    )


def _metrics(
    task: LoadedTask,
    samples: Sequence[SampleResult],
    arguments: _Arguments,
    blind: Mapping[str, object] | None = None,
    *,
    arm: str = DEFAULT_ARM,
    retrieval_candidate_limit: int | None = None,
) -> dict[str, object]:
    seed = _task_seed(arguments.seed, task.spec.name)

    def official(metric_name: str, *, uses_judge: bool = False) -> bool:
        # A baseline arm answers outside the pinned protocol, so none of its numbers are
        # upstream-comparable however faithful the scorer was.
        return arm == DEFAULT_ARM and metric_is_official(
            task.spec.name, metric_name, judge_model, uses_judge=uses_judge
        )

    primary_name = task_primary_metric(task.spec.name)
    scored = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
        for sample in samples
        if sample.score is not None
    )
    primary = summarize(
        scored,
        seed=seed,
        bootstrap_samples=arguments.bootstrap_samples,
    )
    metric_rows: dict[str, dict[str, object]] = {}
    metric_names = sorted({name for sample in samples for name in sample.metrics})
    judge_models = sorted(
        {sample.judge_model for sample in samples if sample.judge_model is not None}
    )
    judge_model = judge_models[0] if len(judge_models) == 1 else ""
    for metric_name in metric_names:
        metric_values = tuple(
            ScoredValue(sample.sample_id, sample.unit_id, sample.metrics[metric_name])
            for sample in samples
            if metric_name in sample.metrics
        )
        uses_judge = any(
            sample.judge_model is not None and metric_name in sample.metrics for sample in samples
        )
        clamp = (
            (0.0, 5.0)
            if metric_name == "judge_score_0_5"
            else (0.0, 2.0)
            if metric_name == "judge_score_0_2"
            else (1.0, 5.0)
            if metric_name == "llm_match_score_1_5"
            else (0.0, 100.0)
            if metric_name == "llm_match"
            else (0.0, 1.0)
        )
        metric_rows[metric_name] = {
            "official_metric": official(metric_name, uses_judge=uses_judge),
            **summarize(
                metric_values,
                seed=_task_seed(seed, metric_name),
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=clamp,
            ),
        }
    latencies = sorted(sample.latency_ms for sample in samples if sample.latency_ms > 0)
    retrieval = (
        _retrieval_quality(
            samples,
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
            recall_limit=arguments.recall_limit,
            retrieval_candidate_limit=retrieval_candidate_limit,
        )
        if _Arm(arm).retrieves
        else {"unavailable_reason": f"the {arm} arm does not run ranked retrieval"}
    )
    error_count = sum(sample.error_code is not None for sample in samples)
    retrieval_diagnostic_error_count = sum(
        sample.retrieval_diagnostic_error is not None for sample in samples
    )
    ingest_failure_count = sum(
        max(sample.ingest_failure_count for sample in samples if sample.unit_id == unit_id)
        for unit_id in {sample.unit_id for sample in samples}
    )
    unavailable_units = dict(getattr(task, "unavailable_units", {}))
    evaluated_units = getattr(task, "units", ())
    evaluated_unit_count = (
        len(evaluated_units) if evaluated_units else len({sample.unit_id for sample in samples})
    )
    full_dataset_selected = getattr(arguments, "offset", 0) == 0 and getattr(
        arguments, "limit", None
    ) in (None, -1)
    result: dict[str, object] = {
        "arm": arm,
        "primary_metric": primary_name,
        "official_metric": bool(
            primary_name in metric_rows and metric_rows[primary_name]["official_metric"]
        ),
        "scorer_protocol": scorer_protocol(task.spec.name),
        "official_judge_model": official_judge_model(task.spec.name),
        "judge_model": judge_models[0] if len(judge_models) == 1 else judge_models or None,
        "judge_model_official": (
            None
            if not judge_models or official_judge_model(task.spec.name) is None
            else len(judge_models) == 1 and judge_model_is_official(task.spec.name, judge_models[0])
        ),
        "score": primary,
        # An answer or judge failure scores zero and stays in the mean; only a store that
        # lost writes, a retrieval diagnostic that failed, or a unit the dataset could not
        # supply makes the score something other than the system's result on the task.
        "score_valid": (
            ingest_failure_count == 0
            and retrieval_diagnostic_error_count == 0
            and not unavailable_units
        ),
        "dataset_coverage": {
            "complete": not unavailable_units,
            "evaluated_unit_count": evaluated_unit_count,
            "unavailable_unit_count": len(unavailable_units),
            "unavailable_units": unavailable_units,
            "score_comparable_to_full_dataset": (full_dataset_selected and not unavailable_units),
        },
        "metrics": metric_rows,
        "exact_match": metric_rows.get("exact_match"),
        "question_count": len(samples),
        "scored_question_count": len(scored),
        "error_count": error_count,
        "retrieval_diagnostic_error_count": retrieval_diagnostic_error_count,
        "ingest_failure_count": ingest_failure_count,
        "abstentions": _abstentions(samples),
        "answer_latency_ms": {
            "measures": (
                "memory.ask wall clock per question, timed after concurrency admission so it "
                "is response latency and not queue depth"
            ),
            "count": len(latencies),
            "p50": percentile(latencies, 0.50, presorted=True),
            "p95": percentile(latencies, 0.95, presorted=True),
            "p99": percentile(latencies, 0.99, presorted=True),
        },
        "retrieval": retrieval,
        "controls": _controls(
            task.spec.name,
            retrieval,
            blind,
            is_blind_run=arguments.blind,
            retrieves=_Arm(arm).retrieves,
        ),
        "noise_floor": _noise_floor(scored, primary),
        "cross_harness_comparable": False,
        "comparability_note": (
            "scores are comparable only against runs of this harness at the same runner "
            "version, dataset revision, and scorer protocol. LoCoMo has ranged from 28.0 to "
            "92.5 across harnesses on identical data"
        ),
        "breakdowns": _metric_breakdowns(task, samples, arguments),
    }
    if task.spec.name == "video-mme-v2" and scored:
        rating = _video_mme_v2_rating(
            samples,
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
        )
        accuracy = _video_mme_v2_accuracy(
            samples,
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
            official_metric=official("accuracy"),
        )
        result.update(
            {
                "primary_metric": "rating",
                "official_metric": official("rating"),
                "score": rating,
                "accuracy": accuracy,
            }
        )
        metric_rows["rating"] = {"official_metric": official("rating"), **rating}
        metric_rows["accuracy"] = {
            "official_metric": official("accuracy"),
            **cast(Mapping[str, object], accuracy["overall"]),
        }
    if task.spec.name == "personamem-v3":
        from mindbridge.benchmarks._official.personamem_v3_scoring import (
            COMPLETE_HEADLINE_COVERAGE,
        )

        scored_rows = tuple(sample for sample in samples if sample.score is not None)
        score_coverage_complete = COMPLETE_HEADLINE_COVERAGE and len(scored_rows) == len(samples)
        supported_subset_accuracy = summarize(
            tuple(
                ScoredValue(sample.sample_id, sample.unit_id, 100.0 * cast(float, sample.score))
                for sample in scored_rows
            ),
            seed=seed,
            bootstrap_samples=arguments.bootstrap_samples,
            clamp=(0.0, 100.0),
        )
        accuracy = (
            supported_subset_accuracy
            if score_coverage_complete
            else summarize(
                (),
                seed=seed,
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=(0.0, 100.0),
            )
        )
        accuracy_official = score_coverage_complete and official(
            "accuracy_pct_micro", uses_judge=bool(judge_models)
        )
        metric_rows["accuracy_pct_micro"] = {
            "official_metric": accuracy_official,
            **accuracy,
        }
        result.update(
            {
                "primary_metric": "accuracy_pct_micro",
                "official_metric": accuracy_official,
                "score": accuracy,
                "score_valid": bool(result["score_valid"]) and score_coverage_complete,
                "score_coverage": {
                    "complete": score_coverage_complete,
                    "scored_question_count": len(scored_rows),
                    "unscored_question_count": len(samples) - len(scored_rows),
                    "supported_subset_accuracy_pct_micro": supported_subset_accuracy,
                },
            }
        )
        if not score_coverage_complete:
            result["unavailable_metrics"] = {
                "accuracy_pct_micro": (
                    "the official PersonaMem-v3 micro score requires every selected task family; "
                    "structured-action, threaded-cluster, and paired-row protocols are unavailable"
                )
            }
    if task.spec.name == "supermemory-vqa" and scored:
        result["answerability"] = {
            "official_metric": official("answerability"),
            **_answerability(samples),
        }
        result["unavailable_metrics"] = {
            "qa_mrr": "answer-option scores are not exposed by the MindBridge answer backend"
        }
    if task.spec.name == "worldmemarena":
        result["unavailable_metrics"] = {
            "memory_snapshot": (
                "the unified runner has no per-session memory-snapshot export required by the "
                "official recall/correctness, update-handling, and interference protocols"
            ),
            "retrieval_coverage": (
                "the official semantic evidence-coverage judge is not part of checkpoint QA; "
                "the generic exact-ID retrieval block remains a MindBridge diagnostic"
            ),
            "retrieval_ranking": (
                "the official fuzzy memory/session/content matching protocol cannot be reproduced "
                "from MindBridge source IDs"
            ),
        }
    reference_scores = tuple(
        ScoredValue(sample.sample_id, sample.unit_id, sample.ref_at_300)
        for sample in samples
        if sample.ref_at_300 is not None
    )
    if reference_scores:
        ref_summary = {
            "official_metric": official("ref_at_300"),
            **summarize(
                reference_scores,
                seed=seed,
                bootstrap_samples=arguments.bootstrap_samples,
                clamp=(0.0, 1.0),
            ),
        }
        result["ref_at_300"] = ref_summary
        metric_rows["ref_at_300"] = ref_summary
    return result


def _metadata_ids(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _retrieved_sources(sample: SampleResult) -> tuple[str, ...]:
    """Return the retriever's ranked source IDs in rank order, deduplicated.

    This is the ranked list observed inside ``Memory.ask``, not ``sample.evidence``. Evidence is
    narrowed to the hits the generator saw, so scoring it would report grounding behaviour under
    the name of retrieval recall.
    """
    return tuple(dict.fromkeys(source for source in sample.ranked_source_ids if source))


def _retrieval_quality(
    samples: Sequence[SampleResult],
    *,
    seed: int,
    bootstrap_samples: int,
    recall_limit: int,
    retrieval_candidate_limit: int | None = None,
) -> dict[str, object]:
    """Report recall at every cutoff next to the random-ranker expectation.

    R@20 is the measured retrieval ceiling on this harness and a perfect reranker buys only a
    few points, so R@1 is never reported without it. The random-ranker row is the exact
    expectation for a uniform ranker over the same candidate pool, which is what makes a high
    recall interpretable: a pool of ten candidates already gives R@10 = 1.0 by chance.
    """
    ranked_limit = (
        min(RETRIEVAL_CANDIDATE_LIMIT, recall_limit * 3)
        if retrieval_candidate_limit is None
        else retrieval_candidate_limit
    )
    key = next(
        (
            name
            for name in GOLD_SOURCE_KEYS
            if any(gold_source_groups(sample.metadata, name) for sample in samples)
        ),
        None,
    )
    # A published evidence ID that named no stored memory. An adapter that joins a
    # separate label list onto its own source IDs reports what did not match, so a
    # release whose label vocabulary is not the source-ID vocabulary shows up here
    # instead of as a plausible recall number over the handful that happened to join.
    unresolved = sum(
        len(_metadata_ids(sample.metadata, _UNRESOLVED_EVIDENCE_KEY)) for sample in samples
    )
    if key is None:
        return {
            "gold_evidence_key": None,
            "recall_limit": recall_limit,
            "retrieval_candidate_limit": ranked_limit,
            "labelled_question_count": 0,
            "unranked_labelled_question_count": 0,
            "recall_at_k": {},
            "random_ranker_recall_at_k": {},
            "unresolved_gold_evidence_ids": unresolved,
            "unavailable_reason": (
                "this task adapter carries no gold evidence source IDs, so retrieval quality "
                "cannot be measured at these retrieval settings"
            ),
        }
    labelled_all = tuple(sample for sample in samples if gold_source_groups(sample.metadata, key))
    # A labelled question whose run never completed the ranked query is excluded and counted.
    # A completed query with zero hits is measured recall zero, not mistaken for missing data.
    labelled = tuple(sample for sample in labelled_all if sample.ranked_source_ids_complete)
    unranked = len(labelled_all) - len(labelled)
    if not labelled:
        return {
            "gold_evidence_key": key,
            "recall_limit": recall_limit,
            "retrieval_candidate_limit": ranked_limit,
            "labelled_question_count": 0,
            "unranked_labelled_question_count": unranked,
            "recall_at_k": {},
            "random_ranker_recall_at_k": {},
            "unresolved_gold_evidence_ids": unresolved,
            "unavailable_reason": (
                "no labelled question carries the retriever's ranked source list, so retrieval "
                "quality cannot be measured from this run"
            ),
        }
    measured: dict[int, list[ScoredValue]] = {cutoff: [] for cutoff in _RECALL_CUTOFFS}
    random_ranker: dict[int, list[ScoredValue]] = {cutoff: [] for cutoff in _RECALL_CUTOFFS}
    pool_sizes = []
    for sample in labelled:
        gold = gold_source_groups(sample.metadata, key)
        retrieved = _retrieved_sources(sample)
        pool = sample.candidate_count
        pool_sizes.append(pool)
        for cutoff in _RECALL_CUTOFFS:
            window = set(retrieved[:cutoff])
            measured[cutoff].append(
                ScoredValue(
                    sample.sample_id,
                    sample.unit_id,
                    sum(1 for group in gold if window.intersection(group)) / len(gold),
                )
            )
            if pool > 0:
                random_ranker[cutoff].append(
                    ScoredValue(
                        sample.sample_id,
                        sample.unit_id,
                        statistics.fmean(
                            _random_group_hit(pool, len(group), cutoff) for group in gold
                        ),
                    )
                )

    def rows(values: Mapping[int, Sequence[ScoredValue]]) -> dict[str, object]:
        return {
            str(cutoff): summarize(
                tuple(values[cutoff]),
                seed=_task_seed(seed, f"recall@{cutoff}"),
                bootstrap_samples=bootstrap_samples,
                clamp=(0.0, 1.0),
            )
            for cutoff in _RECALL_CUTOFFS
            if values[cutoff]
        }

    return {
        "gold_evidence_key": key,
        "recall_limit": recall_limit,
        "retrieval_candidate_limit": ranked_limit,
        "labelled_question_count": len(labelled),
        "unranked_labelled_question_count": unranked,
        "unresolved_gold_evidence_ids": unresolved,
        "recall_at_k": rows(measured),
        "random_ranker_recall_at_k": rows(random_ranker),
        "random_ranker_method": (
            "exact expectation for a uniformly random ranker: min(1, k / candidate_pool_size) "
            "per gold source, or the hypergeometric chance of ranking any member of a gold "
            "group in the top k, averaged over a question's groups"
        ),
        "candidate_pool_size": {
            "min": min(pool_sizes, default=None),
            "max": max(pool_sizes, default=None),
            "mean": statistics.fmean(pool_sizes) if pool_sizes else None,
        },
        "truncated_cutoffs": [cutoff for cutoff in _RECALL_CUTOFFS if cutoff > ranked_limit],
    }


def _noise_floor(scored: Sequence[ScoredValue], primary: Mapping[str, object]) -> dict[str, object]:
    """Report the smallest difference this run size can resolve."""
    values = tuple(value.value for value in scored)
    error = primary.get("cluster_standard_error")
    standard_error = error if isinstance(error, float) else None
    resolvable = NOISE_FLOOR
    if standard_error is not None:
        resolvable = max(NOISE_FLOOR, 1.959963984540054 * standard_error * math.sqrt(2))
    return {
        "floor": NOISE_FLOOR,
        "per_question_standard_deviation": (statistics.stdev(values) if len(values) > 1 else None),
        "cluster_standard_error": standard_error,
        "minimum_meaningful_difference": resolvable,
        "note": (
            "a difference smaller than minimum_meaningful_difference is inside this run's "
            "noise band and is not a result"
        ),
    }


def _controls(
    task_name: str,
    retrieval: Mapping[str, object],
    blind: Mapping[str, object] | None,
    *,
    is_blind_run: bool,
    retrieves: bool = True,
) -> dict[str, object]:
    """Report the three controls that make a score interpretable, and which are absent.

    Each of these has independently invalidated a conclusion on this project: a random ranker
    reached R@10 = 0.9941 on one benchmark, blind answering already scores 0.383 on another,
    and R@1 moving without R@20 moving is noise.

    An arm that never retrieves (blind, full-context) has no retrieval to control for, so the two
    retrieval controls do not apply to it rather than being missing. Requiring them turned
    `controls_complete` false on every run that carried a blind arm, which hid the difference
    between a task with no gold labels and a healthy one.
    """
    recall = retrieval.get("recall_at_k")
    random_ranker = retrieval.get("random_ranker_recall_at_k")
    recall_rows = recall if isinstance(recall, Mapping) else {}
    random_rows = random_ranker if isinstance(random_ranker, Mapping) else {}
    present = {
        "random_ranker": not retrieves or bool(random_rows),
        "recall_at_20": not retrieves or ("20" in recall_rows and "1" in recall_rows),
        "blind": is_blind_run or blind is not None,
    }
    missing = tuple(name for name in _MANDATORY_CONTROLS if not present[name])
    return {
        "random_ranker": {str(k): v for k, v in random_rows.items()} or None,
        "recall_at_1": recall_rows.get("1"),
        "recall_at_20": recall_rows.get("20"),
        "blind": None if blind is None else dict(blind),
        "is_blind_run": is_blind_run,
        "retrieval_controls_applicable": retrieves,
        "missing": list(missing),
        "interpretable": not missing,
        "reason": (
            None
            if not missing
            else (
                f"{task_name} reports a score without "
                + ", ".join(missing)
                + "; the score is not interpretable as a memory-quality result"
            )
        ),
    }


def _in_run_blind_rows(
    arguments: _Arguments,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
) -> dict[str, dict[str, object]]:
    """Report this run's own blind arm as the blind control, when it was one of the arms."""
    if "blind" not in arguments.arms:
        return {}
    rows: dict[str, dict[str, object]] = {}
    for task in tasks:
        scores = [
            sample.score
            for sample in samples
            if sample.task == task.spec.name and sample.arm == "blind" and sample.score is not None
        ]
        if not scores:
            continue
        rows[task.spec.name] = {
            "run_id": arguments.run_id,
            "primary_metric": task_primary_metric(task.spec.name),
            "mean": statistics.fmean(scores),
            "question_count": len(scores),
            "source": f"blind arm of this run, prompt {BLIND_PROMPT_VERSION}",
        }
    return rows


def _blind_baseline_rows(
    path: Path | None, tasks: Sequence[LoadedTask]
) -> dict[str, dict[str, object]]:
    """Load per-task scores from a prior --blind run of the same evaluation inputs."""
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    document_path = resolved / _RESULTS_FILE if resolved.is_dir() else resolved
    if not document_path.is_file():
        raise FileNotFoundError(f"blind baseline does not exist: {document_path}")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("blind baseline must be a results.jsonl document")
    if document.get("blind") is not True:
        raise ValueError("blind baseline must come from a run started with --blind")
    if document.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise ValueError("blind baseline schema version is unsupported")
    digests = {task.spec.name: task.evaluation_sha256 for task in tasks}
    rows: dict[str, dict[str, object]] = {}
    for row in document.get("tasks", ()):
        name = row.get("task") if isinstance(row, Mapping) else None
        if not isinstance(name, str) or name not in digests:
            continue
        if row.get("evaluation_sha256") != digests[name]:
            raise ValueError(f"blind baseline evaluation inputs differ for {name}")
        score = row.get("score")
        rows[name] = {
            "run_id": document.get("run_id"),
            "primary_metric": row.get("primary_metric"),
            "mean": score.get("mean") if isinstance(score, Mapping) else None,
            "question_count": row.get("question_count"),
        }
    return rows


def _abstentions(samples: Sequence[SampleResult]) -> dict[str, object]:
    count = sum(sample.abstained for sample in samples)
    reasons = {
        reason: sum(sample.abstention_reason == reason for sample in samples)
        for reason in sorted(
            {sample.abstention_reason for sample in samples if sample.abstention_reason is not None}
        )
    }
    return {
        "count": count,
        "rate": 0.0 if not samples else count / len(samples),
        "reasons": reasons,
    }


def _metric_breakdowns(
    task: LoadedTask,
    samples: Sequence[SampleResult],
    arguments: _Arguments,
) -> dict[str, object]:
    family = task_family(task.spec.name)
    fields = _BREAKDOWN_FIELDS.get(family or "", ())
    result: dict[str, object] = {}
    for field_name in fields:
        grouped: dict[str, list[ScoredValue]] = {}
        for sample in samples:
            if sample.score is None:
                continue
            raw = sample.metadata.get(field_name)
            labels = (
                tuple(str(value) for value in raw)
                if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
                else (str(raw),)
            )
            for label in labels:
                if label and label != "None":
                    grouped.setdefault(label, []).append(
                        ScoredValue(sample.sample_id, sample.unit_id, sample.score)
                    )
        if grouped:
            result[field_name] = {
                label: summarize(
                    tuple(rows),
                    seed=_task_seed(arguments.seed, f"{task.spec.name}:{field_name}:{label}"),
                    bootstrap_samples=arguments.bootstrap_samples,
                )
                for label, rows in sorted(grouped.items())
            }
    return result


def _video_mme_v2_rating(
    samples: Sequence[SampleResult], *, seed: int, bootstrap_samples: int
) -> dict[str, object]:
    return summarize(
        _video_mme_v2_group_values(
            tuple(
                {
                    "unit_id": sample.unit_id,
                    "score": sample.score,
                    "metadata": sample.metadata,
                }
                for sample in samples
            )
        ),
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        clamp=(0.0, 100.0),
    )


def _video_mme_v2_accuracy(
    samples: Sequence[SampleResult],
    *,
    seed: int,
    bootstrap_samples: int,
    official_metric: bool,
) -> dict[str, object]:
    """Reproduce the released `_acc.json` scale, denominator, and taxonomy cells."""

    def summary(selected: Sequence[SampleResult], suffix: str) -> dict[str, object]:
        return summarize(
            tuple(
                ScoredValue(sample.sample_id, sample.unit_id, 100.0 * float(sample.score or 0.0))
                for sample in selected
            ),
            seed=_task_seed(seed, suffix),
            bootstrap_samples=bootstrap_samples,
            clamp=(0.0, 100.0),
        )

    answered = tuple(sample for sample in samples if sample.parsed_choice is not None)
    taxonomy_fields = ("group_type", "level", "second_head", "third_head")
    cells: dict[str, object] = {}
    for taxonomy_field in taxonomy_fields:
        labels = sorted({str(sample.metadata.get(taxonomy_field, "")) for sample in samples})
        cells[f"by_{taxonomy_field}"] = {
            (f"level_{label}" if taxonomy_field == "level" else label): summary(
                tuple(
                    sample
                    for sample in samples
                    if str(sample.metadata.get(taxonomy_field, "")) == label
                ),
                f"accuracy:{taxonomy_field}:{label}",
            )
            for label in labels
            if label
        }
    return {
        "official_metric": official_metric,
        "overall": summary(samples, "accuracy"),
        "answered_accuracy": (
            summary(answered, "answered_accuracy")
            if answered
            else {
                **summary((), "answered_accuracy"),
                "mean": 0.0,
            }
        ),
        "question_count": len(samples),
        "answered_count": len(answered),
        "correct_count": sum(sample.score == 1.0 for sample in samples),
        "error_count": sum(sample.error_code is not None for sample in samples),
        **cells,
    }


def _video_mme_v2_group_values(
    rows: Sequence[Mapping[str, object]],
) -> tuple[ScoredValue, ...]:
    from mindbridge.benchmarks.video_mme_v2 import score_group_answers

    groups: dict[str, list[tuple[int, float, Mapping[str, object]]]] = {}
    for row in rows:
        unit_id, score, metadata = row.get("unit_id"), row.get("score"), row.get("metadata")
        if not isinstance(unit_id, str) or not unit_id:
            raise ValueError("Video-MME-v2 sample unit ID must be a non-empty string")
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("Video-MME-v2 sample score must be numeric")
        if not isinstance(metadata, Mapping):
            raise ValueError("Video-MME-v2 sample metadata must be an object")
        position = metadata.get("position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("Video-MME-v2 sample position must be an integer")
        groups.setdefault(unit_id, []).append((position, float(score), metadata))

    ratings = []
    for unit_id in sorted(groups):
        group = sorted(groups[unit_id], key=lambda item: item[0])
        if tuple(item[0] for item in group) != (1, 2, 3, 4):
            raise ValueError(f"Video-MME-v2 group {unit_id} must hold positions 1 to 4")
        metadata = group[0][2]
        rating = score_group_answers(
            str(metadata["group_type"]),
            str(metadata.get("group_structure", "")),
            tuple(item[1] == 1.0 for item in group),
        )
        ratings.append(ScoredValue(unit_id, unit_id, rating))
    return tuple(ratings)


def _answerability(samples: Sequence[SampleResult]) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    for sample in samples:
        actual = bool(sample.metadata["is_answerable"])
        predicted = (
            sample.parsed_choice is not None
            and sample.parsed_choice != sample.metadata["unanswerable_choice"]
        )
        true_positive += actual and predicted
        false_positive += not actual and predicted
        false_negative += actual and not predicted
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _comparisons(
    arguments: _Arguments,
    tasks: Sequence[LoadedTask],
    samples: Sequence[SampleResult],
) -> list[dict[str, object]]:
    if arguments.compare is None:
        return []
    baseline = _baseline_samples(arguments.compare)
    rows = []
    for task in tasks:
        # A regression guard compares like with like: only the product arm, on both sides.
        task_samples = tuple(
            sample
            for sample in samples
            if sample.task == task.spec.name and sample.arm == DEFAULT_ARM
        )
        previous_rows = tuple(
            row
            for row in baseline
            if row.get("task") == task.spec.name and row.get("arm", DEFAULT_ARM) == DEFAULT_ARM
        )
        digests = {row.get("evaluation_sha256") for row in previous_rows}
        if previous_rows and digests != {task.evaluation_sha256}:
            raise ValueError(f"baseline evaluation inputs differ for {task.spec.name}")
        current_protocols = {sample.scorer_protocol for sample in task_samples}
        previous_protocols = {row.get("scorer_protocol") for row in previous_rows}
        if previous_rows and previous_protocols != current_protocols:
            raise ValueError(f"baseline scorer protocol differs for {task.spec.name}")
        current_judges = {
            sample.judge_model for sample in task_samples if sample.judge_model is not None
        }
        previous_judges = {
            model for row in previous_rows if isinstance((model := row.get("judge_model")), str)
        }
        if current_judges and previous_judges and current_judges != previous_judges:
            raise ValueError(f"baseline judge model differs for {task.spec.name}")
        current_scored = tuple(sample for sample in task_samples if sample.score is not None)
        previous_scored = tuple(row for row in previous_rows if _has_score(row))
        if task.spec.name == "video-mme-v2":
            current_ids = tuple(sample.sample_id for sample in current_scored)
            previous_ids = tuple(
                sample_id
                for row in previous_scored
                if isinstance((sample_id := row.get("sample_id")), str)
            )
            if (
                len(previous_ids) != len(previous_scored)
                or len(set(previous_ids)) != len(previous_ids)
                or set(current_ids) != set(previous_ids)
            ):
                raise ValueError("candidate and baseline must contain identical scored samples")
            current = _video_mme_v2_group_values(
                tuple(
                    {
                        "unit_id": sample.unit_id,
                        "score": sample.score,
                        "metadata": sample.metadata,
                    }
                    for sample in current_scored
                )
            )
            previous = _video_mme_v2_group_values(previous_scored)
        else:
            current = tuple(
                ScoredValue(sample.sample_id, sample.unit_id, cast(float, sample.score))
                for sample in current_scored
            )
            previous = tuple(_baseline_value(row) for row in previous_scored)
        if not current:
            continue
        if not previous:
            raise ValueError(f"baseline has no scored samples for {task.spec.name}")
        comparison = paired_comparison(
            current,
            previous,
            seed=_task_seed(arguments.seed, task.spec.name),
            bootstrap_samples=arguments.bootstrap_samples,
        )
        delta = comparison.get("mean")
        rows.append(
            {
                "task": task.spec.name,
                "metric": _comparison_metric(task),
                "noise_floor": NOISE_FLOOR,
                "below_noise_floor": (
                    None if not isinstance(delta, float) else abs(delta) < NOISE_FLOOR
                ),
                **comparison,
            }
        )
    return rows


def _comparison_metric(task: LoadedTask) -> str:
    return task_primary_metric(task.spec.name)


def _has_score(row: Mapping[str, object]) -> bool:
    value = row.get("score")
    return not isinstance(value, bool) and isinstance(value, int | float)


def _baseline_value(row: Mapping[str, object]) -> ScoredValue:
    value = row["score"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("baseline score must be numeric")
    sample_id, unit_id = row.get("sample_id"), row.get("unit_id")
    if not isinstance(sample_id, str) or not isinstance(unit_id, str):
        raise ValueError("baseline sample and unit IDs must be strings")
    return ScoredValue(sample_id, unit_id, float(value))


def _baseline_samples(path: Path) -> tuple[dict[str, object], ...]:
    resolved = path.expanduser().resolve()
    sample_path = resolved / _SAMPLES_FILE if resolved.is_dir() else resolved
    if sample_path.name == _RESULTS_FILE:
        sample_path = sample_path.with_name(_SAMPLES_FILE)
    if not sample_path.is_file():
        raise FileNotFoundError(f"baseline samples do not exist: {sample_path}")
    rows = []
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("baseline sample rows must be JSON objects")
        if row.get("schema_version") != EVAL_SCHEMA_VERSION:
            raise ValueError("baseline sample schema version is unsupported")
        rows.append(row)
    return tuple(rows)


def _task_seed(seed: int, task: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _evidence_budget_chars(memory_config: MindBridgeConfig | None) -> int | None:
    """The budget the product arm grounds with, which is the product default when no file set it.

    A run without a config file builds its `Memory` from `MemoryConfig()` plus the harness's own
    two overrides, so the default budget applies there too; reading `None` for that case made
    the reported candidate limit disagree with the window `ask` actually ranked.
    """
    settings = MemoryConfig() if memory_config is None else memory_config.settings
    return settings.evidence_budget_chars


def _random_group_hit(pool: int, size: int, cutoff: int) -> float:
    """Exact chance a uniform ranker over `pool` puts any of `size` gold items in its top `cutoff`.

    One gold item reduces to `min(1, cutoff / pool)`, the row published before groups existed,
    so singleton labels keep the number they always had. `comb(pool - size, cutoff)` is zero once
    the window is wider than the non-gold pool, which is the certain hit.
    """
    size, cutoff = min(size, pool), min(cutoff, pool)
    return 1.0 - math.comb(pool - size, cutoff) / math.comb(pool, cutoff)
