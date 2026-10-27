"""Five observed evaluation boundaries; source coverage is never evidence truth."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from mindbridge.benchmarks.eval_cache import EvidenceInterval
from mindbridge.benchmarks.official_scorers import GOLD_SOURCE_KEYS, gold_source_groups

if TYPE_CHECKING:
    from mindbridge.benchmarks.eval_results import SampleResult

LAYERS = ("input", "retrieval", "organization", "reader_input", "answer")


def quality_diagnostics(sample: SampleResult) -> dict[str, object]:
    """Join observed boundaries without inferring missing spans or assigning causal blame."""
    key = next((key for key in GOLD_SOURCE_KEYS if gold_source_groups(sample.metadata, key)), None)
    groups = () if key is None else gold_source_groups(sample.metadata, key)
    unresolved = sample.metadata.get("unresolved_evidence_ids", ())

    def coverage(sources: Sequence[str] | None) -> dict[str, object] | None:
        if sources is None or not groups or unresolved or any(not item.strip() for item in sources):
            return None
        present = set(sources)
        matched = [index for index, group in enumerate(groups) if present.intersection(group)]
        return {
            "matched_groups": matched,
            "group_count": len(groups),
            "recall": len(matched) / len(groups),
            "all_present": len(matched) == len(groups),
        }

    def layer(sources: Sequence[str] | None, measurement: str) -> dict[str, object]:
        return {
            "observed": sources is not None,
            "measurement": measurement,
            "source_ids": None if sources is None else list(dict.fromkeys(sources)),
            "source_coverage": coverage(sources),
        }

    organized = _sources(sample.generation_evidence)
    layers = {
        "input": layer(sample.input_source_ids, "eligible_labelled_input_sources"),
        "retrieval": layer(
            sample.ranked_source_ids if sample.ranked_source_ids_complete else None,
            "ranked_candidates_before_recall_and_generation",
        ),
        "organization": layer(organized, "last_observed_answerer_input"),
    }
    layers["input"].update(
        eligible_source_count=sample.candidate_count,
        ingest_failure_count=sample.ingest_failure_count,
        representation_quality=None,
        index_quality=None,
    )
    layers["organization"].update(
        observed=sample.generation_evidence is not None,
        partial_source_ids=list(
            dict.fromkeys(item.source_id for item in sample.excerpt_evidence if item.source_id)
        ),
    )
    request = sample.grounding_request
    record_ids = None if request is None else _ids(request.get("record_ids"))
    # Returned hits cannot map records removed by backend packing. The independently observed
    # answerer input includes those records, and can also include recall-plan additions.
    by_id = {
        item.memory_id: item.source_id
        for item in sample.generation_evidence or ()
        if item.source_id is not None
    }
    unmapped = [] if record_ids is None else [item for item in record_ids if item not in by_id]
    delivered = (
        None if record_ids is None or unmapped else tuple(by_id[item] for item in record_ids)
    )
    layers["reader_input"] = {
        **layer(delivered, "last_prepared_generation_request"),
        "observed": record_ids is not None,
        "record_ids": record_ids,
        "unmapped_record_ids": unmapped,
        "asset_ids": None if request is None else request.get("asset_ids"),
        "text_bytes": None if request is None else request.get("text_bytes"),
        "media_fallback": None if request is None else request.get("media_fallback"),
        "successful": None if request is None else request.get("successful"),
        "perception_quality": None,
    }
    layers["answer"] = {
        "observed": bool(sample.prediction) or sample.error_code is not None,
        "score": sample.score,
        "abstained": sample.abstained,
        "error_code": sample.error_code,
        "scorer_error": sample.scorer_error,
        "reader_quality": None,
    }
    return {
        "trace_id": sample.trace_id,
        "answer_span_id": sample.answer_span_id,
        "observation_origin": "response_cache" if sample.cached else "execution",
        "gold_source_key": key,
        "label_granularity": None if key is None else "source_group",
        "fact_span_coverage": None,
        "causal_attribution": None,
        "layers": layers,
    }


def _sources(evidence: Sequence[EvidenceInterval] | None) -> tuple[str, ...] | None:
    if evidence is None or any(item.source_id is None for item in evidence):
        return None
    return tuple(cast(str, item.source_id) for item in evidence)


def _ids(value: object) -> list[str] | None:
    if not isinstance(value, list | tuple) or any(not isinstance(item, str) for item in value):
        return None
    return list(value)


def summarize_diagnostics(samples: Sequence[SampleResult]) -> dict[str, object]:
    """Report separate observation, label and score denominators for every boundary."""
    diagnostics = [quality_diagnostics(sample) for sample in samples]
    layers: dict[str, object] = {}
    for name in LAYERS[:-1]:
        rows = [
            cast(dict[str, dict[str, object]], diagnostic["layers"])[name]
            for diagnostic in diagnostics
        ]
        measured = [row for row in rows if row["source_coverage"] is not None]
        complete = [
            (sample, row)
            for sample, row in zip(samples, rows, strict=True)
            if row["source_coverage"] is not None
            and cast(dict[str, object], row["source_coverage"])["all_present"] is True
        ]
        scores = [sample.score for sample, _ in complete if sample.score is not None]
        layers[name] = {
            "observed_count": sum(row["observed"] is True for row in rows),
            "coverage_measured_count": len(measured),
            "all_labelled_sources_present_count": len(complete),
            "scored_with_all_labelled_sources_count": len(scores),
            "mean_score_with_all_labelled_sources": statistics.fmean(scores) if scores else None,
            "abstained_with_all_labelled_sources_count": sum(
                sample.abstained for sample, _ in complete
            ),
            "errors_with_all_labelled_sources_count": sum(
                sample.error_code is not None or sample.scorer_error is not None
                for sample, _ in complete
            ),
        }
    scores = [sample.score for sample in samples if sample.score is not None]
    layers["answer"] = {
        "observed_count": sum(
            bool(sample.prediction) or sample.error_code is not None for sample in samples
        ),
        "scored_count": len(scores),
        "mean_score": statistics.fmean(scores) if scores else None,
        "error_count": sum(sample.error_code is not None for sample in samples),
        "scorer_error_count": sum(sample.scorer_error is not None for sample in samples),
        "abstained_count": sum(sample.abstained for sample in samples),
    }
    return {
        "question_count": len(samples),
        "source_labelled_question_count": sum(
            diagnostic["gold_source_key"] is not None for diagnostic in diagnostics
        ),
        "fact_span_labelled_question_count": 0,
        "causal_attribution": None,
        "layers": layers,
    }
