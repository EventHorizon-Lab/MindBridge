"""Checks for bounded benchmark timing and token aggregation."""

from __future__ import annotations

from typing import cast

import pytest

from mindbridge._telemetry import (
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    MODEL_MODULE,
    MODEL_TTFT,
    SPAN_KIND,
    mark_model_requests,
    record_model_usage,
)
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    EvaluationTelemetry,
)


def test_evaluation_telemetry_aggregates_nodes_ttft_and_modal_tokens() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with (
            telemetry.tracer.start_as_current_span(
                BENCHMARK_TASK_SPAN,
                attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
            ),
            telemetry.tracer.start_as_current_span(
                "mindbridge.ask",
                attributes={SPAN_KIND: "operation"},
            ),
            telemetry.tracer.start_as_current_span(
                "mindbridge.model.generation",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "generation"},
            ) as span,
        ):
            span.set_attribute(MODEL_TTFT, 0.025)
            span.set_attribute(GROUNDING_MEDIA_ELIDED, 2)
            span.set_attribute(GROUNDING_HITS_DROPPED, 1)
            record_model_usage(
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
                input_by_modality={"text": 7, "image": 5},
                output_by_modality={"text": 3},
            )
        result = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    duration = cast(dict[str, float], result["duration_seconds"])
    assert duration["total"] >= 0
    assert duration["average"] == pytest.approx(duration["total"] / 2)
    nodes = cast(dict[str, dict[str, object]], result["nodes"])
    generation = nodes["mindbridge.model.generation"]
    assert generation["count"] == 1
    assert cast(dict[str, float], generation["ttft_ms"])["average"] == pytest.approx(25)
    usage = cast(dict[str, object], result["token_usage"])
    assert usage["complete"] is True
    assert usage["total_tokens"] == 15
    assert usage["average_tokens"] == pytest.approx(7.5)
    assert usage["modality_breakdown_complete"] is True
    assert usage["input_by_modality"] == {"image": 5, "text": 7}
    assert result["grounding"] == {"dropped_hits": 1, "media_elided_hits": 2}


def test_evaluation_telemetry_does_not_turn_missing_usage_into_zero() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with (
            telemetry.tracer.start_as_current_span(
                BENCHMARK_TASK_SPAN,
                attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
            ),
            telemetry.tracer.start_as_current_span(
                "mindbridge.model.generation",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "generation"},
            ),
        ):
            mark_model_requests(1)
            record_model_usage(
                input_tokens=5,
                output_tokens=None,
                total_tokens=None,
                input_by_modality={"text": 5},
                expected_requests=1,
                reported_requests=0,
            )
        result = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    usage = cast(dict[str, object], result["token_usage"])
    assert usage["complete"] is False
    assert usage["total_tokens"] is None
    assert usage["reported_total_tokens"] == 0
    assert usage["input_tokens"] == 5
    assert usage["input_by_modality"] == {"text": 5}
