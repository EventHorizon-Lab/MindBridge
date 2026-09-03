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
    VISION_BATCHES_FAILED,
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


def test_product_token_cost_survives_an_incomplete_judge() -> None:
    """One judge request without usage must not hide a fully reported product cost."""
    telemetry = EvaluationTelemetry()
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            with telemetry.tracer.start_as_current_span(
                "mindbridge.model.generation",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "generation"},
            ):
                mark_model_requests(1)
                record_model_usage(
                    input_tokens=70,
                    output_tokens=10,
                    total_tokens=80,
                    expected_requests=1,
                    reported_requests=1,
                )
            with telemetry.tracer.start_as_current_span(
                "mindbridge.model.judge",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "judge"},
            ):
                mark_model_requests(1)
                record_model_usage(
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    expected_requests=1,
                    reported_requests=0,
                )
        result = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    usage = cast(dict[str, object], result["token_usage"])
    product = cast(dict[str, object], usage["product"])
    assert usage["complete"] is False
    assert usage["total_tokens"] is None
    assert product["modules"] == ["generation"]
    assert product["complete"] is True
    assert product["total_tokens"] == 80
    assert product["average_tokens"] == pytest.approx(40.0)


def test_evaluation_telemetry_sums_failed_description_batches_per_task() -> None:
    """A write-path caption loss has to reach the artifact, because tokens cannot show it.

    A batch whose reply could not be used reports no usage at all, so it is invisible in
    `token_usage`; what it leaves behind is a media memory with an empty full-text document.
    """
    telemetry = EvaluationTelemetry()
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            for failures in (1, 1):
                with (
                    telemetry.tracer.start_as_current_span(
                        "mindbridge.add",
                        attributes={SPAN_KIND: "operation"},
                    ),
                    telemetry.tracer.start_as_current_span(
                        "mindbridge.model.vision",
                        attributes={SPAN_KIND: "model", MODEL_MODULE: "vision"},
                    ) as span,
                ):
                    span.set_attribute(VISION_BATCHES_FAILED, failures)
            # One batch that succeeded records no counter and must not be counted as a loss.
            with (
                telemetry.tracer.start_as_current_span(
                    "mindbridge.add",
                    attributes={SPAN_KIND: "operation"},
                ),
                telemetry.tracer.start_as_current_span(
                    "mindbridge.model.vision",
                    attributes={SPAN_KIND: "model", MODEL_MODULE: "vision"},
                ),
            ):
                pass
            # Counted from model spans only. An enclosing span carrying the same attribute must
            # not add to the total: that is what would silently double-count every failure if the
            # counter were ever published one level up.
            with telemetry.tracer.start_as_current_span(
                "mindbridge.add",
                attributes={SPAN_KIND: "operation", VISION_BATCHES_FAILED: 7},
            ):
                pass
        performance = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    assert performance["vision"] == {"failed_batches": 2}
