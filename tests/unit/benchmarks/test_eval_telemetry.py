"""Checks for benchmark timing and token aggregation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

import mindbridge.benchmarks.eval_telemetry as eval_telemetry
from mindbridge._telemetry import (
    GEN_AI_TTFC,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    MODEL_MODULE,
    MODEL_REQUEST_COUNT,
    MODEL_TTFT,
    OPERATION_TTFT,
    SPAN_KIND,
    TOKEN_EXPECTED_REQUEST_COUNT,
    TOKEN_REPORTED_REQUEST_COUNT,
    TOKEN_TOTAL,
    VISION_BATCHES_FAILED,
    mark_model_requests,
    record_model_usage,
)
from mindbridge.benchmarks.eval_telemetry import (
    BENCHMARK_ANSWER_SPAN,
    BENCHMARK_ARM_SPAN,
    BENCHMARK_DIAGNOSTIC_SPAN,
    BENCHMARK_JUDGE_SPAN,
    BENCHMARK_PURPOSE,
    BENCHMARK_SAMPLE,
    BENCHMARK_TASK,
    BENCHMARK_TASK_SPAN,
    DIAGNOSTIC_PURPOSE,
    EvaluationTelemetry,
)


def test_telemetry_distributions_retain_every_observation() -> None:
    durations = eval_telemetry._Durations()
    samples = eval_telemetry._Samples()
    tokens = eval_telemetry._Tokens()

    for index in range(3):
        durations.add(
            cast(
                ReadableSpan,
                SimpleNamespace(
                    start_time=index * 10,
                    end_time=index * 10 + 5,
                    attributes={
                        OPERATION_TTFT: float(index + 1),
                        GEN_AI_TTFC: (index + 1) / 1_000,
                    },
                ),
            )
        )
        samples.add(float(index + 1))
        tokens.add(
            {
                MODEL_REQUEST_COUNT: 1,
                TOKEN_EXPECTED_REQUEST_COUNT: 1,
                TOKEN_REPORTED_REQUEST_COUNT: 1,
                TOKEN_TOTAL: index + 1,
            }
        )

    assert (
        len(durations.durations_ns)
        == len(durations.intervals_ns)
        == len(durations.ttft_ms)
        == len(durations.ttfc_ms)
        == 3
    )
    assert len(samples.values) == len(tokens.exact_call_tokens) == 3
    assert durations.latency_ms() == {
        "count": 3,
        "retained_count": 3,
        "complete": True,
        "average": 5 / 1_000_000,
        "p50": 5 / 1_000_000,
        "p95": 5 / 1_000_000,
        "p99": 5 / 1_000_000,
    }
    duration_seconds = cast(
        dict[str, object], eval_telemetry._TaskTelemetry(run=durations).json(3)["duration_seconds"]
    )
    assert duration_seconds["mindbridge"] == pytest.approx(15 / 1_000_000_000)
    assert duration_seconds["total"] == pytest.approx(15 / 1_000_000_000)
    assert cast(dict[str, object], durations.ttft_json())["average"] == 2.0
    assert cast(dict[str, object], durations.ttfc_json())["average"] == 2.0
    assert cast(dict[str, object], durations.ttft_json())["p99"] == pytest.approx(2.98)
    assert cast(dict[str, object], durations.ttfc_json())["p99"] == pytest.approx(2.98)
    assert samples.json()["p99"] == pytest.approx(2.98)
    per_call = cast(dict[str, object], tokens.json(3)["per_call_total_tokens"])
    assert per_call["complete"] is True
    assert per_call["average"] == 2.0
    assert per_call["p99"] == pytest.approx(2.98)

    other_tokens = eval_telemetry._Tokens()
    for value in (4, 5):
        other_tokens.add(
            {
                MODEL_REQUEST_COUNT: 1,
                TOKEN_EXPECTED_REQUEST_COUNT: 1,
                TOKEN_REPORTED_REQUEST_COUNT: 1,
                TOKEN_TOTAL: value,
            }
        )
    tokens.merge(other_tokens)
    assert tokens.exact_call_tokens == [1.0, 2.0, 3.0, 4.0, 5.0]


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
    assert usage["input_tokens"] is None
    assert usage["reported_input_tokens"] == 5
    assert usage["input_tokens_complete"] is False
    assert usage["output_tokens"] is None
    assert usage["observed_output_tokens_per_second"] is None
    assert usage["input_by_modality"] == {"text": 5}


def test_component_tokens_and_incomplete_per_call_average_stay_unknown() -> None:
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
                record_model_usage(
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=10,
                    expected_requests=1,
                    reported_requests=1,
                )
            with telemetry.tracer.start_as_current_span(
                "mindbridge.model.generation",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "generation"},
            ):
                mark_model_requests(1)
        result = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    usage = cast(dict[str, object], result["token_usage"])
    assert usage["complete"] is False
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["cached_input_tokens"] is None
    assert usage["reasoning_output_tokens"] is None
    assert usage["observed_output_tokens_per_second"] is None
    per_call = cast(dict[str, object], usage["per_call_total_tokens"])
    assert per_call == {
        "count": 2,
        "retained_count": 1,
        "observed_count": 1,
        "complete": False,
        "average": None,
        "p50": 10.0,
        "p95": 10.0,
        "p99": 10.0,
        "retained_average": 10.0,
    }


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


def test_ttft_distribution_marks_missing_observations_incomplete() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            for index in range(2):
                with telemetry.tracer.start_as_current_span(
                    BENCHMARK_ANSWER_SPAN,
                    attributes={
                        BENCHMARK_SAMPLE: f"fixture/unit/q{index}",
                        SPAN_KIND: "benchmark",
                    },
                ) as span:
                    if index == 0:
                        span.set_attribute(OPERATION_TTFT, 12.0)
        result = telemetry.result("fixture", question_count=2)
    finally:
        telemetry.close()

    answer = cast(dict[str, object], result["answer"])
    ttft = cast(dict[str, object], answer["end_to_end_time_to_first_token_ms"])
    assert ttft == {
        "count": 2,
        "retained_count": 1,
        "observed_count": 1,
        "complete": False,
        "average": None,
        "p50": 12.0,
        "p95": 12.0,
        "p99": 12.0,
        "retained_average": 12.0,
    }


def test_diagnostic_search_has_caller_latency_and_separate_tokens() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with (
            telemetry.tracer.start_as_current_span(
                BENCHMARK_TASK_SPAN,
                attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
            ),
            telemetry.tracer.start_as_current_span(
                BENCHMARK_DIAGNOSTIC_SPAN,
                attributes={
                    BENCHMARK_SAMPLE: "fixture/unit/q1",
                    BENCHMARK_PURPOSE: DIAGNOSTIC_PURPOSE,
                    SPAN_KIND: "benchmark",
                },
            ),
            telemetry.tracer.start_as_current_span(
                "mindbridge.search",
                attributes={SPAN_KIND: "operation"},
            ),
            telemetry.tracer.start_as_current_span(
                "mindbridge.model.embedding",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "embedding"},
            ),
        ):
            record_model_usage(
                input_tokens=4,
                output_tokens=0,
                total_tokens=4,
                expected_requests=1,
                reported_requests=1,
            )
        result = telemetry.result("fixture", question_count=0)
    finally:
        telemetry.close()

    diagnostic = cast(dict[str, object], result["diagnostic"])
    caller = cast(dict[str, object], diagnostic["search_e2e"])
    sdk = cast(dict[str, object], diagnostic["sdk_operation"])
    usage = cast(dict[str, object], diagnostic["token_usage"])
    product_search = cast(dict[str, object], result["search_e2e"])
    assert caller["span"] == BENCHMARK_DIAGNOSTIC_SPAN
    assert caller["attempt_count"] == caller["count"] == 1
    assert sdk["span"] == "mindbridge.search"
    assert sdk["attempt_count"] == sdk["count"] == 1
    assert product_search["measurement"] == "post_answer_warm_store_replay"
    assert product_search["planned_count"] == product_search["success_count"] == 1
    assert product_search["complete"] is True
    assert product_search["latency_ms"] == caller["latency_ms"]
    assert product_search["sdk_operation"] == sdk
    assert usage["request_count"] == 1
    assert usage["total_tokens"] == 4
    assert result["nodes"] == {}
    assert cast(dict[str, object], result["token_usage"])["request_count"] == 0


def test_zero_request_model_span_is_not_an_inference_node() -> None:
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
            mark_model_requests(0, token_usage_expected=0)
        result = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    assert "mindbridge.model.generation" not in cast(dict[str, object], result["nodes"])
    assert cast(dict[str, object], result["token_usage"])["request_count"] == 0


def test_asr_ratios_use_matching_successful_call_sets() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            with telemetry.tracer.start_as_current_span(
                "mindbridge.model.transcription",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "transcription"},
            ):
                record_model_usage(
                    input_tokens=0,
                    output_tokens=1,
                    total_tokens=1,
                    audio_seconds=2.0,
                )
            with telemetry.tracer.start_as_current_span(
                "mindbridge.model.transcription",
                attributes={SPAN_KIND: "model", MODEL_MODULE: "transcription"},
            ) as failed:
                record_model_usage(
                    input_tokens=0,
                    output_tokens=1,
                    total_tokens=1,
                    audio_seconds=100.0,
                )
                failed.set_status(StatusCode.ERROR)
        result = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    asr = cast(dict[str, object], result["asr"])
    successful = cast(dict[str, float], asr["successful_inference_latency_ms"])
    assert asr["call_count"] == 2
    assert asr["success_count"] == 1
    assert asr["error_count"] == 1
    assert asr["audio_seconds"] == 2.0
    assert asr["compute_seconds"] == pytest.approx(successful["average"] / 1_000)
    assert asr["real_time_factor"] == pytest.approx(cast(float, asr["compute_seconds"]) / 2.0)


def test_cached_product_and_uncached_judges_share_the_union_denominator() -> None:
    telemetry = EvaluationTelemetry()
    try:
        with telemetry.tracer.start_as_current_span(
            BENCHMARK_TASK_SPAN,
            attributes={BENCHMARK_TASK: "fixture", SPAN_KIND: "benchmark"},
        ):
            with (
                telemetry.tracer.start_as_current_span(
                    BENCHMARK_ARM_SPAN,
                    attributes={SPAN_KIND: "benchmark"},
                ),
                telemetry.tracer.start_as_current_span(
                    BENCHMARK_ANSWER_SPAN,
                    attributes={
                        BENCHMARK_SAMPLE: "fixture/unit/q1",
                        SPAN_KIND: "benchmark",
                    },
                ),
                telemetry.tracer.start_as_current_span(
                    "mindbridge.model.generation",
                    attributes={SPAN_KIND: "model", MODEL_MODULE: "generation"},
                ),
            ):
                record_model_usage(input_tokens=8, output_tokens=2, total_tokens=10)
            for sample in ("fixture/unit/q1", "fixture/unit/q2"):
                with telemetry.tracer.start_as_current_span(
                    BENCHMARK_JUDGE_SPAN,
                    attributes={
                        BENCHMARK_SAMPLE: sample,
                        SPAN_KIND: "model",
                        MODEL_MODULE: "judge",
                    },
                ):
                    record_model_usage(input_tokens=4, output_tokens=1, total_tokens=5)
        result = telemetry.result("fixture", question_count=1)
    finally:
        telemetry.close()

    duration = cast(dict[str, object], result["duration_seconds"])
    usage = cast(dict[str, object], result["token_usage"])
    by_module = cast(dict[str, dict[str, object]], usage["by_module"])
    assert duration["measured_product_question_count"] == 1
    assert duration["measured_judge_question_count"] == 2
    assert duration["measured_question_count"] == 2
    assert duration["average_denominator_question_count"] == 2
    assert usage["average_denominator_question_count"] == 2
    assert usage["average_tokens"] == pytest.approx(10.0)
    assert by_module["generation"]["average_tokens"] == pytest.approx(10.0)
    assert by_module["judge"]["average_tokens"] == pytest.approx(5.0)
