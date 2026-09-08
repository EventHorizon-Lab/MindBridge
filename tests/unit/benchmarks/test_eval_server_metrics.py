from __future__ import annotations

from typing import Any, cast

import pytest

from mindbridge.benchmarks.eval_server_metrics import (
    MetricsSnapshot,
    metrics_window,
    parse_prometheus,
)


def test_prometheus_parser_keeps_only_selected_vllm_series() -> None:
    values, counts = parse_prometheus(
        """
# HELP vllm:request_success_total Requests.
vllm:request_success_total{model_name="one",finished_reason="stop"} 2
vllm:request_success_total{model_name="one",finished_reason="length"} 1
vllm:e2e_request_latency_seconds_sum{model_name="one"} 9.5
vllm:e2e_request_latency_seconds_count{model_name="one"} 3
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 2
vllm:num_requests_running{model_name="one"} 2
vllm:num_requests_running{model_name="two"} 4
process_cpu_seconds_total 999
"""
    )

    assert values == {
        "vllm:request_success_total": 3.0,
        "vllm:e2e_request_latency_seconds_sum": 9.5,
        "vllm:e2e_request_latency_seconds_count": 3.0,
        "vllm:num_requests_running": 6.0,
    }
    assert counts["vllm:request_success_total"] == 2
    assert counts["vllm:num_requests_running"] == 2


def test_metrics_window_reports_counter_histogram_and_gauge_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = MetricsSnapshot(
        "2026-09-04T00:00:00+00:00",
        {
            "vllm:request_success_total": 10.0,
            "vllm:prompt_tokens_total": 100.0,
            "vllm:e2e_request_latency_seconds_sum": 20.0,
            "vllm:e2e_request_latency_seconds_count": 10.0,
            "vllm:num_requests_running": 2.0,
        },
        {"vllm:num_requests_running": 1},
    )
    after = MetricsSnapshot(
        "2026-09-04T00:01:00+00:00",
        {
            "vllm:request_success_total": 14.0,
            "vllm:prompt_tokens_total": 180.0,
            "vllm:e2e_request_latency_seconds_sum": 30.0,
            "vllm:e2e_request_latency_seconds_count": 14.0,
            "vllm:num_requests_running": 4.0,
        },
        {"vllm:num_requests_running": 2},
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_server_metrics.capture_metrics",
        lambda *_args, **_kwargs: after,
    )

    result = metrics_window("https://models.example/metrics", before)
    payload = cast(dict[str, Any], result)

    assert payload["scope"] == "server_process_global"
    assert payload["exclusive_attribution"] is False
    assert payload["request_count_delta"] == 4.0
    assert payload["counters"]["vllm:prompt_tokens_total"]["delta"] == 80.0
    histogram = payload["histograms"]["vllm:e2e_request_latency_seconds"]
    assert histogram["count_delta"] == 4.0
    assert histogram["mean"] == 2.5
    assert payload["gauges"]["vllm:num_requests_running"]["end_average"] == 2.0


def test_metrics_window_marks_counter_resets_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    before = MetricsSnapshot(
        "start", {"vllm:request_success_total": 10.0}, {"vllm:request_success_total": 1}
    )
    after = MetricsSnapshot(
        "end", {"vllm:request_success_total": 2.0}, {"vllm:request_success_total": 1}
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_server_metrics.capture_metrics",
        lambda *_args, **_kwargs: after,
    )

    result = metrics_window("https://models.example/metrics", before)

    assert result["status"] == "partial"
    assert result["request_count_delta"] is None
    assert result["counter_resets"] == ["vllm:request_success_total"]


def test_metrics_window_subtracts_interleaved_judge_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(label: str, requests: float, tokens: float) -> MetricsSnapshot:
        return MetricsSnapshot(
            label,
            {
                "vllm:request_success_total": requests,
                "vllm:prompt_tokens_total": tokens,
                "vllm:e2e_request_latency_seconds_sum": requests * 2,
                "vllm:e2e_request_latency_seconds_count": requests,
            },
            {},
        )

    before = snapshot("start", 10, 100)
    judge_start = snapshot("judge-start", 12, 120)
    judge_end = snapshot("judge-end", 15, 180)
    after = snapshot("end", 19, 260)
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_server_metrics.capture_metrics",
        lambda *_args, **_kwargs: after,
    )

    result = cast(
        dict[str, Any],
        metrics_window(
            "https://models.example/metrics",
            before,
            excluded=((judge_start, judge_end),),
        ),
    )

    assert result["excluded_window_count"] == 1
    assert result["request_count_delta"] == 6.0
    assert result["counters"]["vllm:prompt_tokens_total"]["delta"] == 100.0
    histogram = result["histograms"]["vllm:e2e_request_latency_seconds"]
    assert histogram["count_delta"] == 6.0
    assert histogram["sum_delta"] == 12.0


def test_metrics_window_preserves_snapshot_failure_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = MetricsSnapshot("start", {}, {}, "TimeoutError")
    after = MetricsSnapshot("end", {}, {}, "URLError")
    monkeypatch.setattr(
        "mindbridge.benchmarks.eval_server_metrics.capture_metrics",
        lambda *_args, **_kwargs: after,
    )

    result = metrics_window("https://models.example/metrics", before)

    assert result["status"] == "unavailable"
    assert result["start_error_type"] == "TimeoutError"
    assert result["end_error_type"] == "URLError"
