from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from mindbridge.benchmarks.eval_regression import load_result, performance_comparisons


def _result() -> dict[str, Any]:
    return {
        "schema_version": 11,
        "runner_version": "mindbridge_eval_official_v11",
        "status": "completed",
        "limit": -1,
        "offset": 0,
        "unit_concurrency": 1,
        "request_concurrency": 1,
        "recall_limit": 20,
        "seeds": [0, 1234, 1234, 1234],
        "predict_only": False,
        "response_cache": None,
        "arms": {"selected": ["mindbridge"], "ingest": "add"},
        "measurement_protocol": {"state": "cold", "repeat_index": 0},
        "model": {
            "adapter": "mindbridge",
            "embedding_model": "tencent/WeMM-Embedding-9B",
            "embedding_revision": None,
            "embedding_dimension": 4096,
            "embedding_warmup": {"count": 1, "task": "query"},
            "device": "remote,cuda",
            "generation_model": "Qwen3.8-27B",
            "generation_base_url": "http://xyrobot-vl.xyrobot.com/v1",
            "generation_modalities": ["text"],
            "generation_seed": 0,
            "generation_temperature": 0.0,
            "generation_kwargs": "temperature=0,do_sample=false,seed=0",
            "generation_min_video_seconds": None,
            "transcription_model": None,
            "timeout_seconds": 300.0,
            "memory_config": {"embedding": {"dimension": 4096}},
        },
        "environment": {
            "python_version": "3.12.0",
            "platform": "Linux-fixture",
            "runtime_versions": {"torch": "2.8.0"},
            "acceleration_runtime": {"torch_cuda_version": "12.8"},
            "hardware": {
                "machine": "x86_64",
                "processor": "x86_64",
                "cpu_model": "Fixture CPU",
                "logical_cores": 32,
                "ram_total_bytes": 64 * 1024**3,
                "cuda_device_uuids": {"0": "GPU-fixture"},
                "gpus": [
                    {
                        "index": 0,
                        "name": "NVIDIA GeForce RTX 5090",
                        "uuid": "GPU-fixture",
                        "memory_total_mib": 32607,
                        "driver_version": "580.178.04",
                        "power_limit_watts": 575.0,
                    }
                ],
            },
        },
        "tasks": [
            {
                "task": "fixture",
                "arm": "mindbridge",
                "evaluation_sha256": "e" * 64,
                "batch_size": 1,
                "input_modalities": ["text"],
                "question_count": 10,
                "performance": {
                    "answer": {
                        "end_to_end_time_to_first_token_ms": {
                            "complete": True,
                            "p95": 100.0,
                        },
                        "latency_ms": {"complete": True, "p95": 500.0},
                        "throughput_per_active_second": 2.0,
                    },
                    "ask_retrieval_core": {"latency_ms": {"p95": 40.0}},
                    "search_e2e": {
                        "planned_count": 10,
                        "attempt_count": 10,
                        "count": 10,
                        "success_count": 10,
                        "error_count": 0,
                        "complete": True,
                        "latency_ms": {"complete": True, "p95": 40.0},
                    },
                    "token_usage": {"product": {"average_tokens_per_request": 250.0}},
                },
            }
        ],
    }


def test_performance_comparison_gates_latency_tokens_and_throughput() -> None:
    baseline = _result()
    candidate = deepcopy(baseline)
    performance = candidate["tasks"][0]["performance"]
    performance["answer"]["end_to_end_time_to_first_token_ms"]["p95"] = 120.0
    performance["answer"]["latency_ms"]["p95"] = 540.0
    performance["answer"]["throughput_per_active_second"] = 1.7
    performance["ask_retrieval_core"]["latency_ms"]["p95"] = 400.0
    performance["search_e2e"]["latency_ms"]["p95"] = 42.0
    performance["token_usage"]["product"]["average_tokens_per_request"] = 300.0

    rows = performance_comparisons(
        candidate,
        baseline,
        {
            "answer_e2e_ttft_p95": 0.10,
            "answer_e2e_latency_p95": 0.10,
            "retrieval_e2e_latency_p95": 0.10,
            "tokens_per_call": 0.10,
            "answer_throughput": 0.10,
        },
    )

    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["answer_e2e_ttft_p95"]["regressed"] is True
    assert by_metric["answer_e2e_latency_p95"]["regressed"] is False
    assert by_metric["retrieval_e2e_latency_p95"]["regressed"] is False
    assert by_metric["retrieval_e2e_latency_p95"]["candidate"] == 42.0
    assert by_metric["tokens_per_call"]["regressed"] is True
    assert by_metric["answer_throughput"]["regressed"] is True


def test_performance_comparison_rejects_different_concurrency() -> None:
    candidate = _result()
    baseline = _result()
    baseline["request_concurrency"] = 4

    with pytest.raises(ValueError, match="request_concurrency differs"):
        performance_comparisons(candidate, baseline, {"answer_e2e_latency_p95": 0.1})


def test_performance_comparison_rejects_different_ingest_modes() -> None:
    candidate = _result()
    baseline = _result()
    candidate["arms"]["ingest"] = "capture"

    with pytest.raises(ValueError, match=r"arms\.ingest differs"):
        performance_comparisons(candidate, baseline, {"answer_e2e_latency_p95": 0.1})


def test_performance_comparison_rejects_incomplete_token_usage() -> None:
    candidate = _result()
    baseline = _result()
    candidate["tasks"][0]["performance"]["token_usage"]["product"]["average_tokens_per_request"] = (
        None
    )

    with pytest.raises(ValueError, match="no complete tokens_per_call"):
        performance_comparisons(candidate, baseline, {"tokens_per_call": 0.1})


def test_performance_comparison_rejects_incomplete_ttft_distribution() -> None:
    candidate = _result()
    baseline = _result()
    candidate["tasks"][0]["performance"]["answer"]["end_to_end_time_to_first_token_ms"][
        "complete"
    ] = False

    with pytest.raises(ValueError, match="no complete answer_e2e_ttft_p95"):
        performance_comparisons(candidate, baseline, {"answer_e2e_ttft_p95": 0.1})


def test_performance_comparison_rejects_incomplete_answer_latency_distribution() -> None:
    candidate = _result()
    baseline = _result()
    candidate["tasks"][0]["performance"]["answer"]["latency_ms"]["complete"] = False

    with pytest.raises(ValueError, match="no complete answer_e2e_latency_p95"):
        performance_comparisons(candidate, baseline, {"answer_e2e_latency_p95": 0.1})


def test_performance_comparison_rejects_incomplete_search_replay() -> None:
    candidate = _result()
    baseline = _result()
    candidate["tasks"][0]["performance"]["search_e2e"]["complete"] = False

    with pytest.raises(ValueError, match="no complete retrieval_e2e_latency_p95"):
        performance_comparisons(candidate, baseline, {"retrieval_e2e_latency_p95": 0.1})


def test_performance_comparison_rejects_incomplete_search_latency_distribution() -> None:
    candidate = _result()
    baseline = _result()
    candidate["tasks"][0]["performance"]["search_e2e"]["latency_ms"]["complete"] = False

    with pytest.raises(ValueError, match="no complete retrieval_e2e_latency_p95"):
        performance_comparisons(candidate, baseline, {"retrieval_e2e_latency_p95": 0.1})


def test_load_result_accepts_legacy_results_json_beside_samples(tmp_path: Path) -> None:
    result = _result()
    result["schema_version"] = 10
    (tmp_path / "results.json").write_text(json.dumps(result), encoding="utf-8")
    samples = tmp_path / "samples.jsonl"
    samples.write_text("{}\n", encoding="utf-8")

    assert load_result(samples)["schema_version"] == 10
