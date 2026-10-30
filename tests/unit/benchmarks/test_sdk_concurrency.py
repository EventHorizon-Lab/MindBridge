"""SDK load metrics must count acknowledged work and separate immediate rejection."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from mindbridge import Memory, StorageError
from mindbridge.benchmarks import sdk_concurrency as benchmark


@pytest.mark.parametrize("workload", ("get", "search", "add", "mixed"))
def test_load_case_checks_results_and_counts_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workload: str
) -> None:
    monkeypatch.setattr(benchmark, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path))
    result = benchmark.run_case(workload=workload, concurrency=2, requests=20, rows=4, dimension=64)
    assert result["successful"] == 20
    assert result["rejected"] == 0
    assert result["rejection_latency_ms"] == {"p50": None, "p95": None, "p99": None}
    assert result["stored_rows"] == {"get": 4, "search": 4, "add": 24, "mixed": 6}[workload]
    assert list(tmp_path.iterdir()) == []


def test_load_seed_can_exceed_one_public_sdk_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "TemporaryDirectory", partial(TemporaryDirectory, dir=tmp_path))
    result = benchmark.run_case(
        workload="search", concurrency=1, requests=1, rows=1_100, dimension=64
    )
    assert result["successful"] == 1
    assert result["stored_rows"] == 1_100
    assert list(tmp_path.iterdir()) == []


def test_rejections_are_excluded_from_success_metrics_and_other_failures_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = benchmark.SyntheticEmbedder(rows=1, dimension=64, latency_ms=0)
    with Memory(tmp_path, embedder=embedder) as memory:
        records = (memory.add("Synthetic memory row 0"),)

        def reject(*args: object, **kwargs: object) -> None:
            raise StorageError("overloaded", reason="capacity_exceeded", stage="admission")

        monkeypatch.setattr(memory, "search", reject)
        successful, errors, rejected, written = benchmark._measure_worker(
            memory, records, number=0, workload="search", concurrency=1, requests=3
        )
        assert successful == []
        assert errors == {"admission:capacity_exceeded": 3}
        assert len(rejected) == 3
        assert written == []

        def fail(*args: object, **kwargs: object) -> None:
            raise StorageError("broken storage", reason="io_failed")

        monkeypatch.setattr(memory, "search", fail)
        with pytest.raises(StorageError, match="broken storage"):
            benchmark._measure_worker(
                memory, records, number=0, workload="search", concurrency=1, requests=1
            )


def test_load_latency_percentiles_use_milliseconds_and_preserve_empty_samples() -> None:
    assert benchmark._latencies((0.001, 0.003)) == pytest.approx(
        {"p50": 2.0, "p95": 2.9, "p99": 2.98}
    )
    assert benchmark._latencies(()) == {"p50": None, "p95": None, "p99": None}
