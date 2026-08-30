"""Small checks for the standalone local index benchmark."""

from __future__ import annotations

import json
import math
import random
import struct
from pathlib import Path
from typing import cast

import pytest

from mindbridge.benchmarks.local_index_benchmark import (
    _normalized_fp32,
    main,
    run_benchmark,
)
from mindbridge.infrastructure.local import LocalStore


def test_fp32_vectors_are_deterministic_and_normalized() -> None:
    fp32 = struct.Struct("<8f")
    first = _normalized_fp32(random.Random(7), fp32)
    second = _normalized_fp32(random.Random(7), fp32)

    assert first == second
    assert fp32.pack(*first) == fp32.pack(*second)
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0, abs=1e-6)


def test_small_benchmark_reports_valid_metrics(tmp_path: Path) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.7 native wheel is not installed")
    result = run_benchmark(tmp_path, rows=32, dimension=8, queries=4, k=3, seed=7)

    assert set(result) == {
        "rows",
        "dimension",
        "queries",
        "k",
        "seed",
        "quantization",
        "ingest_seconds",
        "optimize_seconds",
        "recall_at_k",
        "query_latency_ms",
        "query_qps",
        "disk_bytes",
    }
    assert result["rows"] == 32
    assert result["dimension"] == 8
    assert result["queries"] == 4
    assert result["k"] == 3
    assert result["seed"] == 7
    assert result["quantization"] == "none"
    assert cast(float, result["ingest_seconds"]) > 0.0
    assert cast(float, result["optimize_seconds"]) >= 0.0
    assert 0.0 <= cast(float, result["recall_at_k"]) <= 1.0
    assert cast(float, result["query_qps"]) > 0.0

    latency = cast(dict[str, float], result["query_latency_ms"])
    assert set(latency) == {"p50", "p95", "p99"}
    assert 0.0 <= latency["p50"] <= latency["p95"] <= latency["p99"]
    disk = cast(dict[str, int], result["disk_bytes"])
    assert set(disk) == {"sqlite", "zvec", "total"}
    assert disk["sqlite"] > 0
    assert disk["zvec"] > 0
    assert disk["total"] == disk["sqlite"] + disk["zvec"]
    json.dumps(result)

    with LocalStore(tmp_path) as store:
        assert store.pending_index_operations() == ()


def test_cli_prints_one_json_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("zvec", reason="Zvec 0.7 native wheel is not installed")
    data_dir = tmp_path / "cli"

    assert (
        main(
            [
                "--rows",
                "16",
                "--dimension",
                "4",
                "--queries",
                "2",
                "--k",
                "2",
                "--seed",
                "3",
                "--data-dir",
                str(data_dir),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == 16
    assert report["quantization"] == "none"
    assert 0.0 <= report["recall_at_k"] <= 1.0


def test_benchmark_rejects_invalid_or_dirty_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="k must not exceed rows"):
        run_benchmark(tmp_path / "invalid", rows=1, dimension=2, queries=1, k=2)

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        run_benchmark(dirty, rows=1, dimension=2, queries=1, k=1)
