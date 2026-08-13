"""Checks for committed reproducible benchmark manifests."""

import json
from pathlib import Path

from mindbridge.benchmarks.dataset_smoke import DatasetAdapterSmokeResult
from mindbridge.benchmarks.jina_smoke import JinaSmokeResult


def test_jina_smoke_manifest_matches_current_schema() -> None:
    """A committed model result cannot drift from its executable schema."""
    manifest_path = (
        Path(__file__).parents[3] / "benchmarks" / "manifests" / "jina-omni-small-smoke.json"
    )

    result = JinaSmokeResult.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    assert result.passed is True
    assert result.dimension == 1_024


def test_dataset_adapter_manifest_matches_current_schema() -> None:
    """Pinned official annotation counts make upstream schema drift visible."""
    manifest_path = (
        Path(__file__).parents[3] / "benchmarks" / "manifests" / "dataset-adapters-smoke.json"
    )

    result = DatasetAdapterSmokeResult.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )

    assert result.passed is True
    assert {dataset.benchmark: dataset.question_count for dataset in result.datasets} == {
        "LoCoMo": 1_986,
        "M3-Bench-robot": 1_276,
        "M3-Bench-web": 3_214,
        "EgoLifeQA": 500,
        "SuperMemory-VQA": 4_853,
    }


def test_locomo_optimization_manifest_preserves_reported_category_metrics() -> None:
    manifest_path = (
        Path(__file__).parents[3] / "benchmarks" / "manifests" / "locomo-conv-26-optimization.json"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = {run["name"]: run for run in manifest["runs"]}

    assert manifest["dataset"]["revision"] == ("3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376")
    assert runs["baseline"]["metrics"]["answer_f1"] == 0.5459446372710516
    assert runs["optimized"]["metrics"]["answer_f1"] == 0.6136889267575371
    assert runs["reflection_v8"]["metrics"]["answer_f1"] == 0.649383027799884
    assert runs["reflection_v8"]["metrics"]["evidence_recall"] == 0.7772194304857621
    assert all(
        sum(category["question_count"] for category in run["metrics"]["categories"])
        == manifest["scope"]["question_count"]
        for run in runs.values()
    )
    assert all(len(run["predictions_sha256"]) == 64 for run in runs.values())
