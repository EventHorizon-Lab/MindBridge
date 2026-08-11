"""Checks for committed reproducible benchmark manifests."""

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
    }
