"""Checks for committed reproducible benchmark manifests."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    PluginSnapshot,
    load_deployment_snapshot,
)
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
    assert result.embedder_plugin == "jina"


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
        "LoCoMo-Refined": 1_382,
        "M3-Bench-robot": 1_276,
        "M3-Bench-web": 3_214,
        "Video-MME": 2_700,
        "EgoLifeQA": 500,
        "EgoTempo": 500,
        "EgoMemReason": 500,
        "MEMLENS-32K": 789,
        "MM-Lifelong-day-test": 200,
        "MM-Lifelong-week-test": 200,
        "MM-Lifelong-month-train": 266,
        "MM-Lifelong-month-val": 623,
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
    assert runs["refinement_v9_uniform_l20"]["metrics"]["answer_f1"] == 0.63756575483928
    assert runs["refinement_v9_uniform_l50"]["metrics"]["answer_f1"] == 0.6531770085437091
    assert runs["refinement_v9_uniform_l50"]["metrics"]["evidence_recall"] == 0.855108877721943
    assert runs["reflection_v8"]["runner_version"] == "locomo_production_api_v6"
    assert runs["refinement_v9_uniform_l20"]["runner_version"] == "locomo_production_api_v7"
    assert runs["refinement_v9_uniform_l50"]["runner_version"] == "locomo_production_api_v7"
    assert all(
        sum(category["question_count"] for category in run["metrics"]["categories"])
        == manifest["scope"]["question_count"]
        for run in runs.values()
    )
    assert all(len(run["predictions_sha256"]) == 64 for run in runs.values())


@pytest.mark.parametrize(
    "config",
    [
        {"api_key": "must-not-be-recorded"},
        {"headers": {"Authorization": "Bearer must-not-be-recorded"}},
        {"aws_access_key": "must-not-be-recorded"},
        {"clientSecret": "must-not-be-recorded"},
        {"private_key": "must-not-be-recorded"},
    ],
)
def test_benchmark_plugin_snapshot_rejects_credentials(
    config: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValidationError, match="must not contain credentials"):
        PluginSnapshot(
            plugin="openai",
            distribution="mindbridge",
            version="0.1.0",
            config=config,
        )


def test_benchmark_plugin_snapshot_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValidationError, match="must contain JSON values"):
        PluginSnapshot(
            plugin="openai",
            distribution="mindbridge",
            version="0.1.0",
            config={"temperature": float("nan")},
        )


@pytest.mark.parametrize("name", ["OpenAI", " openai", "openai "])
def test_benchmark_plugin_snapshot_rejects_non_canonical_names(name: str) -> None:
    with pytest.raises(ValidationError, match="trimmed lowercase"):
        PluginSnapshot(plugin=name, distribution="mindbridge", version="0.1.0")


def test_benchmark_plugin_snapshot_requires_implementation_identity() -> None:
    with pytest.raises(ValidationError):
        PluginSnapshot.model_validate({"plugin": "anthropic"})


def test_raw_media_deployment_requires_a_complete_worker(tmp_path: Path) -> None:
    implementation = {"distribution": "mindbridge", "version": "0.1.0"}
    server = {
        "server_generator": {"plugin": "openai", **implementation},
        "server_embedder": {"plugin": "openai", **implementation},
    }
    with pytest.raises(ValidationError, match="must be provided together"):
        DeploymentSnapshot.model_validate(
            server | {"worker_generator": {"plugin": "openai", **implementation}}
        )

    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(server), encoding="utf-8")
    with pytest.raises(ValueError, match="raw-media benchmarks require"):
        load_deployment_snapshot(path, require_worker=True)


def test_deployment_snapshot_and_hash_use_the_same_frozen_bytes(tmp_path: Path) -> None:
    path = tmp_path / "deployment.json"
    encoded = json.dumps(
        {
            "server_generator": {
                "plugin": "openai",
                "distribution": "mindbridge",
                "version": "0.1.0",
            },
            "server_embedder": {
                "plugin": "openai",
                "distribution": "mindbridge",
                "version": "0.1.0",
            },
        }
    ).encode("utf-8")
    path.write_bytes(encoded)

    loaded = load_deployment_snapshot(path)
    path.write_text("{}", encoding="utf-8")

    assert loaded.snapshot.server_generator.plugin == "openai"
    assert loaded.sha256 == hashlib.sha256(encoded).hexdigest()
