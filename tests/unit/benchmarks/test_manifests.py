"""Checks for committed reproducible benchmark manifests."""

from pathlib import Path

from mindbridge.benchmarks.jina_smoke import JinaSmokeResult


def test_jina_smoke_manifest_matches_current_schema() -> None:
    """A committed model result cannot drift from its executable schema."""
    manifest_path = (
        Path(__file__).parents[3] / "benchmarks" / "manifests" / "jina-omni-small-smoke.json"
    )

    result = JinaSmokeResult.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    assert result.passed is True
    assert result.dimension == 1_024
