"""The run identity, selection, and artifact writing every benchmark CLI shares."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest
from benchmark_deployment import write_deployment_snapshot

from mindbridge.benchmarks.artifacts import (
    MediaRunManifestBase,
    RunManifestBase,
    common_benchmark_parser,
    load_deployment_snapshot,
    media_benchmark_parser,
    predictions_document,
    select_by_id,
    sha256_text,
    sidecar_manifest_path,
    write_run_artifacts,
)

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


class _Question:
    def __init__(self, question_id: str) -> None:
        self.question_id = question_id


def test_selection_returns_everything_when_nothing_is_requested() -> None:
    questions = (_Question("a"), _Question("b"))

    assert select_by_id(questions, (), key=lambda q: q.question_id, label="ids") == questions


def test_selection_keeps_the_order_of_the_official_release() -> None:
    questions = (_Question("a"), _Question("b"), _Question("c"))

    selected = select_by_id(questions, ("c", "a"), key=lambda q: q.question_id, label="ids")

    assert [question.question_id for question in selected] == ["a", "c"]


def test_selection_rejects_unknown_and_duplicated_requests() -> None:
    questions = (_Question("a"),)

    with pytest.raises(ValueError, match="unknown"):
        select_by_id(questions, ("missing",), key=lambda q: q.question_id, label="ids")
    with pytest.raises(ValueError, match="duplicate"):
        select_by_id(questions, ("a", "a"), key=lambda q: q.question_id, label="ids")


def test_selection_reports_integer_keys_in_numeric_not_lexicographic_order() -> None:
    with pytest.raises(ValueError, match=r"unknown example IDs: 9, 10, 100$"):
        select_by_id((), (100, 9, 10), key=lambda item: 0, label="example IDs")


def test_selection_reports_string_keys_alphabetically() -> None:
    with pytest.raises(ValueError, match=r"unknown ids: a, b, c$"):
        select_by_id((), ("c", "a", "b"), key=lambda item: "", label="ids")


def test_shared_parser_supplies_the_eleven_universal_flags() -> None:
    parser = common_benchmark_parser(tenant_prefix="benchmark_locomo")

    parsed = parser.parse_args(
        [
            "--dataset",
            "data.json",
            "--output",
            "out.json",
            "--api-base-url",
            "https://memory.example.test",
            "--code-revision",
            "commit",
            "--deployment-config",
            "deployment.json",
            "--run-id",
            "run_01",
        ]
    )

    assert parsed.dataset == Path("data.json")
    assert parsed.deployment_config == Path("deployment.json")
    assert parsed.tenant_prefix == "benchmark_locomo"
    assert parsed.recall_limit == 20
    assert parsed.request_concurrency == 4
    assert parsed.request_timeout_seconds == 1_800.0
    assert parsed.overwrite is False


def test_shared_parser_is_composable_as_an_argparse_parent() -> None:
    import argparse

    parser = argparse.ArgumentParser(parents=[common_benchmark_parser(tenant_prefix="benchmark_x")])
    parser.add_argument("--split", required=True)

    parsed = parser.parse_args(
        [
            "--dataset",
            "d.json",
            "--output",
            "o.json",
            "--api-base-url",
            "u",
            "--code-revision",
            "c",
            "--deployment-config",
            "dep.json",
            "--run-id",
            "r",
            "--split",
            "day_test",
        ]
    )

    assert parsed.split == "day_test"
    assert parsed.tenant_prefix == "benchmark_x"


def test_artifacts_write_predictions_and_their_manifest_together(tmp_path: Path) -> None:
    deployment_path = write_deployment_snapshot(tmp_path, worker=False)
    output_path = tmp_path / "run" / "predictions.json"
    predictions = predictions_document([{"answer": "B"}])
    manifest = _Manifest(
        runner_version="test_v1",
        adapter_version="adapter_v1",
        code_revision="commit",
        deployment=load_deployment_snapshot(deployment_path).snapshot,
        deployment_sha256="a" * 64,
        answer_prompt_version="answer_v1",
        retrieval_task="document",
        run_id="run_01",
        tenant_prefix="benchmark_test",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        predictions_sha256=sha256_text(predictions),
        completed_at=NOW,
    )

    write_run_artifacts(output_path, predictions, manifest)

    assert json.loads(output_path.read_text(encoding="utf-8")) == [{"answer": "B"}]
    written = _Manifest.model_validate_json(
        sidecar_manifest_path(output_path).read_text(encoding="utf-8")
    )
    assert written.benchmark == "Test"
    assert written.predictions_sha256 == sha256_text(predictions)


def test_predictions_document_keeps_non_ascii_readable_and_newline_terminated() -> None:
    document = predictions_document({"answer": "中文"})

    assert "中文" in document
    assert document.endswith("\n")


class _Manifest(RunManifestBase):
    benchmark: Literal["Test"] = "Test"


def test_media_tier_adds_the_ingest_identity_to_the_shared_run_identity(tmp_path: Path) -> None:
    deployment_path = write_deployment_snapshot(tmp_path)
    manifest = _MediaManifest(
        runner_version="test_v1",
        adapter_version="adapter_v1",
        code_revision="commit",
        deployment=load_deployment_snapshot(deployment_path, require_worker=True).snapshot,
        deployment_sha256="a" * 64,
        answer_prompt_version="answer_v1",
        retrieval_task="document",
        run_id="run_01",
        tenant_prefix="benchmark_test",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        predictions_sha256="b" * 64,
        completed_at=NOW,
        annotation_sha256="c" * 64,
        device_id="test_camera",
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
    )

    assert manifest.device_id == "test_camera"
    assert manifest.run_id == "run_01"
    assert _MediaManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_media_parser_extends_the_shared_parser() -> None:
    parser = media_benchmark_parser(tenant_prefix="benchmark_test")

    parsed = parser.parse_args(
        [
            "--dataset",
            "d.json",
            "--output",
            "o.json",
            "--api-base-url",
            "u",
            "--code-revision",
            "c",
            "--deployment-config",
            "dep.json",
            "--run-id",
            "r",
            "--device-id",
            "custom_camera",
        ]
    )

    assert parsed.device_id == "custom_camera"
    assert parsed.poll_interval_seconds == 1.0
    assert parsed.processing_timeout_seconds == 1_800.0
    assert parsed.tenant_prefix == "benchmark_test"


class _MediaManifest(MediaRunManifestBase):
    benchmark: Literal["MediaTest"] = "MediaTest"
