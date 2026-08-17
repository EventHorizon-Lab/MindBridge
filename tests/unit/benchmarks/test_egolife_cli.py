"""Artifact and selection checks for the reproducible EgoLifeQA CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from benchmark_deployment import SERVER_GENERATOR_REVISION, write_deployment_snapshot

import mindbridge.benchmarks.egolife_cli as egolife_cli
from mindbridge.benchmarks.artifacts import load_deployment_snapshot, select_by_id
from mindbridge.benchmarks.egolife_cli import (
    EgoLifeRunManifest,
    _Arguments,
    _write_artifacts,
)
from mindbridge.benchmarks.egolife_qa import EgoLifeQuestion
from mindbridge.benchmarks.egolife_runner import (
    EgoLifePreparedClip,
    EgoLifePreparedStream,
    EgoLifeQuestionResult,
)
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def test_egolife_validates_deployment_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text("{}", encoding="utf-8")
    arguments = _arguments(
        tmp_path / "qa.json",
        tmp_path / "prepared.json",
        tmp_path / "predictions.json",
        deployment_path,
    )
    monkeypatch.setattr(egolife_cli, "_parse_arguments", lambda: arguments)
    monkeypatch.setattr(egolife_cli, "load_egolife_qa", lambda _path: (_question(),))
    monkeypatch.setattr(egolife_cli, "load_prepared_egolife", lambda _path: _prepared())

    async def fail_if_run(*_arguments: object) -> None:
        raise AssertionError("inference started before deployment validation")

    monkeypatch.setattr(egolife_cli, "_run", fail_if_run)

    with pytest.raises(ValueError):
        egolife_cli.main()


def test_egolife_artifacts_pin_inputs_models_metrics_and_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "qa.json"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.json"
    question = _question()
    prepared = _prepared()
    result = _result()

    arguments = _arguments(
        dataset_path, prepared_path, output_path, write_deployment_snapshot(tmp_path)
    )
    _write_artifacts(
        arguments,
        (question,),
        prepared,
        (result,),
        load_deployment_snapshot(arguments.deployment_config_path, require_worker=True),
    )

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = EgoLifeRunManifest.model_validate_json(
        output_path.with_suffix(".json.manifest.json").read_text(encoding="utf-8")
    )
    assert predictions["accuracy"] == 1.0
    assert predictions["results"][0]["model_option"] == "B"
    assert manifest.dataset_revision == "dataset-revision"
    assert manifest.evaluator_revision == "evaluator-revision"
    assert manifest.deployment.server_generator.config["model_revision"] == (
        SERVER_GENERATOR_REVISION
    )
    assert manifest.request_timeout_seconds == 1_800.0
    assert manifest.run_id == "run_01"
    assert manifest.metrics.correct_count == 1
    assert manifest.media_clip_count == 1
    assert manifest.caption_clip_count == 0
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()


def test_egolife_selection_rejects_unknown_question() -> None:
    question = _question()
    assert select_by_id((question,), ("1",), key=lambda item: item.question_id, label="ids") == (
        question,
    )
    with pytest.raises(ValueError, match="unknown"):
        select_by_id((question,), ("missing",), key=lambda item: item.question_id, label="ids")


def _arguments(
    dataset_path: Path, prepared_path: Path, output_path: Path, deployment_path: Path
) -> _Arguments:
    return _Arguments(
        dataset_path=dataset_path,
        prepared_media_path=prepared_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        dataset_revision="dataset-revision",
        evaluator_revision="evaluator-revision",
        code_revision="mindbridge-commit",
        deployment_config_path=deployment_path,
        run_id="run_01",
        tenant_prefix="benchmark_egolife",
        device_id="egolife_camera",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
        question_ids=(),
        overwrite=False,
    )


def _question() -> EgoLifeQuestion:
    return EgoLifeQuestion(
        question_id="1",
        question="Who used it?",
        choices=("Tasha", "Alice", "Shure", "Lucia"),
        correct_option="B",
        query_day=1,
        query_timecode="11210217",
        query_offset_ms=40_862_850,
        question_type="EntityLog",
        needs_audio=False,
        needs_name=True,
        asks_last_time=False,
    )


def _prepared() -> EgoLifePreparedStream:
    return EgoLifePreparedStream(
        subject_id="A1_JAKE",
        timeline_origin=NOW,
        clips=(
            EgoLifePreparedClip(
                day=1,
                start_timecode="11210000",
                media_object=MediaObjectInput(
                    media_object_id="media_01",
                    kind=MediaKind.VIDEO,
                    uri="s3://benchmark/media_01.mp4",
                    sha256="a" * 64,
                    size_bytes=1_024,
                    created_at=NOW,
                    duration_ms=30_000,
                ),
            ),
        ),
    )


def _result() -> EgoLifeQuestionResult:
    return EgoLifeQuestionResult(
        id="1",
        subject_id="A1_JAKE",
        question="Who used it?",
        answer="B",
        model_option="B",
        model_answer="B",
        question_type="EntityLog",
        query_day=1,
        query_timecode="11210217",
        mindbridge_confidence=0.9,
        mindbridge_memory_ids=("memory_01",),
        mindbridge_evidence_ids=("evidence_01",),
        mindbridge_trace_id="trace_01",
    )
