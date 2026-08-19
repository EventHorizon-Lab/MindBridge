"""Artifact and selection checks for the reproducible SuperMemory-VQA CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.artifacts import load_deployment_snapshot
from mindbridge.benchmarks.supermemory_cli import (
    SuperMemoryRunManifest,
    _Arguments,
    _select_questions,
    _write_artifacts,
)
from mindbridge.benchmarks.supermemory_runner import (
    SuperMemoryPreparedSegment,
    SuperMemoryPreparedSubject,
    SuperMemoryPreparedVideo,
    SuperMemoryQuestionResult,
)
from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

NOW = datetime(2026, 3, 10, tzinfo=timezone.utc)


def test_supermemory_artifacts_pin_inputs_models_metrics_and_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "all_qa.json"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.json"
    question = _question(1)
    prepared = _prepared()
    result = _result(1)

    arguments = _arguments(dataset_path, prepared_path, output_path)
    _write_artifacts(
        arguments,
        (question,),
        prepared,
        (result,),
        load_deployment_snapshot(arguments.deployment_config_path, require_worker=True),
    )

    predictions = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = SuperMemoryRunManifest.model_validate_json(
        output_path.with_suffix(".json.manifest.json").read_text(encoding="utf-8")
    )
    assert predictions["metrics"]["qa_accuracy"] == 1.0
    assert predictions["results"][0]["predicted_option_index"] == 1
    assert manifest.dataset_revision == "dataset-revision"
    assert manifest.source_revision == "source-revision"
    assert manifest.deployment.server_generator.config["model_revision"] == ("answer-fingerprint")
    assert manifest.request_timeout_seconds == 1_800.0
    assert manifest.run_id == "run_01"
    assert manifest.metrics.answerability_f1 == 1.0
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()


def test_supermemory_selection_is_subject_scoped_and_fail_closed() -> None:
    questions = (_question(1), _question(2, subject=2))
    assert _select_questions(questions, 1, ()) == (questions[0],)
    with pytest.raises(ValueError, match="unknown"):
        _select_questions(questions, 1, (2,))


def _arguments(dataset_path: Path, prepared_path: Path, output_path: Path) -> _Arguments:
    deployment_path = dataset_path.parent / "deployment.json"
    _write_deployment(deployment_path)
    return _Arguments(
        dataset_path=dataset_path,
        prepared_media_path=prepared_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        subject=1,
        dataset_revision="dataset-revision",
        source_revision="source-revision",
        code_revision="mindbridge-commit",
        deployment_config_path=deployment_path,
        run_id="run_01",
        tenant_prefix="benchmark_supermemory",
        device_id="supermemory_glasses",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
        question_ids=(),
        overwrite=False,
        quiet=True,
    )


def _write_deployment(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "server_generator": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"model_revision": "answer-fingerprint"},
                },
                "server_embedder": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {},
                },
                "worker_generator": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {"model_revision": "perception-fingerprint"},
                },
                "worker_media_embedder": {
                    "plugin": "jina",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {},
                },
                "worker_text_embedder": {
                    "plugin": "openai",
                    "distribution": "mindbridge",
                    "version": "0.1.0",
                    "config": {},
                },
            }
        ),
        encoding="utf-8",
    )


def _question(question_id: int, *, subject: int = 1) -> SuperMemoryQuestion:
    return SuperMemoryQuestion(
        question_id=question_id,
        subject=subject,
        question="Where is the mug?",
        choices=(
            "This question can not be answered.",
            "In the sink",
            "On the table",
            "In a drawer",
        ),
        correct_option_index=1,
        unanswerable_option_index=0,
        is_answerable=True,
        skill="object_location_memory",
        source_video_ids=(f"Person_{subject}_session_1",),
        question_video_id=f"Person_{subject}_session_1",
        question_ended_at=NOW,
    )


def _prepared() -> SuperMemoryPreparedSubject:
    return SuperMemoryPreparedSubject(
        subject=1,
        videos=(
            SuperMemoryPreparedVideo(
                video_id="Person_1_session_1",
                started_at=NOW,
                segments=(
                    SuperMemoryPreparedSegment(
                        start_seconds=0,
                        duration_ms=30_000,
                        media_objects=(
                            MediaObjectInput(
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
                ),
            ),
        ),
    )


def _result(question_id: int) -> SuperMemoryQuestionResult:
    return SuperMemoryQuestionResult(
        question_id=question_id,
        predicted_option_index=1,
        ranked_option_indices=(1, 2, 3, 0),
        model_answer="B, C, D, A",
        mindbridge_confidence=0.9,
        mindbridge_memory_ids=("memory_01",),
        mindbridge_evidence_ids=("evidence_01",),
        mindbridge_trace_id="trace_01",
    )
