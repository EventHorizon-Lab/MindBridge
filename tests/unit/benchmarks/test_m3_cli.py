"""Artifact and selection checks for the reproducible M3-Bench CLI."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.benchmarks.m3_bench import M3BenchQuestion, M3BenchVideo
from mindbridge.benchmarks.m3_cli import (
    M3RunManifest,
    _Arguments,
    _prepared_by_video,
    _select_videos,
    _validate_subset,
    _write_artifacts,
)
from mindbridge.benchmarks.m3_runner import (
    M3OfficialQuestionResult,
    M3PreparedClip,
    M3PreparedVideo,
    load_prepared_m3,
)
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind
from mindbridge.models.jina import DEFAULT_JINA_OMNI_MODEL_ID, DEFAULT_JINA_OMNI_REVISION

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_m3_artifacts_pin_media_models_code_and_jsonl_output(tmp_path: Path) -> None:
    dataset_path = tmp_path / "robot.json"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.jsonl"
    video = _video()
    prepared = _prepared()
    result = M3OfficialQuestionResult(
        id="video_01_Q01",
        question="What happened?",
        answer="A person entered",
        type=("Temporal",),
        timestamp_seconds=10,
        before_clip=0,
        response="A person entered",
        mindbridge_confidence=0.9,
        mindbridge_memory_ids=("memory_01",),
        mindbridge_evidence_ids=("evidence_01",),
        mindbridge_trace_id="trace_01",
    )

    _write_artifacts(
        _arguments(dataset_path, prepared_path, output_path),
        (video,),
        {video.video_id: prepared},
        (result,),
    )

    prediction = json.loads(output_path.read_text(encoding="utf-8"))
    manifest = M3RunManifest.model_validate_json(
        output_path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert prediction["id"] == "video_01_Q01"
    assert prediction["response"] == "A person entered"
    assert manifest.source_revision == "official-revision"
    assert manifest.media_revision == "official-media-revision"
    assert manifest.perception_model_revision == "perception-serving-fingerprint"
    assert manifest.answer_model_revision == "answer-serving-fingerprint"
    assert manifest.run_id == "run_01"
    assert manifest.clip_count == 1
    assert manifest.question_count == 1
    assert manifest.predictions_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()


def test_m3_media_manifest_and_selection_fail_closed(tmp_path: Path) -> None:
    video = _video()
    assert _select_videos((video,), ("video_01",)) == (video,)
    _validate_subset((video,), "robot")
    with pytest.raises(ValueError, match="web subset"):
        _validate_subset((video,), "web")
    with pytest.raises(ValueError, match="unknown"):
        _select_videos((video,), ("missing",))
    with pytest.raises(ValueError, match="missing prepared"):
        _prepared_by_video((video,), ())

    empty_manifest = tmp_path / "empty.json"
    empty_manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        load_prepared_m3(empty_manifest)


def test_m3_artifacts_reject_missing_predictions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="question order"):
        _write_artifacts(
            _arguments(tmp_path / "robot.json", tmp_path / "prepared.json", tmp_path / "out"),
            (_video(),),
            {"video_01": _prepared()},
            (),
        )


def _arguments(dataset_path: Path, prepared_path: Path, output_path: Path) -> _Arguments:
    return _Arguments(
        dataset_path=dataset_path,
        prepared_media_path=prepared_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        subset="robot",
        source_revision="official-revision",
        media_revision="official-media-revision",
        code_revision="mindbridge-commit",
        perception_model_id="qwen3.8-max",
        perception_model_revision="perception-serving-fingerprint",
        answer_model_id="qwen3.8-max",
        answer_model_revision="answer-serving-fingerprint",
        embedding_model_id=DEFAULT_JINA_OMNI_MODEL_ID,
        embedding_model_revision=DEFAULT_JINA_OMNI_REVISION,
        run_id="run_01",
        tenant_prefix="benchmark_m3",
        device_id="m3_bench_camera",
        recall_limit=20,
        request_concurrency=4,
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
        video_ids=(),
        overwrite=False,
    )


def _video() -> M3BenchVideo:
    return M3BenchVideo(
        video_id="video_01",
        video_path="data/videos/robot/video_01.mp4",
        questions=(
            M3BenchQuestion(
                question_id="video_01_Q01",
                question="What happened?",
                reference_answer="A person entered",
                question_types=("Temporal",),
                timestamp_seconds=10,
                before_clip_index=0,
            ),
        ),
    )


def _prepared() -> M3PreparedVideo:
    return M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(
            M3PreparedClip(
                clip_index=0,
                media_object=MediaObjectInput(
                    media_object_id="media_01",
                    kind=MediaKind.VIDEO,
                    uri="s3://benchmark/tenants/benchmark_m3_video_01/clip-0.mp4",
                    sha256="a" * 64,
                    size_bytes=1_024,
                    created_at=NOW,
                    duration_ms=30_000,
                ),
            ),
        ),
    )
