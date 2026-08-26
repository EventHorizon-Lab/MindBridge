"""Video-MME long-subset scoping and the disclosure that keeps a run comparable."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from benchmark_deployment import write_deployment_snapshot
from pydantic import ValidationError

from mindbridge.benchmarks.artifacts import load_deployment_snapshot
from mindbridge.benchmarks.cli_common import require_declared_transcripts
from mindbridge.benchmarks.runtime import PreparedVideo, PreparedVideoSegment
from mindbridge.benchmarks.video_mme import (
    VideoMMEDuration,
    VideoMMEMetrics,
    VideoMMEQuestion,
    VideoMMEQuestionResult,
    VideoMMEVideo,
    VideoMMEVideoResult,
    evaluate_video_mme,
)
from mindbridge.benchmarks.video_mme_cli import (
    VideoMMERunManifest,
    _Arguments,
    _select_videos,
    _write_artifacts,
)
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_metrics_break_out_the_long_subset_reported_by_the_leaderboard() -> None:
    metrics = evaluate_video_mme(
        (
            _result("long_1", "long", correct=True),
            _result("long_2", "long", correct=False),
            _result("short_1", "short", correct=True),
        )
    )

    assert metrics.question_count == 3
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.by_duration["long"].question_count == 2
    assert metrics.by_duration["long"].accuracy == pytest.approx(0.5)
    assert metrics.by_duration["short"].accuracy == pytest.approx(1.0)
    assert "medium" not in metrics.by_duration


def test_duration_breakdown_does_not_nest_further() -> None:
    metrics = evaluate_video_mme((_result("long_1", "long", correct=True),))

    assert metrics.by_duration["long"].by_duration == {}


def test_selection_scopes_a_run_to_one_official_duration() -> None:
    videos = (_video("long_1", "long"), _video("short_1", "short"))

    assert _select_videos(videos, (), ("long",), None) == (videos[0],)
    assert _select_videos(videos, (), (), None) == videos
    with pytest.raises(ValueError, match="no Video-MME videos"):
        _select_videos((videos[1],), (), ("long",), None)


def test_a_limit_applies_after_the_duration_filter_not_before_it() -> None:
    """Applied before, `--limit 1 --duration long` truncates to a short video and keeps none."""
    videos = (_video("short_1", "short"), _video("long_1", "long"))

    assert _select_videos(videos, (), ("long",), 1) == (videos[1],)
    assert _select_videos(videos, (), (), 1) == (videos[0],)


def test_declaring_no_subtitles_while_feeding_transcripts_is_refused() -> None:
    with_transcript = (_prepared("long_1", transcript="spoken words"),)
    without_transcript = (_prepared("long_1", transcript=None),)

    require_declared_transcripts(without_transcript, "none")
    require_declared_transcripts(with_transcript, "official_subtitles")
    with pytest.raises(ValueError, match="no transcript"):
        require_declared_transcripts(with_transcript, "none")
    with pytest.raises(ValueError, match="carries no transcript"):
        require_declared_transcripts(without_transcript, "asr")


def test_manifest_records_the_transcript_source_behind_the_number(tmp_path: Path) -> None:
    dataset_path = tmp_path / "videomme.parquet"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.json"
    arguments = _arguments(
        dataset_path, prepared_path, output_path, write_deployment_snapshot(tmp_path)
    )

    _write_artifacts(
        arguments,
        (_video("long_1", "long"),),
        (_prepared("long_1", transcript=None),),
        (_result("long_1", "long", correct=True),),
        load_deployment_snapshot(arguments.deployment_config_path, require_worker=True),
    )

    manifest = VideoMMERunManifest.model_validate_json(
        output_path.with_suffix(".json.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.transcript_source == "none"
    assert manifest.durations == ("long",)
    assert manifest.metrics.by_duration["long"].accuracy == pytest.approx(1.0)


def _arguments(
    dataset_path: Path,
    prepared_path: Path,
    output_path: Path,
    deployment_path: Path,
) -> _Arguments:
    return _Arguments(
        dataset_path=dataset_path,
        prepared_media_path=prepared_path,
        output_path=output_path,
        api_base_url="https://memory.example.test",
        deployment_config_path=deployment_path,
        run_id="run_01",
        tenant_prefix="benchmark_video_mme",
        device_id="video_mme_camera",
        recall_limit=20,
        request_concurrency=4,
        request_timeout_seconds=1_800.0,
        limit=None,
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
        video_ids=(),
        durations=("long",),
        transcript_source="none",
        overwrite=False,
        predict_only=False,
        quiet=True,
    )


def _video(video_id: str, duration: VideoMMEDuration) -> VideoMMEVideo:
    return VideoMMEVideo(
        video_id=video_id,
        duration=duration,
        domain="Knowledge",
        sub_category="Humanity",
        source_url="https://www.youtube.com/watch?v=example",
        source_video_id=f"source_{video_id}",
        questions=(
            VideoMMEQuestion(
                question_id=f"{video_id}_q1",
                task_type="Temporal Reasoning",
                question="What happened first?",
                options=("A. one", "B. two", "C. three", "D. four"),
                answer="B",
            ),
        ),
    )


def _prepared(video_id: str, *, transcript: str | None) -> PreparedVideo:
    return PreparedVideo(
        video_id=video_id,
        timeline_origin=NOW,
        segments=(
            PreparedVideoSegment(
                segment_id=f"{video_id}_s1",
                start_seconds=0.0,
                duration_ms=30_000,
                media_objects=(
                    MediaObjectInput(
                        media_object_id=f"media_{video_id}",
                        kind=MediaKind.VIDEO,
                        uri=f"s3://benchmark/{video_id}.mp4",
                        sha256="a" * 64,
                        size_bytes=1_024,
                        created_at=NOW,
                        duration_ms=30_000,
                    ),
                ),
                transcript=transcript,
            ),
        ),
    )


def _result(
    video_id: str,
    duration: VideoMMEDuration,
    *,
    correct: bool,
) -> VideoMMEVideoResult:
    return VideoMMEVideoResult(
        video_id=video_id,
        duration=duration,
        domain="Knowledge",
        sub_category="Humanity",
        questions=(
            VideoMMEQuestionResult(
                question_id=f"{video_id}_q1",
                task_type="Temporal Reasoning",
                question="What happened first?",
                options=("A. one", "B. two", "C. three", "D. four"),
                answer="B",
                response="B" if correct else "C",
                mindbridge_model_answer="B" if correct else "C",
                mindbridge_confidence=0.9,
                mindbridge_memory_ids=("memory_01",),
                mindbridge_evidence_ids=("evidence_01",),
                mindbridge_trace_id=f"trace_{video_id}",
            ),
        ),
    )


def test_duration_cells_are_validated_rather_than_copied_in_unchecked() -> None:
    metrics = evaluate_video_mme(
        (_result("long_1", "long", correct=True), _result("short_1", "short", correct=False))
    )

    assert VideoMMEMetrics.model_validate_json(metrics.model_dump_json()) == metrics
    with pytest.raises(ValidationError, match="cover every scored question"):
        metrics.model_copy(deep=True).__class__(
            question_count=2,
            answered_count=2,
            correct_count=1,
            error_count=0,
            accuracy=0.5,
            strict_accuracy=0.5,
            by_duration={"long": metrics.by_duration["long"]},
        )
