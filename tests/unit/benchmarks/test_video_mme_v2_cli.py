"""Video-MME-v2 run scoping and the disclosure that keeps its rating quotable."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from benchmark_deployment import write_deployment_snapshot

from mindbridge.benchmarks.artifacts import load_deployment_snapshot
from mindbridge.benchmarks.runtime import PreparedVideo, PreparedVideoSegment
from mindbridge.benchmarks.video_mme_v2 import (
    VideoMMEV2Group,
    VideoMMEV2GroupResult,
    VideoMMEV2GroupType,
    VideoMMEV2Question,
    VideoMMEV2QuestionResult,
)
from mindbridge.benchmarks.video_mme_v2_cli import (
    VideoMMEV2RunManifest,
    _Arguments,
    _select_groups,
    _select_prepared,
    _write_artifacts,
)
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def test_selection_scopes_a_run_to_whole_groups() -> None:
    groups = (_group("001", "logic"), _group("002", "relevance"))

    assert _select_groups(groups, ("001",), (), None) == (groups[0],)
    assert _select_groups(groups, (), ("relevance",), None) == (groups[1],)
    assert _select_groups(groups, (), (), None) == groups
    with pytest.raises(ValueError, match="no Video-MME-v2 groups"):
        _select_groups((groups[0],), (), ("relevance",), None)


def test_selecting_an_unknown_video_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError, match="unknown Video-MME-v2 video IDs: 404"):
        _select_groups((_group("001", "logic"),), ("404",), (), None)


def test_prepared_media_must_cover_every_selected_group() -> None:
    """A group missing its video would otherwise score zero and look like a wrong answer."""
    with pytest.raises(ValueError, match="missing prepared Video-MME-v2 videos: 002"):
        _select_prepared(
            (_prepared("001", transcript=None),),
            (_group("001", "logic"), _group("002", "relevance")),
        )


def test_manifest_records_the_group_denominator_and_both_scoring_views(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.parquet"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    output_path = tmp_path / "run" / "predictions.json"
    arguments = _arguments(
        dataset_path, prepared_path, output_path, write_deployment_snapshot(tmp_path)
    )

    _write_artifacts(
        arguments,
        (_group("001", "logic"),),
        (_prepared("001", transcript=None),),
        (_result("001", "logic", correct=(True, True, False, False)),),
        load_deployment_snapshot(arguments.deployment_config_path, require_worker=True),
    )

    manifest = VideoMMEV2RunManifest.model_validate_json(
        output_path.with_suffix(".json.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.benchmark == "Video-MME-v2"
    assert manifest.transcript_source == "none"
    assert manifest.group_count == 1
    assert manifest.group_types == ("logic",)
    # Two of four correct on a linear chain: half the questions, a quarter of the score.
    assert manifest.metrics.accuracy.overall == pytest.approx(50.0)
    assert manifest.metrics.rating.overall == pytest.approx(25.0)
    assert manifest.metrics.rating.by_level == {"level_1": pytest.approx(25.0)}


def test_predictions_out_of_annotation_order_are_refused(tmp_path: Path) -> None:
    """The released scorer reads groups positionally, so order is part of the artifact."""
    dataset_path = tmp_path / "test.parquet"
    dataset_path.write_text("official-annotation", encoding="utf-8")
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("uploaded-media", encoding="utf-8")
    arguments = _arguments(
        dataset_path,
        prepared_path,
        tmp_path / "run" / "predictions.json",
        write_deployment_snapshot(tmp_path),
    )

    with pytest.raises(ValueError, match="must match annotation group order"):
        _write_artifacts(
            arguments,
            (_group("001", "logic"), _group("002", "relevance")),
            (_prepared("001", transcript=None), _prepared("002", transcript=None)),
            (
                _result("002", "relevance", correct=(True,) * 4),
                _result("001", "logic", correct=(True,) * 4),
            ),
            load_deployment_snapshot(arguments.deployment_config_path, require_worker=True),
        )


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
        tenant_prefix="benchmark_video_mme_v2",
        device_id="video_mme_v2_camera",
        recall_limit=20,
        request_concurrency=4,
        unit_concurrency=1,
        request_timeout_seconds=1_800.0,
        limit=None,
        poll_interval_seconds=1.0,
        processing_timeout_seconds=1_800.0,
        video_ids=(),
        group_types=(),
        transcript_source="none",
        overwrite=False,
        predict_only=False,
        quiet=True,
    )


def _group(video_id: str, group_type: VideoMMEV2GroupType) -> VideoMMEV2Group:
    return VideoMMEV2Group(
        video_id=video_id,
        source_url="https://www.youtube.com/watch?v=example",
        group_type=group_type,
        group_structure="[1,2,3,4]" if group_type == "logic" else "4",
        questions=tuple(
            VideoMMEV2Question(
                question_id=f"{video_id}-{position}",
                position=position,
                question=f"What happened at step {position}?",
                options=tuple(f"{label}. Option {label}." for label in "ABCDEFGH"),
                answer="B",
                level="1",
                second_head="Order",
                third_head="Causal Reasoning",
            )
            for position in range(1, 5)
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
    video_id: str, group_type: VideoMMEV2GroupType, *, correct: tuple[bool, ...]
) -> VideoMMEV2GroupResult:
    return VideoMMEV2GroupResult(
        video_id=video_id,
        group_type=group_type,
        group_structure="[1,2,3,4]" if group_type == "logic" else "4",
        questions=tuple(
            VideoMMEV2QuestionResult(
                question_id=f"{video_id}-{position}",
                position=position,
                question=f"What happened at step {position}?",
                options=tuple(f"{label}. Option {label}." for label in "ABCDEFGH"),
                answer="B",
                level="1",
                second_head="Order",
                third_head="Causal Reasoning",
                response="B" if hit else "C",
                mindbridge_model_answer="B" if hit else "C",
                mindbridge_confidence=0.9,
                mindbridge_memory_ids=("memory_01",),
                mindbridge_evidence_ids=("evidence_01",),
                mindbridge_trace_id=f"trace_{video_id}_{position}",
            )
            for position, hit in enumerate(correct, start=1)
        ),
    )
