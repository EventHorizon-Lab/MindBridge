"""Reproducible M3-Bench runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    MediaBenchmarkRunManifest,
    add_media_arguments,
    connected_memory,
    core_parser,
    index_prepared,
    media_arguments,
    media_manifest,
    predictions_jsonl,
    write_run_artifacts,
)
from mindbridge.benchmarks.m3_bench import M3_BENCH_ADAPTER_VERSION, M3BenchVideo, load_m3_bench
from mindbridge.benchmarks.m3_runner import (
    M3_CLIP_DURATION_SECONDS,
    M3OfficialQuestionResult,
    M3PreparedVideo,
    load_prepared_m3,
    run_m3_video,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

M3_RUNNER_VERSION = "m3_production_api_v9"


class M3RunManifest(MediaBenchmarkRunManifest):
    """Immutable data, deployment, code, and output identity for one M3 run."""

    benchmark: Literal["M3-Bench"] = "M3-Bench"
    subset: Literal["robot", "web"]
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    media_repository: NonEmptyString
    media_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    clip_duration_seconds: int = Field(gt=0)
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)
    question_count: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    subset: Literal["robot", "web"]
    source_revision: str
    media_revision: str
    video_ids: tuple[str, ...]


def main() -> None:
    """Run selected official videos and emit JSONL predictions plus a manifest."""
    arguments = _parse_arguments()
    videos = _select_videos(load_m3_bench(arguments.dataset_path), arguments.video_ids)
    _validate_subset(videos, arguments.subset)
    prepared = _prepared_by_video(videos, load_prepared_m3(arguments.prepared_media_path))
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            clip.media_object is not None
            for video in videos
            for clip in prepared[video.video_id].clips
        ),
    )
    results = asyncio.run(_run_videos(arguments, videos, prepared))
    _write_artifacts(arguments, videos, prepared, results, deployment)


async def _run_videos(
    arguments: _Arguments,
    videos: tuple[M3BenchVideo, ...],
    prepared: dict[str, M3PreparedVideo],
) -> tuple[M3OfficialQuestionResult, ...]:
    async with connected_memory(arguments) as memory:
        results: list[M3OfficialQuestionResult] = []
        for video in videos:
            results.extend(
                await run_m3_video(
                    memory,
                    video,
                    prepared[video.video_id],
                    run_id=arguments.run_id,
                    tenant_prefix=arguments.tenant_prefix,
                    device_id=arguments.device_id,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                    processing_timeout_seconds=arguments.processing_timeout_seconds,
                )
            )
        return tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    videos: tuple[M3BenchVideo, ...],
    prepared: dict[str, M3PreparedVideo],
    results: tuple[M3OfficialQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    expected_ids = tuple(question.question_id for video in videos for question in video.questions)
    if tuple(result.id for result in results) != expected_ids:
        raise ValueError("M3-Bench predictions must match annotation question order")
    media_clip_count = sum(
        clip.media_object is not None for video in videos for clip in prepared[video.video_id].clips
    )
    predictions = predictions_jsonl(results)
    manifest = media_manifest(
        M3RunManifest,
        arguments,
        deployment,
        runner_version=M3_RUNNER_VERSION,
        adapter_version=M3_BENCH_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        subset=arguments.subset,
        source_repository="ByteDance-Seed/m3-agent",
        source_revision=arguments.source_revision,
        media_repository="ByteDance-Seed/M3-Bench",
        media_revision=arguments.media_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        clip_duration_seconds=M3_CLIP_DURATION_SECONDS,
        video_ids=tuple(video.video_id for video in videos),
        clip_count=sum(len(prepared[video.video_id].clips) for video in videos),
        media_clip_count=media_clip_count,
        caption_clip_count=sum(
            clip.caption is not None for video in videos for clip in prepared[video.video_id].clips
        ),
        question_count=sum(len(video.questions) for video in videos),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_videos(
    videos: tuple[M3BenchVideo, ...],
    video_ids: tuple[str, ...],
) -> tuple[M3BenchVideo, ...]:
    if not video_ids:
        return videos
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video IDs must not contain duplicates")
    requested = set(video_ids)
    selected = tuple(video for video in videos if video.video_id in requested)
    missing = requested - {video.video_id for video in selected}
    if missing:
        raise ValueError(f"unknown M3-Bench video IDs: {', '.join(sorted(missing))}")
    return selected


def _prepared_by_video(
    videos: tuple[M3BenchVideo, ...],
    prepared: tuple[M3PreparedVideo, ...],
) -> dict[str, M3PreparedVideo]:
    return index_prepared(
        (video.video_id for video in videos),
        prepared,
        key=lambda video: video.video_id,
        label="M3-Bench videos",
    )


def _validate_subset(videos: tuple[M3BenchVideo, ...], subset: Literal["robot", "web"]) -> None:
    expected = (True, True) if subset == "robot" else (False, False)
    timing_shapes = {
        (question.timestamp_seconds is not None, question.before_clip_index is not None)
        for video in videos
        for question in video.questions
    }
    if timing_shapes != {expected}:
        raise ValueError(f"M3-Bench annotations do not match the {subset} subset")


def _parse_arguments() -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_m3"),
        device_id="m3_bench_camera",
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--subset", choices=("robot", "web"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--media-revision", required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parsed = parser.parse_args()
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        subset=cast(Literal["robot", "web"], parsed.subset),
        source_revision=parsed.source_revision,
        media_revision=parsed.media_revision,
        video_ids=tuple(parsed.video_id),
    )


if __name__ == "__main__":
    main()
