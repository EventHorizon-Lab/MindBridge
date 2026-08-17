"""Reproducible Video-MME runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.prompts import VIDEO_MME_QUERY_PROMPT
from mindbridge.benchmarks.runner_cli import (
    MediaBenchmarkArguments,
    MediaRunManifest,
    add_media_arguments,
    benchmark_parser,
    completed_now,
    connected_memory,
    index_prepared,
    json_predictions,
    media_argument_values,
    predictions_digest,
    select_by_id,
    shared_argument_values,
    write_run_artifacts,
)
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.benchmarks.video_mme import (
    VIDEO_MME_ADAPTER_VERSION,
    VideoMMEMetrics,
    VideoMMEVideo,
    VideoMMEVideoResult,
    evaluate_video_mme,
    load_video_mme,
    run_video_mme_video,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
)

VIDEO_MME_RUNNER_VERSION = "video_mme_production_api_v2"


class VideoMMERunManifest(MediaRunManifest):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["Video-MME"] = "Video-MME"
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    benchmark_prompt_version: NonEmptyString
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)
    metrics: VideoMMEMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaBenchmarkArguments):
    prepared_media_path: Path
    dataset_revision: str
    evaluator_revision: str
    video_ids: tuple[str, ...]


def main() -> None:
    """Run selected videos and emit the official nested prediction JSON."""
    arguments = _parse_arguments()
    videos = _select_videos(load_video_mme(arguments.dataset_path), arguments.video_ids)
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), videos)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            segment.media_objects for video in prepared for segment in video.segments
        ),
    )
    results = asyncio.run(_run(arguments, videos, prepared))
    _write_artifacts(arguments, videos, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    videos: tuple[VideoMMEVideo, ...],
    prepared: tuple[PreparedVideo, ...],
) -> tuple[VideoMMEVideoResult, ...]:
    async with connected_memory(arguments) as memory:
        return tuple(
            [
                await run_video_mme_video(
                    memory,
                    video,
                    prepared_video,
                    run_id=arguments.run_id,
                    tenant_prefix=arguments.tenant_prefix,
                    device_id=arguments.device_id,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                    processing_timeout_seconds=arguments.processing_timeout_seconds,
                )
                for video, prepared_video in zip(videos, prepared, strict=True)
            ]
        )


def _write_artifacts(
    arguments: _Arguments,
    videos: tuple[VideoMMEVideo, ...],
    prepared: tuple[PreparedVideo, ...],
    results: tuple[VideoMMEVideoResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.video_id for result in results) != tuple(video.video_id for video in videos):
        raise ValueError("Video-MME predictions must match annotation video order")
    predictions = json_predictions(results)
    segments = tuple(segment for video in prepared for segment in video.segments)
    manifest = VideoMMERunManifest(
        runner_version=VIDEO_MME_RUNNER_VERSION,
        adapter_version=VIDEO_MME_ADAPTER_VERSION,
        dataset_repository="lmms-eval/Video-MME",
        dataset_revision=arguments.dataset_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        evaluator_repository="thanku-all/parse_answer",
        evaluator_revision=arguments.evaluator_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        benchmark_prompt_version=VIDEO_MME_QUERY_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        video_ids=tuple(video.video_id for video in videos),
        segment_count=len(segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in segments),
        transcript_segment_count=sum(segment.transcript is not None for segment in segments),
        metrics=evaluate_video_mme(results),
        predictions_sha256=predictions_digest(predictions),
        completed_at=completed_now(),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_videos(
    videos: tuple[VideoMMEVideo, ...], video_ids: tuple[str, ...]
) -> tuple[VideoMMEVideo, ...]:
    return select_by_id(
        videos,
        video_ids,
        identify=lambda video: video.video_id,
        label="Video-MME video",
    )


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], videos: tuple[VideoMMEVideo, ...]
) -> tuple[PreparedVideo, ...]:
    by_id = index_prepared(
        tuple(video.video_id for video in videos),
        prepared,
        identify=lambda video: video.video_id,
        label="Video-MME videos",
    )
    return tuple(by_id[video.video_id] for video in videos)


def _parse_arguments() -> _Arguments:
    parser = benchmark_parser(tenant_prefix="benchmark_video_mme")
    add_media_arguments(parser, device_id="video_mme_camera")
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        **shared_argument_values(parsed),
        **media_argument_values(parsed),
        prepared_media_path=parsed.prepared_media,
        dataset_revision=parsed.dataset_revision,
        evaluator_revision=parsed.evaluator_revision,
        video_ids=tuple(parsed.video_id),
    )


if __name__ == "__main__":
    main()
