"""Reproducible Video-MME runner against a deployed MindBridge API."""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    MediaArguments,
    MediaRunManifestBase,
    load_deployment_snapshot,
    media_benchmark_parser,
    predictions_document,
    require_writable_output_pair,
    select_by_id,
    sha256_text,
    write_run_artifacts,
)
from mindbridge.benchmarks.prompts import VIDEO_MME_QUERY_PROMPT
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.benchmarks.video_mme import (
    VIDEO_MME_ADAPTER_VERSION,
    VideoMMEDuration,
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
from mindbridge.sdk import MindBridge

VIDEO_MME_RUNNER_VERSION = "video_mme_production_api_v3"
# Video-MME publishes separate "with subtitles" and "without subtitles" tables, and prepared
# segments may legitimately carry either MindBridge's own ASR or the released subtitle track.
# The two are the same field to the ingest path and a different benchmark cell to a reader, so
# every run must name which one it fed and is refused when the media disagrees.
VideoMMETranscriptSource = Literal["none", "asr", "official_subtitles"]


class VideoMMERunManifest(MediaRunManifestBase):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["Video-MME"] = "Video-MME"
    perception_prompt_version: NonEmptyString
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    benchmark_prompt_version: NonEmptyString
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    durations: tuple[VideoMMEDuration, ...] = Field(min_length=1)
    transcript_source: VideoMMETranscriptSource
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)
    metrics: VideoMMEMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    dataset_revision: str
    evaluator_revision: str
    video_ids: tuple[str, ...]
    durations: tuple[VideoMMEDuration, ...]
    transcript_source: VideoMMETranscriptSource


def main() -> None:
    """Run selected videos and emit the official nested prediction JSON."""
    arguments = _parse_arguments()
    videos = _select_videos(
        load_video_mme(arguments.dataset_path), arguments.video_ids, arguments.durations
    )
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), videos)
    _require_declared_transcripts(prepared, arguments.transcript_source)
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
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
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
    finally:
        await memory.close()


def _write_artifacts(
    arguments: _Arguments,
    videos: tuple[VideoMMEVideo, ...],
    prepared: tuple[PreparedVideo, ...],
    results: tuple[VideoMMEVideoResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.video_id for result in results) != tuple(video.video_id for video in videos):
        raise ValueError("Video-MME predictions must match annotation video order")
    predictions = predictions_document(
        [result.model_dump(mode="json") for result in results],
    )
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
        durations=tuple(dict.fromkeys(video.duration for video in videos)),
        transcript_source=arguments.transcript_source,
        segment_count=len(segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in segments),
        transcript_segment_count=sum(segment.transcript is not None for segment in segments),
        metrics=evaluate_video_mme(results),
        predictions_sha256=sha256_text(predictions),
        completed_at=datetime.now(timezone.utc),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_videos(
    videos: tuple[VideoMMEVideo, ...],
    video_ids: tuple[str, ...],
    durations: tuple[VideoMMEDuration, ...],
) -> tuple[VideoMMEVideo, ...]:
    videos = select_by_id(
        videos,
        video_ids,
        key=lambda video: video.video_id,
        label="Video-MME video IDs",
    )
    if durations:
        videos = tuple(video for video in videos if video.duration in set(durations))
    if not videos:
        raise ValueError("no Video-MME videos match the requested IDs and durations")
    return videos


def _require_declared_transcripts(
    prepared: tuple[PreparedVideo, ...],
    transcript_source: VideoMMETranscriptSource,
) -> None:
    """Refuse a run whose declared subtitle setting disagrees with its prepared media."""
    present = any(
        segment.transcript is not None for video in prepared for segment in video.segments
    )
    if present and transcript_source == "none":
        raise ValueError(
            "prepared media carries transcripts; a run declaring no transcript source would "
            "report a with-subtitles result in the without-subtitles column"
        )
    if not present and transcript_source != "none":
        raise ValueError(
            f"prepared media carries no transcript, so it cannot be {transcript_source}"
        )


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], videos: tuple[VideoMMEVideo, ...]
) -> tuple[PreparedVideo, ...]:
    by_id = {video.video_id: video for video in prepared}
    missing = {video.video_id for video in videos} - set(by_id)
    if missing:
        raise ValueError(f"missing prepared Video-MME videos: {', '.join(sorted(missing))}")
    return tuple(by_id[video.video_id] for video in videos)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        parents=[
            media_benchmark_parser(
                tenant_prefix="benchmark_video_mme", device_id="video_mme_camera"
            )
        ]
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument(
        "--duration", action="append", default=[], choices=("short", "medium", "long")
    )
    parser.add_argument(
        "--transcript-source", required=True, choices=("none", "asr", "official_subtitles")
    )
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        prepared_media_path=parsed.prepared_media,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        dataset_revision=parsed.dataset_revision,
        evaluator_revision=parsed.evaluator_revision,
        code_revision=parsed.code_revision,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        device_id=parsed.device_id,
        recall_limit=parsed.recall_limit,
        request_concurrency=parsed.request_concurrency,
        request_timeout_seconds=parsed.request_timeout_seconds,
        poll_interval_seconds=parsed.poll_interval_seconds,
        processing_timeout_seconds=parsed.processing_timeout_seconds,
        video_ids=tuple(parsed.video_id),
        durations=tuple(dict.fromkeys(parsed.duration)),
        transcript_source=parsed.transcript_source,
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
