"""Reproducible Video-MME-v2 runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    MediaBenchmarkRunManifest,
    TranscriptSource,
    add_media_arguments,
    add_transcript_source_argument,
    connected_memory,
    core_parser,
    index_prepared,
    limit_units,
    media_arguments,
    media_manifest,
    report,
    require_declared_transcripts,
    run_units,
    scoring_snapshot,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.prompts import VIDEO_MME_V2_QUERY_PROMPT
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.benchmarks.scoring import require_scoring_is_possible
from mindbridge.benchmarks.video_mme_v2 import (
    GROUP_SIZE,
    VIDEO_MME_V2_ADAPTER_VERSION,
    VideoMMEV2Group,
    VideoMMEV2GroupResult,
    VideoMMEV2GroupType,
    VideoMMEV2Metrics,
    evaluate_video_mme_v2,
    load_video_mme_v2,
    run_video_mme_v2_group,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

VIDEO_MME_V2_RUNNER_VERSION = "video_mme_v2_production_api_v1"


class VideoMMEV2RunManifest(MediaBenchmarkRunManifest):
    """Immutable data, evaluator, deployment, and output identity for one run.

    `group_count` is pinned beside `video_ids` because the rating's denominator is groups, not
    questions, and a reader comparing two runs needs the denominator without re-deriving it.
    """

    benchmark: Literal["Video-MME-v2"] = "Video-MME-v2"
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    benchmark_prompt_version: NonEmptyString
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    group_count: int = Field(gt=0)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)
    group_types: tuple[VideoMMEV2GroupType, ...] = Field(min_length=1)
    transcript_source: TranscriptSource
    metrics: VideoMMEV2Metrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    video_ids: tuple[str, ...]
    group_types: tuple[VideoMMEV2GroupType, ...]
    transcript_source: TranscriptSource


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected groups and emit the official prediction JSON."""
    arguments = _parse_arguments(argv, prog)
    groups = _select_groups(
        load_video_mme_v2(arguments.dataset_path),
        arguments.video_ids,
        arguments.group_types,
        arguments.limit,
    )
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), groups)
    require_declared_transcripts(prepared, arguments.transcript_source)
    require_scoring_is_possible("video-mme-v2", predict_only=arguments.predict_only)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            segment.media_objects for video in prepared for segment in video.segments
        ),
    )
    report(f"running {len(groups)} groups", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, groups, prepared))
    _write_artifacts(arguments, groups, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    groups: tuple[VideoMMEV2Group, ...],
    prepared: tuple[PreparedVideo, ...],
) -> tuple[VideoMMEV2GroupResult, ...]:
    async with connected_memory(arguments) as memory:
        return await run_units(
            tuple(zip(groups, prepared, strict=True)),
            label=lambda pair: f"group {pair[0].video_id} ({pair[0].group_type})",
            unit_concurrency=arguments.unit_concurrency,
            quiet=arguments.quiet,
            run=lambda pair: run_video_mme_v2_group(
                memory,
                pair[0],
                pair[1],
                run_id=arguments.run_id,
                tenant_prefix=arguments.tenant_prefix,
                device_id=arguments.device_id,
                recall_limit=arguments.recall_limit,
                request_concurrency=arguments.request_concurrency,
                poll_interval_seconds=arguments.poll_interval_seconds,
                processing_timeout_seconds=arguments.processing_timeout_seconds,
            ),
        )


def _write_artifacts(
    arguments: _Arguments,
    groups: tuple[VideoMMEV2Group, ...],
    prepared: tuple[PreparedVideo, ...],
    results: tuple[VideoMMEV2GroupResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.video_id for result in results) != tuple(group.video_id for group in groups):
        raise ValueError("Video-MME-v2 predictions must match annotation group order")
    predictions = (
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    segments = tuple(segment for video in prepared for segment in video.segments)
    metrics = evaluate_video_mme_v2(results)
    scoring = scoring_snapshot(
        "video-mme-v2",
        arguments,
        metrics={
            "rating.overall": metrics.rating.overall,
            "accuracy.overall": metrics.accuracy.overall,
        },
    )
    manifest = media_manifest(
        VideoMMEV2RunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=VIDEO_MME_V2_RUNNER_VERSION,
        adapter_version=VIDEO_MME_V2_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="MME-Benchmarks/Video-MME-v2",
        # Unlike Video-MME, whose parser lives in a separate space, v2 ships its scorer in the
        # benchmark repository itself as `evaluation/test_video_mme_v2.py`.
        evaluator_repository="MME-Benchmarks/Video-MME-v2",
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        benchmark_prompt_version=VIDEO_MME_V2_QUERY_PROMPT.version,
        video_ids=tuple(group.video_id for group in groups),
        group_count=len(groups),
        segment_count=len(segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in segments),
        transcript_segment_count=sum(segment.transcript is not None for segment in segments),
        group_types=tuple(dict.fromkeys(group.group_type for group in groups)),
        transcript_source=arguments.transcript_source,
        metrics=metrics,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_groups(
    groups: tuple[VideoMMEV2Group, ...],
    video_ids: tuple[str, ...],
    group_types: tuple[VideoMMEV2GroupType, ...],
    limit: int | None,
) -> tuple[VideoMMEV2Group, ...]:
    """Narrow a run to whole groups, which is the only subset the rating is defined over.

    Selection is by video and by group type, both of which are constant across a group. There
    is deliberately no `--level` counterpart to Video-MME's `--duration`: `level` varies
    between the four questions of a group, so filtering on it would either split a group or
    silently mean "groups whose fourth question is level N", and neither is a subset anyone
    would mean by the flag.
    """
    groups = select_by_id(
        groups,
        video_ids,
        key=lambda group: group.video_id,
        label="Video-MME-v2 video IDs",
    )
    if group_types:
        groups = tuple(group for group in groups if group.group_type in set(group_types))
    if not groups:
        raise ValueError("no Video-MME-v2 groups match the requested IDs and group types")
    return limit_units(groups, limit, label="Video-MME-v2 groups")


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], groups: tuple[VideoMMEV2Group, ...]
) -> tuple[PreparedVideo, ...]:
    by_id = index_prepared(
        (group.video_id for group in groups),
        prepared,
        key=lambda video: video.video_id,
        label="Video-MME-v2 videos",
    )
    return tuple(by_id[group.video_id] for group in groups)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_video_mme_v2", prog=prog, description=__doc__),
        device_id="video_mme_v2_camera",
    )
    parser.add_argument(
        "--prepared-media", type=Path, required=True, help="manifest of prepared video segments"
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help=(
            f"official video to run, carrying all {GROUP_SIZE} of its questions; "
            "repeatable, default the whole release"
        ),
    )
    parser.add_argument(
        "--group-type",
        action="append",
        default=[],
        choices=("relevance", "logic"),
        help="official group type to keep; repeatable, default both",
    )
    add_transcript_source_argument(parser)
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        video_ids=tuple(parsed.video_id),
        group_types=tuple(dict.fromkeys(parsed.group_type)),
        transcript_source=parsed.transcript_source,
    )


if __name__ == "__main__":
    main()
