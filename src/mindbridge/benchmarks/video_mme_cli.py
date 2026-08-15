"""Reproducible Video-MME runner against a deployed MindBridge API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field

from mindbridge.application.recall import RETRIEVAL_DOCUMENT_EMBEDDING_TASK
from mindbridge.benchmarks.artifacts import (
    require_writable_output_pair,
    sidecar_manifest_path,
    write_text_atomically,
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
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models.jina import DEFAULT_JINA_OMNI_MODEL_ID, DEFAULT_JINA_OMNI_REVISION
from mindbridge.models.openai_chat import REASONING_EFFORT_VALUES
from mindbridge.models.openai_omni import DEFAULT_OMNI_MODEL_ID
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
    VIDEO_MME_QUERY_PROMPT,
)
from mindbridge.sdk import AsyncMindBridge

VIDEO_MME_RUNNER_VERSION = "video_mme_production_api_v1"


class VideoMMERunManifest(ContractModel):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["Video-MME"] = "Video-MME"
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    annotation_sha256: Sha256Hex
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    code_revision: NonEmptyString
    perception_model_id: NonEmptyString
    perception_model_revision: NonEmptyString
    perception_prompt_version: NonEmptyString
    answer_model_id: NonEmptyString
    answer_model_revision: NonEmptyString
    answer_prompt_version: NonEmptyString
    benchmark_prompt_version: NonEmptyString
    reasoning_effort: NonEmptyString
    embedding_model_id: NonEmptyString
    embedding_model_revision: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    device_id: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)
    metrics: VideoMMEMetrics
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_path: Path
    prepared_media_path: Path
    output_path: Path
    api_base_url: str
    dataset_revision: str
    evaluator_revision: str
    code_revision: str
    perception_model_id: str
    perception_model_revision: str
    answer_model_id: str
    answer_model_revision: str
    answer_reasoning_effort: str
    embedding_model_id: str
    embedding_model_revision: str
    run_id: str
    tenant_prefix: str
    device_id: str
    recall_limit: int
    request_concurrency: int
    request_timeout_seconds: float
    poll_interval_seconds: float
    processing_timeout_seconds: float
    video_ids: tuple[str, ...]
    overwrite: bool


def main() -> None:
    """Run selected videos and emit the official nested prediction JSON."""
    arguments = _parse_arguments()
    videos = _select_videos(load_video_mme(arguments.dataset_path), arguments.video_ids)
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), videos)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    results = asyncio.run(_run(arguments, videos, prepared))
    _write_artifacts(arguments, videos, prepared, results)


async def _run(
    arguments: _Arguments,
    videos: tuple[VideoMMEVideo, ...],
    prepared: tuple[PreparedVideo, ...],
) -> tuple[VideoMMEVideoResult, ...]:
    memory = AsyncMindBridge.connect(
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
) -> None:
    if tuple(result.video_id for result in results) != tuple(video.video_id for video in videos):
        raise ValueError("Video-MME predictions must match annotation video order")
    predictions = (
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
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
        perception_model_id=arguments.perception_model_id,
        perception_model_revision=arguments.perception_model_revision,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        answer_model_id=arguments.answer_model_id,
        answer_model_revision=arguments.answer_model_revision,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        benchmark_prompt_version=VIDEO_MME_QUERY_PROMPT.version,
        reasoning_effort=arguments.answer_reasoning_effort,
        embedding_model_id=arguments.embedding_model_id,
        embedding_model_revision=arguments.embedding_model_revision,
        retrieval_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
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
        predictions_sha256=hashlib.sha256(predictions.encode()).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _select_videos(
    videos: tuple[VideoMMEVideo, ...], video_ids: tuple[str, ...]
) -> tuple[VideoMMEVideo, ...]:
    if not video_ids:
        return videos
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video IDs must not contain duplicates")
    requested = set(video_ids)
    selected = tuple(video for video in videos if video.video_id in requested)
    missing = requested - {video.video_id for video in selected}
    if missing:
        raise ValueError(f"unknown Video-MME video IDs: {', '.join(sorted(missing))}")
    return selected


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], videos: tuple[VideoMMEVideo, ...]
) -> tuple[PreparedVideo, ...]:
    by_id = {video.video_id: video for video in prepared}
    missing = {video.video_id for video in videos} - set(by_id)
    if missing:
        raise ValueError(f"missing prepared Video-MME videos: {', '.join(sorted(missing))}")
    return tuple(by_id[video.video_id] for video in videos)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--perception-model-id", default=DEFAULT_OMNI_MODEL_ID)
    parser.add_argument("--perception-model-revision", required=True)
    parser.add_argument("--answer-model-id", default=DEFAULT_OMNI_MODEL_ID)
    parser.add_argument("--answer-model-revision", required=True)
    parser.add_argument(
        "--answer-reasoning-effort",
        choices=("omitted", *REASONING_EFFORT_VALUES),
        required=True,
    )
    parser.add_argument("--embedding-model-id", default=DEFAULT_JINA_OMNI_MODEL_ID)
    parser.add_argument("--embedding-model-revision", default=DEFAULT_JINA_OMNI_REVISION)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="benchmark_video_mme")
    parser.add_argument("--device-id", default="video_mme_camera")
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        prepared_media_path=parsed.prepared_media,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        dataset_revision=parsed.dataset_revision,
        evaluator_revision=parsed.evaluator_revision,
        code_revision=parsed.code_revision,
        perception_model_id=parsed.perception_model_id,
        perception_model_revision=parsed.perception_model_revision,
        answer_model_id=parsed.answer_model_id,
        answer_model_revision=parsed.answer_model_revision,
        answer_reasoning_effort=parsed.answer_reasoning_effort,
        embedding_model_id=parsed.embedding_model_id,
        embedding_model_revision=parsed.embedding_model_revision,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        device_id=parsed.device_id,
        recall_limit=parsed.recall_limit,
        request_concurrency=parsed.request_concurrency,
        request_timeout_seconds=parsed.request_timeout_seconds,
        poll_interval_seconds=parsed.poll_interval_seconds,
        processing_timeout_seconds=parsed.processing_timeout_seconds,
        video_ids=tuple(parsed.video_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
