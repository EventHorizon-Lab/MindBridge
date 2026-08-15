"""Reproducible M3-Bench runner against a deployed MindBridge API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import AwareDatetime, Field

from mindbridge.application.recall import RETRIEVAL_DOCUMENT_EMBEDDING_TASK
from mindbridge.benchmarks.artifacts import (
    require_writable_output_pair,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.m3_bench import M3_BENCH_ADAPTER_VERSION, M3BenchVideo, load_m3_bench
from mindbridge.benchmarks.m3_runner import (
    M3_CLIP_DURATION_SECONDS,
    M3OfficialQuestionResult,
    M3PreparedVideo,
    load_prepared_m3,
    run_m3_video,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
)
from mindbridge.models.openai_chat import REASONING_EFFORT_VALUES
from mindbridge.models.openai_omni import (
    ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
    DEFAULT_OMNI_MODEL_ID,
)
from mindbridge.models.openai_perception import (
    PERCEIVE_EVENTS_PROMPT_VERSION,
)
from mindbridge.sdk import AsyncMindBridge

M3_RUNNER_VERSION = "m3_production_api_v8"


class M3RunManifest(ContractModel):
    """Immutable data, deployment, code, and output identity for one M3 run."""

    benchmark: Literal["M3-Bench"] = "M3-Bench"
    subset: Literal["robot", "web"]
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    annotation_sha256: Sha256Hex
    media_repository: NonEmptyString
    media_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    code_revision: NonEmptyString
    perception_model_id: NonEmptyString
    perception_model_revision: NonEmptyString
    perception_prompt_version: NonEmptyString
    answer_model_id: NonEmptyString
    answer_model_revision: NonEmptyString
    answer_prompt_version: NonEmptyString
    reasoning_effort: NonEmptyString
    embedding_model_id: NonEmptyString
    embedding_model_revision: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    device_id: Identifier
    clip_duration_seconds: int = Field(gt=0)
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    video_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)
    question_count: int = Field(gt=0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_path: Path
    prepared_media_path: Path
    output_path: Path
    api_base_url: str
    subset: Literal["robot", "web"]
    source_revision: str
    media_revision: str
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
    """Run selected official videos and emit JSONL predictions plus a manifest."""
    arguments = _parse_arguments()
    videos = _select_videos(load_m3_bench(arguments.dataset_path), arguments.video_ids)
    _validate_subset(videos, arguments.subset)
    prepared = _prepared_by_video(videos, load_prepared_m3(arguments.prepared_media_path))
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    results = asyncio.run(_run_videos(arguments, videos, prepared))
    _write_artifacts(arguments, videos, prepared, results)


async def _run_videos(
    arguments: _Arguments,
    videos: tuple[M3BenchVideo, ...],
    prepared: dict[str, M3PreparedVideo],
) -> tuple[M3OfficialQuestionResult, ...]:
    memory = AsyncMindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
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
    finally:
        await memory.close()


def _write_artifacts(
    arguments: _Arguments,
    videos: tuple[M3BenchVideo, ...],
    prepared: dict[str, M3PreparedVideo],
    results: tuple[M3OfficialQuestionResult, ...],
) -> None:
    expected_ids = tuple(question.question_id for video in videos for question in video.questions)
    if tuple(result.id for result in results) != expected_ids:
        raise ValueError("M3-Bench predictions must match annotation question order")
    predictions = "".join(result.model_dump_json() + "\n" for result in results)
    manifest = M3RunManifest(
        subset=arguments.subset,
        runner_version=M3_RUNNER_VERSION,
        adapter_version=M3_BENCH_ADAPTER_VERSION,
        source_repository="ByteDance-Seed/m3-agent",
        source_revision=arguments.source_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        media_repository="ByteDance-Seed/M3-Bench",
        media_revision=arguments.media_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        code_revision=arguments.code_revision,
        perception_model_id=arguments.perception_model_id,
        perception_model_revision=arguments.perception_model_revision,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT_VERSION,
        answer_model_id=arguments.answer_model_id,
        answer_model_revision=arguments.answer_model_revision,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
        reasoning_effort=arguments.answer_reasoning_effort,
        embedding_model_id=arguments.embedding_model_id,
        embedding_model_revision=arguments.embedding_model_revision,
        retrieval_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        clip_duration_seconds=M3_CLIP_DURATION_SECONDS,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        video_ids=tuple(video.video_id for video in videos),
        clip_count=sum(len(prepared[video.video_id].clips) for video in videos),
        media_clip_count=sum(
            clip.media_object is not None
            for video in videos
            for clip in prepared[video.video_id].clips
        ),
        caption_clip_count=sum(
            clip.caption is not None for video in videos for clip in prepared[video.video_id].clips
        ),
        question_count=sum(len(video.questions) for video in videos),
        predictions_sha256=hashlib.sha256(predictions.encode("utf-8")).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


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
    by_id = {video.video_id: video for video in prepared}
    missing = {video.video_id for video in videos} - by_id.keys()
    if missing:
        raise ValueError(f"missing prepared M3-Bench videos: {', '.join(sorted(missing))}")
    return by_id


def _validate_subset(videos: tuple[M3BenchVideo, ...], subset: Literal["robot", "web"]) -> None:
    expected = (True, True) if subset == "robot" else (False, False)
    profiles = {
        (question.timestamp_seconds is not None, question.before_clip_index is not None)
        for video in videos
        for question in video.questions
    }
    if profiles != {expected}:
        raise ValueError(f"M3-Bench annotations do not match the {subset} subset")


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--subset", choices=("robot", "web"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--media-revision", required=True)
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
    parser.add_argument("--tenant-prefix", default="benchmark_m3")
    parser.add_argument("--device-id", default="m3_bench_camera")
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
        subset=cast(Literal["robot", "web"], parsed.subset),
        source_revision=parsed.source_revision,
        media_revision=parsed.media_revision,
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
