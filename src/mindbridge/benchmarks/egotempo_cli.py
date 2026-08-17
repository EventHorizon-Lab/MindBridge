"""Reproducible EgoTempo runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.egotempo import (
    EGOTEMPO_ADAPTER_VERSION,
    EgoTempoQuestion,
    EgoTempoQuestionResult,
    load_egotempo,
    run_egotempo_clip,
)
from mindbridge.benchmarks.prompts import EGOTEMPO_QUERY_PROMPT
from mindbridge.benchmarks.runner_cli import (
    MediaBenchmarkArguments,
    MediaRunManifest,
    add_media_arguments,
    benchmark_parser,
    completed_now,
    connected_memory,
    index_prepared,
    media_argument_values,
    predictions_digest,
    select_by_id,
    shared_argument_values,
    write_run_artifacts,
)
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
)

EGOTEMPO_RUNNER_VERSION = "egotempo_production_api_v2"


class EgoTempoRunManifest(MediaRunManifest):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["EgoTempo"] = "EgoTempo"
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    benchmark_prompt_version: NonEmptyString
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_ids: tuple[Identifier, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaBenchmarkArguments):
    prepared_media_path: Path
    source_revision: str
    evaluator_revision: str
    question_ids: tuple[str, ...]


def main() -> None:
    """Run selected questions and emit JSON accepted by the official judge notebook."""
    arguments = _parse_arguments()
    questions = _select_questions(load_egotempo(arguments.dataset_path), arguments.question_ids)
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), questions)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            segment.media_objects for video in prepared for segment in video.segments
        ),
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoTempoQuestion, ...],
    prepared: tuple[PreparedVideo, ...],
) -> tuple[EgoTempoQuestionResult, ...]:
    questions_by_clip: dict[str, list[EgoTempoQuestion]] = {}
    for question in questions:
        questions_by_clip.setdefault(question.clip_id, []).append(question)
    prepared_by_id = {video.video_id: video for video in prepared}
    async with connected_memory(arguments) as memory:
        unordered: list[EgoTempoQuestionResult] = []
        for clip_id, clip_questions in questions_by_clip.items():
            unordered.extend(
                await run_egotempo_clip(
                    memory,
                    tuple(clip_questions),
                    prepared_by_id[clip_id],
                    run_id=arguments.run_id,
                    tenant_prefix=arguments.tenant_prefix,
                    device_id=arguments.device_id,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                    processing_timeout_seconds=arguments.processing_timeout_seconds,
                )
            )
        by_id = {result.question_id: result for result in unordered}
        return tuple(by_id[question.question_id] for question in questions)


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[EgoTempoQuestion, ...],
    prepared: tuple[PreparedVideo, ...],
    results: tuple[EgoTempoQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.question_id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("EgoTempo predictions must match annotation question order")
    predictions = (
        json.dumps(
            [result.model_dump(mode="json", by_alias=True) for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    segments = tuple(segment for video in prepared for segment in video.segments)
    manifest = EgoTempoRunManifest(
        runner_version=EGOTEMPO_RUNNER_VERSION,
        adapter_version=EGOTEMPO_ADAPTER_VERSION,
        source_repository="google-research-datasets/egotempo",
        source_revision=arguments.source_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        evaluator_revision=arguments.evaluator_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        benchmark_prompt_version=EGOTEMPO_QUERY_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_ids=tuple(question.question_id for question in questions),
        clip_ids=tuple(dict.fromkeys(question.clip_id for question in questions)),
        segment_count=len(segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in segments),
        transcript_segment_count=sum(segment.transcript is not None for segment in segments),
        predictions_sha256=predictions_digest(predictions),
        completed_at=completed_now(),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_questions(
    questions: tuple[EgoTempoQuestion, ...], question_ids: tuple[str, ...]
) -> tuple[EgoTempoQuestion, ...]:
    return select_by_id(
        questions,
        question_ids,
        identify=lambda question: question.question_id,
        label="EgoTempo question",
    )


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], questions: tuple[EgoTempoQuestion, ...]
) -> tuple[PreparedVideo, ...]:
    clip_ids = tuple(dict.fromkeys(question.clip_id for question in questions))
    by_id = index_prepared(
        clip_ids,
        prepared,
        identify=lambda video: video.video_id,
        label="EgoTempo clips",
    )
    return tuple(by_id[clip_id] for clip_id in clip_ids)


def _parse_arguments() -> _Arguments:
    parser = benchmark_parser(tenant_prefix="benchmark_egotempo")
    add_media_arguments(parser, device_id="egotempo_camera")
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--question-id", action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        **shared_argument_values(parsed),
        **media_argument_values(parsed),
        prepared_media_path=parsed.prepared_media,
        source_revision=parsed.source_revision,
        evaluator_revision=parsed.evaluator_revision,
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
