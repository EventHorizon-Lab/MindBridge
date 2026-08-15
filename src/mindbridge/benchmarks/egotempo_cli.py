"""Reproducible EgoTempo runner against a deployed MindBridge API."""

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

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.egotempo import (
    EGOTEMPO_ADAPTER_VERSION,
    EgoTempoQuestion,
    EgoTempoQuestionResult,
    load_egotempo,
    run_egotempo_clip,
)
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    EGOTEMPO_QUERY_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
)
from mindbridge.sdk import MindBridge

EGOTEMPO_RUNNER_VERSION = "egotempo_production_api_v2"


class EgoTempoRunManifest(ContractModel):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["EgoTempo"] = "EgoTempo"
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    annotation_sha256: Sha256Hex
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    code_revision: NonEmptyString
    deployment: DeploymentSnapshot
    deployment_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    answer_prompt_version: NonEmptyString
    benchmark_prompt_version: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    device_id: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_ids: tuple[Identifier, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_path: Path
    prepared_media_path: Path
    output_path: Path
    api_base_url: str
    source_revision: str
    evaluator_revision: str
    code_revision: str
    deployment_config_path: Path
    run_id: str
    tenant_prefix: str
    device_id: str
    recall_limit: int
    request_concurrency: int
    request_timeout_seconds: float
    poll_interval_seconds: float
    processing_timeout_seconds: float
    question_ids: tuple[str, ...]
    overwrite: bool


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
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
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
    finally:
        await memory.close()


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
        predictions_sha256=hashlib.sha256(predictions.encode()).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _select_questions(
    questions: tuple[EgoTempoQuestion, ...], question_ids: tuple[str, ...]
) -> tuple[EgoTempoQuestion, ...]:
    if not question_ids:
        return questions
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question IDs must not contain duplicates")
    requested = set(question_ids)
    selected = tuple(question for question in questions if question.question_id in requested)
    missing = requested - {question.question_id for question in selected}
    if missing:
        raise ValueError(f"unknown EgoTempo question IDs: {', '.join(sorted(missing))}")
    return selected


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], questions: tuple[EgoTempoQuestion, ...]
) -> tuple[PreparedVideo, ...]:
    by_id = {video.video_id: video for video in prepared}
    clip_ids = tuple(dict.fromkeys(question.clip_id for question in questions))
    missing = set(clip_ids) - set(by_id)
    if missing:
        raise ValueError(f"missing prepared EgoTempo clips: {', '.join(sorted(missing))}")
    return tuple(by_id[clip_id] for clip_id in clip_ids)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="benchmark_egotempo")
    parser.add_argument("--device-id", default="egotempo_camera")
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        prepared_media_path=parsed.prepared_media,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        source_revision=parsed.source_revision,
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
        question_ids=tuple(parsed.question_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
