"""Reproducible SuperMemory-VQA runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
import os
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
    add_media_arguments,
    core_parser,
    media_arguments,
    media_manifest,
    write_run_artifacts,
)
from mindbridge.benchmarks.supermemory_runner import (
    SuperMemoryMetrics,
    SuperMemoryPreparedSubject,
    SuperMemoryQuestionResult,
    evaluate_supermemory_vqa,
    load_prepared_supermemory,
    run_supermemory_vqa,
)
from mindbridge.benchmarks.supermemory_vqa import (
    SUPERMEMORY_VQA_ADAPTER_VERSION,
    SuperMemoryQuestion,
    load_supermemory_vqa,
)
from mindbridge.contracts import NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT
from mindbridge.sdk import MindBridge

SUPERMEMORY_RUNNER_VERSION = "supermemory_production_api_v7"


class SuperMemoryRunManifest(MediaBenchmarkRunManifest):
    """Immutable data, deployment, code, and output identity for one run."""

    benchmark: Literal["SuperMemory-VQA"] = "SuperMemory-VQA"
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    subject: int = Field(gt=0)
    question_ids: tuple[int, ...] = Field(min_length=1)
    video_count: int = Field(gt=0)
    segment_count: int = Field(gt=0)
    metrics: SuperMemoryMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    subject: int
    dataset_revision: str
    source_revision: str
    question_ids: tuple[int, ...]


def main() -> None:
    """Run one participant and emit predictions, official metrics, and a manifest."""
    arguments = _parse_arguments()
    questions = _select_questions(
        load_supermemory_vqa(arguments.dataset_path),
        arguments.subject,
        arguments.question_ids,
    )
    prepared = load_prepared_supermemory(arguments.prepared_media_path)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            segment.media_objects for video in prepared.videos for segment in video.segments
        ),
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[SuperMemoryQuestion, ...],
    prepared: SuperMemoryPreparedSubject,
) -> tuple[SuperMemoryQuestionResult, ...]:
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        return await run_supermemory_vqa(
            memory,
            questions,
            prepared,
            run_id=arguments.run_id,
            tenant_prefix=arguments.tenant_prefix,
            device_id=arguments.device_id,
            recall_limit=arguments.recall_limit,
            request_concurrency=arguments.request_concurrency,
            poll_interval_seconds=arguments.poll_interval_seconds,
            processing_timeout_seconds=arguments.processing_timeout_seconds,
        )
    finally:
        await memory.close()


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[SuperMemoryQuestion, ...],
    prepared: SuperMemoryPreparedSubject,
    results: tuple[SuperMemoryQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.question_id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("SuperMemory-VQA predictions must match annotation question order")
    metrics = evaluate_supermemory_vqa(questions, results)
    predictions = (
        json.dumps(
            {
                "metrics": metrics.model_dump(mode="json"),
                "results": [result.model_dump(mode="json") for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = media_manifest(
        SuperMemoryRunManifest,
        arguments,
        deployment,
        runner_version=SUPERMEMORY_RUNNER_VERSION,
        adapter_version=SUPERMEMORY_VQA_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
        dataset_revision=arguments.dataset_revision,
        source_repository="AIoT-MLSys-Lab/supermemory-vqa",
        source_revision=arguments.source_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        subject=arguments.subject,
        question_ids=tuple(question.question_id for question in questions),
        video_count=len(prepared.videos),
        segment_count=sum(len(video.segments) for video in prepared.videos),
        metrics=metrics,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_questions(
    questions: tuple[SuperMemoryQuestion, ...],
    subject: int,
    question_ids: tuple[int, ...],
) -> tuple[SuperMemoryQuestion, ...]:
    if subject <= 0:
        raise ValueError("subject must be positive")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question IDs must not contain duplicates")
    requested = set(question_ids)
    subject_questions = tuple(question for question in questions if question.subject == subject)
    missing = requested - {question.question_id for question in subject_questions}
    if missing:
        raise ValueError(
            "unknown SuperMemory-VQA question IDs for subject "
            f"{subject}: {', '.join(map(str, sorted(missing)))}"
        )
    selected = tuple(
        question
        for question in subject_questions
        if not requested or question.question_id in requested
    )
    if not selected:
        raise ValueError(f"SuperMemory-VQA subject {subject} has no selected questions")
    return selected


def _parse_arguments() -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_supermemory"),
        device_id="supermemory_glasses",
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--question-id", type=int, action="append", default=[])
    parsed = parser.parse_args()
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        subject=parsed.subject,
        dataset_revision=parsed.dataset_revision,
        source_revision=parsed.source_revision,
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
