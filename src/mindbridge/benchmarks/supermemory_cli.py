"""Reproducible SuperMemory-VQA runner against a deployed MindBridge API."""

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
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, PERCEIVE_EVENTS_PROMPT
from mindbridge.sdk import MindBridge

SUPERMEMORY_RUNNER_VERSION = "supermemory_production_api_v7"


class SuperMemoryRunManifest(MediaRunManifestBase):
    """Immutable data, deployment, code, and output identity for one run."""

    benchmark: Literal["SuperMemory-VQA"] = "SuperMemory-VQA"
    perception_prompt_version: NonEmptyString
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
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
    predictions = predictions_document(
        {
            "metrics": metrics.model_dump(mode="json"),
            "results": [result.model_dump(mode="json") for result in results],
        },
    )
    manifest = SuperMemoryRunManifest(
        runner_version=SUPERMEMORY_RUNNER_VERSION,
        adapter_version=SUPERMEMORY_VQA_ADAPTER_VERSION,
        dataset_repository="OSU-AIoT-MLSys-Lab/SuperMemory-VQA",
        dataset_revision=arguments.dataset_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        source_repository="AIoT-MLSys-Lab/supermemory-vqa",
        source_revision=arguments.source_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        subject=arguments.subject,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_ids=tuple(question.question_id for question in questions),
        video_count=len(prepared.videos),
        segment_count=sum(len(video.segments) for video in prepared.videos),
        metrics=metrics,
        predictions_sha256=sha256_text(predictions),
        completed_at=datetime.now(timezone.utc),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_questions(
    questions: tuple[SuperMemoryQuestion, ...],
    subject: int,
    question_ids: tuple[int, ...],
) -> tuple[SuperMemoryQuestion, ...]:
    if subject <= 0:
        raise ValueError("subject must be positive")
    selected = select_by_id(
        tuple(question for question in questions if question.subject == subject),
        question_ids,
        key=lambda question: question.question_id,
        label=f"SuperMemory-VQA question IDs for subject {subject}",
    )
    if not selected:
        raise ValueError(f"SuperMemory-VQA subject {subject} has no selected questions")
    return selected


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        parents=[
            media_benchmark_parser(
                tenant_prefix="benchmark_supermemory", device_id="supermemory_glasses"
            )
        ]
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--question-id", type=int, action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        prepared_media_path=parsed.prepared_media,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        subject=parsed.subject,
        dataset_revision=parsed.dataset_revision,
        source_revision=parsed.source_revision,
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
