"""Reproducible EgoLifeQA runner against a deployed MindBridge API."""

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
from mindbridge.benchmarks.egolife_qa import (
    EGOLIFE_QA_ADAPTER_VERSION,
    EgoLifeQuestion,
    load_egolife_qa,
)
from mindbridge.benchmarks.egolife_runner import (
    EgoLifeMetrics,
    EgoLifePreparedStream,
    EgoLifeQuestionResult,
    evaluate_egolife_qa,
    load_prepared_egolife,
    run_egolife_qa,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, PERCEIVE_EVENTS_PROMPT
from mindbridge.sdk import MindBridge

EGOLIFE_RUNNER_VERSION = "egolife_production_api_v7"


class EgoLifeRunManifest(MediaRunManifestBase):
    """Immutable data, deployment, code, and output identity for one run."""

    benchmark: Literal["EgoLifeQA"] = "EgoLifeQA"
    perception_prompt_version: NonEmptyString
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    subject_id: Identifier
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)
    metrics: EgoLifeMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    dataset_revision: str
    evaluator_revision: str
    question_ids: tuple[str, ...]


def main() -> None:
    """Run selected official questions and emit scored predictions plus a manifest."""
    arguments = _parse_arguments()
    questions = select_by_id(
        load_egolife_qa(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="EgoLifeQA question IDs",
    )
    prepared = load_prepared_egolife(arguments.prepared_media_path)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(clip.media_object is not None for clip in prepared.clips),
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoLifeQuestion, ...],
    prepared: EgoLifePreparedStream,
) -> tuple[EgoLifeQuestionResult, ...]:
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        return await run_egolife_qa(
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
    questions: tuple[EgoLifeQuestion, ...],
    prepared: EgoLifePreparedStream,
    results: tuple[EgoLifeQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("EgoLifeQA predictions must match annotation question order")
    media_clip_count = sum(clip.media_object is not None for clip in prepared.clips)
    metrics = evaluate_egolife_qa(results)
    predictions = predictions_document(
        {
            "accuracy": metrics.accuracy,
            "results": [result.model_dump(mode="json") for result in results],
        },
    )
    manifest = EgoLifeRunManifest(
        runner_version=EGOLIFE_RUNNER_VERSION,
        adapter_version=EGOLIFE_QA_ADAPTER_VERSION,
        dataset_repository="lmms-lab/EgoLife",
        dataset_revision=arguments.dataset_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        evaluator_repository="EvolvingLMMs-Lab/EgoLife",
        evaluator_revision=arguments.evaluator_revision,
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
        subject_id=prepared.subject_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_ids=tuple(question.question_id for question in questions),
        clip_count=len(prepared.clips),
        media_clip_count=media_clip_count,
        caption_clip_count=sum(clip.caption is not None for clip in prepared.clips),
        metrics=metrics,
        predictions_sha256=sha256_text(predictions),
        completed_at=datetime.now(timezone.utc),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        parents=[
            media_benchmark_parser(tenant_prefix="benchmark_egolife", device_id="egolife_camera")
        ]
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--question-id", action="append", default=[])
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
        question_ids=tuple(parsed.question_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
