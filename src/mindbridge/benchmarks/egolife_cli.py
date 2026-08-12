"""Reproducible EgoLifeQA runner against a deployed MindBridge API."""

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
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
)
from mindbridge.models.openai_omni import (
    ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
    DEFAULT_OMNI_MODEL_ID,
)
from mindbridge.models.openai_perception import (
    PERCEIVE_EVENTS_PROMPT_VERSION,
)
from mindbridge.sdk import AsyncMindBridge

EGOLIFE_RUNNER_VERSION = "egolife_production_api_v1"


class EgoLifeRunManifest(ContractModel):
    """Immutable data, deployment, code, and output identity for one run."""

    benchmark: Literal["EgoLifeQA"] = "EgoLifeQA"
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
    embedding_model_id: NonEmptyString
    embedding_model_revision: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    device_id: Identifier
    subject_id: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    metrics: EgoLifeMetrics
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
    embedding_model_id: str
    embedding_model_revision: str
    run_id: str
    tenant_prefix: str
    device_id: str
    recall_limit: int
    request_concurrency: int
    poll_interval_seconds: float
    processing_timeout_seconds: float
    question_ids: tuple[str, ...]
    overwrite: bool


def main() -> None:
    """Run selected official questions and emit scored predictions plus a manifest."""
    arguments = _parse_arguments()
    questions = _select_questions(load_egolife_qa(arguments.dataset_path), arguments.question_ids)
    prepared = load_prepared_egolife(arguments.prepared_media_path)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoLifeQuestion, ...],
    prepared: EgoLifePreparedStream,
) -> tuple[EgoLifeQuestionResult, ...]:
    memory = AsyncMindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
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
) -> None:
    if tuple(result.id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("EgoLifeQA predictions must match annotation question order")
    metrics = evaluate_egolife_qa(results)
    predictions = (
        json.dumps(
            {
                "accuracy": metrics.accuracy,
                "results": [result.model_dump(mode="json") for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
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
        perception_model_id=arguments.perception_model_id,
        perception_model_revision=arguments.perception_model_revision,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT_VERSION,
        answer_model_id=arguments.answer_model_id,
        answer_model_revision=arguments.answer_model_revision,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
        embedding_model_id=arguments.embedding_model_id,
        embedding_model_revision=arguments.embedding_model_revision,
        retrieval_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        subject_id=prepared.subject_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_ids=tuple(question.question_id for question in questions),
        clip_count=len(prepared.clips),
        metrics=metrics,
        predictions_sha256=hashlib.sha256(predictions.encode()).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _select_questions(
    questions: tuple[EgoLifeQuestion, ...],
    question_ids: tuple[str, ...],
) -> tuple[EgoLifeQuestion, ...]:
    if not question_ids:
        return questions
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question IDs must not contain duplicates")
    requested = set(question_ids)
    selected = tuple(question for question in questions if question.question_id in requested)
    missing = requested - {question.question_id for question in selected}
    if missing:
        raise ValueError(f"unknown EgoLifeQA question IDs: {', '.join(sorted(missing))}")
    return selected


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
    parser.add_argument("--embedding-model-id", default=DEFAULT_JINA_OMNI_MODEL_ID)
    parser.add_argument("--embedding-model-revision", default=DEFAULT_JINA_OMNI_REVISION)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="benchmark_egolife")
    parser.add_argument("--device-id", default="egolife_camera")
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
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
        dataset_revision=parsed.dataset_revision,
        evaluator_revision=parsed.evaluator_revision,
        code_revision=parsed.code_revision,
        perception_model_id=parsed.perception_model_id,
        perception_model_revision=parsed.perception_model_revision,
        answer_model_id=parsed.answer_model_id,
        answer_model_revision=parsed.answer_model_revision,
        embedding_model_id=parsed.embedding_model_id,
        embedding_model_revision=parsed.embedding_model_revision,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        device_id=parsed.device_id,
        recall_limit=parsed.recall_limit,
        request_concurrency=parsed.request_concurrency,
        poll_interval_seconds=parsed.poll_interval_seconds,
        processing_timeout_seconds=parsed.processing_timeout_seconds,
        question_ids=tuple(parsed.question_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
