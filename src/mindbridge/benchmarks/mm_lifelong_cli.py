"""Reproducible MM-Lifelong runner against a deployed MindBridge API."""

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

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.mm_lifelong import (
    MM_LIFELONG_ADAPTER_VERSION,
    MMLifelongQuestion,
    MMLifelongSplit,
    load_mm_lifelong,
)
from mindbridge.benchmarks.mm_lifelong_runner import (
    MMLifelongPreparedTimeline,
    MMLifelongQuestionResult,
    load_prepared_mm_lifelong,
    run_mm_lifelong,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, PERCEIVE_EVENTS_PROMPT
from mindbridge.sdk import MindBridge

MM_LIFELONG_RUNNER_VERSION = "mm_lifelong_production_api_v2"


class MMLifelongRunManifest(ContractModel):
    """Immutable data, media, deployment, model, and output identity for one run."""

    benchmark: Literal["MM-Lifelong"] = "MM-Lifelong"
    split: MMLifelongSplit
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    annotation_sha256: Sha256Hex
    prepared_media_manifest_sha256: Sha256Hex
    code_revision: NonEmptyString
    deployment: DeploymentSnapshot
    deployment_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    answer_prompt_version: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    device_id: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    question_indices: tuple[int, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    caption_segment_count: int = Field(ge=0)
    mean_unofficial_ref_at_300: float = Field(ge=0.0, le=1.0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_path: Path
    prepared_media_path: Path
    output_path: Path
    api_base_url: str
    split: MMLifelongSplit
    source_revision: str
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
    question_indices: tuple[int, ...]
    overwrite: bool


def main() -> None:
    """Run one official split and emit JSONL accepted by its released evaluators."""
    arguments = _parse_arguments()
    questions = _select_questions(
        load_mm_lifelong(arguments.dataset_path, arguments.split),
        arguments.question_indices,
    )
    prepared = load_prepared_mm_lifelong(arguments.prepared_media_path)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(segment.media_objects for segment in prepared.segments),
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[MMLifelongQuestion, ...],
    prepared: MMLifelongPreparedTimeline,
) -> tuple[MMLifelongQuestionResult, ...]:
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        return await run_mm_lifelong(
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
    questions: tuple[MMLifelongQuestion, ...],
    prepared: MMLifelongPreparedTimeline,
    results: tuple[MMLifelongQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.index for result in results) != tuple(question.index for question in questions):
        raise ValueError("MM-Lifelong predictions must match annotation question order")
    predictions = "".join(result.model_dump_json() + "\n" for result in results)
    manifest = MMLifelongRunManifest(
        split=arguments.split,
        runner_version=MM_LIFELONG_RUNNER_VERSION,
        adapter_version=MM_LIFELONG_ADAPTER_VERSION,
        source_repository="MM-Lifelong/MM-Lifelong",
        source_revision=arguments.source_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
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
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_indices=tuple(question.index for question in questions),
        segment_count=len(prepared.segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in prepared.segments),
        caption_segment_count=sum(segment.caption is not None for segment in prepared.segments),
        mean_unofficial_ref_at_300=(
            sum(result.mindbridge_unofficial_ref_at_300 for result in results) / len(results)
        ),
        predictions_sha256=hashlib.sha256(predictions.encode("utf-8")).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _select_questions(
    questions: tuple[MMLifelongQuestion, ...],
    question_indices: tuple[int, ...],
) -> tuple[MMLifelongQuestion, ...]:
    if not question_indices:
        return questions
    if len(set(question_indices)) != len(question_indices):
        raise ValueError("question indices must not contain duplicates")
    requested = set(question_indices)
    selected = tuple(question for question in questions if question.index in requested)
    missing = requested - {question.index for question in selected}
    if missing:
        raise ValueError(
            f"unknown MM-Lifelong question indices: {', '.join(map(str, sorted(missing)))}"
        )
    return selected


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument(
        "--split",
        choices=("day_test", "week_test", "month_train", "month_val"),
        required=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="benchmark_mm_lifelong")
    parser.add_argument("--device-id", default="mm_lifelong_camera")
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--question-index", type=int, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        prepared_media_path=parsed.prepared_media,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        split=cast(MMLifelongSplit, parsed.split),
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
        question_indices=tuple(parsed.question_index),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
