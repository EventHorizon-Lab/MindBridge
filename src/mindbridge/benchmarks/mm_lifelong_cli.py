"""Reproducible MM-Lifelong runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
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
from mindbridge.benchmarks.runner_cli import (
    MediaBenchmarkArguments,
    MediaRunManifest,
    add_media_arguments,
    benchmark_parser,
    completed_now,
    connected_memory,
    jsonl_predictions,
    media_argument_values,
    predictions_digest,
    select_by_id,
    shared_argument_values,
    write_run_artifacts,
)
from mindbridge.contracts import NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, PERCEIVE_EVENTS_PROMPT

MM_LIFELONG_RUNNER_VERSION = "mm_lifelong_production_api_v2"


class MMLifelongRunManifest(MediaRunManifest):
    """Immutable data, media, deployment, model, and output identity for one run."""

    benchmark: Literal["MM-Lifelong"] = "MM-Lifelong"
    split: MMLifelongSplit
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    question_indices: tuple[int, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    caption_segment_count: int = Field(ge=0)
    mean_unofficial_ref_at_300: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaBenchmarkArguments):
    prepared_media_path: Path
    split: MMLifelongSplit
    source_revision: str
    question_indices: tuple[int, ...]


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
    async with connected_memory(arguments) as memory:
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


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[MMLifelongQuestion, ...],
    prepared: MMLifelongPreparedTimeline,
    results: tuple[MMLifelongQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.index for result in results) != tuple(question.index for question in questions):
        raise ValueError("MM-Lifelong predictions must match annotation question order")
    predictions = jsonl_predictions(results)
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
        predictions_sha256=predictions_digest(predictions),
        completed_at=completed_now(),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_questions(
    questions: tuple[MMLifelongQuestion, ...],
    question_indices: tuple[int, ...],
) -> tuple[MMLifelongQuestion, ...]:
    return select_by_id(
        questions,
        question_indices,
        identify=lambda question: question.index,
        label="MM-Lifelong question index",
    )


def _parse_arguments() -> _Arguments:
    parser = benchmark_parser(tenant_prefix="benchmark_mm_lifelong")
    add_media_arguments(parser, device_id="mm_lifelong_camera")
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("day_test", "week_test", "month_train", "month_val"),
        required=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--question-index", type=int, action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        **shared_argument_values(parsed),
        **media_argument_values(parsed),
        prepared_media_path=parsed.prepared_media,
        split=cast(MMLifelongSplit, parsed.split),
        source_revision=parsed.source_revision,
        question_indices=tuple(parsed.question_index),
    )


if __name__ == "__main__":
    main()
