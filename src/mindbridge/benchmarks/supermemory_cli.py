"""Reproducible SuperMemory-VQA runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
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
    connected_memory,
    core_parser,
    media_arguments,
    media_manifest,
    report,
    select_by_id,
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
    abstained_question_count: int = Field(ge=0)
    metrics: SuperMemoryMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    subject: int
    dataset_revision: str
    source_revision: str
    question_ids: tuple[int, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run one participant and emit predictions, official metrics, and a manifest."""
    arguments = _parse_arguments(argv, prog)
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
    # Per-unit lines would have to come from inside the runner, which owns the
    # concurrency this benchmark ingests with; the run announces its size instead.
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    questions: tuple[SuperMemoryQuestion, ...],
    prepared: SuperMemoryPreparedSubject,
) -> tuple[SuperMemoryQuestionResult, ...]:
    async with connected_memory(arguments) as memory:
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
        abstained_question_count=sum(result.mindbridge_abstained for result in results),
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
    selected = select_by_id(
        tuple(question for question in questions if question.subject == subject),
        question_ids,
        key=lambda question: question.question_id,
        label=f"SuperMemory-VQA subject {subject} question IDs",
    )
    if not selected:
        raise ValueError(f"SuperMemory-VQA subject {subject} has no selected questions")
    return selected


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_supermemory", prog=prog, description=__doc__),
        device_id="supermemory_glasses",
    )
    parser.add_argument(
        "--prepared-media", type=Path, required=True, help="manifest of clips prepared for ingest"
    )
    parser.add_argument(
        "--subject", type=int, required=True, help="official subject whose videos this run replays"
    )
    parser.add_argument(
        "--dataset-revision", required=True, help="revision of the official dataset release"
    )
    parser.add_argument(
        "--source-revision", required=True, help="revision of the official video release"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        action="append",
        default=[],
        help="official question to run; repeatable, default the whole subject",
    )
    parsed = parser.parse_args(argv)
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
