"""Reproducible EgoLifeQA runner against a deployed MindBridge API."""

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
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

EGOLIFE_RUNNER_VERSION = "egolife_production_api_v6"


class EgoLifeRunManifest(MediaBenchmarkRunManifest):
    """Immutable data, deployment, code, and output identity for one run."""

    benchmark: Literal["EgoLifeQA"] = "EgoLifeQA"
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    subject_id: Identifier
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)
    metrics: EgoLifeMetrics


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    question_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected official questions and emit scored predictions plus a manifest."""
    arguments = _parse_arguments(argv, prog)
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
    # Per-unit lines would have to come from inside the runner, which owns the
    # concurrency this benchmark ingests with; the run announces its size instead.
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoLifeQuestion, ...],
    prepared: EgoLifePreparedStream,
) -> tuple[EgoLifeQuestionResult, ...]:
    async with connected_memory(arguments) as memory:
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
    manifest = media_manifest(
        EgoLifeRunManifest,
        arguments,
        deployment,
        runner_version=EGOLIFE_RUNNER_VERSION,
        adapter_version=EGOLIFE_QA_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="lmms-lab/EgoLife",
        evaluator_repository="EvolvingLMMs-Lab/EgoLife",
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        subject_id=prepared.subject_id,
        question_ids=tuple(question.question_id for question in questions),
        clip_count=len(prepared.clips),
        media_clip_count=media_clip_count,
        caption_clip_count=sum(clip.caption is not None for clip in prepared.clips),
        metrics=metrics,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_egolife", prog=prog, description=__doc__),
        device_id="egolife_camera",
    )
    parser.add_argument(
        "--prepared-media",
        type=Path,
        required=True,
        help="manifest of clips prepared from the official stream",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="official question to run; repeatable, default the whole release",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
