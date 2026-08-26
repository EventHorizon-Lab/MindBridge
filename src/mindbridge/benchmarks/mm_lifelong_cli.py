"""Reproducible MM-Lifelong runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
    predictions_jsonl,
    report,
    scoring_snapshot,
    select_by_id,
    write_run_artifacts,
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
from mindbridge.benchmarks.scoring import JudgedAnswer
from mindbridge.contracts import NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

MM_LIFELONG_RUNNER_VERSION = "mm_lifelong_production_api_v2"


class MMLifelongRunManifest(MediaBenchmarkRunManifest):
    """Immutable data, media, deployment, model, and output identity for one run."""

    benchmark: Literal["MM-Lifelong"] = "MM-Lifelong"
    split: MMLifelongSplit
    source_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    question_indices: tuple[int, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    caption_segment_count: int = Field(ge=0)
    mean_unofficial_ref_at_300: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    split: MMLifelongSplit
    question_indices: tuple[int, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run one official split and emit JSONL accepted by its released evaluators."""
    arguments = _parse_arguments(argv, prog)
    questions = select_by_id(
        load_mm_lifelong(arguments.dataset_path, arguments.split),
        arguments.question_indices,
        key=lambda question: question.index,
        label="MM-Lifelong question indices",
        limit=arguments.limit,
    )
    prepared = load_prepared_mm_lifelong(arguments.prepared_media_path)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(segment.media_objects for segment in prepared.segments),
    )
    # Per-unit lines would have to come from inside the runner, which owns the
    # concurrency this benchmark ingests with; the run announces its size instead.
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


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
    predictions = predictions_jsonl(results)
    scoring = scoring_snapshot(
        "mm-lifelong",
        arguments,
        answers=tuple(JudgedAnswer(row.question, row.answer, row.pred.answer) for row in results),
        # The interval half is numeric and the runner already computes it, but it is not the
        # released Ref@N -- `unofficial_reference_at_n` says why -- so it travels under the name
        # that says so rather than as the benchmark's own metric.
        metrics={
            "unofficial_reference_at_300": (
                sum(row.mindbridge_unofficial_ref_at_300 for row in results) / len(results)
            )
        },
    )
    manifest = media_manifest(
        MMLifelongRunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=MM_LIFELONG_RUNNER_VERSION,
        adapter_version=MM_LIFELONG_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        split=arguments.split,
        source_repository="MM-Lifelong/MM-Lifelong",
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        question_indices=tuple(question.index for question in questions),
        segment_count=len(prepared.segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in prepared.segments),
        caption_segment_count=sum(segment.caption is not None for segment in prepared.segments),
        mean_unofficial_ref_at_300=(
            sum(result.mindbridge_unofficial_ref_at_300 for result in results) / len(results)
        ),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_mm_lifelong", prog=prog, description=__doc__),
        device_id="mm_lifelong_camera",
    )
    parser.add_argument(
        "--prepared-media", type=Path, required=True, help="manifest of prepared timeline segments"
    )
    parser.add_argument(
        "--split",
        choices=("day_test", "week_test", "month_train", "month_val"),
        required=True,
        help="official split to replay",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        action="append",
        default=[],
        help="official question index to run; repeatable, default the whole split",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        split=cast(MMLifelongSplit, parsed.split),
        question_indices=tuple(parsed.question_index),
    )


if __name__ == "__main__":
    main()
