"""Reproducible EgoTempo runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
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
    index_prepared,
    media_arguments,
    media_manifest,
    report,
    run_units,
    scoring_snapshot,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.egotempo import (
    EGOTEMPO_ADAPTER_VERSION,
    EgoTempoQuestion,
    EgoTempoQuestionResult,
    load_egotempo,
    run_egotempo_clip,
)
from mindbridge.benchmarks.prompts import EGOTEMPO_QUERY_PROMPT
from mindbridge.benchmarks.runtime import PreparedVideo, load_prepared_videos
from mindbridge.benchmarks.scoring import JudgedAnswer, require_scoring_is_possible
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

EGOTEMPO_RUNNER_VERSION = "egotempo_production_api_v2"


class EgoTempoRunManifest(MediaBenchmarkRunManifest):
    """Immutable data, evaluator, deployment, and output identity for one run."""

    benchmark: Literal["EgoTempo"] = "EgoTempo"
    source_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    benchmark_prompt_version: NonEmptyString
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    clip_ids: tuple[Identifier, ...] = Field(min_length=1)
    segment_count: int = Field(gt=0)
    media_segment_count: int = Field(ge=0)
    transcript_segment_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    question_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected questions and emit JSON accepted by the official judge notebook."""
    arguments = _parse_arguments(argv, prog)
    questions = select_by_id(
        load_egotempo(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="EgoTempo question IDs",
        limit=arguments.limit,
    )
    prepared = _select_prepared(load_prepared_videos(arguments.prepared_media_path), questions)
    require_scoring_is_possible("egotempo", predict_only=arguments.predict_only)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            segment.media_objects for video in prepared for segment in video.segments
        ),
    )
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoTempoQuestion, ...],
    prepared: tuple[PreparedVideo, ...],
) -> tuple[EgoTempoQuestionResult, ...]:
    questions_by_clip: dict[str, list[EgoTempoQuestion]] = {}
    for question in questions:
        questions_by_clip.setdefault(question.clip_id, []).append(question)
    prepared_by_id = {video.video_id: video for video in prepared}
    async with connected_memory(arguments) as memory:
        per_clip = await run_units(
            tuple(questions_by_clip.items()),
            label=lambda clip: f"clip {clip[0]}",
            unit_concurrency=arguments.unit_concurrency,
            quiet=arguments.quiet,
            run=lambda clip: run_egotempo_clip(
                memory,
                tuple(clip[1]),
                prepared_by_id[clip[0]],
                run_id=arguments.run_id,
                tenant_prefix=arguments.tenant_prefix,
                device_id=arguments.device_id,
                recall_limit=arguments.recall_limit,
                request_concurrency=arguments.request_concurrency,
                poll_interval_seconds=arguments.poll_interval_seconds,
                processing_timeout_seconds=arguments.processing_timeout_seconds,
            ),
        )
        by_id = {result.question_id: result for result in chain.from_iterable(per_clip)}
        return tuple(by_id[question.question_id] for question in questions)


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
    scoring = scoring_snapshot(
        "egotempo",
        arguments,
        answers=tuple(
            JudgedAnswer(row.question, row.reference_answer, row.model_answer) for row in results
        ),
    )
    manifest = media_manifest(
        EgoTempoRunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=EGOTEMPO_RUNNER_VERSION,
        adapter_version=EGOTEMPO_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        source_repository="google-research-datasets/egotempo",
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        benchmark_prompt_version=EGOTEMPO_QUERY_PROMPT.version,
        question_ids=tuple(question.question_id for question in questions),
        clip_ids=tuple(dict.fromkeys(question.clip_id for question in questions)),
        segment_count=len(segments),
        media_segment_count=sum(bool(segment.media_objects) for segment in segments),
        transcript_segment_count=sum(segment.transcript is not None for segment in segments),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_prepared(
    prepared: tuple[PreparedVideo, ...], questions: tuple[EgoTempoQuestion, ...]
) -> tuple[PreparedVideo, ...]:
    clip_ids = tuple(dict.fromkeys(question.clip_id for question in questions))
    by_id = index_prepared(
        clip_ids,
        prepared,
        key=lambda video: video.video_id,
        label="EgoTempo clips",
    )
    return tuple(by_id[clip_id] for clip_id in clip_ids)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_egotempo", prog=prog, description=__doc__),
        device_id="egotempo_camera",
    )
    parser.add_argument(
        "--prepared-media", type=Path, required=True, help="manifest of clips prepared for ingest"
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
