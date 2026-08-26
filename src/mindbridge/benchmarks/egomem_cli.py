"""Reproducible EgoMemReason runner against a deployed MindBridge API."""

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
from mindbridge.benchmarks.egolife_runner import (
    EgoLifePreparedStream,
    EgoMemReasonResult,
    load_prepared_egomem,
    run_egomem_reason,
)
from mindbridge.benchmarks.egomem_reason import (
    EGOMEM_REASON_ADAPTER_VERSION,
    EgoMemReasonQuestion,
    load_egomem_reason,
)
from mindbridge.benchmarks.prompts import EGOMEM_REASON_QUERY_PROMPT
from mindbridge.benchmarks.scoring import require_scoring_is_possible
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

EGOMEM_RUNNER_VERSION = "egomem_production_api_v3"


class EgoMemRunManifest(MediaBenchmarkRunManifest):
    """Immutable input, deployment, model, and submission identity for one run."""

    benchmark: Literal["EgoMemReason"] = "EgoMemReason"
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    query_prompt_version: NonEmptyString
    identities: tuple[Identifier, ...] = Field(min_length=1)
    example_ids: tuple[int, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)
    abstained_question_count: int = Field(ge=0)
    error_question_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    example_ids: tuple[int, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected questions and emit the exact public leaderboard submission shape."""
    arguments = _parse_arguments(argv, prog)
    questions = select_by_id(
        load_egomem_reason(arguments.dataset_path),
        arguments.example_ids,
        key=lambda question: question.example_id,
        label="EgoMemReason example IDs",
        limit=arguments.limit,
    )
    prepared = _prepared_by_identity(questions, load_prepared_egomem(arguments.prepared_media_path))
    require_scoring_is_possible("egomem", predict_only=arguments.predict_only)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            clip.media_object is not None for stream in prepared.values() for clip in stream.clips
        ),
    )
    report(f"running {len(questions)} questions", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoMemReasonQuestion, ...],
    prepared: dict[str, EgoLifePreparedStream],
) -> tuple[EgoMemReasonResult, ...]:
    async with connected_memory(arguments) as memory:
        identities = tuple(dict.fromkeys(question.identity for question in questions))
        per_identity = await run_units(
            identities,
            label=lambda identity: f"identity {identity}",
            unit_concurrency=arguments.unit_concurrency,
            quiet=arguments.quiet,
            run=lambda identity: run_egomem_reason(
                memory,
                tuple(question for question in questions if question.identity == identity),
                prepared[identity],
                run_id=arguments.run_id,
                tenant_prefix=arguments.tenant_prefix,
                device_id=arguments.device_id,
                recall_limit=arguments.recall_limit,
                request_concurrency=arguments.request_concurrency,
                poll_interval_seconds=arguments.poll_interval_seconds,
                processing_timeout_seconds=arguments.processing_timeout_seconds,
            ),
        )
        by_example_id = {result.example_id: result for result in chain.from_iterable(per_identity)}
        return tuple(by_example_id[question.example_id] for question in questions)


def _write_artifacts(
    arguments: _Arguments,
    questions: tuple[EgoMemReasonQuestion, ...],
    prepared: dict[str, EgoLifePreparedStream],
    results: tuple[EgoMemReasonResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.example_id for result in results) != tuple(
        question.example_id for question in questions
    ):
        raise ValueError("EgoMemReason predictions must match annotation question order")
    predictions = (
        json.dumps(
            [
                {
                    "example_id": result.example_id,
                    "predicted_answer": result.predicted_answer,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    streams = tuple(prepared.values())
    # EgoMemReason withholds its answers, which is MMBench's test split upstream: the run emits
    # what a leaderboard would read and declares `submission` rather than reporting a score.
    scoring = scoring_snapshot("egomem", arguments)
    manifest = media_manifest(
        EgoMemRunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=EGOMEM_RUNNER_VERSION,
        adapter_version=EGOMEM_REASON_ADAPTER_VERSION,
        annotation_sha256=sha256_file(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="Ted412/EgoMemReason",
        evaluator_repository="Ziyang412/EgoMemReason",
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        query_prompt_version=EGOMEM_REASON_QUERY_PROMPT.version,
        identities=tuple(prepared),
        example_ids=tuple(question.example_id for question in questions),
        clip_count=sum(len(stream.clips) for stream in streams),
        media_clip_count=sum(
            clip.media_object is not None for stream in streams for clip in stream.clips
        ),
        caption_clip_count=sum(
            clip.caption is not None for stream in streams for clip in stream.clips
        ),
        abstained_question_count=sum(result.mindbridge_abstained for result in results),
        error_question_count=sum(result.mindbridge_error_code is not None for result in results),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _prepared_by_identity(
    questions: tuple[EgoMemReasonQuestion, ...],
    streams: tuple[EgoLifePreparedStream, ...],
) -> dict[str, EgoLifePreparedStream]:
    identities = tuple(dict.fromkeys(question.identity for question in questions))
    by_identity = index_prepared(
        identities,
        streams,
        key=lambda stream: stream.subject_id,
        label="EgoMemReason identities",
    )
    return {identity: by_identity[identity] for identity in identities}


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_egomem", prog=prog, description=__doc__),
        device_id="egolife_camera",
    )
    parser.add_argument(
        "--prepared-media",
        type=Path,
        required=True,
        help="manifest of clips prepared from the official streams",
    )
    parser.add_argument(
        "--example-id",
        type=int,
        action="append",
        default=[],
        help="official example to run; repeatable, default the whole release",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_media_path=parsed.prepared_media,
        example_ids=tuple(parsed.example_id),
    )


if __name__ == "__main__":
    main()
