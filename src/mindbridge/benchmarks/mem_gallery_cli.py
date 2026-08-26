"""Reproducible Mem-Gallery runner against a deployed MindBridge API."""

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
    report_unit,
    scoring_snapshot,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryTopic,
    load_mem_gallery,
    mem_gallery_dialog_digest,
)
from mindbridge.benchmarks.mem_gallery_runner import (
    MemGalleryPreparedImages,
    MemGalleryQuestionResult,
    load_prepared_mem_gallery,
    run_mem_gallery_topic,
    validate_mem_gallery_images,
)
from mindbridge.benchmarks.prompts import MEM_GALLERY_QUERY_PROMPT
from mindbridge.benchmarks.scoring import JudgedAnswer, require_scoring_is_possible
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.prompts import PERCEIVE_EVENTS_PROMPT

MEM_GALLERY_RUNNER_VERSION = "mem_gallery_production_api_v1"


class MemGalleryRunManifest(MediaBenchmarkRunManifest):
    """Immutable source, protocol, deployment, model, and prediction identity."""

    benchmark: Literal["Mem-Gallery"] = "Mem-Gallery"
    dataset_repository: NonEmptyString
    evaluator_repository: NonEmptyString
    prepared_images_manifest_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    query_prompt_version: NonEmptyString
    topics: tuple[Identifier, ...] = Field(min_length=1)
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    session_count: int = Field(gt=0)
    round_count: int = Field(gt=0)
    image_reference_count: int = Field(ge=0)
    question_image_count: int = Field(ge=0)
    ingest_failure_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_images_path: Path
    topics: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run one tenant per topic over the official dialogue directory."""
    arguments = _parse_arguments(argv, prog)
    topics = select_by_id(
        load_mem_gallery(arguments.dataset_path),
        arguments.topics,
        key=lambda topic: topic.topic,
        label="selected Mem-Gallery topics",
        limit=arguments.limit,
    )
    if not topics:
        raise ValueError("Mem-Gallery selection must not be empty")
    prepared = load_prepared_mem_gallery(arguments.prepared_images_path)
    validate_mem_gallery_images(topics, prepared)
    require_scoring_is_possible("mem-gallery", predict_only=arguments.predict_only)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path, require_worker=True)
    report(f"running {len(topics)} topics", quiet=arguments.quiet)
    results = asyncio.run(_run(arguments, topics, prepared))
    _write_artifacts(arguments, topics, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run(
    arguments: _Arguments,
    topics: tuple[MemGalleryTopic, ...],
    prepared: MemGalleryPreparedImages,
) -> tuple[MemGalleryQuestionResult, ...]:
    async with connected_memory(arguments) as memory:
        results: list[MemGalleryQuestionResult] = []
        for index, topic in enumerate(topics, start=1):
            report_unit(
                f"topic {topic.topic}", index=index, total=len(topics), quiet=arguments.quiet
            )
            results.extend(
                await run_mem_gallery_topic(
                    memory,
                    topic,
                    run_id=arguments.run_id,
                    prepared=prepared,
                    tenant_prefix=arguments.tenant_prefix,
                    device_id=arguments.device_id,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                    poll_interval_seconds=arguments.poll_interval_seconds,
                    processing_timeout_seconds=arguments.processing_timeout_seconds,
                )
            )
        return tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    topics: tuple[MemGalleryTopic, ...],
    results: tuple[MemGalleryQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    expected = tuple(question.question_id for topic in topics for question in topic.questions)
    if tuple(result.question_id for result in results) != expected:
        raise ValueError("Mem-Gallery predictions must match annotation question order")
    # The official evaluator reads a list of per-question objects with `point` as category.
    predictions = (
        json.dumps(
            [
                {
                    "question_id": result.question_id,
                    "topic": result.topic,
                    "point": result.point,
                    "question": result.question,
                    "answer": result.reference_answer,
                    "prediction": result.prediction,
                    "clue": list(result.clue_round_ids),
                    "retrieved_ids": list(result.mindbridge_round_ids),
                    "retrieved_clue_round_count": result.retrieved_clue_round_count,
                    "retrieved_image_ids": list(result.mindbridge_media_object_ids),
                    "mindbridge_confidence": result.mindbridge_confidence,
                    "mindbridge_memory_ids": list(result.mindbridge_memory_ids),
                    "mindbridge_trace_id": result.mindbridge_trace_id,
                    "mindbridge_ingest_failure_count": result.mindbridge_ingest_failure_count,
                }
                for result in results
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    # Every result in one topic carries that topic's own ingest failure count, so keying by
    # topic and summing dedupes them into the run-wide total without needing `_run` to thread
    # a separate count alongside `results` -- the same count `AtmRunManifest` pins, just
    # summed across this release's twenty tenants instead of ATM's one.
    ingest_failure_count = sum(
        {result.topic: result.mindbridge_ingest_failure_count for result in results}.values()
    )
    scoring = scoring_snapshot(
        "mem-gallery",
        arguments,
        answers=tuple(
            JudgedAnswer(row.question, row.reference_answer, row.prediction) for row in results
        ),
    )
    manifest = media_manifest(
        MemGalleryRunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=MEM_GALLERY_RUNNER_VERSION,
        adapter_version=MEM_GALLERY_ADAPTER_VERSION,
        annotation_sha256=mem_gallery_dialog_digest(arguments.dataset_path),
        predictions=predictions,
        dataset_repository="Ethan-Bei/Mem-Gallery",
        evaluator_repository="YuanchenBei/Mem-Gallery",
        prepared_images_manifest_sha256=sha256_file(arguments.prepared_images_path),
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        query_prompt_version=MEM_GALLERY_QUERY_PROMPT.version,
        topics=tuple(topic.topic for topic in topics),
        question_ids=expected,
        session_count=sum(len(topic.sessions) for topic in topics),
        round_count=sum(len(session.rounds) for topic in topics for session in topic.sessions),
        image_reference_count=sum(
            1
            for topic in topics
            for session in topic.sessions
            for round_ in session.rounds
            if round_.image_id is not None
        ),
        question_image_count=sum(
            1
            for topic in topics
            for question in topic.questions
            if question.question_image_path is not None
        ),
        ingest_failure_count=ingest_failure_count,
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = add_media_arguments(
        core_parser(tenant_prefix="benchmark_mem_gallery", prog=prog, description=__doc__),
        device_id="mem_gallery_conversation",
    )
    parser.add_argument(
        "--prepared-images",
        type=Path,
        required=True,
        help="manifest of staged dialogue and question images",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="official topic to run; repeatable, default all twenty",
    )
    parsed = parser.parse_args(argv)
    return media_arguments(
        _Arguments,
        parsed,
        prepared_images_path=parsed.prepared_images,
        topics=tuple(parsed.topic),
    )


if __name__ == "__main__":
    main()
