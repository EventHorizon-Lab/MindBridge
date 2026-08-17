"""Reproducible EgoMemReason runner against a deployed MindBridge API."""

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
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
)
from mindbridge.sdk import MindBridge

EGOMEM_RUNNER_VERSION = "egomem_production_api_v3"


class EgoMemRunManifest(MediaRunManifestBase):
    """Immutable input, deployment, model, and submission identity for one run."""

    benchmark: Literal["EgoMemReason"] = "EgoMemReason"
    perception_prompt_version: NonEmptyString
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    prepared_media_manifest_sha256: Sha256Hex
    query_prompt_version: NonEmptyString
    identities: tuple[Identifier, ...] = Field(min_length=1)
    example_ids: tuple[int, ...] = Field(min_length=1)
    clip_count: int = Field(gt=0)
    media_clip_count: int = Field(ge=0)
    caption_clip_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(MediaArguments):
    prepared_media_path: Path
    dataset_revision: str
    evaluator_revision: str
    example_ids: tuple[int, ...]


def main() -> None:
    """Run selected questions and emit the exact public leaderboard submission shape."""
    arguments = _parse_arguments()
    questions = select_by_id(
        load_egomem_reason(arguments.dataset_path),
        arguments.example_ids,
        key=lambda question: question.example_id,
        label="EgoMemReason example IDs",
    )
    prepared = _prepared_by_identity(questions, load_prepared_egomem(arguments.prepared_media_path))
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=any(
            clip.media_object is not None for stream in prepared.values() for clip in stream.clips
        ),
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, prepared, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[EgoMemReasonQuestion, ...],
    prepared: dict[str, EgoLifePreparedStream],
) -> tuple[EgoMemReasonResult, ...]:
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        by_example_id: dict[int, EgoMemReasonResult] = {}
        for identity in dict.fromkeys(question.identity for question in questions):
            identity_questions = tuple(
                question for question in questions if question.identity == identity
            )
            results = await run_egomem_reason(
                memory,
                identity_questions,
                prepared[identity],
                run_id=arguments.run_id,
                tenant_prefix=arguments.tenant_prefix,
                device_id=arguments.device_id,
                recall_limit=arguments.recall_limit,
                request_concurrency=arguments.request_concurrency,
                poll_interval_seconds=arguments.poll_interval_seconds,
                processing_timeout_seconds=arguments.processing_timeout_seconds,
            )
            by_example_id.update((result.example_id, result) for result in results)
        return tuple(by_example_id[question.example_id] for question in questions)
    finally:
        await memory.close()


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
    predictions = predictions_document(
        [
            {
                "example_id": result.example_id,
                "predicted_answer": result.predicted_answer,
            }
            for result in results
        ],
    )
    streams = tuple(prepared.values())
    manifest = EgoMemRunManifest(
        runner_version=EGOMEM_RUNNER_VERSION,
        adapter_version=EGOMEM_REASON_ADAPTER_VERSION,
        dataset_repository="Ted412/EgoMemReason",
        dataset_revision=arguments.dataset_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        evaluator_repository="Ziyang412/EgoMemReason",
        evaluator_revision=arguments.evaluator_revision,
        prepared_media_manifest_sha256=sha256_file(arguments.prepared_media_path),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        perception_prompt_version=PERCEIVE_EVENTS_PROMPT.version,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        query_prompt_version=EGOMEM_REASON_QUERY_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        identities=tuple(prepared),
        example_ids=tuple(question.example_id for question in questions),
        clip_count=sum(len(stream.clips) for stream in streams),
        media_clip_count=sum(
            clip.media_object is not None for stream in streams for clip in stream.clips
        ),
        caption_clip_count=sum(
            clip.caption is not None for stream in streams for clip in stream.clips
        ),
        predictions_sha256=sha256_text(predictions),
        completed_at=datetime.now(timezone.utc),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _prepared_by_identity(
    questions: tuple[EgoMemReasonQuestion, ...],
    streams: tuple[EgoLifePreparedStream, ...],
) -> dict[str, EgoLifePreparedStream]:
    by_identity = {stream.subject_id: stream for stream in streams}
    missing = {question.identity for question in questions} - by_identity.keys()
    if missing:
        raise ValueError(f"missing prepared EgoMemReason identities: {', '.join(sorted(missing))}")
    return {
        identity: by_identity[identity]
        for identity in dict.fromkeys(question.identity for question in questions)
    }


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        parents=[
            media_benchmark_parser(tenant_prefix="benchmark_egomem", device_id="egolife_camera")
        ]
    )
    parser.add_argument("--prepared-media", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--example-id", type=int, action="append", default=[])
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
        example_ids=tuple(parsed.example_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
