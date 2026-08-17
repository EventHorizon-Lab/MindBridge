"""Reproducible MEMLENS memory-agent runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import Field, model_validator

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.memlens import (
    MEMLENS_ADAPTER_VERSION,
    MemLensQuestion,
    load_memlens,
    load_memlens_agent_subset,
)
from mindbridge.benchmarks.memlens_runner import (
    MemLensPreparedImages,
    MemLensQuestionResult,
    load_prepared_memlens,
    run_memlens_question,
    validate_memlens_images,
)
from mindbridge.benchmarks.prompts import MEMLENS_QUERY_PROMPT
from mindbridge.benchmarks.runner_cli import (
    BenchmarkRunManifest,
    MediaBenchmarkArguments,
    add_media_arguments,
    benchmark_parser,
    completed_now,
    connected_memory,
    media_argument_values,
    predictions_digest,
    select_by_id,
    shared_argument_values,
    write_run_artifacts,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import (
    ANSWER_FROM_EVIDENCE_PROMPT,
    PERCEIVE_EVENTS_PROMPT,
)

MEMLENS_RUNNER_VERSION = "memlens_production_api_v3"
MemLensContextWindow = Literal["32k", "64k", "128k", "256k"]


class MemLensRunManifest(BenchmarkRunManifest):
    """Immutable source, protocol, deployment, model, and prediction identity.

    Perception identity is optional here, unlike other media runners: a text-only MEMLENS run
    ingests no media at all, so it must record the absence rather than a version it never used.
    """

    benchmark: Literal["MEMLENS"] = "MEMLENS"
    context_window: MemLensContextWindow
    dataset_repository: NonEmptyString
    dataset_revision: NonEmptyString
    annotation_sha256: Sha256Hex
    evaluator_repository: NonEmptyString
    evaluator_revision: NonEmptyString
    agent_subset_sha256: Sha256Hex | None = None
    prepared_images_manifest_sha256: Sha256Hex | None = None
    text_only: bool
    perception_prompt_version: NonEmptyString | None = None
    query_prompt_version: NonEmptyString
    device_id: Identifier
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)
    question_ids: tuple[Identifier, ...] = Field(min_length=1)
    session_count: int = Field(gt=0)
    turn_count: int = Field(gt=0)
    image_reference_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_perception_identity_for_multimodal_run(self) -> MemLensRunManifest:
        perception = (
            self.perception_prompt_version,
            self.prepared_images_manifest_sha256,
        )
        if (self.text_only and any(item is not None for item in perception)) or (
            not self.text_only and any(item is None for item in perception)
        ):
            raise ValueError("MEMLENS multimodal identity must be present exactly when enabled")
        return self


@dataclass(frozen=True, slots=True)
class _Arguments(MediaBenchmarkArguments):
    prepared_images_path: Path | None
    agent_subset_path: Path | None
    context_window: MemLensContextWindow
    dataset_revision: str
    evaluator_revision: str
    text_only: bool
    question_ids: tuple[str, ...]


def main() -> None:
    """Run a full context split or its canonical 195-question agent subset."""
    arguments = _parse_arguments()
    subset_ids = (
        load_memlens_agent_subset(arguments.agent_subset_path)
        if arguments.agent_subset_path is not None
        else None
    )
    questions = _select_questions(
        load_memlens(arguments.dataset_path), subset_ids, arguments.question_ids
    )
    prepared = (
        load_prepared_memlens(arguments.prepared_images_path)
        if arguments.prepared_images_path is not None
        else None
    )
    if arguments.text_only:
        if prepared is not None:
            raise ValueError("text-only MEMLENS runs must omit prepared images")
    elif prepared is None:
        raise ValueError("multimodal MEMLENS runs require prepared images")
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(
        arguments.deployment_config_path,
        require_worker=not arguments.text_only,
    )
    results = asyncio.run(_run(arguments, questions, prepared))
    _write_artifacts(arguments, questions, results, deployment)


async def _run(
    arguments: _Arguments,
    questions: tuple[MemLensQuestion, ...],
    prepared: MemLensPreparedImages | None,
) -> tuple[MemLensQuestionResult, ...]:
    validate_memlens_images(questions, prepared, text_only=arguments.text_only)
    async with connected_memory(arguments) as memory:
        results = []
        for question in questions:
            results.append(
                await run_memlens_question(
                    memory,
                    question,
                    run_id=arguments.run_id,
                    prepared_images=prepared,
                    text_only=arguments.text_only,
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
    questions: tuple[MemLensQuestion, ...],
    results: tuple[MemLensQuestionResult, ...],
    deployment: LoadedDeployment,
) -> None:
    if tuple(result.question_id for result in results) != tuple(
        question.question_id for question in questions
    ):
        raise ValueError("MEMLENS predictions must match annotation question order")
    predictions = (
        json.dumps(
            {"data": [result.model_dump(mode="json") for result in results]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = MemLensRunManifest(
        context_window=arguments.context_window,
        runner_version=MEMLENS_RUNNER_VERSION,
        adapter_version=MEMLENS_ADAPTER_VERSION,
        dataset_repository="xiyuRenBill/MEMLENS",
        dataset_revision=arguments.dataset_revision,
        annotation_sha256=sha256_file(arguments.dataset_path),
        evaluator_repository="xrenaf/MEMLENS",
        evaluator_revision=arguments.evaluator_revision,
        agent_subset_sha256=(
            sha256_file(arguments.agent_subset_path)
            if arguments.agent_subset_path is not None
            else None
        ),
        prepared_images_manifest_sha256=(
            sha256_file(arguments.prepared_images_path)
            if arguments.prepared_images_path is not None
            else None
        ),
        text_only=arguments.text_only,
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        perception_prompt_version=(None if arguments.text_only else PERCEIVE_EVENTS_PROMPT.version),
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        query_prompt_version=MEMLENS_QUERY_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        device_id=arguments.device_id,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        poll_interval_seconds=arguments.poll_interval_seconds,
        processing_timeout_seconds=arguments.processing_timeout_seconds,
        question_ids=tuple(question.question_id for question in questions),
        session_count=sum(len(question.sessions) for question in questions),
        turn_count=sum(
            len(session.turns) for question in questions for session in question.sessions
        ),
        image_reference_count=sum(
            len(turn.images)
            for question in questions
            for session in question.sessions
            for turn in session.turns
        ),
        predictions_sha256=predictions_digest(predictions),
        completed_at=completed_now(),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_questions(
    questions: tuple[MemLensQuestion, ...],
    subset_ids: tuple[Identifier, ...] | None,
    question_ids: tuple[str, ...],
) -> tuple[MemLensQuestion, ...]:
    if subset_ids is not None:
        missing_subset = set(subset_ids) - {question.question_id for question in questions}
        if missing_subset:
            raise ValueError(
                f"MEMLENS agent subset contains unknown IDs: {', '.join(sorted(missing_subset))}"
            )
        questions = tuple(question for question in questions if question.question_id in subset_ids)
    selected = select_by_id(
        questions,
        question_ids,
        identify=lambda question: question.question_id,
        label="selected MEMLENS question",
    )
    if not selected:
        raise ValueError("MEMLENS selection must not be empty")
    return selected


def _parse_arguments() -> _Arguments:
    parser = benchmark_parser(tenant_prefix="benchmark_memlens")
    add_media_arguments(parser, device_id="memlens_conversation")
    parser.add_argument("--prepared-images", type=Path)
    parser.add_argument("--agent-subset-index", type=Path)
    parser.add_argument("--context-window", choices=("32k", "64k", "128k", "256k"), required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--evaluator-revision", required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--question-id", action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        **shared_argument_values(parsed),
        **media_argument_values(parsed),
        prepared_images_path=parsed.prepared_images,
        agent_subset_path=parsed.agent_subset_index,
        context_window=cast(MemLensContextWindow, parsed.context_window),
        dataset_revision=parsed.dataset_revision,
        evaluator_revision=parsed.evaluator_revision,
        text_only=parsed.text_only,
        question_ids=tuple(parsed.question_id),
    )


if __name__ == "__main__":
    main()
