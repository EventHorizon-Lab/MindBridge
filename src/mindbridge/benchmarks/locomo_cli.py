"""Reproducible LoCoMo command line runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.locomo import LOCOMO_ADAPTER_VERSION, LoCoMoConversation, load_locomo
from mindbridge.benchmarks.locomo_runner import (
    LOCOMO_ABSTENTION,
    LOCOMO_PREDICTION_KEY,
    LoCoMoOfficialConversationResult,
    run_locomo_conversation,
)
from mindbridge.benchmarks.runner_cli import (
    BenchmarkArguments,
    BenchmarkRunManifest,
    benchmark_parser,
    completed_now,
    connected_memory,
    json_predictions,
    predictions_digest,
    select_by_id,
    shared_argument_values,
    write_run_artifacts,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT

LOCOMO_RUNNER_VERSION = "locomo_production_api_v9"


class LoCoMoRunManifest(BenchmarkRunManifest):
    """Immutable dataset, deployment, code, and output identity for one run."""

    benchmark: Literal["LoCoMo"] = "LoCoMo"
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    source_sha256: Sha256Hex
    prediction_key: NonEmptyString
    abstention_text: NonEmptyString
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    memory_item_count: int = Field(gt=0)
    question_count: int = Field(gt=0)
    abstained_question_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _Arguments(BenchmarkArguments):
    source_revision: str
    sample_ids: tuple[str, ...]


def main() -> None:
    """Run selected official conversations and emit predictions plus a manifest."""
    arguments = _parse_arguments()
    conversations = _select_conversations(load_locomo(arguments.dataset_path), arguments.sample_ids)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    results = asyncio.run(_run_conversations(arguments, conversations))
    _write_artifacts(arguments, conversations, results, deployment)


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
) -> tuple[LoCoMoOfficialConversationResult, ...]:
    async with connected_memory(arguments) as memory:
        results: list[LoCoMoOfficialConversationResult] = []
        for conversation in conversations:
            results.append(
                await run_locomo_conversation(
                    memory,
                    conversation,
                    run_id=arguments.run_id,
                    tenant_prefix=arguments.tenant_prefix,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                )
            )
        return tuple(results)


def _write_artifacts(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
    results: tuple[LoCoMoOfficialConversationResult, ...],
    deployment: LoadedDeployment,
) -> None:
    predictions = json_predictions(results)
    manifest = LoCoMoRunManifest(
        runner_version=LOCOMO_RUNNER_VERSION,
        adapter_version=LOCOMO_ADAPTER_VERSION,
        source_repository="snap-research/locomo",
        source_revision=arguments.source_revision,
        source_sha256=sha256_file(arguments.dataset_path),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT.version,
        retrieval_task=EmbedTask.DOCUMENT.value,
        prediction_key=LOCOMO_PREDICTION_KEY,
        abstention_text=LOCOMO_ABSTENTION,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        sample_ids=tuple(conversation.sample_id for conversation in conversations),
        memory_item_count=sum(len(conversation.turns) for conversation in conversations),
        question_count=sum(len(conversation.questions) for conversation in conversations),
        abstained_question_count=sum(
            question.mindbridge_abstained for result in results for question in result.qa
        ),
        predictions_sha256=predictions_digest(predictions),
        completed_at=completed_now(),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _select_conversations(
    conversations: tuple[LoCoMoConversation, ...],
    sample_ids: tuple[str, ...],
) -> tuple[LoCoMoConversation, ...]:
    return select_by_id(
        conversations,
        sample_ids,
        identify=lambda conversation: conversation.sample_id,
        label="LoCoMo sample",
    )


def _parse_arguments() -> _Arguments:
    parser = benchmark_parser(tenant_prefix="benchmark_locomo")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        **shared_argument_values(parsed),
        source_revision=parsed.source_revision,
        sample_ids=tuple(parsed.sample_id),
    )


if __name__ == "__main__":
    main()
