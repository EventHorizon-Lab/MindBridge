"""Reproducible LoCoMo command line runner against a deployed MindBridge API."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    CommonArguments,
    LoadedDeployment,
    RunManifestBase,
    common_benchmark_parser,
    load_deployment_snapshot,
    predictions_document,
    require_writable_output_pair,
    select_by_id,
    sha256_text,
    write_run_artifacts,
)
from mindbridge.benchmarks.locomo import LOCOMO_ADAPTER_VERSION, LoCoMoConversation, load_locomo
from mindbridge.benchmarks.locomo_runner import (
    LOCOMO_ABSTENTION,
    LOCOMO_PREDICTION_KEY,
    LoCoMoOfficialConversationResult,
    run_locomo_conversation,
)
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT
from mindbridge.sdk import MindBridge

LOCOMO_RUNNER_VERSION = "locomo_production_api_v10"


class LoCoMoRunManifest(RunManifestBase):
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
    # Vendors publish LoCoMo over four categories and the official release has five, so the same
    # system reports two different headline numbers. Recording the per-category denominators is
    # what lets a reader tell which of the two a run is quoting.
    category_question_counts: dict[int, int] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Arguments(CommonArguments):
    source_revision: str
    sample_ids: tuple[str, ...]


def main() -> None:
    """Run selected official conversations and emit predictions plus a manifest."""
    arguments = _parse_arguments()
    conversations = select_by_id(
        load_locomo(arguments.dataset_path),
        arguments.sample_ids,
        key=lambda conversation: conversation.sample_id,
        label="LoCoMo sample IDs",
    )
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    results = asyncio.run(_run_conversations(arguments, conversations))
    _write_artifacts(arguments, conversations, results, deployment)


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
) -> tuple[LoCoMoOfficialConversationResult, ...]:
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
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
    finally:
        await memory.close()


def _write_artifacts(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
    results: tuple[LoCoMoOfficialConversationResult, ...],
    deployment: LoadedDeployment,
) -> None:
    predictions = predictions_document([result.model_dump(mode="json") for result in results])
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
        category_question_counts=Counter(
            question.category
            for conversation in conversations
            for question in conversation.questions
        ),
        predictions_sha256=sha256_text(predictions),
        completed_at=datetime.now(timezone.utc),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser(
        parents=[common_benchmark_parser(tenant_prefix="benchmark_locomo")]
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        source_revision=parsed.source_revision,
        code_revision=parsed.code_revision,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        recall_limit=parsed.recall_limit,
        request_concurrency=parsed.request_concurrency,
        request_timeout_seconds=parsed.request_timeout_seconds,
        sample_ids=tuple(parsed.sample_id),
        overwrite=parsed.overwrite,
    )


if __name__ == "__main__":
    main()
