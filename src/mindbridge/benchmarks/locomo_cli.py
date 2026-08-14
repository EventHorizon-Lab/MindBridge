"""Reproducible LoCoMo command line runner against a deployed MindBridge API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, Field

from mindbridge.application.recall import RETRIEVAL_DOCUMENT_EMBEDDING_TASK
from mindbridge.benchmarks.artifacts import (
    require_writable_output_pair,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.locomo import LOCOMO_ADAPTER_VERSION, LoCoMoConversation, load_locomo
from mindbridge.benchmarks.locomo_runner import (
    LOCOMO_ABSTENTION,
    LOCOMO_PREDICTION_KEY,
    LoCoMoOfficialConversationResult,
    run_locomo_conversation,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file
from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
)
from mindbridge.models.openai_chat import REASONING_EFFORT_VALUES
from mindbridge.models.openai_omni import (
    ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
    DEFAULT_OMNI_MODEL_ID,
)
from mindbridge.sdk import AsyncMindBridge

LOCOMO_RUNNER_VERSION = "locomo_production_api_v8"


class LoCoMoRunManifest(ContractModel):
    """Immutable dataset, deployment, code, and output identity for one run."""

    benchmark: Literal["LoCoMo"] = "LoCoMo"
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    source_repository: NonEmptyString
    source_revision: NonEmptyString
    source_sha256: Sha256Hex
    code_revision: NonEmptyString
    answer_model_id: NonEmptyString
    answer_model_revision: NonEmptyString
    answer_prompt_version: NonEmptyString
    reasoning_effort: NonEmptyString
    embedding_model_id: NonEmptyString
    embedding_model_revision: NonEmptyString
    retrieval_task: NonEmptyString
    prediction_key: NonEmptyString
    abstention_text: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    memory_item_count: int = Field(gt=0)
    question_count: int = Field(gt=0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    dataset_path: Path
    output_path: Path
    api_base_url: str
    source_revision: str
    code_revision: str
    answer_model_id: str
    answer_model_revision: str
    answer_reasoning_effort: str
    embedding_model_id: str
    embedding_model_revision: str
    run_id: str
    tenant_prefix: str
    recall_limit: int
    request_concurrency: int
    request_timeout_seconds: float
    sample_ids: tuple[str, ...]
    overwrite: bool


def main() -> None:
    """Run selected official conversations and emit predictions plus a manifest."""
    arguments = _parse_arguments()
    conversations = _select_conversations(load_locomo(arguments.dataset_path), arguments.sample_ids)
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    results = asyncio.run(_run_conversations(arguments, conversations))
    _write_artifacts(arguments, conversations, results)


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
) -> tuple[LoCoMoOfficialConversationResult, ...]:
    memory = AsyncMindBridge.connect(
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
) -> None:
    predictions = (
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = LoCoMoRunManifest(
        runner_version=LOCOMO_RUNNER_VERSION,
        adapter_version=LOCOMO_ADAPTER_VERSION,
        source_repository="snap-research/locomo",
        source_revision=arguments.source_revision,
        source_sha256=sha256_file(arguments.dataset_path),
        code_revision=arguments.code_revision,
        answer_model_id=arguments.answer_model_id,
        answer_model_revision=arguments.answer_model_revision,
        answer_prompt_version=ANSWER_FROM_EVIDENCE_PROMPT_VERSION,
        reasoning_effort=arguments.answer_reasoning_effort,
        embedding_model_id=arguments.embedding_model_id,
        embedding_model_revision=arguments.embedding_model_revision,
        retrieval_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
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
        predictions_sha256=hashlib.sha256(predictions.encode("utf-8")).hexdigest(),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(arguments.output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _select_conversations(
    conversations: tuple[LoCoMoConversation, ...],
    sample_ids: tuple[str, ...],
) -> tuple[LoCoMoConversation, ...]:
    if not sample_ids:
        return conversations
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must not contain duplicates")
    requested = set(sample_ids)
    selected = tuple(item for item in conversations if item.sample_id in requested)
    missing = requested - {item.sample_id for item in selected}
    if missing:
        raise ValueError(f"unknown LoCoMo sample IDs: {', '.join(sorted(missing))}")
    return selected


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--answer-model-id", default=DEFAULT_OMNI_MODEL_ID)
    parser.add_argument("--answer-model-revision", required=True)
    parser.add_argument(
        "--answer-reasoning-effort",
        choices=("omitted", *REASONING_EFFORT_VALUES),
        required=True,
    )
    parser.add_argument("--embedding-model-id", default=DEFAULT_JINA_OMNI_MODEL_ID)
    parser.add_argument("--embedding-model-revision", default=DEFAULT_JINA_OMNI_REVISION)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="benchmark_locomo")
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args()
    return _Arguments(
        dataset_path=parsed.dataset,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        source_revision=parsed.source_revision,
        code_revision=parsed.code_revision,
        answer_model_id=parsed.answer_model_id,
        answer_model_revision=parsed.answer_model_revision,
        answer_reasoning_effort=parsed.answer_reasoning_effort,
        embedding_model_id=parsed.embedding_model_id,
        embedding_model_revision=parsed.embedding_model_revision,
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
