"""Reproducible LoCoMo command line runner against a deployed MindBridge API."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from mindbridge.benchmarks.artifacts import (
    LoadedDeployment,
    load_deployment_snapshot,
    require_writable_output_pair,
)
from mindbridge.benchmarks.cli_common import (
    BenchmarkRunManifest,
    CoreArguments,
    connected_memory,
    core_arguments,
    core_manifest,
    core_parser,
    report,
    report_unit,
    select_by_id,
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
    # Vendor LoCoMo numbers are mostly four-category, dropping adversarial (category 5), which is
    # the single largest reason two published scores are not comparable. Keyed by official
    # category so a reader can tell which protocol a number came from without rerunning it.
    category_question_counts: dict[int, int] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Arguments(CoreArguments):
    source_revision: str
    sample_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected official conversations and emit predictions plus a manifest."""
    arguments = _parse_arguments(argv, prog)
    conversations = select_by_id(
        load_locomo(arguments.dataset_path),
        arguments.sample_ids,
        key=lambda conversation: conversation.sample_id,
        label="LoCoMo sample IDs",
    )
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    report(f"running {len(conversations)} conversations", quiet=arguments.quiet)
    results = asyncio.run(_run_conversations(arguments, conversations))
    _write_artifacts(arguments, conversations, results, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoConversation, ...],
) -> tuple[LoCoMoOfficialConversationResult, ...]:
    async with connected_memory(arguments) as memory:
        results: list[LoCoMoOfficialConversationResult] = []
        for index, conversation in enumerate(conversations, start=1):
            report_unit(
                f"conversation {conversation.sample_id}",
                index=index,
                total=len(conversations),
                quiet=arguments.quiet,
            )
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
    predictions = (
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    manifest = core_manifest(
        LoCoMoRunManifest,
        arguments,
        deployment,
        runner_version=LOCOMO_RUNNER_VERSION,
        adapter_version=LOCOMO_ADAPTER_VERSION,
        predictions=predictions,
        source_repository="snap-research/locomo",
        source_revision=arguments.source_revision,
        source_sha256=sha256_file(arguments.dataset_path),
        prediction_key=LOCOMO_PREDICTION_KEY,
        abstention_text=LOCOMO_ABSTENTION,
        sample_ids=tuple(conversation.sample_id for conversation in conversations),
        memory_item_count=sum(len(conversation.turns) for conversation in conversations),
        question_count=sum(len(conversation.questions) for conversation in conversations),
        abstained_question_count=sum(
            question.mindbridge_abstained for result in results for question in result.qa
        ),
        category_question_counts=dict(
            sorted(
                Counter(question.category for result in results for question in result.qa).items()
            )
        ),
    )
    write_run_artifacts(arguments.output_path, predictions, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = core_parser(tenant_prefix="benchmark_locomo", prog=prog, description=__doc__)
    parser.add_argument(
        "--source-revision", required=True, help="revision of the official LoCoMo release"
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="official conversation to run; repeatable, default the whole release",
    )
    parsed = parser.parse_args(argv)
    return core_arguments(
        _Arguments,
        parsed,
        source_revision=parsed.source_revision,
        sample_ids=tuple(parsed.sample_id),
    )


if __name__ == "__main__":
    main()
