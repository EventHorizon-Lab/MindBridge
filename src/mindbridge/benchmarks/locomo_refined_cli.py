"""Reproducible LoCoMo-Refined runner against a deployed MindBridge API.

Writes `predictions.jsonl` in the shape `mem-eval-suite/LoCoMo_refined` scores
directly: one `{"qa_id", "predicted_answer"}` object per line, ready for that
repository's `./scripts/run_eval.sh`.
"""

from __future__ import annotations

import asyncio
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
    predictions_jsonl,
    report,
    report_unit,
    scoring_snapshot,
    select_by_id,
    write_run_artifacts,
)
from mindbridge.benchmarks.locomo_refined import (
    LOCOMO_REFINED_ADAPTER_VERSION,
    LoCoMoRefinedConversation,
    load_locomo_refined,
)
from mindbridge.benchmarks.locomo_refined_runner import (
    LOCOMO_REFINED_PREDICTION_KEY,
    LoCoMoRefinedPrediction,
    run_locomo_refined_conversation,
)
from mindbridge.benchmarks.scoring import JudgedAnswer
from mindbridge.contracts import Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file

LOCOMO_REFINED_RUNNER_VERSION = "locomo_refined_production_api_v1"


class LoCoMoRefinedRunManifest(BenchmarkRunManifest):
    """Immutable dataset, deployment, code, and output identity for one run."""

    benchmark: Literal["LoCoMo-Refined"] = "LoCoMo-Refined"
    source_repository: NonEmptyString
    source_sha256: Sha256Hex
    prediction_key: NonEmptyString
    sample_ids: tuple[Identifier, ...] = Field(min_length=1)
    memory_item_count: int = Field(gt=0)
    question_count: int = Field(gt=0)
    # Rows whose recall produced no answer at all, written as an empty
    # `predicted_answer`. LoCoMo-Refined has no adversarial category and therefore no
    # gold abstention, so these are plain misses rather than a second protocol -- but a
    # run whose score is dragged down by silence rather than by wrong answers is a
    # different diagnosis, and only this count separates the two.
    unanswered_question_count: int = Field(ge=0)
    # LoCoMo's four-versus-five-category ambiguity is gone with the adversarial split, so
    # these counts no longer decide whether two numbers are comparable. They stay because
    # the refined release is deliberately uneven across categories (802 of 1,382 questions
    # are category 4), which a subset run can skew without saying so.
    category_question_counts: dict[int, int] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Arguments(CoreArguments):
    sample_ids: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Run selected official conversations and emit predictions plus a manifest."""
    arguments = _parse_arguments(argv, prog)
    conversations = select_by_id(
        load_locomo_refined(arguments.dataset_path),
        arguments.sample_ids,
        key=lambda conversation: conversation.sample_id,
        label="LoCoMo-Refined sample IDs",
        limit=arguments.limit,
    )
    require_writable_output_pair(arguments.output_path, overwrite=arguments.overwrite)
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    report(f"running {len(conversations)} conversations", quiet=arguments.quiet)
    predictions = asyncio.run(_run_conversations(arguments, conversations))
    _write_artifacts(arguments, conversations, predictions, deployment)
    report(f"wrote {arguments.output_path}", quiet=arguments.quiet)


async def _run_conversations(
    arguments: _Arguments,
    conversations: tuple[LoCoMoRefinedConversation, ...],
) -> tuple[LoCoMoRefinedPrediction, ...]:
    async with connected_memory(arguments) as memory:
        predictions: list[LoCoMoRefinedPrediction] = []
        for index, conversation in enumerate(conversations, start=1):
            report_unit(
                f"conversation {conversation.sample_id}",
                index=index,
                total=len(conversations),
                quiet=arguments.quiet,
            )
            predictions.extend(
                await run_locomo_refined_conversation(
                    memory,
                    conversation,
                    run_id=arguments.run_id,
                    tenant_prefix=arguments.tenant_prefix,
                    recall_limit=arguments.recall_limit,
                    request_concurrency=arguments.request_concurrency,
                )
            )
        return tuple(predictions)


def _write_artifacts(
    arguments: _Arguments,
    conversations: tuple[LoCoMoRefinedConversation, ...],
    predictions: tuple[LoCoMoRefinedPrediction, ...],
    deployment: LoadedDeployment,
) -> None:
    categories = Counter(
        question.category for conversation in conversations for question in conversation.questions
    )
    rows = predictions_jsonl(predictions)
    # A refined question may accept several phrasings, and handing the judge only the first
    # would mark a correct answer wrong. MM-Vet spells the same thing `<OR>` inside its ground
    # truth; the vendored pipeline's one-answer limit is its own and does not apply here.
    asked = tuple(question for conversation in conversations for question in conversation.questions)
    scoring = scoring_snapshot(
        "locomo-refined",
        arguments,
        answers=tuple(
            JudgedAnswer(
                question.question,
                " OR ".join(question.reference_answers),
                prediction.predicted_answer,
            )
            for question, prediction in zip(asked, predictions, strict=True)
        ),
    )
    manifest = core_manifest(
        LoCoMoRefinedRunManifest,
        arguments,
        deployment,
        scoring=scoring,
        runner_version=LOCOMO_REFINED_RUNNER_VERSION,
        adapter_version=LOCOMO_REFINED_ADAPTER_VERSION,
        predictions=rows,
        source_repository="mem-eval-suite/LoCoMo_refined",
        source_sha256=sha256_file(arguments.dataset_path),
        prediction_key=LOCOMO_REFINED_PREDICTION_KEY,
        sample_ids=tuple(conversation.sample_id for conversation in conversations),
        memory_item_count=sum(len(conversation.turns) for conversation in conversations),
        question_count=len(predictions),
        unanswered_question_count=sum(
            not prediction.mindbridge_answered for prediction in predictions
        ),
        category_question_counts=dict(sorted(categories.items())),
    )
    write_run_artifacts(arguments.output_path, rows, manifest)


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = core_parser(tenant_prefix="benchmark_locomo_refined", prog=prog, description=__doc__)
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
        sample_ids=tuple(parsed.sample_id),
    )


if __name__ == "__main__":
    main()
