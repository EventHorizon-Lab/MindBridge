"""Run LoCoMo-Refined through the public MindBridge remember and recall contract."""

from __future__ import annotations

import asyncio

from pydantic import Field

from mindbridge.benchmarks.locomo_refined import (
    LOCOMO_REFINED_ADAPTER_VERSION,
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.benchmarks.runtime import (
    answer_failure_trace_id,
    benchmark_tenant_id,
    settle_answers,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    NonEmptyString,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MemoryType
from mindbridge.sdk import MindBridge

LOCOMO_REFINED_PREDICTION_KEY = "predicted_answer"


class LoCoMoRefinedPrediction(ContractModel):
    """One official prediction row plus reproducibility diagnostics its scorer ignores.

    `qa_id` and `predicted_answer` are exactly what `mem-eval-suite/LoCoMo_refined`'s
    `scripts/run_eval.sh` reads; it merges each row into its own question record by
    `qa_id` and never looks at the other keys, so the diagnostics ride along for free.
    """

    qa_id: Identifier
    # Deliberately not `NonEmptyString`: LoCoMo-Refined dropped the adversarial category,
    # so there is no gold abstention wording left to substitute. A recall that produced no
    # answer is simply a wrong answer, and the empty string is how the official evaluator
    # already represents one -- `mindbridge_answered` below is what makes the two
    # distinguishable in a run's own diagnostics.
    predicted_answer: str
    mindbridge_answered: bool
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_prediction_context: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    mindbridge_error_code: NonEmptyString | None = None
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


async def run_locomo_refined_conversation(
    memory: MindBridge,
    conversation: LoCoMoRefinedConversation,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_locomo_refined",
    recall_limit: int = 20,
    request_concurrency: int = 4,
) -> tuple[LoCoMoRefinedPrediction, ...]:
    """Ingest one conversation, then answer its questions without label leakage."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError("recall_limit must be between 1 and 100; concurrency must be positive")
    tenant_id = benchmark_tenant_id(tenant_prefix, conversation.sample_id, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)
    remembered = await asyncio.gather(
        *(
            _remember_turn(memory, tenant_id, conversation.sample_id, turn, semaphore)
            for turn in conversation.turns
        ),
        return_exceptions=True,
    )
    ingest_failures = _count_ingest_failures(remembered)
    dialog_id_by_memory_id = {
        outcome[1]: outcome[0] for outcome in remembered if not isinstance(outcome, BaseException)
    }
    answers = await asyncio.gather(
        *(
            _answer_question(
                memory,
                tenant_id,
                question,
                dialog_id_by_memory_id,
                recall_limit,
                semaphore,
                ingest_failures,
            )
            for question in conversation.questions
        ),
        return_exceptions=True,
    )
    return settle_answers(
        conversation.questions,
        answers,
        lambda question, code: _failed_prediction(question, code, ingest_failures),
    )


def _failed_prediction(
    question: LoCoMoRefinedQuestion,
    error_code: str,
    ingest_failures: int,
) -> LoCoMoRefinedPrediction:
    """Represent a question whose recall raised as the wrong answer the official scorer reads.

    The empty string is how `run_eval.sh` already represents a wrong answer, so the row still
    scores; `mindbridge_error_code` is what keeps it from being read as an abstention.
    """
    return LoCoMoRefinedPrediction(
        qa_id=question.question_id,
        predicted_answer="",
        mindbridge_answered=False,
        mindbridge_confidence=0.0,
        mindbridge_prediction_context=(),
        mindbridge_trace_id=answer_failure_trace_id(question.question_id),
        mindbridge_error_code=error_code,
        mindbridge_ingest_failure_count=ingest_failures,
    )


def _count_ingest_failures(outcomes: list[tuple[str, str] | BaseException]) -> int:
    """Count the turns that failed so a single bad turn cannot discard its whole conversation.

    A bare gather made the first exception the whole conversation's result, throwing away every
    turn already written. The count rides along on every prediction, so a run still tells missing
    history apart from a wrong answer.
    """
    return sum(isinstance(outcome, BaseException) for outcome in outcomes)


async def _remember_turn(
    memory: MindBridge,
    tenant_id: str,
    sample_id: str,
    turn: LoCoMoRefinedTurn,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        stored = await memory.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=_turn_summary(turn),
                memory_type=MemoryType.EPISODIC,
                occurred_at=turn.occurred_at,
                idempotency_key=f"{LOCOMO_REFINED_ADAPTER_VERSION}:{sample_id}:{turn.dialog_id}",
            )
        )
    return turn.dialog_id, stored.memory_id


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: LoCoMoRefinedQuestion,
    dialog_id_by_memory_id: dict[str, str],
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    ingest_failures: int,
) -> LoCoMoRefinedPrediction:
    async with semaphore:
        result = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(text=question.question),
                limit=recall_limit,
                include_evidence=False,
            )
        )
    retrieved_dialog_ids = tuple(
        dialog_id_by_memory_id[item.memory_id]
        for item in result.memories
        if item.memory_id in dialog_id_by_memory_id
    )
    return LoCoMoRefinedPrediction(
        qa_id=question.question_id,
        predicted_answer=result.answer or "",
        mindbridge_answered=bool(result.answer),
        mindbridge_confidence=result.confidence,
        mindbridge_prediction_context=retrieved_dialog_ids,
        mindbridge_trace_id=result.trace_id,
        mindbridge_ingest_failure_count=ingest_failures,
    )


def _turn_summary(turn: LoCoMoRefinedTurn) -> str:
    statement = f'{turn.speaker} said: "{turn.text}"'
    if turn.image_caption is None:
        return statement
    return f'{statement}\n{turn.speaker} shared an image described as: "{turn.image_caption}"'
