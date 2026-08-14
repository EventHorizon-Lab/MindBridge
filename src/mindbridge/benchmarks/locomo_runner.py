"""Run LoCoMo through the public MindBridge remember and recall contract."""

from __future__ import annotations

import asyncio

from pydantic import Field

from mindbridge.benchmarks.locomo import (
    LOCOMO_ADAPTER_VERSION,
    LoCoMoConversation,
    LoCoMoQuestion,
    LoCoMoTurn,
)
from mindbridge.benchmarks.runtime import benchmark_tenant_id
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

LOCOMO_PREDICTION_KEY = "mindbridge_prediction"
LOCOMO_ABSTENTION = "Not mentioned in the conversation"


class LoCoMoOfficialQuestionResult(ContractModel):
    """One official QA row plus reproducibility diagnostics ignored by its scorer."""

    question: NonEmptyString
    answer: NonEmptyString | tuple[NonEmptyString, ...]
    evidence: tuple[Identifier, ...]
    category: int = Field(ge=1, le=5)
    mindbridge_prediction: NonEmptyString
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_prediction_context: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier


class LoCoMoOfficialConversationResult(ContractModel):
    """Prediction shape accepted by the official conversation-level evaluator."""

    sample_id: Identifier
    qa: tuple[LoCoMoOfficialQuestionResult, ...] = Field(min_length=1)


async def run_locomo_conversation(
    memory: MindBridge,
    conversation: LoCoMoConversation,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_locomo",
    recall_limit: int = 20,
    request_concurrency: int = 4,
) -> LoCoMoOfficialConversationResult:
    """Ingest one conversation, then answer its questions without label leakage."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError("recall_limit must be between 1 and 100; concurrency must be positive")
    tenant_id = benchmark_tenant_id(tenant_prefix, conversation.sample_id, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)
    remembered = await asyncio.gather(
        *(
            _remember_turn(memory, tenant_id, conversation.sample_id, turn, semaphore)
            for turn in conversation.turns
        )
    )
    dialog_id_by_memory_id = {memory_id: dialog_id for dialog_id, memory_id in remembered}
    qa = await asyncio.gather(
        *(
            _answer_question(
                memory,
                tenant_id,
                question,
                dialog_id_by_memory_id,
                recall_limit,
                semaphore,
            )
            for question in conversation.questions
        )
    )
    return LoCoMoOfficialConversationResult(sample_id=conversation.sample_id, qa=tuple(qa))


async def _remember_turn(
    memory: MindBridge,
    tenant_id: str,
    sample_id: str,
    turn: LoCoMoTurn,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        stored = await memory.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=_turn_summary(turn),
                memory_type=MemoryType.EPISODIC,
                occurred_at=turn.occurred_at,
                idempotency_key=f"{LOCOMO_ADAPTER_VERSION}:{sample_id}:{turn.dialog_id}",
            )
        )
    return turn.dialog_id, stored.memory_id


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: LoCoMoQuestion,
    dialog_id_by_memory_id: dict[str, str],
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> LoCoMoOfficialQuestionResult:
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
    return LoCoMoOfficialQuestionResult(
        question=question.question,
        answer=(
            question.reference_answers[0]
            if len(question.reference_answers) == 1
            else question.reference_answers
        ),
        evidence=question.evidence_dialog_ids,
        category=question.category,
        mindbridge_prediction=result.answer or LOCOMO_ABSTENTION,
        mindbridge_confidence=result.confidence,
        mindbridge_prediction_context=retrieved_dialog_ids,
        mindbridge_trace_id=result.trace_id,
    )


def _turn_summary(turn: LoCoMoTurn) -> str:
    statement = f'{turn.speaker} said: "{turn.text}"'
    if turn.image_caption is None:
        return statement
    return f'{statement}\n{turn.speaker} shared an image described as: "{turn.image_caption}"'
