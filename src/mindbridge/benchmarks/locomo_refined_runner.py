"""Run one LoCoMo-Refined conversation through the public local API."""

from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from mindbridge import AsyncMemory, MindBridgeError
from mindbridge.benchmarks.locomo_refined import (
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.models.openai_http import UNKNOWN_ANSWER

LOCOMO_REFINED_PREDICTION_KEY = "predicted_answer"

_Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
_FailureCount = Annotated[int, Field(ge=0)]


class LoCoMoRefinedPrediction(BaseModel):
    """One official prediction row plus scorer-ignored diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    qa_id: _Identifier
    predicted_answer: str
    mindbridge_answered: bool
    mindbridge_confidence: _Confidence
    mindbridge_prediction_context: tuple[_Identifier, ...]
    mindbridge_error_code: _Identifier | None = None
    mindbridge_ingest_failure_count: _FailureCount = 0


async def run_locomo_refined_conversation(
    memory: AsyncMemory,
    conversation: LoCoMoRefinedConversation,
    *,
    recall_limit: int = 20,
    request_concurrency: int = 4,
) -> tuple[LoCoMoRefinedPrediction, ...]:
    """Ingest one isolated conversation, then answer without using its gold labels."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError("recall_limit must be between 1 and 100; concurrency must be positive")

    semaphore = asyncio.Semaphore(request_concurrency)
    remembered = await asyncio.gather(
        *(
            _add_turn(memory, conversation.sample_id, turn, semaphore)
            for turn in conversation.turns
        ),
        return_exceptions=True,
    )
    for remembered_outcome in remembered:
        if isinstance(remembered_outcome, BaseException) and not isinstance(
            remembered_outcome, Exception
        ):
            raise remembered_outcome
    ingest_failures = sum(isinstance(item, Exception) for item in remembered)
    dialog_id_by_memory_id = {
        item[1]: item[0] for item in remembered if not isinstance(item, BaseException)
    }
    answers = await asyncio.gather(
        *(
            _answer_question(
                memory,
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
    predictions: list[LoCoMoRefinedPrediction] = []
    for question, answer_outcome in zip(conversation.questions, answers, strict=True):
        if not isinstance(answer_outcome, BaseException):
            predictions.append(answer_outcome)
        elif isinstance(answer_outcome, Exception):
            predictions.append(
                _failed_prediction(question, _error_code(answer_outcome), ingest_failures)
            )
        else:
            raise answer_outcome
    return tuple(predictions)


async def _add_turn(
    memory: AsyncMemory,
    sample_id: str,
    turn: LoCoMoRefinedTurn,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    async with semaphore:
        stored = await memory.add(
            _turn_content(turn),
            occurred_at=turn.occurred_at,
            metadata={
                "benchmark": "locomo-refined",
                "sample_id": sample_id,
                "dialog_id": turn.dialog_id,
            },
        )
    return turn.dialog_id, stored.id


async def _answer_question(
    memory: AsyncMemory,
    question: LoCoMoRefinedQuestion,
    dialog_id_by_memory_id: dict[str, str],
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    ingest_failures: int,
) -> LoCoMoRefinedPrediction:
    async with semaphore:
        result = await memory.ask(question.question, limit=recall_limit)
    return LoCoMoRefinedPrediction(
        qa_id=question.question_id,
        predicted_answer=result.answer,
        # `AnswerResult` rejects a blank answer, so truthiness is always true. A refusal is
        # the model emitting the grounded-abstention sentence, which is what this counts.
        mindbridge_answered=result.answer.strip() != UNKNOWN_ANSWER,
        mindbridge_confidence=max((hit.score for hit in result.hits), default=0.0),
        mindbridge_prediction_context=tuple(
            dialog_id_by_memory_id[hit.id]
            for hit in result.hits
            if hit.id in dialog_id_by_memory_id
        ),
        mindbridge_ingest_failure_count=ingest_failures,
    )


def _failed_prediction(
    question: LoCoMoRefinedQuestion,
    error_code: str,
    ingest_failures: int,
) -> LoCoMoRefinedPrediction:
    return LoCoMoRefinedPrediction(
        qa_id=question.question_id,
        predicted_answer="",
        mindbridge_answered=False,
        mindbridge_confidence=0.0,
        mindbridge_prediction_context=(),
        mindbridge_error_code=error_code,
        mindbridge_ingest_failure_count=ingest_failures,
    )


def _error_code(error: Exception) -> str:
    return error.code if isinstance(error, MindBridgeError) else type(error).__name__


def _turn_content(turn: LoCoMoRefinedTurn) -> str:
    statement = f'{turn.speaker} said: "{turn.text}"'
    if turn.image_caption is None:
        return statement
    return f'{statement}\n{turn.speaker} shared an image described as: "{turn.image_caption}"'
