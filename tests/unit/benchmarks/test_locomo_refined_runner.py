"""Public-API checks for the isolated LoCoMo-Refined runner."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from inspect import signature
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import AnswerResult, AsyncMemory, MemoryRecord, ModelError, SearchHit
from mindbridge.benchmarks.locomo_refined import (
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.benchmarks.locomo_refined_runner import run_locomo_refined_conversation

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class _RecordingMemory:
    def __init__(self) -> None:
        self.add_requests: list[tuple[str, datetime | None, Mapping[str, object] | None]] = []
        self.ask_requests: list[tuple[str, int]] = []
        self.records: list[MemoryRecord] = []
        self.answer = "A new course."
        self.scores: tuple[float, ...] = ()

    async def add(
        self,
        content: str,
        *,
        occurred_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        self.add_requests.append((content, occurred_at, metadata))
        dialog_id = None if metadata is None else metadata.get("dialog_id")
        assert isinstance(dialog_id, str)
        record = MemoryRecord(
            id=f"memory:{dialog_id}",
            content=content,
            created_at=NOW,
            occurred_at=occurred_at,
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    async def ask(self, question: str, *, limit: int = 5) -> AnswerResult:
        self.ask_requests.append((question, limit))
        hits = tuple(
            SearchHit(
                id=record.id,
                content=record.content,
                score=score,
                created_at=record.created_at,
                occurred_at=record.occurred_at,
                metadata=record.metadata,
            )
            for record, score in zip(self.records, self.scores, strict=False)
        )
        return AnswerResult(answer=self.answer, hits=hits)


async def test_runner_uses_only_public_add_and_ask_without_gold_or_identity_fields() -> None:
    memory = _RecordingMemory()
    memory.scores = (0.73,)
    conversation = _conversation(reference_answer="SECRET REFERENCE ANSWER")

    predictions = await run_locomo_refined_conversation(
        cast(AsyncMemory, memory), conversation, recall_limit=17
    )

    content, occurred_at, metadata = memory.add_requests[0]
    assert content == (
        'Caroline said: "I started a new course."\n'
        'Caroline shared an image described as: "a classroom at sunrise"'
    )
    assert occurred_at == NOW
    assert metadata == {
        "benchmark": "locomo-refined",
        "sample_id": "conv-26",
        "dialog_id": "D1:1",
    }
    assert metadata is not None
    assert {"tenant", "tenant_id", "user", "user_id", "run", "run_id"}.isdisjoint(metadata)
    assert memory.ask_requests == [("What did Caroline start?", 17)]
    assert "SECRET REFERENCE ANSWER" not in repr(memory.add_requests)
    assert "SECRET REFERENCE ANSWER" not in repr(memory.ask_requests)
    assert {"run", "run_id", "tenant_prefix", "tenant_id", "user", "user_id"}.isdisjoint(
        signature(run_locomo_refined_conversation).parameters
    )
    assert predictions[0].qa_id == "conv-26#q0000"
    assert predictions[0].predicted_answer == "A new course."
    assert predictions[0].mindbridge_prediction_context == ("D1:1",)
    assert predictions[0].mindbridge_confidence == pytest.approx(0.73)
    serialized = predictions[0].model_dump()
    assert "mindbridge_trace_id" not in serialized
    assert not any("tenant" in key for key in serialized)
    assert "SECRET REFERENCE ANSWER" not in predictions[0].model_dump_json()


async def test_confidence_is_zero_when_ask_returns_no_hits() -> None:
    memory = _RecordingMemory()

    prediction = (
        await run_locomo_refined_conversation(cast(AsyncMemory, memory), _conversation())
    )[0]

    assert prediction.mindbridge_confidence == 0.0
    assert prediction.mindbridge_prediction_context == ()


async def test_one_failed_turn_does_not_discard_the_other_turns() -> None:
    class _FailingMemory(_RecordingMemory):
        async def add(
            self,
            content: str,
            *,
            occurred_at: datetime | None = None,
            metadata: Mapping[str, object] | None = None,
        ) -> MemoryRecord:
            if metadata is not None and metadata.get("dialog_id") == "D1:2":
                raise ModelError("turn could not be embedded")
            return await super().add(content, occurred_at=occurred_at, metadata=metadata)

    memory = _FailingMemory()
    memory.scores = (0.4, 0.8)
    conversation = _conversation().model_copy(
        update={
            "turns": tuple(
                LoCoMoRefinedTurn(
                    dialog_id=dialog_id,
                    speaker="Caroline",
                    text=f"turn {dialog_id}",
                    occurred_at=NOW,
                )
                for dialog_id in ("D1:1", "D1:2", "D1:3")
            )
        }
    )

    prediction = (await run_locomo_refined_conversation(cast(AsyncMemory, memory), conversation))[0]

    assert [record.metadata["dialog_id"] for record in memory.records] == ["D1:1", "D1:3"]
    assert prediction.mindbridge_prediction_context == ("D1:1", "D1:3")
    assert prediction.mindbridge_confidence == pytest.approx(0.8)
    assert prediction.mindbridge_ingest_failure_count == 1


async def test_one_failed_question_keeps_other_predictions_and_stable_error_codes() -> None:
    class _FailingMemory(_RecordingMemory):
        async def ask(self, question: str, *, limit: int = 5) -> AnswerResult:
            if "second" in question:
                raise ModelError("model unavailable")
            if "third" in question:
                raise RuntimeError("unexpected failure")
            return await super().ask(question, limit=limit)

    memory = _FailingMemory()
    conversation = _conversation().model_copy(
        update={
            "questions": tuple(
                LoCoMoRefinedQuestion(
                    question_id=f"conv-26#q{index:04d}",
                    question=f"What happened {word}?",
                    reference_answers=("GOLD",),
                    evidence_dialog_ids=("D1:1",),
                    category=1,
                    is_multi_modality=False,
                )
                for index, word in enumerate(("first", "second", "third"))
            )
        }
    )

    predictions = await run_locomo_refined_conversation(cast(AsyncMemory, memory), conversation)

    assert [prediction.qa_id for prediction in predictions] == [
        "conv-26#q0000",
        "conv-26#q0001",
        "conv-26#q0002",
    ]
    assert [prediction.mindbridge_error_code for prediction in predictions] == [
        None,
        "model_error",
        "RuntimeError",
    ]
    assert [prediction.predicted_answer for prediction in predictions] == [
        "A new course.",
        "",
        "",
    ]


async def test_runner_rejects_invalid_concurrency_or_recall_limits() -> None:
    memory = cast(AsyncMemory, _RecordingMemory())
    with pytest.raises(ValueError, match="positive"):
        await run_locomo_refined_conversation(memory, _conversation(), request_concurrency=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_locomo_refined_conversation(memory, _conversation(), recall_limit=101)


def test_dataset_values_are_strict_and_frozen() -> None:
    conversation = _conversation()

    with pytest.raises(ValidationError):
        LoCoMoRefinedConversation(
            sample_id=cast(str, 26),
            turns=conversation.turns,
            questions=conversation.questions,
        )
    with pytest.raises(ValidationError):
        conversation.sample_id = "another"


def _conversation(*, reference_answer: str = "Hello") -> LoCoMoRefinedConversation:
    return LoCoMoRefinedConversation(
        sample_id="conv-26",
        turns=(
            LoCoMoRefinedTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="I started a new course.",
                occurred_at=NOW,
                image_caption="a classroom at sunrise",
            ),
        ),
        questions=(
            LoCoMoRefinedQuestion(
                question_id="conv-26#q0000",
                question="What did Caroline start?",
                reference_answers=(reference_answer,),
                evidence_dialog_ids=("D1:1",),
                category=2,
                is_multi_modality=False,
            ),
        ),
    )
