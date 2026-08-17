"""Production-contract checks for the LoCoMo runner."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks import (
    LOCOMO_ABSTENTION,
    LoCoMoConversation,
    LoCoMoQuestion,
    LoCoMoTurn,
    run_locomo_conversation,
)
from mindbridge.contracts import (
    MemoryView,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import MemoryState, VerificationStatus

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self.memories: list[MemoryView] = []
        self.answer: str | None = None

    async def remember(self, request: RememberRequest) -> MemoryView:
        self.remember_requests.append(request)
        memory = MemoryView(
            memory_id=f"memory_{len(self.remember_requests)}",
            memory_type=request.memory_type,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
        )
        self.memories.append(memory)
        return memory

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer=self.answer,
            confidence=0.0 if self.answer is None else 0.9,
            memories=tuple(self.memories[:1]),
            evidence=(),
            trace_id="trace_locomo",
        )


async def test_locomo_uses_only_source_turns_and_questions_in_api_requests() -> None:
    api = RecordingMemoryApi()
    conversation = LoCoMoConversation(
        sample_id="conv-01",
        turns=(
            LoCoMoTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="I started a new course.",
                occurred_at=NOW,
                image_caption="a classroom at sunrise",
            ),
        ),
        questions=(
            LoCoMoQuestion(
                question_id="conv-01_Q0001",
                question="What did Caroline start?",
                reference_answers=("SECRET REFERENCE ANSWER",),
                evidence_dialog_ids=("D1:1",),
                category=2,
            ),
        ),
    )

    result = await run_locomo_conversation(cast(MindBridge, api), conversation, run_id="run_01")

    assert api.remember_requests[0].summary == (
        'Caroline said: "I started a new course."\n'
        'Caroline shared an image described as: "a classroom at sunrise"'
    )
    assert api.remember_requests[0].idempotency_key == "locomo_official_v1:conv-01:D1:1"
    assert api.recall_requests[0].query.text == "What did Caroline start?"
    assert "SECRET REFERENCE ANSWER" not in api.remember_requests[0].model_dump_json()
    assert "SECRET REFERENCE ANSWER" not in api.recall_requests[0].model_dump_json()
    assert result.qa[0].answer == "SECRET REFERENCE ANSWER"
    assert result.qa[0].mindbridge_prediction == LOCOMO_ABSTENTION
    assert result.qa[0].mindbridge_abstained is True
    assert result.qa[0].mindbridge_prediction_context == ("D1:1",)
    assert result.qa[0].mindbridge_trace_id == "trace_locomo"


async def test_locomo_marks_an_answered_question_as_not_abstained() -> None:
    """The adversarial reference string is also a valid answer, so scoring needs the flag."""
    api = RecordingMemoryApi()
    api.answer = "A new course."

    result = await run_locomo_conversation(cast(MindBridge, api), _conversation(), run_id="run_01")

    assert result.qa[0].mindbridge_prediction == "A new course."
    assert result.qa[0].mindbridge_abstained is False


async def test_locomo_rejects_unbounded_or_empty_request_pool() -> None:
    with pytest.raises(ValueError, match="positive"):
        await run_locomo_conversation(
            cast(MindBridge, RecordingMemoryApi()),
            _conversation(),
            run_id="run_01",
            request_concurrency=0,
        )


async def test_locomo_uses_the_same_recall_budget_for_every_question_wording() -> None:
    api = RecordingMemoryApi()
    conversation = _conversation().model_copy(
        update={
            "questions": (
                LoCoMoQuestion(
                    question_id="conv-01_Q0001",
                    question="Would Caroline likely enjoy another course?",
                    reference_answers=("Yes",),
                    evidence_dialog_ids=("D1:1",),
                    category=1,
                ),
            )
        }
    )

    await run_locomo_conversation(
        cast(MindBridge, api),
        conversation,
        run_id="run_01",
        recall_limit=20,
    )

    assert api.recall_requests[0].limit == 20


def _conversation() -> LoCoMoConversation:
    return LoCoMoConversation(
        sample_id="conv-01",
        turns=(
            LoCoMoTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="Hello",
                occurred_at=NOW,
            ),
        ),
        questions=(
            LoCoMoQuestion(
                question_id="conv-01_Q0001",
                question="What happened?",
                reference_answers=("Hello",),
                evidence_dialog_ids=("D1:1",),
                category=1,
            ),
        ),
    )
