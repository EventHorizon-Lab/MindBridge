"""Production-contract checks for the LoCoMo-Refined runner."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.locomo_refined import (
    LoCoMoRefinedConversation,
    LoCoMoRefinedQuestion,
    LoCoMoRefinedTurn,
)
from mindbridge.benchmarks.locomo_refined_runner import run_locomo_refined_conversation
from mindbridge.contracts import (
    MemoryView,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import MemoryState, VerificationStatus
from mindbridge.sdk import MindBridgeError

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
            trace_id="trace_locomo_refined",
        )


async def test_locomo_refined_uses_only_source_turns_and_questions_in_api_requests() -> None:
    api = RecordingMemoryApi()
    conversation = LoCoMoRefinedConversation(
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
                reference_answers=("SECRET REFERENCE ANSWER",),
                evidence_dialog_ids=("D1:1",),
                category=2,
                is_multi_modality=False,
            ),
        ),
    )

    predictions = await run_locomo_refined_conversation(
        cast(MindBridge, api), conversation, run_id="run_01"
    )

    assert api.remember_requests[0].summary == (
        'Caroline said: "I started a new course."\n'
        'Caroline shared an image described as: "a classroom at sunrise"'
    )
    assert api.remember_requests[0].idempotency_key == "locomo_refined_v1:conv-26:D1:1"
    assert api.recall_requests[0].query.text == "What did Caroline start?"
    assert "SECRET REFERENCE ANSWER" not in api.remember_requests[0].model_dump_json()
    assert "SECRET REFERENCE ANSWER" not in api.recall_requests[0].model_dump_json()
    # The official evaluator joins on `qa_id` alone, so the row must carry the release's own
    # id and nothing that would leak the gold answer back into the scored artifact.
    assert predictions[0].qa_id == "conv-26#q0000"
    assert predictions[0].predicted_answer == ""
    assert predictions[0].mindbridge_answered is False
    assert predictions[0].mindbridge_prediction_context == ("D1:1",)
    assert predictions[0].mindbridge_trace_id == "trace_locomo_refined"


async def test_locomo_refined_writes_the_recalled_answer_as_the_prediction() -> None:
    api = RecordingMemoryApi()
    api.answer = "A new course."

    predictions = await run_locomo_refined_conversation(
        cast(MindBridge, api), _conversation(), run_id="run_01"
    )

    assert predictions[0].predicted_answer == "A new course."
    assert predictions[0].mindbridge_answered is True


async def test_locomo_refined_rejects_unbounded_or_empty_request_pool() -> None:
    with pytest.raises(ValueError, match="positive"):
        await run_locomo_refined_conversation(
            cast(MindBridge, RecordingMemoryApi()),
            _conversation(),
            run_id="run_01",
            request_concurrency=0,
        )


async def test_locomo_refined_uses_the_same_recall_budget_for_every_question_wording() -> None:
    api = RecordingMemoryApi()
    conversation = _conversation().model_copy(
        update={
            "questions": (
                LoCoMoRefinedQuestion(
                    question_id="conv-26#q0000",
                    question="Would Caroline likely enjoy another course?",
                    reference_answers=("Yes",),
                    evidence_dialog_ids=("D1:1",),
                    category=3,
                    is_multi_modality=False,
                ),
            )
        }
    )

    await run_locomo_refined_conversation(
        cast(MindBridge, api),
        conversation,
        run_id="run_01",
        recall_limit=20,
    )

    assert api.recall_requests[0].limit == 20


async def test_locomo_refined_keeps_stored_turns_when_one_turn_fails_to_remember() -> None:
    """A turn that cannot be written used to discard every turn written beside it."""

    class FailingMemoryApi(RecordingMemoryApi):
        async def remember(self, request: RememberRequest) -> MemoryView:
            if request.idempotency_key == "locomo_refined_v1:conv-26:D1:2":
                raise MindBridgeError(
                    "turn could not be written",
                    code="model_request_failed",
                    status_code=502,
                    trace_id="trace_ingest_error",
                )
            return await super().remember(request)

    api = FailingMemoryApi()
    api.answer = "A new course."
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

    predictions = await run_locomo_refined_conversation(
        cast(MindBridge, api), conversation, run_id="run_01"
    )

    assert [request.summary for request in api.remember_requests] == [
        'Caroline said: "turn D1:1"',
        'Caroline said: "turn D1:3"',
    ]
    assert predictions[0].predicted_answer == "A new course."
    # The surviving turns must still resolve to their own dialog ids, not shift onto the dead one.
    assert predictions[0].mindbridge_prediction_context == ("D1:1",)
    assert predictions[0].mindbridge_ingest_failure_count == 1


def _conversation() -> LoCoMoRefinedConversation:
    return LoCoMoRefinedConversation(
        sample_id="conv-26",
        turns=(
            LoCoMoRefinedTurn(
                dialog_id="D1:1",
                speaker="Caroline",
                text="Hello",
                occurred_at=NOW,
            ),
        ),
        questions=(
            LoCoMoRefinedQuestion(
                question_id="conv-26#q0000",
                question="What happened?",
                reference_answers=("Hello",),
                evidence_dialog_ids=("D1:1",),
                category=1,
                is_multi_modality=False,
            ),
        ),
    )
