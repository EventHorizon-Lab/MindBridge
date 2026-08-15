"""Production-contract checks for the question-isolated MEMLENS runner."""

import asyncio
from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks import (
    MemLensImage,
    MemLensPreparedImage,
    MemLensPreparedImages,
    MemLensQuestion,
    MemLensSession,
    MemLensTurn,
    run_memlens_question,
)
from mindbridge.benchmarks.memlens_runner import validate_memlens_images
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind

NOW = datetime(2024, 5, 31, 7, 58, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_01",
            evidence_ids=("evidence_01",),
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        self.calls.append(f"job:{job_id}")
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_01",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=NOW,
            updated_at=NOW,
            trace_id=f"trace_{tenant_id}",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.calls.append(f"remember:{request.summary}")
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.calls.append("recall")
        self.recall_requests.append(request)
        return RecallResult(
            answer="$260.00",
            confidence=0.9,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )


async def test_memlens_binds_turn_images_and_keeps_question_tenant_isolated() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_memlens_question(
            cast(MindBridge, api),
            _question(),
            run_id="run_01",
            prepared_images=_prepared_images(),
            recall_limit=101,
        )
    assert not api.calls

    result = await run_memlens_question(
        cast(MindBridge, api),
        _question(),
        run_id="run_01",
        prepared_images=_prepared_images(),
        poll_interval_seconds=0.001,
    )

    assert api.calls == [
        "observe:0",
        "job:job_01",
        'remember:User said: "I paid $150. [image]"',
        'remember:Assistant said: "Noted."',
        "recall",
    ]
    assert api.observe_requests[0].tenant_id == "benchmark_memlens_q_01_run_01"
    assert api.remember_requests[0].evidence_ids == ("evidence_01",)
    assert api.recall_requests[0].query.text is not None
    assert "Question date: 2024/05/31 07:58 UTC" in api.recall_requests[0].query.text
    assert result.prediction == "$260.00"
    assert result.reference_answer == "$260.00"


async def test_memlens_requires_images_unless_text_only() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="missing prepared"):
        await run_memlens_question(
            cast(MindBridge, api),
            _question(),
            run_id="run_01",
        )

    result = await run_memlens_question(
        cast(MindBridge, api),
        _question(),
        run_id="text_01",
        text_only=True,
    )

    assert not api.observe_requests
    assert api.remember_requests[0].summary == 'User said: "I paid $150. [image]"'
    assert result.prediction == "$260.00"


async def test_memlens_ingests_turns_in_source_order() -> None:
    class DelayedMemoryApi(RecordingMemoryApi):
        async def remember(self, request: RememberRequest) -> object:
            if request.summary == 'User said: "first"':
                await asyncio.sleep(0.01)
            return await super().remember(request)

    api = DelayedMemoryApi()
    question = _question()
    ordered = question.model_copy(
        update={
            "question_id": "q_order",
            "sessions": (
                MemLensSession(
                    session_id="sess_order",
                    occurred_at=NOW,
                    turns=(
                        MemLensTurn(turn_id="turn_0", role="user", content="first"),
                        MemLensTurn(turn_id="turn_1", role="assistant", content="second"),
                    ),
                ),
            ),
        }
    )

    await run_memlens_question(
        cast(MindBridge, api),
        ordered,
        run_id="run_01",
        text_only=True,
        request_concurrency=2,
    )

    assert [request.summary for request in api.remember_requests] == [
        'User said: "first"',
        'Assistant said: "second"',
    ]


def test_memlens_validates_all_prepared_images_before_run() -> None:
    question = _question()
    missing_turn = (
        question.sessions[0]
        .turns[0]
        .model_copy(update={"images": (MemLensImage(source_file="needle_images/missing.jpg"),)})
    )
    late_question = question.model_copy(
        update={
            "question_id": "q_late",
            "sessions": (question.sessions[0].model_copy(update={"turns": (missing_turn,)}),),
        }
    )

    with pytest.raises(ValueError, match=r"needle_images/missing\.jpg"):
        validate_memlens_images(
            (question, late_question),
            _prepared_images(),
            text_only=False,
        )


async def test_memlens_splits_long_turns_without_losing_content() -> None:
    api = RecordingMemoryApi()
    question = MemLensQuestion(
        question_id="q_long",
        question_type="information_extraction",
        question="What was said?",
        reference_answer="x",
        question_date=NOW,
        sessions=(
            MemLensSession(
                session_id="sess_long",
                occurred_at=NOW,
                turns=(
                    MemLensTurn(
                        turn_id="sess_long_T0000",
                        role="user",
                        content="x" * 4_000,
                    ),
                ),
            ),
        ),
    )

    await run_memlens_question(
        cast(MindBridge, api),
        question,
        run_id="run_01",
        text_only=True,
    )

    assert len(api.remember_requests) == 3
    assert all(len(request.summary) <= 2_048 for request in api.remember_requests)
    assert sum(request.summary.count("x") for request in api.remember_requests) == 4_000


def _question() -> MemLensQuestion:
    return MemLensQuestion(
        question_id="q_01",
        question_type="multi_session_reasoning",
        question="How much total did I spend?",
        reference_answer="$260.00",
        question_date=NOW,
        sessions=(
            MemLensSession(
                session_id="sess_01",
                occurred_at=datetime(2024, 5, 6, 17, 17, tzinfo=timezone.utc),
                turns=(
                    MemLensTurn(
                        turn_id="sess_01_T0000",
                        role="user",
                        content="I paid $150. <image>",
                        images=(MemLensImage(source_file="needle_images/receipt.jpg"),),
                    ),
                ),
            ),
            MemLensSession(
                session_id="sess_02",
                occurred_at=datetime(2024, 5, 7, 21, 42, tzinfo=timezone.utc),
                turns=(
                    MemLensTurn(
                        turn_id="sess_02_T0000",
                        role="assistant",
                        content="Noted.",
                    ),
                ),
            ),
        ),
    )


def _prepared_images() -> MemLensPreparedImages:
    return MemLensPreparedImages(
        images=(
            MemLensPreparedImage(
                source_file="needle_images/receipt.jpg",
                media_object=MediaObjectInput(
                    media_object_id="memlens_receipt",
                    kind=MediaKind.IMAGE,
                    uri="s3://benchmark/memlens/receipt.jpg",
                    sha256="a" * 64,
                    size_bytes=1_024,
                    created_at=NOW,
                ),
            ),
        )
    )
