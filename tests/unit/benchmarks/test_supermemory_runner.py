"""Production-contract checks for the causal SuperMemory-VQA runner."""

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import AsyncMindBridge
from mindbridge.benchmarks import (
    SuperMemoryPreparedSegment,
    SuperMemoryPreparedSubject,
    SuperMemoryPreparedVideo,
    SuperMemoryQuestion,
    SuperMemoryQuestionResult,
    evaluate_supermemory_vqa,
    run_supermemory_vqa,
)
from mindbridge.contracts import (
    MediaObjectInput,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind, MemoryState, MemoryType, VerificationStatus

ORIGIN = datetime(2026, 3, 10, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, answer: str | None = "Ranking: C, A, D, B") -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.recall_requests: list[RecallRequest] = []
        self.remember_requests: list[RememberRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        return ObservationReceipt(
            observation_id=f"observation_{request.sequence}",
            processing_job_id=f"job_{request.sequence}",
            evidence_ids=(f"evidence_{request.sequence}",),
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
            observation_id=job_id.replace("job", "observation"),
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=ORIGIN,
            updated_at=ORIGIN,
            trace_id=f"trace_{tenant_id}",
        )

    async def remember(self, request: RememberRequest) -> MemoryView:
        self.calls.append(f"remember:{request.summary.splitlines()[0]}")
        self.remember_requests.append(request)
        return MemoryView(
            memory_id="memory_transcript",
            memory_type=MemoryType.EPISODIC,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=ORIGIN,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.calls.append("recall")
        self.recall_requests.append(request)
        return RecallResult(
            answer=self.answer,
            confidence=0.8,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )


async def test_supermemory_ingests_through_question_boundary_without_future_segment() -> None:
    api = RecordingMemoryApi()

    result = await run_supermemory_vqa(
        cast(AsyncMindBridge, api),
        (_question(1, question_ended_at=ORIGIN + timedelta(seconds=45)),),
        _prepared_subject(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert set(api.calls[:-1]) == {
        "observe:0",
        "job:job_0",
        "remember:B said the mug was in the sink.",
        "observe:1",
        "job:job_1",
    }
    assert api.calls[-1] == "recall"
    assert api.calls.index("observe:0") < api.calls.index("job:job_0")
    assert api.calls.index("job:job_0") < api.calls.index(
        "remember:B said the mug was in the sink."
    )
    assert api.calls.index("observe:1") < api.calls.index("job:job_1")
    assert api.remember_requests[0].evidence_ids == ("evidence_0",)
    assert result[0].ranked_option_indices == (2, 0, 3, 1)
    assert api.recall_requests[0].filters.occurred_before == ORIGIN + timedelta(seconds=45)
    request_json = api.recall_requests[0].model_dump_json()
    assert "correct_option_index" not in request_json
    assert "is_answerable" not in request_json


async def test_supermemory_maps_production_abstention_to_explicit_choice() -> None:
    api = RecordingMemoryApi(answer=None)

    result = await run_supermemory_vqa(
        cast(AsyncMindBridge, api),
        (_question(1, question_ended_at=ORIGIN + timedelta(seconds=45)),),
        _prepared_subject(),
        run_id="run_02",
        poll_interval_seconds=0.001,
    )

    assert result[0].predicted_option_index == 0
    assert result[0].ranked_option_indices == (0,)


async def test_supermemory_rejects_missing_question_boundary_before_api_calls() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="question boundary"):
        await run_supermemory_vqa(
            cast(AsyncMindBridge, api),
            (_question(1, question_ended_at=ORIGIN + timedelta(seconds=45)),),
            SuperMemoryPreparedSubject(
                subject=1,
                videos=(
                    _video(
                        "Person_1_session_1",
                        ORIGIN,
                        (_segment(0, "media_0"), _segment(30, "media_1")),
                    ),
                ),
            ),
            run_id="run_03",
        )

    assert api.calls == []


async def test_supermemory_accepts_question_at_video_start_without_ingesting_future() -> None:
    api = RecordingMemoryApi()
    prepared = _prepared_subject()

    await run_supermemory_vqa(
        cast(AsyncMindBridge, api),
        (_question(1, question_ended_at=ORIGIN),),
        prepared,
        run_id="run_01",
    )

    assert api.calls == ["recall"]


def test_supermemory_metrics_use_answerability_and_full_ranking() -> None:
    questions = (
        _question(1, question_ended_at=ORIGIN + timedelta(seconds=45)),
        _question(
            2,
            question_ended_at=ORIGIN + timedelta(seconds=46),
            correct_option_index=0,
            is_answerable=False,
        ),
    )
    results = (
        _result(1, (1, 2, 0, 3)),
        _result(2, (0,)),
    )

    metrics = evaluate_supermemory_vqa(questions, results)

    assert metrics.answerability_f1 == 1.0
    assert metrics.qa_accuracy == 0.5
    assert metrics.qa_mean_reciprocal_rank == 0.75


def test_prepared_supermemory_rejects_globally_overlapping_segments() -> None:
    with pytest.raises(ValidationError, match="must not overlap globally"):
        SuperMemoryPreparedSubject(
            subject=1,
            videos=(
                _video("video_1", ORIGIN, (_segment(0, "media_1"),)),
                _video("video_2", ORIGIN + timedelta(seconds=10), (_segment(0, "media_2"),)),
            ),
        )


def _question(
    question_id: int,
    *,
    question_ended_at: datetime,
    correct_option_index: int = 2,
    is_answerable: bool = True,
) -> SuperMemoryQuestion:
    return SuperMemoryQuestion(
        question_id=question_id,
        subject=1,
        question="Where did I leave the mug?",
        choices=(
            "This question can not be answered.",
            "On the counter",
            "In the sink",
            "On the table",
        ),
        correct_option_index=correct_option_index,
        unanswerable_option_index=0,
        is_answerable=is_answerable,
        skill="object_location_memory",
        source_video_ids=("Person_1_session_1",),
        question_video_id="Person_1_session_1",
        question_ended_at=question_ended_at,
    )


def _prepared_subject() -> SuperMemoryPreparedSubject:
    return SuperMemoryPreparedSubject(
        subject=1,
        videos=(
            _video(
                "Person_1_session_1",
                ORIGIN,
                (
                    _segment(0, "media_0", transcript="B said the mug was in the sink."),
                    _segment(30, "media_1", duration_ms=15_000),
                    _segment(45, "media_2"),
                ),
            ),
        ),
    )


def _video(
    video_id: str,
    started_at: datetime,
    segments: tuple[SuperMemoryPreparedSegment, ...],
) -> SuperMemoryPreparedVideo:
    return SuperMemoryPreparedVideo(
        video_id=video_id,
        started_at=started_at,
        segments=segments,
    )


def _segment(
    start_seconds: float,
    media_id: str,
    *,
    duration_ms: int = 30_000,
    transcript: str | None = None,
) -> SuperMemoryPreparedSegment:
    return SuperMemoryPreparedSegment(
        start_seconds=start_seconds,
        duration_ms=duration_ms,
        media_objects=(
            MediaObjectInput(
                media_object_id=media_id,
                kind=MediaKind.VIDEO,
                uri=f"s3://benchmark/{media_id}.mp4",
                sha256="a" * 64,
                size_bytes=1_024,
                created_at=ORIGIN,
                duration_ms=duration_ms,
            ),
        ),
        transcript=transcript,
    )


def _result(question_id: int, ranking: tuple[int, ...]) -> SuperMemoryQuestionResult:
    return SuperMemoryQuestionResult(
        question_id=question_id,
        predicted_option_index=ranking[0] if ranking else None,
        ranked_option_indices=ranking,
        model_answer="prediction",
        mindbridge_confidence=0.8,
        mindbridge_memory_ids=(),
        mindbridge_evidence_ids=(),
        mindbridge_trace_id=f"trace_{question_id}",
    )
