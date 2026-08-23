"""Production-contract checks for the causal SuperMemory-VQA runner."""

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import MindBridge
from mindbridge.benchmarks.supermemory_runner import (
    SuperMemoryPreparedSegment,
    SuperMemoryPreparedSubject,
    SuperMemoryPreparedVideo,
    SuperMemoryQuestionResult,
    evaluate_supermemory_vqa,
    run_supermemory_vqa,
)
from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion
from mindbridge.contracts import (
    IdentityObservationInput,
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
from mindbridge.core import (
    IdentityKind,
    JobState,
    MediaKind,
    MemoryState,
    MemoryType,
    VerificationStatus,
)
from mindbridge.sdk import MindBridgeError

ORIGIN = datetime(2026, 3, 10, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, answer: str | None = "Ranking: C, A, D, B") -> None:
        self.answer = answer
        self.calls: list[str] = []
        self.observe_requests: list[ObserveRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self.remember_requests: list[RememberRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        self.observe_requests.append(request)
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
        cast(MindBridge, api),
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
    assert api.observe_requests[0].identity_observations == (_identity(),)
    assert result[0].ranked_option_indices == (2, 0, 3, 1)
    assert api.recall_requests[0].filters.occurred_before == ORIGIN + timedelta(seconds=45)
    request_json = api.recall_requests[0].model_dump_json()
    assert "correct_option_index" not in request_json
    assert "is_answerable" not in request_json


async def test_supermemory_abstention_is_not_the_dataset_unanswerable_choice() -> None:
    """Substituting the choice scored an abstention correct on unanswerable questions."""
    api = RecordingMemoryApi(answer=None)
    unanswerable = _question(
        1,
        question_ended_at=ORIGIN + timedelta(seconds=45),
        correct_option_index=0,
        is_answerable=False,
    )

    results = await run_supermemory_vqa(
        cast(MindBridge, api),
        (unanswerable,),
        _prepared_subject(),
        run_id="run_02",
        poll_interval_seconds=0.001,
    )

    assert results[0].predicted_option_index is None
    assert results[0].ranked_option_indices == ()
    assert results[0].mindbridge_abstained is True
    # unanswerable_option_index is 0 and so is correct_option_index here, so the old substitution
    # handed a retrieval failure both a correct answer and a reciprocal rank of 1.
    metrics = evaluate_supermemory_vqa((unanswerable,), results)
    assert metrics.qa_accuracy == 0.0
    assert metrics.qa_mean_reciprocal_rank == 0.0
    # Answerability is unaffected: None already reads as "not answerable".
    assert metrics.answerability_recall == 0.0


async def test_supermemory_rejects_missing_question_boundary_before_api_calls() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="question boundary"):
        await run_supermemory_vqa(
            cast(MindBridge, api),
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
        cast(MindBridge, api),
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


def test_supermemory_identity_requires_source_media() -> None:
    with pytest.raises(ValidationError, match="identity observations require source media"):
        SuperMemoryPreparedSegment(
            start_seconds=0,
            duration_ms=30_000,
            transcript="A person spoke.",
            identity_observations=(_identity(),),
        )


def test_supermemory_allows_short_container_tail_but_rejects_overrun() -> None:
    payload = _segment(0, "media_tail", duration_ms=9_000).model_dump()
    payload["media_objects"][0]["duration_ms"] = 8_200

    segment = SuperMemoryPreparedSegment.model_validate(payload)

    assert segment.media_objects[0].duration_ms == 8_200
    payload["media_objects"][0]["duration_ms"] = 9_001
    with pytest.raises(ValidationError, match="must be positive and not exceed its segment"):
        SuperMemoryPreparedSegment.model_validate(payload)
    payload["media_objects"][0]["duration_ms"] = 0
    with pytest.raises(ValidationError, match="must be positive and not exceed its segment"):
        SuperMemoryPreparedSegment.model_validate(payload)
    payload["media_objects"][0]["duration_ms"] = None
    with pytest.raises(ValidationError, match="must be positive and not exceed its segment"):
        SuperMemoryPreparedSegment.model_validate(payload)
    payload["media_objects"][0]["duration_ms"] = 999
    payload["identity_observations"] = [_identity().model_dump()]
    with pytest.raises(ValidationError, match="identity observation exceeds its source media"):
        SuperMemoryPreparedSegment.model_validate(payload)


async def test_supermemory_answers_later_boundaries_when_one_segment_never_ingests() -> None:
    """A dead segment used to abort the subject, discarding every segment already ingested."""

    class FailingMemoryApi(RecordingMemoryApi):
        async def observe(self, request: ObserveRequest) -> ObservationReceipt:
            if request.sequence == 1:
                raise MindBridgeError(
                    "segment could not be observed",
                    code="model_request_failed",
                    status_code=502,
                    trace_id="trace_ingest_error",
                )
            return await super().observe(request)

    api = FailingMemoryApi()

    results = await run_supermemory_vqa(
        cast(MindBridge, api),
        (
            _question(1, question_ended_at=ORIGIN),
            _question(2, question_ended_at=ORIGIN + timedelta(seconds=45)),
        ),
        _prepared_subject(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert [request.sequence for request in api.observe_requests] == [0]
    assert "remember:B said the mug was in the sink." in api.calls
    assert results[1].ranked_option_indices == (2, 0, 3, 1)
    # The first question was answered before the failure, so it must not inherit it.
    assert results[0].mindbridge_ingest_failure_count == 0
    assert results[1].mindbridge_ingest_failure_count == 1


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
                    _segment(
                        0,
                        "media_0",
                        transcript="B said the mug was in the sink.",
                        identity_observations=(_identity(),),
                    ),
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
    identity_observations: tuple[IdentityObservationInput, ...] = (),
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
        identity_observations=identity_observations,
    )


def _identity() -> IdentityObservationInput:
    return IdentityObservationInput(
        identity_id="person_device_01",
        kind=IdentityKind.FACE,
        start_ms=0,
        end_ms=1_000,
        confidence=0.9,
        model_id="insightface/buffalo_l",
        visual_bbox_xyxy=(0.1, 0.1, 0.5, 0.8),
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
