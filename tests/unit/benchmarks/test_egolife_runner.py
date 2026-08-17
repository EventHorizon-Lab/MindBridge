"""Production-contract checks for the causal EgoLifeQA runner."""

import asyncio
from datetime import datetime, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import MindBridge
from mindbridge.benchmarks import (
    EgoLifeOption,
    EgoLifePreparedClip,
    EgoLifePreparedStream,
    EgoLifeQuestion,
    EgoLifeQuestionResult,
    EgoMemReasonQuestion,
    evaluate_egolife_qa,
    run_egolife_qa,
    run_egomem_reason,
)
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

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
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

    async def recall(self, request: RecallRequest) -> RecallResult:
        assert request.query.text is not None
        self.calls.append(f"recall:{request.query.text.splitlines()[0]}")
        self.recall_requests.append(request)
        return RecallResult(
            answer="B",
            confidence=0.8,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.calls.append(f"remember:{request.summary}")
        self.remember_requests.append(request)
        return object()


async def test_egolife_answers_before_ingesting_a_clip_that_crosses_query_time() -> None:
    api = RecordingMemoryApi()
    questions = (
        _question("q_late", "10004500"),
        _question("q_early", "10001500"),
    )

    results = await run_egolife_qa(
        cast(MindBridge, api),
        questions,
        _prepared_stream(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.calls == [
        "recall:Who used it?",
        "observe:0",
        "job:job_0",
        "recall:Who used it?",
    ]
    assert [result.id for result in results] == ["q_late", "q_early"]
    assert results[0].model_option == "B"
    assert api.recall_requests[0].filters.occurred_before == ORIGIN.replace(
        hour=10, minute=0, second=15
    )
    assert "SECRET" not in api.recall_requests[0].model_dump_json()


async def test_egolife_ingests_official_caption_without_reprocessing_video() -> None:
    api = RecordingMemoryApi()
    prepared = EgoLifePreparedStream(
        subject_id="A1_JAKE",
        timeline_origin=ORIGIN,
        clips=(
            EgoLifePreparedClip(
                day=1,
                start_timecode="10000000",
                caption="Jake passes the phone to Alice.",
                duration_ms=30_000,
            ),
        ),
    )

    await run_egolife_qa(
        cast(MindBridge, api),
        (_question("q_1", "10004500"),),
        prepared,
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.calls == [
        "remember:Jake passes the phone to Alice.",
        "recall:Who used it?",
    ]
    assert api.remember_requests[0].occurred_at == ORIGIN.replace(hour=10)
    assert api.remember_requests[0].ended_at == ORIGIN.replace(hour=10, second=30)


async def test_egolife_binds_released_caption_to_source_video() -> None:
    api = RecordingMemoryApi()
    prepared = EgoLifePreparedStream(
        subject_id="A1_JAKE",
        timeline_origin=ORIGIN,
        clips=(
            EgoLifePreparedClip(
                day=1,
                start_timecode="10000000",
                media_object=_clip("10000000", "media_0", duration_ms=30_000).media_object,
                caption="Jake passes the phone to Alice.",
                duration_ms=30_000,
            ),
        ),
    )

    await run_egolife_qa(
        cast(MindBridge, api),
        (_question("q_1", "10004500"),),
        prepared,
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.remember_requests[0].evidence_ids == ("evidence_0",)


async def test_egolife_keeps_released_visual_and_audio_memories_separate() -> None:
    class SerialMemoryApi(RecordingMemoryApi):
        remembering = False

        async def remember(self, request: RememberRequest) -> object:
            assert not self.remembering
            self.remembering = True
            await asyncio.sleep(0)
            result = await super().remember(request)
            self.remembering = False
            return result

    api = SerialMemoryApi()
    prepared = EgoLifePreparedStream(
        subject_id="A1_JAKE",
        timeline_origin=ORIGIN,
        clips=(
            EgoLifePreparedClip(
                day=1,
                start_timecode="10000000",
                caption=(
                    "Visual 10000100-10000200: Jake picks up the phone.\n"
                    "Audio 10000200-10000300: Jake: I found it."
                ),
                duration_ms=30_000,
            ),
        ),
    )

    await run_egolife_qa(
        cast(MindBridge, api),
        (_question("q_1", "10004500"),),
        prepared,
        run_id="run_01",
    )

    assert [request.summary for request in api.remember_requests] == [
        "Visual 10000100-10000200: Jake picks up the phone.",
        "Audio 10000200-10000300: Jake: I found it.",
    ]


def test_prepared_egolife_rejects_overlapping_clips() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        EgoLifePreparedStream(
            subject_id="A1_JAKE",
            timeline_origin=ORIGIN,
            clips=(
                _clip("10000000", "media_0", duration_ms=30_000),
                _clip("10002000", "media_1", duration_ms=30_000),
            ),
        )


def test_egolife_metrics_use_exact_option_match() -> None:
    results = (
        _result("q_1", answer="B", prediction="B"),
        _result("q_2", answer="A", prediction=None),
    )

    metrics = evaluate_egolife_qa(results)

    assert metrics.question_count == 2
    assert metrics.correct_count == 1
    assert metrics.accuracy == 0.5


async def test_egomem_reason_reuses_causal_stream_and_supports_ten_options() -> None:
    api = RecordingMemoryApi()
    choices = tuple(f"Choice {index}" for index in range(10))
    questions = (
        EgoMemReasonQuestion(
            example_id=2,
            question_id="A1_JAKE_late",
            identity="A1_JAKE",
            query_time="DAY1, 10:00:45",
            query_offset_ms=36_045_000,
            question="Which choice is correct?",
            choices=choices,
            query_type="Event Ordering",
        ),
        EgoMemReasonQuestion(
            example_id=1,
            question_id="A1_JAKE_early",
            identity="A1_JAKE",
            query_time="DAY1, 10:00:15",
            query_offset_ms=36_015_000,
            question="Which choice is correct?",
            choices=choices,
            query_type="Event Ordering",
        ),
    )

    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_egomem_reason(
            cast(MindBridge, api),
            questions,
            _prepared_stream(),
            run_id="run_01",
            recall_limit=101,
        )
    assert not api.calls

    results = await run_egomem_reason(
        cast(MindBridge, api),
        questions,
        _prepared_stream(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.calls[:2] == [
        "recall:Query time reference: DAY1, 10:00:15",
        "observe:0",
    ]
    assert api.observe_requests[0].boot_id == "egomem_reason_official_v1"
    assert api.recall_requests[0].query.text is not None
    assert "Query time reference: DAY1, 10:00:15" in api.recall_requests[0].query.text
    assert "J. Choice 9" in api.recall_requests[0].query.text
    assert [result.example_id for result in results] == [2, 1]
    assert [result.predicted_answer for result in results] == ["B", "B"]


def _question(question_id: str, timecode: str) -> EgoLifeQuestion:
    hours, minutes, seconds, frames = map(
        int,
        (timecode[0:2], timecode[2:4], timecode[4:6], timecode[6:8]),
    )
    return EgoLifeQuestion(
        question_id=question_id,
        question="Who used it?",
        choices=("Tasha", "Alice", "Shure", "Lucia"),
        correct_option="B",
        query_day=1,
        query_timecode=timecode,
        query_offset_ms=(hours * 3_600 + minutes * 60 + seconds) * 1_000 + frames * 50,
        question_type="EntityLog",
        needs_audio=False,
        needs_name=True,
        asks_last_time=False,
    )


def _prepared_stream() -> EgoLifePreparedStream:
    return EgoLifePreparedStream(
        subject_id="A1_JAKE",
        timeline_origin=ORIGIN,
        clips=(
            _clip("10000000", "media_0", duration_ms=30_000),
            _clip("10003000", "media_1", duration_ms=30_000),
        ),
    )


def _clip(timecode: str, media_id: str, *, duration_ms: int) -> EgoLifePreparedClip:
    return EgoLifePreparedClip(
        day=1,
        start_timecode=timecode,
        media_object=MediaObjectInput(
            media_object_id=media_id,
            kind=MediaKind.VIDEO,
            uri=f"s3://benchmark/{media_id}.mp4",
            sha256="a" * 64,
            size_bytes=1_024,
            created_at=ORIGIN,
            duration_ms=duration_ms,
        ),
    )


def _result(
    question_id: str,
    *,
    answer: str,
    prediction: str | None,
) -> EgoLifeQuestionResult:
    return EgoLifeQuestionResult(
        id=question_id,
        subject_id="A1_JAKE",
        question="Who used it?",
        answer=cast(EgoLifeOption, answer),
        model_option=cast(EgoLifeOption | None, prediction),
        model_answer=prediction or "",
        question_type="EntityLog",
        query_day=1,
        query_timecode="10001500",
        mindbridge_confidence=0.8,
        mindbridge_memory_ids=(),
        mindbridge_evidence_ids=(),
        mindbridge_trace_id=f"trace_{question_id}",
    )


def test_metrics_break_out_the_five_official_question_types() -> None:
    metrics = evaluate_egolife_qa(
        (
            _typed_result("q_1", "EntityLog", correct=True),
            _typed_result("q_2", "EntityLog", correct=False),
            _typed_result("q_3", "RelationMap", correct=True),
        )
    )

    assert metrics.question_count == 3
    assert metrics.accuracy == pytest.approx(2 / 3)
    by_type = {category.question_type: category for category in metrics.categories}
    assert by_type["EntityLog"].question_count == 2
    assert by_type["EntityLog"].accuracy == pytest.approx(0.5)
    assert by_type["RelationMap"].accuracy == pytest.approx(1.0)


def test_category_order_is_stable_so_two_runs_compare_line_by_line() -> None:
    metrics = evaluate_egolife_qa(
        (
            _typed_result("q_1", "TaskMaster", correct=True),
            _typed_result("q_2", "EntityLog", correct=True),
            _typed_result("q_3", "HabitInsight", correct=False),
        )
    )

    assert tuple(category.question_type for category in metrics.categories) == (
        "EntityLog",
        "HabitInsight",
        "TaskMaster",
    )


def _typed_result(question_id: str, question_type: str, *, correct: bool) -> EgoLifeQuestionResult:
    return EgoLifeQuestionResult(
        id=question_id,
        subject_id="A1_JAKE",
        question="Who used it?",
        answer="B",
        model_option="B" if correct else "C",
        model_answer="B" if correct else "C",
        question_type=question_type,
        query_day=1,
        query_timecode="10001500",
        mindbridge_confidence=0.8,
        mindbridge_memory_ids=(),
        mindbridge_evidence_ids=(),
        mindbridge_trace_id=f"trace_{question_id}",
    )
