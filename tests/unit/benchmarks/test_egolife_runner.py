"""Production-contract checks for the causal EgoLifeQA runner."""

from datetime import datetime, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import AsyncMindBridge
from mindbridge.benchmarks import (
    EgoLifePreparedClip,
    EgoLifePreparedStream,
    EgoLifeQuestion,
    run_egolife_qa,
)
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
)
from mindbridge.core import JobState, MediaKind

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.observe_requests: list[ObserveRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id=f"observation_{request.sequence}",
            processing_job_id=f"job_{request.sequence}",
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


async def test_egolife_answers_before_ingesting_a_clip_that_crosses_query_time() -> None:
    api = RecordingMemoryApi()
    questions = (
        _question("q_late", "10004500"),
        _question("q_early", "10001500"),
    )

    results = await run_egolife_qa(
        cast(AsyncMindBridge, api),
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


def _question(question_id: str, timecode: str) -> EgoLifeQuestion:
    hours, minutes, seconds, centiseconds = map(
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
        query_offset_ms=(hours * 3_600 + minutes * 60 + seconds) * 1_000 + centiseconds * 10,
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
