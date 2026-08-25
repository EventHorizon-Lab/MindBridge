"""Production-contract checks for MM-Lifelong answer and Ref@300 output."""

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.mm_lifelong import MMLifelongQuestion
from mindbridge.benchmarks.mm_lifelong_runner import (
    MMLifelongPreparedSegment,
    MMLifelongPreparedTimeline,
    run_mm_lifelong,
    unofficial_reference_at_n,
)
from mindbridge.contracts import MemoryView, RecallRequest, RecallResult, RememberRequest
from mindbridge.core import MemoryState, MemoryType, VerificationStatus
from mindbridge.sdk import MindBridgeError

ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def remember(self, request: RememberRequest) -> object:
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer="A meeting",
            confidence=0.8,
            memories=(_memory(),),
            evidence=(),
            trace_id="trace_recall",
        )


async def test_mm_lifelong_emits_official_answer_shape_and_local_diagnostic() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_mm_lifelong(
            cast(MindBridge, api),
            (_question(),),
            _prepared(),
            run_id="run_01",
            recall_limit=101,
        )
    assert not api.remember_requests

    results = await run_mm_lifelong(
        cast(MindBridge, api),
        (_question(),),
        _prepared(),
        run_id="run_01",
    )

    assert api.remember_requests[0].occurred_at == ORIGIN
    assert api.recall_requests[0].tenant_id == "benchmark_mm_lifelong_day_test_run_01"
    assert results[0].pred.answer == "A meeting"
    assert results[0].pred.intervals == ((100.0, 200.0),)
    assert results[0].mindbridge_unofficial_ref_at_300 == 1.0


async def test_mm_lifelong_rejects_timeline_that_cannot_cover_labels() -> None:
    api = RecordingMemoryApi()
    question = _question(reference_intervals=((700.0, 710.0),))

    with pytest.raises(ValueError, match="does not cover"):
        await run_mm_lifelong(
            cast(MindBridge, api),
            (question,),
            _prepared(),
            run_id="run_01",
        )

    assert not api.remember_requests


def test_mm_lifelong_unofficial_reference_at_n_uses_bucket_jaccard() -> None:
    assert unofficial_reference_at_n(((0.0, 600.0),), ((300.0, 600.0),), 600.0) == 0.5


async def test_mm_lifelong_keeps_ingested_segments_when_one_segment_fails() -> None:
    """A segment that cannot be written used to discard the whole timeline behind it."""

    class FailingMemoryApi(RecordingMemoryApi):
        async def remember(self, request: RememberRequest) -> object:
            if request.summary == "segment_02 happened.":
                raise MindBridgeError(
                    "segment could not be written",
                    code="model_request_failed",
                    status_code=502,
                    trace_id="trace_ingest_error",
                )
            return await super().remember(request)

    api = FailingMemoryApi()

    results = await run_mm_lifelong(
        cast(MindBridge, api),
        (_question(),),
        _prepared(segment_count=3),
        run_id="run_01",
    )

    assert [request.summary for request in api.remember_requests] == [
        "A meeting took place.",
        "segment_03 happened.",
    ]
    assert results[0].pred.answer == "A meeting"
    assert results[0].mindbridge_ingest_failure_count == 1


def _question(
    *, reference_intervals: tuple[tuple[float, float], ...] = ((100.0, 200.0),)
) -> MMLifelongQuestion:
    return MMLifelongQuestion(
        index=0,
        split="day_test",
        question="What happened?",
        reference_answer="A meeting",
        question_type="Event Recognition",
        temporal_certificate="Short",
        clue_interval_count=1,
        reference_intervals=reference_intervals,
    )


def _prepared(*, segment_count: int = 1) -> MMLifelongPreparedTimeline:
    return MMLifelongPreparedTimeline(
        split="day_test",
        timeline_origin=ORIGIN,
        segments=tuple(
            MMLifelongPreparedSegment(
                segment_id=f"segment_{index + 1:02d}",
                start_seconds=index * 600,
                duration_ms=600_000,
                caption=(
                    "A meeting took place." if index == 0 else f"segment_{index + 1:02d} happened."
                ),
            )
            for index in range(segment_count)
        ),
    )


def _memory() -> MemoryView:
    return MemoryView(
        memory_id="memory_01",
        memory_type=MemoryType.EPISODIC,
        summary="A meeting took place.",
        evidence_ids=(),
        occurred_at=ORIGIN + timedelta(seconds=100),
        ended_at=ORIGIN + timedelta(seconds=200),
        created_at=ORIGIN,
        verification_status=VerificationStatus.ATTESTED,
        state=MemoryState.ACTIVE,
    )


async def test_mm_lifelong_keeps_answers_when_one_question_recall_raises() -> None:
    """One raising recall used to abort every question in the same concurrency window."""

    class RecallFailingMemoryApi(RecordingMemoryApi):
        async def recall(self, request: RecallRequest) -> RecallResult:
            if request.query.text == "What happened first?":
                raise MindBridgeError(
                    "recall could not be served",
                    code="internal_error",
                    status_code=500,
                    trace_id="trace_recall_error",
                )
            return await super().recall(request)

    api = RecallFailingMemoryApi()
    questions = tuple(
        _question().model_copy(update={"index": index, "question": text})
        for index, text in enumerate(("What happened first?", "What happened next?"))
    )

    results = await run_mm_lifelong(
        cast(MindBridge, api),
        questions,
        _prepared(),
        run_id="run_01",
    )

    assert [result.index for result in results] == [0, 1]
    assert [result.mindbridge_error_code for result in results] == ["internal_error", None]
    assert results[0].pred.answer == ""
    assert results[0].pred.intervals == ()
    assert results[0].mindbridge_unofficial_ref_at_300 == 0.0
    assert results[1].pred.answer == "A meeting"
