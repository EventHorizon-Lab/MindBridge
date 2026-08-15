"""Production-contract checks for MM-Lifelong answer and Ref@300 output."""

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from mindbridge import AsyncMindBridge
from mindbridge.benchmarks import (
    MMLifelongPreparedSegment,
    MMLifelongPreparedTimeline,
    MMLifelongQuestion,
    reference_at_n,
    run_mm_lifelong,
)
from mindbridge.contracts import MemoryView, RecallRequest, RecallResult, RememberRequest
from mindbridge.core import MemoryState, MemoryType, VerificationStatus

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


async def test_mm_lifelong_emits_official_answer_shape_and_ref_300() -> None:
    api = RecordingMemoryApi()

    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_mm_lifelong(
            cast(AsyncMindBridge, api),
            (_question(),),
            _prepared(),
            run_id="run_01",
            recall_limit=101,
        )
    assert not api.remember_requests

    results = await run_mm_lifelong(
        cast(AsyncMindBridge, api),
        (_question(),),
        _prepared(),
        run_id="run_01",
    )

    assert api.remember_requests[0].occurred_at == ORIGIN
    assert api.recall_requests[0].tenant_id == "benchmark_mm_lifelong_day_test_run_01"
    assert results[0].pred.answer == "A meeting"
    assert results[0].pred.intervals == ((100.0, 200.0),)
    assert results[0].ref_300 == 1.0


async def test_mm_lifelong_rejects_timeline_that_cannot_cover_labels() -> None:
    api = RecordingMemoryApi()
    question = _question(reference_intervals=((700.0, 710.0),))

    with pytest.raises(ValueError, match="does not cover"):
        await run_mm_lifelong(
            cast(AsyncMindBridge, api),
            (question,),
            _prepared(),
            run_id="run_01",
        )

    assert not api.remember_requests


def test_mm_lifelong_reference_at_n_matches_official_bucket_jaccard() -> None:
    assert reference_at_n(((0.0, 600.0),), ((300.0, 600.0),), 600.0) == 0.5


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


def _prepared() -> MMLifelongPreparedTimeline:
    return MMLifelongPreparedTimeline(
        split="day_test",
        timeline_origin=ORIGIN,
        segments=(
            MMLifelongPreparedSegment(
                segment_id="segment_01",
                start_seconds=0,
                duration_ms=600_000,
                caption="A meeting took place.",
            ),
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
