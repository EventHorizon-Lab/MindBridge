"""Production-contract checks for Video-MME and EgoTempo runners."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import AsyncMindBridge
from mindbridge.benchmarks import (
    EgoTempoQuestion,
    PreparedVideo,
    PreparedVideoSegment,
    VideoMMEQuestion,
    VideoMMEVideo,
    evaluate_video_mme,
    run_egotempo_clip,
    run_video_mme_video,
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
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id="observation_0",
            processing_job_id="job_0",
            evidence_ids=("evidence_0",),
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_0",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            memory_ids=(),
            created_at=ORIGIN,
            updated_at=ORIGIN,
            trace_id=f"trace_{tenant_id}",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer=self.answer,
            confidence=0.8,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )


def test_prepared_video_accepts_decimal_contiguous_segments() -> None:
    video = PreparedVideo(
        video_id="video_01",
        timeline_origin=ORIGIN,
        segments=(
            PreparedVideoSegment(
                segment_id="segment_0",
                start_seconds=0.1,
                duration_ms=200,
                transcript="First segment.",
            ),
            PreparedVideoSegment(
                segment_id="segment_1",
                start_seconds=0.3,
                duration_ms=100,
                transcript="Second segment.",
            ),
        ),
    )

    assert len(video.segments) == 2


async def test_video_mme_emits_official_nested_response() -> None:
    api = RecordingMemoryApi("The best answer is C.")
    video = VideoMMEVideo(
        video_id="001",
        duration="short",
        domain="Knowledge",
        sub_category="Humanity & History",
        source_url="https://www.youtube.com/watch?v=source",
        source_video_id="source",
        questions=(
            VideoMMEQuestion(
                question_id="001-1",
                task_type="Counting Problem",
                question="Which decoration appears most?",
                options=("A. Apples.", "B. Candles.", "C. Berries.", "D. Equal."),
                answer="C",
            ),
        ),
    )

    result = await run_video_mme_video(
        cast(AsyncMindBridge, api), video, _prepared("001", with_media=True), run_id="run_01"
    )

    query = api.recall_requests[0].query.text
    assert query is not None
    assert api.remember_requests[0].tenant_id == "benchmark_video_mme_001_run_01"
    assert api.remember_requests[0].evidence_ids == ("evidence_0",)
    assert api.observe_requests[0].media_objects[0].kind is MediaKind.VIDEO
    assert "Respond with only the letter (A, B, C, or D)" in query
    assert result.questions[0].response == "C"
    assert result.questions[0].mindbridge_model_answer == "The best answer is C."
    assert evaluate_video_mme((result,)).accuracy == 1.0


async def test_egotempo_emits_official_judge_fields_without_answer_leakage() -> None:
    api = RecordingMemoryApi("A spoon.")
    question = EgoTempoQuestion(
        question_id="source_0.0_2.0_0",
        clip_id="source_0.0_2.0",
        source_video_id="source",
        clip_start_seconds=0,
        clip_end_seconds=2,
        question_type="action-specific object",
        question="What did the person pick up?",
        reference_answer="SECRET A spoon.",
    )

    results = await run_egotempo_clip(
        cast(AsyncMindBridge, api),
        (question,),
        _prepared("source_0.0_2.0"),
        run_id="run_01",
    )

    official = results[0].model_dump(mode="json", by_alias=True)
    query = api.recall_requests[0].query.text
    assert query is not None
    assert api.recall_requests[0].tenant_id == "benchmark_egotempo_source_0.0_2.0_run_01"
    assert "person recording the video" in query
    assert "SECRET" not in api.recall_requests[0].model_dump_json()
    assert {key: official[key] for key in ("V", "Q", "QA", "A", "C", "M")} == {
        "V": "source_0.0_2.0",
        "Q": "What did the person pick up?",
        "QA": query,
        "A": "A spoon.",
        "C": "SECRET A spoon.",
        "M": "action-specific object",
    }


async def test_new_video_runners_validate_before_ingestion() -> None:
    api = RecordingMemoryApi("C")
    video = VideoMMEVideo(
        video_id="001",
        duration="short",
        domain="Knowledge",
        sub_category="History",
        source_url="https://example.com/video",
        source_video_id="source",
        questions=(
            VideoMMEQuestion(
                question_id="001-1",
                task_type="Counting",
                question="Which?",
                options=("A. One.", "B. Two.", "C. Three.", "D. Four."),
                answer="C",
            ),
        ),
    )

    with pytest.raises(ValueError, match="between 1 and 100"):
        await run_video_mme_video(
            cast(AsyncMindBridge, api),
            video,
            _prepared("001"),
            run_id="run_01",
            recall_limit=101,
        )

    assert not api.remember_requests


def _prepared(video_id: str, *, with_media: bool = False) -> PreparedVideo:
    media_objects = (
        (
            MediaObjectInput(
                media_object_id=f"media_{video_id}",
                kind=MediaKind.VIDEO,
                uri=f"s3://benchmark/{video_id}.mp4",
                sha256="a" * 64,
                size_bytes=100,
                created_at=ORIGIN,
                duration_ms=1_000,
            ),
        )
        if with_media
        else ()
    )
    return PreparedVideo(
        video_id=video_id,
        timeline_origin=ORIGIN,
        segments=(
            PreparedVideoSegment(
                segment_id="segment_0",
                start_seconds=0,
                duration_ms=1_000,
                media_objects=media_objects,
                transcript="The person picked up a spoon near some berries.",
            ),
        ),
    )
