"""Production-contract checks for the M3-Bench runner."""

import asyncio
from datetime import datetime, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import AsyncMindBridge
from mindbridge.benchmarks import (
    M3BenchQuestion,
    M3BenchVideo,
    M3PreparedClip,
    M3PreparedVideo,
    run_m3_video,
    wait_for_observation_job,
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

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, job_states: list[JobState] | None = None) -> None:
        self.calls: list[str] = []
        self.observe_requests: list[ObserveRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self.job_states = job_states or [JobState.SUCCEEDED]

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id=f"observation_{request.sequence}",
            processing_job_id=f"job_{request.sequence}",
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id=f"trace_observe_{request.sequence}",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        state = self.job_states.pop(0) if len(self.job_states) > 1 else self.job_states[0]
        self.calls.append(f"job:{job_id}:{state.value}")
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id=job_id.replace("job", "observation"),
            state=state,
            attempt=0 if state is JobState.PENDING else 1,
            error_code="model_unavailable" if state is JobState.FAILED else None,
            created_at=NOW,
            updated_at=NOW,
            trace_id=f"trace_{tenant_id}",
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.calls.append(f"recall:{request.query.text}")
        self.recall_requests.append(request)
        return RecallResult(
            answer="grounded prediction",
            confidence=0.8,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )


async def test_m3_answers_at_clip_boundaries_without_future_ingestion() -> None:
    api = RecordingMemoryApi()
    results = await run_m3_video(
        cast(AsyncMindBridge, api),
        _annotation(),
        _prepared_video(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.calls == [
        "observe:0",
        "job:job_0:succeeded",
        "recall:What was visible first?",
        "observe:1",
        "job:job_1:succeeded",
        "recall:What happened overall?",
    ]
    assert api.observe_requests[0].occurred_at == NOW
    assert api.observe_requests[1].occurred_at.timestamp() - NOW.timestamp() == 30
    assert "SECRET FIRST ANSWER" not in api.recall_requests[0].model_dump_json()
    assert results[0].answer == "SECRET FIRST ANSWER"
    assert results[0].response == "grounded prediction"
    assert results[0].before_clip == 0


async def test_m3_rejects_out_of_range_boundary_before_api_calls() -> None:
    api = RecordingMemoryApi()
    annotation = _annotation(
        questions=(
            M3BenchQuestion(
                question_id="video_01_Q99",
                question="What happens in a future clip?",
                reference_answer="SECRET",
                question_types=("Temporal",),
                before_clip_index=2,
            ),
        )
    )

    with pytest.raises(ValueError, match="boundary"):
        await run_m3_video(
            cast(AsyncMindBridge, api), annotation, _prepared_video(), run_id="run_01"
        )

    assert api.calls == []


async def test_job_waiter_allows_failed_attempt_to_be_retried() -> None:
    api = RecordingMemoryApi([JobState.FAILED, JobState.SUCCEEDED])

    job = await wait_for_observation_job(
        cast(AsyncMindBridge, api),
        "tenant",
        "job_0",
        poll_interval_seconds=0.001,
        timeout_seconds=1.0,
    )

    assert job.state is JobState.SUCCEEDED


async def test_job_waiter_times_out_when_status_request_never_returns() -> None:
    class HangingMemoryApi(RecordingMemoryApi):
        async def get_observation_job(
            self, tenant_id: str, job_id: str
        ) -> ObservationProcessingJobView:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    with pytest.raises(TimeoutError, match="last state was unavailable"):
        await wait_for_observation_job(
            cast(AsyncMindBridge, HangingMemoryApi()),
            "tenant",
            "job_0",
            poll_interval_seconds=0.001,
            timeout_seconds=0.01,
        )


def test_prepared_video_requires_contiguous_video_clips() -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        M3PreparedVideo(video_id="video_01", timeline_origin=NOW, clips=(_clip(1),))
    with pytest.raises(ValidationError, match="video media"):
        M3PreparedClip(
            clip_index=0,
            media_object=_media("media_image", MediaKind.IMAGE),
        )
    with pytest.raises(ValidationError, match="must not exceed 30 seconds"):
        M3PreparedClip(
            clip_index=0,
            media_object=_media("media_long", MediaKind.VIDEO, duration_ms=30_001),
        )
    with pytest.raises(ValidationError, match="before the final clip"):
        M3PreparedVideo(
            video_id="video_01",
            timeline_origin=NOW,
            clips=(
                M3PreparedClip(
                    clip_index=0,
                    media_object=_media("media_short", MediaKind.VIDEO, duration_ms=29_999),
                ),
                _clip(1),
            ),
        )


def _annotation(*, questions: tuple[M3BenchQuestion, ...] | None = None) -> M3BenchVideo:
    return M3BenchVideo(
        video_id="video_01",
        video_path="data/videos/robot/video_01.mp4",
        questions=questions
        or (
            M3BenchQuestion(
                question_id="video_01_Q01",
                question="What was visible first?",
                reference_answer="SECRET FIRST ANSWER",
                question_types=("Visual",),
                timestamp_seconds=10,
                before_clip_index=0,
            ),
            M3BenchQuestion(
                question_id="video_01_Q02",
                question="What happened overall?",
                reference_answer="SECRET OVERALL ANSWER",
                question_types=("Temporal",),
            ),
        ),
    )


def _prepared_video() -> M3PreparedVideo:
    return M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(_clip(0), _clip(1)),
    )


def _clip(index: int) -> M3PreparedClip:
    return M3PreparedClip(
        clip_index=index,
        media_object=_media(f"media_{index}", MediaKind.VIDEO),
    )


def _media(
    media_object_id: str,
    kind: MediaKind,
    *,
    duration_ms: int = 30_000,
) -> MediaObjectInput:
    return MediaObjectInput(
        media_object_id=media_object_id,
        kind=kind,
        uri=f"s3://benchmark/{media_object_id}.mp4",
        sha256="a" * 64,
        size_bytes=1_024,
        created_at=NOW,
        duration_ms=duration_ms,
    )
