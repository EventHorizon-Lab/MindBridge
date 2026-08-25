"""Production-contract checks for the M3-Bench runner."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from pydantic import ValidationError

from mindbridge import MindBridge
from mindbridge.benchmarks.m3_bench import M3BenchQuestion, M3BenchVideo
from mindbridge.benchmarks.m3_runner import M3PreparedClip, M3PreparedVideo, run_m3_video
from mindbridge.benchmarks.runtime import wait_for_observation_job
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

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, job_states: list[JobState] | None = None) -> None:
        self.calls: list[str] = []
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self.job_states = job_states or [JobState.SUCCEEDED]

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.calls.append(f"observe:{request.sequence}")
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id=f"observation_{request.sequence}",
            processing_job_id=f"job_{request.sequence}",
            evidence_ids=(f"evidence_{request.sequence}",),
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

    async def remember(self, request: RememberRequest) -> MemoryView:
        self.calls.append(f"remember:{request.summary}")
        self.remember_requests.append(request)
        return MemoryView(
            memory_id="memory_caption",
            memory_type=MemoryType.EPISODIC,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
        )


class IngestFailingMemoryApi(RecordingMemoryApi):
    """Fail one clip's ingestion permanently, the way a clip the perceiver cannot read does."""

    def __init__(self, failing_sequence: int) -> None:
        super().__init__()
        self.failing_sequence = failing_sequence

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        if request.sequence == self.failing_sequence:
            self.calls.append(f"observe_failed:{request.sequence}")
            raise MindBridgeError(
                "clip could not be observed",
                code="model_request_failed",
                status_code=502,
                trace_id="trace_ingest_error",
            )
        return await super().observe(request)


async def test_m3_answers_at_clip_boundaries_without_future_ingestion() -> None:
    api = RecordingMemoryApi()
    results = await run_m3_video(
        cast(MindBridge, api),
        _annotation(),
        _prepared_video(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert set(api.calls[:4]) == {
        "observe:0",
        "job:job_0:succeeded",
        "recall:What was visible first?",
        "observe:1",
    }
    assert api.calls.index("observe:0") < api.calls.index("job:job_0:succeeded")
    assert api.calls.index("job:job_0:succeeded") < api.calls.index(
        "recall:What was visible first?"
    )
    assert api.calls[-2:] == [
        "job:job_1:succeeded",
        "recall:What happened overall?",
    ]
    assert api.observe_requests[0].occurred_at == NOW
    assert api.observe_requests[1].occurred_at.timestamp() - NOW.timestamp() == 30
    assert api.recall_requests[0].filters.occurred_before == NOW + timedelta(seconds=30)
    assert api.recall_requests[1].filters.occurred_before == NOW + timedelta(seconds=60)
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
        await run_m3_video(cast(MindBridge, api), annotation, _prepared_video(), run_id="run_01")

    assert api.calls == []


async def test_m3_caption_protocol_uses_same_causal_boundary() -> None:
    api = RecordingMemoryApi()
    prepared = M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(M3PreparedClip(clip_index=0, caption="A person entered.", duration_ms=30_000),),
    )
    annotation = _annotation(questions=(_annotation().questions[0],))

    await run_m3_video(cast(MindBridge, api), annotation, prepared, run_id="run_01")

    assert api.calls == [
        "remember:A person entered.",
        "recall:What was visible first?",
    ]
    assert api.remember_requests[0].ended_at == NOW.replace(second=30)


async def test_m3_binds_released_caption_to_source_video() -> None:
    api = RecordingMemoryApi()
    prepared = M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(
            M3PreparedClip(
                clip_index=0,
                media_object=_media("media_0", MediaKind.VIDEO),
                caption="A person entered.",
                duration_ms=30_000,
                identity_observations=(_identity(),),
            ),
        ),
    )

    await run_m3_video(
        cast(MindBridge, api),
        _annotation(questions=(_annotation().questions[0],)),
        prepared,
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert api.remember_requests[0].evidence_ids == ("evidence_0",)
    assert api.observe_requests[0].identity_observations == (_identity(),)


async def test_m3_keeps_released_events_and_inferences_in_separate_memory_types() -> None:
    class SerialMemoryApi(RecordingMemoryApi):
        remembering = False

        async def remember(self, request: RememberRequest) -> MemoryView:
            assert not self.remembering
            self.remembering = True
            await asyncio.sleep(0)
            result = await super().remember(request)
            self.remembering = False
            return result

    api = SerialMemoryApi()
    prepared = M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(
            M3PreparedClip(
                clip_index=0,
                caption=(
                    "[Event] A person placed a mug on the table.\n"
                    "[Inference] The person appears to be preparing tea."
                ),
                duration_ms=30_000,
            ),
        ),
    )

    await run_m3_video(
        cast(MindBridge, api),
        _annotation(questions=(_annotation().questions[0],)),
        prepared,
        run_id="run_01",
    )

    assert [(request.memory_type, request.summary) for request in api.remember_requests] == [
        (MemoryType.EPISODIC, "[Event] A person placed a mug on the table."),
        (MemoryType.SEMANTIC, "[Inference] The person appears to be preparing tea."),
    ]


async def test_m3_counts_bounded_model_failures_as_incorrect_and_continues() -> None:
    class ModelFailingMemoryApi(RecordingMemoryApi):
        async def recall(self, request: RecallRequest) -> RecallResult:
            if not self.recall_requests:
                self.recall_requests.append(request)
                raise MindBridgeError(
                    "invalid model output",
                    code="model_output_invalid",
                    status_code=502,
                    trace_id="trace_model_error",
                )
            return await super().recall(request)

    results = await run_m3_video(
        cast(MindBridge, ModelFailingMemoryApi()),
        _annotation(),
        _prepared_video(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert results[0].response == ""
    assert results[0].mindbridge_error_code == "model_output_invalid"
    assert results[0].mindbridge_trace_id == "trace_model_error"
    assert results[1].response == "grounded prediction"


async def test_job_waiter_allows_failed_attempt_to_be_retried() -> None:
    api = RecordingMemoryApi([JobState.FAILED, JobState.SUCCEEDED])

    job = await wait_for_observation_job(
        cast(MindBridge, api),
        "tenant",
        "job_0",
        poll_interval_seconds=0.001,
        timeout_seconds=1.0,
    )

    assert job.state is JobState.SUCCEEDED


async def test_job_waiter_gives_up_on_a_stuck_failure_and_reports_its_error_code() -> None:
    """A failure that never advances its attempt used to burn the whole processing timeout."""
    api = RecordingMemoryApi([JobState.FAILED])

    with pytest.raises(RuntimeError, match="error code model_unavailable"):
        await wait_for_observation_job(
            cast(MindBridge, api),
            "tenant",
            "job_0",
            poll_interval_seconds=0.001,
            timeout_seconds=30.0,
            failed_grace_seconds=0.01,
        )


async def test_job_waiter_times_out_when_status_request_never_returns() -> None:
    class HangingMemoryApi(RecordingMemoryApi):
        async def get_observation_job(
            self, tenant_id: str, job_id: str
        ) -> ObservationProcessingJobView:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    with pytest.raises(TimeoutError, match="last state was unavailable"):
        await wait_for_observation_job(
            cast(MindBridge, HangingMemoryApi()),
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
    with pytest.raises(ValidationError, match="require video or a caption"):
        M3PreparedClip(clip_index=0)
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
    with pytest.raises(ValidationError, match="identity observation exceeds"):
        M3PreparedClip(
            clip_index=0,
            media_object=_media("media_identity", MediaKind.VIDEO),
            identity_observations=(_identity(end_ms=30_001),),
        )
    with pytest.raises(ValidationError, match="identity observations require source video"):
        M3PreparedClip(
            clip_index=0,
            caption="A person entered.",
            duration_ms=30_000,
            identity_observations=(_identity(),),
        )


async def test_m3_boundary_batch_survives_one_clip_that_never_ingests() -> None:
    """A dead clip used to abort the whole video, discarding every clip already ingested."""
    api = IngestFailingMemoryApi(1)
    prepared = M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(_clip(0), _clip(1), _clip(2)),
    )
    annotation = _annotation(
        questions=(
            M3BenchQuestion(
                question_id="video_01_Q01",
                question="What was visible first?",
                reference_answer="SECRET FIRST ANSWER",
                question_types=("Visual",),
                before_clip_index=2,
            ),
        )
    )

    results = await run_m3_video(
        cast(MindBridge, api),
        annotation,
        prepared,
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert [request.sequence for request in api.observe_requests] == [0, 2]
    assert {"job:job_0:succeeded", "job:job_2:succeeded"} <= set(api.calls)
    assert results[0].response == "grounded prediction"
    assert results[0].mindbridge_ingest_failure_count == 1


async def test_m3_final_batch_survives_one_clip_that_never_ingests() -> None:
    """The trailing ingest is its own fan-out, so it needs its own accounting."""
    api = IngestFailingMemoryApi(2)
    prepared = M3PreparedVideo(
        video_id="video_01",
        timeline_origin=NOW,
        clips=(_clip(0), _clip(1), _clip(2)),
    )

    results = await run_m3_video(
        cast(MindBridge, api),
        _annotation(),
        prepared,
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert [request.sequence for request in api.observe_requests] == [0, 1]
    assert "job:job_1:succeeded" in api.calls
    assert results[1].response == "grounded prediction"
    # The first question was answered before the failure, so it must not inherit it.
    assert results[0].mindbridge_ingest_failure_count == 0
    assert results[1].mindbridge_ingest_failure_count == 1


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


_SUFFIX_BY_KIND = {MediaKind.VIDEO: "mp4", MediaKind.IMAGE: "png", MediaKind.AUDIO: "wav"}


def _media(
    media_object_id: str,
    kind: MediaKind,
    *,
    duration_ms: int = 30_000,
) -> MediaObjectInput:
    return MediaObjectInput(
        media_object_id=media_object_id,
        kind=kind,
        # The extension has to follow the kind, or MediaObjectInput rejects the pair before the
        # M3 clip rules under test ever run.
        uri=f"s3://benchmark/{media_object_id}.{_SUFFIX_BY_KIND[kind]}",
        sha256="a" * 64,
        size_bytes=1_024,
        created_at=NOW,
        duration_ms=duration_ms,
    )


def _identity(*, end_ms: int = 1_000) -> IdentityObservationInput:
    return IdentityObservationInput(
        identity_id="person_device_01",
        kind=IdentityKind.FACE,
        start_ms=0,
        end_ms=end_ms,
        confidence=0.9,
        model_id="insightface/buffalo_l",
        visual_bbox_xyxy=(0.1, 0.1, 0.5, 0.8),
    )


async def test_m3_boundary_cohort_survives_one_question_whose_recall_raises() -> None:
    """One raising recall used to abort every question sharing its clip boundary."""

    class RecallFailingMemoryApi(RecordingMemoryApi):
        async def recall(self, request: RecallRequest) -> RecallResult:
            if request.query.text == "What was visible first?":
                raise MindBridgeError(
                    "recall could not be served",
                    code="internal_error",
                    status_code=500,
                    trace_id="trace_recall_error",
                )
            return await super().recall(request)

    api = RecallFailingMemoryApi()
    annotation = _annotation(
        questions=(
            M3BenchQuestion(
                question_id="video_01_Q01",
                question="What was visible first?",
                reference_answer="SECRET FIRST ANSWER",
                question_types=("Visual",),
                before_clip_index=0,
            ),
            M3BenchQuestion(
                question_id="video_01_Q02",
                question="What else was visible?",
                reference_answer="SECRET SECOND ANSWER",
                question_types=("Visual",),
                before_clip_index=0,
            ),
        )
    )

    results = await run_m3_video(
        cast(MindBridge, api),
        annotation,
        _prepared_video(),
        run_id="run_01",
        poll_interval_seconds=0.001,
    )

    assert [result.id for result in results] == ["video_01_Q01", "video_01_Q02"]
    assert [result.mindbridge_error_code for result in results] == ["internal_error", None]
    assert [result.response for result in results] == ["", "grounded prediction"]
