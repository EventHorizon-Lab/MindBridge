"""Production-contract checks for Video-MME, Video-MME-v2, and EgoTempo runners."""

from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.egotempo import EgoTempoQuestion, run_egotempo_clip
from mindbridge.benchmarks.runtime import PreparedVideo, PreparedVideoSegment
from mindbridge.benchmarks.video_mme import (
    VideoMMEOption,
    VideoMMEQuestion,
    VideoMMEQuestionResult,
    VideoMMEVideo,
    VideoMMEVideoResult,
    evaluate_video_mme,
    run_video_mme_video,
)
from mindbridge.benchmarks.video_mme_v2 import (
    VideoMMEV2Group,
    VideoMMEV2GroupType,
    VideoMMEV2Question,
    evaluate_video_mme_v2,
    run_video_mme_v2_group,
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
        cast(MindBridge, api), video, _prepared("001", with_media=True), run_id="run_01"
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


def test_video_mme_metrics_separate_official_accuracy_from_unanswered_questions() -> None:
    """An unparsed or failed row must not silently leave the reported denominator."""
    results = (
        _video_mme_result("001-1", answer="C", response="C"),
        _video_mme_result("001-2", answer="A", response=""),
        _video_mme_result("001-3", answer="B", response="", error_code="model_request_failed"),
    )

    metrics = evaluate_video_mme(results)

    assert metrics.question_count == 3
    assert metrics.answered_count == 1
    assert metrics.correct_count == 1
    assert metrics.error_count == 1
    assert metrics.accuracy == 1.0
    assert metrics.strict_accuracy == pytest.approx(1 / 3)


def _video_mme_result(
    question_id: str,
    *,
    answer: VideoMMEOption,
    response: str,
    error_code: str | None = None,
) -> VideoMMEVideoResult:
    return VideoMMEVideoResult(
        video_id="001",
        duration="short",
        domain="Knowledge",
        sub_category="Humanity & History",
        questions=(
            VideoMMEQuestionResult(
                question_id=question_id,
                task_type="Counting Problem",
                question="Which decoration appears most?",
                options=("A. Apples.", "B. Candles.", "C. Berries.", "D. Equal."),
                answer=answer,
                response=response,
                mindbridge_model_answer=response,
                mindbridge_confidence=0.0,
                mindbridge_memory_ids=(),
                mindbridge_evidence_ids=(),
                mindbridge_trace_id=f"trace_{question_id}",
                mindbridge_error_code=error_code,
            ),
        ),
    )


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
        cast(MindBridge, api),
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


class SegmentFailingMemoryApi(RecordingMemoryApi):
    """Fails the first segment's observation and serves every other request."""

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        if request.sequence == 0:
            raise RuntimeError("clip media could not be read")
        return await super().observe(request)


async def test_video_mme_answers_every_question_when_one_segment_fails_to_ingest() -> None:
    """A bare gather made the first exception the whole video's result: the segments that had
    already succeeded were cancelled with it and nothing was answered at all."""
    api = SegmentFailingMemoryApi("The best answer is C.")
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
            VideoMMEQuestion(
                question_id="001-2",
                task_type="Counting Problem",
                question="Which decoration appears least?",
                options=("A. Apples.", "B. Candles.", "C. Berries.", "D. Equal."),
                answer="C",
            ),
        ),
    )

    result = await run_video_mme_video(
        cast(MindBridge, api), video, _prepared_pair("001"), run_id="run_01"
    )

    # The surviving segment was ingested rather than cancelled alongside its failed sibling.
    assert [request.sequence for request in api.observe_requests] == [1]
    assert [question.response for question in result.questions] == ["C", "C"]
    # And the loss rides along on every answer, so an answer given over incomplete memory is
    # not read as a wrong one.
    assert [question.mindbridge_ingest_failure_count for question in result.questions] == [1, 1]


async def test_egotempo_answers_every_question_when_one_segment_fails_to_ingest() -> None:
    api = SegmentFailingMemoryApi("A spoon.")
    questions = tuple(
        EgoTempoQuestion(
            question_id=f"source_0.0_2.0_{index}",
            clip_id="source_0.0_2.0",
            source_video_id="source",
            clip_start_seconds=0,
            clip_end_seconds=2,
            question_type="action-specific object",
            question="What did the person pick up?",
            reference_answer="A spoon.",
        )
        for index in range(2)
    )

    results = await run_egotempo_clip(
        cast(MindBridge, api), questions, _prepared_pair("source_0.0_2.0"), run_id="run_01"
    )

    assert [request.sequence for request in api.observe_requests] == [1]
    assert [result.model_answer for result in results] == ["A spoon.", "A spoon."]
    assert [result.mindbridge_ingest_failure_count for result in results] == [1, 1]


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
            cast(MindBridge, api),
            video,
            _prepared("001"),
            run_id="run_01",
            recall_limit=101,
        )

    assert not api.remember_requests


def _prepared_pair(video_id: str) -> PreparedVideo:
    """Two segments with media, so one can fail while the other is in the same gather."""
    return PreparedVideo(
        video_id=video_id,
        timeline_origin=ORIGIN,
        segments=tuple(
            PreparedVideoSegment(
                segment_id=f"segment_{index}",
                start_seconds=index,
                duration_ms=1_000,
                media_objects=(
                    MediaObjectInput(
                        media_object_id=f"media_{video_id}_{index}",
                        kind=MediaKind.VIDEO,
                        uri=f"s3://benchmark/{video_id}_{index}.mp4",
                        sha256="a" * 64,
                        size_bytes=100,
                        created_at=ORIGIN,
                        duration_ms=1_000,
                    ),
                ),
                transcript="The person picked up a spoon near some berries.",
            )
            for index in range(2)
        ),
    )


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


def _video_mme_v2_group(group_type: VideoMMEV2GroupType = "logic") -> VideoMMEV2Group:
    return VideoMMEV2Group(
        video_id="001",
        source_url="https://www.youtube.com/watch?v=source",
        group_type=group_type,
        group_structure="[1,2,3,4]" if group_type == "logic" else "4",
        questions=tuple(
            VideoMMEV2Question(
                question_id=f"001-{position}",
                position=position,
                question=f"Question {position}?",
                options=tuple(f"{label}. Option {label}." for label in "ABCDEFGH"),
                answer="F",
                level="1",
                second_head="Frames & Audio",
                third_head="Visual-Audio Collaborative Reasoning",
            )
            for position in range(1, 5)
        ),
    )


async def test_video_mme_v2_answers_a_whole_group_with_the_eight_option_instruction() -> None:
    api = RecordingMemoryApi("The best answer is F.")

    result = await run_video_mme_v2_group(
        cast(MindBridge, api),
        _video_mme_v2_group(),
        _prepared("001", with_media=True),
        run_id="run_01",
    )

    query = api.recall_requests[0].query.text
    assert query is not None
    assert api.remember_requests[0].tenant_id == "benchmark_video_mme_v2_001_run_01"
    assert "Respond with only the letter (A, B, C, D, E, F, G, or H)" in query
    assert [question.response for question in result.questions] == ["F", "F", "F", "F"]
    # Four of four right is the only outcome worth full marks under either group type.
    assert evaluate_video_mme_v2((result,)).rating.overall == pytest.approx(100.0)


async def test_video_mme_v2_scores_zero_for_a_group_it_answered_consistently_wrong() -> None:
    """The rating, unlike an accuracy, gives nothing back for a uniformly wrong group."""
    api = RecordingMemoryApi("The best answer is A.")

    result = await run_video_mme_v2_group(
        cast(MindBridge, api),
        _video_mme_v2_group(),
        _prepared("001", with_media=True),
        run_id="run_01",
    )

    metrics = evaluate_video_mme_v2((result,))
    assert metrics.rating.overall == pytest.approx(0.0)
    assert metrics.accuracy.overall == pytest.approx(0.0)
    assert metrics.accuracy.answered_count == 4


async def test_video_mme_v2_carries_one_ingest_loss_onto_all_four_questions() -> None:
    """A dropped segment can cost a whole group its rating, not one question its point."""
    api = SegmentFailingMemoryApi("The best answer is F.")

    result = await run_video_mme_v2_group(
        cast(MindBridge, api),
        _video_mme_v2_group(),
        _prepared_pair("001"),
        run_id="run_01",
    )

    assert [request.sequence for request in api.observe_requests] == [1]
    assert [question.mindbridge_ingest_failure_count for question in result.questions] == [
        1,
        1,
        1,
        1,
    ]


async def test_video_mme_v2_refuses_prepared_media_for_a_different_video() -> None:
    api = RecordingMemoryApi("The best answer is F.")

    with pytest.raises(ValueError, match="IDs must match"):
        await run_video_mme_v2_group(
            cast(MindBridge, api),
            _video_mme_v2_group(),
            _prepared("002", with_media=True),
            run_id="run_01",
        )
