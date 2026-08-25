"""Production-contract checks for the per-topic Mem-Gallery runner."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.mem_gallery import (
    MemGalleryProfile,
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
    MemGalleryTopic,
)
from mindbridge.benchmarks.mem_gallery_runner import (
    MemGalleryPreparedImage,
    MemGalleryPreparedImages,
    run_mem_gallery_topic,
    validate_mem_gallery_images,
)
from mindbridge.contracts import (
    EvidenceView,
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
    JobState,
    MediaKind,
    MemoryState,
    MemoryType,
    VerificationStatus,
)
from mindbridge.sdk import MindBridgeError

NOW = datetime(2024, 6, 24, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self) -> None:
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self.observe_requests.append(request)
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_01",
            evidence_ids=("evidence_01",),
            idempotency_key=request.idempotency_key or "generated",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_01",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )

    async def remember(self, request: RememberRequest) -> object:
        self.remember_requests.append(request)
        return object()

    async def recall(self, request: RecallRequest) -> RecallResult:
        self.recall_requests.append(request)
        return RecallResult(
            answer="30 litres or more.",
            confidence=0.7,
            memories=(
                MemoryView(
                    memory_id="memory_01",
                    memory_type=MemoryType.EPISODIC,
                    summary="D1:1 User said: Can you tell me the basics?",
                    evidence_ids=(),
                    occurred_at=NOW,
                    ended_at=NOW,
                    created_at=NOW,
                    verification_status=VerificationStatus.ATTESTED,
                    state=MemoryState.ACTIVE,
                ),
            ),
            evidence=(
                EvidenceView(
                    evidence_id="evidence_01",
                    media_object_id="D1:IMG_001",
                    start_ms=0,
                    end_ms=0,
                    media_url="https://example.invalid/signed",
                    media_url_expires_at=NOW + timedelta(minutes=5),
                ),
            ),
            trace_id="trace_recall",
        )


async def test_mem_gallery_registers_a_question_image_before_querying_with_it() -> None:
    """`RecallQuery` resolves media ids against the tenant, so an unobserved one aborts.

    Nothing else registers these: the release keeps question images in their own `QA_IMG_*`
    files and, measured on the pinned corpus, all 487 image-bearing questions name one that no
    dialogue round uses. Before this every one of them raised
    `recall query references unknown media` -- after ingesting the whole persona.
    """

    class UnknownMediaMemoryApi(RecordingMemoryApi):
        """Refuses a query image it was never handed, exactly as `_resolve_query_media` does."""

        async def recall(self, request: RecallRequest) -> RecallResult:
            registered = {
                media.media_object_id
                for observed in self.observe_requests
                for media in observed.media_objects
            }
            unknown = set(request.query.media_object_ids) - registered
            if unknown:
                raise MindBridgeError(
                    "recall query references unknown media", code="domain_invariant_failed"
                )
            return await super().recall(request)

    api = UnknownMediaMemoryApi()

    results = await run_mem_gallery_topic(
        cast(MindBridge, api), _topic(), run_id="run_01", prepared=_prepared()
    )

    assert [result.question_id for result in results] == ["Baking:1", "Baking:2"]
    assert any(
        "QA_IMG_001" in media.uri
        for observed in api.observe_requests
        for media in observed.media_objects
    ), "the question image was never registered with the tenant"
    # Registered once, not once per question that references it.
    assert (
        sum(
            "question-image" in (observed.idempotency_key or "")
            for observed in api.observe_requests
        )
        == 1
    )


def _topic() -> MemGalleryTopic:
    return MemGalleryTopic(
        topic="Baking",
        profile=MemGalleryProfile(
            name="Maya",
            persona_summary="A librarian who bakes.",
            traits=("curious",),
            conversation_style="Earnest.",
        ),
        sessions=(
            MemGallerySession(
                session_id="D1",
                occurred_at=NOW,
                rounds=(
                    MemGalleryRound(
                        round_id="D1:1",
                        user="Can you tell me the basics?",
                        assistant="Start with a 30 litre oven.",
                    ),
                    MemGalleryRound(
                        round_id="D1:2",
                        user="What is in this picture?",
                        assistant="A tray of shortbread.",
                        image_id="D1:IMG_001",
                        image_path="../image/Baking/D1_IMG_001.jpg",
                        image_caption="Pale shortbread fingers.",
                    ),
                ),
            ),
        ),
        questions=(
            MemGalleryQuestion(
                question_id="Baking:1",
                point="FR",
                question="What oven size was recommended?",
                reference_answer="30 litres or more.",
                session_ids=("D1",),
                clue_round_ids=("D1:1",),
            ),
            MemGalleryQuestion(
                question_id="Baking:2",
                point="VS",
                question="Which image shows shortbread?",
                reference_answer="D1:IMG_001",
                session_ids=("D1",),
                clue_round_ids=("D1:2",),
                question_image_path="../image/Baking/QA_IMG_001.jpg",
                question_image_caption="A tray of biscuits.",
            ),
        ),
    )


def _prepared() -> MemGalleryPreparedImages:
    return MemGalleryPreparedImages(
        images=(
            MemGalleryPreparedImage(
                image_key="../image/Baking/D1_IMG_001.jpg",
                media_object=MediaObjectInput(
                    media_object_id="D1:IMG_001",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/mem-gallery/Baking/D1_IMG_001.jpg",
                    sha256="b" * 64,
                    size_bytes=52_144,
                    created_at=NOW,
                ),
            ),
            MemGalleryPreparedImage(
                image_key="../image/Baking/QA_IMG_001.jpg",
                media_object=MediaObjectInput(
                    media_object_id="Baking:QA_IMG_001",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/mem-gallery/Baking/QA_IMG_001.jpg",
                    sha256="c" * 64,
                    size_bytes=41_002,
                    created_at=NOW,
                ),
            ),
        )
    )


async def test_rounds_are_written_per_speaker_and_images_observed_with_their_round_text() -> None:
    api = RecordingMemoryApi()

    results = await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    # One tenant for the whole topic, not one per question.
    assert {request.tenant_id for request in api.recall_requests} == {
        request.tenant_id for request in api.remember_requests
    }
    # The image round is observed with its official image_id as the media object ID.
    assert [item.media_object_id for item in api.observe_requests[0].media_objects] == [
        "D1:IMG_001"
    ]
    # Two rounds, two speakers each, and the image round's text is written too.
    assert len(api.remember_requests) == 4
    assert all(request.summary.startswith(("D1:1 ", "D1:2 ")) for request in api.remember_requests)
    assert len(results) == 2
    # The image's caption is folded into the User turn that introduced it, and only that turn.
    image_round_user = next(
        request.summary
        for request in api.remember_requests
        if request.summary.startswith("D1:2 User")
    )
    image_round_assistant = next(
        request.summary
        for request in api.remember_requests
        if request.summary.startswith("D1:2 Assistant")
    )
    assert "Pale shortbread fingers." in image_round_user
    assert "Pale shortbread fingers." not in image_round_assistant


async def test_a_question_image_is_sent_as_a_recall_query_object() -> None:
    api = RecordingMemoryApi()

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert api.recall_requests[0].query.media_object_ids == ()
    assert api.recall_requests[1].query.media_object_ids == ("Baking:QA_IMG_001",)


async def test_official_constraints_are_applied_only_to_ar_cd_and_vs() -> None:
    api = RecordingMemoryApi()

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    factual_query = api.recall_requests[0].query.text or ""
    search_query = api.recall_requests[1].query.text or ""
    assert "Return the image_id" not in factual_query
    assert "Return the image_id" in search_query
    # The official wording names the speakers, and the constraint arrives as its own
    # paragraph rather than trailing the question on one line.
    assert "between user (Maya) and assistant" in factual_query
    assert search_query.endswith(
        "\n\nReturn the image_id of the image(s). If there are "
        "multiple images, sort them in ascending order and separate "
        "them by commas. Format example: “D2:IMG_003, "
        "D2:IMG_010, D10:IMG_002” (for format reference only)."
    )


async def test_clue_recall_counts_rounds_the_recall_actually_returned() -> None:
    api = RecordingMemoryApi()

    results = await run_mem_gallery_topic(
        cast(MindBridge, api),
        _topic(),
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert results[0].mindbridge_round_ids == ("D1:1",)
    assert results[0].retrieved_clue_round_count == 1
    assert results[1].retrieved_clue_round_count == 0
    assert results[0].mindbridge_confidence == pytest.approx(0.7)


def test_a_run_refuses_to_start_without_every_referenced_image() -> None:
    prepared = MemGalleryPreparedImages(images=_prepared().images[:1])

    with pytest.raises(ValueError, match="missing prepared Mem-Gallery images"):
        validate_mem_gallery_images((_topic(),), prepared)


def _staged_image(
    image_key: str, media_object_id: str, sha256_byte: str
) -> MemGalleryPreparedImage:
    return MemGalleryPreparedImage(
        image_key=image_key,
        media_object=MediaObjectInput(
            media_object_id=media_object_id,
            kind=MediaKind.IMAGE,
            uri=f"s3://mindbridge-media/mem-gallery/{image_key}",
            sha256=sha256_byte * 64,
            size_bytes=1,
            created_at=NOW,
        ),
    )


def test_prepared_images_allow_a_media_object_id_shared_across_two_topics() -> None:
    """The official `image_id` is release-relative, not archive-unique: two topics staged
    into one manifest legitimately share it. Measured on the pinned release, `D1:IMG_001`
    alone names a different picture in all twenty topics, and 127 of 182 distinct `image_id`
    values are shared across more than one. `image_key` -- the release-relative path -- is
    what actually disambiguates the two topics' pictures, and stays required to be unique.
    """
    prepared = MemGalleryPreparedImages(
        images=(
            _staged_image("Baking/D1_IMG_001.jpg", "D1:IMG_001", "b"),
            _staged_image("Gardening/D1_IMG_001.jpg", "D1:IMG_001", "c"),
        )
    )

    assert len(prepared.images) == 2


def test_prepared_images_still_refuse_a_duplicate_image_key() -> None:
    with pytest.raises(ValueError, match="prepared image keys must be unique"):
        MemGalleryPreparedImages(
            images=(
                _staged_image("Baking/D1_IMG_001.jpg", "D1:IMG_001", "b"),
                _staged_image("Baking/D1_IMG_001.jpg", "Baking:IMG_002", "c"),
            )
        )


async def test_sessions_overlap_in_ingest_while_rounds_inside_one_stay_serial() -> None:
    """Sessions carry distinct occurred_at so they may overlap; rounds inside one share it.

    Mirrors `mindbridge.benchmarks.memlens_runner`'s identical session/turn split and its own
    `test_memlens_overlaps_sessions_while_keeping_each_session_ordered`. A draft that awaits
    every round at the top level leaves `request_concurrency` with nothing to bound -- this
    would still pass every other test in this file, since none of them delay a call, so this
    is the one test that fails if that wiring is removed.
    """

    class DelayedMemoryApi(RecordingMemoryApi):
        async def remember(self, request: RememberRequest) -> object:
            if request.summary.startswith("D1:1 User"):
                await asyncio.sleep(0.01)
            return await super().remember(request)

    api = DelayedMemoryApi()
    topic = _topic().model_copy(
        update={
            "sessions": tuple(
                MemGallerySession(
                    session_id=session_id,
                    occurred_at=NOW - timedelta(days=day),
                    rounds=(
                        MemGalleryRound(
                            round_id=f"{session_id}:1",
                            user=f"{session_id} question",
                            assistant=f"{session_id} answer",
                        ),
                    ),
                )
                for day, session_id in ((2, "D1"), (1, "D2"))
            ),
            "questions": (
                MemGalleryQuestion(
                    question_id="Baking:1",
                    point="FR",
                    question="Did D2 finish first?",
                    reference_answer="Yes.",
                    session_ids=("D1", "D2"),
                ),
            ),
        }
    )

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        topic,
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    remember_summaries = [request.summary for request in api.remember_requests]
    # Each session stays internally ordered even though D1's first write was delayed.
    for session_id in ("D1", "D2"):
        ordered = [summary for summary in remember_summaries if summary.startswith(session_id)]
        assert ordered == [
            f"{session_id}:1 User said: {session_id} question",
            f"{session_id}:1 Assistant said: {session_id} answer",
        ]
    # D2 did not wait behind the delayed D1 round, which is the point of batching sessions.
    assert remember_summaries[0] == "D2:1 User said: D2 question"


async def test_questions_are_answered_concurrently() -> None:
    """An independent second question must not wait behind a slow first one."""

    class DelayedMemoryApi(RecordingMemoryApi):
        async def recall(self, request: RecallRequest) -> RecallResult:
            if request.query.text is not None and "first" in request.query.text:
                await asyncio.sleep(0.01)
            return await super().recall(request)

    api = DelayedMemoryApi()
    topic = _topic().model_copy(
        update={
            "questions": (
                MemGalleryQuestion(
                    question_id="Baking:1",
                    point="FR",
                    question="first question?",
                    reference_answer="A1.",
                    session_ids=("D1",),
                ),
                MemGalleryQuestion(
                    question_id="Baking:2",
                    point="FR",
                    question="second question?",
                    reference_answer="A2.",
                    session_ids=("D1",),
                ),
            )
        }
    )

    await run_mem_gallery_topic(
        cast(MindBridge, api),
        topic,
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    # The delayed first question did not block the second: it lands before the first one wakes,
    # and the slow one only completes afterward -- proving the two ran concurrently rather than
    # in sequence, where "first" would always be recorded before "second".
    assert api.recall_requests[0].query.text is not None
    assert "second" in api.recall_requests[0].query.text
    assert api.recall_requests[-1].query.text is not None
    assert "first" in api.recall_requests[-1].query.text


async def test_a_failing_round_is_counted_but_other_rounds_still_land() -> None:
    """A round that cannot be written used to discard every round written beside it."""

    class FailingMemoryApi(RecordingMemoryApi):
        async def remember(self, request: RememberRequest) -> object:
            if request.summary.startswith("sess_b:1"):
                raise MindBridgeError(
                    "round could not be written",
                    code="model_request_failed",
                    status_code=502,
                    trace_id="trace_ingest_error",
                )
            return await super().remember(request)

    api = FailingMemoryApi()
    topic = _topic().model_copy(
        update={
            "sessions": (
                MemGallerySession(
                    session_id="sess_a",
                    occurred_at=NOW - timedelta(days=3),
                    rounds=(
                        MemGalleryRound(
                            round_id="sess_a:1",
                            user="sess_a question 1",
                            assistant="sess_a answer 1",
                        ),
                    ),
                ),
                MemGallerySession(
                    session_id="sess_b",
                    occurred_at=NOW - timedelta(days=2),
                    rounds=(
                        # This round fails; the next one in the SAME session must still land --
                        # the guard is per round, not per session.
                        MemGalleryRound(
                            round_id="sess_b:1",
                            user="sess_b question 1",
                            assistant="sess_b answer 1",
                        ),
                        MemGalleryRound(
                            round_id="sess_b:2",
                            user="sess_b question 2",
                            assistant="sess_b answer 2",
                        ),
                    ),
                ),
                MemGallerySession(
                    session_id="sess_c",
                    occurred_at=NOW - timedelta(days=1),
                    rounds=(
                        MemGalleryRound(
                            round_id="sess_c:1",
                            user="sess_c question 1",
                            assistant="sess_c answer 1",
                        ),
                    ),
                ),
            ),
            "questions": (
                MemGalleryQuestion(
                    question_id="Baking:1",
                    point="FR",
                    question="Did the other rounds still land?",
                    reference_answer="Yes.",
                    session_ids=("sess_a", "sess_b", "sess_c"),
                ),
            ),
        }
    )

    results = await run_mem_gallery_topic(
        cast(MindBridge, api),
        topic,
        run_id="run1",
        prepared=_prepared(),
        tenant_prefix="benchmark_mem_gallery",
        device_id="mem_gallery_conversation",
        recall_limit=20,
        request_concurrency=3,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    # The one failing round is counted...
    assert results[0].mindbridge_ingest_failure_count == 1
    # ...but every other round still landed, including the one right after it in sess_b, and
    # neither of the failing round's own two speaker turns (it raised on the first) leaked through.
    assert sorted(request.summary for request in api.remember_requests) == sorted(
        [
            "sess_a:1 User said: sess_a question 1",
            "sess_a:1 Assistant said: sess_a answer 1",
            "sess_b:2 User said: sess_b question 2",
            "sess_b:2 Assistant said: sess_b answer 2",
            "sess_c:1 User said: sess_c question 1",
            "sess_c:1 Assistant said: sess_c answer 1",
        ]
    )
