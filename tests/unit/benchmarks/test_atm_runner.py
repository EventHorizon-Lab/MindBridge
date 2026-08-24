"""Production-contract checks for the single-archive ATM-Bench runner."""

from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.atm_bench import (
    AtmBenchQuestion,
    AtmEmail,
    AtmSgmRecord,
)
from mindbridge.benchmarks.atm_bench_runner import (
    AtmPreparedArchive,
    AtmPreparedMedia,
    answer_atm_question,
    ingest_atm_archive,
    validate_prepared_atm,
)
from mindbridge.contracts import (
    EvidenceView,
    MediaObjectInput,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallMode,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import JobState, MediaKind, MemoryState, MemoryType, VerificationStatus

NOW = datetime(2025, 2, 23, 13, 2, 49, tzinfo=timezone.utc)


class RecordingMemoryApi:
    def __init__(self, *, evidence: tuple[str, ...] = ()) -> None:
        self.observe_requests: list[ObserveRequest] = []
        self.remember_requests: list[RememberRequest] = []
        self.recall_requests: list[RecallRequest] = []
        self._evidence = evidence

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
            answer="£799.74",
            confidence=0.82,
            memories=(
                MemoryView(
                    memory_id="memory_01",
                    memory_type=MemoryType.EPISODIC,
                    summary="ID: email202411160004",
                    evidence_ids=(),
                    occurred_at=NOW,
                    ended_at=NOW,
                    created_at=NOW,
                    verification_status=VerificationStatus.ATTESTED,
                    state=MemoryState.ACTIVE,
                ),
            ),
            evidence=tuple(
                EvidenceView(
                    evidence_id=f"evidence_{index}",
                    media_object_id=media_object_id,
                    start_ms=0,
                    end_ms=0,
                    media_url="https://example.invalid/signed",
                    media_url_expires_at=NOW + timedelta(minutes=5),
                )
                for index, media_object_id in enumerate(self._evidence, start=1)
            ),
            trace_id="trace_recall",
        )


def _question(**overrides: object) -> AtmBenchQuestion:
    fields: dict[str, object] = {
        "question_id": "question_01",
        "question": "How much did I pay for accommodation?",
        "reference_answer": "£799.74",
        "qtype": "number",
        "evidence_ids": ("email202411160004", "20250223_130249"),
    }
    return AtmBenchQuestion.model_validate(fields | overrides)


def _prepared() -> AtmPreparedArchive:
    return AtmPreparedArchive(
        media=(
            AtmPreparedMedia(
                media_id="20250223_130249",
                media_object=MediaObjectInput(
                    media_object_id="20250223_130249",
                    kind=MediaKind.IMAGE,
                    uri="s3://mindbridge-media/atm-bench/20250223_130249.jpg",
                    sha256="a" * 64,
                    size_bytes=100_686,
                    created_at=NOW,
                ),
            ),
        )
    )


def _sgm_record() -> AtmSgmRecord:
    return AtmSgmRecord(
        media_id="20250223_130249",
        media_kind=MediaKind.IMAGE,
        occurred_at=NOW,
        raw_timestamp="2025-02-23 13:02:49",
        location_name="Porto, Portugal",
        city="Porto, Portugal",
        short_caption="A steel bridge over a river.",
        caption="A wide steel arch bridge spans the Douro.",
        ocr_text="",
        tags=("bridge", "porto"),
        size_bytes=100_686,
    )


def _email() -> AtmEmail:
    return AtmEmail(
        email_id="email202411160004",
        occurred_at=NOW,
        summary="Hotel confirmation",
        body="Total £799.74 for four nights.",
    )


async def test_raw_arm_observes_one_media_object_per_observation_named_by_its_stem() -> None:
    api = RecordingMemoryApi()

    failures = await ingest_atm_archive(
        cast(MindBridge, api),
        tenant_id="benchmark_atm_archive_run1",
        device_id="atm_archive",
        media_source="raw",
        prepared=_prepared(),
        sgm_records=(_sgm_record(),),
        emails=(_email(),),
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert failures == 0
    assert len(api.observe_requests) == 1
    request = api.observe_requests[0]
    assert [item.media_object_id for item in request.media_objects] == ["20250223_130249"]
    assert request.occurred_at == NOW
    # Emails are written in both arms; the raw arm writes no SGM text.
    summaries = [item.summary for item in api.remember_requests]
    assert any(summary.startswith("ID: email202411160004") for summary in summaries)
    assert not any(summary.startswith("ID: 20250223_130249") for summary in summaries)


async def test_sgm_arm_writes_official_blocks_and_observes_nothing() -> None:
    api = RecordingMemoryApi()

    failures = await ingest_atm_archive(
        cast(MindBridge, api),
        tenant_id="benchmark_atm_archive_run1",
        device_id="atm_archive",
        media_source="sgm",
        prepared=None,
        sgm_records=(_sgm_record(),),
        emails=(_email(),),
        request_concurrency=2,
        poll_interval_seconds=0.01,
        processing_timeout_seconds=1.0,
    )

    assert failures == 0
    assert api.observe_requests == []
    summaries = [item.summary for item in api.remember_requests]
    assert any(summary.startswith("ID: 20250223_130249\nType: image") for summary in summaries)
    assert any(summary.startswith("ID: email202411160004") for summary in summaries)


async def test_list_recall_questions_enumerate_and_others_answer() -> None:
    api = RecordingMemoryApi()

    await answer_atm_question(
        cast(MindBridge, api),
        _question(qtype="list_recall"),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )
    await answer_atm_question(
        cast(MindBridge, api),
        _question(qtype="open_end"),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )

    assert api.recall_requests[0].mode is RecallMode.ENUMERATE
    assert api.recall_requests[1].mode is RecallMode.ANSWER


async def test_retrieval_recall_counts_only_gold_evidence_the_recall_returned() -> None:
    api = RecordingMemoryApi(evidence=("20250223_130249", "20220430_132212"))

    result = await answer_atm_question(
        cast(MindBridge, api),
        _question(),
        tenant_id="benchmark_atm_archive_run1",
        recall_limit=20,
    )

    assert result.prediction == "£799.74"
    assert result.mindbridge_media_object_ids == ("20250223_130249", "20220430_132212")
    # One of the two gold evidence items came back; the distractor does not count.
    assert result.retrieved_gold_evidence_count == 1
    assert result.mindbridge_confidence == pytest.approx(0.82)


def test_raw_arm_refuses_to_start_without_every_cited_media_item() -> None:
    with pytest.raises(ValueError, match="missing prepared ATM-Bench media"):
        validate_prepared_atm(
            (_question(evidence_ids=("20250223_130249", "20991231_235959")),),
            _prepared(),
            media_source="raw",
        )

    # The SGM arm needs no prepared media at all.
    validate_prepared_atm((_question(),), None, media_source="sgm")
