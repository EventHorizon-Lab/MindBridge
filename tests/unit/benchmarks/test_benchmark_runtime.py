"""Checks for shared production benchmark runtime behavior."""

import asyncio
from datetime import datetime, timezone
from typing import cast

import pytest

from mindbridge import MindBridge
from mindbridge.benchmarks.runtime import (
    PreparedVideo,
    PreparedVideoSegment,
    answer_failure_trace_id,
    benchmark_tenant_id,
    ingest_prepared_video,
    multiple_choice_query,
    parse_option_ranking,
    settle_answers,
)
from mindbridge.contracts import (
    MediaObjectInput,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
)
from mindbridge.core import JobState, MediaKind
from mindbridge.sdk import MindBridgeError

_ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_benchmark_tenant_requires_an_isolated_run_id() -> None:
    assert benchmark_tenant_id("benchmark", "subject_1", "run_01") == ("benchmark_subject_1_run_01")
    with pytest.raises(ValueError, match="run_id"):
        benchmark_tenant_id("benchmark", "subject_1", " ")


def test_multiple_choice_query_contains_only_inference_inputs() -> None:
    query = multiple_choice_query(
        "Where is the mug?",
        ("On the table", "In the sink", "In a drawer", "Unknown"),
        rank_all=True,
    )

    assert "A. On the table" in query
    assert query.endswith("best to worst, separated by commas.")


def test_option_parser_accepts_constrained_outputs_and_rejects_prose() -> None:
    choices = ("On the table", "In the sink", "In a drawer", "Unknown")

    assert parse_option_ranking("Ranking: B, A, D, C", choices) == (1, 0, 3, 2)
    assert parse_option_ranking("In the sink", choices) == (1,)
    assert parse_option_ranking("B because the memory says so", choices) == ()
    assert parse_option_ranking("B, B, A, C", choices) == ()


def test_multiple_choice_runtime_supports_egomem_reason_a_to_j() -> None:
    choices = tuple(f"Choice {index}" for index in range(10))

    query = multiple_choice_query("Pick one", choices, rank_all=False)

    assert "J. Choice 9" in query
    assert parse_option_ranking("J", choices) == (9,)


def test_settle_answers_replaces_only_the_question_that_failed() -> None:
    outcomes = [
        "answered q0",
        MindBridgeError("gateway timeout", code="model_unavailable"),
        RuntimeError("connection dropped"),
    ]

    settled = settle_answers(
        ("q0", "q1", "q2"),
        outcomes,
        lambda question, code: f"{question} failed with {code}",
    )

    assert settled == (
        "answered q0",
        "q1 failed with model_unavailable",
        "q2 failed with RuntimeError",
    )


def test_settle_answers_refuses_to_score_a_cancelled_run() -> None:
    """A cancelled cohort is a run that ended, not a cohort of wrong answers."""
    with pytest.raises(asyncio.CancelledError):
        settle_answers(("q0",), [asyncio.CancelledError()], lambda question, code: "recorded")


def test_settle_answers_rejects_outcomes_that_do_not_line_up_with_their_questions() -> None:
    with pytest.raises(ValueError, match="shorter"):
        settle_answers(("q0", "q1"), ["answered q0"], lambda question, code: "recorded")


def test_answer_failure_trace_id_stays_inside_the_identifier_limit() -> None:
    assert len(answer_failure_trace_id("q" * 400)) == 255


class _SlowFirstSegmentApi:
    """Answer every job immediately except the first, and log the order things happened in.

    The first segment's job stays RUNNING for a fixed number of polls. That is what separates a
    fan-out that streams from one that stops at a cohort boundary: under a bounded stream the
    permits its fast siblings release are taken by segments further down the video, so a later
    segment is observed while the slow one is still in flight. Under a cohort barrier no segment
    outside the first cohort can be observed until every member of it, the slow one included,
    has finished.
    """

    def __init__(self, *, slow_polls: int) -> None:
        self.slow_polls = slow_polls
        self.log: list[str] = []
        self._polls: dict[str, int] = {}
        self._observed = 0

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        self._observed += 1
        job_id = f"job_{self._observed:04d}"
        self.log.append(f"observe {job_id}")
        return ObservationReceipt(
            observation_id=f"obs_{self._observed:04d}",
            processing_job_id=job_id,
            idempotency_key=request.idempotency_key or "key_1",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def get_observation_job(
        self, tenant_id: str, job_id: str
    ) -> ObservationProcessingJobView:
        self._polls[job_id] = self._polls.get(job_id, 0) + 1
        running = job_id == "job_0001" and self._polls[job_id] <= self.slow_polls
        if not running:
            self.log.append(f"done {job_id}")
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="obs_0001",
            state=JobState.RUNNING if running else JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            created_at=_ORIGIN,
            updated_at=_ORIGIN,
            trace_id="trace_job",
        )


def _prepared_video(segments: int) -> PreparedVideo:
    return PreparedVideo(
        video_id="video_1",
        timeline_origin=_ORIGIN,
        segments=tuple(
            PreparedVideoSegment(
                segment_id=f"segment_{index:04d}",
                start_seconds=float(index * 30),
                duration_ms=30_000,
                media_objects=(
                    MediaObjectInput(
                        media_object_id=f"object_{index:04d}",
                        kind=MediaKind.VIDEO,
                        uri=f"s3://bucket/object_{index:04d}.mp4",
                        sha256=f"{index:064x}",
                        size_bytes=1_024,
                        created_at=_ORIGIN,
                        duration_ms=30_000,
                    ),
                ),
            )
            for index in range(segments)
        ),
    )


async def test_prepared_ingest_keeps_its_permits_busy_past_the_first_cohort() -> None:
    """One slow segment must not stall the ones a permit is free for.

    The fan-out is bounded by `request_concurrency` permits and nothing else. Chunking the
    segments into cohorts of that size instead drains the permits to zero at every cohort
    boundary, so a video's in-flight count sawtooths between the ceiling and one -- against a
    Worker whose queue only stays full while the permits do.
    """
    concurrency = 3
    api = _SlowFirstSegmentApi(slow_polls=8)

    failures = await ingest_prepared_video(
        cast(MindBridge, api),
        "tenant_1",
        "device_1",
        _prepared_video(12),
        adapter_version="adapter_v1",
        request_concurrency=concurrency,
        poll_interval_seconds=0.001,
        processing_timeout_seconds=30.0,
    )

    assert failures == 0
    observed_before_the_slow_one_finished = api.log.index("done job_0001")
    started_before_then = sum(
        entry.startswith("observe") for entry in api.log[:observed_before_the_slow_one_finished]
    )
    assert started_before_then > concurrency, (
        f"only {started_before_then} segments were in flight before the slow one finished; "
        f"a stream bounded by {concurrency} permits reaches further than one cohort"
    )
