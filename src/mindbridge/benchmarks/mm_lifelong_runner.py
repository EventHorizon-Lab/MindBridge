"""Run MM-Lifelong through public MindBridge observation and recall contracts."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

from pydantic import AwareDatetime, Field, model_validator

from mindbridge.benchmarks.mm_lifelong import (
    MM_LIFELONG_ADAPTER_VERSION,
    MMLifelongQuestion,
    MMLifelongSplit,
)
from mindbridge.benchmarks.runtime import (
    benchmark_tenant_id,
    ingest_media,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    MemoryView,
    NonEmptyString,
    ObserveRequest,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge


class MMLifelongPreparedSegment(ContractModel):
    """One globally aligned chunk of a Day, Week, or Month timeline."""

    segment_id: Identifier
    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: int = Field(gt=0)
    media_objects: tuple[MediaObjectInput, ...] = ()
    caption: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_aligned_content(self) -> MMLifelongPreparedSegment:
        if not self.media_objects and self.caption is None:
            raise ValueError("MM-Lifelong segments require media or a caption")
        media_ids = tuple(item.media_object_id for item in self.media_objects)
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("MM-Lifelong segment media_object_ids must be unique")
        if any(
            item.kind in {MediaKind.VIDEO, MediaKind.AUDIO}
            and (item.duration_ms is None or not 0 < item.duration_ms <= self.duration_ms)
            for item in self.media_objects
        ):
            raise ValueError("MM-Lifelong timed media must fit its prepared segment")
        return self


class MMLifelongPreparedTimeline(ContractModel):
    """Prepared media aligned to the official split-wide total_intervals clock."""

    split: MMLifelongSplit
    timeline_origin: AwareDatetime
    segments: tuple[MMLifelongPreparedSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_non_overlapping_segments(self) -> MMLifelongPreparedTimeline:
        starts = tuple(segment.start_seconds for segment in self.segments)
        if starts != tuple(sorted(starts)) or len(set(starts)) != len(starts):
            raise ValueError("MM-Lifelong segments must have unique chronological starts")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        media_ids = tuple(
            media.media_object_id for segment in self.segments for media in segment.media_objects
        )
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("MM-Lifelong segment IDs must be unique")
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("MM-Lifelong media_object_ids must be globally unique")
        previous_end = 0.0
        for segment in self.segments:
            if segment.start_seconds < previous_end:
                raise ValueError("MM-Lifelong prepared segments must not overlap")
            previous_end = segment.start_seconds + segment.duration_ms / 1_000
        return self


class MMLifelongPrediction(ContractModel):
    """Official evaluator candidate answer plus retrieved global intervals."""

    answer: str
    intervals: tuple[tuple[float, float], ...]


class MMLifelongQuestionResult(ContractModel):
    """One official prediction row plus deterministic retrieval diagnostics."""

    index: int = Field(ge=0)
    question: NonEmptyString
    answer: NonEmptyString
    question_type: NonEmptyString
    temporal_certificate: NonEmptyString
    total_intervals: tuple[tuple[float, float], ...] = Field(min_length=1)
    pred: MMLifelongPrediction
    mindbridge_unofficial_ref_at_300: float = Field(ge=0.0, le=1.0)
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier


def load_prepared_mm_lifelong(path: Path) -> MMLifelongPreparedTimeline:
    """Load already uploaded timeline metadata without adding media tooling."""
    return MMLifelongPreparedTimeline.model_validate_json(path.read_bytes())


async def run_mm_lifelong(
    memory: MindBridge,
    questions: tuple[MMLifelongQuestion, ...],
    prepared: MMLifelongPreparedTimeline,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_mm_lifelong",
    device_id: str = "mm_lifelong_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[MMLifelongQuestionResult, ...]:
    """Ingest one complete official timeline, then answer and localize each question."""
    if not questions:
        raise ValueError("MM-Lifelong questions must not be empty")
    if {question.split for question in questions} != {prepared.split}:
        raise ValueError("MM-Lifelong questions and prepared timeline split must match")
    if len({question.index for question in questions}) != len(questions):
        raise ValueError("MM-Lifelong questions must have unique indices")
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    total_seconds = _timeline_end_seconds(prepared)
    if any(
        end > total_seconds for question in questions for _, end in question.reference_intervals
    ):
        raise ValueError("MM-Lifelong prepared timeline does not cover reference intervals")

    tenant_id = benchmark_tenant_id(tenant_prefix, prepared.split, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)
    for offset in range(0, len(prepared.segments), request_concurrency):
        await asyncio.gather(
            *(
                _ingest_segment(
                    memory,
                    tenant_id,
                    device_id,
                    prepared,
                    segment,
                    offset + index,
                    poll_interval_seconds,
                    processing_timeout_seconds,
                    semaphore,
                )
                for index, segment in enumerate(
                    prepared.segments[offset : offset + request_concurrency]
                )
            )
        )

    results = []
    for offset in range(0, len(questions), request_concurrency):
        results.extend(
            await asyncio.gather(
                *(
                    _answer_question(
                        memory,
                        tenant_id,
                        question,
                        prepared,
                        total_seconds,
                        recall_limit,
                        semaphore,
                    )
                    for question in questions[offset : offset + request_concurrency]
                )
            )
        )
    return tuple(results)


async def _ingest_segment(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    prepared: MMLifelongPreparedTimeline,
    segment: MMLifelongPreparedSegment,
    sequence: int,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        occurred_at = prepared.timeline_origin + timedelta(seconds=segment.start_seconds)
        ended_at = occurred_at + timedelta(milliseconds=segment.duration_ms)
        evidence_ids: tuple[str, ...] = ()
        if segment.media_objects:
            evidence_ids = await ingest_media(
                memory,
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=MM_LIFELONG_ADAPTER_VERSION,
                    sequence=sequence,
                    sensor=(
                        SensorKind.MICROPHONE
                        if all(item.kind is MediaKind.AUDIO for item in segment.media_objects)
                        else SensorKind.CAMERA
                    ),
                    media_objects=segment.media_objects,
                    occurred_at=occurred_at,
                    ended_at=ended_at,
                    observed_at=ended_at,
                    idempotency_key=(
                        f"{MM_LIFELONG_ADAPTER_VERSION}:{prepared.split}:{segment.segment_id}:media"
                    ),
                ),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
        if segment.caption is not None:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=segment.caption,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=occurred_at,
                    ended_at=ended_at,
                    evidence_ids=evidence_ids,
                    idempotency_key=(
                        f"{MM_LIFELONG_ADAPTER_VERSION}:{prepared.split}:"
                        f"{segment.segment_id}:caption"
                    ),
                )
            )


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: MMLifelongQuestion,
    prepared: MMLifelongPreparedTimeline,
    total_seconds: float,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> MMLifelongQuestionResult:
    async with semaphore:
        recalled = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(text=question.question),
                limit=recall_limit,
            )
        )
    intervals = _memory_intervals(recalled.memories, prepared, total_seconds)
    return MMLifelongQuestionResult(
        index=question.index,
        question=question.question,
        answer=question.reference_answer,
        question_type=question.question_type,
        temporal_certificate=question.temporal_certificate,
        total_intervals=question.reference_intervals,
        pred=MMLifelongPrediction(answer=recalled.answer or "", intervals=intervals),
        mindbridge_unofficial_ref_at_300=unofficial_reference_at_n(
            question.reference_intervals, intervals, total_seconds
        ),
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_evidence_ids=tuple(item.evidence_id for item in recalled.evidence),
        mindbridge_trace_id=recalled.trace_id,
    )


def _memory_intervals(
    memories: tuple[MemoryView, ...],
    prepared: MMLifelongPreparedTimeline,
    total_seconds: float,
) -> tuple[tuple[float, float], ...]:
    intervals = set()
    for memory in memories:
        start = max(0.0, (memory.occurred_at - prepared.timeline_origin).total_seconds())
        end = min(total_seconds, (memory.ended_at - prepared.timeline_origin).total_seconds())
        if end > start:
            intervals.add((start, end))
    return tuple(sorted(intervals))


def unofficial_reference_at_n(
    reference: tuple[tuple[float, float], ...],
    prediction: tuple[tuple[float, float], ...],
    total_seconds: float,
    bucket_size: float = 300.0,
) -> float:
    """Estimate interval-localization quality as a bucketed Jaccard overlap.

    This is an in-repo diagnostic, not the official Ref@N. Bucket edges and rounding are not
    verified against the released scorer, so a published number must come from running that
    scorer over the emitted `pred` rows.
    """
    if total_seconds <= 0 or bucket_size <= 0:
        raise ValueError("MM-Lifelong total duration and bucket size must be positive")

    def buckets(intervals: tuple[tuple[float, float], ...]) -> set[int]:
        result: set[int] = set()
        for start, end in intervals:
            start = max(0.0, start)
            end = min(total_seconds, end)
            if start >= end:
                continue
            first = int(start // bucket_size)
            last = int((end - 1e-9) // bucket_size)
            result.update(range(first, last + 1))
        return result

    reference_buckets = buckets(reference)
    prediction_buckets = buckets(prediction)
    union = reference_buckets | prediction_buckets
    return len(reference_buckets & prediction_buckets) / len(union) if union else 0.0


def _timeline_end_seconds(prepared: MMLifelongPreparedTimeline) -> float:
    last = prepared.segments[-1]
    return last.start_seconds + last.duration_ms / 1_000
