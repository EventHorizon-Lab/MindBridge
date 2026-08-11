"""Run SuperMemory-VQA through public MindBridge ingestion and recall contracts."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from itertools import groupby, pairwise
from pathlib import Path

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from mindbridge.benchmarks.runtime import (
    multiple_choice_query,
    parse_option_ranking,
    wait_for_observation_job,
)
from mindbridge.benchmarks.supermemory_vqa import (
    SUPERMEMORY_VQA_ADAPTER_VERSION,
    SuperMemoryQuestion,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallFilters,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import AsyncMindBridge


class SuperMemoryPreparedSegment(ContractModel):
    """One addressable, time-aligned segment of an official recording."""

    start_seconds: float = Field(ge=0, allow_inf_nan=False)
    duration_ms: int = Field(gt=0)
    media_objects: tuple[MediaObjectInput, ...] = ()
    transcript: NonEmptyString | None = None

    @model_validator(mode="after")
    def require_aligned_content(self) -> SuperMemoryPreparedSegment:
        """Accept released transcript-only spans but validate every supplied medium."""
        if not self.media_objects and self.transcript is None:
            raise ValueError("SuperMemory-VQA segments require media or a transcript")
        media_ids = tuple(item.media_object_id for item in self.media_objects)
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("SuperMemory-VQA segment media_object_ids must be unique")
        if any(item.duration_ms != self.duration_ms for item in self.media_objects):
            raise ValueError("SuperMemory-VQA media duration must match its segment")
        return self


class SuperMemoryPreparedVideo(ContractModel):
    """Prepared segments and absolute start time for one official source video."""

    video_id: Identifier
    started_at: AwareDatetime
    segments: tuple[SuperMemoryPreparedSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_non_overlapping_segments(self) -> SuperMemoryPreparedVideo:
        """Keep source-local timing deterministic."""
        starts = tuple(segment.start_seconds for segment in self.segments)
        if starts != tuple(sorted(starts)) or len(set(starts)) != len(starts):
            raise ValueError("SuperMemory-VQA segments must have unique chronological starts")
        previous_end = -1.0
        for segment in self.segments:
            if segment.start_seconds < previous_end:
                raise ValueError("SuperMemory-VQA segments must not overlap")
            previous_end = segment.start_seconds + segment.duration_ms / 1_000
        return self


class SuperMemoryPreparedSubject(ContractModel):
    """Uploaded multimodal and transcript context for one benchmark participant."""

    subject: int = Field(gt=0)
    videos: tuple[SuperMemoryPreparedVideo, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_sources_and_media(self) -> SuperMemoryPreparedSubject:
        """Prevent ambiguous source or object identity across the timeline."""
        video_ids = tuple(video.video_id for video in self.videos)
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("SuperMemory-VQA prepared video IDs must be unique")
        media_ids = tuple(
            media.media_object_id
            for video in self.videos
            for segment in video.segments
            for media in segment.media_objects
        )
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("SuperMemory-VQA media_object_ids must be globally unique")
        intervals = sorted(
            (
                _segment_bounds(video, segment)
                for video in self.videos
                for segment in video.segments
            ),
            key=lambda interval: interval[0],
        )
        if any(current[0] < previous[1] for previous, current in pairwise(intervals)):
            raise ValueError("SuperMemory-VQA prepared segments must not overlap globally")
        return self


class SuperMemoryQuestionResult(ContractModel):
    """One model ranking with retrieval diagnostics and no ground-truth fields."""

    question_id: int = Field(gt=0)
    predicted_option_index: int | None = Field(default=None, ge=0, lt=4)
    ranked_option_indices: tuple[int, ...] = Field(max_length=4)
    model_answer: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier

    @model_validator(mode="after")
    def require_consistent_ranking(self) -> SuperMemoryQuestionResult:
        """Keep top-one and ranking metrics based on the same prediction."""
        if len(set(self.ranked_option_indices)) != len(self.ranked_option_indices):
            raise ValueError("ranked_option_indices must be unique")
        if any(index < 0 or index >= 4 for index in self.ranked_option_indices):
            raise ValueError("ranked_option_indices must be between zero and three")
        expected = self.ranked_option_indices[0] if self.ranked_option_indices else None
        if self.predicted_option_index != expected:
            raise ValueError("predicted_option_index must be the first ranked option")
        return self


class SuperMemoryMetrics(ContractModel):
    """The three metrics defined by the official dataset card."""

    question_count: int = Field(gt=0)
    answerability_precision: float = Field(ge=0.0, le=1.0)
    answerability_recall: float = Field(ge=0.0, le=1.0)
    answerability_f1: float = Field(ge=0.0, le=1.0)
    qa_accuracy: float = Field(ge=0.0, le=1.0)
    qa_mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)


def load_prepared_supermemory(path: Path) -> SuperMemoryPreparedSubject:
    """Load an uploaded-media manifest without adding another transfer client."""
    return SuperMemoryPreparedSubject.model_validate_json(path.read_bytes())


async def run_supermemory_vqa(
    memory: AsyncMindBridge,
    questions: tuple[SuperMemoryQuestion, ...],
    prepared: SuperMemoryPreparedSubject,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_supermemory",
    device_id: str = "supermemory_glasses",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[SuperMemoryQuestionResult, ...]:
    """Ingest completed segments and answer at each official question timestamp."""
    if not questions:
        raise ValueError("SuperMemory-VQA questions must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("SuperMemory-VQA questions must have unique IDs")
    if {question.subject for question in questions} != {prepared.subject}:
        raise ValueError("SuperMemory-VQA questions and prepared subject must match")
    if recall_limit <= 0 or request_concurrency <= 0:
        raise ValueError("recall_limit and request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    tenant_id = TypeAdapter(Identifier).validate_python(
        f"{tenant_prefix}_{prepared.subject}_{run_id}"
    )
    segments = sorted(
        (
            (*_segment_bounds(video, segment), video.video_id, segment)
            for video in prepared.videos
            for segment in video.segments
        ),
        key=lambda item: item[0],
    )
    ordered = sorted(questions, key=lambda question: question.question_at)
    answers: dict[int, SuperMemoryQuestionResult] = {}
    next_segment = 0
    semaphore = asyncio.Semaphore(request_concurrency)
    for question_at, group in groupby(ordered, key=lambda question: question.question_at):
        while next_segment < len(segments) and segments[next_segment][1] <= question_at:
            occurred_at, ended_at, video_id, segment = segments[next_segment]
            await _ingest_segment(
                memory,
                tenant_id,
                device_id,
                prepared.subject,
                video_id,
                segment,
                next_segment,
                occurred_at,
                ended_at,
                poll_interval_seconds,
                processing_timeout_seconds,
            )
            next_segment += 1
        results = await asyncio.gather(
            *(
                _answer_question(memory, tenant_id, question, recall_limit, semaphore)
                for question in group
            )
        )
        answers.update((result.question_id, result) for result in results)
    return tuple(answers[question.question_id] for question in questions)


def evaluate_supermemory_vqa(
    questions: tuple[SuperMemoryQuestion, ...],
    results: tuple[SuperMemoryQuestionResult, ...],
) -> SuperMemoryMetrics:
    """Compute Ans-F1, QA-Acc, and QA-MRR from model-only prediction rows."""
    if not questions or len(questions) != len(results):
        raise ValueError("SuperMemory-VQA questions and results must have equal non-zero length")
    by_id = {result.question_id: result for result in results}
    if len(by_id) != len(results) or set(by_id) != {question.question_id for question in questions}:
        raise ValueError("SuperMemory-VQA result IDs must match questions exactly")

    true_positive = false_positive = false_negative = correct = 0
    reciprocal_rank = 0.0
    for question in questions:
        result = by_id[question.question_id]
        predicted_answerable = (
            result.predicted_option_index is not None
            and result.predicted_option_index != question.unanswerable_option_index
        )
        true_positive += int(predicted_answerable and question.is_answerable)
        false_positive += int(predicted_answerable and not question.is_answerable)
        false_negative += int(not predicted_answerable and question.is_answerable)
        correct += int(result.predicted_option_index == question.correct_option_index)
        if question.correct_option_index in result.ranked_option_indices:
            reciprocal_rank += 1 / (
                result.ranked_option_indices.index(question.correct_option_index) + 1
            )

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return SuperMemoryMetrics(
        question_count=len(questions),
        answerability_precision=precision,
        answerability_recall=recall,
        answerability_f1=_ratio(2 * precision * recall, precision + recall),
        qa_accuracy=correct / len(questions),
        qa_mean_reciprocal_rank=reciprocal_rank / len(questions),
    )


async def _ingest_segment(
    memory: AsyncMindBridge,
    tenant_id: str,
    device_id: str,
    subject: int,
    video_id: str,
    segment: SuperMemoryPreparedSegment,
    sequence: int,
    occurred_at: AwareDatetime,
    ended_at: AwareDatetime,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    source_key = f"subject:{subject}:video:{video_id}:start:{segment.start_seconds:g}"
    if segment.media_objects:
        receipt = await memory.observe(
            ObserveRequest(
                tenant_id=tenant_id,
                device_id=device_id,
                boot_id=SUPERMEMORY_VQA_ADAPTER_VERSION,
                sequence=sequence,
                sensor=(
                    SensorKind.CAMERA
                    if any(item.kind is not MediaKind.AUDIO for item in segment.media_objects)
                    else SensorKind.MICROPHONE
                ),
                media_objects=segment.media_objects,
                occurred_at=occurred_at,
                ended_at=ended_at,
                observed_at=ended_at,
                idempotency_key=f"{SUPERMEMORY_VQA_ADAPTER_VERSION}:{source_key}:media",
            )
        )
        await wait_for_observation_job(
            memory,
            tenant_id,
            receipt.processing_job_id,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=processing_timeout_seconds,
        )
    if segment.transcript is not None:
        await memory.remember(
            RememberRequest(
                tenant_id=tenant_id,
                summary=segment.transcript,
                memory_type=MemoryType.EPISODIC,
                occurred_at=occurred_at,
                ended_at=ended_at,
                idempotency_key=f"{SUPERMEMORY_VQA_ADAPTER_VERSION}:{source_key}:transcript",
            )
        )


async def _answer_question(
    memory: AsyncMindBridge,
    tenant_id: str,
    question: SuperMemoryQuestion,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> SuperMemoryQuestionResult:
    async with semaphore:
        result = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(
                    text=multiple_choice_query(question.question, question.choices, rank_all=True)
                ),
                filters=RecallFilters(occurred_before=question.question_at),
                limit=recall_limit,
            )
        )
    ranking = (
        (question.unanswerable_option_index,)
        if result.answer is None
        else parse_option_ranking(result.answer, question.choices)
    )
    return SuperMemoryQuestionResult(
        question_id=question.question_id,
        predicted_option_index=ranking[0] if ranking else None,
        ranked_option_indices=ranking,
        model_answer=result.answer or "",
        mindbridge_confidence=result.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in result.memories),
        mindbridge_evidence_ids=tuple(item.evidence_id for item in result.evidence),
        mindbridge_trace_id=result.trace_id,
    )


def _segment_bounds(
    video: SuperMemoryPreparedVideo,
    segment: SuperMemoryPreparedSegment,
) -> tuple[AwareDatetime, AwareDatetime]:
    occurred_at = video.started_at + timedelta(seconds=segment.start_seconds)
    return occurred_at, occurred_at + timedelta(milliseconds=segment.duration_ms)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
