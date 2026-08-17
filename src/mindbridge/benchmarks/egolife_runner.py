"""Run EgoLifeQA through the public MindBridge observe and recall contract."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from itertools import groupby
from pathlib import Path
from typing import cast

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from mindbridge.benchmarks.egolife_qa import (
    EGOLIFE_QA_ADAPTER_VERSION,
    EgoLifeOption,
    EgoLifeQuestion,
    egolife_timecode_offset_ms,
)
from mindbridge.benchmarks.egomem_reason import (
    EGOMEM_REASON_ADAPTER_VERSION,
    EgoMemReasonQuestion,
)
from mindbridge.benchmarks.prompts import EGOMEM_REASON_QUERY_PROMPT
from mindbridge.benchmarks.runtime import (
    OPTION_LABELS,
    benchmark_tenant_id,
    ingest_media,
    multiple_choice_query,
    parse_option_ranking,
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
from mindbridge.sdk import MindBridge


class EgoLifePreparedClip(ContractModel):
    """One raw or officially captioned clip on EgoLife's seven-day clock."""

    day: int = Field(ge=1)
    start_timecode: str = Field(pattern=r"^[0-9]{8}$")
    media_object: MediaObjectInput | None = None
    caption: NonEmptyString | None = None
    duration_ms: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_video_with_duration(self) -> EgoLifePreparedClip:
        """Require enough metadata to derive a leak-free interval."""
        if self.media_object is None:
            if self.caption is None or self.duration_ms is None:
                raise ValueError("EgoLifeQA clips require video or a caption with duration_ms")
        else:
            if self.media_object.kind is not MediaKind.VIDEO:
                raise ValueError("EgoLifeQA clips must be video media objects")
            if not self.media_object.duration_ms:
                raise ValueError("EgoLifeQA clips must have a positive duration_ms")
            if self.duration_ms is not None and self.duration_ms != self.media_object.duration_ms:
                raise ValueError("EgoLifeQA clip durations must match")
        egolife_timecode_offset_ms(self.day, self.start_timecode)
        return self


class EgoLifePreparedStream(ContractModel):
    """Uploaded media manifest for one EgoLife wearer."""

    subject_id: Identifier
    timeline_origin: AwareDatetime
    clips: tuple[EgoLifePreparedClip, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_ordered_non_overlapping_clips(self) -> EgoLifePreparedStream:
        """Make causal ingestion deterministic and unambiguous."""
        starts = tuple(_clip_start_ms(clip) for clip in self.clips)
        if starts != tuple(sorted(starts)) or len(set(starts)) != len(starts):
            raise ValueError("EgoLifeQA clips must have unique chronological start times")
        media_ids = tuple(
            clip.media_object.media_object_id
            for clip in self.clips
            if clip.media_object is not None
        )
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("EgoLifeQA clip media_object_ids must be unique")
        previous_end = -1
        for clip, start in zip(self.clips, starts, strict=True):
            if start < previous_end:
                raise ValueError("EgoLifeQA clips must not overlap")
            previous_end = start + _clip_duration_ms(clip)
        return self


class EgoLifeQuestionResult(ContractModel):
    """One official answer plus production retrieval diagnostics."""

    id: Identifier
    subject_id: Identifier
    question: NonEmptyString
    answer: EgoLifeOption
    model_option: EgoLifeOption | None
    model_answer: str
    question_type: NonEmptyString
    query_day: int = Field(ge=1)
    query_timecode: str = Field(pattern=r"^[0-9]{8}$")
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier


class EgoLifeCategoryMetrics(ContractModel):
    """Exact-choice accuracy for one of the five official question types."""

    question_type: NonEmptyString
    question_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class EgoLifeMetrics(ContractModel):
    """Official exact-choice accuracy for one EgoLifeQA run.

    `accuracy` is the question-weighted figure published as a method's EgoLifeQA average, so it
    is only comparable once every wearer's questions are pooled into a single run. `categories`
    carries the five-column breakdown papers print beside that average.
    """

    question_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    categories: tuple[EgoLifeCategoryMetrics, ...] = Field(min_length=1)


class EgoMemReasonResult(ContractModel):
    """One leaderboard prediction plus MindBridge retrieval diagnostics."""

    example_id: int = Field(gt=0)
    question_id: Identifier
    predicted_answer: NonEmptyString
    model_answer: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier


def load_prepared_egolife(path: Path) -> EgoLifePreparedStream:
    """Load already uploaded media metadata without owning download or storage."""
    return EgoLifePreparedStream.model_validate_json(path.read_bytes())


def load_prepared_egomem(path: Path) -> tuple[EgoLifePreparedStream, ...]:
    """Load one prepared EgoLife stream per EgoMemReason identity."""
    streams = TypeAdapter(tuple[EgoLifePreparedStream, ...]).validate_json(path.read_bytes())
    if not streams:
        raise ValueError("EgoMemReason prepared media manifest must not be empty")
    subject_ids = tuple(stream.subject_id for stream in streams)
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("EgoMemReason prepared media contains duplicate subject IDs")
    media_ids = tuple(
        clip.media_object.media_object_id
        for stream in streams
        for clip in stream.clips
        if clip.media_object is not None
    )
    if len(set(media_ids)) != len(media_ids):
        raise ValueError("EgoMemReason media_object_ids must be globally unique")
    return streams


def evaluate_egolife_qa(results: tuple[EgoLifeQuestionResult, ...]) -> EgoLifeMetrics:
    """Compute the exact option accuracy used by the official EgoRAG code."""
    if not results:
        raise ValueError("EgoLifeQA results must not be empty")
    if len({result.id for result in results}) != len(results):
        raise ValueError("EgoLifeQA results must have unique IDs")
    correct = sum(result.model_option == result.answer for result in results)
    by_type: dict[str, list[EgoLifeQuestionResult]] = {}
    for result in results:
        by_type.setdefault(result.question_type, []).append(result)
    return EgoLifeMetrics(
        question_count=len(results),
        correct_count=correct,
        accuracy=correct / len(results),
        categories=tuple(
            EgoLifeCategoryMetrics(
                question_type=question_type,
                question_count=len(group),
                correct_count=sum(item.model_option == item.answer for item in group),
                accuracy=sum(item.model_option == item.answer for item in group) / len(group),
            )
            for question_type, group in sorted(by_type.items())
        ),
    )


async def run_egolife_qa(
    memory: MindBridge,
    questions: tuple[EgoLifeQuestion, ...],
    prepared: EgoLifePreparedStream,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_egolife",
    device_id: str = "egolife_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[EgoLifeQuestionResult, ...]:
    """Answer chronologically while withholding every clip that crosses query time."""
    if not questions:
        raise ValueError("EgoLifeQA questions must not be empty")
    if len({question.question_id for question in questions}) != len(questions):
        raise ValueError("EgoLifeQA questions must have unique IDs")
    if recall_limit <= 0 or request_concurrency <= 0:
        raise ValueError("recall_limit and request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")

    tenant_id = benchmark_tenant_id(tenant_prefix, prepared.subject_id, run_id)
    ordered = sorted(questions, key=lambda question: question.query_offset_ms)
    answers: dict[str, EgoLifeQuestionResult] = {}
    next_clip = 0
    semaphore = asyncio.Semaphore(request_concurrency)
    for query_offset_ms, group in groupby(ordered, key=lambda question: question.query_offset_ms):
        due_clips = []
        while next_clip < len(prepared.clips):
            clip = prepared.clips[next_clip]
            if _clip_end_ms(clip) > query_offset_ms:
                break
            due_clips.append(
                _ingest_clip(
                    memory,
                    tenant_id,
                    device_id,
                    prepared,
                    clip,
                    next_clip,
                    adapter_version=EGOLIFE_QA_ADAPTER_VERSION,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                    semaphore=semaphore,
                )
            )
            next_clip += 1
        await asyncio.gather(*due_clips)
        cutoff = prepared.timeline_origin + timedelta(milliseconds=query_offset_ms)
        results = await asyncio.gather(
            *(
                _answer_question(
                    memory,
                    tenant_id,
                    prepared.subject_id,
                    question,
                    cutoff,
                    recall_limit,
                    semaphore,
                )
                for question in group
            )
        )
        answers.update((result.id, result) for result in results)
    return tuple(answers[question.question_id] for question in questions)


async def run_egomem_reason(
    memory: MindBridge,
    questions: tuple[EgoMemReasonQuestion, ...],
    prepared: EgoLifePreparedStream,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_egomem",
    device_id: str = "egolife_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[EgoMemReasonResult, ...]:
    """Answer one wearer's questions without ingesting clips beyond each query time."""
    if not questions:
        raise ValueError("EgoMemReason questions must not be empty")
    if {question.identity for question in questions} != {prepared.subject_id}:
        raise ValueError("EgoMemReason questions and prepared subject must match")
    if len({question.example_id for question in questions}) != len(questions):
        raise ValueError("EgoMemReason questions must have unique example IDs")
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")

    tenant_id = benchmark_tenant_id(tenant_prefix, prepared.subject_id, run_id)
    ordered = sorted(questions, key=lambda question: question.query_offset_ms)
    answers: dict[int, EgoMemReasonResult] = {}
    next_clip = 0
    semaphore = asyncio.Semaphore(request_concurrency)
    for query_offset_ms, group in groupby(ordered, key=lambda question: question.query_offset_ms):
        due_clips = []
        while next_clip < len(prepared.clips):
            clip = prepared.clips[next_clip]
            if _clip_end_ms(clip) > query_offset_ms:
                break
            due_clips.append(
                _ingest_clip(
                    memory,
                    tenant_id,
                    device_id,
                    prepared,
                    clip,
                    next_clip,
                    adapter_version=EGOMEM_REASON_ADAPTER_VERSION,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                    semaphore=semaphore,
                )
            )
            next_clip += 1
        await asyncio.gather(*due_clips)
        cutoff = prepared.timeline_origin + timedelta(milliseconds=query_offset_ms)
        results = await asyncio.gather(
            *(
                _answer_egomem_question(
                    memory,
                    tenant_id,
                    question,
                    cutoff,
                    recall_limit,
                    semaphore,
                )
                for question in group
            )
        )
        answers.update((result.example_id, result) for result in results)
    return tuple(answers[question.example_id] for question in questions)


async def _ingest_clip(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    prepared: EgoLifePreparedStream,
    clip: EgoLifePreparedClip,
    sequence: int,
    *,
    adapter_version: str,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        occurred_at = prepared.timeline_origin + timedelta(milliseconds=_clip_start_ms(clip))
        ended_at = occurred_at + timedelta(milliseconds=_clip_duration_ms(clip))
        source_key = f"day:{clip.day}:clip:{clip.start_timecode}"
        evidence_ids: tuple[str, ...] = ()
        if clip.media_object is not None:
            evidence_ids = await ingest_media(
                memory,
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=adapter_version,
                    sequence=sequence,
                    sensor=SensorKind.CAMERA,
                    media_objects=(clip.media_object,),
                    occurred_at=occurred_at,
                    ended_at=ended_at,
                    observed_at=ended_at,
                    idempotency_key=(f"{adapter_version}:{prepared.subject_id}:{source_key}:media"),
                ),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
        if clip.caption is not None:
            for suffix, summary in _caption_memories(clip.caption):
                await memory.remember(
                    RememberRequest(
                        tenant_id=tenant_id,
                        summary=summary,
                        memory_type=MemoryType.EPISODIC,
                        occurred_at=occurred_at,
                        ended_at=ended_at,
                        evidence_ids=evidence_ids,
                        idempotency_key=(
                            f"{adapter_version}:{prepared.subject_id}:{source_key}:{suffix}"
                        ),
                    )
                )


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    subject_id: str,
    question: EgoLifeQuestion,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> EgoLifeQuestionResult:
    async with semaphore:
        result = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(
                    text=multiple_choice_query(question.question, question.choices, rank_all=False)
                ),
                filters=RecallFilters(occurred_before=cutoff),
                limit=recall_limit,
            )
        )
    ranking = parse_option_ranking(result.answer, question.choices)
    model_option = cast(EgoLifeOption, OPTION_LABELS[ranking[0]]) if ranking else None
    return EgoLifeQuestionResult(
        id=question.question_id,
        subject_id=subject_id,
        question=question.question,
        answer=question.correct_option,
        model_option=model_option,
        model_answer=result.answer or "",
        question_type=question.question_type,
        query_day=question.query_day,
        query_timecode=question.query_timecode,
        mindbridge_confidence=result.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in result.memories),
        mindbridge_evidence_ids=tuple(item.evidence_id for item in result.evidence),
        mindbridge_trace_id=result.trace_id,
    )


async def _answer_egomem_question(
    memory: MindBridge,
    tenant_id: str,
    question: EgoMemReasonQuestion,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> EgoMemReasonResult:
    async with semaphore:
        result = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(text=_egomem_question_query(question)),
                filters=RecallFilters(occurred_before=cutoff),
                limit=recall_limit,
            )
        )
    ranking = parse_option_ranking(result.answer, question.choices)
    if not ranking:
        raise ValueError(
            f"EgoMemReason question {question.example_id} did not produce a valid option label"
        )
    return EgoMemReasonResult(
        example_id=question.example_id,
        question_id=question.question_id,
        predicted_answer=OPTION_LABELS[ranking[0]],
        model_answer=result.answer or "",
        mindbridge_confidence=result.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in result.memories),
        mindbridge_evidence_ids=tuple(item.evidence_id for item in result.evidence),
        mindbridge_trace_id=result.trace_id,
    )


def _egomem_question_query(question: EgoMemReasonQuestion) -> str:
    return EGOMEM_REASON_QUERY_PROMPT.text.format(
        query_time=question.query_time,
        question_with_options=multiple_choice_query(
            question.question,
            question.choices,
            rank_all=False,
        ),
    )


def _clip_start_ms(clip: EgoLifePreparedClip) -> int:
    return egolife_timecode_offset_ms(clip.day, clip.start_timecode)


def _clip_end_ms(clip: EgoLifePreparedClip) -> int:
    return _clip_start_ms(clip) + _clip_duration_ms(clip)


def _caption_memories(caption: str) -> tuple[tuple[str, str], ...]:
    """Keep released visual and audio observations independently retrievable."""
    lines = tuple(line.strip() for line in caption.splitlines() if line.strip())
    visual = tuple(line for line in lines if line.startswith("Visual "))
    audio = tuple(line for line in lines if line.startswith("Audio "))
    other = tuple(line for line in lines if not line.startswith(("Visual ", "Audio ")))
    if not visual or not audio:
        return (("caption", caption),)
    memories = [
        ("visual", "\n".join(visual)),
        ("audio", "\n".join(audio)),
    ]
    if other:
        memories.append(("caption", "\n".join(other)))
    return tuple(memories)


def _clip_duration_ms(clip: EgoLifePreparedClip) -> int:
    duration_ms = (
        clip.media_object.duration_ms if clip.media_object is not None else clip.duration_ms
    )
    assert duration_ms is not None
    return duration_ms
