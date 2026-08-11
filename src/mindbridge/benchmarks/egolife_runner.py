"""Run EgoLifeQA through the public MindBridge observe and recall contract."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from itertools import groupby
from pathlib import Path
from typing import cast

from pydantic import AwareDatetime, Field, model_validator

from mindbridge.benchmarks.egolife_qa import (
    EGOLIFE_QA_ADAPTER_VERSION,
    EgoLifeOption,
    EgoLifeQuestion,
)
from mindbridge.benchmarks.runtime import (
    OPTION_LABELS,
    benchmark_tenant_id,
    multiple_choice_query,
    parse_option_ranking,
    wait_for_observation_job,
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
)
from mindbridge.core import MediaKind, SensorKind
from mindbridge.sdk import AsyncMindBridge


class EgoLifePreparedClip(ContractModel):
    """One uploaded EgoLife source clip on its official seven-day clock."""

    day: int = Field(ge=1)
    start_timecode: str = Field(pattern=r"^[0-9]{8}$")
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_video_with_duration(self) -> EgoLifePreparedClip:
        """Require enough metadata to derive a leak-free interval."""
        if self.media_object.kind is not MediaKind.VIDEO:
            raise ValueError("EgoLifeQA clips must be video media objects")
        if not self.media_object.duration_ms:
            raise ValueError("EgoLifeQA clips must have a positive duration_ms")
        _timecode_offset_ms(self.day, self.start_timecode)
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
        media_ids = tuple(clip.media_object.media_object_id for clip in self.clips)
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("EgoLifeQA clip media_object_ids must be unique")
        previous_end = -1
        for clip, start in zip(self.clips, starts, strict=True):
            if start < previous_end:
                raise ValueError("EgoLifeQA clips must not overlap")
            assert clip.media_object.duration_ms is not None
            previous_end = start + clip.media_object.duration_ms
        return self


class EgoLifeQuestionResult(ContractModel):
    """One official answer plus production retrieval diagnostics."""

    id: Identifier
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


def load_prepared_egolife(path: Path) -> EgoLifePreparedStream:
    """Load already uploaded media metadata without owning download or storage."""
    return EgoLifePreparedStream.model_validate_json(path.read_bytes())


async def run_egolife_qa(
    memory: AsyncMindBridge,
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
        while next_clip < len(prepared.clips):
            clip = prepared.clips[next_clip]
            if _clip_end_ms(clip) > query_offset_ms:
                break
            receipt = await memory.observe(
                _observe_request(tenant_id, device_id, prepared, clip, next_clip)
            )
            await wait_for_observation_job(
                memory,
                tenant_id,
                receipt.processing_job_id,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=processing_timeout_seconds,
            )
            next_clip += 1
        cutoff = prepared.timeline_origin + timedelta(milliseconds=query_offset_ms)
        results = await asyncio.gather(
            *(
                _answer_question(memory, tenant_id, question, cutoff, recall_limit, semaphore)
                for question in group
            )
        )
        answers.update((result.id, result) for result in results)
    return tuple(answers[question.question_id] for question in questions)


def _observe_request(
    tenant_id: str,
    device_id: str,
    prepared: EgoLifePreparedStream,
    clip: EgoLifePreparedClip,
    sequence: int,
) -> ObserveRequest:
    occurred_at = prepared.timeline_origin + timedelta(milliseconds=_clip_start_ms(clip))
    assert clip.media_object.duration_ms is not None
    ended_at = occurred_at + timedelta(milliseconds=clip.media_object.duration_ms)
    return ObserveRequest(
        tenant_id=tenant_id,
        device_id=device_id,
        boot_id=EGOLIFE_QA_ADAPTER_VERSION,
        sequence=sequence,
        sensor=SensorKind.CAMERA,
        media_objects=(clip.media_object,),
        occurred_at=occurred_at,
        ended_at=ended_at,
        observed_at=ended_at,
        idempotency_key=(
            f"{EGOLIFE_QA_ADAPTER_VERSION}:{prepared.subject_id}:"
            f"day:{clip.day}:clip:{clip.start_timecode}"
        ),
    )


async def _answer_question(
    memory: AsyncMindBridge,
    tenant_id: str,
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


def _clip_start_ms(clip: EgoLifePreparedClip) -> int:
    return _timecode_offset_ms(clip.day, clip.start_timecode)


def _clip_end_ms(clip: EgoLifePreparedClip) -> int:
    assert clip.media_object.duration_ms is not None
    return _clip_start_ms(clip) + clip.media_object.duration_ms


def _timecode_offset_ms(day: int, timecode: str) -> int:
    if len(timecode) != 8 or not timecode.isdigit():
        raise ValueError(f"invalid EgoLife timecode: {timecode}")
    hours, minutes, seconds, centiseconds = (
        int(timecode[0:2]),
        int(timecode[2:4]),
        int(timecode[4:6]),
        int(timecode[6:8]),
    )
    if day < 1 or hours >= 24 or minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid EgoLife timecode: {timecode}")
    return ((day - 1) * 86_400 + hours * 3_600 + minutes * 60 + seconds) * 1_000 + centiseconds * 10
