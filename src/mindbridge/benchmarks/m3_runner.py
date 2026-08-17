"""Run M3-Bench through the public MindBridge observe and recall contract."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from pydantic import AwareDatetime, Field, TypeAdapter, model_validator

from mindbridge.benchmarks.m3_bench import (
    M3_BENCH_ADAPTER_VERSION,
    M3BenchQuestion,
    M3BenchVideo,
)
from mindbridge.benchmarks.runtime import (
    benchmark_tenant_id,
    ingest_media,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    IdentityObservationInput,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallFilters,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge, MindBridgeError

M3_CLIP_DURATION_SECONDS = 30


class M3PreparedClip(ContractModel):
    """One raw or precomputed clip from the official 30-second split."""

    clip_index: int = Field(ge=0)
    media_object: MediaObjectInput | None = None
    caption: NonEmptyString | None = None
    duration_ms: int | None = Field(default=None, gt=0)
    identity_observations: tuple[IdentityObservationInput, ...] = Field(default=(), max_length=512)

    @model_validator(mode="after")
    def require_video_with_duration(self) -> M3PreparedClip:
        """Reject media that cannot define a video observation interval."""
        if self.media_object is None:
            if self.caption is None or self.duration_ms is None:
                raise ValueError("M3-Bench clips require video or a caption with duration_ms")
            if self.identity_observations:
                raise ValueError("M3-Bench identity observations require source video")
        else:
            if self.media_object.kind is not MediaKind.VIDEO:
                raise ValueError("M3-Bench clips must be video media objects")
            if not self.media_object.duration_ms:
                raise ValueError("M3-Bench clips must have a positive duration_ms")
            if self.duration_ms is not None and self.duration_ms != self.media_object.duration_ms:
                raise ValueError("M3-Bench clip durations must match")
        if _clip_duration_ms(self) > M3_CLIP_DURATION_SECONDS * 1_000:
            raise ValueError("M3-Bench clips must not exceed 30 seconds")
        if any(
            identity.end_ms > _clip_duration_ms(self) for identity in self.identity_observations
        ):
            raise ValueError("M3-Bench identity observation exceeds its clip")
        return self


class M3PreparedVideo(ContractModel):
    """Uploaded clip manifest aligned to one official M3-Bench video."""

    video_id: Identifier
    timeline_origin: AwareDatetime
    clips: tuple[M3PreparedClip, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_contiguous_unique_clips(self) -> M3PreparedVideo:
        """Keep official zero-based clip boundaries unambiguous."""
        indices = tuple(clip.clip_index for clip in self.clips)
        if indices != tuple(range(len(self.clips))):
            raise ValueError("M3-Bench clip indices must be contiguous and start at zero")
        media_ids = tuple(
            clip.media_object.media_object_id
            for clip in self.clips
            if clip.media_object is not None
        )
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("M3-Bench clip media_object_ids must be unique")
        if any(
            _clip_duration_ms(clip) != M3_CLIP_DURATION_SECONDS * 1_000 for clip in self.clips[:-1]
        ):
            raise ValueError("M3-Bench clips before the final clip must be exactly 30 seconds")
        return self


class M3OfficialQuestionResult(ContractModel):
    """One official M3-Bench prediction plus retrieval diagnostics."""

    id: Identifier
    question: NonEmptyString
    answer: NonEmptyString
    type: tuple[NonEmptyString, ...] = Field(min_length=1)
    timestamp_seconds: int | None = Field(default=None, ge=0)
    before_clip: int | None = Field(default=None, ge=0)
    response: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    mindbridge_error_code: NonEmptyString | None = None


def load_prepared_m3(path: Path) -> tuple[M3PreparedVideo, ...]:
    """Load uploaded media metadata without adding a benchmark-specific downloader."""
    videos = TypeAdapter(tuple[M3PreparedVideo, ...]).validate_json(path.read_bytes())
    video_ids = tuple(video.video_id for video in videos)
    if not videos:
        raise ValueError("M3-Bench prepared media manifest must not be empty")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("M3-Bench prepared media manifest contains duplicate video IDs")
    return videos


async def run_m3_video(
    memory: MindBridge,
    annotation: M3BenchVideo,
    prepared: M3PreparedVideo,
    *,
    run_id: str,
    tenant_prefix: str = "benchmark_m3",
    device_id: str = "m3_bench_camera",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[M3OfficialQuestionResult, ...]:
    """Stream one video and answer each question before future clips are ingested."""
    if annotation.video_id != prepared.video_id:
        raise ValueError("M3-Bench annotation and prepared video IDs must match")
    if recall_limit <= 0 or request_concurrency <= 0:
        raise ValueError("recall_limit and request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    if len({question.question_id for question in annotation.questions}) != len(
        annotation.questions
    ):
        raise ValueError("M3-Bench questions must have unique IDs")
    maximum_clip_index = len(prepared.clips) - 1
    if any(
        question.before_clip_index is not None and question.before_clip_index > maximum_clip_index
        for question in annotation.questions
    ):
        raise ValueError("M3-Bench question boundary exceeds prepared clips")

    tenant_id = benchmark_tenant_id(tenant_prefix, annotation.video_id, run_id)
    questions_by_boundary: dict[int | None, list[M3BenchQuestion]] = defaultdict(list)
    for question in annotation.questions:
        questions_by_boundary[question.before_clip_index].append(question)

    answers: dict[str, M3OfficialQuestionResult] = {}
    semaphore = asyncio.Semaphore(request_concurrency)
    next_clip = 0
    for boundary in sorted(value for value in questions_by_boundary if value is not None):
        await asyncio.gather(
            *(
                _ingest_clip(
                    memory,
                    tenant_id,
                    device_id,
                    annotation.video_id,
                    prepared,
                    clip,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                    semaphore=semaphore,
                )
                for clip in prepared.clips[next_clip : boundary + 1]
            )
        )
        answers.update(
            await _answer_questions(
                memory,
                tenant_id,
                questions_by_boundary[boundary],
                _clip_end(prepared, prepared.clips[boundary]),
                recall_limit,
                semaphore,
            )
        )
        next_clip = boundary + 1

    await asyncio.gather(
        *(
            _ingest_clip(
                memory,
                tenant_id,
                device_id,
                annotation.video_id,
                prepared,
                clip,
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
                semaphore=semaphore,
            )
            for clip in prepared.clips[next_clip:]
        )
    )

    answers.update(
        await _answer_questions(
            memory,
            tenant_id,
            questions_by_boundary[None],
            _clip_end(prepared, prepared.clips[-1]),
            recall_limit,
            semaphore,
        )
    )
    return tuple(answers[question.question_id] for question in annotation.questions)


async def _ingest_clip(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    video_id: str,
    prepared: M3PreparedVideo,
    clip: M3PreparedClip,
    *,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        evidence_ids: tuple[str, ...] = ()
        if clip.media_object is not None:
            evidence_ids = await ingest_media(
                memory,
                _observe_request(tenant_id, device_id, video_id, prepared, clip),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
        if clip.caption is not None:
            occurred_at = prepared.timeline_origin + timedelta(
                seconds=M3_CLIP_DURATION_SECONDS * clip.clip_index
            )
            for suffix, memory_type, summary in _caption_memories(clip.caption):
                await memory.remember(
                    RememberRequest(
                        tenant_id=tenant_id,
                        summary=summary,
                        memory_type=memory_type,
                        occurred_at=occurred_at,
                        ended_at=occurred_at + timedelta(milliseconds=_clip_duration_ms(clip)),
                        evidence_ids=evidence_ids,
                        idempotency_key=(
                            f"{M3_BENCH_ADAPTER_VERSION}:{video_id}:clip:{clip.clip_index}:{suffix}"
                        ),
                    )
                )


def _observe_request(
    tenant_id: str,
    device_id: str,
    video_id: str,
    prepared: M3PreparedVideo,
    clip: M3PreparedClip,
) -> ObserveRequest:
    occurred_at = prepared.timeline_origin + timedelta(
        seconds=M3_CLIP_DURATION_SECONDS * clip.clip_index
    )
    media_object = clip.media_object
    assert media_object is not None
    duration_ms = _clip_duration_ms(clip)
    ended_at = occurred_at + timedelta(milliseconds=duration_ms)
    return ObserveRequest(
        tenant_id=tenant_id,
        device_id=device_id,
        boot_id=M3_BENCH_ADAPTER_VERSION,
        sequence=clip.clip_index,
        sensor=SensorKind.CAMERA,
        media_objects=(media_object,),
        occurred_at=occurred_at,
        ended_at=ended_at,
        observed_at=ended_at,
        identity_observations=clip.identity_observations,
        idempotency_key=(f"{M3_BENCH_ADAPTER_VERSION}:{video_id}:clip:{clip.clip_index}"),
    )


def _caption_memories(caption: str) -> tuple[tuple[str, MemoryType, str], ...]:
    """Keep released observations and inferences in their native memory channels."""
    lines = tuple(line.strip() for line in caption.splitlines() if line.strip())
    inferences = tuple(line for line in lines if line.startswith("[Inference]"))
    events = tuple(line for line in lines if not line.startswith("[Inference]"))
    if not inferences:
        return (("caption", MemoryType.EPISODIC, caption),)
    if not events:
        return (("inferences", MemoryType.SEMANTIC, "\n".join(inferences)),)
    return (
        ("events", MemoryType.EPISODIC, "\n".join(events)),
        ("inferences", MemoryType.SEMANTIC, "\n".join(inferences)),
    )


def _clip_end(prepared: M3PreparedVideo, clip: M3PreparedClip) -> AwareDatetime:
    return prepared.timeline_origin + timedelta(
        seconds=M3_CLIP_DURATION_SECONDS * clip.clip_index,
        milliseconds=_clip_duration_ms(clip),
    )


async def _answer_questions(
    memory: MindBridge,
    tenant_id: str,
    questions: list[M3BenchQuestion],
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, M3OfficialQuestionResult]:
    results = await asyncio.gather(
        *(
            _answer_question(memory, tenant_id, question, cutoff, recall_limit, semaphore)
            for question in questions
        )
    )
    return {result.id: result for result in results}


async def _answer_question(
    memory: MindBridge,
    tenant_id: str,
    question: M3BenchQuestion,
    cutoff: AwareDatetime,
    recall_limit: int,
    semaphore: asyncio.Semaphore,
) -> M3OfficialQuestionResult:
    try:
        async with semaphore:
            result = await memory.recall(
                RecallRequest(
                    tenant_id=tenant_id,
                    query=RecallQuery(text=question.question),
                    filters=RecallFilters(occurred_before=cutoff),
                    limit=recall_limit,
                )
            )
    except MindBridgeError as error:
        if error.code not in {"model_output_invalid", "model_request_failed"}:
            raise
        return M3OfficialQuestionResult(
            id=question.question_id,
            question=question.question,
            answer=question.reference_answer,
            type=question.question_types,
            timestamp_seconds=question.timestamp_seconds,
            before_clip=question.before_clip_index,
            response="",
            mindbridge_confidence=0.0,
            mindbridge_memory_ids=(),
            mindbridge_evidence_ids=(),
            mindbridge_trace_id=(error.trace_id or f"trace_model_error_{question.question_id}"),
            mindbridge_error_code=error.code,
        )
    return M3OfficialQuestionResult(
        id=question.question_id,
        question=question.question,
        answer=question.reference_answer,
        type=question.question_types,
        timestamp_seconds=question.timestamp_seconds,
        before_clip=question.before_clip_index,
        response=result.answer or "",
        mindbridge_confidence=result.confidence,
        mindbridge_memory_ids=tuple(memory.memory_id for memory in result.memories),
        mindbridge_evidence_ids=tuple(evidence.evidence_id for evidence in result.evidence),
        mindbridge_trace_id=result.trace_id,
    )


def _clip_duration_ms(clip: M3PreparedClip) -> int:
    duration_ms = (
        clip.media_object.duration_ms if clip.media_object is not None else clip.duration_ms
    )
    assert duration_ms is not None
    return duration_ms
