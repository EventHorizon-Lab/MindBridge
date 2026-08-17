"""Run MEMLENS through public MindBridge ingestion and recall contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pydantic import Field, model_validator

from mindbridge.benchmarks.memlens import (
    MEMLENS_ADAPTER_VERSION,
    MemLensQuestion,
    MemLensSession,
    MemLensTurn,
)
from mindbridge.benchmarks.prompts import MEMLENS_QUERY_PROMPT
from mindbridge.benchmarks.runtime import (
    benchmark_tenant_id,
    ingest_media,
)
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge


class MemLensPreparedImage(ContractModel):
    """One uploaded image keyed by its official MEMLENS relative path."""

    source_file: NonEmptyString
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_image(self) -> MemLensPreparedImage:
        if self.media_object.kind is not MediaKind.IMAGE:
            raise ValueError("MEMLENS prepared media objects must be images")
        return self


class MemLensPreparedImages(ContractModel):
    """Uploaded image lookup shared by question-isolated MEMLENS runs."""

    images: tuple[MemLensPreparedImage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_images(self) -> MemLensPreparedImages:
        source_files = tuple(image.source_file for image in self.images)
        media_ids = tuple(image.media_object.media_object_id for image in self.images)
        if len(set(source_files)) != len(source_files):
            raise ValueError("MEMLENS prepared image paths must be unique")
        if len(set(media_ids)) != len(media_ids):
            raise ValueError("MEMLENS prepared media_object_ids must be unique")
        return self


class MemLensQuestionResult(ContractModel):
    """One official judge-compatible prediction with retrieval diagnostics."""

    question_id: Identifier
    question: NonEmptyString
    question_type: NonEmptyString
    question_subtype: NonEmptyString | None = None
    reference_answer: NonEmptyString
    old_answer: NonEmptyString | None = None
    prediction: str
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_evidence_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier


def load_prepared_memlens(path: Path) -> MemLensPreparedImages:
    """Load already uploaded image metadata without owning transfer or storage."""
    return MemLensPreparedImages.model_validate_json(path.read_bytes())


def validate_memlens_images(
    questions: tuple[MemLensQuestion, ...],
    prepared_images: MemLensPreparedImages | None,
    *,
    text_only: bool,
) -> None:
    """Reject incomplete prepared media before a MEMLENS run starts."""
    if text_only:
        return
    available = (
        {image.source_file for image in prepared_images.images}
        if prepared_images is not None
        else set()
    )
    required = {
        image.source_file
        for question in questions
        for session in question.sessions
        for turn in session.turns
        for image in turn.images
    }
    missing = required - available
    if missing:
        raise ValueError(f"missing prepared MEMLENS images: {', '.join(sorted(missing))}")


async def run_memlens_question(
    memory: MindBridge,
    question: MemLensQuestion,
    *,
    run_id: str,
    prepared_images: MemLensPreparedImages | None = None,
    text_only: bool = False,
    tenant_prefix: str = "benchmark_memlens",
    device_id: str = "memlens_conversation",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> MemLensQuestionResult:
    """Evaluate one question in a fresh tenant, matching the official agent protocol."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    image_by_source = (
        {image.source_file: image.media_object for image in prepared_images.images}
        if prepared_images is not None
        else {}
    )
    validate_memlens_images((question,), prepared_images, text_only=text_only)

    tenant_id = benchmark_tenant_id(tenant_prefix, question.question_id, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)
    sequence = 0
    for session in question.sessions:
        for index, turn in enumerate(session.turns):
            await _ingest_turn(
                memory,
                tenant_id,
                device_id,
                question.question_id,
                session,
                turn,
                sequence + index,
                image_by_source,
                text_only,
                poll_interval_seconds,
                processing_timeout_seconds,
                semaphore,
            )
        sequence += len(session.turns)

    async with semaphore:
        recalled = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(text=_question_query(question)),
                limit=recall_limit,
            )
        )
    prediction = recalled.answer or "Insufficient information"
    return MemLensQuestionResult(
        question_id=question.question_id,
        question=question.question,
        question_type=question.question_type,
        question_subtype=question.question_subtype,
        reference_answer=question.reference_answer,
        old_answer=question.old_answer,
        prediction=prediction,
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_evidence_ids=tuple(item.evidence_id for item in recalled.evidence),
        mindbridge_trace_id=recalled.trace_id,
    )


async def _ingest_turn(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    question_id: str,
    session: MemLensSession,
    turn: MemLensTurn,
    sequence: int,
    image_by_source: dict[str, MediaObjectInput],
    text_only: bool,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        evidence_ids: tuple[str, ...] = ()
        media_objects = (
            () if text_only else tuple(image_by_source[image.source_file] for image in turn.images)
        )
        if media_objects:
            evidence_ids = await ingest_media(
                memory,
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=MEMLENS_ADAPTER_VERSION,
                    sequence=sequence,
                    sensor=SensorKind.CAMERA,
                    media_objects=media_objects,
                    occurred_at=session.occurred_at,
                    ended_at=session.occurred_at,
                    observed_at=session.occurred_at,
                    idempotency_key=(
                        f"{MEMLENS_ADAPTER_VERSION}:{question_id}:{session.session_id}:"
                        f"{turn.turn_id}:media"
                    ),
                ),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
        for index, summary in enumerate(_turn_summaries(turn)):
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=summary,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=session.occurred_at,
                    evidence_ids=evidence_ids,
                    idempotency_key=(
                        f"{MEMLENS_ADAPTER_VERSION}:{question_id}:{session.session_id}:"
                        f"{turn.turn_id}:text:{index}"
                    ),
                )
            )


def _turn_summaries(turn: MemLensTurn) -> tuple[str, ...]:
    content = turn.content.replace("<image>", "[image]").strip()
    if not content:
        content = "[image]"
    speaker = "User" if turn.role == "user" else "Assistant"
    chunks = tuple(content[offset : offset + 1_900] for offset in range(0, len(content), 1_900))
    if len(chunks) == 1:
        return (f'{speaker} said: "{content}"',)
    return tuple(
        f'{speaker} said (part {index}/{len(chunks)}): "{chunk}"'
        for index, chunk in enumerate(chunks, start=1)
    )


def _question_query(question: MemLensQuestion) -> str:
    date = question.question_date.strftime("%Y/%m/%d %H:%M UTC")
    return MEMLENS_QUERY_PROMPT.text.format(question_date=date, question=question.question)
