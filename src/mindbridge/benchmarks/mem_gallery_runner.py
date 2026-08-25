"""Run Mem-Gallery through public MindBridge ingestion and recall contracts.

One tenant per topic: the release is twenty independent personas, and a shared store would
leak one persona's memory into another's questions. Rounds are written per speaker and keyed
by their official round ID, which is what makes the release's `clue` annotation measurable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, model_validator

from mindbridge.benchmarks.mem_gallery import (
    MEM_GALLERY_ADAPTER_VERSION,
    MemGalleryPoint,
    MemGalleryQuestion,
    MemGalleryRound,
    MemGallerySession,
    MemGalleryTopic,
)
from mindbridge.benchmarks.prompts import (
    MEM_GALLERY_QUERY_PROMPT,
    mem_gallery_format_constraint,
)
from mindbridge.benchmarks.runtime import benchmark_tenant_id, ingest_media
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


class MemGalleryPreparedImage(ContractModel):
    """One staged image, keyed by the release-relative path that references it."""

    image_key: NonEmptyString
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_image(self) -> MemGalleryPreparedImage:
        if self.media_object.kind is not MediaKind.IMAGE:
            raise ValueError("Mem-Gallery prepared media objects must be images")
        return self


class MemGalleryPreparedImages(ContractModel):
    """Staged image lookup shared by every topic in one run."""

    images: tuple[MemGalleryPreparedImage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_images(self) -> MemGalleryPreparedImages:
        keys = tuple(image.image_key for image in self.images)
        object_ids = tuple(image.media_object.media_object_id for image in self.images)
        if len(set(keys)) != len(keys):
            raise ValueError("Mem-Gallery prepared image keys must be unique")
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("Mem-Gallery prepared media_object_ids must be unique")
        return self


class MemGalleryQuestionResult(ContractModel):
    """One official-shaped prediction with this run's retrieval diagnostics."""

    question_id: Identifier
    topic: Identifier
    point: MemGalleryPoint
    question: NonEmptyString
    reference_answer: NonEmptyString
    prediction: str
    clue_round_ids: tuple[Identifier, ...]
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_round_ids: tuple[Identifier, ...]
    mindbridge_media_object_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    retrieved_clue_round_count: int = Field(ge=0)
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


def load_prepared_mem_gallery(path: Path) -> MemGalleryPreparedImages:
    """Load already staged image metadata without owning transfer or storage."""
    return MemGalleryPreparedImages.model_validate_json(path.read_bytes())


def validate_mem_gallery_images(
    topics: Sequence[MemGalleryTopic],
    prepared: MemGalleryPreparedImages,
) -> None:
    """Refuse a run that cannot ground every image its topics and questions reference."""
    available = {image.image_key for image in prepared.images}
    required = {
        round_.image_path
        for topic in topics
        for session in topic.sessions
        for round_ in session.rounds
        if round_.image_path is not None
    } | {
        question.question_image_path
        for topic in topics
        for question in topic.questions
        if question.question_image_path is not None
    }
    missing = required - available
    if missing:
        raise ValueError(f"missing prepared Mem-Gallery images: {', '.join(sorted(missing))}")


async def run_mem_gallery_topic(
    memory: MindBridge,
    topic: MemGalleryTopic,
    *,
    run_id: str,
    prepared: MemGalleryPreparedImages,
    tenant_prefix: str = "benchmark_mem_gallery",
    device_id: str = "mem_gallery_conversation",
    recall_limit: int = 20,
    request_concurrency: int = 4,
    poll_interval_seconds: float = 1.0,
    processing_timeout_seconds: float = 1_800.0,
) -> tuple[MemGalleryQuestionResult, ...]:
    """Ingest one persona's whole dialogue, then answer every question over it."""
    if not 1 <= recall_limit <= 100 or request_concurrency <= 0:
        raise ValueError(
            "recall_limit must be between 1 and 100; request_concurrency must be positive"
        )
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    validate_mem_gallery_images((topic,), prepared)
    by_key = {image.image_key: image.media_object for image in prepared.images}
    tenant_id = benchmark_tenant_id(tenant_prefix, topic.topic, run_id)
    semaphore = asyncio.Semaphore(request_concurrency)

    ingest_failures = await _ingest_topic_sessions(
        memory,
        topic,
        tenant_id,
        device_id,
        by_key,
        semaphore,
        request_concurrency,
        poll_interval_seconds,
        processing_timeout_seconds,
    )
    return await _answer_topic_questions(
        memory,
        topic,
        tenant_id,
        by_key,
        recall_limit,
        semaphore,
        request_concurrency,
        ingest_failures,
    )


async def _ingest_topic_sessions(
    memory: MindBridge,
    topic: MemGalleryTopic,
    tenant_id: str,
    device_id: str,
    by_key: dict[str, MediaObjectInput],
    semaphore: asyncio.Semaphore,
    request_concurrency: int,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> int:
    """Ingest every session concurrently, returning how many of their rounds failed.

    Sessions run concurrently; rounds inside one session must not. Every round of a session
    shares that session's occurred_at, so nothing but insertion order records which round came
    first -- reordering them rewrites the dialogue. Distinct sessions carry distinct occurred_at,
    so their relative order survives batching. Awaiting each round at the top level would also
    make `request_concurrency` inert, because the semaphore below is only ever acquired
    uncontended inside a serial await -- the same failure mode `memlens_runner` documents at
    `_ingest_session_turns` and was fixed the same way there.
    """
    session_starts = tuple(
        sum(len(previous.rounds) for previous in topic.sessions[:index])
        for index in range(len(topic.sessions))
    )
    failures = 0
    for offset in range(0, len(topic.sessions), request_concurrency):
        outcomes = await asyncio.gather(
            *(
                _ingest_session_rounds(
                    memory,
                    tenant_id,
                    device_id,
                    session,
                    session_starts[offset + index],
                    by_key,
                    semaphore,
                    poll_interval_seconds,
                    processing_timeout_seconds,
                )
                for index, session in enumerate(
                    topic.sessions[offset : offset + request_concurrency]
                )
            ),
            return_exceptions=True,
        )
        failures += sum(outcome if isinstance(outcome, int) else 1 for outcome in outcomes)
    return failures


async def _ingest_session_rounds(
    memory: MindBridge,
    tenant_id: str,
    device_id: str,
    session: MemGallerySession,
    first_sequence: int,
    by_key: dict[str, MediaObjectInput],
    semaphore: asyncio.Semaphore,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> int:
    """Ingest one session's rounds strictly in order, counting the ones that fail."""
    failures = 0
    for index, round_ in enumerate(session.rounds):
        try:
            await _ingest_round(
                memory,
                tenant_id=tenant_id,
                device_id=device_id,
                session=session,
                round_=round_,
                sequence=first_sequence + index,
                media_object=(by_key[round_.image_path] if round_.image_path is not None else None),
                semaphore=semaphore,
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
        except Exception:
            # A bad round must not discard the rest of this session or the topic.
            failures += 1
    return failures


async def _ingest_round(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    session: MemGallerySession,
    round_: MemGalleryRound,
    sequence: int,
    media_object: MediaObjectInput | None,
    semaphore: asyncio.Semaphore,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    """Write one round: its image as an observation, then its two speaker turns."""
    evidence_ids: tuple[str, ...] = ()
    if media_object is not None:
        async with semaphore:
            evidence_ids = await ingest_media(
                memory,
                ObserveRequest(
                    tenant_id=tenant_id,
                    device_id=device_id,
                    boot_id=MEM_GALLERY_ADAPTER_VERSION,
                    sequence=sequence,
                    sensor=SensorKind.CAMERA,
                    media_objects=(media_object,),
                    occurred_at=session.occurred_at,
                    ended_at=session.occurred_at,
                    observed_at=session.occurred_at,
                    idempotency_key=(f"{MEM_GALLERY_ADAPTER_VERSION}:media:{round_.round_id}"),
                ),
                poll_interval_seconds=poll_interval_seconds,
                processing_timeout_seconds=processing_timeout_seconds,
            )
    for role, content in (("User", round_.user), ("Assistant", round_.assistant)):
        summary = f"{round_.round_id} {role} said: {content}"
        if round_.image_caption is not None and role == "User" and round_.image_id is not None:
            summary = (
                f"{round_.round_id} {role} said: {content} "
                f"[image {round_.image_id}: {round_.image_caption}]"
            )
        async with semaphore:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=summary[:2_048],
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=session.occurred_at,
                    evidence_ids=evidence_ids,
                    idempotency_key=(
                        f"{MEM_GALLERY_ADAPTER_VERSION}:text:{round_.round_id}:{role.lower()}"
                    ),
                )
            )


async def _answer_topic_questions(
    memory: MindBridge,
    topic: MemGalleryTopic,
    tenant_id: str,
    by_key: dict[str, MediaObjectInput],
    recall_limit: int,
    semaphore: asyncio.Semaphore,
    request_concurrency: int,
    ingest_failure_count: int,
) -> tuple[MemGalleryQuestionResult, ...]:
    """Answer every question concurrently; each is independent of every other.

    A question's own failure has no tolerance path -- there is no way to report a missing
    prediction as anything other than an incomplete run -- so any exception here is re-raised
    exactly as an unguarded `await memory.recall(...)` would propagate one. Only the scheduling
    changes; a bare `asyncio.gather` would still cancel sibling questions on the first failure,
    so outcomes are still collected with `return_exceptions=True` and reduced explicitly.
    """
    results: list[MemGalleryQuestionResult] = []
    for offset in range(0, len(topic.questions), request_concurrency):
        outcomes = await asyncio.gather(
            *(
                _answer_question(
                    memory,
                    topic,
                    question,
                    tenant_id=tenant_id,
                    recall_limit=recall_limit,
                    question_image=(
                        by_key[question.question_image_path]
                        if question.question_image_path is not None
                        else None
                    ),
                    semaphore=semaphore,
                    ingest_failure_count=ingest_failure_count,
                )
                for question in topic.questions[offset : offset + request_concurrency]
            ),
            return_exceptions=True,
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            results.append(outcome)
    return tuple(results)


async def _answer_question(
    memory: MindBridge,
    topic: MemGalleryTopic,
    question: MemGalleryQuestion,
    *,
    tenant_id: str,
    recall_limit: int,
    question_image: MediaObjectInput | None,
    semaphore: asyncio.Semaphore,
    ingest_failure_count: int,
) -> MemGalleryQuestionResult:
    async with semaphore:
        recalled = await memory.recall(
            RecallRequest(
                tenant_id=tenant_id,
                query=RecallQuery(
                    text=_question_query(topic, question),
                    media_object_ids=(
                        () if question_image is None else (question_image.media_object_id,)
                    ),
                ),
                limit=recall_limit,
            )
        )
    round_ids = tuple(
        dict.fromkeys(
            summary.split(" ", 1)[0]
            for summary in (item.summary for item in recalled.memories)
            if ":" in summary.split(" ", 1)[0]
        )
    )
    return MemGalleryQuestionResult(
        question_id=question.question_id,
        topic=topic.topic,
        point=question.point,
        question=question.question,
        reference_answer=question.reference_answer,
        prediction=recalled.answer or "",
        clue_round_ids=question.clue_round_ids,
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_round_ids=round_ids,
        mindbridge_media_object_ids=tuple(item.media_object_id for item in recalled.evidence),
        mindbridge_trace_id=recalled.trace_id,
        retrieved_clue_round_count=len(set(question.clue_round_ids) & set(round_ids)),
        mindbridge_ingest_failure_count=ingest_failure_count,
    )


def _question_query(topic: MemGalleryTopic, question: MemGalleryQuestion) -> str:
    """Reproduce the official query, including the speaker framing it names.

    Upstream resolves `speaker_a` to `user (<persona name>)` and `speaker_b` to
    `assistant`, so which persona the dialogue belongs to is part of what the model is
    asked. The adapter already carries that name, so dropping the clause would tell the
    model less than the benchmark tells its own baselines.
    """
    return MEM_GALLERY_QUERY_PROMPT.text.format(
        speaker_a=f"user ({topic.profile.name})",
        speaker_b="assistant",
        question=question.question,
        format_constraint=mem_gallery_format_constraint(question.point),
    )
