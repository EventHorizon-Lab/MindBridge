"""Run ATM-Bench through public MindBridge ingestion and recall contracts.

One archive, one tenant: the release is a single person's three and a half years, and every
question is asked of the whole of it. The two media arms differ only in what is written —
`raw` sends the bytes through MindBridge's own perception, `sgm` writes the official
schema-guided text — and emails are written in both.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mindbridge.benchmarks.atm_bench import (
    ATM_BENCH_ADAPTER_VERSION,
    AtmBenchQuestion,
    AtmEmail,
    AtmQuestionType,
    AtmSgmRecord,
    atm_email_block,
    atm_evidence_kind,
    atm_memory_chunks,
    atm_sgm_block,
)
from mindbridge.benchmarks.prompts import ATM_BENCH_QUERY_PROMPT
from mindbridge.benchmarks.runtime import ingest_media
from mindbridge.contracts import (
    ContractModel,
    Identifier,
    MediaObjectInput,
    NonEmptyString,
    ObserveRequest,
    RecallMode,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import MediaKind, MemoryType, SensorKind
from mindbridge.sdk import MindBridge

AtmMediaSource = Literal["raw", "sgm"]

_FORMAT_CONSTRAINTS: dict[str, str] = {
    "number": "Answer with the number alone, including its unit or currency symbol.",
    "list_recall": (
        "Answer with the matching evidence IDs alone, separated by commas, and nothing else."
    ),
    "open_end": "Answer concisely, using only what the memories support.",
}


class AtmPreparedMedia(ContractModel):
    """One archive item already staged in the object store, keyed by its official stem."""

    media_id: Identifier
    media_object: MediaObjectInput

    @model_validator(mode="after")
    def require_official_media_object_id(self) -> AtmPreparedMedia:
        if self.media_object.media_object_id != self.media_id:
            raise ValueError("ATM-Bench media_object_id must be the official media stem")
        if self.media_object.kind not in (MediaKind.IMAGE, MediaKind.VIDEO):
            raise ValueError("ATM-Bench media must be an image or a video")
        return self


class AtmPreparedArchive(ContractModel):
    """The staged archive one `raw` run reads."""

    media: tuple[AtmPreparedMedia, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_media(self) -> AtmPreparedArchive:
        ids = tuple(item.media_id for item in self.media)
        if len(set(ids)) != len(ids):
            raise ValueError("ATM-Bench prepared media IDs must be unique")
        return self


class AtmQuestionResult(ContractModel):
    """One official-shaped prediction with this run's retrieval diagnostics."""

    question_id: Identifier
    question: NonEmptyString
    qtype: AtmQuestionType
    reference_answer: NonEmptyString
    prediction: str
    evidence_ids: tuple[Identifier, ...]
    mindbridge_confidence: float = Field(ge=0.0, le=1.0)
    mindbridge_memory_ids: tuple[Identifier, ...]
    mindbridge_media_object_ids: tuple[Identifier, ...]
    mindbridge_trace_id: Identifier
    retrieved_gold_evidence_count: int = Field(ge=0)
    mindbridge_ingest_failure_count: int = Field(default=0, ge=0)


def load_prepared_atm(path: Path) -> AtmPreparedArchive:
    """Load already staged archive metadata without owning transfer or storage."""
    return AtmPreparedArchive.model_validate_json(path.read_bytes())


def validate_prepared_atm(
    questions: Sequence[AtmBenchQuestion],
    prepared: AtmPreparedArchive | None,
    *,
    media_source: AtmMediaSource,
) -> None:
    """Refuse a `raw` run that cannot ground every media item its questions cite."""
    if media_source == "sgm":
        return
    available = {item.media_id for item in prepared.media} if prepared is not None else set()
    required = {
        evidence_id
        for question in questions
        for evidence_id in question.evidence_ids
        if atm_evidence_kind(evidence_id) == "media"
    }
    missing = required - available
    if missing:
        raise ValueError(f"missing prepared ATM-Bench media: {', '.join(sorted(missing))}")


async def ingest_atm_archive(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    media_source: AtmMediaSource,
    prepared: AtmPreparedArchive | None,
    sgm_records: Sequence[AtmSgmRecord],
    emails: Sequence[AtmEmail],
    request_concurrency: int,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> int:
    """Write the whole archive once, returning how many items failed to land.

    A failure count rather than an exception: an archive of 4,292 media items and 6,742
    emails takes hours, and one bad item must not discard the rest of the run.
    """
    if request_concurrency <= 0:
        raise ValueError("request_concurrency must be positive")
    if poll_interval_seconds <= 0 or processing_timeout_seconds <= 0:
        raise ValueError("poll interval and processing timeout must be positive")
    semaphore = asyncio.Semaphore(request_concurrency)
    by_id = {record.media_id: record for record in sgm_records}
    failures = 0

    if media_source == "raw":
        staged = prepared.media if prepared is not None else ()
        failures += await _gather_units(
            tuple(
                _observe_media(
                    memory,
                    tenant_id=tenant_id,
                    device_id=device_id,
                    item=item,
                    record=by_id.get(item.media_id),
                    sequence=sequence,
                    semaphore=semaphore,
                    poll_interval_seconds=poll_interval_seconds,
                    processing_timeout_seconds=processing_timeout_seconds,
                )
                for sequence, item in enumerate(staged)
            ),
            request_concurrency,
        )
    else:
        failures += await _gather_units(
            tuple(
                _remember_blocks(
                    memory,
                    tenant_id=tenant_id,
                    evidence_id=record.media_id,
                    block=atm_sgm_block(record),
                    occurred_at=record.occurred_at,
                    semaphore=semaphore,
                )
                for record in sgm_records
            ),
            request_concurrency,
        )

    failures += await _gather_units(
        tuple(
            _remember_blocks(
                memory,
                tenant_id=tenant_id,
                evidence_id=email.email_id,
                block=atm_email_block(email),
                occurred_at=email.occurred_at,
                semaphore=semaphore,
            )
            for email in emails
        ),
        request_concurrency,
    )
    return failures


async def answer_atm_question(
    memory: MindBridge,
    question: AtmBenchQuestion,
    *,
    tenant_id: str,
    recall_limit: int,
    ingest_failure_count: int = 0,
) -> AtmQuestionResult:
    """Ask one question of the whole archive and record what came back."""
    if not 1 <= recall_limit <= 100:
        raise ValueError("recall_limit must be between 1 and 100")
    mode = RecallMode.ENUMERATE if question.qtype == "list_recall" else RecallMode.ANSWER
    recalled = await memory.recall(
        RecallRequest(
            tenant_id=tenant_id,
            query=RecallQuery(text=_question_query(question)),
            mode=mode,
            limit=recall_limit,
        )
    )
    media_object_ids = tuple(item.media_object_id for item in recalled.evidence)
    return AtmQuestionResult(
        question_id=question.question_id,
        question=question.question,
        qtype=question.qtype,
        reference_answer=question.reference_answer,
        prediction=recalled.answer or "",
        evidence_ids=question.evidence_ids,
        mindbridge_confidence=recalled.confidence,
        mindbridge_memory_ids=tuple(item.memory_id for item in recalled.memories),
        mindbridge_media_object_ids=media_object_ids,
        mindbridge_trace_id=recalled.trace_id,
        retrieved_gold_evidence_count=len(set(question.evidence_ids) & set(media_object_ids)),
        mindbridge_ingest_failure_count=ingest_failure_count,
    )


def _question_query(question: AtmBenchQuestion) -> str:
    return ATM_BENCH_QUERY_PROMPT.text.format(
        question=question.question,
        format_constraint=_FORMAT_CONSTRAINTS[question.qtype],
    )


async def _gather_units(
    units: tuple[Coroutine[object, object, None], ...], request_concurrency: int
) -> int:
    """Await coroutines in bounded batches, counting failures instead of raising them."""
    failures = 0
    for offset in range(0, len(units), request_concurrency):
        outcomes = await asyncio.gather(
            *units[offset : offset + request_concurrency], return_exceptions=True
        )
        failures += sum(isinstance(outcome, BaseException) for outcome in outcomes)
    return failures


async def _observe_media(
    memory: MindBridge,
    *,
    tenant_id: str,
    device_id: str,
    item: AtmPreparedMedia,
    record: AtmSgmRecord | None,
    sequence: int,
    semaphore: asyncio.Semaphore,
    poll_interval_seconds: float,
    processing_timeout_seconds: float,
) -> None:
    """Observe exactly one media object, so returned evidence names one archive item."""
    duration = record.duration_seconds if record is not None else None
    started = item.media_object.created_at
    ended = started if duration is None else started + timedelta(seconds=duration)
    async with semaphore:
        await ingest_media(
            memory,
            ObserveRequest(
                tenant_id=tenant_id,
                device_id=device_id,
                boot_id=ATM_BENCH_ADAPTER_VERSION,
                sequence=sequence,
                sensor=SensorKind.CAMERA,
                media_objects=(item.media_object,),
                occurred_at=started,
                ended_at=ended,
                observed_at=ended,
                idempotency_key=f"{ATM_BENCH_ADAPTER_VERSION}:media:{item.media_id}",
            ),
            poll_interval_seconds=poll_interval_seconds,
            processing_timeout_seconds=processing_timeout_seconds,
        )


async def _remember_blocks(
    memory: MindBridge,
    *,
    tenant_id: str,
    evidence_id: str,
    block: str,
    occurred_at: datetime,
    semaphore: asyncio.Semaphore,
) -> None:
    """Write one serialized block, chunked where it exceeds the summary limit."""
    for index, chunk in enumerate(atm_memory_chunks(block, evidence_id)):
        async with semaphore:
            await memory.remember(
                RememberRequest(
                    tenant_id=tenant_id,
                    summary=chunk,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=occurred_at,
                    idempotency_key=(f"{ATM_BENCH_ADAPTER_VERSION}:text:{evidence_id}:{index}"),
                )
            )
