"""Narrow boundaries between memory use cases and external systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DomainInvariantError,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MemoryRecord,
    Observation,
    TenantId,
)


@dataclass(frozen=True, slots=True)
class ObservationWriteResult:
    """Stored observation and whether this call created it."""

    observation: Observation
    created: bool


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Stored memory and whether this call created it."""

    memory: MemoryRecord
    created: bool


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Atomic evidence payload persisted for one observation."""

    media_objects: tuple[MediaObject, ...]
    observation: Observation
    evidence_spans: tuple[EvidenceSpan, ...]


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Validated output from a frozen answer or verification model."""

    answer: str | None
    confidence: float

    def __post_init__(self) -> None:
        if self.answer is not None and not self.answer.strip():
            raise DomainInvariantError("answer must be non-empty when present")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("confidence must be between 0 and 1")
        if self.answer is None and self.confidence != 0.0:
            raise DomainInvariantError("confidence must be zero when no answer is present")


class MemoryStore(Protocol):
    """Persistence operations required by the stable use cases."""

    async def write_observation(
        self,
        batch: ObservationBatch,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ObservationWriteResult: ...

    async def write_memory(
        self,
        memory: MemoryRecord,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> MemoryWriteResult: ...

    async def search_memories(self, request: RecallRequest) -> tuple[MemoryRecord, ...]: ...

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]: ...


class MemoryAnswerer(Protocol):
    """Frozen model boundary used only after candidate retrieval."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[EvidenceSpan, ...],
    ) -> GeneratedAnswer: ...
