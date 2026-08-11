"""Narrow boundaries between memory use cases and external systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias

from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingRecord,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    MemoryRecord,
    ModelReference,
    Observation,
    TenantId,
)

EmbeddingInput: TypeAlias = str | bytes | tuple[str | bytes, ...]


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
class PresignedMediaUpload:
    """A constrained PUT request for one immutable media object."""

    upload_url: str
    expires_at: datetime
    content_type: str
    checksum_sha256_base64: str

    @property
    def required_headers(self) -> dict[str, str]:
        """Return headers covered by the object-store signature."""
        return {
            "Content-Type": self.content_type,
            "x-amz-checksum-sha256": self.checksum_sha256_base64,
        }


@dataclass(frozen=True, slots=True)
class PresignedMediaDownload:
    """A short-lived GET URL for one tenant-owned media object."""

    download_url: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ResolvedEvidence:
    """An exact evidence span joined to openable source media."""

    evidence_span: EvidenceSpan
    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime

    def __post_init__(self) -> None:
        if self.evidence_span.tenant_id != self.media_object.tenant_id:
            raise DomainInvariantError("evidence and media tenants must match")
        if self.evidence_span.media_object_id != self.media_object.media_object_id:
            raise DomainInvariantError("evidence must resolve to its referenced media object")
        if not self.media_url.strip():
            raise DomainInvariantError("media_url must not be empty")
        if self.media_url_expires_at.utcoffset() is None:
            raise DomainInvariantError("media_url_expires_at must be timezone-aware")


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


@dataclass(frozen=True, slots=True)
class EmbeddingSearch:
    """One normalized query against a single frozen embedding space."""

    tenant_id: TenantId
    values: tuple[float, ...]
    model_reference: ModelReference
    document_task: str
    object_types: tuple[EmbeddedObjectType, ...]
    limit: int

    def __post_init__(self) -> None:
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise DomainInvariantError("embedding search values must be finite and non-empty")
        if not self.document_task.strip():
            raise DomainInvariantError("document_task must not be empty")
        if not self.object_types or len(set(self.object_types)) != len(self.object_types):
            raise DomainInvariantError("object_types must be non-empty and unique")
        if not 1 <= self.limit <= 1_000:
            raise DomainInvariantError("embedding search limit must be between 1 and 1000")


@dataclass(frozen=True, slots=True)
class EmbeddingMatch:
    """Cosine-ranked object returned by the semantic index."""

    embedding_id: str
    object_type: EmbeddedObjectType
    object_id: str
    similarity: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.similarity):
            raise DomainInvariantError("embedding similarity must be finite")


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

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]: ...


class MemoryAnswerer(Protocol):
    """Frozen model boundary used only after candidate retrieval."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
    ) -> GeneratedAnswer: ...


class MediaUrlSigner(Protocol):
    """Short-lived access to immutable media without proxying its bytes."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload: ...


class EmbeddingIndex(Protocol):
    """Version-aware semantic vector persistence and retrieval."""

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool: ...

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]: ...


class OmniEmbedder(Protocol):
    """Query/document-aware frozen multimodal encoder."""

    @property
    def model_reference(self) -> ModelReference: ...

    @property
    def dimension(self) -> int: ...

    async def encode_queries(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]: ...

    async def encode_documents(
        self,
        inputs: tuple[EmbeddingInput, ...],
    ) -> tuple[tuple[float, ...], ...]: ...
