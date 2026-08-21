"""Narrow boundaries between memory use cases and external systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from mindbridge.application.observation_processing import (
    ObservationBatch,
    ObservationProcessingOutput,
)
from mindbridge.application.perception import EventPerception, ResolvedEvidence
from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DeletionTombstone,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    EvidenceId,
    EvidenceSpan,
    FeedbackId,
    FeedbackType,
    JobId,
    MediaObject,
    MediaObjectId,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    MemoryState,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
    require_aware_datetime,
    require_non_empty,
    require_similarity,
)


@dataclass(frozen=True, slots=True)
class ObservationWriteResult:
    """Stored observation and whether this call created it."""

    observation: Observation
    processing_job_id: JobId
    created: bool


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """Stored memory and whether this call created it."""

    memory: MemoryRecord
    created: bool


@dataclass(frozen=True, slots=True)
class FeedbackWriteResult:
    """Stored feedback and the lifecycle snapshot produced by that event."""

    feedback_id: FeedbackId
    feedback_type: FeedbackType
    memory_id: MemoryId | None
    created_at: datetime
    resulting_state: MemoryState | None
    resulting_strength: float | None
    corrected_memory: MemoryRecord | None
    created: bool

    def __post_init__(self) -> None:
        if (self.resulting_state is None) != (self.resulting_strength is None):
            raise DomainInvariantError("feedback lifecycle state and strength must be paired")


@dataclass(frozen=True, slots=True)
class ForgetPlan:
    """Durable tombstone plus immutable objects that still require S3 deletion."""

    tombstone: DeletionTombstone
    media_objects: tuple[MediaObject, ...]


@dataclass(frozen=True, slots=True)
class PresignedMediaDownload:
    """A short-lived GET URL for one tenant-owned media object."""

    download_url: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ResolvedQueryMedia:
    """One tenant-owned query object with short-lived model access."""

    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.media_url, "query media URL")
        require_aware_datetime(self.media_url_expires_at, "query media URL expiry")


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Validated output from a frozen answer or verification model."""

    answer: str | None
    confidence: float
    retrieval_queries: tuple[str, ...] = ()
    temporal_order: Literal["relevance", "newest", "oldest"] = "relevance"

    def __post_init__(self) -> None:
        if self.answer is not None and not self.answer.strip():
            raise DomainInvariantError("answer must be non-empty when present")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise DomainInvariantError("confidence must be between 0 and 1")
        if self.answer is None and self.confidence != 0.0:
            raise DomainInvariantError("confidence must be zero when no answer is present")
        if len(self.retrieval_queries) > 2 or any(
            not query.strip() or query != query.strip() or len(query) > 2_048
            for query in self.retrieval_queries
        ):
            raise DomainInvariantError(
                "retrieval queries must contain at most two non-empty values"
            )
        if len(set(self.retrieval_queries)) != len(self.retrieval_queries):
            raise DomainInvariantError("retrieval queries must be unique")
        if self.temporal_order not in {"relevance", "newest", "oldest"}:
            raise DomainInvariantError("unknown temporal candidate order")


@dataclass(frozen=True, slots=True)
class EmbeddingSearch:
    """One normalized query against a single frozen embedding space."""

    tenant_id: TenantId
    values: tuple[float, ...]
    space_reference: EmbeddingSpaceReference
    document_task: str
    object_types: tuple[EmbeddedObjectType, ...]
    limit: int
    minimum_similarity: float = 0.0

    def __post_init__(self) -> None:
        if not self.values or not all(math.isfinite(value) for value in self.values):
            raise DomainInvariantError("embedding search values must be finite and non-empty")
        if not self.document_task.strip():
            raise DomainInvariantError("document_task must not be empty")
        if not self.object_types or len(set(self.object_types)) != len(self.object_types):
            raise DomainInvariantError("object_types must be non-empty and unique")
        if not 1 <= self.limit <= 1_000:
            raise DomainInvariantError("embedding search limit must be between 1 and 1000")
        require_similarity(self.minimum_similarity, "minimum_similarity")


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

    async def read_memory(
        self,
        tenant_id: TenantId,
        memory_id: MemoryId,
    ) -> MemoryRecord: ...

    async def record_feedback(
        self,
        feedback: MemoryFeedback,
        corrected_memory: MemoryRecord | None,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> FeedbackWriteResult: ...

    async def prepare_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        idempotency_key: str,
        content_digest: str,
    ) -> ForgetPlan: ...

    async def complete_forget(
        self,
        tombstone: DeletionTombstone,
        *,
        completed_at: datetime,
    ) -> DeletionTombstone: ...

    async def mark_forget_failed(
        self,
        tombstone: DeletionTombstone,
        *,
        error_code: str,
    ) -> DeletionTombstone: ...

    async def read_deletion_tombstone(
        self,
        tenant_id: TenantId,
        tombstone_id: str,
    ) -> DeletionTombstone: ...

    async def list_deletion_tombstones(
        self,
        tenant_id: TenantId,
        *,
        after_tombstone_id: str | None,
        limit: int,
    ) -> tuple[DeletionTombstone, ...]: ...

    async def search_memories(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def list_memories_for_enumeration(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def search_memories_by_evidence(
        self,
        request: RecallRequest,
        ranked_evidence_ids: tuple[EvidenceId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def search_memories_by_ids(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def search_memories_by_hierarchy(
        self,
        request: RecallRequest,
        ranked_memory_ids: tuple[MemoryId, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def search_memories_by_graph_objects(
        self,
        request: RecallRequest,
        ranked_objects: tuple[EmbeddingMatch, ...],
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]: ...

    async def record_memory_accesses(
        self,
        tenant_id: TenantId,
        memory_ids: tuple[MemoryId, ...],
        *,
        accessed_at: datetime,
    ) -> tuple[MemoryRecord, ...]: ...

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

    async def read_evidence_clip_media(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> dict[EvidenceId, MediaObject]: ...

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob: ...


class Answerer(Protocol):
    """Provider-neutral answer policy used by recall."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer: ...


class OccurrenceVerifier(Protocol):
    """Provider-neutral exact-occurrence policy used by enumeration."""

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]: ...


class Perceiver(Protocol):
    """Provider-neutral perception policy used by observation processing."""

    async def perceive_events(
        self,
        observation: Observation,
        evidence: tuple[ResolvedEvidence, ...],
    ) -> EventPerception: ...


class ObservationProcessingStore(Protocol):
    """Transactional persistence needed by the shared processing use case."""

    async def claim_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationJobClaim: ...

    async def read_observation_batch(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
    ) -> ObservationBatch: ...

    async def commit_observation_processing(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        output: ObservationProcessingOutput,
    ) -> ObservationProcessingJob: ...

    async def mark_observation_processing_failed(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
        *,
        attempt: int,
        error_code: str,
    ) -> ObservationProcessingJob: ...


class MediaUrlSigner(Protocol):
    """Short-lived access to immutable media without proxying its bytes."""

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload: ...


class DerivedMediaStore(MediaUrlSigner, Protocol):
    """Everything clip derivation needs from object storage."""

    async def read_media(self, media_object: MediaObject) -> bytes: ...
    async def upload_media(self, media_object: MediaObject, content: bytes) -> None: ...
    async def delete_media(self, media_object: MediaObject) -> None: ...


class DerivedMediaJanitor(Protocol):
    """Key-level access used to reclaim clips whose registration never committed."""

    async def list_media_keys(
        self,
        tenant_id: str,
        prefix: str,
    ) -> tuple[tuple[str, datetime], ...]: ...
    async def delete_media_key(self, tenant_id: str, key: str) -> None: ...


class ClipDigestStore(Protocol):
    """Reads which stored clip digests the system of record still references."""

    async def list_known_clip_digests(
        self,
        tenant_id: TenantId,
        digests: tuple[str, ...],
    ) -> frozenset[str]: ...


class MediaDeleter(Protocol):
    """Idempotent physical deletion for tenant-validated immutable media."""

    async def delete_media(self, media_object: MediaObject) -> None:
        """Delete one object; repeated calls must remain successful."""
        ...


class ObservationJobPublisher(Protocol):
    """At-least-once delivery for one already durable observation job."""

    async def publish_observation_processing_job(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> None: ...


class EmbeddingIndex(Protocol):
    """Version-aware semantic vector persistence and retrieval."""

    async def has_embedding(self, tenant_id: TenantId, embedding_id: EmbeddingId) -> bool: ...

    async def write_embedding(
        self,
        embedding: EmbeddingRecord,
        *,
        allow_reencoding: bool = False,
    ) -> bool: ...

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]: ...
