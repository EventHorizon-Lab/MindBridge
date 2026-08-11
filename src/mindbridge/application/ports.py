"""Narrow boundaries between memory use cases and external systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias

from mindbridge.contracts import RecallRequest
from mindbridge.core import (
    DeletionTombstone,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    Event,
    EvidenceId,
    EvidenceSpan,
    FeedbackId,
    FeedbackType,
    IdentityMention,
    JobId,
    MediaObject,
    MediaObjectId,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    MemoryState,
    ModelReference,
    Observation,
    ObservationId,
    ObservationJobClaim,
    ObservationProcessingJob,
    TenantId,
)

EmbeddingInput: TypeAlias = str | bytes | tuple[str | bytes, ...]


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
class ObservationBatch:
    """Atomic evidence payload persisted for one observation."""

    media_objects: tuple[MediaObject, ...]
    observation: Observation
    evidence_spans: tuple[EvidenceSpan, ...]


@dataclass(frozen=True, slots=True)
class ObservationProcessingOutput:
    """Derived records committed together with successful job state."""

    events: tuple[Event, ...]
    memories: tuple[MemoryRecord, ...]
    embeddings: tuple[EmbeddingRecord, ...]
    identity_mentions: tuple[IdentityMention, ...] = ()

    def __post_init__(self) -> None:
        if len(self.events) != len(self.memories):
            raise DomainInvariantError("each derived event must have one episodic memory")
        for event, memory in zip(self.events, self.memories, strict=True):
            if (
                event.tenant_id != memory.tenant_id
                or event.description != memory.summary
                or event.evidence_ids != memory.evidence_ids
            ):
                raise DomainInvariantError("derived event and memory provenance must match")
        embedding_ids = [embedding.embedding_id for embedding in self.embeddings]
        if len(set(embedding_ids)) != len(embedding_ids):
            raise DomainInvariantError("derived embedding IDs must be unique")
        mention_ids = [mention.mention_id for mention in self.identity_mentions]
        if len(set(mention_ids)) != len(mention_ids):
            raise DomainInvariantError("derived identity mention IDs must be unique")


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
class ResolvedQueryMedia:
    """One tenant-owned query object with short-lived model access."""

    media_object: MediaObject
    media_url: str
    media_url_expires_at: datetime

    def __post_init__(self) -> None:
        if not self.media_url.strip():
            raise DomainInvariantError("query media URL must not be empty")
        if self.media_url_expires_at.utcoffset() is None:
            raise DomainInvariantError("query media URL expiry must be timezone-aware")


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
class PerceivedEvent:
    """One schema-validated semantic interval proposed by an Omni model."""

    start_ms: int
    end_ms: int
    description: str
    salience: float
    evidence_ids: tuple[EvidenceId, ...]

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise DomainInvariantError("perceived event time range is invalid")
        if not self.description.strip():
            raise DomainInvariantError("perceived event description must not be empty")
        if not math.isfinite(self.salience) or not 0.0 <= self.salience <= 1.0:
            raise DomainInvariantError("perceived event salience must be between 0 and 1")
        if not self.evidence_ids or len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainInvariantError("perceived event evidence IDs must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class EventPerception:
    """Reproducible event proposals and the frozen model that produced them."""

    events: tuple[PerceivedEvent, ...]
    model_reference: ModelReference
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.prompt_version.strip():
            raise DomainInvariantError("perception prompt version must not be empty")


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
        if not math.isfinite(self.minimum_similarity) or not -1.0 <= self.minimum_similarity <= 1.0:
            raise DomainInvariantError("minimum_similarity must be between -1 and 1")


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

    async def read_observation_processing_job(
        self,
        tenant_id: TenantId,
        job_id: JobId,
    ) -> ObservationProcessingJob: ...


class MemoryAnswerer(Protocol):
    """Frozen model boundary used only after candidate retrieval."""

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> GeneratedAnswer: ...

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]: ...


class ObservationPerceiver(Protocol):
    """Frozen Omni boundary that inspects original observation evidence."""

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

    async def write_embedding(self, embedding: EmbeddingRecord) -> bool: ...

    async def search_embeddings(self, search: EmbeddingSearch) -> tuple[EmbeddingMatch, ...]: ...


class OmniEmbedder(Protocol):
    """Query/document-aware frozen multimodal encoder."""

    @property
    def model_reference(self) -> ModelReference: ...

    @property
    def space_reference(self) -> EmbeddingSpaceReference: ...

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
