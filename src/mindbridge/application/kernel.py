"""Stable observe, remember, and recall use cases."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from datetime import datetime, timezone

from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.application.ports import (
    EmbeddingIndex,
    MediaDeleter,
    MediaUrlSigner,
    MemoryAnswerer,
    MemoryStore,
    ObservationJobPublisher,
)
from mindbridge.application.recall import (
    RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
    RecallEmbedder,
    RecallMemories,
    memory_view,
)
from mindbridge.contracts import (
    ContractModel,
    DeletionListRequest,
    DeletionPage,
    DeletionTombstoneView,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    MediaObjectInput,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import (
    AnonymousIdentityObservation,
    DeletionPropagationState,
    DeletionTombstone,
    DeviceId,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EvidenceId,
    EvidenceSpan,
    FeedbackId,
    JobId,
    MediaObject,
    MediaObjectId,
    MemoryFeedback,
    MemoryId,
    MemoryRecord,
    ModelReference,
    ObjectStorageError,
    Observation,
    TenantId,
    TombstoneId,
    VerificationStatus,
    derive_observation_id,
    derive_stable_id,
)
from mindbridge.telemetry import (
    current_trace_id,
    set_current_span_attributes,
    trace_operation,
)


class MemoryKernel:
    """Single application path shared by every protocol adapter."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: MemoryAnswerer,
        *,
        embedding_index: EmbeddingIndex,
        media_deleter: MediaDeleter,
        media_url_signer: MediaUrlSigner,
        observation_job_publisher: ObservationJobPublisher,
        recall_embedder: RecallEmbedder,
        minimum_embedding_similarity: float = 0.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            not math.isfinite(minimum_embedding_similarity)
            or not -1.0 <= minimum_embedding_similarity <= 1.0
        ):
            raise ValueError("minimum_embedding_similarity must be between -1 and 1")
        self._store = store
        self._embedding_index = embedding_index
        self._media_deleter = media_deleter
        self._observation_job_publisher = observation_job_publisher
        self._recall_embedder = recall_embedder
        self._clock = clock or _utc_now
        self._recall = RecallMemories(
            store,
            answerer,
            embedding_index=embedding_index,
            media_url_signer=media_url_signer,
            recall_embedder=recall_embedder,
            minimum_embedding_similarity=minimum_embedding_similarity,
            clock=self._clock,
        )

    @trace_operation("mindbridge.observe")
    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        """Persist one observation atomically and acknowledge retries."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.device.id": request.device_id,
                "mindbridge.observation.sequence": request.sequence,
                "mindbridge.media.count": len(request.media_objects),
                "mindbridge.identity.count": len(request.identity_observations),
            }
        )
        observation = _build_observation(request)
        batch = ObservationBatch(
            media_objects=tuple(
                _build_media_object(item, request.tenant_id) for item in request.media_objects
            ),
            observation=observation,
            evidence_spans=tuple(
                _build_evidence_span(item, observation) for item in request.media_objects
            ),
        )
        idempotency_key = request.idempotency_key or observation.idempotency_key
        result = await self._store.write_observation(
            batch,
            idempotency_key=idempotency_key,
            content_digest=_request_digest(request),
        )
        await self._observation_job_publisher.publish_observation_processing_job(
            result.observation.tenant_id,
            result.observation.observation_id,
            result.processing_job_id,
        )
        return ObservationReceipt(
            observation_id=result.observation.observation_id,
            processing_job_id=result.processing_job_id,
            idempotency_key=idempotency_key,
            status=(ObservationStatus.ACCEPTED if result.created else ObservationStatus.DUPLICATE),
            trace_id=current_trace_id(),
        )

    @trace_operation("mindbridge.remember")
    async def remember(self, request: RememberRequest) -> MemoryView:
        """Persist explicit content without pretending unsupported input is fact."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.memory.type": request.memory_type.value,
                "mindbridge.evidence.count": len(request.evidence_ids),
            }
        )
        idempotency_key = request.idempotency_key or f"remember_{_request_digest(request)}"
        memory = MemoryRecord(
            memory_id=MemoryId(derive_stable_id("memory", request.tenant_id, idempotency_key)),
            tenant_id=TenantId(request.tenant_id),
            memory_type=request.memory_type,
            summary=request.summary,
            evidence_ids=tuple(EvidenceId(value) for value in request.evidence_ids),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=self._clock(),
            verification_status=(
                VerificationStatus.VERIFIED if request.evidence_ids else VerificationStatus.ATTESTED
            ),
        )
        result = await self._store.write_memory(
            memory,
            idempotency_key=idempotency_key,
            content_digest=_request_digest(request),
        )
        stored_memory = result.memory
        await self._index_memory(stored_memory)
        return memory_view(stored_memory)

    @trace_operation("mindbridge.record_feedback")
    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record an explainable learning signal and create a correction version when needed."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.feedback.type": request.feedback_type.value,
                "mindbridge.feedback.has_correction": request.correction_summary is not None,
            }
        )
        content_digest = _request_digest(request)
        idempotency_key = request.idempotency_key or f"feedback_{content_digest}"
        feedback = MemoryFeedback(
            feedback_id=FeedbackId(
                derive_stable_id("feedback", request.tenant_id, idempotency_key)
            ),
            tenant_id=TenantId(request.tenant_id),
            feedback_type=request.feedback_type,
            memory_id=MemoryId(request.memory_id) if request.memory_id is not None else None,
            recall_trace_id=request.recall_trace_id,
            correction_summary=request.correction_summary,
            created_at=self._clock(),
        )
        corrected_memory = (
            await self._build_corrected_memory(feedback)
            if feedback.correction_summary is not None
            else None
        )
        result = await self._store.record_feedback(
            feedback,
            corrected_memory,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )
        if result.corrected_memory is not None:
            await self._index_memory(result.corrected_memory)
        return FeedbackReceipt(
            feedback_id=result.feedback_id,
            feedback_type=result.feedback_type,
            memory_id=result.memory_id,
            corrected_memory_id=(
                result.corrected_memory.memory_id if result.corrected_memory is not None else None
            ),
            resulting_state=result.resulting_state,
            resulting_strength=result.resulting_strength,
            created_at=result.created_at,
            trace_id=current_trace_id(),
        )

    @trace_operation("mindbridge.forget")
    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        """Explicitly erase one scope through a durable, retry-safe tombstone."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.forget.target_type": request.target_type.value,
                "mindbridge.forget.target_id": request.target_id,
            }
        )
        content_digest = _request_digest(request)
        idempotency_key = request.idempotency_key or f"forget_{content_digest}"
        tombstone = DeletionTombstone(
            tombstone_id=TombstoneId(
                derive_stable_id(
                    "tombstone",
                    request.tenant_id,
                    request.target_type.value,
                    request.target_id,
                )
            ),
            tenant_id=TenantId(request.tenant_id),
            target_type=request.target_type,
            target_id=request.target_id,
            propagation_state=DeletionPropagationState.PENDING,
            requested_at=self._clock(),
        )
        plan = await self._store.prepare_forget(
            tombstone,
            idempotency_key=idempotency_key,
            content_digest=content_digest,
        )
        if plan.tombstone.propagation_state is not DeletionPropagationState.COMPLETE:
            try:
                for media_object in plan.media_objects:
                    await self._media_deleter.delete_media(media_object)
            except ObjectStorageError:
                await self._store.mark_forget_failed(
                    plan.tombstone,
                    error_code="object_storage_unavailable",
                )
                raise
            completed = await self._store.complete_forget(
                plan.tombstone,
                completed_at=self._clock(),
            )
        else:
            completed = plan.tombstone
        return _forget_receipt(completed)

    @trace_operation("mindbridge.get_forget_status")
    async def get_forget_status(self, tenant_id: str, tombstone_id: str) -> ForgetReceipt:
        """Return content-free deletion progress for one tenant-owned tombstone."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": tenant_id,
                "mindbridge.tombstone.id": tombstone_id,
            }
        )
        tombstone = await self._store.read_deletion_tombstone(
            TenantId(tenant_id),
            tombstone_id,
        )
        return _forget_receipt(tombstone)

    @trace_operation("mindbridge.list_deletions")
    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        """List stable deletion barriers for reconnecting edge devices."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.page.limit": request.limit,
            }
        )
        tombstones = await self._store.list_deletion_tombstones(
            TenantId(request.tenant_id),
            after_tombstone_id=request.cursor,
            limit=request.limit + 1,
        )
        page = tombstones[: request.limit]
        return DeletionPage(
            items=tuple(_deletion_view(tombstone) for tombstone in page),
            next_cursor=(page[-1].tombstone_id if len(tombstones) > request.limit else None),
            trace_id=current_trace_id(),
        )

    @trace_operation("mindbridge.get_observation_job")
    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        """Return one tenant-owned observation processing state."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": tenant_id,
                "mindbridge.job.id": job_id,
            }
        )
        job = await self._store.read_observation_processing_job(
            TenantId(tenant_id),
            JobId(job_id),
        )
        return ObservationProcessingJobView(
            job_id=job.job_id,
            observation_id=job.observation_id,
            state=job.state,
            attempt=job.attempt,
            error_code=job.error_code,
            created_at=job.created_at,
            updated_at=job.updated_at,
            trace_id=current_trace_id(),
        )

    @trace_operation("mindbridge.get_memory")
    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryView:
        """Return one tenant-owned memory through the shared stable view."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": tenant_id,
                "mindbridge.memory.id": memory_id,
            }
        )
        memory = await self._store.read_memory(TenantId(tenant_id), MemoryId(memory_id))
        return memory_view(memory)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Delegate recall to its focused application use case."""
        return await self._recall.run(request)

    async def _build_corrected_memory(self, feedback: MemoryFeedback) -> MemoryRecord:
        assert feedback.memory_id is not None
        assert feedback.correction_summary is not None
        original = await self._store.read_memory(feedback.tenant_id, feedback.memory_id)
        return MemoryRecord(
            memory_id=MemoryId(derive_stable_id("memory_correction", feedback.feedback_id)),
            tenant_id=feedback.tenant_id,
            memory_type=original.memory_type,
            summary=feedback.correction_summary,
            evidence_ids=original.evidence_ids,
            occurred_at=original.occurred_at,
            ended_at=original.ended_at,
            created_at=feedback.created_at,
            verification_status=VerificationStatus.ATTESTED,
            salience=original.salience,
            strength=original.salience,
            supersedes_memory_id=original.memory_id,
        )

    @trace_operation("mindbridge.index_memory")
    async def _index_memory(self, memory: MemoryRecord) -> None:
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": memory.tenant_id,
                "mindbridge.memory.id": memory.memory_id,
                "mindbridge.model.id": self._recall_embedder.document_model_reference.model_id,
                "mindbridge.embedding.dimension": self._recall_embedder.dimension,
            }
        )
        values = await self._recall_embedder.encode_memory_document(memory.summary)
        await self._embedding_index.write_embedding(
            EmbeddingRecord(
                embedding_id=EmbeddingId(
                    derive_stable_id(
                        "embedding",
                        memory.tenant_id,
                        memory.memory_id,
                        self._recall_embedder.document_model_reference.model_id,
                        self._recall_embedder.document_model_reference.revision,
                        self._recall_embedder.space_reference.space_id,
                        self._recall_embedder.space_reference.revision,
                        RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                    )
                ),
                tenant_id=memory.tenant_id,
                object_type=EmbeddedObjectType.MEMORY_RECORD,
                object_id=memory.memory_id,
                values=values,
                model_reference=self._recall_embedder.document_model_reference,
                space_reference=self._recall_embedder.space_reference,
                task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                dimension=self._recall_embedder.dimension,
                normalized=True,
                created_at=memory.created_at,
            )
        )


def _build_observation(request: ObserveRequest) -> Observation:
    observation_id = derive_observation_id(
        request.tenant_id,
        request.device_id,
        request.boot_id,
        request.sequence,
    )
    return Observation(
        observation_id=observation_id,
        tenant_id=TenantId(request.tenant_id),
        device_id=DeviceId(request.device_id),
        boot_id=request.boot_id,
        sequence=request.sequence,
        sensor=request.sensor,
        media_object_ids=tuple(
            MediaObjectId(item.media_object_id) for item in request.media_objects
        ),
        occurred_at=request.occurred_at,
        ended_at=request.ended_at,
        observed_at=request.observed_at,
        clock_offset_ms=request.clock_offset_ms,
        identity_observations=tuple(
            AnonymousIdentityObservation(
                identity_id=item.identity_id,
                kind=item.kind,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                confidence=item.confidence,
                model_reference=ModelReference(
                    model_id=item.model_id,
                    revision=item.model_revision,
                ),
            )
            for item in request.identity_observations
        ),
    )


def _build_media_object(item: MediaObjectInput, tenant_id: str) -> MediaObject:
    return MediaObject(
        media_object_id=MediaObjectId(item.media_object_id),
        tenant_id=TenantId(tenant_id),
        kind=item.kind,
        uri=item.uri,
        sha256=item.sha256,
        size_bytes=item.size_bytes,
        created_at=item.created_at,
        duration_ms=item.duration_ms,
    )


def _build_evidence_span(item: MediaObjectInput, observation: Observation) -> EvidenceSpan:
    end_ms = item.duration_ms or 0
    return EvidenceSpan(
        evidence_id=EvidenceId(
            derive_stable_id(
                "evidence",
                observation.observation_id,
                item.media_object_id,
                0,
                end_ms,
            )
        ),
        tenant_id=observation.tenant_id,
        observation_id=observation.observation_id,
        media_object_id=MediaObjectId(item.media_object_id),
        start_ms=0,
        end_ms=end_ms,
        created_at=observation.observed_at,
    )


def _forget_receipt(tombstone: DeletionTombstone) -> ForgetReceipt:
    return ForgetReceipt(
        **_deletion_view(tombstone).model_dump(),
        trace_id=current_trace_id(),
    )


def _deletion_view(tombstone: DeletionTombstone) -> DeletionTombstoneView:
    return DeletionTombstoneView(
        tombstone_id=tombstone.tombstone_id,
        target_type=tombstone.target_type,
        target_id=tombstone.target_id,
        propagation_state=tombstone.propagation_state,
        requested_at=tombstone.requested_at,
        completed_at=tombstone.completed_at,
        error_code=tombstone.error_code,
    )


def _request_digest(request: ContractModel) -> str:
    payload = request.model_dump(mode="json", exclude={"idempotency_key"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
