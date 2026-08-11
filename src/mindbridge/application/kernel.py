"""Stable observe, remember, and recall use cases."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from mindbridge.application.evidence import resolve_evidence_media
from mindbridge.application.ports import (
    EmbeddingIndex,
    EmbeddingSearch,
    MediaDeleter,
    MediaUrlSigner,
    MemoryAnswerer,
    MemoryStore,
    ObservationBatch,
    ObservationJobPublisher,
    ResolvedEvidence,
)
from mindbridge.application.ranking import fuse_memory_rankings
from mindbridge.application.recall import (
    RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
    RecallEmbedder,
    RecallEmbeddingQuery,
    ResolvedQueryMedia,
)
from mindbridge.contracts import (
    ContractModel,
    DeletionListRequest,
    DeletionPage,
    DeletionTombstoneView,
    EvidenceView,
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
    RecallMode,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import (
    DeletionPropagationState,
    DeletionTombstone,
    DeviceId,
    DomainInvariantError,
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
    MemoryIntegrityError,
    MemoryRecord,
    ObjectStorageError,
    Observation,
    TenantId,
    TombstoneId,
    VerificationStatus,
    derive_observation_id,
    derive_stable_id,
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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._answerer = answerer
        self._embedding_index = embedding_index
        self._media_deleter = media_deleter
        self._media_url_signer = media_url_signer
        self._observation_job_publisher = observation_job_publisher
        self._recall_embedder = recall_embedder
        self._clock = clock or _utc_now

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        """Persist one observation atomically and acknowledge retries."""
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
            trace_id=_new_id("trace"),
        )

    async def remember(self, request: RememberRequest) -> MemoryView:
        """Persist explicit content without pretending unsupported input is fact."""
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
        return _memory_view(stored_memory)

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record an explainable learning signal and create a correction version when needed."""
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
            trace_id=_new_id("trace"),
        )

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        """Explicitly erase one scope through a durable, retry-safe tombstone."""
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

    async def get_forget_status(self, tenant_id: str, tombstone_id: str) -> ForgetReceipt:
        """Return content-free deletion progress for one tenant-owned tombstone."""
        tombstone = await self._store.read_deletion_tombstone(
            TenantId(tenant_id),
            tombstone_id,
        )
        return _forget_receipt(tombstone)

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        """List stable deletion barriers for reconnecting edge devices."""
        tombstones = await self._store.list_deletion_tombstones(
            TenantId(request.tenant_id),
            after_tombstone_id=request.cursor,
            limit=request.limit + 1,
        )
        page = tombstones[: request.limit]
        return DeletionPage(
            items=tuple(_deletion_view(tombstone) for tombstone in page),
            next_cursor=(page[-1].tombstone_id if len(tombstones) > request.limit else None),
            trace_id=_new_id("trace"),
        )

    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        """Return one tenant-owned observation processing state."""
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
            trace_id=_new_id("trace"),
        )

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryView:
        """Return one tenant-owned memory through the shared stable view."""
        memory = await self._store.read_memory(TenantId(tenant_id), MemoryId(memory_id))
        return _memory_view(memory)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Retrieve memories, inspect evidence, and answer only when supported."""
        candidate_limit = min(request.limit * 4, 100)
        semantic_search = self._search_semantic_memories(request, limit=candidate_limit)
        if request.query.text is None:
            memories = (await semantic_search)[: request.limit]
        else:
            sparse_request = request.model_copy(update={"limit": candidate_limit})
            sparse, semantic = await asyncio.gather(
                self._store.search_memories(sparse_request),
                semantic_search,
            )
            memories = fuse_memory_rankings(
                (semantic, sparse),
                limit=request.limit,
            )
        should_read_evidence = request.include_evidence or request.mode is not RecallMode.SEARCH
        evidence = (
            await self._read_recall_evidence(request, memories) if should_read_evidence else ()
        )
        answer = None
        confidence = 0.0
        supported_memories = tuple(
            memory
            for memory in memories
            if memory.evidence_ids or memory.verification_status is VerificationStatus.ATTESTED
        )
        if supported_memories and request.mode is not RecallMode.SEARCH:
            generated = await self._answerer.answer(request, supported_memories, evidence)
            answer = generated.answer
            confidence = generated.confidence
        return RecallResult(
            answer=answer,
            confidence=confidence,
            memories=tuple(_memory_view(memory) for memory in memories),
            evidence=(
                tuple(_evidence_view(item) for item in evidence) if request.include_evidence else ()
            ),
            trace_id=_new_id("trace"),
        )

    async def _search_semantic_memories(
        self,
        request: RecallRequest,
        *,
        limit: int,
    ) -> tuple[MemoryRecord, ...]:
        query = RecallEmbeddingQuery(
            text=request.query.text,
            media=await self._resolve_query_media(request),
        )
        values = await self._recall_embedder.encode_query(query)
        searches = {
            object_type: EmbeddingSearch(
                tenant_id=TenantId(request.tenant_id),
                values=values,
                model_reference=self._recall_embedder.model_reference,
                document_task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                object_types=(object_type,),
                limit=limit,
            )
            for object_type in (
                EmbeddedObjectType.EVIDENCE_SPAN,
                EmbeddedObjectType.MEMORY_RECORD,
            )
        }
        evidence_matches, memory_matches = await asyncio.gather(
            self._embedding_index.search_embeddings(searches[EmbeddedObjectType.EVIDENCE_SPAN]),
            self._embedding_index.search_embeddings(searches[EmbeddedObjectType.MEMORY_RECORD]),
        )
        evidence_ids = tuple(
            dict.fromkeys(EvidenceId(match.object_id) for match in evidence_matches)
        )
        memory_ids = tuple(dict.fromkeys(MemoryId(match.object_id) for match in memory_matches))
        evidence_memories, direct_memories = await asyncio.gather(
            self._store.search_memories_by_evidence(request, evidence_ids, limit=limit),
            self._store.search_memories_by_ids(request, memory_ids, limit=limit),
        )
        return fuse_memory_rankings(
            (evidence_memories, direct_memories),
            limit=limit,
        )

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

    async def _index_memory(self, memory: MemoryRecord) -> None:
        values = await self._recall_embedder.encode_memory_document(memory.summary)
        await self._embedding_index.write_embedding(
            EmbeddingRecord(
                embedding_id=EmbeddingId(
                    derive_stable_id(
                        "embedding",
                        memory.tenant_id,
                        memory.memory_id,
                        self._recall_embedder.model_reference.model_id,
                        self._recall_embedder.model_reference.revision,
                        RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                    )
                ),
                tenant_id=memory.tenant_id,
                object_type=EmbeddedObjectType.MEMORY_RECORD,
                object_id=memory.memory_id,
                values=values,
                model_reference=self._recall_embedder.model_reference,
                task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                dimension=self._recall_embedder.dimension,
                normalized=True,
                created_at=memory.created_at,
            )
        )

    async def _resolve_query_media(
        self,
        request: RecallRequest,
    ) -> tuple[ResolvedQueryMedia, ...]:
        requested_ids = tuple(MediaObjectId(value) for value in request.query.media_object_ids)
        if not requested_ids:
            return ()
        tenant_id = TenantId(request.tenant_id)
        media_objects = await self._store.read_media_objects(tenant_id, requested_ids)
        if any(item.tenant_id != tenant_id for item in media_objects):
            raise MemoryIntegrityError("media store returned a cross-tenant query object")
        media_by_id = {item.media_object_id: item for item in media_objects}
        if len(media_by_id) != len(requested_ids) or set(media_by_id) != set(requested_ids):
            raise DomainInvariantError("recall query references unknown media")
        downloads = await asyncio.gather(
            *(
                self._media_url_signer.create_presigned_download(media_by_id[media_object_id])
                for media_object_id in requested_ids
            )
        )
        return tuple(
            ResolvedQueryMedia(
                media_object=media_by_id[media_object_id],
                media_url=download.download_url,
                media_url_expires_at=download.expires_at,
            )
            for media_object_id, download in zip(requested_ids, downloads, strict=True)
        )

    async def _read_recall_evidence(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[ResolvedEvidence, ...]:
        evidence_ids = tuple(
            dict.fromkeys(evidence_id for memory in memories for evidence_id in memory.evidence_ids)
        )
        if not evidence_ids:
            return ()
        tenant_id = TenantId(request.tenant_id)
        evidence_spans = await self._store.read_evidence(tenant_id, evidence_ids)
        if len(evidence_spans) != len(evidence_ids):
            raise MemoryIntegrityError("memory references missing evidence")
        media_object_ids = tuple(
            dict.fromkeys(evidence.media_object_id for evidence in evidence_spans)
        )
        media_objects = await self._store.read_media_objects(tenant_id, media_object_ids)
        if len(media_objects) != len(media_object_ids):
            raise MemoryIntegrityError("evidence references missing media")
        return await resolve_evidence_media(
            evidence_spans,
            media_objects,
            self._media_url_signer,
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


def _memory_view(memory: MemoryRecord) -> MemoryView:
    return MemoryView(
        memory_id=memory.memory_id,
        memory_type=memory.memory_type,
        summary=memory.summary,
        evidence_ids=memory.evidence_ids,
        occurred_at=memory.occurred_at,
        ended_at=memory.ended_at,
        created_at=memory.created_at,
        verification_status=memory.verification_status,
        state=memory.state,
        salience=memory.salience,
        strength=memory.strength,
        useful_access_count=memory.useful_access_count,
        positive_feedback_count=memory.positive_feedback_count,
        negative_feedback_count=memory.negative_feedback_count,
        last_accessed_at=memory.last_accessed_at,
        supersedes_memory_id=memory.supersedes_memory_id,
        superseded_at=memory.superseded_at,
    )


def _evidence_view(evidence: ResolvedEvidence) -> EvidenceView:
    return EvidenceView(
        evidence_id=evidence.evidence_span.evidence_id,
        media_object_id=evidence.media_object.media_object_id,
        start_ms=evidence.evidence_span.start_ms,
        end_ms=evidence.evidence_span.end_ms,
        media_url=evidence.media_url,
        media_url_expires_at=evidence.media_url_expires_at,
    )


def _forget_receipt(tombstone: DeletionTombstone) -> ForgetReceipt:
    return ForgetReceipt(
        **_deletion_view(tombstone).model_dump(),
        trace_id=_new_id("trace"),
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


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
