"""Stable observe, remember, and recall use cases."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timedelta
from typing import cast, overload

from mindbridge.application.capabilities import (
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedTask,
    ModelInput,
    TextPart,
)
from mindbridge.application.evidence import read_resolved_memory_evidence
from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.application.ports import (
    Answerer,
    EmbeddingIndex,
    MediaDeleter,
    MediaUrlSigner,
    MemoryStore,
    ObservationJobPublisher,
    OccurrenceVerifier,
)
from mindbridge.application.recall import RecallMemories, memory_result
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
    MemoryResult,
    MemoryWriteStatus,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
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
    ModelOutputError,
    ModelReference,
    ObjectStorageError,
    Observation,
    ObservationProcessingJob,
    TenantId,
    TombstoneId,
    VerificationStatus,
    derive_observation_id,
    derive_stable_id,
    utc_now,
)
from mindbridge.telemetry import (
    current_trace_id,
    operation_span,
    set_current_span_attributes,
    trace_operation,
)

JOB_POLL_INTERVAL_SECONDS = 1.0
JOB_WATCH_MAXIMUM_SECONDS = 300.0
# ponytail: one global bound on a batch's write fan-out. Per-tenant bounds only if one
# tenant's batch is ever shown to starve another's.
_MAX_CONCURRENT_MEMORY_WRITES = 8


class MemoryKernel:
    """Single application path shared by every protocol adapter."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: Answerer,
        occurrence_verifier: OccurrenceVerifier,
        *,
        embedding_index: EmbeddingIndex,
        media_deleter: MediaDeleter,
        media_url_signer: MediaUrlSigner,
        observation_job_publisher: ObservationJobPublisher,
        embedder: Embedder,
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
        self._media_url_signer = media_url_signer
        self._observation_job_publisher = observation_job_publisher
        self._embedder = embedder
        self._clock = clock or utc_now
        self._recall = RecallMemories(
            store,
            answerer,
            occurrence_verifier,
            embedding_index=embedding_index,
            media_url_signer=media_url_signer,
            embedder=embedder,
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
            evidence_ids=tuple(span.evidence_id for span in batch.evidence_spans),
            idempotency_key=idempotency_key,
            status=(ObservationStatus.ACCEPTED if result.created else ObservationStatus.DUPLICATE),
            trace_id=current_trace_id(),
        )

    @overload
    async def remember(self, request: RememberRequest) -> RememberResult: ...

    @overload
    async def remember(
        self,
        request: tuple[RememberRequest, ...],
    ) -> tuple[RememberResult, ...]: ...

    @trace_operation("mindbridge.remember")
    async def remember(
        self,
        request: RememberRequest | tuple[RememberRequest, ...],
    ) -> RememberResult | tuple[RememberResult, ...]:
        """Persist explicit content without pretending unsupported input is fact.

        Takes one memory or a batch of them. A batch costs one encoder round trip
        instead of N: `EmbedRequest.inputs` is a sequence and the bundled encoder
        sends a whole batch as one request body, so a caller already holding N
        memories should hand over all N rather than calling N times. One request
        in, one result out; a batch in, results in request order out.

        Dispatch tests for the single request rather than for the tuple. Asking
        `isinstance(request, tuple)` would read a list as one request and fail
        later on a missing attribute, whereas anything that is not one request is
        a sequence of them.
        """
        requests = (request,) if isinstance(request, RememberRequest) else tuple(request)
        set_current_span_attributes({"mindbridge.memory.batch_size": len(requests)})
        tenant_ids = {item.tenant_id for item in requests}
        if len(tenant_ids) == 1:
            # Unlike `memory.type` and `evidence.count`, which genuinely vary per item, a batch
            # has one tenant in every caller today -- so gating this on the single-request path
            # left batch writes, the whole reason the batch API exists, invisible to a
            # per-tenant span query. Still conditional: a future mixed batch should omit the
            # attribute rather than label every span in it with whichever request came first.
            set_current_span_attributes({"mindbridge.tenant.id": next(iter(tenant_ids))})
        if len(requests) == 1:
            set_current_span_attributes(
                {
                    "mindbridge.memory.type": requests[0].memory_type.value,
                    "mindbridge.evidence.count": len(requests[0].evidence_ids),
                }
            )
        if not requests:
            return ()
        embeddings = await self._embed_summaries(tuple(item.summary for item in requests))
        # Bounds the write fan-out one batch can aim at the store: each write is up to
        # three pooled connections, so an unbounded gather over a large extraction
        # would queue behind itself inside the pool.
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT_MEMORY_WRITES)

        async def write(item: RememberRequest, embedding: Embedding) -> RememberResult:
            async with semaphore:
                return await self._write_remembered(item, embedding)

        # Every write settles before the first failure propagates. A bare gather returns on
        # the first exception while its siblings keep running detached, so the caller would be
        # told the batch failed while writes it cannot see continue against a store whose
        # request scope is already unwinding -- and a second failure among them would surface
        # only as "Task exception was never retrieved", with nothing tying it to the request.
        settled = await asyncio.gather(
            *(write(item, embedding) for item, embedding in zip(requests, embeddings, strict=True)),
            return_exceptions=True,
        )
        for outcome in settled:
            if isinstance(outcome, BaseException):
                raise outcome
        results = cast("tuple[RememberResult, ...]", tuple(settled))
        return results[0] if isinstance(request, RememberRequest) else results

    async def _write_remembered(
        self,
        request: RememberRequest,
        embedding: Embedding,
    ) -> RememberResult:
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
            verification_status=VerificationStatus.ATTESTED,
        )
        result = await self._store.write_memory(
            memory,
            idempotency_key=idempotency_key,
            content_digest=_request_digest(request),
        )
        stored_memory = result.memory
        await self._index_memory(
            stored_memory,
            embedding,
            skip_existing=not result.created,
        )
        return await self._remember_result(stored_memory, created=result.created)

    @trace_operation("mindbridge.record_feedback")
    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record an explainable learning signal and create a correction version when needed."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": request.tenant_id,
                "mindbridge.feedback.type": request.feedback_type.value,
                "mindbridge.feedback.has_correction": request.correction_summary is not None,
                # The recall that produced this feedback recorded its own query shape under the
                # same trace id, so emitting it here is what makes "which query shapes get the
                # worst feedback" answerable in the observability backend — no table, and nothing
                # written on the read path. MISSING feedback carries no memory_id, so this is the
                # only thing that ties a retrieval failure back to what was asked.
                "mindbridge.feedback.recall_trace_id": request.recall_trace_id or "",
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
            (embedding,) = await self._embed_summaries((result.corrected_memory.summary,))
            await self._index_memory(result.corrected_memory, embedding)
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
        return _job_view(job)

    async def watch_observation_job(
        self,
        tenant_id: str,
        job_id: str,
        *,
        after_updated_at: datetime | None = None,
        poll_interval_seconds: float = JOB_POLL_INTERVAL_SECONDS,
        maximum_duration_seconds: float = JOB_WATCH_MAXIMUM_SECONDS,
    ) -> AsyncGenerator[ObservationProcessingJobView, None]:
        """Yield one complete job view per observed change until this attempt settles.

        Every view is self-contained, so a caller that reconnects only needs the current state
        and never a replayed history. `after_updated_at` suppresses a state the caller already
        saw. Changes that occur between two reads coalesce into the newer state; the caller can
        therefore miss an intermediate attempt, but never observes a stale one.
        """
        if poll_interval_seconds < 0:
            raise ValueError("poll_interval_seconds must not be negative")
        if maximum_duration_seconds <= 0:
            raise ValueError("maximum_duration_seconds must be positive")
        deadline = self._clock() + timedelta(seconds=maximum_duration_seconds)
        previous: ObservationProcessingJob | None = None
        with operation_span("mindbridge.watch_observation_job"):
            while True:
                job = await self._store.read_observation_processing_job(
                    TenantId(tenant_id),
                    JobId(job_id),
                )
                changed = job != previous
                previous = job
                if changed and (after_updated_at is None or job.updated_at > after_updated_at):
                    yield _job_view(job)
                if job.state.is_settled or self._clock() >= deadline:
                    return
                await asyncio.sleep(poll_interval_seconds)

    @trace_operation("mindbridge.get_memory")
    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
        """Return one tenant-owned memory through the shared stable view."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": tenant_id,
                "mindbridge.memory.id": memory_id,
            }
        )
        memory = await self._store.read_memory(TenantId(tenant_id), MemoryId(memory_id))
        return await self._memory_result(memory)

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
            strength=original.strength,
            supersedes_memory_id=original.memory_id,
        )

    async def _remember_result(
        self,
        memory: MemoryRecord,
        *,
        created: bool,
    ) -> RememberResult:
        """Say whether this write stored the memory or matched one already under its key.

        `observe` has always answered `accepted` or `duplicate`; a caller retrying `remember`
        could only infer it from a memory_id it had seen before. The store already knows.
        """
        result = await self._memory_result(memory)
        # `__dict__` hands over the EvidenceView instances already built; dumping and
        # revalidating would rebuild every one of them, so the cost of attaching a single
        # enum would grow with the evidence count.
        return RememberResult(
            **result.__dict__,
            status=MemoryWriteStatus.CREATED if created else MemoryWriteStatus.DUPLICATE,
        )

    async def _memory_result(self, memory: MemoryRecord) -> MemoryResult:
        evidence = await read_resolved_memory_evidence(
            self._store,
            self._media_url_signer,
            memory.tenant_id,
            (memory,),
        )
        return memory_result(memory, evidence)

    async def _embed_summaries(self, summaries: tuple[str, ...]) -> tuple[Embedding, ...]:
        """Encode every summary in one call, so a batch costs one round trip."""
        result = await self._embedder.embed(
            EmbedRequest(
                inputs=tuple(ModelInput((TextPart(summary),)) for summary in summaries),
                task=EmbedTask.DOCUMENT,
            )
        )
        if len(result.embeddings) != len(summaries):
            raise ModelOutputError("embedder returned the wrong memory vector count")
        return result.embeddings

    @trace_operation("mindbridge.index_memory")
    async def _index_memory(
        self,
        memory: MemoryRecord,
        embedding: Embedding,
        *,
        skip_existing: bool = False,
    ) -> None:
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": memory.tenant_id,
                "mindbridge.memory.id": memory.memory_id,
                "mindbridge.model.id": embedding.model_reference.model_id,
                "mindbridge.embedding.dimension": embedding.dimension,
            }
        )
        embedding_id = EmbeddingId(
            derive_stable_id(
                "embedding",
                memory.tenant_id,
                memory.memory_id,
                embedding.model_reference.model_id,
                embedding.model_reference.revision,
                embedding.space_reference.space_id,
                embedding.space_reference.revision,
                EmbedTask.DOCUMENT.value,
            )
        )
        if skip_existing and await self._embedding_index.has_embedding(
            memory.tenant_id, embedding_id
        ):
            return
        # This ID pins its text, so a differing stored vector is encoder noise, not drift.
        # `embedding_id` derives from `memory_id`, which derives from the idempotency key,
        # and `write_memory` has already refused that key if it carried a different content
        # digest -- so reaching here means the stored vector encodes this exact summary.
        # It matters because a batch's vectors depend on the batch's composition: two
        # concurrent `remember` batches that share one memory encode it beside different
        # neighbours, and the difference is far outside the write's equality tolerance.
        await self._embedding_index.write_embedding(
            EmbeddingRecord(
                embedding_id=embedding_id,
                tenant_id=memory.tenant_id,
                object_type=EmbeddedObjectType.MEMORY_RECORD,
                object_id=memory.memory_id,
                values=embedding.values,
                model_reference=embedding.model_reference,
                space_reference=embedding.space_reference,
                task=EmbedTask.DOCUMENT.value,
                dimension=embedding.dimension,
                normalized=True,
                created_at=memory.created_at,
            ),
            allow_reencoding=True,
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
                scope=item.scope,
                transcript=item.transcript,
                visual_bbox_xyxy=item.visual_bbox_xyxy,
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
    end_ms = item.duration_ms
    if end_ms is None:
        end_ms = round((observation.ended_at - observation.occurred_at).total_seconds() * 1_000)
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


def _job_view(job: ObservationProcessingJob) -> ObservationProcessingJobView:
    return ObservationProcessingJobView(
        job_id=job.job_id,
        observation_id=job.observation_id,
        state=job.state,
        attempt=job.attempt,
        error_code=job.error_code,
        memory_ids=job.memory_ids,
        created_at=job.created_at,
        updated_at=job.updated_at,
        trace_id=current_trace_id(),
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
