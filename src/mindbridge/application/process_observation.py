"""Shared retry-safe use case for deriving memory from one observation."""

from __future__ import annotations

from datetime import datetime, timedelta

from mindbridge.application.evidence import resolve_evidence_media
from mindbridge.application.ports import (
    EventPerception,
    MediaUrlSigner,
    ObservationPerceiver,
    ObservationProcessingOutput,
    ObservationProcessingStore,
    OmniEmbedder,
    PerceivedEvent,
    ResolvedEvidence,
)
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    Event,
    EventId,
    JobId,
    JobState,
    MemoryId,
    MemoryIntegrityError,
    MemoryRecord,
    MemoryType,
    ModelOutputError,
    ModelUnavailableError,
    ObjectStorageError,
    Observation,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
    VerificationStatus,
    derive_stable_id,
)

DOCUMENT_EMBEDDING_TASK = "retrieval_document"


class ProcessObservation:
    """Turn original AV evidence into queryable events through one application path."""

    def __init__(
        self,
        store: ObservationProcessingStore,
        perceiver: ObservationPerceiver,
        embedder: OmniEmbedder,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._perceiver = perceiver
        self._embedder = embedder
        self._media_url_signer = media_url_signer

    async def run(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        """Process one claimed attempt or return its existing durable state."""
        claim = await self._store.claim_observation_processing_job(
            tenant_id,
            observation_id,
            job_id,
        )
        if not claim.acquired:
            if claim.job.state not in {JobState.RUNNING, JobState.SUCCEEDED}:
                raise MemoryIntegrityError("unclaimed observation job has invalid state")
            return claim.job

        try:
            batch = await self._store.read_observation_batch(tenant_id, observation_id)
            evidence = await resolve_evidence_media(
                batch.evidence_spans,
                batch.media_objects,
                self._media_url_signer,
            )
            perception = await self._perceiver.perceive_events(batch.observation, evidence)
            _require_grounded_perception(batch.observation, evidence, perception.events)
            events = _events(batch.observation, perception, claim.job.created_at)
            output = ObservationProcessingOutput(
                events=events,
                memories=tuple(_event_memory(event) for event in events),
                embeddings=await _evidence_embeddings(
                    tenant_id,
                    evidence,
                    self._embedder,
                    claim.job.created_at,
                ),
            )
            return await self._store.commit_observation_processing(
                tenant_id,
                observation_id,
                job_id,
                attempt=claim.job.attempt,
                output=output,
            )
        except Exception as error:
            await self._store.mark_observation_processing_failed(
                tenant_id,
                observation_id,
                job_id,
                attempt=claim.job.attempt,
                error_code=_processing_error_code(error),
            )
            raise


def _events(
    observation: Observation,
    perception: EventPerception,
    created_at: datetime,
) -> tuple[Event, ...]:
    return tuple(
        Event(
            event_id=EventId(
                derive_stable_id(
                    "event",
                    observation.tenant_id,
                    observation.observation_id,
                    perception.prompt_version,
                    ordinal,
                )
            ),
            tenant_id=observation.tenant_id,
            observation_ids=(observation.observation_id,),
            evidence_ids=event.evidence_ids,
            occurred_at=observation.occurred_at + timedelta(milliseconds=event.start_ms),
            ended_at=observation.occurred_at + timedelta(milliseconds=event.end_ms),
            description=event.description,
            salience=event.salience,
            created_at=created_at,
            model_reference=perception.model_reference,
            prompt_version=perception.prompt_version,
        )
        for ordinal, event in enumerate(perception.events)
    )


def _event_memory(event: Event) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId(derive_stable_id("memory", event.event_id)),
        tenant_id=event.tenant_id,
        memory_type=MemoryType.EPISODIC,
        summary=event.description,
        evidence_ids=event.evidence_ids,
        occurred_at=event.occurred_at,
        ended_at=event.ended_at,
        created_at=event.created_at,
        verification_status=VerificationStatus.VERIFIED,
        model_reference=event.model_reference,
    )


async def _evidence_embeddings(
    tenant_id: TenantId,
    evidence: tuple[ResolvedEvidence, ...],
    embedder: OmniEmbedder,
    created_at: datetime,
) -> tuple[EmbeddingRecord, ...]:
    media_urls = tuple(dict.fromkeys(item.media_url for item in evidence))
    vectors = await embedder.encode_documents(media_urls)
    vector_by_url = dict(zip(media_urls, vectors, strict=True))
    return tuple(
        EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    tenant_id,
                    item.evidence_span.evidence_id,
                    embedder.model_reference.model_id,
                    embedder.model_reference.revision,
                    DOCUMENT_EMBEDDING_TASK,
                )
            ),
            tenant_id=tenant_id,
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=item.evidence_span.evidence_id,
            values=vector_by_url[item.media_url],
            model_reference=embedder.model_reference,
            task=DOCUMENT_EMBEDDING_TASK,
            dimension=embedder.dimension,
            normalized=True,
            created_at=created_at,
        )
        for item in evidence
    )


def _require_grounded_perception(
    observation: Observation,
    evidence: tuple[ResolvedEvidence, ...],
    events: tuple[PerceivedEvent, ...],
) -> None:
    duration_ms = round((observation.ended_at - observation.occurred_at).total_seconds() * 1000)
    evidence_by_id = {item.evidence_span.evidence_id: item.evidence_span for item in evidence}
    evidence_ids = set(evidence_by_id)
    for event in events:
        if event.end_ms > duration_ms or not set(event.evidence_ids) <= evidence_ids:
            raise DomainInvariantError("perceived event is outside source evidence")
        if any(
            evidence_by_id[evidence_id].end_ms < event.start_ms
            or evidence_by_id[evidence_id].start_ms > event.end_ms
            for evidence_id in event.evidence_ids
        ):
            raise DomainInvariantError("perceived event does not overlap source evidence")


def _processing_error_code(error: Exception) -> str:
    if isinstance(error, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(error, ModelOutputError):
        return "model_output_invalid"
    if isinstance(error, ObjectStorageError):
        return "object_storage_unavailable"
    if isinstance(error, MemoryIntegrityError):
        return "memory_integrity_failed"
    if isinstance(error, DomainInvariantError):
        return "domain_invariant_failed"
    return "observation_processing_failed"
