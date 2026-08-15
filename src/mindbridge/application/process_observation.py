"""Shared retry-safe use case for deriving memory from one observation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

from mindbridge.application.derive_observation_graph import (
    derive_observation_graph,
    embed_observation_graph,
)
from mindbridge.application.evidence import resolve_evidence_media
from mindbridge.application.observation_processing import ObservationProcessingOutput
from mindbridge.application.perception import (
    EventPerception,
    PerceivedEvent,
    ResolvedEvidence,
    time_ranges_overlap,
)
from mindbridge.application.ports import (
    MediaUrlSigner,
    ObservationPerceiver,
    ObservationProcessingStore,
    OmniEmbedder,
    TextDocumentEmbedder,
)
from mindbridge.application.recall import RETRIEVAL_DOCUMENT_EMBEDDING_TASK
from mindbridge.core import (
    DatabaseUnavailableError,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    Event,
    EventId,
    EvidenceId,
    EvidenceSpan,
    JobId,
    JobState,
    MemoryIntegrityError,
    ModelOutputError,
    ModelRequestError,
    ModelUnavailableError,
    ObjectStorageError,
    Observation,
    ObservationId,
    ObservationProcessingJob,
    TenantId,
    derive_stable_id,
)
from mindbridge.telemetry import set_current_span_attributes, trace_operation


class ProcessObservation:
    """Turn original AV evidence into queryable events through one application path."""

    def __init__(
        self,
        store: ObservationProcessingStore,
        perceiver: ObservationPerceiver,
        embedder: OmniEmbedder,
        text_embedder: TextDocumentEmbedder,
        *,
        media_url_signer: MediaUrlSigner,
    ) -> None:
        self._store = store
        self._perceiver = perceiver
        self._embedder = embedder
        self._text_embedder = text_embedder
        self._media_url_signer = media_url_signer

    @trace_operation("mindbridge.process_observation")
    async def run(
        self,
        tenant_id: TenantId,
        observation_id: ObservationId,
        job_id: JobId,
    ) -> ObservationProcessingJob:
        """Process one claimed attempt or return its existing durable state."""
        set_current_span_attributes(
            {
                "mindbridge.tenant.id": tenant_id,
                "mindbridge.observation.id": observation_id,
                "mindbridge.job.id": job_id,
            }
        )
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
            perception, event_evidence = _ground_events(
                batch.observation,
                perception,
                batch.evidence_spans,
                claim.job.created_at,
            )
            events = _events(batch.observation, perception, claim.job.created_at)
            graph = derive_observation_graph(
                batch.observation,
                perception,
                events,
                event_evidence,
                claim.job.created_at,
            )
            set_current_span_attributes(
                {
                    "mindbridge.event.count": len(events),
                    "mindbridge.evidence.count": len(event_evidence),
                    "mindbridge.model.id": perception.model_reference.model_id,
                    "mindbridge.prompt.version": perception.prompt_version,
                }
            )
            embedding_evidence = await resolve_evidence_media(
                batch.evidence_spans,
                batch.media_objects,
                self._media_url_signer,
            )
            evidence_embeddings, graph_embeddings = await asyncio.gather(
                _evidence_embeddings(
                    tenant_id,
                    embedding_evidence,
                    self._embedder,
                    claim.job.created_at,
                ),
                embed_observation_graph(
                    tenant_id,
                    events,
                    graph.claims,
                    self._text_embedder,
                    claim.job.created_at,
                ),
            )
            output = ObservationProcessingOutput(
                evidence_spans=event_evidence,
                events=events,
                entities=graph.entities,
                entity_mentions=graph.entity_mentions,
                claims=graph.claims,
                memories=graph.memories,
                relations=graph.relations,
                embeddings=evidence_embeddings + graph_embeddings,
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


def _ground_events(
    observation: Observation,
    perception: EventPerception,
    source_evidence: tuple[EvidenceSpan, ...],
    created_at: datetime,
) -> tuple[EventPerception, tuple[EvidenceSpan, ...]]:
    """Replace whole-observation references with deterministic event-time subspans."""
    source_by_id = {span.evidence_id: span for span in source_evidence}
    grounded_events = []
    event_spans = []
    for ordinal, event in enumerate(perception.events):
        remapped: dict[EvidenceId, EvidenceId] = {}
        for source_id in event.evidence_ids:
            source = source_by_id[source_id]
            start_ms = max(source.start_ms, event.start_ms)
            end_ms = min(source.end_ms, event.end_ms)
            evidence_id = EvidenceId(
                derive_stable_id(
                    "event_evidence",
                    observation.tenant_id,
                    observation.observation_id,
                    perception.prompt_version,
                    ordinal,
                    source_id,
                    start_ms,
                    end_ms,
                )
            )
            remapped[source_id] = evidence_id
            unchanged_range = start_ms == source.start_ms and end_ms == source.end_ms
            event_spans.append(
                EvidenceSpan(
                    evidence_id=evidence_id,
                    tenant_id=source.tenant_id,
                    observation_id=source.observation_id,
                    media_object_id=source.media_object_id,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    created_at=created_at,
                    frame_start=source.frame_start if unchanged_range else None,
                    frame_end=source.frame_end if unchanged_range else None,
                    region=source.region,
                    audio_track=source.audio_track,
                )
            )
        grounded_events.append(
            replace(
                event,
                evidence_ids=tuple(remapped[item] for item in event.evidence_ids),
                entities=tuple(
                    replace(
                        entity,
                        evidence_ids=tuple(remapped[item] for item in entity.evidence_ids),
                    )
                    for entity in event.entities
                ),
                claims=tuple(
                    replace(
                        item,
                        evidence_ids=tuple(remapped[value] for value in item.evidence_ids),
                    )
                    for item in event.claims
                ),
            )
        )
    return (
        replace(perception, events=tuple(grounded_events)),
        tuple(event_spans),
    )


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
                    RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
                )
            ),
            tenant_id=tenant_id,
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=item.evidence_span.evidence_id,
            values=vector_by_url[item.media_url],
            model_reference=embedder.model_reference,
            space_reference=embedder.space_reference,
            task=RETRIEVAL_DOCUMENT_EMBEDDING_TASK,
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
            not time_ranges_overlap(
                evidence_by_id[evidence_id].start_ms,
                evidence_by_id[evidence_id].end_ms,
                event.start_ms,
                event.end_ms,
            )
            for evidence_id in event.evidence_ids
        ):
            raise DomainInvariantError("perceived event does not overlap source evidence")


def _processing_error_code(error: Exception) -> str:
    if isinstance(error, DatabaseUnavailableError):
        return "database_unavailable"
    if isinstance(error, ModelUnavailableError):
        return "model_unavailable"
    if isinstance(error, ModelOutputError):
        return "model_output_invalid"
    if isinstance(error, ModelRequestError):
        return "model_request_failed"
    if isinstance(error, ObjectStorageError):
        return "object_storage_unavailable"
    if isinstance(error, MemoryIntegrityError):
        return "memory_integrity_failed"
    if isinstance(error, DomainInvariantError):
        return "domain_invariant_failed"
    return "observation_processing_failed"
