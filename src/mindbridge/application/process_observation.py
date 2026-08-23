"""Shared retry-safe use case for deriving memory from one observation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta
from time import perf_counter

from mindbridge.application.capabilities import (
    Embedder,
)
from mindbridge.application.derive_observation_graph import (
    derive_observation_graph,
    embed_observation_graph,
)
from mindbridge.application.evidence import resolve_evidence_media
from mindbridge.application.evidence_clips import (
    ClipSampling,
    derive_evidence_clips,
    generation_proxies,
)
from mindbridge.application.observation_processing import ObservationProcessingOutput
from mindbridge.application.perception import (
    EventPerception,
    PerceivedEvent,
    ResolvedEvidence,
    time_ranges_overlap,
)
from mindbridge.application.ports import (
    DerivedMediaStore,
    ObservationProcessingStore,
    Perceiver,
)
from mindbridge.core import (
    ClaimType,
    DatabaseUnavailableError,
    DomainInvariantError,
    Event,
    EventId,
    EvidenceId,
    EvidenceSpan,
    JobId,
    JobState,
    MediaObject,
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
from mindbridge.media.clipping import (
    ClipRequest,
    MediaClip,
    cut_clips,
    cut_generation_proxy,
)
from mindbridge.telemetry import (
    operation_span,
    record_output_repairs,
    record_stage_duration,
    set_current_span_attributes,
)


class ProcessObservation:
    """Turn original AV evidence into queryable events through one application path."""

    def __init__(
        self,
        store: ObservationProcessingStore,
        perceiver: Perceiver,
        media_embedder: Embedder,
        text_embedder: Embedder,
        *,
        media_url_signer: DerivedMediaStore,
        clip_sampling: ClipSampling | None = None,
        clip_cutter: Callable[[bytes, ClipRequest], tuple[MediaClip, ...]] = cut_clips,
        proxy_cutter: Callable[[bytes, ClipRequest], MediaClip] = cut_generation_proxy,
    ) -> None:
        if media_embedder.space_reference != text_embedder.space_reference:
            raise ValueError(
                "media and text embedders must write into one search space: "
                f"{media_embedder.space_reference} != {text_embedder.space_reference}"
            )
        self._store = store
        self._perceiver = perceiver
        self._media_embedder = media_embedder
        self._text_embedder = text_embedder
        self._media_url_signer = media_url_signer
        self._clip_sampling = clip_sampling or ClipSampling()
        self._clip_cutter = clip_cutter
        self._proxy_cutter = proxy_cutter

    @operation_span("mindbridge.process_observation")
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

        processing_started_at = perf_counter()
        job_age_at_claim_seconds = max(
            0.0,
            (claim.job.updated_at - claim.job.created_at).total_seconds(),
        )
        set_current_span_attributes({"mindbridge.job.attempt": claim.job.attempt})
        if claim.job.attempt == 1:
            set_current_span_attributes(
                {"mindbridge.process_observation.queue_lag_seconds": job_age_at_claim_seconds}
            )
            record_stage_duration("cloud.job_to_first_claim", job_age_at_claim_seconds)

        try:
            batch = await self._store.read_observation_batch(tenant_id, observation_id)
            evidence = await resolve_evidence_media(
                batch.evidence_spans,
                batch.media_objects,
                self._media_url_signer,
            )
            # Perception already asks the model for exactly this frame rate and pixel budget,
            # so it reads a copy cut to them rather than making the model download the frames it
            # is about to discard. The copies are lent for the length of the call: nothing
            # registers them, so scoping them here is what keeps them from outliving it.
            # `ClaimType` resolves a listed alias rather than rejecting the observation over
            # it, which means an aliased claim is indistinguishable from a native one by the
            # time it lands. The delta across this attempt is the only place the substitution
            # is still visible, so it is reported next to the event and prompt counts: a
            # vocabulary that shifts, or one that stops shifting after a prompt change, should
            # be readable rather than inferred. A worker child runs one attempt at a time, so
            # the delta is this observation's; a process that perceived concurrently would see
            # the two attempts' aliases pooled.
            aliases_before = sum(ClaimType.alias_uses().values())
            async with generation_proxies(
                tenant_id,
                evidence,
                store=self._media_url_signer,
                sampling=self._clip_sampling,
                # This attempt, not this observation: a claim reclaimed after the stale window
                # leaves two attempts running, and each must only reclaim its own copies.
                scope=f"{job_id}:{claim.job.attempt}",
                cut=self._proxy_cutter,
            ) as perceived_evidence:
                perception = await self._perceiver.perceive_events(
                    batch.observation, perceived_evidence
                )
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
            record_output_repairs(
                {
                    "mindbridge.perception.claim_type_alias_count": (
                        sum(ClaimType.alias_uses().values()) - aliases_before
                    )
                }
            )
            # Clip the grounded event spans, not the whole-file source spans:
            # a vector is only as precise as the span it was cut from.
            embedding_evidence = await resolve_evidence_media(
                event_evidence,
                _referenced_media(batch.media_objects, event_evidence),
                self._media_url_signer,
            )
            derived_clips, graph_embeddings = await asyncio.gather(
                derive_evidence_clips(
                    tenant_id,
                    embedding_evidence,
                    store=self._media_url_signer,
                    embedder=self._media_embedder,
                    sampling=self._clip_sampling,
                    created_at=claim.job.created_at,
                    cut=self._clip_cutter,
                ),
                embed_observation_graph(
                    tenant_id,
                    events,
                    graph.claims,
                    graph.entities,
                    graph.memories,
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
                media_objects=derived_clips.media_objects,
                evidence_clips=derived_clips.clips,
                embeddings=derived_clips.embeddings + graph_embeddings,
            )
            async with operation_span("mindbridge.process_observation.commit"):
                completed = await self._store.commit_observation_processing(
                    tenant_id,
                    observation_id,
                    job_id,
                    attempt=claim.job.attempt,
                    output=output,
                )
            ready_seconds = job_age_at_claim_seconds + max(
                0.0,
                perf_counter() - processing_started_at,
            )
            set_current_span_attributes(
                {"mindbridge.process_observation.searchable_ready_seconds": ready_seconds}
            )
            record_stage_duration("cloud.job_to_searchable_ready", ready_seconds)
            return completed
        except Exception as error:
            await self._store.mark_observation_processing_failed(
                tenant_id,
                observation_id,
                job_id,
                attempt=claim.job.attempt,
                error_code=_processing_error_code(error),
            )
            raise


async def record_unclaimed_processing_failure(
    store: ObservationProcessingStore,
    tenant_id: TenantId,
    observation_id: ObservationId,
    job_id: JobId,
    error: Exception,
) -> None:
    """Record a failure that happened before `run` above could claim the row.

    `task_acks_late` acks and discards the message of a task that raised, so anything that fails
    before the claim leaves no trace anywhere: no `failed` row, because the handler that writes
    one sits after the claim, and no message, because the broker already dropped it. That is the
    split the 2026-08-21 evaluation ended with -- 479 CUDA out-of-memory errors against ~17
    `failed` rows and ~318 rows stranded `pending`.

    The claim is taken here only in order to record, and that is the whole design rather than a
    detail to tidy away later. Claiming up front instead, before the models load, is broken:
    `run` claims for itself, so a second claim comes back unacquired, `run` returns the row's
    `RUNNING` state, and the Worker's running-state loop re-delivers the observation 40 times at
    30 second intervals without ever processing it. Claiming up front is also slower to recover
    from the failure that ran alongside those out-of-memory errors, a host `global_oom` that
    kills the child outright: a row already `running` is invisible to `mindbridge jobs
    --republish` until the 960 second stale window expires, while a row left `pending` is
    republished immediately. Recording is one write long, so that window never opens.

    An unacquired claim is another delivery's or a finished job's, and is left exactly as found.
    """
    claim = await store.claim_observation_processing_job(tenant_id, observation_id, job_id)
    if not claim.acquired:
        return
    await store.mark_observation_processing_failed(
        tenant_id,
        observation_id,
        job_id,
        attempt=claim.job.attempt,
        # Not the generic fallthrough: this observation was never perceived, so an operator can
        # fix the environment and republish it without paying again for a rejection it already
        # bought. `observation_processing_failed` cannot say that.
        error_code=_processing_error_code(error, "worker_setup_failed"),
    )


def _referenced_media(
    media_objects: tuple[MediaObject, ...],
    spans: tuple[EvidenceSpan, ...],
) -> tuple[MediaObject, ...]:
    """Keep only the media the given spans point at, as the resolver requires."""
    required = {span.media_object_id for span in spans}
    return tuple(item for item in media_objects if item.media_object_id in required)


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


def _processing_error_code(
    error: Exception,
    default: str = "observation_processing_failed",
) -> str:
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
    return default
