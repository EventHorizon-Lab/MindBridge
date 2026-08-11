"""Stable observe, remember, and recall use cases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from mindbridge.application.ports import MemoryAnswerer, MemoryStore, ObservationBatch
from mindbridge.contracts import (
    EvidenceView,
    MediaObjectInput,
    MemoryView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallMode,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import (
    DeviceId,
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryRecord,
    Observation,
    ObservationId,
    TenantId,
    VerificationStatus,
)


class MemoryKernel:
    """Single application path shared by every protocol adapter."""

    def __init__(
        self,
        store: MemoryStore,
        answerer: MemoryAnswerer,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._answerer = answerer
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
        return ObservationReceipt(
            observation_id=result.observation.observation_id,
            idempotency_key=idempotency_key,
            status=(ObservationStatus.ACCEPTED if result.created else ObservationStatus.DUPLICATE),
            trace_id=_new_id("trace"),
        )

    async def remember(self, request: RememberRequest) -> MemoryView:
        """Persist explicit content without pretending unsupported input is fact."""
        idempotency_key = request.idempotency_key or f"remember_{_request_digest(request)}"
        memory = MemoryRecord(
            memory_id=MemoryId(_stable_id("memory", request.tenant_id, idempotency_key)),
            tenant_id=TenantId(request.tenant_id),
            memory_type=request.memory_type,
            summary=request.summary,
            evidence_ids=tuple(EvidenceId(value) for value in request.evidence_ids),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=self._clock(),
            verification_status=(
                VerificationStatus.VERIFIED
                if request.evidence_ids
                else VerificationStatus.UNVERIFIED
            ),
        )
        result = await self._store.write_memory(
            memory,
            idempotency_key=idempotency_key,
            content_digest=_request_digest(request),
        )
        return _memory_view(result.memory)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Retrieve memories, inspect evidence, and answer only when supported."""
        memories = await self._store.search_memories(request)
        should_read_evidence = request.include_evidence or request.mode is not RecallMode.SEARCH
        evidence = (
            await self._read_recall_evidence(request, memories) if should_read_evidence else ()
        )
        answer = None
        confidence = 0.0
        if memories and request.mode is not RecallMode.SEARCH:
            generated = await self._answerer.answer(request, memories, evidence)
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

    async def _read_recall_evidence(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
    ) -> tuple[EvidenceSpan, ...]:
        evidence_ids = tuple(
            dict.fromkeys(evidence_id for memory in memories for evidence_id in memory.evidence_ids)
        )
        if not evidence_ids:
            return ()
        return await self._store.read_evidence(TenantId(request.tenant_id), evidence_ids)


def _build_observation(request: ObserveRequest) -> Observation:
    observation_id = ObservationId(
        _stable_id(
            "observation",
            request.tenant_id,
            request.device_id,
            request.boot_id,
            request.sequence,
        )
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
            _stable_id(
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
    )


def _evidence_view(evidence: EvidenceSpan) -> EvidenceView:
    return EvidenceView(
        evidence_id=evidence.evidence_id,
        media_object_id=evidence.media_object_id,
        start_ms=evidence.start_ms,
        end_ms=evidence.end_ms,
    )


def _request_digest(request: ObserveRequest | RememberRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"idempotency_key"})
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *components: object) -> str:
    canonical = json.dumps(components, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:26]
    return f"{prefix}_{digest}"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
