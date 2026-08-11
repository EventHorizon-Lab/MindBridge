"""Shared resolution of exact evidence spans to short-lived source media."""

from __future__ import annotations

import asyncio

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    MediaUrlSigner,
    MemoryStore,
    PresignedMediaDownload,
)
from mindbridge.core import (
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    MemoryRecord,
    TenantId,
)


async def read_resolved_memory_evidence(
    store: MemoryStore,
    signer: MediaUrlSigner,
    tenant_id: TenantId,
    memories: tuple[MemoryRecord, ...],
) -> tuple[ResolvedEvidence, ...]:
    """Read and sign every distinct evidence span referenced by memories."""
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for memory in memories for evidence_id in memory.evidence_ids)
    )
    if not evidence_ids:
        return ()
    evidence_spans = await store.read_evidence(tenant_id, evidence_ids)
    if len(evidence_spans) != len(evidence_ids):
        raise MemoryIntegrityError("memory references missing evidence")
    media_object_ids = tuple(dict.fromkeys(evidence.media_object_id for evidence in evidence_spans))
    media_objects = await store.read_media_objects(tenant_id, media_object_ids)
    if len(media_objects) != len(media_object_ids):
        raise MemoryIntegrityError("evidence references missing media")
    return await resolve_evidence_media(evidence_spans, media_objects, signer)


async def resolve_evidence_media(
    evidence_spans: tuple[EvidenceSpan, ...],
    media_objects: tuple[MediaObject, ...],
    signer: MediaUrlSigner,
    *,
    max_concurrency: int = 8,
) -> tuple[ResolvedEvidence, ...]:
    """Join and sign evidence media without unbounded external I/O fan-out."""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    media_by_id = {item.media_object_id: item for item in media_objects}
    required_media_ids = tuple(
        dict.fromkeys(evidence.media_object_id for evidence in evidence_spans)
    )
    if len(media_by_id) != len(media_objects) or set(media_by_id) != set(required_media_ids):
        raise MemoryIntegrityError("evidence media set is incomplete or ambiguous")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sign(
        media_object: MediaObject,
    ) -> tuple[MediaObjectId, PresignedMediaDownload]:
        async with semaphore:
            return media_object.media_object_id, await signer.create_presigned_download(
                media_object
            )

    downloads = dict(await asyncio.gather(*(sign(item) for item in media_objects)))
    return tuple(
        ResolvedEvidence(
            evidence_span=evidence,
            media_object=media_by_id[evidence.media_object_id],
            media_url=downloads[evidence.media_object_id].download_url,
            media_url_expires_at=downloads[evidence.media_object_id].expires_at,
        )
        for evidence in evidence_spans
    )
