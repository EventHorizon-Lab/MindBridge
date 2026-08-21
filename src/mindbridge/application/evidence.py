"""Shared resolution of exact evidence spans to short-lived source media."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    MediaUrlSigner,
    PresignedMediaDownload,
    ResolvedQueryMedia,
)
from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    MemoryIntegrityError,
    MemoryRecord,
    TenantId,
)


class EvidenceReader(Protocol):
    """Narrow persistence boundary needed to resolve exact evidence media."""

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


async def read_resolved_memory_evidence(
    store: EvidenceReader,
    signer: MediaUrlSigner,
    tenant_id: TenantId,
    memories: tuple[MemoryRecord, ...],
) -> tuple[ResolvedEvidence, ...]:
    """Read and sign every distinct evidence span referenced by memories."""
    evidence_ids = tuple(
        dict.fromkeys(evidence_id for memory in memories for evidence_id in memory.evidence_ids)
    )
    return await read_resolved_evidence(store, signer, tenant_id, evidence_ids)


async def read_resolved_evidence(
    store: EvidenceReader,
    signer: MediaUrlSigner,
    tenant_id: TenantId,
    evidence_ids: tuple[EvidenceId, ...],
) -> tuple[ResolvedEvidence, ...]:
    """Read, validate, and sign a unique ordered set of exact evidence IDs."""
    evidence_ids = tuple(dict.fromkeys(evidence_ids))
    if not evidence_ids:
        return ()
    evidence_spans = await store.read_evidence(tenant_id, evidence_ids)
    if tuple(evidence.evidence_id for evidence in evidence_spans) != evidence_ids or any(
        evidence.tenant_id != tenant_id for evidence in evidence_spans
    ):
        raise MemoryIntegrityError("derived record references missing or cross-tenant evidence")
    media_object_ids = tuple(dict.fromkeys(evidence.media_object_id for evidence in evidence_spans))
    media_objects = await store.read_media_objects(tenant_id, media_object_ids)
    if tuple(media.media_object_id for media in media_objects) != media_object_ids or any(
        media.tenant_id != tenant_id for media in media_objects
    ):
        raise MemoryIntegrityError("evidence references missing media")
    clip_media = await store.read_evidence_clip_media(tenant_id, evidence_ids)
    return await resolve_evidence_media(
        evidence_spans, media_objects, signer, clip_media=clip_media
    )


async def sign_query_media(
    media_objects: tuple[MediaObject, ...],
    signer: MediaUrlSigner,
) -> tuple[ResolvedQueryMedia, ...]:
    """Give one model stage fresh access to already tenant-validated query media."""
    downloads = await asyncio.gather(
        *(signer.create_presigned_download(media_object) for media_object in media_objects)
    )
    return tuple(
        ResolvedQueryMedia(
            media_object=media_object,
            media_url=download.download_url,
            media_url_expires_at=download.expires_at,
        )
        for media_object, download in zip(media_objects, downloads, strict=True)
    )


async def resolve_evidence_media(
    evidence_spans: tuple[EvidenceSpan, ...],
    media_objects: tuple[MediaObject, ...],
    signer: MediaUrlSigner,
    *,
    max_concurrency: int = 8,
    clip_media: Mapping[EvidenceId, MediaObject] | None = None,
) -> tuple[ResolvedEvidence, ...]:
    """Join and sign evidence media without unbounded external I/O fan-out.

    A span whose derived clip is in `clip_media` is signed to that clip instead of its
    source, which is what `ResolvedEvidence.media_url` already documents. The span still
    reports its source `media_object`, so every id, uri and duration a caller reads is
    unchanged; only the bytes a model opens get smaller.
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    media_by_id = {item.media_object_id: item for item in media_objects}
    required_media_ids = tuple(
        dict.fromkeys(evidence.media_object_id for evidence in evidence_spans)
    )
    if len(media_by_id) != len(media_objects) or set(media_by_id) != set(required_media_ids):
        raise MemoryIntegrityError("evidence media set is incomplete or ambiguous")

    semaphore = asyncio.Semaphore(max_concurrency)
    clip_media = dict(clip_media or {})
    # One presign per distinct object, whether it is a source or a substituted clip.
    to_sign = {item.media_object_id: item for item in media_objects}
    to_sign.update({item.media_object_id: item for item in clip_media.values()})

    async def sign(
        media_object: MediaObject,
    ) -> tuple[MediaObjectId, PresignedMediaDownload]:
        async with semaphore:
            return media_object.media_object_id, await signer.create_presigned_download(
                media_object
            )

    downloads = dict(await asyncio.gather(*(sign(item) for item in to_sign.values())))

    def attached(evidence: EvidenceSpan) -> MediaObjectId:
        clip = clip_media.get(evidence.evidence_id)
        return clip.media_object_id if clip is not None else evidence.media_object_id

    return tuple(
        ResolvedEvidence(
            evidence_span=evidence,
            media_object=media_by_id[evidence.media_object_id],
            media_url=downloads[attached(evidence)].download_url,
            media_url_expires_at=downloads[attached(evidence)].expires_at,
        )
        for evidence in evidence_spans
    )
