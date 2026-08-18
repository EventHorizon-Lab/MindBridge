"""Derive one stored clip per evidence window and embed that clip.

Embedding the whole source recording gave every span of a file the same
vector. Cutting the span first, persisting it as derived media, and embedding
that clip makes each vector mean what its EvidenceSpan claims, bounds audio to
what the encoder can actually consume, and puts frame sampling under the
deployment's control.

Clip object keys are content addressed, so a retried job overwrites the same
object instead of leaving one orphan per attempt.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from urllib.parse import quote, urlsplit

from mindbridge.application.capabilities import (
    Embedder,
    EmbedRequest,
    EmbedTask,
    MediaPart,
    ModelInput,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import (
    ClipDigestStore,
    DerivedMediaJanitor,
    DerivedMediaStore,
)
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EvidenceClip,
    EvidenceSpan,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    TenantId,
    derive_stable_id,
    utc_now,
)
from mindbridge.media.clipping import (
    DEFAULT_IMAGE_MAX_PIXELS,
    DEFAULT_VIDEO_FRAMES_PER_SECOND,
    DEFAULT_VIDEO_MAX_PIXELS,
    ClipRequest,
    MediaClip,
    cut_clips,
)
from mindbridge.telemetry import operation_span, set_current_span_attributes


@dataclass(frozen=True, slots=True)
class ClipSampling:
    """Deployment-chosen sampling applied when a clip is cut."""

    frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS


@dataclass(frozen=True, slots=True)
class DerivedEvidenceClips:
    """Everything one observation's clip derivation adds to the write batch."""

    media_objects: tuple[MediaObject, ...]
    clips: tuple[EvidenceClip, ...]
    embeddings: tuple[EmbeddingRecord, ...]


@dataclass(frozen=True, slots=True)
class _StoredClip:
    """One uploaded clip, holding metadata only so bytes can be released."""

    evidence: ResolvedEvidence
    ordinal: int
    start_ms: int
    end_ms: int
    media_object: MediaObject


@operation_span("mindbridge.evidence_clips.derive")
async def derive_evidence_clips(
    tenant_id: TenantId,
    evidence: tuple[ResolvedEvidence, ...],
    *,
    store: DerivedMediaStore,
    embedder: Embedder,
    sampling: ClipSampling,
    created_at: datetime,
    max_concurrency: int = 4,
    cut: Callable[[bytes, ClipRequest], tuple[MediaClip, ...]] = cut_clips,
) -> DerivedEvidenceClips:
    """Cut, store, and embed every window of every span exactly once.

    Work is grouped per source object so one file is read once and its bytes are
    released as soon as its clips are uploaded; peak memory is therefore bounded
    by the concurrency limit rather than by the observation's total media size.
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not evidence:
        return DerivedEvidenceClips((), (), ())

    semaphore = asyncio.Semaphore(max_concurrency)
    groups: dict[MediaObjectId, list[ResolvedEvidence]] = {}
    for item in evidence:
        groups.setdefault(item.media_object.media_object_id, []).append(item)
    batches = await asyncio.gather(
        *(
            _store_source_group(
                tenant_id,
                items,
                store=store,
                sampling=sampling,
                cut=cut,
                semaphore=semaphore,
            )
            for items in groups.values()
        )
    )
    stored = tuple(item for batch in batches for item in batch)
    set_current_span_attributes(
        {
            "mindbridge.evidence.count": len(evidence),
            "mindbridge.evidence.clip_count": len(stored),
        }
    )
    embeddings = await _embed_stored_clips(tenant_id, stored, store, embedder, created_at)
    return DerivedEvidenceClips(
        media_objects=tuple(item.media_object for item in _unique_by_media_object(stored)),
        clips=tuple(
            EvidenceClip(
                tenant_id=tenant_id,
                evidence_id=item.evidence.evidence_span.evidence_id,
                ordinal=item.ordinal,
                media_object_id=item.media_object.media_object_id,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
                created_at=created_at,
            )
            for item in stored
        ),
        embeddings=embeddings,
    )


async def _store_source_group(
    tenant_id: TenantId,
    items: list[ResolvedEvidence],
    *,
    store: DerivedMediaStore,
    sampling: ClipSampling,
    cut: Callable[[bytes, ClipRequest], tuple[MediaClip, ...]],
    semaphore: asyncio.Semaphore,
) -> tuple[_StoredClip, ...]:
    """Read one source once, cut every span on it, upload, then drop the bytes."""
    async with semaphore:
        source = await store.read_media(items[0].media_object)
        stored: list[_StoredClip] = []
        for item in items:
            span = item.evidence_span
            request = ClipRequest(
                kind=item.media_object.kind,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                frames_per_second=sampling.frames_per_second,
                max_pixels=sampling.max_pixels,
                image_max_pixels=sampling.image_max_pixels,
                region=_span_region(span),
            )
            clips = await asyncio.to_thread(cut, source, request)
            for ordinal, clip in enumerate(clips):
                media_object = _clip_media_object(tenant_id, item, clip, created_at=span.created_at)
                await store.upload_media(media_object, clip.content)
                stored.append(
                    _StoredClip(
                        evidence=item,
                        ordinal=ordinal,
                        start_ms=clip.start_ms,
                        end_ms=clip.end_ms,
                        media_object=media_object,
                    )
                )
        del source
        return tuple(stored)


async def _embed_stored_clips(
    tenant_id: TenantId,
    stored: tuple[_StoredClip, ...],
    store: DerivedMediaStore,
    embedder: Embedder,
    created_at: datetime,
) -> tuple[EmbeddingRecord, ...]:
    """Sign immediately before encoding so no URL ages during the upload phase."""
    if not stored:
        return ()
    unique = _unique_by_media_object(stored)
    signed = await asyncio.gather(
        *(store.create_presigned_download(item.media_object) for item in unique)
    )
    urls = {
        item.media_object.media_object_id: download.download_url
        for item, download in zip(unique, signed, strict=True)
    }
    result = await embedder.embed(
        EmbedRequest(
            inputs=tuple(
                ModelInput(
                    (
                        MediaPart(
                            kind=item.media_object.kind,
                            url=urls[item.media_object.media_object_id],
                            source_uri=item.media_object.uri,
                        ),
                    )
                )
                for item in stored
            ),
            task=EmbedTask.DOCUMENT,
        )
    )
    if len(result.embeddings) != len(stored):
        raise ModelOutputError("embedder returned the wrong evidence clip vector count")
    return tuple(
        EmbeddingRecord(
            embedding_id=EmbeddingId(
                derive_stable_id(
                    "embedding",
                    tenant_id,
                    item.evidence.evidence_span.evidence_id,
                    str(item.ordinal),
                    embedding.model_reference.model_id,
                    embedding.model_reference.revision,
                    EmbedTask.DOCUMENT.value,
                )
            ),
            tenant_id=tenant_id,
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            object_id=item.evidence.evidence_span.evidence_id,
            values=embedding.values,
            model_reference=embedding.model_reference,
            space_reference=embedding.space_reference,
            task=EmbedTask.DOCUMENT.value,
            dimension=embedding.dimension,
            normalized=True,
            created_at=created_at,
        )
        for item, embedding in zip(stored, result.embeddings, strict=True)
    )


def _span_region(span: EvidenceSpan) -> tuple[int, int, int, int] | None:
    """Pass the span's region of interest through to image cropping."""
    if span.region is None:
        return None
    return (span.region.x_min, span.region.y_min, span.region.x_max, span.region.y_max)


def _clip_media_object(
    tenant_id: TenantId,
    item: ResolvedEvidence,
    clip: MediaClip,
    *,
    created_at: datetime,
) -> MediaObject:
    digest = sha256(clip.content).hexdigest()
    return MediaObject(
        media_object_id=MediaObjectId(derive_stable_id("media_clip", tenant_id, digest)),
        tenant_id=tenant_id,
        kind=item.media_object.kind,
        uri=_clip_uri(item.media_object.uri, tenant_id, digest, clip.suffix),
        sha256=digest,
        size_bytes=len(clip.content),
        created_at=created_at,
        duration_ms=clip.end_ms - clip.start_ms,
        derived_from_media_object_id=item.media_object.media_object_id,
    )


def _clip_uri(source_uri: str, tenant_id: TenantId, digest: str, suffix: str) -> str:
    """Address the clip by content inside its source object's bucket."""
    bucket = urlsplit(source_uri).netloc
    return f"s3://{bucket}/tenants/{quote(tenant_id, safe='')}/clips/{digest}{suffix}"


def _unique_by_media_object(stored: tuple[_StoredClip, ...]) -> tuple[_StoredClip, ...]:
    """Identical clip content deduplicates onto one stored object."""
    seen: dict[MediaObjectId, _StoredClip] = {}
    for item in stored:
        seen.setdefault(item.media_object.media_object_id, item)
    return tuple(seen.values())


CLIP_KEY_PREFIX = "clips/"
# A clip is uploaded before the transaction that registers it, so a freshly
# written object is not evidence of an orphan. The grace period must exceed
# OBSERVATION_JOB_STALE_AFTER_SECONDS so no in-flight attempt is ever robbed.
CLIP_RECLAIM_GRACE_SECONDS = 3_600


@dataclass(frozen=True, slots=True)
class ClipReclaimSummary:
    """Content-free totals for one orphan-clip sweep."""

    tenant_id: TenantId
    scanned_count: int
    skipped_recent_count: int
    reclaimed_count: int


@operation_span("mindbridge.evidence_clips.reclaim")
async def reclaim_orphan_clips(
    tenant_id: TenantId,
    *,
    janitor: DerivedMediaJanitor,
    digests: ClipDigestStore,
    now: datetime | None = None,
    grace_seconds: int = CLIP_RECLAIM_GRACE_SECONDS,
    batch_size: int = 500,
) -> ClipReclaimSummary:
    """Delete stored clips the system of record never registered.

    Keys are content addressed, so an orphan is a digest the database does not
    know. Objects younger than the grace period are left alone because a Worker
    may still be between its upload and its commit.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if grace_seconds < 0:
        raise ValueError("grace_seconds must not be negative")
    cutoff = (now or utc_now()) - timedelta(seconds=grace_seconds)
    prefix = f"tenants/{quote(tenant_id, safe='')}/{CLIP_KEY_PREFIX}"
    listed = await janitor.list_media_keys(tenant_id, prefix)
    settled = tuple(key for key, modified_at in listed if modified_at <= cutoff)
    reclaimed = 0
    for offset in range(0, len(settled), batch_size):
        batch = settled[offset : offset + batch_size]
        known = await digests.list_known_clip_digests(
            tenant_id, tuple(_key_digest(key) for key in batch)
        )
        for key in batch:
            if _key_digest(key) not in known:
                await janitor.delete_media_key(tenant_id, key)
                reclaimed += 1
    set_current_span_attributes(
        {
            "mindbridge.tenant.id": tenant_id,
            "mindbridge.clip.scanned_count": len(listed),
            "mindbridge.clip.skipped_recent_count": len(listed) - len(settled),
            "mindbridge.clip.reclaimed_count": reclaimed,
        }
    )
    return ClipReclaimSummary(
        tenant_id=tenant_id,
        scanned_count=len(listed),
        skipped_recent_count=len(listed) - len(settled),
        reclaimed_count=reclaimed,
    )


def _key_digest(key: str) -> str:
    """Recover the content digest a clip key was named after."""
    return key.rsplit("/", 1)[-1].split(".", 1)[0]
