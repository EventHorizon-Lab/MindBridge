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
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from hashlib import sha256
from typing import TypeVar
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
    PresignedMediaDownload,
)
from mindbridge.core import (
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EvidenceClip,
    EvidenceSpan,
    MediaKind,
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
    cut_generation_proxy,
)
from mindbridge.telemetry import operation_span, set_current_span_attributes

_Budget = TypeVar("_Budget", float, int)


@dataclass(frozen=True, slots=True)
class ClipSampling:
    """Deployment-chosen sampling applied when a clip is cut."""

    frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS
    # Off only for a generator that reads the same storage the Worker does, where the encode
    # costs more than the transfer it removes.
    generation_proxy: bool = True


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
    embeddings = await _embed_stored_clips(tenant_id, stored, store, embedder, created_at, sampling)
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


@asynccontextmanager
async def generation_proxies(
    tenant_id: TenantId,
    evidence: tuple[ResolvedEvidence, ...],
    *,
    store: DerivedMediaStore,
    sampling: ClipSampling,
    scope: str,
    cut: Callable[[bytes, ClipRequest], MediaClip] = cut_generation_proxy,
    max_concurrency: int = 4,
) -> AsyncIterator[tuple[ResolvedEvidence, ...]]:
    """Lend sampled copies for the length of one model call, then take them back.

    A proxy is never registered, so nothing downstream can reach it: it is not in the write
    batch, it is not cited as provenance, and `forget()` would not find it. Scoping it to the
    call that needs it is what keeps that from meaning "kept forever" — including for an
    observation a tenant later asks to erase, and including on the attempt that raised.

    `scope` has to name that one call, not the observation: a claim reclaimed after the stale
    window leaves two attempts running at once, and each has to delete only its own copies.
    """
    # Collected as they are uploaded, not returned at the end: a source that fails to sign after
    # an earlier one uploaded would otherwise leave that copy behind, which is the leak this
    # scope exists to prevent.
    uploaded: list[MediaObject] = []
    try:
        yield await _derive_generation_proxies(
            tenant_id,
            evidence,
            store=store,
            sampling=sampling,
            scope=scope,
            cut=cut,
            max_concurrency=max_concurrency,
            uploaded=uploaded,
        )
    finally:
        outcomes = await asyncio.gather(
            *(store.delete_media(proxy) for proxy in uploaded), return_exceptions=True
        )
        failed = sum(isinstance(outcome, BaseException) for outcome in outcomes)
        if failed:
            # The whole justification for the scope is that nothing else can reach these, so a
            # silent cleanup failure is exactly the one that needs to be visible.
            set_current_span_attributes({"mindbridge.generation_proxy.undeleted": failed})


@operation_span("mindbridge.evidence_clips.generation_proxy")
async def _derive_generation_proxies(
    tenant_id: TenantId,
    evidence: tuple[ResolvedEvidence, ...],
    *,
    store: DerivedMediaStore,
    sampling: ClipSampling,
    scope: str,
    cut: Callable[[bytes, ClipRequest], MediaClip] = cut_generation_proxy,
    max_concurrency: int = 4,
    uploaded: list[MediaObject],
) -> tuple[ResolvedEvidence, ...]:
    """Point generation at the sampled copy of each video the model would produce anyway.

    Every generation request already carries the frame rate and pixel budget the model must
    apply, so handing over untouched source video makes the provider download frames it
    immediately discards. Each video source is cut once at that same budget and the sampled copy
    becomes what the model fetches. The span keeps citing its original object, non-video evidence
    is untouched, and a proxy that did not come out smaller is discarded rather than uploaded.
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not sampling.generation_proxy:
        return evidence
    groups: dict[MediaObjectId, list[ResolvedEvidence]] = {}
    for item in evidence:
        if item.media_object.kind is MediaKind.VIDEO:
            groups.setdefault(item.media_object.media_object_id, []).append(item)
    if not groups:
        return evidence

    semaphore = asyncio.Semaphore(max_concurrency)
    proxies = {
        media_object_id: download
        for media_object_id, download in zip(
            groups,
            await asyncio.gather(
                *(
                    _store_generation_proxy(
                        tenant_id,
                        items,
                        store=store,
                        sampling=sampling,
                        scope=scope,
                        cut=cut,
                        semaphore=semaphore,
                        uploaded=uploaded,
                    )
                    for items in groups.values()
                )
            ),
            strict=True,
        )
        if download is not None
    }
    set_current_span_attributes({"mindbridge.generation_proxy.count": len(proxies)})
    return tuple(
        item
        if (download := proxies.get(item.media_object.media_object_id)) is None
        else replace(
            item,
            media_url=download.download_url,
            media_url_expires_at=download.expires_at,
            sampled_frames_per_second=sampling.frames_per_second,
            sampled_max_pixels=sampling.max_pixels,
        )
        for item in evidence
    )


async def _store_generation_proxy(
    tenant_id: TenantId,
    items: list[ResolvedEvidence],
    *,
    store: DerivedMediaStore,
    sampling: ClipSampling,
    scope: str,
    cut: Callable[[bytes, ClipRequest], MediaClip],
    semaphore: asyncio.Semaphore,
    uploaded: list[MediaObject],
) -> PresignedMediaDownload | None:
    """Cut, store, and sign one sampled copy covering every span read from one source.

    Everything here is best-effort, storage included. Reading a source, writing a copy and
    signing it are work the observation did not do before, so a blip in any of them must degrade
    to the untouched source rather than fail an attempt whose perception would have succeeded.
    """
    source_object = items[0].media_object
    request = ClipRequest(
        kind=source_object.kind,
        start_ms=min(item.evidence_span.start_ms for item in items),
        end_ms=max(item.evidence_span.end_ms for item in items),
        frames_per_second=sampling.frames_per_second,
        max_pixels=sampling.max_pixels,
    )
    # The muxer ceiling is a frame count, so the frame rate decides it as much as the span does.
    # Checked before the read because the alternative is paying for a full source download and a
    # doomed encode on every observation.
    frames = (request.end_ms - request.start_ms) / 1_000 * request.frames_per_second
    if frames > MAX_PROXY_SAMPLED_FRAMES:
        set_current_span_attributes(
            {
                "mindbridge.generation_proxy.skipped": True,
                "mindbridge.generation_proxy.skipped_reason": "span_exceeds_frame_ceiling",
            }
        )
        return None
    async with semaphore:
        try:
            source = await store.read_media(source_object)
            try:
                proxy = await asyncio.to_thread(cut, source, request)
                # Not source_object.size_bytes: that is client-declared and never verified, so
                # one wrong number would silently disable the feature after paying for the read.
                shrank = len(proxy.content) < len(source)
            finally:
                del source
            if not shrank:
                set_current_span_attributes(
                    {
                        "mindbridge.generation_proxy.skipped": True,
                        "mindbridge.generation_proxy.skipped_reason": "no_smaller_than_source",
                    }
                )
                return None
            media_object = _clip_media_object(
                tenant_id,
                items[0],
                proxy,
                created_at=items[0].evidence_span.created_at,
                scope=scope,
            )
            await store.upload_media(media_object, proxy.content)
            uploaded.append(media_object)
            return await store.create_presigned_download(media_object)
        except Exception as error:
            # Deliberately broad, and covering storage as well as the encoder: the muxer raises a
            # bare ValueError for a span it will not interleave, PyAV raises IndexError for a
            # container it cannot read, and object storage raises for a transient outage. None of
            # them is worth losing an observation over.
            set_current_span_attributes(
                {
                    "mindbridge.generation_proxy.skipped": True,
                    "mindbridge.generation_proxy.skipped_reason": type(error).__name__,
                }
            )
            return None


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
    sampling: ClipSampling,
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
                            frames_per_second=_video_only(
                                sampling.frames_per_second, item.media_object.kind
                            ),
                            max_pixels=_video_only(sampling.max_pixels, item.media_object.kind),
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


def _video_only(value: _Budget, kind: MediaKind) -> _Budget | None:
    """Declare a sampling budget only where it means anything, since MediaPart rejects it off video."""
    return value if kind is MediaKind.VIDEO else None


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
    scope: str | None = None,
) -> MediaObject:
    """Address one derived clip by its content, or by content and scope when it is transient.

    A registered clip dedupes on content, which is what makes a retry overwrite the same object.
    A proxy must not: for a silent video it cuts bytes identical to the evidence clip of the same
    span, and deleting it at the end of its scope would then erase evidence a committed attempt
    still cites. `scope` also keeps two live attempts on one source off each other's object.
    """
    digest = sha256(clip.content).hexdigest()
    name = (
        digest
        if scope is None
        else f"{PROXY_KEY_INFIX}{sha256(f'{scope}:{digest}'.encode()).hexdigest()}"
    )
    return MediaObject(
        media_object_id=MediaObjectId(derive_stable_id("media_clip", tenant_id, name)),
        tenant_id=tenant_id,
        kind=item.media_object.kind,
        uri=_clip_uri(item.media_object.uri, tenant_id, name, clip.suffix),
        sha256=digest,
        size_bytes=len(clip.content),
        created_at=created_at,
        duration_ms=clip.end_ms - clip.start_ms,
        derived_from_media_object_id=item.media_object.media_object_id,
    )


def _clip_uri(source_uri: str, tenant_id: TenantId, name: str, suffix: str) -> str:
    """Address the clip by content inside its source object's bucket."""
    bucket = urlsplit(source_uri).netloc
    return f"s3://{bucket}/tenants/{quote(tenant_id, safe='')}/{CLIP_KEY_PREFIX}{name}{suffix}"


def _unique_by_media_object(stored: tuple[_StoredClip, ...]) -> tuple[_StoredClip, ...]:
    """Identical clip content deduplicates onto one stored object."""
    seen: dict[MediaObjectId, _StoredClip] = {}
    for item in stored:
        seen.setdefault(item.media_object.media_object_id, item)
    return tuple(seen.values())


CLIP_KEY_PREFIX = "clips/"
# Transient proxies share the swept prefix so a leaked one is still reclaimable, but never a
# registered clip's key.
PROXY_KEY_INFIX = "proxy-"
# The MP4 muxer refuses to interleave a sparse sampled video track with continuous audio past
# roughly this many frames. Checking it before reading the source turns a doomed full decode and
# encode into a decision.
MAX_PROXY_SAMPLED_FRAMES = 40
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
    dry_run: bool = False,
) -> ClipReclaimSummary:
    """Delete stored clips the system of record never registered.

    Keys are content addressed, so an orphan is a digest the database does not
    know. Objects younger than the grace period are left alone because a Worker
    may still be between its upload and its commit.

    `dry_run` counts what it would delete and deletes nothing, so an operator can see
    the size of an irreversible sweep before running it.
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
                if not dry_run:
                    await janitor.delete_media_key(tenant_id, key)
                reclaimed += 1
    set_current_span_attributes(
        {
            "mindbridge.tenant.id": tenant_id,
            "mindbridge.clip.scanned_count": len(listed),
            "mindbridge.clip.skipped_recent_count": len(listed) - len(settled),
            "mindbridge.clip.reclaimed_count": reclaimed,
            # Without this a preview is indistinguishable from a deletion in any dashboard
            # summing reclaimed_count, which would report storage that is still there.
            "mindbridge.clip.dry_run": dry_run,
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
