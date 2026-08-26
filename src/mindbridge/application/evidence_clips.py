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
from math import ceil
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
    EmbeddingRecord,
    EvidenceClip,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    TenantId,
    derive_embedding_id,
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
from mindbridge.telemetry import log_fields, logger, operation_span, set_current_span_attributes

_Budget = TypeVar("_Budget", float, int)

_LOGGER = logger("mindbridge.application.evidence_clips")


# Past roughly this many sampled frames the proxy encode fails, on the flush that drains the
# encoder rather than on any single frame: `av.error.ValueError: [Errno 22] Invalid argument`.
# Checking it before reading the source turns a doomed full decode and encode into a decision.
#
# This was documented as the MP4 muxer refusing to interleave a sparse video track with
# continuous audio. That is wrong, and measurably so: a silent source cut with
# `include_audio=False` -- no audio anywhere in the pipeline -- fails at 45 frames exactly as an
# audiovisual one does, so dropping audio does not buy a single frame. Also ruled out by
# bisection: timestamp magnitude (45 frames fails across 45 s, 22.5 s and 9 s spans alike),
# holding frames past their decode container, and the picture type inherited from the source.
# It reproduces only for frames obtained by decoding a source -- 120 synthetic frames encode
# fine through the identical loop -- and the mechanism is not yet identified. The number stays
# where measurement put it; what changed is that it is no longer attributed to a cause that
# would suggest audio is the thing to give up.
MAX_PROXY_SAMPLED_FRAMES = 40
# The other end of the same bound, and the reason `_sampled_window` exists: Qwen3-VL's video
# processor merges frames in temporal pairs, so anything it is given has to carry at least two.
MIN_SAMPLED_FRAMES = 2
# How far that floor may reach in wall clock, which is not the same bound. Two sampling
# intervals is a length only once the rate is fixed, and the rate is a deployment setting:
# `{"frames_per_second": 0.2}` made the floor ten seconds, so every prompt-v11 event -- 2 to 5
# seconds of them -- was cut as a ten second clip that recall then attached and the answering
# model then read. Past this the clip samples its own window faster instead, which lands the
# same two frames on a span that still means what it says. The number is the floor the default
# 1 fps already produces: that is the widening evidence resolution was measured against.
MAX_SAMPLING_FLOOR_MS = 2_000


@dataclass(frozen=True, slots=True)
class ClipSampling:
    """Deployment-chosen sampling applied when a clip is cut."""

    frames_per_second: float = DEFAULT_VIDEO_FRAMES_PER_SECOND
    max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS
    image_max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS
    # Off only for a generator that reads the same storage the Worker does, where the encode
    # costs more than the transfer it removes.
    generation_proxy: bool = True
    # Off for a generator that cannot hear. Measured against the evaluation's endpoint on one
    # 15 s clip: `prompt_tokens` was 1009 whether or not the file carried its audio track, so
    # the track was never ingested, while the file was 336 KiB with it against 212 KiB without.
    # This does *not* raise the frame ceiling below -- see the note there.
    proxy_audio: bool = True


@dataclass(frozen=True, slots=True)
class DerivedEvidenceClips:
    """Everything one observation's clip derivation adds to the write batch."""

    media_objects: tuple[MediaObject, ...]
    clips: tuple[EvidenceClip, ...]
    embeddings: tuple[EmbeddingRecord, ...]


@dataclass(frozen=True, slots=True)
class _StoredProxy:
    """One uploaded generation proxy and the sampling it was really cut at."""

    download: PresignedMediaDownload
    frames_per_second: float


@dataclass(frozen=True, slots=True)
class _StoredClip:
    """One uploaded clip, holding metadata only so bytes can be released."""

    evidence: ResolvedEvidence
    ordinal: int
    start_ms: int
    end_ms: int
    media_object: MediaObject
    frames_per_second: float


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
        media_object_id: proxy
        for media_object_id, proxy in zip(
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
        if proxy is not None
    }
    set_current_span_attributes({"mindbridge.generation_proxy.count": len(proxies)})
    return tuple(
        item
        if (proxy := proxies.get(item.media_object.media_object_id)) is None
        else replace(
            item,
            media_url=proxy.download.download_url,
            media_url_expires_at=proxy.download.expires_at,
            # The rate the copy was cut at, not the one asked for: a proxy sampled faster than
            # the setting because its window was too short is a copy the generator has to be
            # told about, or it re-samples it back down to one frame.
            sampled_frames_per_second=proxy.frames_per_second,
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
) -> _StoredProxy | None:
    """Cut, store, and sign one sampled copy covering every span read from one source.

    Everything here is best-effort, storage included. Reading a source, writing a copy and
    signing it are work the observation did not do before, so a blip in any of them must degrade
    to the untouched source rather than fail an attempt whose perception would have succeeded.
    """
    source_object = items[0].media_object
    start_ms, end_ms, frames_per_second = _sampled_window(
        min(item.evidence_span.start_ms for item in items),
        max(item.evidence_span.end_ms for item in items),
        source_object.kind,
        sampling.frames_per_second,
    )
    request = ClipRequest(
        kind=source_object.kind,
        start_ms=start_ms,
        end_ms=end_ms,
        frames_per_second=frames_per_second,
        max_pixels=sampling.max_pixels,
        include_audio=sampling.proxy_audio,
    )
    # The ceiling is a frame count, so the frame rate decides it as much as the span does.
    # Checked before the read because the alternative is paying for a full source download and a
    # doomed encode on every observation.
    frames = (request.end_ms - request.start_ms) / 1_000 * request.frames_per_second
    if frames > MAX_PROXY_SAMPLED_FRAMES:
        _skipped_proxy(
            "span_exceeds_frame_ceiling",
            media_object_id=str(source_object.media_object_id),
            frames=round(frames),
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
                _skipped_proxy(
                    "no_smaller_than_source",
                    media_object_id=str(source_object.media_object_id),
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
            return _StoredProxy(
                await store.create_presigned_download(media_object),
                request.frames_per_second,
            )
        except Exception as error:
            # Deliberately broad, and covering storage as well as the encoder: the muxer raises a
            # bare ValueError for a span it will not interleave, PyAV raises IndexError for a
            # container it cannot read, and object storage raises for a transient outage. None of
            # them is worth losing an observation over.
            _skipped_proxy(
                type(error).__name__,
                media_object_id=str(source_object.media_object_id),
                error=str(error),
            )
            return None


def _skipped_proxy(reason: str, **fields: str | int) -> None:
    """Report one silently degraded generation proxy on both the span and the log.

    Perception still runs, on the untouched source, so nothing fails and the observation
    looks normal. A quiet downgrade in the media path is precisely the failure this project
    has paid for before, so it is a warning rather than an attribute only a collector sees.
    """
    set_current_span_attributes(
        {
            "mindbridge.generation_proxy.skipped": True,
            "mindbridge.generation_proxy.skipped_reason": reason,
        }
    )
    _LOGGER.warning(
        "generation proxy skipped, perceiving the untouched source",
        extra=log_fields(reason=reason, **fields),
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
            start_ms, end_ms, frames_per_second = _sampled_window(
                span.start_ms,
                span.end_ms,
                item.media_object.kind,
                sampling.frames_per_second,
            )
            request = ClipRequest(
                kind=item.media_object.kind,
                start_ms=start_ms,
                end_ms=end_ms,
                frames_per_second=frames_per_second,
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
                        frames_per_second=frames_per_second,
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
                                item.frames_per_second, item.media_object.kind
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
            embedding_id=derive_embedding_id(
                tenant_id,
                item.evidence.evidence_span.evidence_id,
                str(item.ordinal),
                model_id=embedding.model_reference.model_id,
                space_id=embedding.space_reference.space_id,
                task=EmbedTask.DOCUMENT.value,
            ),
            tenant_id=tenant_id,
            object_type=EmbeddedObjectType.EVIDENCE_SPAN,
            # The span, because `recall` reads this column back as an `EvidenceId` to load it.
            # Which clip of that span is `object_part`, and it is the same ordinal the ID above
            # hashes -- the vectors key needs it because a span cut into several clips is
            # several different sounds, not one object embedded twice.
            object_id=item.evidence.evidence_span.evidence_id,
            object_part=item.ordinal,
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


def _sampled_window(
    start_ms: int,
    end_ms: int,
    kind: MediaKind,
    frames_per_second: float,
) -> tuple[int, int, float]:
    """Widen a video span too short to sample twice, as the ceiling narrows one sampled too often.

    The sampler takes a frame at the start of the span and one every interval after it, so a span
    shorter than a single interval yields exactly one frame -- and a one-frame video is not a video
    to a Qwen3-VL encoder, whose patch embedding merges frames in temporal pairs. It raises
    `t:1 must be larger than temporal_factor:2`, which on 2026-08-21 destroyed the whole
    observation, perception included, on 72 occasions across 105 observations.

    One interval of window is not enough, which a real decode is the only way to find out: the
    sample instants are counted from the requested start, but each one is served by the first real
    frame at or after it, up to one source frame late. A 999 ms window at 1 fps therefore still
    came back with a single frame, and so did a window ending where the source's own last frame
    already had. Two intervals leave room for that lateness at any source frame rate at or above
    the sampling rate, which is every real recording.

    Widening runs backwards by preference: media starts at zero, so earlier is always inside the
    source, while later may be past its end -- and a window past the end samples nothing new.

    Two intervals of window is only two frames while the rate can fill it, and how long that
    window is in seconds is a deployment setting, so the reach is capped and the returned rate
    makes up whatever the cap took away. Both bounds are needed: the window is what covers a
    source whose own frames are sparser than the sampling, and the rate is what keeps a slow
    deployment setting from stretching a two second event into ten seconds of clip.
    """
    if kind is not MediaKind.VIDEO:
        return start_ms, end_ms, frames_per_second
    minimum_ms = min(MAX_SAMPLING_FLOOR_MS, ceil(1_000 * MIN_SAMPLED_FRAMES / frames_per_second))
    # One millisecond at the least, because the divisor below has to be positive: a point span
    # and a frame rate high enough to floor the minimum leave nothing else between them.
    window_ms = max(end_ms - start_ms, minimum_ms, 1)
    sampled_per_second = max(frames_per_second, 1_000 * MIN_SAMPLED_FRAMES / window_ms)
    if end_ms - start_ms >= minimum_ms:
        return start_ms, end_ms, sampled_per_second
    widened_start_ms = max(0, end_ms - minimum_ms)
    return widened_start_ms, widened_start_ms + minimum_ms, sampled_per_second


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
