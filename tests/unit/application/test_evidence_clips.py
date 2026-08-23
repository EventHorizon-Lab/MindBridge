"""Checks that clip derivation stores, dedupes, and embeds the right windows."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from mindbridge.application.evidence_clips import (
    ClipSampling,
    derive_evidence_clips,
    generation_proxies,
    reclaim_orphan_clips,
)
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.core import (
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingSpaceReference,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    ModelReference,
    ObjectStorageError,
    ObservationId,
    TenantId,
)
from mindbridge.media.clipping import ClipRequest, MediaClip, audio_windows
from mindbridge.models import Embedding, EmbedRequest, EmbedResult, EmbedTask, MediaPart

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")
SOURCE = MediaObject(
    media_object_id=MediaObjectId("media_01"),
    tenant_id=TENANT_ID,
    kind=MediaKind.AUDIO,
    uri="s3://memory/tenants/tenant_01/media_01.wav",
    sha256="a" * 64,
    size_bytes=4_096,
    created_at=NOW,
    duration_ms=70_000,
)


async def test_long_span_becomes_one_stored_clip_and_vector_per_window() -> None:
    """A span past the encoder window keeps its tail as extra clips and vectors."""
    store = RecordingStore()
    embedder = RecordingEmbedder()

    derived = await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 70_000),),
        store=store,
        embedder=embedder,
        sampling=ClipSampling(),
        created_at=NOW,
        cut=_stub_cut,
    )

    assert [(clip.start_ms, clip.end_ms) for clip in derived.clips] == [
        (0, 30_000),
        (30_000, 60_000),
        (60_000, 70_000),
    ]
    assert len(derived.media_objects) == 3
    assert len(store.uploaded) == 3
    assert len(derived.embeddings) == 3
    assert {embedding.object_id for embedding in derived.embeddings} == {"evidence_01"}
    assert len({embedding.embedding_id for embedding in derived.embeddings}) == 3
    assert all(
        embedding.object_type is EmbeddedObjectType.EVIDENCE_SPAN
        for embedding in derived.embeddings
    )
    assert embedder.tasks == [EmbedTask.DOCUMENT]


async def test_clip_provenance_and_content_addressed_key() -> None:
    store = RecordingStore()

    derived = await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 1_000),),
        store=store,
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=_stub_cut,
    )

    clip_media = derived.media_objects[0]
    assert clip_media.derived_from_media_object_id == MediaObjectId("media_01")
    assert clip_media.uri == (f"s3://memory/tenants/tenant_01/clips/{clip_media.sha256}.wav")
    assert clip_media.duration_ms == 1_000
    assert clip_media.kind is MediaKind.AUDIO


async def test_identical_spans_share_one_stored_object_but_keep_separate_vectors() -> None:
    """Content addressing dedupes storage without collapsing evidence identity."""
    store = RecordingStore()

    derived = await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 1_000), _evidence("evidence_02", 0, 1_000)),
        store=store,
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=_stub_cut,
    )

    assert len(store.uploaded) == 1
    assert len(derived.media_objects) == 1
    assert len(derived.clips) == 2
    assert {embedding.object_id for embedding in derived.embeddings} == {
        "evidence_01",
        "evidence_02",
    }


async def test_source_media_is_read_once_for_every_span_that_shares_it() -> None:
    store = RecordingStore()

    await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 1_000), _evidence("evidence_02", 2_000, 3_000)),
        store=store,
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=_stub_cut,
    )

    assert store.reads == 1


async def test_vector_count_mismatch_is_rejected() -> None:
    with pytest.raises(ModelOutputError, match="wrong evidence clip vector count"):
        await derive_evidence_clips(
            TENANT_ID,
            (_evidence("evidence_01", 0, 70_000),),
            store=RecordingStore(),
            embedder=ShortEmbedder(),
            sampling=ClipSampling(),
            created_at=NOW,
            cut=_stub_cut,
        )


async def test_no_evidence_derives_nothing() -> None:
    derived = await derive_evidence_clips(
        TENANT_ID,
        (),
        store=RecordingStore(),
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
    )

    assert (derived.media_objects, derived.clips, derived.embeddings) == ((), (), ())


def _evidence(evidence_id: str, start_ms: int, end_ms: int) -> ResolvedEvidence:
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(evidence_id),
            tenant_id=TENANT_ID,
            observation_id=ObservationId("observation_01"),
            media_object_id=SOURCE.media_object_id,
            start_ms=start_ms,
            end_ms=end_ms,
            created_at=NOW,
        ),
        media_object=SOURCE,
        media_url="https://objects.example.test/media_01.wav",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _stub_cut(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
    return tuple(
        MediaClip(
            content=b"clip:%d-%d:" % (start, end) + source,
            suffix=".wav",
            start_ms=start,
            end_ms=end,
        )
        for start, end in audio_windows(request.start_ms, request.end_ms)
    )


class RecordingStore:
    """Object storage double that counts reads and keeps uploaded clip bytes."""

    def __init__(self) -> None:
        self.reads = 0
        self.uploaded: dict[str, bytes] = {}
        self.deleted: tuple[str, ...] = ()

    async def read_media(self, media_object: MediaObject) -> bytes:
        self.reads += 1
        return b"source-bytes" * 128

    async def upload_media(self, media_object: MediaObject, content: bytes) -> None:
        self.uploaded[media_object.uri] = content

    async def delete_media(self, media_object: MediaObject) -> None:
        self.deleted = (*self.deleted, media_object.uri)

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/signed/{media_object.sha256}",
            expires_at=NOW + timedelta(minutes=5),
        )


class RecordingEmbedder:
    space_reference = EmbeddingSpaceReference(space_id="jina-v5")

    def __init__(self) -> None:
        self.tasks: list[EmbedTask] = []
        self.urls: list[str] = []
        self.sampling: tuple[tuple[float | None, int | None], ...] = ()

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.tasks.append(request.task)
        self.sampling = tuple(
            (part.frames_per_second, part.max_pixels)
            for input_value in request.inputs
            for part in input_value.parts
            if isinstance(part, MediaPart)
        )
        self.urls.extend(
            part.url
            for input_value in request.inputs
            for part in input_value.parts
            if isinstance(part, MediaPart)
        )
        return EmbedResult(
            tuple(
                Embedding(
                    (1.0, 0.0),
                    ModelReference(model_id="jina-omni"),
                    EmbeddingSpaceReference(space_id="jina-v5"),
                )
                for _ in request.inputs
            )
        )


class ShortEmbedder(RecordingEmbedder):
    async def embed(self, request: EmbedRequest) -> EmbedResult:
        result = await super().embed(request)
        return EmbedResult(result.embeddings[:1])


async def test_reclaim_deletes_only_clips_the_database_never_registered() -> None:
    """The orphan is the object with no row, not a row with no link."""
    registered, orphan = "1" * 64, "2" * 64
    janitor = RecordingJanitor(
        keys=(
            f"tenants/tenant_01/clips/{registered}.wav",
            f"tenants/tenant_01/clips/{orphan}.wav",
        )
    )

    summary = await reclaim_orphan_clips(
        TENANT_ID,
        janitor=janitor,
        digests=KnownDigests(frozenset({registered})),
        now=NOW,
    )

    assert janitor.listed_prefixes == ["tenants/tenant_01/clips/"]
    assert janitor.deleted == [f"tenants/tenant_01/clips/{orphan}.wav"]
    assert (summary.scanned_count, summary.reclaimed_count) == (2, 1)


async def test_reclaim_counts_the_orphans_a_dry_run_would_delete() -> None:
    """`--dry-run` has to report the size of an irreversible sweep without running it."""
    registered, orphan = "1" * 64, "2" * 64
    janitor = RecordingJanitor(
        keys=(
            f"tenants/tenant_01/clips/{registered}.wav",
            f"tenants/tenant_01/clips/{orphan}.wav",
        )
    )

    summary = await reclaim_orphan_clips(
        TENANT_ID,
        janitor=janitor,
        digests=KnownDigests(frozenset({registered})),
        now=NOW,
        dry_run=True,
    )

    assert janitor.deleted == []
    assert (summary.scanned_count, summary.reclaimed_count) == (2, 1)


async def test_reclaim_batches_digest_lookups() -> None:
    """A large prefix must not become one giant IN clause."""
    keys = tuple(f"tenants/tenant_01/clips/{index:064d}.wav" for index in range(5))
    janitor = RecordingJanitor(keys=keys)
    digests = KnownDigests(frozenset())

    summary = await reclaim_orphan_clips(
        TENANT_ID, janitor=janitor, digests=digests, batch_size=2, now=NOW
    )

    assert [len(batch) for batch in digests.requested] == [2, 2, 1]
    assert summary.reclaimed_count == 5


async def test_reclaim_leaves_an_empty_prefix_alone() -> None:
    janitor = RecordingJanitor(keys=())

    summary = await reclaim_orphan_clips(
        TENANT_ID, janitor=janitor, digests=KnownDigests(frozenset()), now=NOW
    )

    assert janitor.deleted == []
    assert (summary.scanned_count, summary.reclaimed_count) == (0, 0)


class RecordingJanitor:
    def __init__(self, *, keys: tuple[str, ...], age_seconds: float = 7_200.0) -> None:
        self._listed = tuple((key, NOW - timedelta(seconds=age_seconds)) for key in keys)
        self.listed_prefixes: list[str] = []
        self.deleted: list[str] = []

    async def list_media_keys(
        self,
        tenant_id: str,
        prefix: str,
    ) -> tuple[tuple[str, datetime], ...]:
        self.listed_prefixes.append(prefix)
        return self._listed

    async def delete_media_key(self, tenant_id: str, key: str) -> None:
        self.deleted.append(key)


class KnownDigests:
    def __init__(self, known: frozenset[str]) -> None:
        self._known = known
        self.requested: list[tuple[str, ...]] = []

    async def list_known_clip_digests(
        self,
        tenant_id: TenantId,
        digests: tuple[str, ...],
    ) -> frozenset[str]:
        self.requested.append(digests)
        return self._known & frozenset(digests)


async def test_reclaim_leaves_a_freshly_uploaded_clip_for_the_worker() -> None:
    """An upload still between its S3 write and its commit must not be robbed."""
    janitor = RecordingJanitor(
        keys=("tenants/tenant_01/clips/" + "3" * 64 + ".wav",),
        age_seconds=30.0,
    )

    summary = await reclaim_orphan_clips(
        TENANT_ID,
        janitor=janitor,
        digests=KnownDigests(frozenset()),
        now=NOW,
    )

    assert janitor.deleted == []
    assert (summary.scanned_count, summary.skipped_recent_count) == (1, 1)
    assert summary.reclaimed_count == 0


async def test_every_clip_is_uploaded_before_any_url_is_signed() -> None:
    """Signing during upload let the first URLs expire before the batched embed."""
    store = OrderRecordingStore()

    await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 70_000),),
        store=store,
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=_stub_cut,
    )

    assert store.calls.count("upload") == 3
    last_upload = max(index for index, call in enumerate(store.calls) if call == "upload")
    assert store.calls.index("sign") > last_upload


class OrderRecordingStore(RecordingStore):
    """Records the order of storage calls so signing cannot drift earlier."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def read_media(self, media_object: MediaObject) -> bytes:
        self.calls.append("read")
        return await super().read_media(media_object)

    async def upload_media(self, media_object: MediaObject, content: bytes) -> None:
        self.calls.append("upload")
        await super().upload_media(media_object, content)

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.calls.append("sign")
        return await super().create_presigned_download(media_object)


VIDEO_SOURCE = MediaObject(
    media_object_id=MediaObjectId("media_video_01"),
    tenant_id=TENANT_ID,
    kind=MediaKind.VIDEO,
    uri="s3://memory/tenants/tenant_01/media_video_01.mp4",
    sha256="b" * 64,
    size_bytes=14_000_000,
    created_at=NOW,
    duration_ms=30_000,
)


async def test_generation_proxy_hands_the_model_a_sampled_copy_of_the_video() -> None:
    """Perception already asks the provider for one frame per second, so shipping the untouched
    source only moves bytes the model discards on arrival."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert len(store.uploaded) == 1
        proxy_uri, proxy_bytes = next(iter(store.uploaded.items()))
        assert len(proxy_bytes) < VIDEO_SOURCE.size_bytes
        assert resolved[0].media_url != "https://objects.example.test/media_video_01.mp4"
        # Provenance is unchanged: the span still cites the source object the operator uploaded.
        assert resolved[0].media_object == VIDEO_SOURCE
        assert proxy_uri.startswith("s3://memory/tenants/tenant_01/clips/")


async def test_generation_proxy_is_cut_at_the_configured_sampling() -> None:
    """The proxy must carry exactly the frames the generator is told to sample, never fewer."""
    store = RecordingStore()
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> MediaClip:
        requests.append(request)
        return _shrinking_cut(source, request)

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(frames_per_second=0.5, max_pixels=50_176),
        cut=record,
        scope="job_01:1",
    ):
        pass

    assert [(item.frames_per_second, item.max_pixels) for item in requests] == [(0.5, 50_176)]
    assert [(item.start_ms, item.end_ms) for item in requests] == [(0, 30_000)]


async def test_generation_proxy_carries_the_deployment_s_audio_decision_to_the_cutter() -> None:
    """A deployment whose generator cannot hear turns `proxy_audio` off; without this the flag
    would be readable in configuration and have no effect on what is cut, which is the shape of
    bug that leaves a knob documented and dead."""
    seen: list[bool] = []

    def record(source: bytes, request: ClipRequest) -> MediaClip:
        seen.append(request.include_audio)
        return _shrinking_cut(source, request)

    for proxy_audio in (True, False):
        async with generation_proxies(
            TENANT_ID,
            (_video_evidence("evidence_video_01", 0, 30_000),),
            store=RecordingStore(),
            sampling=ClipSampling(proxy_audio=proxy_audio),
            cut=record,
            scope="job_01:1",
        ):
            pass

    assert seen == [True, False]


async def test_generation_proxy_is_dropped_when_it_would_not_shrink_the_source() -> None:
    """Re-encoding media that is already at the sampling budget would cost bytes, not save them."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=lambda source, request: MediaClip(
            content=b"x" * (len(b"source-bytes" * 128) + 1),
            suffix=".mp4",
            start_ms=request.start_ms,
            end_ms=request.end_ms,
        ),
        scope="job_01:1",
    ) as resolved:
        pass

    assert store.uploaded == {}
    assert resolved[0].media_url == "https://objects.example.test/media_video_01.mp4"


async def test_generation_proxy_leaves_non_video_evidence_untouched() -> None:
    """Audio and images are already small; a second encode is pure loss."""
    store = RecordingStore()
    original = _evidence("evidence_01", 0, 30_000)

    async with generation_proxies(
        TENANT_ID,
        (original,),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        pass

    assert resolved == (original,)
    assert store.reads == 0


async def test_generation_proxy_can_be_switched_off_for_a_colocated_model() -> None:
    """A deployment whose generator reads the same disk gains nothing and pays an encode."""
    store = RecordingStore()
    original = _video_evidence("evidence_video_01", 0, 30_000)

    async with generation_proxies(
        TENANT_ID,
        (original,),
        store=store,
        sampling=ClipSampling(generation_proxy=False),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        pass

    assert resolved == (original,)
    assert store.reads == 0


async def test_generation_proxy_reads_each_source_once_for_all_of_its_spans() -> None:
    """Two spans on one file must not download that file twice."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (
            _video_evidence("evidence_video_01", 0, 10_000),
            _video_evidence("evidence_video_02", 10_000, 30_000),
        ),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ):
        pass

    assert store.reads == 1
    assert len(store.uploaded) == 1


SECOND_VIDEO_SOURCE = MediaObject(
    media_object_id=MediaObjectId("media_video_02"),
    tenant_id=TENANT_ID,
    kind=MediaKind.VIDEO,
    uri="s3://memory/tenants/tenant_01/media_video_02.mp4",
    sha256="c" * 64,
    size_bytes=14_000_000,
    created_at=NOW,
    duration_ms=30_000,
)


def _video_evidence(
    evidence_id: str,
    start_ms: int,
    end_ms: int,
    *,
    media: MediaObject | None = None,
) -> ResolvedEvidence:
    source = media or VIDEO_SOURCE
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(evidence_id),
            tenant_id=TENANT_ID,
            observation_id=ObservationId("observation_01"),
            media_object_id=source.media_object_id,
            start_ms=start_ms,
            end_ms=end_ms,
            created_at=NOW,
        ),
        media_object=source,
        media_url=f"https://objects.example.test/{source.media_object_id}.mp4",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


class FailingSignerStore(RecordingStore):
    """Uploads fine, then refuses to sign after a set number of successes."""

    def __init__(self, *, fail_after: int) -> None:
        super().__init__()
        self._remaining = fail_after

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        if self._remaining <= 0:
            raise ObjectStorageError("could not sign S3 GET request")
        self._remaining -= 1
        return await super().create_presigned_download(media_object)


class UnreadableStore(RecordingStore):
    """Object storage that cannot serve the source the proxy wants to sample."""

    async def read_media(self, media_object: MediaObject) -> bytes:
        raise ObjectStorageError("could not read S3 evidence media")


def _shrinking_cut(source: bytes, request: ClipRequest) -> MediaClip:
    """Stand in for the encoder, and actually come out smaller than what it was given."""
    return MediaClip(
        content=b"proxy:%d-%d" % (request.start_ms, request.end_ms),
        suffix=".mp4",
        start_ms=request.start_ms,
        end_ms=request.end_ms,
    )


@pytest.mark.parametrize(
    "error",
    [
        DomainInvariantError("video span selected no frames from its source"),
        ValueError("[Errno 22] Invalid argument"),
        IndexError("tuple index out of range"),
        RuntimeError("decoder exploded"),
    ],
)
async def test_generation_proxy_falls_back_to_the_source_it_could_not_sample(
    error: Exception,
) -> None:
    """The proxy is an optimization, so every way the encoder can fail has to degrade to the
    source rather than turn a working observation into a permanently failed one. The muxer
    raises a bare ValueError for a span it will not interleave, and PyAV raises IndexError for a
    video object with no decodable video stream; neither is a DomainInvariantError."""
    store = RecordingStore()

    def refuse(source: bytes, request: ClipRequest) -> MediaClip:
        raise error

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=refuse,
        scope="job_01:1",
    ) as resolved:
        pass

    assert resolved[0].media_url == "https://objects.example.test/media_video_01.mp4"
    assert store.uploaded == {}


async def test_generation_proxy_reports_the_sampling_it_was_cut_at() -> None:
    """The proxy carries the deployment's sampling, but the model is told a budget from an
    unrelated variable, so the request has to state what the bytes actually are."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(frames_per_second=0.5, max_pixels=50_176),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert resolved[0].sampled_max_pixels == 50_176


async def test_source_evidence_declares_no_sampling_of_its_own() -> None:
    """Untouched source media must keep letting the generator apply its own budget."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(generation_proxy=False),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        pass

    assert resolved[0].sampled_frames_per_second is None
    assert resolved[0].sampled_max_pixels is None


async def test_generation_proxy_is_deleted_once_the_model_has_read_it() -> None:
    """The proxy is never registered, so nothing else can reach it: not the write batch, not
    provenance, and not forget(). Leaving it behind would keep a full re-encoded copy of the
    speech and picture of an observation a tenant asked to erase."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert resolved[0].media_url != "https://objects.example.test/media_video_01.mp4"
        assert len(store.uploaded) == 1

    assert store.deleted == tuple(store.uploaded)


async def test_a_failed_perception_still_deletes_the_proxy_it_uploaded() -> None:
    """A retried observation re-cuts the proxy, so an attempt that raised must not keep one."""
    store = RecordingStore()

    with pytest.raises(RuntimeError, match="perception exploded"):
        async with generation_proxies(
            TENANT_ID,
            (_video_evidence("evidence_video_01", 0, 30_000),),
            store=store,
            sampling=ClipSampling(),
            cut=_shrinking_cut,
            scope="job_01:1",
        ):
            raise RuntimeError("perception exploded")

    assert store.deleted == tuple(store.uploaded)


async def test_a_proxy_can_never_own_a_registered_clips_object_key() -> None:
    """For a silent video the proxy and the evidence clip cut identical bytes, and the key is a
    content digest, so they used to be the same object. Deleting the proxy then erased evidence a
    committed attempt still cites, and the orphan sweep would never flag it because the database
    knows that digest."""
    store = RecordingStore()
    identical = MediaClip(content=b"identical-bytes", suffix=".mp4", start_ms=0, end_ms=30_000)

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=lambda source, request: identical,
        scope="job_01:1",
    ):
        pass
    proxy_uri = next(iter(store.uploaded))

    clips = await derive_evidence_clips(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=lambda source, request: (identical,),
    )

    assert proxy_uri not in {item.uri for item in clips.media_objects}


async def test_two_attempts_on_one_source_do_not_share_a_proxy_object() -> None:
    """A claim reclaimed after the stale window leaves two attempts alive at once; whichever
    finishes first must not delete the copy the other is still handing to the model."""
    store = RecordingStore()

    for attempt in (1, 2):
        async with generation_proxies(
            TENANT_ID,
            (_video_evidence("evidence_video_01", 0, 30_000),),
            store=store,
            sampling=ClipSampling(),
            cut=_shrinking_cut,
            scope=f"job_01:{attempt}",
        ):
            pass

    # Identical bytes, so a content-only key would have been one object deleted twice.
    assert len(store.uploaded) == 2
    assert len(set(store.deleted)) == 2


async def test_a_proxy_uploaded_before_a_later_failure_is_still_reclaimed() -> None:
    """A copy is recorded for cleanup the moment it is uploaded, so a source that fails while
    being signed cannot leave it behind — derivation used to finish before the scope opened."""
    store = FailingSignerStore(fail_after=1)

    async with generation_proxies(
        TENANT_ID,
        (
            _video_evidence("evidence_video_01", 0, 30_000),
            _video_evidence("evidence_video_02", 0, 20_000, media=SECOND_VIDEO_SOURCE),
        ),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        # The source whose signing failed degrades to its own untouched media.
        assert sum(item.sampled_max_pixels is None for item in resolved) == 1

    assert len(store.uploaded) == 2
    assert set(store.deleted) == set(store.uploaded)


async def test_object_storage_trouble_degrades_to_the_source_instead_of_failing() -> None:
    """Reading, writing and signing a proxy are work the observation did not need before. An
    optimization must not be able to fail an attempt whose perception would have succeeded."""
    store = UnreadableStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert resolved[0].media_url == "https://objects.example.test/media_video_01.mp4"


async def test_the_shrink_check_measures_the_source_it_actually_read() -> None:
    """size_bytes is client-declared and never verified, so trusting it let one wrong number
    silently disable the feature after paying for the read and the encode."""
    store = RecordingStore()
    understated = replace(VIDEO_SOURCE, size_bytes=0)

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000, media=understated),),
        store=store,
        sampling=ClipSampling(),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert resolved[0].media_url != "https://objects.example.test/media_video_01.mp4"


async def test_a_span_too_long_to_encode_is_skipped_before_it_is_read() -> None:
    """The muxer ceiling is a frame count, so frames_per_second decides it as much as span
    length. Paying a full source read and a doomed encode per observation is the whole cost the
    proxy exists to remove."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(frames_per_second=4.0),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        assert resolved[0].media_url == "https://objects.example.test/media_video_01.mp4"
    assert store.reads == 0


async def test_the_embedder_is_told_the_sampling_its_clip_was_cut_at() -> None:
    """The clip was cut at the deployment's sampling, but the request used to carry the model
    plugin's own budget, so a served embedder resampled bytes that no longer had those frames."""
    store = RecordingStore()
    embedder = RecordingEmbedder()

    await derive_evidence_clips(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        embedder=embedder,
        sampling=ClipSampling(frames_per_second=2.0, max_pixels=50_176),
        created_at=NOW,
        cut=lambda source, request: (_shrinking_cut(source, request),),
    )

    assert embedder.sampling == ((2.0, 50_176),)


async def test_a_stored_audio_clip_declares_no_frame_rate() -> None:
    """Frame rate is meaningless off video and MediaPart rejects it, so the guard has to hold for
    the audio and image clips this same path stores."""
    store = RecordingStore()
    embedder = RecordingEmbedder()

    await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_01", 0, 30_000),),
        store=store,
        embedder=embedder,
        sampling=ClipSampling(frames_per_second=2.0, max_pixels=50_176),
        created_at=NOW,
        cut=_stub_cut,
    )

    assert embedder.sampling == ((None, None),)


async def test_a_span_too_short_to_sample_twice_is_widened_instead_of_failing() -> None:
    """The defect this closes: 72 clips across 105 observations, and each took its whole
    observation -- perception included -- down with it. At 1 fps a sub-second event was cut into a
    one-frame video, and a one-frame video is not a video to a Qwen3-VL encoder: it raised
    `t:1 must be larger than temporal_factor:2` at the embed call. Two sampling intervals is the
    shortest window a real decode actually yields two frames from."""
    store = RecordingStore()
    embedder = RecordingEmbedder()
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
        requests.append(request)
        return (_shrinking_cut(source, request),)

    derived = await derive_evidence_clips(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 4_000, 4_400),),
        store=store,
        embedder=embedder,
        sampling=ClipSampling(),
        created_at=NOW,
        cut=record,
    )

    assert [(item.start_ms, item.end_ms) for item in requests] == [(2_400, 4_400)]
    # What was cut is what is recorded, so the widening is visible rather than implied.
    assert [(clip.start_ms, clip.end_ms) for clip in derived.clips] == [(2_400, 4_400)]


async def test_the_floor_follows_the_configured_frame_rate() -> None:
    """The floor is two sampling intervals, so it moves with the deployment's frame rate rather
    than being a fixed number of milliseconds."""
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
        requests.append(request)
        return (_shrinking_cut(source, request),)

    for frames_per_second in (0.5, 4.0):
        await derive_evidence_clips(
            TENANT_ID,
            (_video_evidence("evidence_video_01", 8_000, 8_100),),
            store=RecordingStore(),
            embedder=RecordingEmbedder(),
            sampling=ClipSampling(frames_per_second=frames_per_second),
            created_at=NOW,
            cut=record,
        )

    assert [(item.start_ms, item.end_ms) for item in requests] == [(4_100, 8_100), (7_600, 8_100)]


async def test_a_span_at_the_start_of_a_source_is_widened_forwards() -> None:
    """Widening runs backwards by preference, but there is nothing behind the start of a file."""
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
        requests.append(request)
        return (_shrinking_cut(source, request),)

    await derive_evidence_clips(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 300),),
        store=RecordingStore(),
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=record,
    )

    assert [(item.start_ms, item.end_ms) for item in requests] == [(0, 2_000)]


async def test_a_span_long_enough_to_sample_is_cut_exactly_as_it_stands() -> None:
    """The floor must not move a span that never needed it."""
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
        requests.append(request)
        return (_shrinking_cut(source, request),)

    await derive_evidence_clips(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 4_000, 9_000),),
        store=RecordingStore(),
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=record,
    )

    assert [(item.start_ms, item.end_ms) for item in requests] == [(4_000, 9_000)]


async def test_a_short_audio_or_image_span_is_left_where_it_was() -> None:
    """Frames are a video idea. Audio has its own minimum and an image has no temporal factor at
    all, so widening either one would only blur what its vector means."""
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> tuple[MediaClip, ...]:
        requests.append(request)
        return (_shrinking_cut(source, request),)

    await derive_evidence_clips(
        TENANT_ID,
        (_evidence("evidence_audio_01", 4_000, 4_100),),
        store=RecordingStore(),
        embedder=RecordingEmbedder(),
        sampling=ClipSampling(),
        created_at=NOW,
        cut=record,
    )

    assert [(item.start_ms, item.end_ms) for item in requests] == [(4_000, 4_100)]


async def test_the_generation_proxy_gets_the_same_floor_as_the_stored_clip() -> None:
    """The generator is the same family of model as the encoder, so a one-frame proxy is refused
    the same way -- and there the loss lands on perception, which is the expensive half."""
    requests: list[ClipRequest] = []

    def record(source: bytes, request: ClipRequest) -> MediaClip:
        requests.append(request)
        return _shrinking_cut(source, request)

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 6_000, 6_200),),
        store=RecordingStore(),
        sampling=ClipSampling(),
        cut=record,
        scope="job_01:1",
    ):
        pass

    assert [(item.start_ms, item.end_ms) for item in requests] == [(4_200, 6_200)]


async def test_widening_cannot_push_a_proxy_past_the_frame_ceiling() -> None:
    """The two bounds have to compose: the ceiling is checked against the window that will
    actually be cut, not the span that was asked for."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 39_900, 40_000),),
        store=store,
        sampling=ClipSampling(frames_per_second=1.0),
        cut=_shrinking_cut,
        scope="job_01:1",
    ) as resolved:
        # 100 ms asked for, 2 s cut, still far below the 40-frame ceiling.
        assert resolved[0].media_url != "https://objects.example.test/media_video_01.mp4"
    assert store.reads == 1
