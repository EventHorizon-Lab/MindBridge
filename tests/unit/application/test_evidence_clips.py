"""Checks that clip derivation stores, dedupes, and embeds the right windows."""

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
        return b"source-bytes"

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
    space_reference = EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1")

    def __init__(self) -> None:
        self.tasks: list[EmbedTask] = []
        self.urls: list[str] = []

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.tasks.append(request.task)
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
                    ModelReference(model_id="jina-omni", revision="revision-01"),
                    EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
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
    ):
        pass

    assert [(item.frames_per_second, item.max_pixels) for item in requests] == [(0.5, 50_176)]
    assert [(item.start_ms, item.end_ms) for item in requests] == [(0, 30_000)]


async def test_generation_proxy_is_dropped_when_it_would_not_shrink_the_source() -> None:
    """Re-encoding media that is already at the sampling budget would cost bytes, not save them."""
    store = RecordingStore()

    async with generation_proxies(
        TENANT_ID,
        (_video_evidence("evidence_video_01", 0, 30_000),),
        store=store,
        sampling=ClipSampling(),
        cut=lambda source, request: MediaClip(
            content=b"x" * (VIDEO_SOURCE.size_bytes + 1),
            suffix=".mp4",
            start_ms=request.start_ms,
            end_ms=request.end_ms,
        ),
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
    ):
        pass

    assert store.reads == 1
    assert len(store.uploaded) == 1


def _video_evidence(evidence_id: str, start_ms: int, end_ms: int) -> ResolvedEvidence:
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(evidence_id),
            tenant_id=TENANT_ID,
            observation_id=ObservationId("observation_01"),
            media_object_id=VIDEO_SOURCE.media_object_id,
            start_ms=start_ms,
            end_ms=end_ms,
            created_at=NOW,
        ),
        media_object=VIDEO_SOURCE,
        media_url="https://objects.example.test/media_video_01.mp4",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _shrinking_cut(source: bytes, request: ClipRequest) -> MediaClip:
    return MediaClip(
        content=b"proxy:" + source,
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
        ):
            raise RuntimeError("perception exploded")

    assert store.deleted == tuple(store.uploaded)
