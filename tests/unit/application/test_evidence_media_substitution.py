"""The read path attaches the stored clip, not the full-resolution source.

The write path cuts one derived clip per evidence span at the deployment's sampling and
embeds that. Generation used to resolve the span's *source* object instead, so a populated
recall attached full-resolution originals with no ceiling. Measured against the evaluation
endpoint: 12.3k prompt tokens for one source clip against 1.65k for its stored clip, and
four sources exceed a 60 s gateway limit that streaming does not dodge.
"""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from mindbridge.application.evidence import read_resolved_evidence, resolve_evidence_media
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.core import (
    DomainInvariantError,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ObservationId,
    TenantId,
)

TENANT_ID = TenantId("tenant_evidence_substitution")
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class _Signer:
    """Signs each object to a distinguishable URL and records what it was asked for."""

    def __init__(self) -> None:
        self.signed: list[MediaObjectId] = []

    async def create_presigned_download(self, media_object: MediaObject) -> PresignedMediaDownload:
        self.signed.append(media_object.media_object_id)
        return PresignedMediaDownload(
            download_url=f"https://cdn.example/{media_object.media_object_id}?sig=x",
            expires_at=NOW,
        )


def _media(name: str, *, derived_from: str | None = None) -> MediaObject:
    return MediaObject(
        media_object_id=MediaObjectId(name),
        tenant_id=TENANT_ID,
        kind=MediaKind.VIDEO,
        uri=f"s3://media/{TENANT_ID}/{name}.mp4",
        sha256=f"{abs(hash(name)):064x}"[:64],
        size_bytes=1_000,
        created_at=NOW,
        duration_ms=30_000,
        derived_from_media_object_id=(
            MediaObjectId(derived_from) if derived_from is not None else None
        ),
    )


def _span(name: str, source: str, start_ms: int = 0) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=EvidenceId(name),
        tenant_id=TENANT_ID,
        observation_id=ObservationId(f"observation_{name}"),
        media_object_id=MediaObjectId(source),
        start_ms=start_ms,
        end_ms=start_ms + 4_000,
        created_at=NOW,
    )


def _resolved(span: EvidenceSpan, media: MediaObject, url: str) -> ResolvedEvidence:
    return ResolvedEvidence(
        evidence_span=span, media_object=media, media_url=url, media_url_expires_at=NOW
    )


async def test_a_span_with_a_stored_clip_is_signed_to_the_clip_not_the_source() -> None:
    source, clip = _media("source_a"), _media("clip_a", derived_from="source_a")
    span = _span("evidence_a", "source_a")
    signer = _Signer()

    resolved = await resolve_evidence_media(
        (span,), (source,), signer, clip_media={span.evidence_id: clip}
    )

    assert len(resolved) == 1
    # The bytes a model opens are the clip's.
    assert resolved[0].media_url == "https://cdn.example/clip_a?sig=x"
    # The identity a caller reads is still the source's.
    assert resolved[0].media_object.media_object_id == "source_a"
    assert resolved[0].media_object.uri == "s3://media/tenant_evidence_substitution/source_a.mp4"


async def test_a_span_with_no_stored_clip_still_resolves_to_its_source() -> None:
    source = _media("source_b")
    span = _span("evidence_b", "source_b")
    signer = _Signer()

    resolved = await resolve_evidence_media((span,), (source,), signer, clip_media={})

    assert resolved[0].media_url == "https://cdn.example/source_b?sig=x"
    assert signer.signed == [MediaObjectId("source_b")]


async def test_two_spans_of_one_recording_attach_both_clips() -> None:
    """The dedup key is the attached bytes; keying on the source dropped the second clip."""
    source = _media("source_c")
    first, second = _span("evidence_c1", "source_c", 0), _span("evidence_c2", "source_c", 8_000)
    clips = {
        first.evidence_id: _media("clip_c1", derived_from="source_c"),
        second.evidence_id: _media("clip_c2", derived_from="source_c"),
    }

    resolved = await resolve_evidence_media((first, second), (source,), _Signer(), clip_media=clips)
    parts = evidence_parts(resolved)

    attached = [part.url for part in parts if hasattr(part, "url")]
    assert attached == ["https://cdn.example/clip_c1?sig=x", "https://cdn.example/clip_c2?sig=x"]


def test_identical_bytes_are_attached_once() -> None:
    source = _media("source_d")
    span = _span("evidence_d", "source_d")
    url = "https://cdn.example/clip_d?sig=x"
    parts = evidence_parts((_resolved(span, source, url), _resolved(span, source, url)))
    assert sum(1 for part in parts if hasattr(part, "url")) == 1


def test_the_media_part_count_is_capped() -> None:
    source = _media("source_e")
    span = _span("evidence_e", "source_e")
    many = tuple(
        _resolved(span, source, f"https://cdn.example/clip_e{index}?sig=x") for index in range(40)
    )
    assert sum(1 for part in evidence_parts(many, max_media_parts=3) if hasattr(part, "url")) == 3
    # The default ceiling is what keeps a pathological fan-out under the gateway limit.
    assert sum(1 for part in evidence_parts(many) if hasattr(part, "url")) == 24


def test_a_negative_cap_is_rejected_rather_than_silently_dropping_everything() -> None:
    with pytest.raises(ValueError, match="max_media_parts"):
        evidence_parts((), max_media_parts=-1)


class _Reader:
    """The persistence side of the read path, answering exactly what it is asked for."""

    def __init__(self, span: EvidenceSpan, source: MediaObject, clip: MediaObject | None) -> None:
        self._span = span
        self._objects = {source.media_object_id: source}
        self._clip = clip
        if clip is not None:
            self._objects[clip.media_object_id] = clip

    async def read_evidence(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> tuple[EvidenceSpan, ...]:
        return tuple(self._span for evidence_id in evidence_ids)

    async def read_media_objects(
        self,
        tenant_id: TenantId,
        media_object_ids: tuple[MediaObjectId, ...],
    ) -> tuple[MediaObject, ...]:
        return tuple(self._objects[media_object_id] for media_object_id in media_object_ids)

    async def read_evidence_clip_media(
        self,
        tenant_id: TenantId,
        evidence_ids: tuple[EvidenceId, ...],
    ) -> dict[EvidenceId, MediaObject]:
        return {} if self._clip is None else {self._span.evidence_id: self._clip}


async def test_the_read_path_asks_the_store_for_the_clip_it_attaches() -> None:
    """The substitution is only worth anything if the one caller in the product does it.

    Every other test here hands `resolve_evidence_media` a clip map directly, so all of them
    stay green with the store lookup deleted. This one is the wiring.
    """
    source, clip = _media("source_f"), _media("clip_f", derived_from="source_f")
    span = _span("evidence_f", "source_f")

    resolved = await read_resolved_evidence(
        _Reader(span, source, clip), _Signer(), TENANT_ID, (span.evidence_id,)
    )

    assert resolved[0].media_url == "https://cdn.example/clip_f?sig=x"
    assert resolved[0].attached_media_object == clip


async def test_a_clip_belonging_to_another_tenant_is_refused() -> None:
    """The source lookup is validated on the way in; what stands in for it has to be too."""
    source = _media("source_g")
    span = _span("evidence_g", "source_g")
    foreign = replace(_media("clip_g", derived_from="source_g"), tenant_id=TenantId("tenant_other"))

    with pytest.raises(DomainInvariantError, match="attached media"):
        await resolve_evidence_media(
            (span,), (source,), _Signer(), clip_media={span.evidence_id: foreign}
        )


async def test_bytes_are_attached_while_they_are_about_the_span_and_not_after() -> None:
    """A span with no single covering clip falls back to its whole source, which may be hours.

    Attaching that is what the part ceiling exists to prevent, arriving one span at a time, so
    the fallback is attached only while the source is still about the span. Multi-window audio
    is the case: `cut_clips` splits at 30 s, so a 70 s span is stored as three ordinals and no
    one of them covers it.
    """
    span = _span("evidence_h", "source_h", 0)
    long_span = EvidenceSpan(
        evidence_id=EvidenceId("evidence_i"),
        tenant_id=TENANT_ID,
        observation_id=ObservationId("observation_i"),
        media_object_id=MediaObjectId("source_i"),
        start_ms=0,
        end_ms=70_000,
        created_at=NOW,
    )
    short_source = replace(_media("source_h"), duration_ms=span.end_ms)
    recording = replace(_media("source_i"), duration_ms=2 * 60 * 60 * 1_000)

    attached = await resolve_evidence_media(
        (span, long_span), (short_source, recording), _Signer(), clip_media={}
    )
    parts = evidence_parts(attached)

    assert [part.url for part in parts if hasattr(part, "url")] == [
        "https://cdn.example/source_h?sig=x"
    ]
    # Both spans keep their place in the prompt; only the two-hour one sends no bytes.
    assert attached[1].media_url == "https://cdn.example/source_i?sig=x"
