"""Checks that a generation request describes the media bytes it actually attaches."""

from datetime import datetime, timedelta, timezone

from mindbridge.application.capabilities import MediaPart
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines.evidence import evidence_parts
from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ObservationId,
    TenantId,
)

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")
SOURCE = MediaObject(
    media_object_id=MediaObjectId("media_01"),
    tenant_id=TENANT_ID,
    kind=MediaKind.VIDEO,
    uri="s3://memory/tenants/tenant_01/media_01.mp4",
    sha256="a" * 64,
    size_bytes=14_000_000,
    created_at=NOW,
    duration_ms=30_000,
)


def test_a_sampled_copy_states_the_budget_it_was_cut_at() -> None:
    """The generator plugin's own frame rate is a separate variable, so a request that omitted
    the proxy's budget would tell the model to resample an already sampled clip."""
    parts = evidence_parts((_evidence(frames_per_second=0.5, max_pixels=50_176),))

    media = next(part for part in parts if isinstance(part, MediaPart))
    assert media.frames_per_second == 0.5
    assert media.max_pixels == 50_176


def test_untouched_source_media_leaves_the_budget_to_the_generator() -> None:
    """Nothing resampled these bytes, so the plugin's configured budget still applies."""
    parts = evidence_parts((_evidence(),))

    media = next(part for part in parts if isinstance(part, MediaPart))
    assert media.frames_per_second is None
    assert media.max_pixels is None


def _evidence(
    *,
    frames_per_second: float | None = None,
    max_pixels: int | None = None,
) -> ResolvedEvidence:
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId("evidence_01"),
            tenant_id=TENANT_ID,
            observation_id=ObservationId("observation_01"),
            media_object_id=SOURCE.media_object_id,
            start_ms=0,
            end_ms=30_000,
            created_at=NOW,
        ),
        media_object=SOURCE,
        media_url="https://objects.example.test/signed",
        media_url_expires_at=NOW + timedelta(minutes=5),
        sampled_frames_per_second=frames_per_second,
        sampled_max_pixels=max_pixels,
    )
