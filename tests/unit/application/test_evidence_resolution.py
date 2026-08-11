"""Checks for shared evidence-to-media resolution."""

import asyncio
from datetime import datetime, timedelta, timezone

from mindbridge.application import PresignedMediaDownload, resolve_evidence_media
from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ObservationId,
    TenantId,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


class ConcurrencyRecordingSigner:
    """Makes concurrent signing observable without an object store."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.001)
        self.active -= 1
        return PresignedMediaDownload(
            download_url=f"https://objects.example.test/{media_object.media_object_id}",
            expires_at=NOW + timedelta(minutes=5),
        )


async def test_resolver_preserves_order_and_bounds_signing_concurrency() -> None:
    """A large recall cannot create unbounded object-store work."""
    media_objects = tuple(_media(index) for index in range(4))
    evidence = tuple(_evidence(index) for index in range(4))
    signer = ConcurrencyRecordingSigner()

    resolved = await resolve_evidence_media(
        evidence,
        media_objects,
        signer,
        max_concurrency=2,
    )

    assert tuple(item.evidence_span.evidence_id for item in resolved) == tuple(
        item.evidence_id for item in evidence
    )
    assert signer.max_active == 2


def _media(index: int) -> MediaObject:
    media_id = MediaObjectId(f"media_{index}")
    return MediaObject(
        media_object_id=media_id,
        tenant_id=TenantId("tenant_01"),
        kind=MediaKind.IMAGE,
        uri=f"s3://memory/tenants/tenant_01/{media_id}.jpg",
        sha256=f"{index:064x}",
        size_bytes=100,
        created_at=NOW,
    )


def _evidence(index: int) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=EvidenceId(f"evidence_{index}"),
        tenant_id=TenantId("tenant_01"),
        observation_id=ObservationId("observation_01"),
        media_object_id=MediaObjectId(f"media_{index}"),
        start_ms=0,
        end_ms=0,
        created_at=NOW,
    )
