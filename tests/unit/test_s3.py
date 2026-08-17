"""Tests for tenant-safe S3 media access."""

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from botocore.exceptions import ClientError

from mindbridge.core import MediaKind, MediaObject, MediaObjectId, ObjectStorageError, TenantId
from mindbridge.infrastructure.s3 import (
    InvalidMediaLocationError,
    ObjectStorageEnvironment,
    S3MediaAccess,
    object_storage_from_environment,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
SHA256 = "00" * 32
STORAGE = ObjectStorageEnvironment(
    bucket="memory",
    endpoint_url="https://objects.example.test",
)


@pytest.fixture(autouse=True)
def aws_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Boto3 on its local signing path during unit tests."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    # MindBridge no longer passes a region; Boto3's own chain supplies it like any other AWS tool.
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


def media_object(*, tenant_id: str = "tenant_01", uri: str | None = None) -> MediaObject:
    """Build one valid immutable media reference."""
    return MediaObject(
        media_object_id=MediaObjectId("media_01"),
        tenant_id=TenantId(tenant_id),
        kind=MediaKind.VIDEO,
        uri=uri or f"s3://memory/tenants/{tenant_id}/media_01.mp4",
        sha256=SHA256,
        size_bytes=42,
        created_at=NOW,
        duration_ms=1_000,
    )


async def test_presigned_download_rejects_cross_tenant_location() -> None:
    """A valid media identifier cannot be used to sign another tenant's key."""
    access = S3MediaAccess(STORAGE, clock=lambda: NOW)
    foreign_location = media_object(uri="s3://memory/tenants/tenant_02/media_01.mp4")

    with pytest.raises(InvalidMediaLocationError, match="tenant prefix"):
        await access.create_presigned_download(foreign_location)


async def test_presigned_download_uses_get_and_configured_bucket() -> None:
    """Evidence access stays read-only and scoped to the configured object store."""
    access = S3MediaAccess(STORAGE, url_lifetime_seconds=120, clock=lambda: NOW)

    download = await access.create_presigned_download(media_object())
    query = parse_qs(urlsplit(download.download_url).query)

    assert download.download_url.startswith(
        "https://objects.example.test/memory/tenants/tenant_01/media_01.mp4?"
    )
    assert download.expires_at.isoformat() == "2026-08-11T12:02:00+00:00"
    assert query["X-Amz-Expires"] == ["120"]


async def test_delete_normalizes_s3_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    access = S3MediaAccess(STORAGE)

    def deny_delete(**_kwargs: object) -> None:
        raise ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "provider detail"}},
            "DeleteObject",
        )

    monkeypatch.setattr(access._client, "delete_object", deny_delete)

    with pytest.raises(ObjectStorageError, match="could not delete S3 evidence media"):
        await access.delete_media(media_object())


def test_object_storage_contract_is_read_once_for_every_process() -> None:
    """One reader keeps the API, Worker, and consolidation processes on the same variables."""
    storage = object_storage_from_environment(
        {
            "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
            "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL": "  ",
        }
    )

    assert storage == ObjectStorageEnvironment(bucket="memory", endpoint_url=None)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"bucket": " "}, "bucket"),
        ({"endpoint_url": " "}, "endpoint_url"),
    ],
)
def test_object_storage_contract_rejects_blank_values(
    changes: dict[str, str],
    message: str,
) -> None:
    """Blank storage settings must fail at startup, not when the first upload is signed."""
    with pytest.raises(ValueError, match=message):
        ObjectStorageEnvironment(**{"bucket": "memory", **changes})
