"""Tests for the one path that signs a write into tenant object storage."""

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from mindbridge.contracts import MAX_MEDIA_UPLOAD_BYTES
from mindbridge.infrastructure.s3 import (
    _UPLOAD_URL_LIFETIME_SECONDS,
    InvalidMediaLocationError,
    ObjectStorageEnvironment,
    S3MediaAccess,
    tenant_media_upload_uri,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
DIGEST = "a1" * 32
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
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


async def test_upload_is_signed_for_the_content_addressed_key_and_nothing_else() -> None:
    """The key is derived, the digest addresses it, and the length is signed with it."""
    access = S3MediaAccess(STORAGE, clock=lambda: NOW)

    upload = await access.create_presigned_upload(
        "tenant_01",
        sha256=DIGEST,
        suffix=".mp4",
        size_bytes=2_048,
    )
    query = parse_qs(urlsplit(upload.upload_url).query)

    assert upload.uri == (
        "s3://memory/tenants/tenant_01/media/"
        "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1.mp4"
    )
    assert upload.upload_url.startswith(
        "https://objects.example.test/memory/tenants/tenant_01/media/"
        "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1.mp4?"
    )
    # Signed, not merely requested: object storage refuses a PUT of any other length, which is
    # what makes the size bound something a client cannot walk past.
    assert query["X-Amz-SignedHeaders"] == ["content-length;host"]
    assert query["X-Amz-Expires"] == [str(_UPLOAD_URL_LIFETIME_SECONDS)]
    assert upload.expires_at == datetime(2026, 8, 11, 12, 15, tzinfo=timezone.utc)


async def test_upload_key_stays_in_the_tenant_prefix_a_hostile_tenant_id_tried_to_leave() -> None:
    """Nothing in a tenant identifier may become a path separator in the key it is signed for.

    The escape this rules out is the whole point of the endpoint: an identifier that reads as
    `../` is contained by percent-encoding it, and `tenant_s3_object_key` -- which recomputes
    the prefix the same way and refuses a key that does not start with it -- is what catches a
    derivation that ever stops encoding it.
    """
    access = S3MediaAccess(STORAGE, clock=lambda: NOW)

    upload = await access.create_presigned_upload(
        "tenant_01/../tenant_02",
        sha256=DIGEST,
        suffix=".mp4",
        size_bytes=1,
    )

    assert upload.uri == (
        "s3://memory/tenants/tenant_01%2F..%2Ftenant_02/media/"
        "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1.mp4"
    )


async def test_upload_refuses_to_sign_against_a_bucket_its_own_uri_does_not_name() -> None:
    """What gets signed is what `tenant_s3_object_key` returns, or nothing gets signed.

    A bucket configured with a prefix in it -- `memory/evidence` rather than `memory` -- is the
    reachable way the derived URI and the configured location come apart: a naive parse would
    sign `evidence/tenants/...` in a bucket called `memory`, which no read path can resolve,
    because every one of them resolves through this same check.
    """
    access = S3MediaAccess(
        ObjectStorageEnvironment(bucket="memory/evidence", endpoint_url=STORAGE.endpoint_url),
        clock=lambda: NOW,
    )

    with pytest.raises(InvalidMediaLocationError, match="configured bucket and tenant prefix"):
        await access.create_presigned_upload(
            "tenant_01",
            sha256=DIGEST,
            suffix=".mp4",
            size_bytes=1,
        )


@pytest.mark.parametrize(
    "suffix",
    [
        "/../../tenants/tenant_02/pwn.mp4",
        ".mp4/../../tenants/tenant_02/pwn",
        ".mp4%2F..%2Ftenant_02",
        "/etc/passwd",
        ".mp4?x=1",
        ".txt",
        ".",
    ],
)
async def test_upload_refuses_a_suffix_that_is_not_a_container_extension(suffix: str) -> None:
    """The extension is the only caller-supplied text in the key, so it is a closed set."""
    access = S3MediaAccess(STORAGE, clock=lambda: NOW)

    with pytest.raises(InvalidMediaLocationError, match="recognized media extension"):
        await access.create_presigned_upload(
            "tenant_01",
            sha256=DIGEST,
            suffix=suffix,
            size_bytes=1,
        )


@pytest.mark.parametrize(
    "sha256",
    ["../../tenants/tenant_02/pwn", "z" * 64, DIGEST[:-1], f"{DIGEST}/../x", ""],
)
def test_upload_refuses_a_digest_that_is_not_one(sha256: str) -> None:
    """A digest that is not 64 hex characters is caller text pointed at an object key."""
    with pytest.raises(InvalidMediaLocationError, match="sha-256 digest"):
        tenant_media_upload_uri("memory", "tenant_01", sha256=sha256, suffix=".mp4")


def test_case_does_not_split_one_object_into_two() -> None:
    """Two clients naming the same bytes differently must not write two objects."""
    assert tenant_media_upload_uri(
        "memory", "tenant_01", sha256=DIGEST.upper(), suffix=".MP4"
    ) == tenant_media_upload_uri("memory", "tenant_01", sha256=DIGEST, suffix=".mp4")


def test_an_unknown_container_is_addressed_by_its_digest_alone() -> None:
    """A key needs no extension; omitting one is how an unrecognized container stays storable."""
    assert tenant_media_upload_uri("memory", "tenant_01", sha256=DIGEST, suffix=None) == (
        "s3://memory/tenants/tenant_01/media/"
        "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1"
    )


@pytest.mark.parametrize("size_bytes", [0, -1, MAX_MEDIA_UPLOAD_BYTES + 1])
async def test_upload_refuses_to_sign_a_size_it_would_not_accept(size_bytes: int) -> None:
    """The signature is the grant, so the bound has to be applied before it is issued."""
    access = S3MediaAccess(STORAGE, clock=lambda: NOW)

    with pytest.raises(InvalidMediaLocationError, match="upload size"):
        await access.create_presigned_upload(
            "tenant_01",
            sha256=DIGEST,
            suffix=".mp4",
            size_bytes=size_bytes,
        )


async def test_upload_url_never_outlives_a_deployment_shortened_download_url() -> None:
    """A deployment that shortens signed access must not have upload quietly opt out of it."""
    access = S3MediaAccess(STORAGE, url_lifetime_seconds=120, clock=lambda: NOW)

    upload = await access.create_presigned_upload(
        "tenant_01",
        sha256=DIGEST,
        suffix=".mp4",
        size_bytes=1,
    )
    query = parse_qs(urlsplit(upload.upload_url).query)

    assert query["X-Amz-Expires"] == ["120"]
    assert upload.expires_at == datetime(2026, 8, 11, 12, 2, tzinfo=timezone.utc)
