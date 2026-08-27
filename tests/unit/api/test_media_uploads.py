"""Tests for the only route that lets a client put bytes into tenant storage."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.ports import MediaUploadSigner, PresignedMediaUpload
from mindbridge.contracts import MAX_MEDIA_UPLOAD_BYTES
from mindbridge.core import ObjectStorageError
from mindbridge.infrastructure.s3 import ObjectStorageEnvironment, S3MediaAccess

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
DIGEST = "a1" * 32
TENANT_API_KEY = "tenant-01-test-key-00000000000000"


@pytest.fixture
def aws_test_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Boto3 on its local signing path for the one test that signs for real."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@dataclass(frozen=True, slots=True)
class SignedUpload:
    """One call that reached the signer, recorded exactly as the route made it."""

    tenant_id: str
    sha256: str
    suffix: str | None
    size_bytes: int


class RecordingSigner:
    """A signer that reports what it was asked to sign, and refuses nothing itself.

    Refusing nothing is the point: it is what makes "the request was rejected" and "the request
    never became a signature" separable. A double that validated its own input could report a
    403 while still having been handed another tenant's key.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[SignedUpload] = []
        self._error = error

    async def create_presigned_upload(
        self,
        tenant_id: str,
        *,
        sha256: str,
        suffix: str | None,
        size_bytes: int,
    ) -> PresignedMediaUpload:
        self.calls.append(SignedUpload(tenant_id, sha256, suffix, size_bytes))
        if self._error is not None:
            raise self._error
        return PresignedMediaUpload(
            upload_url="https://objects.example.test/signed",
            uri=f"s3://memory/tenants/{tenant_id}/media/{sha256}{suffix or ''}",
            expires_at=NOW,
        )


def _client(signer: MediaUploadSigner) -> TestClient:
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator({"tenant_01": (TENANT_API_KEY,)}),
        media_uploads=signer,
    )
    return TestClient(app, headers={"Authorization": f"Bearer {TENANT_API_KEY}"})


def _body(**changes: object) -> dict[str, object]:
    return {"tenant_id": "tenant_01", "sha256": DIGEST, "size_bytes": 1_024, **changes}


def test_ticket_names_the_key_the_bytes_were_signed_into() -> None:
    signer = RecordingSigner()

    response = _client(signer).post("/v1/media/uploads", json=_body(suffix=".MP4"))

    body = response.json()
    assert response.status_code == 200
    assert body["uri"] == f"s3://memory/tenants/tenant_01/media/{DIGEST}.mp4"
    assert body["upload_url"] == "https://objects.example.test/signed"
    assert body["expires_at"] == "2026-08-11T12:00:00Z"
    assert body["trace_id"].startswith("trace_")
    # The suffix arrives lowercased, so `.MP4` and `.mp4` are one object rather than two.
    assert signer.calls == [SignedUpload("tenant_01", DIGEST, ".mp4", 1_024)]


@pytest.mark.usefixtures("aws_test_credentials")
def test_a_signed_key_is_inside_the_calling_tenants_own_prefix() -> None:
    """End to end through the real signer: what a caller receives points at its own prefix."""
    access = S3MediaAccess(
        ObjectStorageEnvironment(bucket="memory", endpoint_url="https://objects.example.test"),
        clock=lambda: NOW,
    )

    response = _client(access).post("/v1/media/uploads", json=_body(suffix=".mp4"))

    body = response.json()
    assert body["uri"] == f"s3://memory/tenants/tenant_01/media/{DIGEST}.mp4"
    assert body["upload_url"].startswith(
        f"https://objects.example.test/memory/tenants/tenant_01/media/{DIGEST}.mp4?"
    )


def test_a_caller_cannot_be_signed_into_another_tenants_prefix() -> None:
    """Naming another tenant is refused before anything is signed, not after.

    The assertion that matters is the second one. A 403 alone would also be produced by a
    server that signed the key first and then failed to return it, and a signature that was
    issued is a grant that exists whether or not its response was delivered.
    """
    signer = RecordingSigner()

    response = _client(signer).post("/v1/media/uploads", json=_body(tenant_id="other_tenant"))

    assert response.status_code == 403
    assert response.json()["code"] == "tenant_access_denied"
    assert signer.calls == []


def test_an_unauthenticated_caller_is_signed_nothing() -> None:
    signer = RecordingSigner()
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator({"tenant_01": (TENANT_API_KEY,)}),
        media_uploads=signer,
    )

    response = TestClient(app).post("/v1/media/uploads", json=_body())

    assert response.status_code == 401
    assert response.json()["code"] == "authentication_required"
    assert signer.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"suffix": "/../../tenants/other_tenant/pwn.mp4"},
        {"suffix": ".mp4/../../tenants/other_tenant/pwn"},
        {"suffix": ".mp4%2F..%2Fother_tenant"},
        {"suffix": ".txt"},
        {"suffix": ""},
        {"sha256": "../../tenants/other_tenant/pwn"},
        {"sha256": f"{DIGEST}/../x"},
        {"sha256": "z" * 64},
        {"size_bytes": 0},
        {"size_bytes": MAX_MEDIA_UPLOAD_BYTES + 1},
        {"key": "tenants/other_tenant/pwn.mp4"},
        {"uri": "s3://memory/tenants/other_tenant/pwn.mp4"},
    ],
)
def test_no_request_field_can_name_the_key_to_be_signed(changes: dict[str, object]) -> None:
    """Everything a caller can put in the body is either bounded or not a field at all."""
    signer = RecordingSigner()

    response = _client(signer).post("/v1/media/uploads", json=_body(**changes))

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_failed"
    assert signer.calls == []


def test_storage_being_unreachable_is_a_retryable_status() -> None:
    signer = RecordingSigner(ObjectStorageError("could not sign S3 PUT request"))

    response = _client(signer).post("/v1/media/uploads", json=_body())

    assert response.status_code == 503
    assert response.json()["code"] == "object_storage_unavailable"


def test_the_route_is_absent_when_no_deployment_signer_is_configured() -> None:
    """An MCP-only or storage-less build serves the rest of the surface unchanged."""
    app = build_app(
        cast(MemoryKernel, object()),
        authenticator=TenantApiKeyAuthenticator({"tenant_01": (TENANT_API_KEY,)}),
    )

    assert "/v1/media/uploads" not in app.openapi()["paths"]


def test_the_upload_operation_is_published_under_a_stable_id() -> None:
    paths = _client(RecordingSigner()).get("/openapi.json").json()["paths"]

    assert paths["/v1/media/uploads"]["post"]["operationId"] == "createMediaUpload"
