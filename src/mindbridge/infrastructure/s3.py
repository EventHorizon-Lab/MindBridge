"""Short-lived media access through the official S3 SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from string import hexdigits
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit

from mindbridge.application.ports import PresignedMediaDownload, PresignedMediaUpload
from mindbridge.configuration import (
    configuration_source,
    optional_environment_value,
    require_environment_value,
)
from mindbridge.contracts import MAX_MEDIA_UPLOAD_BYTES
from mindbridge.core import (
    MediaObject,
    MemoryIntegrityError,
    ObjectStorageError,
    media_kind_for_suffix,
    utc_now,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# A signed download is handed to a generator that fetches it itself, so it has to outlive the
# model call rather than the request that signs it. At 300s it expired mid-call and object
# storage answered with a permanent "could not download multimodal content" 400 instead of a
# retryable fetch. The default is set from the longest model request the tree can issue --
# OpenAiGenerator's own request_timeout_seconds default of 1800s -- plus the margin the fetch
# itself needs. It belongs here rather than at a call site because all three processes that
# sign media hand the result to a generator; a deployment configuring a request timeout above
# this has to raise the lifetime with it, and _MAX_URL_LIFETIME_SECONDS is the ceiling.
_DEFAULT_URL_LIFETIME_SECONDS = 2_100
_MAX_URL_LIFETIME_SECONDS = 3_600

# A signed upload has no such reader: it is answered to the client that is already holding the
# file and about to send it, so it can be short. S3 checks the expiry when the PUT arrives
# rather than while its body streams, so this bounds how long the grant can be sat on, not how
# long an upload may take.
_UPLOAD_URL_LIFETIME_SECONDS = 900
_SHA256_HEX_LENGTH = 64
MEDIA_KEY_PREFIX = "media/"
"""Where an uploaded original lives inside its tenant prefix, as `edge.capture` also writes it."""


@dataclass(frozen=True, slots=True)
class ObjectStorageEnvironment:
    """The one object storage contract every deployable process reads and validates."""

    bucket: str
    endpoint_url: str | None = None
    # Models fetch signed evidence from outside the deployment, so the name they must use is
    # not always the one the deployment should read and write through. Setting only
    # endpoint_url keeps both on the same address, exactly as before this field existed.
    public_endpoint_url: str | None = None

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise ValueError("object_storage bucket must not be empty")
        for name, value in (
            ("endpoint_url", self.endpoint_url),
            ("public_endpoint_url", self.public_endpoint_url),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"object_storage {name} must not be empty when provided")

    @property
    def signing_endpoint_url(self) -> str | None:
        """The address a presigned URL must name for a reader outside the deployment."""
        return self.public_endpoint_url or self.endpoint_url


def object_storage_from_environment(source: Mapping[str, str]) -> ObjectStorageEnvironment:
    """Read the storage variables once so no process can drift from the documented names."""
    return ObjectStorageEnvironment(
        bucket=require_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"),
        endpoint_url=optional_environment_value(source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"),
        public_endpoint_url=optional_environment_value(
            source, "MINDBRIDGE_OBJECT_STORAGE_PUBLIC_ENDPOINT_URL"
        ),
    )


class InvalidMediaLocationError(MemoryIntegrityError):
    """Raised when a media URI escapes its configured tenant storage prefix.

    An integrity failure rather than a plain ValueError: it is reached by resolving evidence
    a request already accepted, so it must answer in the same envelope every other stored-state
    inconsistency does instead of escaping as a bare 500.
    """


class S3MediaAccess:
    """Create tenant-scoped download URLs and delete immutable media."""

    def __init__(
        self,
        storage: ObjectStorageEnvironment,
        *,
        url_lifetime_seconds: int = _DEFAULT_URL_LIFETIME_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not 1 <= url_lifetime_seconds <= _MAX_URL_LIFETIME_SECONDS:
            raise ValueError("url_lifetime_seconds must be between 1 and 3600")
        import boto3
        from botocore.config import Config

        self._bucket = storage.bucket
        self._url_lifetime_seconds = url_lifetime_seconds
        self._clock = clock or utc_now
        # region_name is deliberately absent: Boto3's own chain (AWS_REGION, AWS_DEFAULT_REGION,
        # ~/.aws/config, instance metadata) already resolves it, and passing an explicit default
        # here silently overrode whatever the deployment had configured for every other AWS tool.
        config = Config(
            signature_version="s3v4",
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=storage.endpoint_url,
            config=config,
        )
        # SigV4 covers the Host header, so a URL a model fetches from outside the deployment has
        # to be signed against the address it will actually use. Only a deployment that sets a
        # separate public name pays for a second client.
        self._signing_client: S3Client = (
            self._client
            if storage.signing_endpoint_url == storage.endpoint_url
            else boto3.client("s3", endpoint_url=storage.signing_endpoint_url, config=config)
        )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> S3MediaAccess:
        """Build tenant-scoped media access from the documented storage contract."""
        return cls(object_storage_from_environment(configuration_source(environ)))

    async def close(self) -> None:
        """Release the SDK connection pools owned by this access adapter."""
        clients = (
            (self._client,)
            if self._signing_client is self._client
            else (
                self._client,
                self._signing_client,
            )
        )
        await asyncio.gather(*(asyncio.to_thread(client.close) for client in clients))

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        """Sign a GET only after validating bucket and tenant ownership."""
        object_key = self._tenant_object_key(media_object)
        expires_at = self._expires_at(self._url_lifetime_seconds)
        download_url = await self._presign(
            "get_object",
            {"Bucket": self._bucket, "Key": object_key},
            "GET",
            self._url_lifetime_seconds,
        )
        return PresignedMediaDownload(
            download_url=download_url,
            expires_at=expires_at,
        )

    async def create_presigned_upload(
        self,
        tenant_id: str,
        *,
        sha256: str,
        suffix: str | None,
        size_bytes: int,
    ) -> PresignedMediaUpload:
        """Sign a PUT for the one key these bytes belong at, and for no other.

        This is the only way bytes enter the deployment without its storage credentials, so the
        caller describes them and is told where they go rather than naming a key: the key is
        derived here by `tenant_media_upload_uri` and then re-read out of the finished URI by
        `tenant_s3_object_key`, the same check every read path applies, so what gets signed is
        what that check returns and nothing else. `ContentLength` is signed with it, which is
        what makes the size bound refusable by object storage rather than advisory -- a PUT of
        any other length fails the signature.
        """
        if not 1 <= size_bytes <= MAX_MEDIA_UPLOAD_BYTES:
            raise InvalidMediaLocationError(
                f"upload size must be between 1 and {MAX_MEDIA_UPLOAD_BYTES} bytes"
            )
        uri = tenant_media_upload_uri(self._bucket, tenant_id, sha256=sha256, suffix=suffix)
        object_key = tenant_s3_object_key(self._bucket, tenant_id, uri)
        lifetime_seconds = min(self._url_lifetime_seconds, _UPLOAD_URL_LIFETIME_SECONDS)
        expires_at = self._expires_at(lifetime_seconds)
        upload_url = await self._presign(
            "put_object",
            {"Bucket": self._bucket, "Key": object_key, "ContentLength": size_bytes},
            "PUT",
            lifetime_seconds,
        )
        return PresignedMediaUpload(upload_url=upload_url, uri=uri, expires_at=expires_at)

    async def read_media(self, media_object: MediaObject) -> bytes:
        """Fetch immutable bytes so a derived clip can be cut from them."""
        from botocore.exceptions import BotoCoreError, ClientError

        object_key = self._tenant_object_key(media_object)
        try:
            response = await asyncio.to_thread(
                self._client.get_object,
                Bucket=self._bucket,
                Key=object_key,
            )
            return await asyncio.to_thread(response["Body"].read)
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("could not read S3 evidence media") from error

    async def upload_media(self, media_object: MediaObject, content: bytes) -> None:
        """Write derived bytes to the object's own tenant-validated key.

        Keys are content addressed by the caller, so a retried job overwrites
        the same object instead of accumulating a new orphan per attempt.
        """
        from botocore.exceptions import BotoCoreError, ClientError

        object_key = self._tenant_object_key(media_object)
        try:
            await asyncio.to_thread(
                self._client.put_object,
                Bucket=self._bucket,
                Key=object_key,
                Body=content,
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("could not upload derived S3 evidence media") from error

    async def list_media_keys(
        self,
        tenant_id: str,
        prefix: str,
    ) -> tuple[tuple[str, datetime], ...]:
        """List keys and modification times under one tenant-scoped prefix."""
        from botocore.exceptions import BotoCoreError, ClientError

        tenant_prefix = f"tenants/{quote(tenant_id, safe='')}/"
        if not prefix.startswith(tenant_prefix):
            raise InvalidMediaLocationError("listing prefix must stay inside the tenant prefix")
        try:
            return await asyncio.to_thread(self._list_keys, prefix)
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("could not list S3 evidence media") from error

    def _list_keys(self, prefix: str) -> tuple[tuple[str, datetime], ...]:
        keys: list[tuple[str, datetime]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend((item["Key"], item["LastModified"]) for item in page.get("Contents", ()))
        return tuple(keys)

    async def delete_media_key(self, tenant_id: str, key: str) -> None:
        """Delete one object by key after re-checking its tenant prefix."""
        from botocore.exceptions import BotoCoreError, ClientError

        tenant_prefix = f"tenants/{quote(tenant_id, safe='')}/"
        if not key.startswith(tenant_prefix) or key.endswith("/"):
            raise InvalidMediaLocationError("deletion key must identify a tenant-owned object")
        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=self._bucket,
                Key=key,
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("could not delete derived S3 evidence media") from error

    async def delete_media(self, media_object: MediaObject) -> None:
        """Delete one tenant-scoped object; repeated S3 deletes remain successful."""
        from botocore.exceptions import BotoCoreError, ClientError

        object_key = self._tenant_object_key(media_object)
        try:
            await asyncio.to_thread(
                self._client.delete_object,
                Bucket=self._bucket,
                Key=object_key,
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError("could not delete S3 evidence media") from error

    async def _presign(
        self,
        operation: str,
        parameters: dict[str, object],
        http_method: str,
        lifetime_seconds: int,
    ) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return await asyncio.to_thread(
                self._signing_client.generate_presigned_url,
                operation,
                Params=parameters,
                ExpiresIn=lifetime_seconds,
                HttpMethod=http_method,
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError(f"could not sign S3 {http_method} request") from error

    def _tenant_object_key(self, media_object: MediaObject) -> str:
        return tenant_s3_object_key(self._bucket, media_object.tenant_id, media_object.uri)

    def _expires_at(self, lifetime_seconds: int) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now + timedelta(seconds=lifetime_seconds)


def tenant_media_upload_uri(
    bucket: str,
    tenant_id: str,
    *,
    sha256: str,
    suffix: str | None,
) -> str:
    """Name the one object these bytes belong at, inside this tenant's own prefix.

    The key is `tenants/<tenant_id>/media/<sha256><suffix>`, which is exactly what
    `edge.capture` writes, so the same file arriving from a device and from an SDK client is
    one object rather than two. Content addressing is not cosmetic here: a retried upload
    overwrites the same object instead of leaving an orphan per attempt, the reason
    `upload_media` gives for the same rule.

    Everything a caller contributes is bounded before it reaches the key. The digest is
    checked to be 64 hex characters, the extension is accepted only from the closed set the
    domain recognizes, and the tenant is percent-encoded -- so a `..`, an absolute key, or an
    encoded separator cannot appear in a key this returns. An unrecognized container is
    rejected rather than passed through: its suffix would be arbitrary caller text, and an
    object key needs no extension at all to be read back.
    """
    if len(sha256) != _SHA256_HEX_LENGTH or not all(char in hexdigits for char in sha256):
        raise InvalidMediaLocationError("upload key must be addressed by a sha-256 digest")
    if suffix is not None and media_kind_for_suffix(suffix) is None:
        raise InvalidMediaLocationError("upload suffix must be a recognized media extension")
    tenant_prefix = f"tenants/{quote(tenant_id, safe='')}/"
    # Both halves are lowered for one reason: the key is an identity, and the same bytes named
    # `.MP4` by one client and `.mp4` by another have to be one object rather than two.
    return (
        f"s3://{bucket}/{tenant_prefix}{MEDIA_KEY_PREFIX}{sha256.lower()}{(suffix or '').lower()}"
    )


def tenant_s3_object_key(bucket: str, tenant_id: str, uri: str) -> str:
    """Resolve an S3 URI only when it stays inside one configured tenant prefix."""
    location = urlsplit(uri)
    object_key = location.path.removeprefix("/")
    tenant_prefix = f"tenants/{quote(tenant_id, safe='')}/"
    if (
        location.scheme != "s3"
        or location.netloc != bucket
        or location.query
        or location.fragment
        or not object_key.startswith(tenant_prefix)
        or object_key.endswith("/")
    ):
        raise InvalidMediaLocationError(
            "media URI must identify an object in the configured bucket and tenant prefix"
        )
    return object_key
