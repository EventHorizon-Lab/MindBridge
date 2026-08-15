"""Short-lived media access through the official S3 SDK."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit

from mindbridge.application.ports import PresignedMediaDownload
from mindbridge.core import MediaObject, ObjectStorageError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

_DEFAULT_URL_LIFETIME_SECONDS = 300
_MAX_URL_LIFETIME_SECONDS = 3_600


class InvalidMediaLocationError(ValueError):
    """Raised when a media URI escapes its configured tenant storage prefix."""


class S3MediaAccess:
    """Create tenant-scoped download URLs and delete immutable media."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        url_lifetime_seconds: int = _DEFAULT_URL_LIFETIME_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("bucket must not be empty")
        if not 1 <= url_lifetime_seconds <= _MAX_URL_LIFETIME_SECONDS:
            raise ValueError("url_lifetime_seconds must be between 1 and 3600")
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._url_lifetime_seconds = url_lifetime_seconds
        self._clock = clock or _utc_now
        self._client: S3Client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    async def create_presigned_download(
        self,
        media_object: MediaObject,
    ) -> PresignedMediaDownload:
        """Sign a GET only after validating bucket and tenant ownership."""
        object_key = self._tenant_object_key(media_object)
        expires_at = self._expires_at()
        download_url = await self._presign(
            "get_object",
            {"Bucket": self._bucket, "Key": object_key},
            "GET",
        )
        return PresignedMediaDownload(
            download_url=download_url,
            expires_at=expires_at,
        )

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
    ) -> str:
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return await asyncio.to_thread(
                self._client.generate_presigned_url,
                operation,
                Params=parameters,
                ExpiresIn=self._url_lifetime_seconds,
                HttpMethod=http_method,
            )
        except (BotoCoreError, ClientError) as error:
            raise ObjectStorageError(f"could not sign S3 {http_method} request") from error

    def _tenant_object_key(self, media_object: MediaObject) -> str:
        return tenant_s3_object_key(self._bucket, media_object.tenant_id, media_object.uri)

    def _expires_at(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now + timedelta(seconds=self._url_lifetime_seconds)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
