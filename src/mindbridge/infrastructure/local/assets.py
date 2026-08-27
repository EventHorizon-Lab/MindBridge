"""Crash-safe content-addressed media files for the embedded runtime."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import stat
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

import httpx

from mindbridge.infrastructure.local.store import StoredAsset, validate_asset_name

_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_REDIRECTS = 3
_MEDIA_MODALITIES = frozenset({"image", "video", "audio"})
_MEDIA_TYPE = re.compile(r"[!#$&^_.+0-9A-Za-z-]+/[!#$&^_.+0-9A-Za-z-]+\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class _RemoteTarget:
    parsed: SplitResult
    host: str
    addresses: tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]


class AssetStoreError(RuntimeError):
    """Base error raised by local media materialization."""


class AssetTooLargeError(AssetStoreError):
    """Raised before an asset can exceed the configured local byte limit."""


class UnsafeAssetUrlError(AssetStoreError):
    """Raised when a URL fails the local SSRF policy."""


class AssetDownloadError(AssetStoreError):
    """Raised when an allowed remote asset cannot be downloaded."""


class AssetStore:
    """Materialize immutable media beneath ``data_dir/assets`` by SHA-256."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        allowed_url_hosts: Iterable[str] = (),
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.assets_dir = self.data_dir / "assets"
        self._staging_dir = self.data_dir / ".asset-staging"
        self.max_bytes = max_bytes
        self.allowed_url_hosts = frozenset(
            _canonical_allowed_host(host) for host in allowed_url_hosts
        )
        self.timeout_seconds = timeout_seconds
        self._transport = transport
        self._write_lock = threading.Lock()
        self._leases: Counter[str] = Counter()
        _ensure_data_directory(self.data_dir)
        self.assets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._staging_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _secure_directory(self.assets_dir)
        _secure_directory(self._staging_dir)
        self._remove_stale_parts()

    def materialize_bytes(
        self,
        content: bytes | bytearray | memoryview,
        *,
        modality: str,
        mime_type: str,
        name: str | None = None,
        created_at: datetime | None = None,
        lease: bool = False,
    ) -> StoredAsset:
        """Durably materialize inline bytes and return their immutable descriptor."""
        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise TypeError("content must be bytes-like")
        _validate_media(modality, mime_type)
        view = memoryview(content)
        return self._materialize_chunks(
            (view[offset : offset + _CHUNK_BYTES] for offset in range(0, len(view), _CHUNK_BYTES)),
            modality=modality,
            mime_type=mime_type,
            name=name,
            created_at=created_at,
            lease=lease,
        )

    def materialize_path(
        self,
        path: str | Path,
        *,
        modality: str,
        mime_type: str,
        name: str | None = None,
        created_at: datetime | None = None,
        lease: bool = False,
    ) -> StoredAsset:
        """Stream a regular local file into the CAS without following a final symlink."""
        _validate_media(modality, mime_type)
        source = Path(path).expanduser()
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(source, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("asset path must refer to a regular file")
            if info.st_size > self.max_bytes:
                raise AssetTooLargeError(
                    f"asset exceeds the configured {self.max_bytes}-byte limit"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return self._materialize_chunks(
                    _read_chunks(stream),
                    modality=modality,
                    mime_type=mime_type,
                    name=source.name if name is None else name,
                    created_at=created_at,
                    lease=lease,
                )
        finally:
            os.close(descriptor)

    def materialize_url(
        self,
        url: str,
        *,
        modality: str,
        mime_type: str,
        name: str | None = None,
        created_at: datetime | None = None,
        lease: bool = False,
    ) -> StoredAsset:
        """Download an explicitly allowed HTTPS URL after validating every redirect."""
        _validate_url_media_hint(modality, mime_type)
        current_url = url
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=self.timeout_seconds,
                transport=self._transport,
                trust_env=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                for redirect_count in range(_MAX_REDIRECTS + 1):
                    target = _validate_remote_url(current_url, self.allowed_url_hosts)
                    result = self._materialize_remote_target(
                        client,
                        target,
                        source_url=current_url,
                        allow_redirect=redirect_count < _MAX_REDIRECTS,
                        modality=modality,
                        mime_type=mime_type,
                        name=name,
                        created_at=created_at,
                        lease=lease,
                    )
                    if isinstance(result, StoredAsset):
                        return result
                    current_url = result
        except AssetStoreError:
            raise
        except (httpx.HTTPError, OSError) as error:
            raise AssetDownloadError("remote asset download failed") from error
        raise AssetDownloadError("remote asset redirect limit exceeded")

    def _materialize_remote_target(
        self,
        client: httpx.Client,
        target: _RemoteTarget,
        *,
        source_url: str,
        allow_redirect: bool,
        modality: str,
        mime_type: str,
        name: str | None,
        created_at: datetime | None,
        lease: bool,
    ) -> StoredAsset | str:
        connect_error: httpx.TransportError | None = None
        for address in target.addresses:
            try:
                with client.stream(
                    "GET",
                    _pinned_url(target, address),
                    headers={"Host": _host_header(target)},
                    extensions={"sni_hostname": target.host},
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if location is None or not allow_redirect:
                            raise AssetDownloadError("remote asset has an invalid redirect")
                        return cast(str, urljoin(source_url, location))
                    if response.status_code < 200 or response.status_code >= 300:
                        raise AssetDownloadError(
                            f"remote asset returned HTTP {response.status_code}"
                        )
                    declared_size = _content_length(response.headers.get("content-length"))
                    if declared_size == 0:
                        raise AssetDownloadError("remote asset is empty")
                    if declared_size is not None and declared_size > self.max_bytes:
                        raise AssetTooLargeError(
                            f"asset exceeds the configured {self.max_bytes}-byte limit"
                        )
                    actual_mime_type = _response_mime_type(
                        response.headers.get("content-type"),
                        modality=modality,
                        expected=mime_type,
                    )
                    return self._materialize_chunks(
                        response.iter_bytes(chunk_size=_CHUNK_BYTES),
                        modality=modality,
                        mime_type=actual_mime_type,
                        name=name,
                        created_at=created_at,
                        lease=lease,
                    )
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                connect_error = error
        raise AssetDownloadError("remote asset download failed") from connect_error

    def resolve(self, asset: StoredAsset) -> Path:
        """Resolve a descriptor to its regular CAS file without trusting stored paths."""
        expected = Path(asset.relative_path)
        if expected.is_absolute() or expected.parts != (
            "assets",
            asset.sha256[:2],
            asset.sha256,
        ):
            raise AssetStoreError("asset descriptor has an invalid relative path")
        path = self.data_dir / expected
        try:
            info = path.lstat()
        except FileNotFoundError:
            raise AssetStoreError(f"asset {asset.asset_id} is missing from local storage") from None
        if not stat.S_ISREG(info.st_mode) or info.st_size != asset.size_bytes:
            raise AssetStoreError(f"asset {asset.asset_id} is not a valid local file")
        return path

    def acquire(self, assets: Iterable[StoredAsset]) -> None:
        """Lease existing CAS files so concurrent garbage collection cannot remove them."""
        supplied = tuple(assets)
        with self._write_lock:
            for asset in supplied:
                self.resolve(asset)
            self._leases.update(asset.asset_id for asset in supplied)

    def release(self, assets: Iterable[StoredAsset]) -> None:
        """Release leases acquired while one public operation was using media."""
        supplied = tuple(assets)
        with self._write_lock:
            counts = Counter(asset.asset_id for asset in supplied)
            if any(self._leases[asset_id] < count for asset_id, count in counts.items()):
                raise AssetStoreError("asset lease accounting is inconsistent")
            self._leases.subtract(counts)
            self._leases += Counter()

    def delete_if_unleased(self, asset: StoredAsset) -> bool:
        """Remove one CAS file only when no active operation holds a lease."""
        with self._write_lock:
            if self._leases[asset.asset_id]:
                return False
            self._validate_delete_target(asset)
            self._delete_path(self.data_dir / asset.relative_path)
        return True

    def delete(self, asset: StoredAsset) -> None:
        """Idempotently remove an unreferenced CAS file described by SQLite."""
        with self._write_lock:
            self._validate_delete_target(asset)
            self._delete_path(self.data_dir / asset.relative_path)

    def _validate_delete_target(self, asset: StoredAsset) -> None:
        path = self.data_dir / asset.relative_path
        expected = self.assets_dir / asset.sha256[:2] / asset.sha256
        if path != expected:
            raise AssetStoreError("asset descriptor has an invalid relative path")

    def list_ids(self) -> tuple[str, ...]:
        """List strictly laid-out regular CAS objects without following links."""
        found = []
        try:
            for bucket in self.assets_dir.iterdir():
                info = bucket.lstat()
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or re.fullmatch(r"[0-9a-f]{2}", bucket.name) is None
                ):
                    raise AssetStoreError("asset store contains an invalid bucket")
                for path in bucket.iterdir():
                    info = path.lstat()
                    if (
                        not stat.S_ISREG(info.st_mode)
                        or _SHA256.fullmatch(path.name) is None
                        or not path.name.startswith(bucket.name)
                    ):
                        raise AssetStoreError("asset store contains an invalid object")
                    found.append(path.name)
        except AssetStoreError:
            raise
        except OSError as error:
            raise AssetStoreError("failed to scan the asset store") from error
        return tuple(sorted(found))

    def delete_id(self, asset_id: str) -> None:
        """Idempotently remove one validated CAS ID without SQLite metadata."""
        if not isinstance(asset_id, str) or _SHA256.fullmatch(asset_id) is None:
            raise AssetStoreError("asset id is invalid")
        with self._write_lock:
            self._delete_path(self.assets_dir / asset_id[:2] / asset_id)

    @staticmethod
    def _delete_path(path: Path) -> None:
        try:
            path.unlink()
            _fsync_directory(path.parent)
        except FileNotFoundError:
            return
        except OSError as error:
            raise AssetStoreError("failed to delete local media") from error

    def _materialize_chunks(
        self,
        chunks: Iterable[bytes | bytearray | memoryview],
        *,
        modality: str,
        mime_type: str,
        name: str | None,
        created_at: datetime | None,
        lease: bool,
    ) -> StoredAsset:
        try:
            return self._materialize_chunks_unchecked(
                chunks,
                modality=modality,
                mime_type=mime_type,
                name=name,
                created_at=created_at,
                lease=lease,
            )
        except AssetStoreError:
            raise
        except OSError as error:
            raise AssetStoreError("failed to materialize local media") from error

    def _materialize_chunks_unchecked(
        self,
        chunks: Iterable[bytes | bytearray | memoryview],
        *,
        modality: str,
        mime_type: str,
        name: str | None,
        created_at: datetime | None,
        lease: bool,
    ) -> StoredAsset:
        if name is not None:
            name = validate_asset_name(name)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._staging_dir,
            prefix="asset-",
            suffix=".part",
        )
        temporary_path = Path(temporary_name)
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise TypeError("asset stream yielded a non-bytes chunk")
                    size_bytes += len(chunk)
                    if size_bytes > self.max_bytes:
                        raise AssetTooLargeError(
                            f"asset exceeds the configured {self.max_bytes}-byte limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                if size_bytes == 0:
                    raise AssetStoreError("asset must not be empty")
                output.flush()
                os.fsync(output.fileno())
            digest_text = digest.hexdigest()
            destination = self.assets_dir / digest_text[:2] / digest_text
            with self._write_lock:
                bucket_existed = destination.parent.exists()
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _secure_directory(destination.parent)
                if not bucket_existed:
                    _fsync_directory(self.assets_dir)
                if destination.exists():
                    info = destination.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_size != size_bytes:
                        raise AssetStoreError(f"content-addressed asset {digest_text} is corrupt")
                    temporary_path.unlink()
                else:
                    os.chmod(temporary_path, 0o600)
                    os.replace(temporary_path, destination)
                    _fsync_directory(destination.parent)
                asset = StoredAsset(
                    asset_id=digest_text,
                    modality=modality,
                    mime_type=mime_type,
                    size_bytes=size_bytes,
                    sha256=digest_text,
                    relative_path=f"assets/{digest_text[:2]}/{digest_text}",
                    name=name,
                    created_at=created_at or datetime.now(timezone.utc),
                )
                if lease:
                    self._leases[asset.asset_id] += 1
                return asset
        finally:
            with suppress(FileNotFoundError):
                temporary_path.unlink()

    def _remove_stale_parts(self) -> None:
        for path in self._staging_dir.glob("asset-*.part"):
            with suppress(FileNotFoundError):
                path.unlink()


def _read_chunks(stream: BinaryIO) -> Iterable[bytes]:
    while chunk := stream.read(_CHUNK_BYTES):
        yield chunk


def _validate_media(modality: str, mime_type: str) -> None:
    if modality not in _MEDIA_MODALITIES:
        raise ValueError("asset modality must be image, video, or audio")
    if (
        not isinstance(mime_type, str)
        or mime_type != mime_type.strip().casefold()
        or _MEDIA_TYPE.fullmatch(mime_type) is None
        or mime_type.split("/", 1)[0] != modality
    ):
        raise ValueError("mime_type must be canonical and match asset modality")


def _validate_url_media_hint(modality: str, mime_type: str) -> None:
    if modality not in _MEDIA_MODALITIES:
        raise ValueError("asset modality must be image, video, or audio")
    if not isinstance(mime_type, str):
        raise ValueError("URL mime_type must be a concrete media type or media range")
    if mime_type.endswith("/*"):
        if mime_type != f"{modality}/*":
            raise ValueError("URL media type range must match asset modality")
        return
    _validate_media(modality, mime_type)


def _response_mime_type(
    value: str | None,
    *,
    modality: str,
    expected: str,
) -> str:
    if value is None:
        raise AssetDownloadError("remote asset response is missing Content-Type")
    actual = value.split(";", 1)[0].strip().casefold()
    try:
        _validate_media(modality, actual)
    except ValueError:
        raise AssetDownloadError(
            "remote asset Content-Type does not match its expected modality"
        ) from None
    if (expected.endswith("/*") and expected.split("/", 1)[0] != actual.split("/", 1)[0]) or (
        not expected.endswith("/*") and expected != actual
    ):
        raise AssetDownloadError("remote asset Content-Type does not match its expected type")
    return actual


def _canonical_allowed_host(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("allowed URL hosts must be non-empty and trimmed")
    if "://" in value or "/" in value or "@" in value or ":" in value:
        raise ValueError("allowed URL hosts must contain only a hostname")
    try:
        return value.rstrip(".").encode("idna").decode("ascii").casefold()
    except UnicodeError:
        raise ValueError("allowed URL host is invalid") from None


def _validate_remote_url(url: str, allowed_hosts: frozenset[str]) -> _RemoteTarget:
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except (TypeError, ValueError):
        raise UnsafeAssetUrlError("remote asset URL is invalid") from None
    if (
        parsed.scheme.casefold() != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise UnsafeAssetUrlError("remote assets require HTTPS without credentials or fragments")
    try:
        host = parsed.hostname.rstrip(".").encode("idna").decode("ascii").casefold()
    except UnicodeError:
        raise UnsafeAssetUrlError("remote asset host is invalid") from None
    if not allowed_hosts or host not in allowed_hosts:
        raise UnsafeAssetUrlError("remote asset host is not explicitly allowed")
    try:
        addresses = {
            ipaddress.ip_address(str(item[4][0]).split("%", 1)[0])
            for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError):
        raise UnsafeAssetUrlError("remote asset host could not be safely resolved") from None
    if not addresses or any(not address.is_global for address in addresses):
        raise UnsafeAssetUrlError("remote asset host resolves to a non-public address")
    return _RemoteTarget(
        parsed=parsed,
        host=host,
        addresses=tuple(sorted(addresses, key=lambda address: (address.version, address.packed))),
    )


def _pinned_url(
    target: _RemoteTarget,
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str:
    host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    port = target.parsed.port or 443
    authority = host if port == 443 else f"{host}:{port}"
    return urlunsplit(("https", authority, target.parsed.path, target.parsed.query, ""))


def _host_header(target: _RemoteTarget) -> str:
    port = target.parsed.port or 443
    return target.host if port == 443 else f"{target.host}:{port}"


def _content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise AssetDownloadError("remote asset has an invalid content length") from None
    if parsed < 0:
        raise AssetDownloadError("remote asset has an invalid content length")
    return parsed


def _secure_directory(path: Path) -> None:
    if os.name != "nt":
        os.chmod(path, 0o700)


def _ensure_data_directory(path: Path) -> None:
    created = False
    try:
        path.mkdir(mode=0o700, parents=True)
        created = True
    except FileExistsError:
        if not path.is_dir():
            raise
    if created or os.name != "nt":
        _secure_directory(path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
