"""Focused safety and durability tests for local content-addressed media."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import httpx
import pytest

from mindbridge.infrastructure.local import (
    AssetDownloadError,
    AssetStore,
    AssetStoreError,
    AssetTooLargeError,
    UnsafeAssetUrlError,
)


def _public_dns(
    _host: str,
    port: int,
    *,
    type: socket.SocketKind,
) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]]:
    assert type is socket.SOCK_STREAM
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", port))]


def test_bytes_and_path_deduplicate_to_one_private_file(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    content = b"same immutable media"
    source = tmp_path / "photo.png"
    source.write_bytes(content)

    inline = store.materialize_bytes(
        content,
        modality="image",
        mime_type="image/png",
        name="inline.png",
    )
    local = store.materialize_path(
        source,
        modality="image",
        mime_type="image/png",
    )

    assert inline.asset_id == local.asset_id
    assert inline.relative_path == local.relative_path
    assert store.resolve(inline).read_bytes() == content
    assert len(tuple(store.assets_dir.rglob(inline.sha256))) == 1
    if os.name != "nt":
        assert stat.S_IMODE(store.resolve(inline).stat().st_mode) == 0o600
        assert stat.S_IMODE(store.assets_dir.stat().st_mode) == 0o700


def test_size_limit_cleans_staging_and_stale_crash_parts(tmp_path: Path) -> None:
    staging = tmp_path / ".asset-staging"
    staging.mkdir(parents=True)
    stale = staging / "asset-crash.part"
    stale.write_bytes(b"incomplete")
    store = AssetStore(tmp_path, max_bytes=4)

    assert not stale.exists()
    with pytest.raises(AssetTooLargeError, match="4-byte"):
        store.materialize_bytes(
            b"12345",
            modality="audio",
            mime_type="audio/wav",
        )
    assert tuple(staging.iterdir()) == ()
    assert tuple(store.assets_dir.rglob("*")) == ()

    with pytest.raises(AssetStoreError, match="empty"):
        store.materialize_bytes(
            b"",
            modality="audio",
            mime_type="audio/wav",
        )

    with pytest.raises(ValueError, match="safe filename"):
        store.materialize_bytes(
            b"valid bytes",
            modality="image",
            mime_type="image/png",
            name="../escape.png",
        )
    assert not any(path.is_file() for path in store.assets_dir.rglob("*"))


@pytest.mark.parametrize(
    "url",
    [
        "http://media.example/image.png",
        "https://user:secret@media.example/image.png",
        "https://not-allowed.example/image.png",
    ],
)
def test_url_requires_safe_https_and_an_explicit_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    url: str,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    store = AssetStore(tmp_path, allowed_url_hosts={"media.example"})

    with pytest.raises(UnsafeAssetUrlError):
        store.materialize_url(
            url,
            modality="image",
            mime_type="image/*",
        )


def test_url_rejects_private_dns_before_network_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = False

    def private_dns(
        _host: str,
        port: int,
        *,
        type: socket.SocketKind,
    ) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]]:
        assert type is socket.SOCK_STREAM
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=b"private")

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    store = AssetStore(
        tmp_path,
        allowed_url_hosts={"media.example"},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UnsafeAssetUrlError, match="non-public"):
        store.materialize_url(
            "https://media.example/private",
            modality="image",
            mime_type="image/*",
        )
    assert requested is False


def test_redirect_is_revalidated_and_cannot_reach_a_private_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def dns(
        host: str,
        port: int,
        *,
        type: socket.SocketKind,
    ) -> list[tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]]:
        address = "127.0.0.1" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, type, socket.IPPROTO_TCP, "", (address, port))]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "media.example"
        assert request.extensions["sni_hostname"] == "media.example"
        return httpx.Response(
            302,
            headers={"location": "https://internal.example/secret"},
        )

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    store = AssetStore(
        tmp_path,
        allowed_url_hosts={"media.example", "internal.example"},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(UnsafeAssetUrlError, match="non-public"):
        store.materialize_url(
            "https://media.example/start",
            modality="audio",
            mime_type="audio/*",
        )


def test_url_persists_actual_content_type_but_not_source_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["token"] == "do-not-persist"
        return httpx.Response(
            200,
            headers={"content-type": "image/webp; charset=binary"},
            content=b"remote image",
        )

    store = AssetStore(
        tmp_path,
        allowed_url_hosts={"media.example"},
        transport=httpx.MockTransport(handler),
    )
    asset = store.materialize_url(
        "https://media.example/photo?id=7&token=do-not-persist",
        modality="image",
        mime_type="image/*",
        name="photo.webp",
    )

    assert asset.mime_type == "image/webp"
    assert "media.example" not in repr(asset)
    assert "do-not-persist" not in repr(asset)
    assert store.resolve(asset).read_bytes() == b"remote image"
    store.delete(asset)
    store.delete(asset)
    assert not (tmp_path / asset.relative_path).exists()


def test_url_rejects_response_with_the_wrong_media_modality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    store = AssetStore(
        tmp_path,
        allowed_url_hosts={"media.example"},
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                content=b"not media",
            )
        ),
    )

    with pytest.raises(AssetDownloadError, match="Content-Type"):
        store.materialize_url(
            "https://media.example/not-media",
            modality="image",
            mime_type="image/*",
        )


def test_url_requires_an_exact_concrete_content_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    store = AssetStore(
        tmp_path,
        allowed_url_hosts={"media.example"},
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "image/jpeg"},
                content=b"jpeg bytes",
            )
        ),
    )

    with pytest.raises(AssetDownloadError, match="expected type"):
        store.materialize_url(
            "https://media.example/image",
            modality="image",
            mime_type="image/png",
        )
