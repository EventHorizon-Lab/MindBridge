"""Focused durability tests for local content-addressed media."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import AssetStore, AssetStoreError, AssetTooLargeError


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
    local = store.materialize_path(source, modality="image", mime_type="image/png")

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
        store.materialize_bytes(b"12345", modality="audio", mime_type="audio/wav")
    assert tuple(staging.iterdir()) == ()
    assert tuple(store.assets_dir.rglob("*")) == ()

    with pytest.raises(AssetStoreError, match="empty"):
        store.materialize_bytes(b"", modality="audio", mime_type="audio/wav")

    with pytest.raises(ValueError, match="safe filename"):
        store.materialize_bytes(
            b"valid bytes",
            modality="image",
            mime_type="image/png",
            name="../escape.png",
        )
    assert not any(path.is_file() for path in store.assets_dir.rglob("*"))
