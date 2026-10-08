"""Focused durability tests for local content-addressed media."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import AssetStore, AssetStoreError, AssetTooLargeError


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX named pipes")
def test_a_named_pipe_is_rejected_without_waiting_for_a_writer(tmp_path: Path) -> None:
    store = AssetStore(tmp_path)
    source = tmp_path / "input.wav"
    os.mkfifo(source)
    done = threading.Event()
    errors: list[Exception] = []

    def materialize() -> None:
        try:
            store.materialize_path(source, modality="audio", mime_type="audio/wav")
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    worker = threading.Thread(target=materialize)
    worker.start()
    promptly_rejected = done.wait(1)
    try:
        if not promptly_rejected:
            # Unblock the buggy reader so a regression fails without leaving a hung thread.
            descriptor = os.open(source, os.O_WRONLY | os.O_NONBLOCK)
            os.close(descriptor)
    finally:
        worker.join(5)
    assert promptly_rejected
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "regular file" in str(errors[0])


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
