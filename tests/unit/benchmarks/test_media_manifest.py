"""Prepared media validates structure before consumers use any files."""

import hashlib
import json
from pathlib import Path

import pytest

from mindbridge.benchmarks.eval_adapters import MediaResolver, load_media_manifest


@pytest.mark.parametrize(
    "task",
    [
        [],
        {"units": []},
        {"units": {"room": []}},
        {"units": {"room": [12]}},
        {"units": {"room": ["clip.mp4"]}},
        {"room": [{"path": "clip.mp4"}]},
        {"units": {"room": [{"text": "clip", "start_seconds": True}]}},
        {"units": {"room": [{"text": "clip", "end_seconds": -1}]}},
        {"units": {"room": [{"text": "clip", "path": 4}]}},
        {"units": {"room": [{"text": "clip", "source_id": []}]}},
        {"units": {}, "unavailable_units": {"room": ""}},
    ],
)
def test_load_manifest_rejects_malformed_nested_structure(tmp_path: Path, task: object) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "tasks": {"fixture": task}}))
    with pytest.raises(ValueError, match="media manifest"):
        load_media_manifest(path)


def test_manifest_preserves_source_hash_and_defers_unused_files(tmp_path: Path) -> None:
    units = {
        "selected": [{"text": "  prepared clip  ", "source_id": " clip ", "end_seconds": 9}],
        "unused": [{"path": "missing.mp4"}],
    }
    task = {"units": units, "unavailable_units": {"absent": "download failed"}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "tasks": {"fixture": task}}))
    manifest, directory = load_media_manifest(path)
    assert manifest is not None
    expected = hashlib.sha256(
        json.dumps(
            units, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert manifest.tasks["fixture"].input_sha256 == expected
    resolver = MediaResolver("fixture", None, manifest, directory)
    (part,) = resolver.parts("selected")
    assert (part.source_id, part.content, part.end_seconds) == ("clip", ("prepared clip",), 9)
    assert resolver.unavailable_units == {"absent": "download failed"}
    with pytest.raises(FileNotFoundError, match="prepared media does not exist"):
        resolver.parts("unused")


def test_manifest_path_uses_resolved_file_and_declared_source_id(tmp_path: Path) -> None:
    target = tmp_path / "original.mp4"
    target.touch()
    (tmp_path / "alias.mp4").symlink_to(target)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": {
                    "fixture": {"units": {"room": [{"path": "alias.mp4", "source_id": "clip"}]}}
                },
            }
        )
    )
    manifest, directory = load_media_manifest(path)
    (part,) = MediaResolver("fixture", None, manifest, directory).parts("room")
    assert (part.source_id, part.content) == ("clip", (target,))
