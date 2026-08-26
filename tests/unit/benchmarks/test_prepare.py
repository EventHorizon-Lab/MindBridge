"""Checks for the prepared-media producers.

Staging reaches an object store and cutting reaches a video encoder, so both are exercised
against doubles or a synthetic source rather than the real corpus. What is checked is what the
manifests have to satisfy: the tenant a URI must sit under, the split's disjoint boundaries, and
the runner's own acceptance of the result.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest

from mindbridge.benchmarks.prepare import (
    SEGMENT_SECONDS,
    STAGED_AT,
    PrepareRequest,
    _key_component,
    _Staging,
    _within,
    prepare_mem_gallery,
    video_segments,
)
from mindbridge.core import MediaKind

pytest.importorskip("av", reason="prepared media is cut with the media extra's decoders")


class _RecordingClient:
    """An S3 double that keeps what it was asked to write."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body


def test_a_staged_object_lands_under_the_tenant_that_will_read_it() -> None:
    """`tenant_s3_object_key` refuses any other prefix, which is why a manifest is per run."""
    client = _RecordingClient()
    staging = _Staging("mindbridge-media", client)

    media_object = staging.stage(
        tenant_id="benchmark_m3_bedroom_01_run-01",
        key="m3/bedroom_01/0.mp4",
        content=b"clip bytes",
        kind=MediaKind.VIDEO,
        media_object_id="m3_bedroom_01_0",
        duration_ms=30_000,
    )

    assert media_object.uri == (
        "s3://mindbridge-media/tenants/benchmark_m3_bedroom_01_run-01/m3/bedroom_01/0.mp4"
    )
    assert "mindbridge-media/tenants/benchmark_m3_bedroom_01_run-01/m3/bedroom_01/0.mp4" in (
        client.objects
    )
    assert media_object.size_bytes == len(b"clip bytes")


def test_two_preparations_of_one_clip_produce_the_same_manifest() -> None:
    """A run manifest pins the prepared manifest's digest, so a wall clock in it would churn."""
    staging = _Staging("bucket", _RecordingClient())

    first = staging.stage(
        tenant_id="tenant_01",
        key="a.jpg",
        content=b"image",
        kind=MediaKind.IMAGE,
        media_object_id="IMG:1",
    )
    second = staging.stage(
        tenant_id="tenant_01",
        key="a.jpg",
        content=b"image",
        kind=MediaKind.IMAGE,
        media_object_id="IMG:1",
    )

    assert first == second
    assert first.created_at == STAGED_AT


def test_the_split_is_disjoint_and_its_durations_are_the_ones_it_declares(
    tmp_path: Path,
) -> None:
    """A closed span shares its last second with the next segment.

    That is the defect this checks for: a benchmark that withholds clips recorded after a
    question's timestamp would have had a second of the future in the clip before it, and the
    manifest would have declared 30 seconds for 31 seconds of content.
    """
    import av

    source = _synthetic_video(tmp_path, seconds=SEGMENT_SECONDS * 2 + 5)

    segments = list(video_segments(source))

    assert [duration for _, duration, _ in segments] == [30_000, 30_000, 5_000]
    assert [index for index, _, _ in segments] == [0, 1, 2]
    for _, declared_ms, content in segments:
        # Probed rather than trusted: the declared duration is what the manifest claims, and the
        # bug this guards let the two disagree by a full second.
        probe = cast(Any, av.open(io.BytesIO(content)))
        with probe:
            assert int(probe.duration / 1_000) == declared_ms


def test_a_limit_stops_the_cut_early_rather_than_cutting_the_whole_source(
    tmp_path: Path,
) -> None:
    """Cutting is the expensive half of preparing, so `--limit` has to bound it too."""
    source = _synthetic_video(tmp_path, seconds=SEGMENT_SECONDS * 2 + 5)

    assert len(list(video_segments(source, limit=1))) == 1


def test_mem_gallery_stages_every_image_its_topics_reference_and_nothing_else(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner refuses a manifest missing a referenced key, and the release holds spares."""
    from mindbridge.benchmarks import prepare
    from mindbridge.benchmarks.mem_gallery_runner import (
        load_prepared_mem_gallery,
        validate_mem_gallery_images,
    )

    dialog, referenced, unreferenced = _mem_gallery_release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare, "staging", lambda: _Staging("bucket", client))
    manifest = tmp_path / "prepared.json"

    prepare_mem_gallery(
        PrepareRequest(
            argv=_mem_gallery_argv(dialog, manifest, tmp_path),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    prepared = load_prepared_mem_gallery(manifest)
    keys = {image.image_key for image in prepared.images}
    assert referenced <= keys
    assert unreferenced not in keys
    # The manifest is only correct if the runner's own validation accepts it.
    from mindbridge.benchmarks.mem_gallery import load_mem_gallery

    validate_mem_gallery_images(load_mem_gallery(dialog), prepared)
    assert all("tenants/benchmark_mem_gallery_topic_01_prep-01/" in key for key in client.objects)


def _mem_gallery_argv(dialog: Path, manifest: Path, root: Path) -> tuple[str, ...]:
    return (
        "--dataset",
        str(dialog),
        "--prepared-images",
        str(manifest),
        "--output",
        str(root / "out.json"),
        "--api-base-url",
        "http://localhost:8000",
        "--deployment-config",
        str(root / "deployment.json"),
        "--run-id",
        "prep-01",
    )


def _synthetic_video(directory: Path, *, seconds: int) -> Path:
    """Encode a small source of a known length, so segment boundaries are checkable."""
    import av
    import numpy

    path = directory / "source.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.thread_count, stream.thread_type = 1, "NONE"
        for index in range(5 * seconds):
            array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return path


def _mem_gallery_release(root: Path) -> tuple[Path, set[str], str]:
    """Write the smallest release the Mem-Gallery adapter reads, plus one unreferenced image."""
    import json

    dialog = root / "mem-gallery" / "data" / "dialog"
    images = root / "mem-gallery" / "data" / "image" / "topic_01"
    dialog.mkdir(parents=True)
    images.mkdir(parents=True)
    for name in ("D1_IMG_001.jpg", "QA_IMG_001.jpg", "D9_IMG_009.jpg"):
        (images / name).write_bytes(b"\xff\xd8\xff" + name.encode())
    referenced = {
        "../image/topic_01/D1_IMG_001.jpg",
        "../image/topic_01/QA_IMG_001.jpg",
    }
    (dialog / "topic_01.json").write_text(
        json.dumps(_mem_gallery_topic()),
        encoding="utf-8",
    )
    return dialog, referenced, "../image/topic_01/D9_IMG_009.jpg"


def _mem_gallery_topic() -> dict[str, Any]:
    """The release's own key names, which are not the adapter's field names."""
    return {
        "character_profile": {
            "name": "Ada",
            "persona_summary": "an engineer",
            "traits": ["curious"],
            "conversation_style": "direct",
        },
        "multi_session_dialogues": [
            {
                "session_id": "S1",
                "date": "2026-01-01",
                "dialogues": [
                    {
                        "round": "1",
                        "user": "Look at this",
                        "assistant": "I see it",
                        "image_id": ["D1:IMG_001"],
                        "input_image": ["../image/topic_01/D1_IMG_001.jpg"],
                        "image_caption": ["a robot"],
                    }
                ],
            }
        ],
        "human-annotated QAs": [
            {
                "point": "VR",
                "question": "What was in the picture?",
                "answer": "a robot",
                "session_id": ["S1"],
                "clue": ["1"],
                "question_image": "../image/topic_01/QA_IMG_001.jpg",
                "image_caption": "a robot again",
            }
        ],
    }


@pytest.mark.parametrize(
    "image_key",
    [
        "../../../../etc/passwd",
        "/etc/passwd",
        "../image/../../../../root/.ssh/id_rsa",
    ],
)
def test_a_release_cannot_name_a_file_outside_the_corpus(tmp_path: Path, image_key: str) -> None:
    """Mem-Gallery's image keys are relative paths out of annotations this command downloads.

    They legitimately climb -- `../image/<topic>/` -- which is why they are joined at all, so
    the boundary has to be checked rather than assumed. Unchecked, the path was read and its
    bytes uploaded into the deployment's bucket, driven entirely by release content.
    """
    dialog = tmp_path / "mem-gallery" / "data" / "dialog"
    dialog.mkdir(parents=True)

    with pytest.raises(ValueError, match="outside the corpus"):
        _within(tmp_path, str(dialog), image_key)


def test_a_key_a_release_supplies_stays_one_object_key_component() -> None:
    """A topic is interpolated straight into the S3 key, so it may not carry a separator."""
    assert _key_component("topic_1", label="topic") == "topic_1"
    for hostile in ("a/b", "..", "", "a\\b"):
        with pytest.raises(ValueError, match="one object-key component"):
            _key_component(hostile, label="topic")


def test_a_legitimate_climbing_key_still_resolves(tmp_path: Path) -> None:
    """The guard must not break the shape the release actually uses."""
    dialog = tmp_path / "mem-gallery" / "data" / "dialog"
    dialog.mkdir(parents=True)

    resolved = _within(tmp_path, str(dialog), "../image/topic_1/a.jpg")

    assert resolved == (tmp_path / "mem-gallery" / "data" / "image" / "topic_1" / "a.jpg").resolve()
