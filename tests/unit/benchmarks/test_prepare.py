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

from mindbridge.benchmarks.prepare import prepare_mem_gallery
from mindbridge.benchmarks.staging import (
    SEGMENT_SECONDS,
    STAGED_AT,
    PrepareRequest,
    Staging,
    key_component,
    video_segments,
    within,
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
    staging = Staging("mindbridge-media", client)

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
    staging = Staging("bucket", _RecordingClient())

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
    monkeypatch.setattr(prepare, "staging", lambda: Staging("bucket", client))
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
        within(tmp_path, str(dialog), image_key)


def test_a_key_a_release_supplies_stays_one_object_key_component() -> None:
    """A topic is interpolated straight into the S3 key, so it may not carry a separator."""
    assert key_component("topic_1", label="topic") == "topic_1"
    for hostile in ("a/b", "..", "", "a\\b"):
        with pytest.raises(ValueError, match="one object-key component"):
            key_component(hostile, label="topic")


def test_a_legitimate_climbing_key_still_resolves(tmp_path: Path) -> None:
    """The guard must not break the shape the release actually uses."""
    dialog = tmp_path / "mem-gallery" / "data" / "dialog"
    dialog.mkdir(parents=True)

    resolved = within(tmp_path, str(dialog), "../image/topic_1/a.jpg")

    assert resolved == (tmp_path / "mem-gallery" / "data" / "image" / "topic_1" / "a.jpg").resolve()


def test_m3_asks_for_the_one_video_it_is_missing_rather_than_the_whole_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `--limit 1` run must not pay for the other 99 videos, or the other 919.

    Two different costs behind one call. `m3-robot` is a Hub download of about 2 GB per video, so
    an unnarrowed fetch to cut one was roughly 200 GB; `m3-web` is acquired one live URL at a
    time, so the same call is 920 downloads. Both come out of the same line, which is why this
    asserts the narrowing rather than the fetch: the producer knows exactly which file is absent,
    and `ensure_media` takes paths relative to the release directory.
    """
    from mindbridge.benchmarks import prepare
    from mindbridge.benchmarks.prepare import prepare_m3

    asked: list[dict[str, object]] = []
    monkeypatch.setattr(
        prepare, "ensure_media", lambda name, **kwargs: asked.append({"release": name, **kwargs})
    )
    monkeypatch.setattr(prepare, "staging", lambda: Staging("bucket", _RecordingClient()))
    dataset = _m3_release(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"videos/robot/video_02\.mp4|is absent"):
        prepare_m3(
            PrepareRequest(
                argv=(
                    "--dataset",
                    str(dataset),
                    "--subset",
                    "robot",
                    "--prepared-media",
                    str(tmp_path / "prepared.json"),
                    "--output",
                    str(tmp_path / "out.jsonl"),
                    "--api-base-url",
                    "http://localhost:8000",
                    "--deployment-config",
                    str(tmp_path / "deployment.json"),
                    "--run-id",
                    "prep-01",
                    "--limit",
                    "1",
                ),
                benchmarks_root=tmp_path,
                quiet=False,
            )
        )

    assert [{key: value for key, value in call.items() if key != "announce"} for call in asked] == [
        {
            "release": "m3-robot",
            "root": tmp_path,
            "only": ("videos/robot/video_01.mp4",),
            "download": True,
        }
    ], "the fetch names the one selected video, and no other"
    # And it says so out loud. This producer passed no `announce` at all, so a 712 MB `m3-web`
    # acquisition and a 2 GB Hub download both ran to completion with nothing on stderr -- found
    # by running the real command, not by any test here, which is why one exists now.
    assert callable(asked[0]["announce"]), "a multi-gigabyte fetch must not be silent"


def _m3_release(root: Path) -> Path:
    """Two videos' worth of annotation, in the release's own `robot.json` shape."""
    import json

    dataset = root / "m3-agent" / "data" / "annotations" / "robot.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        json.dumps(
            {
                f"video_{index:02d}": {
                    "video_path": f"videos/robot/video_{index:02d}.mp4",
                    "qa_list": [
                        {
                            "question_id": f"video_{index:02d}_Q01",
                            "question": "What happened?",
                            "answer": "a robot moved",
                            "type": ["Temporal"],
                            "timestamp": "00:10",
                            "before_clip": 0,
                        }
                    ],
                }
                for index in (1, 2)
            }
        ),
        encoding="utf-8",
    )
    return dataset
