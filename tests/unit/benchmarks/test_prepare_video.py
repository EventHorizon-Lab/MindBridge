"""Checks for the three producers that write `runtime.PreparedVideo`.

Staging reaches an object store and cutting reaches a video encoder, so both run against doubles
and synthetic sources rather than the corpus.

What is checked is what a wrong manifest costs. Preparing a unit the run will not read is a
wasted upload, and preparing the wrong file is a run scored against the wrong video -- which no
later check catches, because both produce a manifest that loads. So each release directory here
holds decoys: files named after the field a producer must *not* key on, of a different length, so
keying on one fails an assertion rather than a lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from mindbridge.benchmarks import prepare_video
from mindbridge.benchmarks.runtime import load_prepared_videos
from mindbridge.benchmarks.staging import SEGMENT_SECONDS, PrepareRequest, Staging

pytest.importorskip("av", reason="prepared media is cut with the media extra's decoders")


class _RecordingClient:
    """An S3 double that keeps what it was asked to write."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body


class _FakeArrowTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def to_pylist(self) -> list[dict[str, object]]:
        return self.rows


class _FakeParquet:
    """Stands in for `pyarrow.parquet`, as the adapter checks do, so the extra stays optional."""

    rows: ClassVar[list[dict[str, object]]] = []

    @classmethod
    def read_table(cls, source: Path) -> _FakeArrowTable:
        return _FakeArrowTable(cls.rows)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _RecordingClient:
    """Point the producers at a bucket double and stop them fetching a release."""
    recorder = _RecordingClient()
    monkeypatch.setattr(prepare_video, "staging", lambda: Staging("bucket", recorder))
    monkeypatch.setattr(prepare_video, "ensure_media", lambda name, *, root, **_: root / name)
    return recorder


def test_video_mme_prepares_the_band_the_runner_keeps_from_the_file_it_is_named_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """Two things a wrong producer gets wrong silently, checked together on one preparation.

    `--duration long --limit 1` selects video `002` and nothing else, so preparing `001` or `003`
    is an upload the run will never read. And Video-MME's files are named by `source_video_id`,
    the YouTube ID, not by the `video_id` the annotation is keyed on: the release directory here
    also holds a `002.mp4`, of a different length, so a producer keying on the wrong field finds
    a file and prepares the wrong video rather than failing.
    """
    from mindbridge.benchmarks.video_mme import load_video_mme
    from mindbridge.benchmarks.video_mme_cli import _parse_arguments, _select_prepared

    dataset = _video_mme_release(tmp_path, monkeypatch)
    manifest = tmp_path / "prepared.json"

    prepare_video.prepare_video_mme(
        _request(
            _video_mme_argv(dataset, manifest, tmp_path, "--duration", "long", "--limit", "1"),
            tmp_path,
        )
    )

    prepared = load_prepared_videos(manifest)
    assert [video.video_id for video in prepared] == ["002"]
    # 65 seconds is `bbb.mp4`; the decoy `002.mp4` beside it is 35 seconds, so the split length
    # is what says which file was read.
    assert [segment.duration_ms for segment in prepared[0].segments] == [30_000, 30_000, 5_000]
    assert [segment.start_seconds for segment in prepared[0].segments] == [0.0, 30.0, 60.0]
    assert all(
        key.startswith("bucket/tenants/benchmark_video_mme_002_prep-01/") for key in client.objects
    )
    # One object per segment: a key that did not vary would leave every segment of the manifest
    # pointing at the last clip's bytes, which nothing downstream would notice.
    assert len(client.objects) == 3
    # The manifest is only right if the runner's own lookup accepts it.
    arguments = _parse_arguments(list(_video_mme_argv(dataset, manifest, tmp_path)), None)
    videos = load_video_mme(arguments.dataset_path)
    assert _select_prepared(prepared, tuple(video for video in videos if video.video_id == "002"))


def test_two_preparations_of_one_video_produce_identical_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """A run manifest pins this file's digest, so anything drawn from a clock would churn it."""
    dataset = _video_mme_release(tmp_path, monkeypatch)
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    for manifest in (first, second):
        prepare_video.prepare_video_mme(
            _request(_video_mme_argv(dataset, manifest, tmp_path, "--limit", "1"), tmp_path)
        )

    assert first.read_bytes() == second.read_bytes()
    # Comparing two preparations cannot see a clock read once at import, and both of this
    # manifest's timestamps are exactly that. So the two constants are pinned by value: any
    # value drawn from a clock is a date in this decade rather than either of these.
    body = first.read_text(encoding="utf-8")
    assert '"created_at": "1970-01-01T00:00:00Z"' in body
    assert '"timeline_origin": "2000-01-01T00:00:00Z"' in body


def test_an_absent_source_is_named_before_any_other_video_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """Cutting and uploading is the expensive half, and these sources are gigabytes each.

    A producer that resolved each source as it reached it would have paid for every video before
    the absent one, so the emptiness of the bucket is the assertion, not the exception.
    """
    dataset = _video_mme_release(tmp_path, monkeypatch)
    (tmp_path / "video-mme" / "data" / "ccc.mp4").unlink()

    with pytest.raises(FileNotFoundError, match=r"ccc\.mp4 is absent from the video-mme media"):
        prepare_video.prepare_video_mme(
            _request(_video_mme_argv(dataset, tmp_path / "prepared.json", tmp_path), tmp_path)
        )

    assert client.objects == {}


def test_a_transcript_source_this_cannot_produce_is_refused_before_it_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """The runner refuses the manifest anyway, but only after preparation has been paid for."""
    dataset = _video_mme_release(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="cannot serve --transcript-source official_subtitles"):
        prepare_video.prepare_video_mme(
            _request(
                _video_mme_argv(
                    dataset,
                    tmp_path / "prepared.json",
                    tmp_path,
                    transcript_source="official_subtitles",
                ),
                tmp_path,
            )
        )

    assert client.objects == {}


def test_a_release_cannot_name_a_source_outside_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """The file read and uploaded is named by release content, so the boundary is checked."""
    dataset = _video_mme_release(tmp_path, monkeypatch, source_video_id="../../../../etc/passwd")

    with pytest.raises(ValueError, match="outside the corpus"):
        prepare_video.prepare_video_mme(
            _request(_video_mme_argv(dataset, tmp_path / "prepared.json", tmp_path), tmp_path)
        )

    assert client.objects == {}


def test_video_mme_v2_prepares_only_the_group_types_its_runner_keeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: _RecordingClient
) -> None:
    """The rating is defined over whole groups, so the subset flag is the runner's own."""
    dataset = _video_mme_v2_release(tmp_path, monkeypatch)
    manifest = tmp_path / "prepared.json"

    prepare_video.prepare_video_mme_v2(
        _request(
            _video_mme_v2_argv(dataset, manifest, tmp_path, "--group-type", "relevance"), tmp_path
        )
    )

    prepared = load_prepared_videos(manifest)
    assert [video.video_id for video in prepared] == ["002"]
    assert [segment.duration_ms for segment in prepared[0].segments] == [30_000, 5_000]
    assert all(
        key.startswith("bucket/tenants/benchmark_video_mme_v2_002_prep-01/")
        for key in client.objects
    )
    assert len(client.objects) == 2


def test_egotempo_prepares_one_clip_per_question_group_the_run_will_ingest(
    tmp_path: Path, client: _RecordingClient
) -> None:
    """Two questions over one clip is one ingest, so a clip is prepared once, not once each."""
    dataset = _egotempo_release(tmp_path)
    manifest = tmp_path / "prepared.json"

    prepare_video.prepare_egotempo(_request(_egotempo_argv(dataset, manifest, tmp_path), tmp_path))

    prepared = load_prepared_videos(manifest)
    assert [video.video_id for video in prepared] == ["ego4d-uid_10.0_75.0", "ego4d-uid_0.0_20.0"]
    assert [segment.duration_ms for segment in prepared[0].segments] == [30_000, 30_000, 5_000]
    assert [segment.duration_ms for segment in prepared[1].segments] == [20_000]
    assert {key.rsplit("/", 2)[-2] for key in client.objects} == {
        "benchmark_egotempo_ego4d-uid_10.0_75.0_prep-01",
        "benchmark_egotempo_ego4d-uid_0.0_20.0_prep-01",
    }


def test_egotempo_asks_its_acquirer_for_the_clips_it_is_missing_and_no_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EgoTempo's clips are acquired from Ego4D one span at a time, so an unnarrowed ask is
    every clip the split names -- tens of terabytes to prepare the one this run selected.

    Video-MME goes through the same `_source` helper and must not narrow: its media ships as 20
    opaque archives with no index of which holds which video, and `ensure_media` refuses `only`
    for those. So the narrowing is per call site, and this pins the EgoTempo one.
    """
    asked: list[dict[str, object]] = []
    monkeypatch.setattr(prepare_video, "staging", lambda: Staging("bucket", _RecordingClient()))
    monkeypatch.setattr(
        prepare_video,
        "ensure_media",
        lambda name, **kwargs: asked.append({"release": name, **kwargs}),
    )
    dataset = _egotempo_release(tmp_path)
    for clip in ("ego4d-uid_10.0_75.0", "ego4d-uid_0.0_20.0"):
        (tmp_path / "egotempo" / "videos" / f"{clip}.mp4").unlink()

    with pytest.raises(FileNotFoundError):
        prepare_video.prepare_egotempo(
            _request(
                _egotempo_argv(dataset, tmp_path / "prepared.json", tmp_path, "--limit", "1"),
                tmp_path,
            )
        )

    assert asked == [
        {
            "release": "egotempo",
            "root": tmp_path,
            "only": ("videos/ego4d-uid_10.0_75.0.mp4",),
            "download": True,
            # `--quiet` reaches the fetch as well as the producer's own progress, which is the
            # other half of the wiring the m3 test pins: this request is quiet, so nothing here
            # may announce.
            "announce": None,
        }
    ]


def _request(argv: tuple[str, ...], root: Path) -> PrepareRequest:
    return PrepareRequest(argv=argv, benchmarks_root=root, quiet=True)


def _core_argv(dataset: Path, manifest: Path, root: Path) -> tuple[str, ...]:
    return (
        "--dataset",
        str(dataset),
        "--prepared-media",
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


def _video_mme_argv(
    dataset: Path,
    manifest: Path,
    root: Path,
    *extra: str,
    transcript_source: str = "none",
) -> tuple[str, ...]:
    return (
        *_core_argv(dataset, manifest, root),
        "--transcript-source",
        transcript_source,
        *extra,
    )


def _video_mme_v2_argv(dataset: Path, manifest: Path, root: Path, *extra: str) -> tuple[str, ...]:
    return (*_core_argv(dataset, manifest, root), "--transcript-source", "none", *extra)


def _egotempo_argv(dataset: Path, manifest: Path, root: Path, *extra: str) -> tuple[str, ...]:
    return (*_core_argv(dataset, manifest, root), *extra)


def _video_mme_release(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_video_id: str | None = None,
) -> Path:
    """Three videos across two bands, plus decoys named after the field to key on wrongly."""
    media = root / "video-mme" / "data"
    media.mkdir(parents=True)
    _synthetic_video(media / "aaa.mp4", seconds=35)
    _synthetic_video(media / "bbb.mp4", seconds=SEGMENT_SECONDS * 2 + 5)
    _synthetic_video(media / "ccc.mp4", seconds=35)
    _synthetic_video(media / "002.mp4", seconds=35)
    _synthetic_video(media / "003.mp4", seconds=35)
    _FakeParquet.rows = [
        {
            "video_id": video_id,
            "duration": duration,
            "domain": "Knowledge",
            "sub_category": "Humanity & History",
            "url": f"https://www.youtube.com/watch?v={youtube_id}",
            "videoID": source_video_id if source_video_id is not None else youtube_id,
            "question_id": f"{video_id}-{index}",
            "task_type": "Counting Problem",
            "question": "Which decoration appears most?",
            "options": ["A. Apples.", "B. Candles.", "C. Berries.", "D. Equal."],
            "answer": "C",
        }
        for video_id, duration, youtube_id in (
            ("001", "short", "aaa"),
            ("002", "long", "bbb"),
            ("003", "long", "ccc"),
        )
        for index in range(1, 4)
    ]
    monkeypatch.setattr("mindbridge.benchmarks.video_mme.import_module", lambda name: _FakeParquet)
    return root / "video-mme" / "videomme" / "test-00000-of-00001.parquet"


def _video_mme_v2_release(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One logic group and one relevance group, so the group-type filter has something to drop."""
    media = root / "video-mme-v2" / "videos"
    media.mkdir(parents=True)
    _synthetic_video(media / "001.mp4", seconds=SEGMENT_SECONDS * 2 + 5)
    _synthetic_video(media / "002.mp4", seconds=35)
    _FakeParquet.rows = [
        {
            "video_id": video_id,
            "url": "https://www.youtube.com/watch?v=AYSYelOQtQI",
            "group_type": group_type,
            "group_structure": structure,
            "question_id": f"{video_id}-{index}",
            "question": f"Question {index}?",
            "options": "\n".join(f"{label}. Option {label}." for label in "ABCDEFGH"),
            "answer": "F",
            "level": "1",
            "second_head": "Frames & Audio",
            "third_head": "Visual-Audio Collaborative Reasoning",
        }
        for video_id, group_type, structure in (
            ("001", "logic", "[1,2,3,4]"),
            ("002", "relevance", "4"),
        )
        for index in range(1, 5)
    ]
    monkeypatch.setattr(
        "mindbridge.benchmarks.video_mme_v2.import_module", lambda name: _FakeParquet
    )
    return root / "video-mme-v2" / "test.parquet"


def _egotempo_release(root: Path) -> Path:
    """Three questions over two clips, laid out the way `releases.UNOBTAINABLE` asks for them.

    Named by `clip_id` and already trimmed to the span that name spells, because the operator is
    the only thing that can produce them: Ego4D is behind a signed agreement. The decoy is the
    full-scale source those spans came from, named by `source_video_id`, which is the field a
    producer must not key on here.
    """
    media = root / "egotempo" / "videos"
    media.mkdir(parents=True)
    _synthetic_video(media / "ego4d-uid.mp4", seconds=90)
    _synthetic_video(media / "ego4d-uid_10.0_75.0.mp4", seconds=65)
    _synthetic_video(media / "ego4d-uid_0.0_20.0.mp4", seconds=20)
    dataset = root / "egotempo" / "egotempo_openQA.json"
    dataset.write_text(
        json.dumps(
            {
                "info": {"release date": "2026-01-01", "version": "1.0"},
                "annotations": [
                    {
                        "question_id": question_id,
                        "clip_id": clip_id,
                        "question_type": "temporal",
                        "question": "What happened first?",
                        "answer": "the kettle boiled",
                    }
                    for question_id, clip_id in (
                        ("q1", "ego4d-uid_10.0_75.0"),
                        ("q2", "ego4d-uid_10.0_75.0"),
                        ("q3", "ego4d-uid_0.0_20.0"),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _synthetic_video(path: Path, *, seconds: int) -> Path:
    """Encode a small source of a known length, so segment boundaries are checkable."""
    import av
    import numpy

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.thread_count, stream.thread_type = 1, "NONE"
        for index in range(5 * seconds):
            array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())
    return path
