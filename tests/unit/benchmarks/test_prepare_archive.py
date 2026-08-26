"""Checks for the ATM-Bench prepared-media producer.

Staging reaches an object store, so it runs against a double; the release is a handful of
synthetic files in the shapes `atm_bench` parses. What is checked is what the manifest has to
satisfy for a run to be worth anything: the archive's own clock, the coupling between an item's
official stem and the object staged for it, and that the arm which ingests captions instead is
not quietly handed three gigabytes of upload.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mindbridge.benchmarks import prepare_archive
from mindbridge.benchmarks.prepare_archive import prepare_atm
from mindbridge.benchmarks.staging import STAGED_AT, PrepareRequest, Staging
from mindbridge.contracts import MediaObjectInput
from mindbridge.core import MediaKind

pytest.importorskip("av", reason="prepared media is cut with the media extra's decoders")

_IMAGE_STEM = "20220703_210745"
_SECOND_IMAGE_STEM = "20220626_181551"
_VIDEO_STEM = "20220502_172850"
_CAPTURED_AT = datetime(2022, 7, 3, 21, 7, 45, tzinfo=timezone.utc)
"""What the release means by `20220703_210745`, written out rather than recomputed.

`atm_capture_time` is the function under test here as much as the producer is: asserting
`created_at == atm_capture_time(stem)` would pass for any two agreeing implementations,
including two that both read the stem backwards.
"""


class _RecordingClient:
    """An S3 double that keeps what it was asked to write."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body


class _RenamingStaging(Staging):
    """A producer that keys the object by something other than the official stem.

    Which is the failure `AtmPreparedMedia`'s validator exists for, and the reason to build the
    manifest through the contract rather than by hand: `media_id` and `media_object_id` are
    supplied from two places in the same call and nothing but that validator ties them.
    """

    def stage(self, **arguments: Any) -> MediaObjectInput:  # noqa: ANN401
        staged = super().stage(**arguments)
        return staged.model_copy(update={"media_object_id": f"other_{staged.media_object_id}"})


class _Fetches:
    """What `ensure_media` was asked for, which is what a prepared corpus must not ask at all."""

    def __init__(self) -> None:
        self.only: list[tuple[str, ...]] = []

    def __call__(
        self, release: str, *, root: Path, only: tuple[str, ...] = (), **_: object
    ) -> Path:
        self.only.append(tuple(only))
        return root / release


@pytest.fixture(autouse=True)
def fetches(monkeypatch: pytest.MonkeyPatch) -> _Fetches:
    """`ensure_media` reaches the Hub, and the fixtures are already on disk."""
    from mindbridge.benchmarks import releases

    recorder = _Fetches()
    monkeypatch.setattr(releases, "ensure_media", recorder)
    return recorder


def test_the_manifest_carries_the_archives_own_capture_time_not_the_upload_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_observe_media` reads `occurred_at` off `created_at`, and the runner refuses a skew.

    `STAGED_AT` is right for every other producer and wrong here: it would place a
    three-and-a-half-year archive at the epoch, and every temporal question with it.
    """
    from mindbridge.benchmarks.atm_bench import load_atm_sgm
    from mindbridge.benchmarks.atm_bench_runner import load_prepared_atm
    from mindbridge.benchmarks.atm_cli import _require_one_clock

    release = _release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))
    manifest = tmp_path / "prepared.json"

    prepare_atm(
        PrepareRequest(argv=_argv(tmp_path, manifest), benchmarks_root=tmp_path, quiet=True)
    )

    prepared = load_prepared_atm(manifest)
    by_id = {item.media_id: item.media_object for item in prepared.media}
    assert by_id[_IMAGE_STEM].created_at == _CAPTURED_AT
    assert by_id[_IMAGE_STEM].created_at != STAGED_AT
    # The runner's own comparison of the two clocks, which is what a skew actually costs.
    _require_one_clock(
        prepared,
        (*load_atm_sgm(release.sgm_image), *load_atm_sgm(release.sgm_video)),
        media_source="raw",
    )


def test_a_video_is_staged_whole_with_its_duration_and_an_image_without_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ATM's items are seconds long and keyed one to one, so nothing here is cut into segments."""
    from mindbridge.benchmarks.atm_bench_runner import load_prepared_atm

    release = _release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))
    manifest = tmp_path / "prepared.json"

    prepare_atm(
        PrepareRequest(argv=_argv(tmp_path, manifest), benchmarks_root=tmp_path, quiet=True)
    )

    by_id = {item.media_id: item.media_object for item in load_prepared_atm(manifest).media}
    assert by_id[_VIDEO_STEM].kind is MediaKind.VIDEO
    assert by_id[_VIDEO_STEM].duration_ms == 3_300
    assert by_id[_IMAGE_STEM].duration_ms is None
    # Byte for byte the release's own file, not a re-encode: the URI's extension is the source's
    # and the object is the source.
    assert by_id[_VIDEO_STEM].uri.endswith(f"/atm/{_VIDEO_STEM}.mp4")
    assert client.objects[by_id[_VIDEO_STEM].uri.removeprefix("s3://")] == (
        release.video.read_bytes()
    )
    assert (
        client.objects[by_id[_IMAGE_STEM].uri.removeprefix("s3://")] == release.image.read_bytes()
    )


def test_an_object_keyed_by_anything_but_the_official_stem_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stem is the evidence ID every question cites, so the two cannot be set independently."""
    _release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: _RenamingStaging("bucket", client))
    manifest = tmp_path / "prepared.json"

    with pytest.raises(ValidationError, match="official media stem"):
        prepare_atm(
            PrepareRequest(argv=_argv(tmp_path, manifest), benchmarks_root=tmp_path, quiet=True)
        )
    assert not manifest.exists()


@pytest.mark.parametrize(
    "hostile",
    [
        f"../../../../etc/{_IMAGE_STEM}.jpg",
        f"/etc/{_IMAGE_STEM}.jpg",
        f"data/raw_memory/../../../../root/.ssh/{_IMAGE_STEM}.jpg",
    ],
)
def test_a_release_supplied_media_path_cannot_leave_the_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile: str,
) -> None:
    """The file each item lives in is named by release content, and read and uploaded from it.

    The climb has to keep the official stem to get this far: `atm_capture_time` refuses a name
    with no timestamp in it, which rules out `../../etc/passwd` before any of this, and a stem
    no question cites is never selected. What is left -- a valid stem under a directory that
    climbs out of the corpus -- is what `within` is the only guard against.
    """
    release = _release(tmp_path)
    entries = json.loads(release.sgm_image.read_text(encoding="utf-8"))
    entries[0]["image_path"] = hostile
    release.sgm_image.write_text(json.dumps(entries), encoding="utf-8")
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))
    manifest = tmp_path / "prepared.json"

    with pytest.raises(ValueError, match="outside the corpus"):
        prepare_atm(
            PrepareRequest(argv=_argv(tmp_path, manifest), benchmarks_root=tmp_path, quiet=True)
        )
    assert client.objects == {}


def test_the_arm_that_ingests_captions_is_not_handed_an_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`atm-*-sgm` is the same benchmark and reads no prepared media.

    `PREPARERS` is keyed by benchmark and the sweep appends `--prepared-media` to whichever task
    does not already carry it, which is exactly the two sgm tasks -- so a producer registered for
    `atm` is aimed at all four unless something refuses.
    """
    _release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))
    manifest = tmp_path / "prepared.json"
    argv = _argv(tmp_path, manifest, media_source="sgm")

    with pytest.raises(ValueError, match="--media-source"):
        prepare_atm(PrepareRequest(argv=argv, benchmarks_root=tmp_path, quiet=True))
    assert client.objects == {}
    assert not manifest.exists()


def test_a_limited_run_stages_the_haystack_its_questions_were_written_against(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--limit` narrows the archive the way `_archive_for_run` narrows the rest of it.

    Gold evidence and the release's own distractors for that question, and nothing a question
    this run does not ask cites -- a haystack of one item is not the question that was written,
    and 4,292 items is not a smoke run.
    """
    from mindbridge.benchmarks.atm_bench_runner import load_prepared_atm

    _release(tmp_path)
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))
    manifest = tmp_path / "prepared.json"

    prepare_atm(
        PrepareRequest(
            argv=(*_argv(tmp_path, manifest), "--limit", "1"),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    staged = {item.media_id for item in load_prepared_atm(manifest).media}
    # q1 cites the image as gold and the video as its distractor; q2 cites the second image and
    # is not selected.
    assert staged == {_IMAGE_STEM, _VIDEO_STEM}


def test_two_preparations_of_one_archive_write_the_same_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run manifest pins this file's digest, so a wall clock anywhere in it would churn."""
    _release(tmp_path)
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", _RecordingClient()))

    first, second = tmp_path / "one.json", tmp_path / "two.json"
    for manifest in (first, second):
        prepare_atm(
            PrepareRequest(argv=_argv(tmp_path, manifest), benchmarks_root=tmp_path, quiet=True)
        )

    assert first.read_bytes() == second.read_bytes()


def test_a_question_whose_evidence_the_archive_cannot_supply_is_refused_before_uploading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's own refusal, reached before three gigabytes of transfer rather than after."""
    release = _release(tmp_path)
    questions = json.loads(release.dataset.read_text(encoding="utf-8"))
    questions[0]["evidence_ids"] = ["20200101_000000"]
    release.dataset.write_text(json.dumps(questions), encoding="utf-8")
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))

    with pytest.raises(ValueError, match="missing prepared ATM-Bench media"):
        prepare_atm(
            PrepareRequest(
                argv=_argv(tmp_path, tmp_path / "prepared.json"),
                benchmarks_root=tmp_path,
                quiet=True,
            )
        )
    assert client.objects == {}


def test_an_archive_already_on_disk_is_not_fetched_at_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetches: _Fetches,
) -> None:
    """`ensure_media` refuses an unobtainable set before it looks at the disk.

    So an operator who placed a release by hand has to be able to prepare from it without the
    fetch they do not need being consulted.
    """
    _release(tmp_path)
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", _RecordingClient()))

    prepare_atm(
        PrepareRequest(
            argv=_argv(tmp_path, tmp_path / "prepared.json"),
            benchmarks_root=tmp_path,
            quiet=True,
        )
    )

    assert fetches.only == []


def test_an_archive_item_the_release_lists_but_does_not_ship_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fetches: _Fetches,
) -> None:
    """Absent media is the ordinary state of a fresh corpus, and has to say which file.

    It is also the only reason to fetch: named file by file, so a scoped run does not pull the
    whole 3.1 GB of raw media to stage two items.
    """
    release = _release(tmp_path)
    release.video.unlink()
    client = _RecordingClient()
    monkeypatch.setattr(prepare_archive, "staging", lambda: Staging("bucket", client))

    with pytest.raises(FileNotFoundError, match=_VIDEO_STEM):
        prepare_atm(
            PrepareRequest(
                argv=_argv(tmp_path, tmp_path / "prepared.json"),
                benchmarks_root=tmp_path,
                quiet=True,
            )
        )
    assert fetches.only == [(f"data/raw_memory/video/{_VIDEO_STEM}.mp4",)]
    assert client.objects == {}


class _Release:
    """Where each synthetic file landed, so a test can corrupt exactly one of them."""

    def __init__(self, root: Path) -> None:
        self.root = root / "atm-bench"
        self.dataset = self.root / "data" / "atm-bench" / "atm-bench.json"
        self.emails = self.root / "data" / "raw_memory" / "email" / "emails.json"
        self.sgm_image = self.root / "data" / "processed_memory" / "image_batch_results.json"
        self.sgm_video = self.root / "data" / "processed_memory" / "video_batch_results.json"
        self.image = self.root / "data" / "raw_memory" / "image" / f"{_IMAGE_STEM}.jpg"
        self.second_image = (
            self.root / "data" / "raw_memory" / "image" / f"{_SECOND_IMAGE_STEM}.jpg"
        )
        self.video = self.root / "data" / "raw_memory" / "video" / f"{_VIDEO_STEM}.mp4"


def _release(root: Path) -> _Release:
    """Write the smallest ATM-Bench release the adapter parses, in the release's own key names."""
    release = _Release(root)
    for path in (release.dataset, release.emails, release.sgm_image, release.image, release.video):
        path.parent.mkdir(parents=True, exist_ok=True)
    release.image.write_bytes(b"\xff\xd8\xff" + _IMAGE_STEM.encode())
    release.second_image.write_bytes(b"\xff\xd8\xff" + _SECOND_IMAGE_STEM.encode())
    release.video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + _VIDEO_STEM.encode())
    release.dataset.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "What flew past?",
                    "answer": "an aeroplane",
                    "qtype": "open_end",
                    "evidence_ids": [_IMAGE_STEM],
                    "niah_evidence_ids": [_IMAGE_STEM, _VIDEO_STEM],
                },
                {
                    "id": "q2",
                    "question": "Where was the churchyard?",
                    "answer": "Aberdeen",
                    "qtype": "open_end",
                    "evidence_ids": [_SECOND_IMAGE_STEM],
                    "niah_evidence_ids": [_SECOND_IMAGE_STEM],
                },
            ]
        ),
        encoding="utf-8",
    )
    release.emails.write_text(
        json.dumps(
            [
                {
                    "id": "email_0001",
                    "timestamp": "2022-07-04 09:00:00",
                    "short_summary": "A booking",
                    "detail": "Confirmed.",
                }
            ]
        ),
        encoding="utf-8",
    )
    release.sgm_image.write_text(
        json.dumps(
            [
                _sgm_entry("image_path", f"data/raw_memory/image/{_IMAGE_STEM}.jpg"),
                _sgm_entry("image_path", f"data/raw_memory/image/{_SECOND_IMAGE_STEM}.jpg"),
            ]
        ),
        encoding="utf-8",
    )
    release.sgm_video.write_text(
        json.dumps(
            [
                {
                    **_sgm_entry("video_path", f"data/raw_memory/video/{_VIDEO_STEM}.mp4"),
                    "duration": 3.300756,
                }
            ]
        ),
        encoding="utf-8",
    )
    return release


def _sgm_entry(field: str, path: str) -> dict[str, Any]:
    """One batch-results row, in the release's own key names rather than the adapter's."""
    return {
        field: path,
        "file_size": 1_024,
        "timestamp": "2022-07-03 21:07:45",
        "location_name": "West Quay Road, Southampton",
        "city": "Southampton, United Kingdom",
        "short_caption": "An aeroplane against a clear sky.",
        "caption": "A small aircraft crosses a cloudless sky.",
        "ocr_text": "",
        "tags": ["airplane", "sky"],
    }


def _argv(root: Path, manifest: Path, *, media_source: str = "raw") -> tuple[str, ...]:
    release = _Release(root)
    return (
        "--dataset",
        str(release.dataset),
        "--split",
        "main",
        "--media-source",
        media_source,
        "--prepared-media",
        str(manifest),
        "--emails",
        str(release.emails),
        "--sgm-image",
        str(release.sgm_image),
        "--sgm-video",
        str(release.sgm_video),
        "--output",
        str(root / "out.json"),
        "--api-base-url",
        "http://localhost:8000",
        "--deployment-config",
        str(root / "deployment.json"),
        "--run-id",
        "prep-01",
    )
