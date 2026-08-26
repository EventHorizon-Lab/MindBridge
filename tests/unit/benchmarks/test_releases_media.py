"""Checks for the media half of the release table — what a producer's source files come from.

Nothing here reaches the network or the corpus. What is checked is everything that decides
*which* bytes would be asked for and *whether* asking twice costs anything: the join from a media
set to the release holding it, the narrowing that keeps a small run off a large release, the
refusal of a media set no download can obtain, and the extraction rule that has to survive being
interrupted halfway through 94 GiB.
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from mindbridge.benchmarks.releases import (
    MEDIA,
    RELEASES,
    UNOBTAINABLE,
    Release,
    _extract,
    ensure_media,
)


def test_every_media_set_names_a_release_the_table_can_reach() -> None:
    """A media set whose release is not in the table is a producer that can never run.

    `m3-robot` is why this is not the identity check it looks like: its annotations are the
    `m3-agent` Git repository and its videos are the `m3-bench` Hub dataset, so the media set's
    name, its release, and its directory are three different strings.
    """
    dangling = {
        name: media.release for name, media in MEDIA.items() if media.release not in RELEASES
    }

    assert dangling == {}
    assert MEDIA["m3-robot"].release == "m3-bench"
    assert RELEASES["m3-bench"].hub is True


def test_a_media_set_is_either_obtainable_or_explained_but_never_both() -> None:
    """Both tables answering to one name would make which behaviour you get depend on lookup order."""
    assert set(MEDIA) & set(UNOBTAINABLE) == set()


def test_every_unobtainable_set_says_where_the_operator_must_put_the_files() -> None:
    """An error that only says "not available" leaves the corpus in exactly the state it found it."""
    for name, reason in UNOBTAINABLE.items():
        assert "<benchmarks-root>/" in reason, f"{name} does not name a destination"


def test_media_that_ships_as_archives_is_the_only_media_named_by_a_zip_pattern() -> None:
    """`ensure_media` reads `.zip` off the pattern to decide whether to unpack, so a loose file
    named `*.zip` would be unpacked and an archive not named one would be left packed."""
    archived = {
        name
        for name, media in MEDIA.items()
        if any(pattern.endswith(".zip") for pattern in media.patterns)
    }

    assert archived == {"video-mme", "video-mme-v2"}


def test_asking_for_media_no_download_can_obtain_says_so_and_says_what_to_do(
    tmp_path: Path,
) -> None:
    """Ego4D is behind a signed agreement; discovering that as an absent file wastes the run."""
    with pytest.raises(FileNotFoundError) as caught:
        ensure_media("egotempo", root=tmp_path, announce=_never)

    message = str(caught.value)
    assert "ego4d-data.org" in message
    assert "<benchmarks-root>/egotempo/videos/<clip_id>.mp4" in message


def test_asking_for_a_media_set_that_does_not_exist_lists_the_ones_that_do(
    tmp_path: Path,
) -> None:
    """Four producers type these keys by hand; a bare KeyError makes a typo a puzzle."""
    with pytest.raises(KeyError) as caught:
        ensure_media("video_mme", root=tmp_path, announce=_never)

    message = str(caught.value)
    assert "video-mme" in message
    assert "m3-web" in message, "the unobtainable sets are keys a producer can legitimately try"


def test_a_whole_media_set_is_fetched_into_its_release_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The directory is the release's, not the media set's, and `m3-robot` is where they differ."""
    asked = _record_downloads(monkeypatch)

    destination = ensure_media("m3-robot", root=tmp_path)

    assert destination == tmp_path / "m3-bench"
    assert asked == [(RELEASES["m3-bench"].repository, ("videos/robot/*",), tmp_path / "m3-bench")]


def test_only_narrows_the_fetch_to_the_paths_a_small_run_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EgoLife is 477 GiB across six identities, and one task reads one of them."""
    asked = _record_downloads(monkeypatch)

    ensure_media("egolife", root=tmp_path, only=("A1_JAKE/DAY1/*",))

    assert [patterns for _, patterns, _ in asked] == [("A1_JAKE/DAY1/*",)]
    assert MEDIA["egolife"].patterns != ("A1_JAKE/DAY1/*",), "the default would fetch all six"


def test_archived_media_refuses_to_be_narrowed_rather_than_fetching_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No index says which of 20 volumes holds a video, so a narrowed fetch would match nothing,
    download nothing, and report success with an empty directory."""
    asked = _record_downloads(monkeypatch)

    with pytest.raises(ValueError, match="cannot be narrowed"):
        ensure_media("video-mme", root=tmp_path, only=("data/PaC3CEkCD6k.mp4",))

    assert asked == [], "refusing after downloading 94 GiB would refuse nothing worth refusing"


def test_an_archived_media_set_is_unpacked_where_its_layout_expects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Video-MME's volumes sit at the release root and name their entries `data/<id>.mp4`, so
    unpacking beside the volume is what produces the documented path."""
    _fake_download(monkeypatch, {"videos_chunked_01.zip": {"data/PaC3CEkCD6k.mp4": b"frames"}})

    destination = ensure_media("video-mme", root=tmp_path)

    assert (destination / "data" / "PaC3CEkCD6k.mp4").read_bytes() == b"frames"


def test_a_volume_held_in_a_subdirectory_unpacks_into_that_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other release packages itself the other way round, and this is the only case that can
    tell "beside the volume" apart from "at the release root".

    Video-MME-v2 keeps its 40 volumes in `videos/` and names their entries `<nnn>.mp4` with no
    directory of their own, where Video-MME does the reverse. Unpacking both into the release
    root would leave 800 videos loose beside `test.parquet` instead of under `videos/`, which
    every Video-MME-v2 test would still pass because that release's own volumes sit at the root.
    """
    _fake_download(monkeypatch, {"videos/001.zip": {"001.mp4": b"frames"}})

    destination = ensure_media("video-mme-v2", root=tmp_path)

    assert (destination / "videos" / "001.mp4").read_bytes() == b"frames"
    assert not (destination / "001.mp4").exists()


def test_a_second_run_over_a_complete_extraction_rewrites_nothing(tmp_path: Path) -> None:
    """The corpus is hundreds of gigabytes and a sweep runs this before every producer."""
    archive = _archive(tmp_path / "videos_chunked_01.zip", {"data/one.mp4": b"frames"})
    _extract(archive, announce=None)
    extracted = tmp_path / "data" / "one.mp4"
    # Same length, different bytes: a re-extraction would put `frames` back, and a check that
    # only compared timestamps or existence could not tell the two apart.
    extracted.write_bytes(b"EDITED")

    _extract(archive, announce=None)

    assert extracted.read_bytes() == b"EDITED"


def test_an_extraction_cut_off_partway_through_a_file_is_redone(tmp_path: Path) -> None:
    """Interruption is the case that matters: the Hub client cannot resume a volume, so the run
    that follows one has to be able to finish the unpacking rather than trust what it finds."""
    archive = _archive(tmp_path / "videos_chunked_01.zip", {"data/one.mp4": b"frames"})
    _extract(archive, announce=None)
    truncated = tmp_path / "data" / "one.mp4"
    truncated.write_bytes(b"fra")

    _extract(archive, announce=None)

    assert truncated.read_bytes() == b"frames"


def test_an_archive_naming_a_path_outside_itself_is_refused(tmp_path: Path) -> None:
    """These volumes come from the network, and extraction runs with the operator's own rights.

    One `..` rather than several on purpose: a guard rooted one directory too high still refuses
    a deep escape, so only the shallowest one says the base is the volume's own directory.
    """
    archive = _archive(tmp_path / "hostile.zip", {"../escaped.mp4": b"frames"})

    with pytest.raises(ValueError, match="escapes"):
        _extract(archive, announce=None)

    assert not (tmp_path.parent / "escaped.mp4").exists()


def test_extraction_is_announced_with_how_much_of_it_is_left(tmp_path: Path) -> None:
    """A volume takes minutes to unpack; silence there reads as a hang."""
    said: list[str] = []
    archive = _archive(
        tmp_path / "videos_chunked_01.zip",
        {"data/one.mp4": b"frames", "data/two.mp4": b"frames"},
    )
    _extract(archive, announce=None)
    (tmp_path / "data" / "two.mp4").unlink()

    _extract(archive, announce=said.append)

    assert said == ["extracting 1 of 2 files from videos_chunked_01.zip"]


def _archive(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as writing:
        for name, content in entries.items():
            writing.writestr(name, content)
    return path


def _record_downloads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple[str, ...], Path]]:
    """Replace the Hub call with a note of what it was asked for."""
    asked: list[tuple[str, tuple[str, ...], Path]] = []

    def record(release: Release, patterns: Sequence[str], *, destination: Path) -> None:
        asked.append((release.repository, tuple(patterns), destination))

    monkeypatch.setattr("mindbridge.benchmarks.releases._download_from_hub", record)
    return asked


def _fake_download(monkeypatch: pytest.MonkeyPatch, archives: dict[str, dict[str, bytes]]) -> None:
    """Replace the Hub call with one that materialises the archives it would have fetched."""

    def materialise(release: Release, patterns: Sequence[str], *, destination: Path) -> None:
        for name, entries in archives.items():
            volume = destination / name
            volume.parent.mkdir(parents=True, exist_ok=True)
            _archive(volume, entries)

    monkeypatch.setattr("mindbridge.benchmarks.releases._download_from_hub", materialise)


def _never(message: str) -> None:
    raise AssertionError(f"nothing should have been announced, but got: {message}")
