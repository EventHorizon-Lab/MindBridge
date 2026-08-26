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

import httpx
import pytest

from mindbridge.benchmarks import releases
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


def test_exactly_these_media_sets_are_the_ones_that_download_themselves() -> None:
    """Which side of the line a benchmark falls on is what its producer is written against.

    Moving one across is a silent behaviour change for a caller that never sees this file: a
    producer calls `ensure_media` on an absent file and either gets it or gets an exception, and
    nothing in the producer's own tests would notice the day that flipped. MM-Lifelong is the
    live example -- its `video_list.txt` of YouTube and bilibili URLs reads like an instruction
    to fetch them by hand, but all three of its scales are on the Hub at the pinned revision, and
    someone acting on that file would move it and break two producers.
    """
    assert set(MEDIA) == {
        "video-mme",
        "video-mme-v2",
        "egolife",
        "mm-lifelong",
        "supermemory-vqa",
        "atm-bench",
        "mem-gallery",
        "m3-robot",
    }
    assert set(UNOBTAINABLE) == {"egotempo", "m3-web"}


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
    tell "beside the volume" apart from "into the release directory".

    Video-MME-v2 keeps its 40 volumes in `videos/` and names their entries `<nnn>.mp4` with no
    directory of their own, where Video-MME does the reverse. Extracting into the release
    directory instead would leave 800 videos loose beside `test.parquet` rather than under
    `videos/` -- and every *Video-MME* test would still pass, because that release keeps its
    volumes at the release root, where the two destinations are the same directory.

    Measured rather than argued: an implementation extracting into the release directory fails
    this test and only this test. Note that moving the base up one level unconditionally is a
    different, cruder bug that several tests catch, so catching *that* would not have shown the
    exclusivity this docstring claims.
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


def test_a_directory_entry_in_an_archive_does_not_become_a_zero_byte_file(
    tmp_path: Path,
) -> None:
    """Skipping directory entries is what stops the first real entry failing to make its parent.

    A directory entry has a name ending in `/` and no content, so unpacking it writes an empty
    *file* where a directory belongs, and the entry that follows cannot create the parent it
    needs. Against a real archive that is `FileExistsError` partway through, which on a 94 GiB
    volume means discovering the packaging halfway through the download it follows.

    Neither archived release exercises this today -- I read the central directories of
    `videos_chunked_20.zip` and `videos/001.zip` at their pins and both hold zero directory
    entries, so deleting the guard is a no-op against them and no test here would notice.
    It is kept because the same publisher's `subtitle.zip` in the Video-MME repository *does*
    carry one (`subtitle/`, 1 of 745), so the packaging style is live rather than hypothetical,
    and `MEDIA` is a table meant to grow a third archived release cheaply.
    """
    archive = tmp_path / "videos_chunked_01.zip"
    with zipfile.ZipFile(archive, "w") as writing:
        # Written first, so an implementation that does not skip it creates `data` as a file
        # before the entry that needs `data` to be a directory.
        writing.writestr(zipfile.ZipInfo("data/"), b"")
        writing.writestr("data/one.mp4", b"frames")

    _extract(archive, announce=None)

    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "one.mp4").read_bytes() == b"frames"


def test_an_archive_naming_a_path_outside_itself_is_refused(tmp_path: Path) -> None:
    """These volumes come from the network, and extraction runs with the operator's own rights.

    One `..` rather than several on purpose: a guard rooted one directory too high still refuses
    a deep escape, so only the shallowest one says the base is the volume's own directory.
    """
    archive = _archive(tmp_path / "hostile.zip", {"../escaped.mp4": b"frames"})

    with pytest.raises(ValueError, match="escapes"):
        _extract(archive, announce=None)

    assert not (tmp_path.parent / "escaped.mp4").exists()


def test_a_fetch_narrowed_to_many_exact_paths_is_announced_as_a_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A producer that narrows to the files it is missing passes one pattern per file.

    ATM-Bench on an empty corpus is 4,292 of them and EgoLife 6,266, so joining them would put
    hundreds of kilobytes on one line immediately before the transfer they are announcing.
    """
    said: list[str] = []
    _record_downloads(monkeypatch)
    many = tuple(f"data/raw_memory/image/{index}.jpg" for index in range(5))

    ensure_media("atm-bench", root=tmp_path, only=many, announce=said.append)

    assert said == ["fetching atm-bench media from Jingbiao/ATM-Bench@78e826dc07e9: 5 patterns"]


def test_a_fetch_narrowed_to_a_few_paths_still_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting the common case would hide which video a one-unit run is about to spend on."""
    said: list[str] = []
    _record_downloads(monkeypatch)

    ensure_media("egolife", root=tmp_path, only=("A1_JAKE/DAY1/*",), announce=said.append)

    assert said == ["fetching egolife media from lmms-lab/EgoLife@143fb319be7a: A1_JAKE/DAY1/*"]


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


def test_no_download_refuses_absent_media_rather_than_fetching_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--no-download` has to reach here, not only the pre-flight over annotations.

    Preparation is where a media benchmark's bytes are actually obtained, so before this the flag
    governed the 40 MB of annotations and let the 94 GiB behind them through -- a flag named
    `--no-download` that permits the largest download in the tree. `ensure_media` is only ever
    called because a file is already missing, so refusing is the whole of the correct behaviour.

    The fetch is replaced rather than trusted not to happen. A regression here does not fail an
    assertion, it starts Video-MME's 94 GiB against the real Hub -- which is how this test first
    ran, and it hung rather than failed.
    """

    def _refuse(*arguments: object, **keywords: object) -> None:
        raise AssertionError("--no-download must refuse before anything reaches the Hub")

    monkeypatch.setattr(releases, "_download_from_hub", _refuse)
    with pytest.raises(ValueError, match=r"--no-download was given"):
        ensure_media("video-mme", root=tmp_path, download=False)


def test_no_download_still_reports_an_unobtainable_set_as_unobtainable(tmp_path: Path) -> None:
    """The flag must not turn a licensing wall into a download complaint.

    `egotempo` can never be fetched by any flag, so the operator's instructions are the useful
    answer whether or not downloading was permitted; reporting `--no-download` instead would send
    them to drop a flag that was never what stood in the way.
    """
    with pytest.raises(FileNotFoundError, match=r"signed access agreement"):
        ensure_media("egotempo", root=tmp_path, download=False)


def test_a_gated_release_names_the_terms_to_accept_rather_than_a_missing_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gated dataset is the one failure whose whole fix lives outside this program.

    `GatedRepoError` subclasses `RepositoryNotFoundError`, so catching the parent first reports a
    dataset the user can see in their browser as one that does not exist -- and sends them looking
    for a typo in a name that is correct. The assertion is on the terms URL and the authorisation
    step, because those two sentences are the entire remedy.
    """
    import huggingface_hub
    from huggingface_hub.errors import GatedRepoError

    def _gated(**_: object) -> None:
        # Built the way the Hub client raises it: `HfHubHTTPError` requires the response, and a
        # stub without one raises TypeError instead, which this test would then pass on for the
        # wrong reason.
        forbidden = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co"))
        raise GatedRepoError("403 Client Error: Forbidden", response=forbidden)

    # Patched on `huggingface_hub` rather than on `releases`: `_download_from_hub` imports the
    # symbol inside the function body, so a name bound on this module is never the one it calls
    # and `raising=False` would have hidden that by creating an attribute nothing reads.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _gated)
    monkeypatch.setitem(releases.MEDIA, "gated-probe", releases.Media("video-mme", ("data/*.mp4",)))

    with pytest.raises(PermissionError) as raised:
        ensure_media("gated-probe", root=tmp_path)

    message = str(raised.value)
    assert "huggingface.co/datasets/lmms-eval/Video-MME" in message, "names the terms to accept"
    assert "hf auth login" in message and "HF_TOKEN" in message, "names how to authorise"
