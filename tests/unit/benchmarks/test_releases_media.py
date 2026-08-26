"""Checks for the media half of the release table — what a producer's source files come from.

Nothing here reaches the network or the corpus. What is checked is everything that decides
*which* bytes would be asked for and *whether* asking twice costs anything: the join from a media
set to the release holding it, the narrowing that keeps a small run off a large release, the
refusal of a media set no download can obtain, and the extraction rule that has to survive being
interrupted halfway through 94 GiB.
"""

from __future__ import annotations

import hashlib
import sys
import types
import zipfile
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from mindbridge.benchmarks import releases
from mindbridge.benchmarks.releases import (
    ACQUIRERS,
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


def test_asking_for_media_nothing_can_obtain_says_so_and_says_what_to_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set with no acquirer refuses by name, rather than being discovered as an absent file.

    Asserted against an entry made here rather than a real one: both real entries now have an
    acquirer, so `ensure_media` dispatches for them and this branch has no live example -- but a
    set added to `UNOBTAINABLE` before anyone writes its acquirer is exactly the state this arm
    is for, and it is one commit away at any time.
    """
    monkeypatch.setitem(
        releases.UNOBTAINABLE,
        "probe",
        "nothing can fetch it; put it in <benchmarks-root>/probe/videos/<clip_id>.mp4 yourself",
    )

    with pytest.raises(FileNotFoundError) as caught:
        ensure_media("probe", root=tmp_path, announce=_never)

    assert "<benchmarks-root>/probe/videos/<clip_id>.mp4" in str(caught.value)


def test_asking_for_a_media_set_that_does_not_exist_lists_the_ones_that_do(
    tmp_path: Path,
) -> None:
    """Four producers type these keys by hand; a bare KeyError makes a typo a puzzle."""
    with pytest.raises(KeyError) as caught:
        ensure_media("video_mme", root=tmp_path, announce=_never)

    message = str(caught.value)
    assert "video-mme" in message
    assert "m3-web" in message, "the unobtainable sets are keys a producer can legitimately try"


def test_the_pin_and_the_narrowing_both_reach_the_hub_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other test here replaces `_download_from_hub`, so none of them can see this.

    A double that accepts an argument and drops it is a fixture lying about the world, and the
    argument being dropped here would be the pin -- the one property this module exists to hold,
    since a branch name makes one task name mean different bytes on different days. `MEDIA` being
    pinned and the pin reaching the Hub client are two halves, and the table test only proves the
    first. Faked one level deeper for that reason: at `snapshot_download` rather than at the
    function that calls it, which is the only level where the request itself is observable.
    """
    import huggingface_hub

    asked: list[dict[str, object]] = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: asked.append(kwargs))

    ensure_media("egolife", root=tmp_path, only=("A1_JAKE/DAY1/*",))

    assert asked == [
        {
            "repo_id": "lmms-lab/EgoLife",
            "repo_type": "dataset",
            # Compared against the table rather than a copied literal: a second copy of the
            # revision here would agree with a table that had been changed to a branch.
            "revision": RELEASES["egolife"].revision,
            "allow_patterns": ["A1_JAKE/DAY1/*"],
            "local_dir": str(tmp_path / "egolife"),
        }
    ]


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

    Telling an operator to drop `--no-download` implies that dropping it would work. Without an
    Ego4D signature it would not, so the flag cannot be the whole message -- which is what this
    asserted before anything could acquire these two, by reporting the wall and not the flag.

    Both now, because after `ACQUIRERS` exactly one of the two facts is knowable. The flag is
    certain and the signature is not, and finding out would mean probing for a credential inside
    the one flag that says not to go and look. Naming only the wall would send an operator who
    does hold the CLI and the credential off to fetch 920 videos by hand that dropping the flag
    would have fetched for them, so the fix is not to choose.
    """
    with pytest.raises(FileNotFoundError, match=r"signed access agreement") as caught:
        ensure_media("egotempo", root=tmp_path, download=False)

    message = str(caught.value)
    assert "--no-download" in message, "the flag that refused is the operator's own doing"
    assert "ego4d-data.org" in message, "and the wall it may not be enough to lift"


def test_no_download_still_reports_a_set_with_no_acquirer_as_unobtainable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A set nothing can obtain is not waiting on permission to download.

    The other arm of the same rule, kept because both real entries now have an acquirer: a row
    added to `UNOBTAINABLE` before its acquirer exists must report the wall alone, since here
    dropping the flag really would change nothing.
    """
    monkeypatch.setitem(
        releases.UNOBTAINABLE, "probe", "nobody can fetch it; see <benchmarks-root>"
    )

    with pytest.raises(FileNotFoundError, match=r"nobody can fetch it") as caught:
        ensure_media("probe", root=tmp_path, download=False)

    assert "--no-download" not in str(caught.value), "no flag is what stands in the way here"


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


def test_every_acquired_media_set_is_named_by_the_three_tables_it_has_to_agree_with() -> None:
    """An acquired set is a row in `ACQUIRERS`, a fallback in `UNOBTAINABLE`, and not in `MEDIA`.

    All three, or the dispatch is silently wrong in a way no producer can see. Missing from
    `UNOBTAINABLE`, `_acquire` raises `KeyError` from inside its own error path -- the operator
    loses the one sentence that would have unblocked them, at the moment they need it. Present in
    `MEDIA` as well, which behaviour you get depends on which table is consulted first.
    """
    assert set(ACQUIRERS) == {"egotempo", "m3-web"}
    assert set(ACQUIRERS) <= set(UNOBTAINABLE), "an acquisition with no manual fallback"
    assert set(ACQUIRERS) & set(MEDIA) == set()
    for name, acquirer in ACQUIRERS.items():
        assert acquirer.release in RELEASES, f"{name} lands in a directory no release owns"
        assert acquirer.annotations, f"{name} declares no annotation, so it has no input"
        for annotation in acquirer.annotations:
            assert annotation.split("/")[0] in RELEASES, f"{name} reads {annotation}"


def test_the_annotation_an_acquirer_reads_is_one_a_catalog_task_already_declares() -> None:
    """`ACQUIRERS` restates a path the catalog also spells, and this is what holds the two together.

    It has to be restated: `releases` cannot import the catalog, because the catalog imports the
    producers and the producers import `releases`. So the duplication is real, and a drifted copy
    would be a fetch of the wrong file followed by an acquirer reading an absent one -- which is
    the failure the fetch was added to prevent. Asserted against the catalog's own declared
    inputs rather than against a literal, so moving a dataset path fails here.
    """
    from mindbridge.benchmarks.task_catalog import TASKS

    root = Path("/corpus")
    declared = {path for task in TASKS.values() for path in task.inputs(root=root)}

    for name, acquirer in ACQUIRERS.items():
        for annotation in acquirer.annotations:
            assert root / annotation in declared, f"{name} reads {annotation}, which no task does"


def test_an_acquired_media_set_is_handed_to_its_acquirer_with_the_units_it_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the seam: `ensure_media` dispatches instead of refusing.

    `only` is asserted because it is not a nicety here. `m3-web` is 920 live URLs and EgoTempo is
    one Ego4D span per clip, so an acquirer handed no narrowing at all does the entire release to
    satisfy a `--limit 1` run. The corpus root is passed rather than the release directory, which
    is the same contract `Media` has: the acquirer joins its own layout.
    """
    seen: list[dict[str, object]] = []
    _install_acquirer(monkeypatch, tmp_path, acquire=lambda **kwargs: seen.append(kwargs))

    destination = ensure_media("egotempo", root=tmp_path, only=("videos/one.mp4",))

    assert destination == tmp_path / "egotempo"
    assert seen == [{"root": tmp_path, "only": ("videos/one.mp4",), "announce": None}]


def test_the_annotation_an_acquirer_reads_is_on_disk_before_it_is_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An acquirer's inputs are inside a release file, so the file is a precondition of calling it.

    The sweep fetches annotations in its own pre-flight, but only for catalog tasks and only for
    the path in argv -- while an acquirer opens the path it derives itself. `--suite`, a bare
    runner invocation, and a `--dataset` pointing elsewhere all reach a producer without it, so
    the ordering is asserted here, where every one of those paths goes through.

    The acquirer reads the file rather than checking that it exists, because that is what its real
    body does with it: recording the order alone would pass on a fetch that wrote the bytes after
    the call.
    """
    order: list[str] = []
    body = '{"annotations": []}'
    # The recorded digest is moved to these bytes rather than switched off, so the fetch this test
    # asserts the ordering of is the same fetch that verifies -- see the test below for what the
    # verification does when they disagree.
    monkeypatch.setitem(
        releases.RECORDED_DIGESTS,
        "egotempo_openQA.json",
        hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )

    def write_annotation(release: Release, within: str, *, destination: Path) -> None:
        order.append(f"fetched {within}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")

    def acquire(*, root: Path, only: Sequence[str], announce: object) -> None:
        order.append((root / "egotempo" / "egotempo_openQA.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(releases, "_download_from_git", write_annotation)
    _install_acquirer(monkeypatch, tmp_path, acquire=acquire, annotation=False)

    ensure_media("egotempo", root=tmp_path)

    assert order == ["fetched egotempo_openQA.json", body]


def test_a_drifted_annotation_stops_the_acquisition_rather_than_driving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The annotation is where every URL and every clip span comes from, so its bytes decide 920
    downloads. Fetching it through `fetch` rather than checking that it exists is what puts the
    recorded digest in front of them: upstream changing the release stops the run here instead of
    acquiring a corpus that no longer matches the questions anyone will be scored on."""
    reached: list[dict[str, object]] = []

    def write_other_bytes(release: Release, within: str, *, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("upstream rewrote this", encoding="utf-8")

    monkeypatch.setattr(releases, "_download_from_git", write_other_bytes)
    _install_acquirer(
        monkeypatch, tmp_path, acquire=lambda **kwargs: reached.append(kwargs), annotation=False
    )

    with pytest.raises(ValueError, match=r"hashed to .*, not the .* recorded in"):
        ensure_media("egotempo", root=tmp_path)

    assert reached == [], "a drifted annotation must not reach the acquirer at all"


def test_an_acquirer_that_is_not_installed_falls_back_to_doing_it_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prerequisite this package does not depend on is absent as an `ImportError`.

    `yt-dlp`, the Ego4D CLI, or the acquirer module itself in an installation that predates it.
    None of those is a broken build, and all three leave the operator in exactly the position
    `UNOBTAINABLE` was written for -- so the sentence is the fallback rather than a traceback
    about an import. Chained, so `MINDBRIDGE_TRACEBACK=1` still names the module.
    """
    _install_acquirer(monkeypatch, tmp_path, module="mindbridge.benchmarks.acquire_nothing")

    with pytest.raises(FileNotFoundError) as caught:
        ensure_media("egotempo", root=tmp_path)

    message = str(caught.value)
    assert "ego4d-data.org" in message, "the manual instruction is the fallback"
    assert "<benchmarks-root>/egotempo/videos/<clip_id>.mp4" in message, "and its destination"
    assert isinstance(caught.value.__cause__, ImportError)


def test_a_prerequisite_missing_inside_the_acquisition_is_still_a_missing_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An acquirer's third-party tool is not necessarily imported at its module scope.

    The Ego4D one imports the `ego4d` CLI inside `acquire`, after checking whether the corpus is
    already cut -- so on an install without it the `ImportError` comes from the call, not from
    `import_module`. Caught only around the import, the ordinary missing-prerequisite case fell
    into the generic arm and read as an unexpected failure rather than as the thing to install.
    """

    def missing_tool(**_: object) -> None:
        raise ImportError("no module named 'ego4d'; install it with `uv pip install ego4d`")

    _install_acquirer(monkeypatch, tmp_path, acquire=missing_tool)

    with pytest.raises(FileNotFoundError) as caught:
        ensure_media("egotempo", root=tmp_path)

    message = str(caught.value)
    assert "cannot be acquired here" in message, "the missing-prerequisite arm, not the generic one"
    assert "uv pip install ego4d" in message, "and what to install"
    assert "ego4d-data.org" in message


def test_an_acquisition_that_fails_says_what_failed_as_well_as_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing credential and a dead URL both surface inside the acquisition, not before it.

    Both sentences are needed and neither is enough: what went wrong is the only thing that says
    whether to retry, and the manual instruction is the only thing that says what to do if not.
    """

    def refuse(**_: object) -> None:
        raise RuntimeError("no Ego4D credential in ~/.aws/credentials")

    _install_acquirer(monkeypatch, tmp_path, acquire=refuse)

    with pytest.raises(FileNotFoundError) as caught:
        ensure_media("egotempo", root=tmp_path)

    message = str(caught.value)
    assert "no Ego4D credential" in message, "what actually went wrong"
    assert "ego4d-data.org" in message, "and what to do about it"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_an_interrupted_acquisition_stays_an_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """920 downloads is where a sweep gets interrupted, and `suite` turns that into a summary.

    Reduced to `FileNotFoundError` here, `_run_task_prepared` would report the task as failed,
    `_run_plans` would carry on into the next benchmark, and the operator's Ctrl-C would start
    the next acquisition instead of stopping the sweep.
    """

    def interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    _install_acquirer(monkeypatch, tmp_path, acquire=interrupt)

    with pytest.raises(KeyboardInterrupt):
        ensure_media("egotempo", root=tmp_path)


def test_no_download_refuses_an_acquisition_before_it_reaches_the_acquirer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-download` governs the largest fetch in the tree, which is now an acquisition.

    Refused before the import and before the annotation fetch, which is what makes the flag hold
    where the acquirer is not installed at all -- and what keeps a flag named `--no-download`
    from performing the pinned download of an annotation on its way to refusing. What the message
    says is the neighbouring test's subject; this one is that nothing happened.
    """
    _install_acquirer(
        monkeypatch,
        tmp_path,
        acquire=lambda **_: pytest.fail("--no-download must refuse before the acquirer runs"),
    )
    monkeypatch.setattr(
        releases,
        "fetch",
        lambda *_, **__: pytest.fail("--no-download must refuse before the annotation fetch"),
    )

    with pytest.raises(FileNotFoundError, match=r"--no-download was given"):
        ensure_media("egotempo", root=tmp_path, download=False, announce=_never)


def _install_acquirer(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    acquire: object = None,
    module: str = "mindbridge_probe_acquirer",
    annotation: bool = True,
) -> None:
    """Register an acquirer for `egotempo` that is this test rather than the Ego4D CLI.

    The real key rather than a synthetic one, because `_acquire` reads `UNOBTAINABLE[release]`
    for its fallback and a made-up key would raise `KeyError` from inside the error path -- which
    is a thing worth failing on, and is asserted by the table test rather than papered over here.

    The annotation is written by default so `fetch` finds it present and no test that is about
    something else reaches the network. `annotation=False` is for the one test that is about the
    fetch itself.
    """
    if annotation:
        path = root / "egotempo" / "egotempo_openQA.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    if acquire is not None:
        probe = types.ModuleType(module)
        probe.acquire = acquire  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module, probe)
    monkeypatch.setitem(
        releases.ACQUIRERS,
        "egotempo",
        releases.Acquirer(
            release="egotempo",
            module=module,
            annotations=ACQUIRERS["egotempo"].annotations,
        ),
    )
