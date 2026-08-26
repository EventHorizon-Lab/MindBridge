"""Checks for the M3-Bench web split's acquisition, which is 920 YouTube URLs and no files.

`yt-dlp` is never run. What is checked is everything around it: the argument vector, which is
where the pacing and the resume archive live; which videos are asked for, because narrowing is
all that separates a `--limit 1` run from a working day of downloads; and what happens to the
ones that fail, since a corpus collected from a public site loses entries permanently and a run
has to walk past those while stopping dead when the source is refusing the address.

The one thing that is real is the file each fake download leaves behind. A downloaded video is
checked for an audio stream, and that check is only worth having if the test writes a container
this repository's own decoder opens -- the regression it exists for was a corpus that was
silently mute while every count said 920 of 920.

Video IDs here are `v0`, `v1`, ... because `load_m3_bench` returns the annotation sorted by ID.
The acquisition order is therefore that sort and not the JSON's own order, which is worth naming:
a test that assumed otherwise would pass or fail on the names it happened to choose.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from mindbridge.benchmarks import acquire_youtube
from mindbridge.benchmarks.acquire_youtube import (
    ARCHIVE_NAME,
    CONSECUTIVE_BLOCK_LIMIT,
    DEFAULT_SLEEP_SECONDS,
    SLEEP_SECONDS_VARIABLE,
    WEB_VIDEOS,
    acquire,
)

pytest.importorskip("av", reason="a downloaded video's audio track is probed with the media extra")

_PRIVATE = "ERROR: [youtube] v1: Private video. Sign in if you have been granted access"
_UNAVAILABLE = (
    "ERROR: [youtube] Video unavailable. This video is no longer available because the "
    "uploader has closed their YouTube account"
)
_BOT_CHECK = "ERROR: [youtube] Sign in to confirm you are not a bot. Use --cookies-from-browser"
_FORBIDDEN = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
_LOGIN_WALL = (
    "ERROR: [youtube] v1: Please sign in. Use --cookies-from-browser or --cookies for the "
    "authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ  for how to manually pass "
    "cookies"
)
"""Verbatim from a real annotation URL, alongside five from the same address that succeeded.

One word from the bot check, and the opposite classification. Kept as the literal text because
paraphrasing it is what would let the two drift back together.
"""


class _FakeYtDlp:
    """Stands in for the process: records each argv and leaves the file it claims to have got.

    Named from the URL's `v=` parameter, which is how `--output %(id)s.%(ext)s` names it for real,
    so a test that asks for one video and is handed another's bytes fails on the file name.
    """

    def __init__(
        self,
        *,
        reasons: dict[str, str] | None = None,
        mute: Sequence[str] = (),
        land: bool = True,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.reasons = reasons or {}
        self.mute = set(mute)
        self.land = land

    def __call__(self, command: Sequence[str]) -> str | None:
        self.commands.append(tuple(command))
        video_id = _video_id(command)
        reason = self.reasons.get(video_id)
        if reason is not None:
            return reason
        if self.land:
            destination = Path(_argument(command, "--paths"))
            content = _sample(audio=video_id not in self.mute)
            (destination / f"{video_id}.mp4").write_bytes(content)
        return None

    @property
    def requested(self) -> list[str]:
        return [_video_id(command) for command in self.commands]


def test_the_argv_paces_every_download_and_records_it_beside_the_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pacing and the resume archive are the whole of this module's contract with `yt-dlp`."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0",))

    acquire(root=tmp_path, announce=None)

    (command,) = downloads.commands
    assert command[:2] == ("uvx", "yt-dlp@latest")
    assert _argument(command, "--download-archive") == str(tmp_path / WEB_VIDEOS / ARCHIVE_NAME)
    assert _argument(command, "--paths") == str(tmp_path / WEB_VIDEOS)
    assert _argument(command, "--output") == "%(id)s.%(ext)s"
    assert _argument(command, "--sleep-interval") == f"{DEFAULT_SLEEP_SECONDS:.0f}"
    assert _argument(command, "--max-sleep-interval") == f"{DEFAULT_SLEEP_SECONDS * 2:.0f}"
    assert float(_argument(command, "--sleep-requests")) >= 1.0
    assert "--no-playlist" in command
    # The URL is release-supplied text, so it arrives after `--` and cannot become a flag.
    assert command[-2:] == ("--", "https://www.youtube.com/watch?v=v0")


def test_the_pacing_is_configurable_and_an_unreadable_setting_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rate that works is a property of the address, so it has to be turnable from outside."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1"))
    monkeypatch.setenv(SLEEP_SECONDS_VARIABLE, "8")

    acquire(root=tmp_path, only=("videos/web/v0.mp4",))

    (command,) = downloads.commands
    assert _argument(command, "--sleep-interval") == "8"
    assert _argument(command, "--max-sleep-interval") == "16"
    assert _argument(command, "--sleep-requests") == "2"

    monkeypatch.setenv(SLEEP_SECONDS_VARIABLE, "half a minute")
    with pytest.raises(ValueError, match="not a number of seconds"):
        acquire(root=tmp_path, only=("videos/web/v1.mp4",))
    assert downloads.requested == ["v0"], "an unreadable setting must not reach a download"


def test_only_narrows_to_the_units_a_limit_run_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is what stands between `--limit 1` and 920 paced downloads.

    The entries are paths relative to the release directory, which is what `ensure_media` means by
    `only` and what `prepare_m3` passes, and the video ID is the stem of each.
    """
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1", "v2"))

    acquire(root=tmp_path, only=("videos/web/v1.mp4",))

    assert downloads.requested == ["v1"]
    assert (tmp_path / WEB_VIDEOS / "v1.mp4").exists()
    assert not (tmp_path / WEB_VIDEOS / "v0.mp4").exists()


def test_an_only_entry_no_annotation_names_is_refused_rather_than_fetching_everything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo that fell back to the whole release would cost a day of downloads to notice."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1"))

    with pytest.raises(ValueError, match="unknown M3-Bench web video IDs: nope"):
        acquire(root=tmp_path, only=("videos/web/nope.mp4",))

    assert downloads.commands == []


def test_a_video_already_on_disk_costs_no_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running has to resume, and a paced download is far too expensive to repeat."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1"))
    lines: list[str] = []

    acquire(root=tmp_path, only=("videos/web/v0.mp4",), announce=lines.append)
    acquire(root=tmp_path, only=("videos/web/v0.mp4",), announce=lines.append)

    assert downloads.requested == ["v0"]
    assert lines[-1] == "1 M3-Bench web videos are ready"


def test_an_archive_entry_that_outlived_its_file_is_reported_not_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`yt-dlp` exits 0 for an ID already in the archive, whether or not the file survived."""
    _prepared(tmp_path, monkeypatch, ("v0",), land=False)

    with pytest.raises(RuntimeError, match=ARCHIVE_NAME):
        acquire(root=tmp_path)


def test_a_muxed_file_with_no_audio_is_a_failure_rather_than_an_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent corpus reporting 920 of 920 is the regression this exists for."""
    _prepared(tmp_path, monkeypatch, ("v0", "v1"), mute=("v1",))
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert (tmp_path / WEB_VIDEOS / "v1.mp4").exists(), "the file arrived; it is unusable anyway"
    assert lines[-1] == (
        "1 of 2 M3-Bench web videos are ready; 1 are not: "
        "v1 (v1.mp4 was muxed with no audio stream)"
    )


def test_a_mute_file_already_on_disk_is_reported_on_every_later_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-fetching it would produce the same bytes, so the report is what has to keep happening."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1"))
    mute = tmp_path / WEB_VIDEOS / "v0.mp4"
    mute.parent.mkdir(parents=True)
    mute.write_bytes(_sample(audio=False))
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v1"], "a present file is not re-fetched even when it is mute"
    assert "1 of 2 M3-Bench web videos are ready" in lines[-1]
    assert "no audio stream" in lines[-1]


def test_a_file_on_disk_that_cannot_be_read_is_reported_not_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated container is one video's problem; raising would end the run at hour six."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1"))
    corrupt = tmp_path / WEB_VIDEOS / "v0.mp4"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not a video at all")
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v1"]
    assert "1 of 2 M3-Bench web videos are ready" in lines[-1]
    assert "v0 (v0.mp4 on disk has no audio stream" in lines[-1]


def test_one_dead_video_does_not_stop_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion and privacy are permanent facts about a public corpus, not a broken run."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0", "v1", "v2"), reasons={"v1": _PRIVATE})
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v0", "v1", "v2"]
    assert "2 of 3 M3-Bench web videos are ready" in lines[-1]
    assert f"v1 ({_PRIVATE})" in lines[-1]


def test_a_cluster_of_unavailable_videos_never_stops_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Annotations are grouped by source, so a terminated channel kills a contiguous run of them.

    Well past `CONSECUTIVE_BLOCK_LIMIT`, and none of it is evidence about this address.
    """
    dead = tuple(f"v{index + 1}" for index in range(CONSECUTIVE_BLOCK_LIMIT + 3))
    downloads = _prepared(
        tmp_path,
        monkeypatch,
        ("v0", *dead),
        reasons=dict.fromkeys(dead, _UNAVAILABLE),
    )
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v0", *dead]
    assert f"1 of {len(dead) + 1} M3-Bench web videos are ready" in lines[-1]


def test_a_login_walled_video_is_not_read_as_the_address_being_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Please sign in.` is a video this has no cookies for, not a throttle to back off from.

    A cluster of them past `CONSECUTIVE_BLOCK_LIMIT` has to be walked, or one login-walled
    channel's worth of annotation entries would stop the whole acquisition.
    """
    walled = tuple(f"v{index + 1}" for index in range(CONSECUTIVE_BLOCK_LIMIT + 1))
    downloads = _prepared(
        tmp_path, monkeypatch, ("v0", *walled), reasons=dict.fromkeys(walled, _LOGIN_WALL)
    )
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v0", *walled]
    assert f"1 of {len(walled) + 1} M3-Bench web videos are ready" in lines[-1]


def test_consecutive_throttles_stop_the_run_with_the_rest_unattempted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bot detection starts around ten downloads; grinding on proves it at 30 seconds a video."""
    throttled = tuple(f"v{index + 1}" for index in range(CONSECUTIVE_BLOCK_LIMIT))
    unattempted = ("v8", "v9")
    downloads = _prepared(
        tmp_path,
        monkeypatch,
        ("v0", *throttled, *unattempted),
        reasons=dict.fromkeys(throttled, _BOT_CHECK),
    )

    with pytest.raises(RuntimeError, match="failed in a row") as failure:
        acquire(root=tmp_path)

    assert downloads.requested == ["v0", *throttled]
    assert f"{len(unattempted)} were not attempted" in str(failure.value)
    # Something had succeeded first, so this is throttling rather than a stale yt-dlp.
    assert SLEEP_SECONDS_VARIABLE in str(failure.value)


def test_a_stop_before_anything_succeeded_names_an_out_of_date_yt_dlp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403 is ambiguous, and a stale copy fails on the first video rather than the eleventh."""
    blocked = tuple(f"v{index}" for index in range(CONSECUTIVE_BLOCK_LIMIT))
    _prepared(tmp_path, monkeypatch, blocked, reasons=dict.fromkeys(blocked, _FORBIDDEN))

    with pytest.raises(RuntimeError, match="out-of-date yt-dlp") as failure:
        acquire(root=tmp_path)

    assert SLEEP_SECONDS_VARIABLE not in str(failure.value)


def test_nothing_usable_at_all_raises_rather_than_reporting_a_finished_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`prepare_m3`'s own message would blame the 2 GB Hub release, which the web split is not."""
    _prepared(
        tmp_path,
        monkeypatch,
        ("v0", "v1"),
        reasons={"v0": _UNAVAILABLE, "v1": _PRIVATE},
    )

    with pytest.raises(RuntimeError, match="none of the 2 M3-Bench web videos"):
        acquire(root=tmp_path)


def test_a_url_that_is_not_http_never_reaches_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An annotation is release-supplied text, and `yt-dlp` reads local paths and stdin too."""
    downloads = _prepared(tmp_path, monkeypatch, ("v0",))
    _write_annotation(
        tmp_path,
        {"v0": "", "v1": "--config-location=/etc/passwd", "v2": None},
    )
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert downloads.requested == ["v0"]
    assert "1 of 3 M3-Bench web videos are ready" in lines[-1]
    assert "is not an HTTP URL" in lines[-1]
    assert "no video_url" in lines[-1]


class _FakeProcess:
    """Stands in for `subprocess.run` itself, so the stderr that reaches classification is real.

    The double one level down replaces `_run` and hands back a reason directly; this one leaves
    that parsing in place, because which line of stderr becomes the reason is the thing under test.
    """

    def __init__(self, *, dead: Sequence[str], stderr: str) -> None:
        self.dead = set(dead)
        self.stderr = stderr

    def __call__(self, command: Sequence[str], **_: object) -> subprocess.CompletedProcess[str]:
        video_id = _video_id(command)
        if video_id in self.dead:
            return subprocess.CompletedProcess(list(command), 1, stdout="", stderr=self.stderr)
        destination = Path(_argument(command, "--paths"))
        (destination / f"{video_id}.mp4").write_bytes(_sample(audio=True))
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")


def test_a_trailing_warning_does_not_hide_the_error_a_failure_is_classified_by(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every YouTube call today ends with a deprecation warning about the JavaScript runtime.

    Reading the last line of stderr would classify each of these removed videos as evidence about
    this address, and `CONSECUTIVE_BLOCK_LIMIT` of them would then stop a run that should walk
    straight past them.
    """
    dead = tuple(f"v{index + 1}" for index in range(CONSECUTIVE_BLOCK_LIMIT))
    _corpus(tmp_path, monkeypatch, ("v0", *dead))
    monkeypatch.setattr(
        subprocess,
        "run",
        _FakeProcess(
            dead=dead,
            stderr=(
                f"WARNING: [youtube] falling back\n{_UNAVAILABLE}\n"
                "WARNING: [youtube] No supported JavaScript runtime could be found\n"
            ),
        ),
    )
    lines: list[str] = []

    acquire(root=tmp_path, announce=lines.append)

    assert f"1 of {len(dead) + 1} M3-Bench web videos are ready" in lines[-1]
    assert _UNAVAILABLE in lines[-1]
    assert "JavaScript runtime" not in lines[-1]


def test_yt_dlp_absent_raises_naming_the_form_that_cannot_rot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never a silent no-op: the media it would have fetched is the whole benchmark."""
    _prepared(tmp_path, monkeypatch, ("v0",))
    monkeypatch.setattr(shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="uvx yt-dlp@latest"):
        acquire(root=tmp_path)


def test_an_absent_annotation_says_the_annotations_have_not_been_fetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The URLs are a declared input of the task, so this is wiring, not a missing corpus."""
    monkeypatch.setattr(acquire_youtube, "_run", _FakeYtDlp())

    with pytest.raises(FileNotFoundError, match="have not been fetched yet"):
        acquire(root=tmp_path)


def _prepared(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_ids: Sequence[str],
    *,
    reasons: dict[str, str] | None = None,
    mute: Sequence[str] = (),
    land: bool = True,
) -> _FakeYtDlp:
    """Write the annotation these URLs come from and put a recording double in `yt-dlp`'s place."""
    _corpus(root, monkeypatch, video_ids)
    downloads = _FakeYtDlp(reasons=reasons, mute=mute, land=land)
    monkeypatch.setattr(acquire_youtube, "_run", downloads)
    return downloads


def _corpus(root: Path, monkeypatch: pytest.MonkeyPatch, video_ids: Sequence[str]) -> None:
    """The annotation, a resolvable `uvx`, and no pacing inherited from the ambient environment."""
    _write_annotation(root, dict.fromkeys(video_ids, ""))
    monkeypatch.delenv(SLEEP_SECONDS_VARIABLE, raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name == "uvx" else None)


def _write_annotation(root: Path, urls: dict[str, str | None]) -> None:
    """The smallest `web.json` the M3-Bench adapter accepts, one entry per video.

    An empty URL means the ordinary YouTube watch URL for that ID; `None` means the entry carries
    no `video_url` at all, which the adapter allows and the release does use.
    """
    path = root / "m3-agent" / "data" / "annotations" / "web.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        video_id: {
            "video_path": f"data/videos/web/{video_id}.mp4",
            **(
                {"video_url": url or f"https://www.youtube.com/watch?v={video_id}"}
                if url is not None
                else {}
            ),
            "qa_list": [
                {
                    "question": "What happened?",
                    "answer": "Something did",
                    "question_id": f"{video_id}_Q1",
                    "type": ["General"],
                }
            ],
        }
        for video_id, url in urls.items()
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _argument(command: Sequence[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _video_id(command: Sequence[str]) -> str:
    return parse_qs(urlparse(command[-1]).query)["v"][0]


@cache
def _sample(*, audio: bool) -> bytes:
    """One second of encoded video, with or without a sound track, cached across the module."""
    import tempfile

    import av
    import numpy

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.mp4"
        with av.open(str(path), mode="w") as container:
            video = container.add_stream("libx264", rate=5)
            video.width, video.height, video.pix_fmt = 128, 96, "yuv420p"
            video.thread_count, video.thread_type = 1, "NONE"
            sound = container.add_stream("aac", rate=16_000) if audio else None
            if sound is not None:
                sound.layout = "mono"
            for index in range(5):
                array = numpy.full((96, 128, 3), index % 255, dtype=numpy.uint8)
                container.mux(video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
            container.mux(video.encode())
            if sound is not None:
                _mux_silence(container, sound)
        return path.read_bytes()


def _mux_silence(container: Any, sound: Any) -> None:  # noqa: ANN401
    import av
    import numpy

    frame = av.AudioFrame.from_ndarray(
        numpy.zeros((1, 16_000), dtype=numpy.int16), format="s16", layout="mono"
    )
    frame.sample_rate = 16_000
    frame.pts = 0
    frame.time_base = Fraction(1, 16_000)
    for packet in sound.encode(frame):
        container.mux(packet)
    container.mux(sound.encode())
