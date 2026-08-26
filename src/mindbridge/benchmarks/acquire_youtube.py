"""Obtaining M3-Bench's web split, whose videos the release distributes as URLs, not as files.

Every other media set in `releases.py` is a download of published bytes at a pinned revision.
This one is 920 `video_url` fields in `m3-agent/data/annotations/web.json`, all of them YouTube,
so the corpus is assembled a video at a time from a site that does not want to be assembled from.
That difference is why this is a module of its own rather than another `MEDIA` entry: `yt-dlp` is
an external process, the failures are per video rather than per release, and the run is measured
in hours of deliberate waiting.

Six properties are forced rather than chosen.

**Paced, and sequentially.** YouTube starts refusing an address after roughly ten downloads, and
pacing is the only lever that has been made to work here. So one URL per invocation, one
invocation at a time, with `--sleep-interval` before each -- no `asyncio`, no worker pool, nothing
to run concurrently and therefore no `gather` to get wrong. Throughput is not the goal; finishing
is. `MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS` is the knob, because the rate that works is a
property of the address the operator is calling from and cannot be known here.

**No cookies.** `yt-dlp` will read a browser's cookie jar and that does raise the ceiling, which
is exactly the problem: it makes a benchmark corpus a function of one person's logged-in account,
and a score is not reproducible if reproducing it needs their session. Left out deliberately, so
an age-restricted video is a video this cannot have.

**`uvx yt-dlp@latest`, not a pinned copy.** YouTube changes its player and an out-of-date `yt-dlp`
answers HTTP 403 -- indistinguishable, from here, from the bot detection above (bilibili answers
412). Someone reading a wall of 403s will reach for the pacing knob, which cannot help, so the
version is resolved fresh at the point of use where it cannot rot into that misdiagnosis.

**No JavaScript runtime is installed here, and `yt-dlp` says so on every call.** Verbatim, on
every YouTube URL: `WARNING: [youtube] No supported JavaScript runtime could be found. Only deno
is enabled by default ... YouTube extraction without a JS runtime has been deprecated, and some
formats may be missing.` Nothing is broken by it today -- a probe of a real annotation URL still
resolved video and AAC audio in MP4 -- but it is where to look first if a format that should
exist is not offered, and `--js-runtimes RUNTIME[:PATH]` or an installed `deno` is the answer. No
runtime is depended on from here: it is a system package, this is a benchmark helper, and the
warning is louder than the problem. Its other consequence is already handled -- `_run` reads the
first `ERROR:` line rather than the last line of stderr, because this warning arrives *after* the
error and would otherwise have been reported as the reason a video failed, and then classified as
evidence about the address rather than about the video.

**The audio track is checked, not assumed.** This repository has already shipped a corpus whose
derived clips carried only h264 while the sources carried aac; the audio-visual benchmarks scored
below their own blind baselines and nothing failed, because nothing looked. A muxed file with no
audio stream is therefore a failure here rather than an acquisition, on the way in and on every
later run that finds it on disk.

**A shortfall is reported, never rounded up to success.** Removed, private and geoblocked videos
are normal in a 920-URL academic corpus; one of them must not abandon the other 919. So each
failure is collected with its reason and announced, and `prepare_m3` -- which calls this only
because a file it wants is already missing -- attributes the specific absent video afterwards.
What does raise is the two cases where continuing is pointless: nothing usable was obtained at
all, or enough consecutive failures of the kind that is *not* about a particular video that the
address is plainly being refused.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from mindbridge.benchmarks.cli_common import select_by_id
from mindbridge.benchmarks.m3_bench import M3BenchVideo, load_m3_bench
from mindbridge.benchmarks.staging import key_component

WEB_ANNOTATION = "m3-agent/data/annotations/web.json"
"""Where the URLs are, relative to the corpus root.

The same file `task_catalog.py` gives `m3-web` as `--dataset`: the annotation is a declared input
of the task and the media is not, so a sweep has fetched and digest-checked it long before
preparation asks for a video. Absence means this was called before that happened, which is a
wiring fault rather than a missing corpus, and `_annotation` says so in those words.
"""

WEB_VIDEOS = "m3-bench/videos/web"
"""Where the videos go, relative to the corpus root.

`prepare_m3` builds `<root>/m3-bench/videos/<subset>/<video_id>.mp4` and the runner requires the
prepared `video_id` to be the annotation's key, so this layout is the contract rather than a
preference.
"""

ARCHIVE_NAME = ".yt-dlp-archive.txt"
"""`--download-archive`, kept inside the video directory on purpose.

Beside the videos rather than above them so that deleting the videos deletes the archive. An
archive that outlives the files it describes is worse than no archive: `yt-dlp` skips every id
recorded in it, so a corpus someone cleared by hand would never refill.
"""

SLEEP_SECONDS_VARIABLE = "MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS"
DEFAULT_SLEEP_SECONDS = 30.0
"""Seconds before each download, jittered up to twice this.

Thirty is slow -- 920 videos is more than seven hours of waiting alone -- and slower than the
address needs on a good day. It is the default anyway: the alternative failure is being refused
part way through, which costs the whole remaining corpus rather than an afternoon.
"""

CONSECUTIVE_BLOCK_LIMIT = 3
"""How many failures that are not about a particular video mean the address is being refused.

Keyed on the class of failure rather than on a raw count, because the deaths in a corpus like
this one arrive in clusters: annotations are grouped by source and a terminated channel takes all
of its videos with it, so five consecutive genuine 404s is an ordinary afternoon with no
throttling anywhere. Those never reach this counter. What does -- a 429, a bot check, a 403, a
transport error -- is systemic by nature, so three in a row is already enough, and stopping there
is what saves the hours the remaining URLs would spend at `DEFAULT_SLEEP_SECONDS` each proving
something known.
"""

_PER_VIDEO_MARKERS = (
    "no video_url",
    "not an http url",
    "no audio stream",
    "http error 404",
    "video unavailable",
    "private video",
    "has been removed",
    "removed for violating",
    "has been terminated",
    "no longer available",
    "not available in your country",
    "not available from your location",
    "members-only",
    "join this channel",
    "please sign in",
    "sign in to confirm your age",
    "age-restricted",
    "who has blocked it",
)
"""Failure text that is a fact about one video rather than about this address.

Enumerated in this direction on purpose: the permanent reasons are a list someone can read and
extend, and the systemic ones are not -- a 429, a bot check, an expired extractor, a DNS failure
and a `yt-dlp` that crashed have no common phrase between them. So anything unrecognised counts
towards `CONSECUTIVE_BLOCK_LIMIT`, which errs towards stopping a run that has started to fail for
a reason nobody wrote down here.

Three of these entries are one word apart from the failure they must *not* match. YouTube's own
wording, observed against real annotation URLs: `Please sign in.` is a video behind a login,
`Sign in to confirm your age` is the age gate this has no cookies for, and
`Sign in to confirm you're not a bot` is the address being refused. The first two belong here and
the third must not, or the only signal that matters would be suppressed by the two that look like
it. `please sign in` is in this list on evidence rather than on reading: a probe of the first six
annotation URLs had five succeed and the sixth answer exactly that, from the same address in the
same minute, which is as direct as "this is about the video" gets.
"""

_TIMEOUT_SECONDS = 3_600.0
"""One video's own ceiling.

These are long videos on a deliberately slow connection, so this is protection against a `yt-dlp`
that has stopped making progress without exiting rather than a target to come in under.
"""

_FAILURES_LISTED = 10
"""How many failures a summary names before it counts the rest.

900 of them is one line of 60 KB, which buries the number in front of it.
"""


def acquire(
    *,
    root: Path,
    only: Sequence[str] = (),
    announce: Callable[[str], None] | None = None,
) -> None:
    """Download the web split's videos into the layout `prepare_m3` reads, and report the rest.

    `only` narrows to the videos a run will actually read, and is the knob that matters here more
    than anywhere else in this package: unnarrowed, this is 920 videos and a working day of
    pacing, so a `--limit 1` run that fetched all of them would make the flag meaningless.

    Its entries are paths relative to the release directory, which is what `ensure_media` already
    means by `only` -- `videos/web/<video_id>.mp4` -- and the video ID is the stem. One definition
    has to serve both tables, so the path form is the canonical one and this derives the ID rather
    than asking the caller for it in a second shape. Empty means the whole release. A stem no
    annotation names is refused rather than ignored: falling back to all 920 videos on a typo is
    the one mistake this cannot afford.

    Idempotent and resumable in two independent ways. A video already on disk with its audio
    intact costs nothing but a header read, not a process; anything past that is `yt-dlp`'s own
    `--download-archive`, which also covers a video that arrived under a name this cannot predict.
    Interruption leaves a `.part` file and no archive entry, so the next run redoes it -- what it
    cannot leave is a truncated file at the final name, which is why arrival is checked against
    the file rather than against the exit status alone.
    """
    videos = select_by_id(
        load_m3_bench(_annotation(root)),
        [PurePosixPath(entry).stem for entry in only],
        key=lambda video: video.video_id,
        label="M3-Bench web video IDs",
    )
    destination = root / WEB_VIDEOS
    pending, failures = _triage(videos, destination)
    if pending:
        command = _yt_dlp_command()
        sleep_seconds = _sleep_seconds()
        destination.mkdir(parents=True, exist_ok=True)
        _say(
            announce,
            f"acquiring {len(pending)} of {len(videos)} M3-Bench web videos with "
            f"{' '.join(command)}, paced at {sleep_seconds:.0f}-{sleep_seconds * 2:.0f}s per "
            "download; expect this to take hours",
        )
        failures += _fetch_each(pending, command, sleep_seconds, destination, announce)
    _report(wanted=len(videos), attempted=len(pending), failures=failures, announce=announce)


def _triage(
    videos: Sequence[M3BenchVideo],
    destination: Path,
) -> tuple[list[M3BenchVideo], list[str]]:
    """Split the selection into what has to be fetched and what is on disk but not usable.

    A file that is present is not re-fetched even when it is mute, because fetching it again
    produces the same bytes -- the archive already holds its ID. It is reported instead, on this
    run and on every run after it, until someone deletes it. That is the opposite of the failure
    this guards: a mute corpus that is reported once and silently accepted afterwards.
    """
    pending: list[M3BenchVideo] = []
    unusable: list[str] = []
    for video in videos:
        target = _target(destination, video.video_id)
        if not target.exists():
            pending.append(video)
        elif not _has_audio(target):
            unusable.append(
                f"{video.video_id} ({target.name} on disk has no audio stream; delete it and "
                f"remove its line from {ARCHIVE_NAME} to fetch it again)"
            )
    return pending, unusable


def _fetch_each(
    pending: Sequence[M3BenchVideo],
    command: Sequence[str],
    sleep_seconds: float,
    destination: Path,
    announce: Callable[[str], None] | None,
) -> list[str]:
    """Fetch each absent video in turn, stopping only if the source is refusing this address."""
    failures: list[str] = []
    blocks = 0
    acquired = 0
    for index, video in enumerate(pending, start=1):
        _say(announce, f"[{index}/{len(pending)}] {video.video_id}")
        reason = _acquire_one(video.video_id, video.video_url, command, sleep_seconds, destination)
        if reason is None:
            acquired += 1
            blocks = 0
            continue
        failures.append(f"{video.video_id} ({reason})")
        if _is_about_this_video(reason):
            continue
        blocks += 1
        if blocks >= CONSECUTIVE_BLOCK_LIMIT:
            raise RuntimeError(
                _blocked(
                    failures,
                    len(pending) - index,
                    sleep_seconds,
                    acquired=acquired,
                )
            )
    return failures


def _blocked(
    failures: Sequence[str],
    unattempted: int,
    sleep_seconds: float,
    *,
    acquired: int,
) -> str:
    """Why the run stopped, phrased for the two things it can be.

    Whether anything succeeded first is the disambiguator, and it is the whole difference between
    the two fixes. An out-of-date `yt-dlp` answers HTTP 403 on the very first video, because the
    player it knows how to read is the one that changed; throttling and bot detection arrive after
    a run has been going, which is when raising the pacing is the thing that helps. Counted over
    this run only -- what a download did yesterday says nothing about the binary resolved today.
    """
    diagnosis = (
        f"raise {SLEEP_SECONDS_VARIABLE} above {sleep_seconds:.0f} and re-run"
        if acquired
        else "check that these are not HTTP 403 from an out-of-date yt-dlp, which no amount of "
        "pacing fixes -- a stale copy fails on the first video, throttling only after several"
    )
    return (
        f"{CONSECUTIVE_BLOCK_LIMIT} M3-Bench web videos failed in a row for reasons that are not "
        f"about those videos, which is what being refused by the source looks like: "
        f"{_summarise(failures)}. {unattempted} were not attempted. What arrived is kept, along "
        f"with {ARCHIVE_NAME}, so re-running resumes rather than restarts; {diagnosis}"
    )


def _acquire_one(
    video_id: str,
    video_url: str | None,
    command: Sequence[str],
    sleep_seconds: float,
    destination: Path,
) -> str | None:
    """Fetch one video, returning None on success or why it is still not usable."""
    target = _target(destination, video_id)
    if video_url is None:
        return "the annotation carries no video_url"
    if urlparse(video_url).scheme not in {"http", "https"}:
        # The URL is release-supplied text on its way into an argument vector. `--` below stops
        # one beginning with `-` from becoming a flag; this stops `yt-dlp`'s non-HTTP inputs --
        # a local path, `-` for stdin -- from being reachable through an annotation at all.
        return f"{video_url!r} is not an HTTP URL"
    reason = _run((*command, *_options(destination, sleep_seconds), "--", video_url))
    if reason is not None:
        return reason
    if not target.exists():
        return (
            f"yt-dlp reported success but {target.name} is absent; if its ID is already in "
            f"{ARCHIVE_NAME} the download was recorded on an earlier run and the file has since "
            "been removed, so delete that line to fetch it again"
        )
    if not target.stat().st_size:
        return f"{target.name} was written empty"
    if not _has_audio(target):
        return f"{target.name} was muxed with no audio stream"
    return None


def _options(destination: Path, sleep_seconds: float) -> tuple[str, ...]:
    """The arguments every fetch shares: where it lands, what it is named, and how slowly."""
    return (
        "--paths",
        str(destination),
        "--output",
        "%(id)s.%(ext)s",
        "--download-archive",
        str(destination / ARCHIVE_NAME),
        # Prefer formats that are already MP4, force a merge into MP4, and remux anything that
        # still is not. All three, because the layout names `<video_id>.mp4` exactly: the sort
        # alone leaves WebM where no MP4 is offered, and `--merge-output-format` only governs the
        # container of a merge that happens. Remuxing does not re-encode.
        "--format-sort",
        "ext:mp4:m4a",
        "--merge-output-format",
        "mp4",
        "--remux-video",
        "mp4",
        # An annotation URL carrying `&list=` would otherwise pull a whole playlist for one ID.
        "--no-playlist",
        # `--sleep-interval` applies before every download including the first, and the ceiling
        # makes the wait random rather than a period anything can lock onto. `--sleep-requests`
        # paces the metadata requests too, at a fraction of the download interval -- at the full
        # value it would roughly double a run that is already measured in hours.
        "--sleep-interval",
        f"{sleep_seconds:.0f}",
        "--max-sleep-interval",
        f"{sleep_seconds * 2:.0f}",
        "--sleep-requests",
        f"{max(1.0, sleep_seconds / 4):.0f}",
        # Progress bars are carriage returns into a captured pipe nobody reads.
        "--no-progress",
    )


def _target(destination: Path, video_id: str) -> Path:
    """Where one video has to land for `prepare_m3` to find it.

    The ID becomes a file name, so it is held to one path component first. It is also what
    `--output` interpolates as `%(id)s`: the annotation's key is the site's own video ID, which
    is what lets the download name itself without a template per video.
    """
    return destination / f"{key_component(video_id, label='M3-Bench web video ID')}.mp4"


def _has_audio(path: Path) -> bool:
    """Whether a downloaded file actually carries sound, answering False if it cannot be read.

    Headers only -- `av.open` reads the container's stream table without decoding a frame, so this
    is cheap enough to run over an already-complete corpus on every call. `av` is the media extra
    and a caller that got this far needs it anyway: `prepare_m3` cuts these videos into clips with
    it immediately afterwards.

    An unreadable file answers False rather than raising, and the two are deliberately the same
    answer because the remedy is: delete it and fetch it again. A truncated container is a fact
    about one video -- a disk that filled during the remux is how one appears -- so it reports as
    that video being unusable. Raising would abandon the other 900 at whatever hour of the run it
    happened to be, which is the shape of failure this module exists to avoid.
    """
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        # Every way this can fail -- a truncated header, a container `av` has no demuxer for, a
        # permission error -- is a file that cannot be ingested, which is what the caller asked.
        return False


def _run(command: Sequence[str]) -> str | None:
    """Run one `yt-dlp` invocation, returning None on success or the reason it failed.

    Monkeypatched by the tests, which is why the process boundary is a function of its own.
    Nothing here parses `yt-dlp`'s output beyond the line it failed with -- but which line that is
    matters twice over, because this text is both what an operator reads and what
    `_is_about_this_video` classifies the failure by. So the first `ERROR:` is preferred over the
    last line of stderr: `yt-dlp` surrounds its errors with warnings -- today every YouTube call
    ends with a deprecation notice about there being no JavaScript runtime installed, which is
    also worth knowing about on its own, since without one `some formats may be missing` and
    `--js-runtimes` is the answer -- and taking the last line would have read a removed video's
    trailing warning as evidence that this address is being throttled.
    """
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"no exit after {_TIMEOUT_SECONDS:.0f}s"
    except OSError as error:
        return str(error)
    if completed.returncode == 0:
        return None
    lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
    errors = [line for line in lines if line.startswith("ERROR")]
    reported = errors or lines or [f"yt-dlp exited {completed.returncode}"]
    return reported[0]


def _is_about_this_video(reason: str) -> bool:
    lowered = reason.casefold()
    return any(marker in lowered for marker in _PER_VIDEO_MARKERS)


def _yt_dlp_command() -> tuple[str, ...]:
    """How to invoke `yt-dlp`, preferring the form that cannot be out of date.

    `uvx yt-dlp@latest` resolves the newest release on each call, which matters because YouTube
    breaks old versions and the symptom is HTTP 403 -- the same thing bot detection looks like,
    and the same thing an expired extractor looks like (bilibili answers 412). A copy on PATH is
    accepted second: it works until it silently does not.
    """
    if shutil.which("uvx"):
        return ("uvx", "yt-dlp@latest")
    if shutil.which("yt-dlp"):
        return ("yt-dlp",)
    raise FileNotFoundError(
        "M3-Bench's web split is 920 URLs and downloading them needs yt-dlp, which is not on "
        "PATH. Install uv so this can run `uvx yt-dlp@latest` -- the form that stays current, "
        "because an out-of-date yt-dlp fails with HTTP 403 rather than a version error -- or put "
        "yt-dlp on PATH yourself"
    )


def _sleep_seconds() -> float:
    """The pacing between downloads, which is the only setting an operator here can usefully turn.

    Refused rather than defaulted when it is set to something unreadable: someone who exports
    `30s` and has it silently replaced by 30 has been told the knob works, and the run that
    follows is the unpaced one they were trying to avoid.
    """
    configured = os.environ.get(SLEEP_SECONDS_VARIABLE, "").strip()
    if not configured:
        return DEFAULT_SLEEP_SECONDS
    try:
        seconds = float(configured)
    except ValueError as error:
        raise ValueError(
            f"{SLEEP_SECONDS_VARIABLE} is {configured!r}, which is not a number of seconds"
        ) from error
    if seconds < 0:
        raise ValueError(f"{SLEEP_SECONDS_VARIABLE} is {configured!r}, which is negative")
    return seconds


def _annotation(root: Path) -> Path:
    path = root / WEB_ANNOTATION
    if not path.exists():
        raise FileNotFoundError(
            f"{path} holds the URLs of M3-Bench's web videos and is absent, so the annotations "
            "have not been fetched yet. It is the m3-web task's own --dataset and therefore "
            "arrives before any producer runs; reaching this means the videos were asked for "
            "first. Fetch it with `mindbridge-bench datasets` or let a sweep of --tasks m3-web "
            "obtain it, then ask for the videos"
        )
    return path


def _report(
    *,
    wanted: int,
    attempted: int,
    failures: Sequence[str],
    announce: Callable[[str], None] | None,
) -> None:
    """Say what the corpus now holds, and refuse only when it holds nothing this run can use.

    Counted in usable videos rather than in successful downloads, because those are different
    numbers and only one of them is the question: a run whose whole selection was already on disk
    downloads nothing and is ready, and a run that acquired nine of ten is ready for nine.
    """
    usable = wanted - len(failures)
    if not failures:
        acquired = f", {attempted} newly acquired" if attempted else ""
        _say(announce, f"{wanted} M3-Bench web videos are ready{acquired}")
        return
    if not usable:
        raise RuntimeError(
            f"none of the {wanted} M3-Bench web videos this run selected is usable: "
            f"{_summarise(failures)}"
        )
    # Announced rather than raised. These videos really are gone -- a corpus of 920 URLs collected
    # from a public site loses entries to deletion, privacy and geoblocking -- and refusing to
    # prepare the ones that did arrive would hold the whole benchmark hostage to any one of them.
    # `prepare_m3` re-checks the file it wants straight after calling this, so an absent video
    # still stops the run that needed it, named.
    _say(
        announce,
        f"{usable} of {wanted} M3-Bench web videos are ready; {len(failures)} are not: "
        f"{_summarise(failures)}",
    )


def _summarise(failures: Sequence[str]) -> str:
    listed = ", ".join(failures[:_FAILURES_LISTED])
    remaining = len(failures) - _FAILURES_LISTED
    return listed if remaining <= 0 else f"{listed}, and {remaining} more"


def _say(announce: Callable[[str], None] | None, message: str) -> None:
    if announce is not None:
        announce(message)
