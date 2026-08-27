"""Obtaining EgoTempo's clips from Ego4D, the one release with a signature in front of it.

EgoTempo publishes 500 questions over 367 clips and not one video. Each clip is a span of an
Ego4D full-scale recording and is named for it -- `<video_uid>_<start_seconds>_<end_seconds>` --
and those 367 clips come out of **222 distinct Ego4D videos**, which is the number that decides
how this module is shaped.

A HuggingFace token is not the missing piece. Six Ego4D mirrors and both EgoTempo datasets on the
Hub were inspected and every one carries zero video files: Ego4D is not distributed through the
Hub at all. It is distributed from S3, to holders of a signed access agreement, and signing that
agreement grants an AWS access key. So exactly one step here is manual and it happens once --
request access at https://ego4d-data.org and put the key it grants where the AWS SDK looks -- and
everything after it is the official `ego4d` CLI, which is what this module drives.

Four properties are deliberate.

**`only` is what makes a smoke run affordable.** Ego4D full-scale videos are hundreds of
megabytes to several gigabytes each, and the 367 clips span 222 of them, so acquiring "EgoTempo"
unnarrowed is a multi-hundred-gigabyte download. `--limit 2` reads two questions, which is one or
two clips out of one or two source videos, and `only` is how that reaches the download rather
than being discovered after it.

**Prerequisites raise, and so does a download that quietly did nothing.** The `ego4d` CLI exits 0
after downloading nothing when S3 refuses its credentials -- it prints `Boto 403 Exception` per
object and `ABORT: All S3 Objects Invalid`, then returns success -- which is exactly what an
unsigned or not-yet-approved agreement looks like. So the source file's presence is checked after
the CLI returns, not the CLI's exit status alone. No credential is ever read, printed, or stored
here: only whether one resolves, and only from the environment and AWS configuration the CLI
itself already reads.

**The clips are cut at the product's own sampling, not as archival trims.** `cut_clips` at its
defaults samples one frame per second and scales to 200 704 pixels, which is what
`prepare_egotempo` re-cuts these files to anyway -- re-sampling a 1 fps source at 1 fps keeps
every frame -- so the run ingests the same frames it would have from a full-rate trim. Full rate
is not a free choice: `cut_clips` holds every sampled frame in memory, so a 30 s 1440p span at
30 fps peaks around 3.7 GB. What makes this safe rather than lossy is that the downloaded Ego4D
sources are kept: changing this decision later is a re-cut, not a re-download.

**The clips keep their audio, and that was measured rather than assumed.** Ego4D has an audio
track and derived clips going silently video-only is a defect this project has shipped before --
the audio-visual benchmarks then scored below their own blind baselines with nothing failing. A
1 fps video track is sparse, and a sparse video track is the case where PyAV cannot interleave
audio, so the answer was not obvious: probed on a 40 s 30 fps source with a 48 kHz AAC track, a
10-25 s cut at 1 fps carries 16 video frames **and 15.0 s of AAC**, and `video_segments`
re-cutting that clip carries the audio through again (10.0 s and 5.0 s across two segments). The
`include_audio` default is what does it, and `_copy_span_audio` runs after the video track is
flushed rather than interleaved with it, which is why sparseness does not cost the track here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from mindbridge.benchmarks.egotempo import EgoTempoQuestion, load_egotempo
from mindbridge.benchmarks.releases import fetch
from mindbridge.benchmarks.staging import media_duration_ms, within
from mindbridge.core import MediaKind

RELEASE = "egotempo"
"""The corpus directory this media set occupies, as `releases.RELEASES` keys it."""

ANNOTATION = "egotempo_openQA.json"
CLIP_DIRECTORY = "videos"
"""Where a cut clip lands: `<root>/egotempo/videos/<clip_id>.mp4`.

The destination `releases.UNOBTAINABLE` has always printed and `prepare_video.EGOTEMPO_MEDIA`
has always read, so an operator who cut these by hand before this module existed keeps them.
"""

EGO4D_DIRECTORY = "ego4d"
"""Where the Ego4D CLI's own tree lives, beside the clips cut out of it rather than inside them.

Kept after cutting, for the reason `releases.ensure_media` keeps its archives: deleting them is
what would make the next run download them again, and here that is gigabytes per clip.
"""

EGO4D_EXECUTABLE = "ego4d"
EGO4D_DATASET = "full_scale"
EGO4D_VERSION = "v2_1"
"""The Ego4D release the clip IDs are resolved against, pinned for the reason every revision in
`releases.py` is: the CLI's own default moves with its version, and a task name has to mean the
same bytes on two days. The CLI writes under the major alone -- `v2_1` lands in `v2/` -- so that
directory is derived from this string rather than written twice."""

ACCESS_URL = "https://ego4d-data.org"


def acquire(
    *,
    root: Path,
    only: Sequence[str] = (),
    announce: Callable[[str], None] | None = None,
) -> None:
    """Download the Ego4D videos behind the selected EgoTempo clips, and cut the clips out.

    `only` holds release-relative paths -- `videos/<clip_id>.mp4`, the same form
    `ensure_media` takes and the same form `prepare_egotempo` looks for -- and empty means every
    clip the annotation declares. The stem is the clip ID, which is why nothing here can be
    handed a question ID by accident: `question_id` is `clip_id` plus an optional `_<n>`, and for
    2 of the 500 questions the two strings are identical, so an ID-shaped channel would have been
    ambiguous where a file name is not.

    Idempotent, and cheap when there is nothing to do: a clip already on disk that opens with a
    declared duration is left alone, and if every selected clip is already there this returns
    before looking for the CLI or for credentials. That ordering is the point -- re-running a
    prepared corpus must not demand an Ego4D signature that the first run already used.

    Raises rather than degrading, in every direction. An absent CLI is an `ImportError` naming
    what to install, absent credentials a `PermissionError` naming what to sign, and a CLI that
    returns without producing a video a `FileNotFoundError` naming the video it did not produce.
    """
    clips = _selected(root, only, announce=announce)
    pending = tuple(clip for clip in clips if not _readable(_clip_path(root, clip.clip_id)))
    if not pending:
        return
    grouped = _by_source_video(pending)
    if announce is not None:
        announce(
            f"acquiring {len(pending)} of {len(clips)} egotempo clips from "
            f"{len(grouped)} Ego4D {EGO4D_DATASET} video(s)"
        )
    sources = _sources(root, tuple(grouped), announce=announce)
    for video_uid, group in grouped.items():
        # One read per source video rather than per clip: 15 of these clips come out of one
        # recording at the extreme, and re-reading a multi-gigabyte file 15 times is the cost
        # this grouping exists to avoid.
        # ponytail: the whole source is held in memory because `cut_clips` takes bytes, so peak
        # RSS is one Ego4D video (0.5-5 GB). Upgrade path is a path-taking encoder in
        # `mindbridge.media.clipping`, which would help every producer here rather than this one.
        content = sources[video_uid].read_bytes()
        for clip in group:
            target = _clip_path(root, clip.clip_id)
            _cut(content, clip, target)
            if announce is not None:
                announce(
                    f"  cut {clip.clip_id}: "
                    f"{clip.clip_start_seconds:.1f}-{clip.clip_end_seconds:.1f}s of {video_uid}"
                )


def _selected(
    root: Path,
    only: Sequence[str],
    *,
    announce: Callable[[str], None] | None,
) -> tuple[EgoTempoQuestion, ...]:
    """Resolve the requested clips against the release's own annotation, in its order.

    The annotation is the only thing that knows which clips exist, and `load_egotempo` already
    parses each clip ID into the source video and span this module downloads and cuts -- so
    there is no second parser here to disagree with the runner's.

    Normally it is already on disk: acquisition is reached from a producer, which runs after the
    annotations a task declares have been fetched. Fetching it here is for the other caller --
    this module used directly against a bare corpus -- and costs one 200 KB digest-verified
    download rather than an instruction to go and get it.
    """
    annotation = within(root, RELEASE, ANNOTATION)
    if not annotation.exists():
        fetch((annotation,), root=root, announce=announce)
    if not annotation.exists():
        raise FileNotFoundError(
            f"{annotation} is absent and could not be fetched, so which Ego4D videos EgoTempo "
            "needs is unknown; fetch the annotation before its media"
        )
    clips = {question.clip_id: question for question in load_egotempo(annotation)}
    if not only:
        return tuple(clips.values())
    requested = tuple(dict.fromkeys(_clip_id(entry) for entry in only))
    unknown = tuple(clip_id for clip_id in requested if clip_id not in clips)
    if unknown:
        raise ValueError(
            f"unknown EgoTempo clip IDs: {', '.join(sorted(unknown))}; {annotation} declares "
            f"{len(clips)} clips and a clip is named for the Ego4D span it is, "
            "`<video_uid>_<start_seconds>_<end_seconds>`"
        )
    return tuple(clips[clip_id] for clip_id in requested)


def _clip_id(entry: str) -> str:
    """Read one clip ID out of the release-relative path that names its file.

    Strict about the shape rather than accepting a bare ID as well: clip IDs carry the span in
    dotted decimals -- `..._120.0_180.0` -- so `Path(...).stem` on one that is not suffixed
    `.mp4` silently returns `..._120` and would then download the wrong video's clip. The same
    check is what keeps a release-supplied `videos/../../etc/passwd` from being treated as a clip.
    """
    path = PurePosixPath(entry)
    if path.parent.as_posix() != CLIP_DIRECTORY or path.suffix != ".mp4":
        raise ValueError(
            f"{entry!r} does not name an EgoTempo clip; `only` takes release-relative paths of "
            f"the shape {CLIP_DIRECTORY}/<clip_id>.mp4"
        )
    return path.stem


def _clip_path(root: Path, clip_id: str) -> Path:
    return within(root, RELEASE, CLIP_DIRECTORY, f"{clip_id}.mp4")


def _readable(path: Path) -> bool:
    """Whether a clip already on disk is one the producer that reads it can use.

    That, and not the span its name spells, is the test. These files are written whole through a
    temporary name, so a truncated one of this module's own making is not a state that exists;
    what does exist is an operator's hand-cut clip from before this module, cut with another
    encoder and its own rounding, and a source that genuinely ends before `clip_end_seconds`.
    Comparing durations would re-cut all of those on every run. Opening the file and reading the
    duration is exactly what `staging.video_segments` needs from it, so a clip that passes here
    is a clip the run can prepare.
    """
    if not path.exists():
        return False
    try:
        media_duration_ms(path)
    except Exception:
        # Anything from an unreadable container to an absent decoder means this file cannot be
        # skipped as already done; the cut below either replaces it or fails naming the decoder.
        return False
    return True


def _by_source_video(clips: Sequence[EgoTempoQuestion]) -> dict[str, list[EgoTempoQuestion]]:
    """Group the pending clips by the Ego4D recording they are spans of, in annotation order."""
    grouped: dict[str, list[EgoTempoQuestion]] = {}
    for clip in clips:
        grouped.setdefault(clip.source_video_id, []).append(clip)
    return grouped


def _sources(
    root: Path,
    video_uids: Sequence[str],
    *,
    announce: Callable[[str], None] | None,
) -> dict[str, Path]:
    """Return each Ego4D video's path on disk, downloading the ones that are not there yet."""
    directory = within(root, RELEASE, EGO4D_DIRECTORY)
    # `v2_1` is published under `v2/`, which is the CLI's own `version.split("_")[0]`.
    downloads = directory / EGO4D_VERSION.split("_")[0] / EGO4D_DATASET
    paths = {video_uid: downloads / f"{video_uid}.mp4" for video_uid in video_uids}
    absent = tuple(video_uid for video_uid, path in paths.items() if not path.exists())
    if not absent:
        return paths
    _run_ego4d(directory, absent, announce=announce)
    still_absent = tuple(video_uid for video_uid in absent if not paths[video_uid].exists())
    if still_absent:
        raise FileNotFoundError(
            f"the {EGO4D_EXECUTABLE} CLI returned without writing "
            f"{', '.join(f'{uid}.mp4' for uid in sorted(still_absent))} under {downloads}. That "
            "is what an Ego4D agreement which is unsigned, not yet approved, or approved for "
            "other universities looks like: S3 refuses each object with a 403, the CLI prints "
            "`ABORT: All S3 Objects Invalid`, and it exits successfully. A video absent from "
            f"Ego4D {EGO4D_VERSION} itself fails the same way, and the CLI's own output above "
            "says which of the two happened."
        )
    return paths


def _run_ego4d(
    directory: Path,
    video_uids: Sequence[str],
    *,
    announce: Callable[[str], None] | None,
) -> None:
    """Drive the official CLI once for every video still missing.

    One invocation rather than one per video: it downloads its manifest once and transfers in
    parallel, and the files it has already written are what make an interrupted acquisition
    resumable. Its output is inherited rather than captured, because a download this size is one
    an operator watches.
    """
    executable = shutil.which(EGO4D_EXECUTABLE)
    if executable is None:
        # ImportError rather than a message of its own: an absent tool is the case the caller
        # falls back to `releases.UNOBTAINABLE` for, and this text is what it wraps.
        raise ImportError(
            f"the official {EGO4D_EXECUTABLE} CLI is not on PATH, and it is the only way to "
            "obtain EgoTempo's clips: they are spans of Ego4D full-scale recordings, which are "
            "not published on the HuggingFace Hub at all. Install it with `uv pip install "
            f"ego4d` (PyPI, tested against 1.7.3), and give it the AWS credentials a signed "
            f"Ego4D agreement grants -- request access at {ACCESS_URL}."
        )
    profile = os.environ.get("AWS_PROFILE") or "default"
    _require_credentials(profile)
    command = [
        executable,
        "--output_directory",
        str(directory),
        "--datasets",
        EGO4D_DATASET,
        "--version",
        EGO4D_VERSION,
        "--aws_profile_name",
        # Named explicitly because the CLI's own default is the literal `default` profile, which
        # ignores AWS_PROFILE; passing it keeps the credentials checked above and the credentials
        # used below the same ones.
        profile,
        # The primary `ego4d.json` describes every video in the release and no clip needs it.
        "--no-metadata",
        "--yes",
        "--video_uids",
        *video_uids,
    ]
    if announce is not None:
        announce(
            f"downloading {len(video_uids)} Ego4D {EGO4D_DATASET} video(s) into {directory} "
            f"with the {EGO4D_EXECUTABLE} CLI"
        )
    # A fixed argv with no shell: the only release-supplied strings in it are Ego4D video
    # UIDs, which the annotation's own clip IDs are parsed out of.
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"the {EGO4D_EXECUTABLE} CLI exited {completed.returncode} while downloading "
            f"{len(video_uids)} Ego4D {EGO4D_DATASET} video(s); its own output above says why. "
            f"Already-downloaded videos are kept under {directory}, so re-running resumes "
            "rather than restarts."
        )


def _require_credentials(profile: str) -> None:
    """Refuse before downloading anything if no AWS credentials resolve for the CLI's profile.

    Presence only. Nothing here reads, logs, or copies a key: `boto3` resolves them the same way
    the CLI's own session does, and this asks whether that resolution produced anything. Checked
    before the subprocess so the failure names the agreement rather than arriving as an S3 403
    inside somebody else's progress bar.

    The message asks for a file rather than for environment variables because the CLI cannot use
    the latter. Naming a profile is what makes botocore drop its environment provider, the CLI
    names one -- `default` unless told otherwise -- and its own `validate_config` constructs a
    session for that profile before anything else, so `AWS_ACCESS_KEY_ID` alone fails there with
    `Could not find AWS profile 'default'`. Verified against `ego4d` 1.7.3's own parser: the same
    absence reaches this function as `ProfileNotFound`, which is why both it and a profile that
    exists without keys in it end at the same instruction.
    """
    import boto3
    from botocore.exceptions import BotoCoreError

    unavailable = (
        f"no AWS credentials resolve for profile {profile!r}, so the {EGO4D_EXECUTABLE} CLI "
        "cannot download anything. Ego4D is released under a signed access agreement: request "
        f"access at {ACCESS_URL}, and put the access key ID and secret access key it grants in "
        f"~/.aws/credentials under [{profile}] -- export AWS_PROFILE to use another profile. "
        "Nothing else about the run changes; re-run the same command."
    )
    try:
        credentials = boto3.Session(profile_name=profile).get_credentials()
    except BotoCoreError as error:
        # A named profile that is not in the AWS configuration at all arrives here.
        raise PermissionError(unavailable) from error
    if credentials is None:
        raise PermissionError(unavailable)


def _cut(content: bytes, clip: EgoTempoQuestion, target: Path) -> None:
    """Cut one clip out of its source video with the encoder the product stores evidence with.

    The span is closed at both ends, which is `cut_clips`' own shape and the right one here: this
    file *is* the unit, so the frame at `clip_end_seconds` belongs to it. `prepare_egotempo` is
    what re-cuts it into disjoint segments, and that is where a half-open span matters.

    Written through a temporary name in the same directory. An interrupted cut has to leave
    nothing rather than a short file, because a short file is indistinguishable from a clip whose
    source ended early and would be skipped as done on the next run.
    """
    from mindbridge.media.clipping import ClipRequest, cut_clips

    cut = cut_clips(
        content,
        ClipRequest(
            kind=MediaKind.VIDEO,
            start_ms=round(clip.clip_start_seconds * 1_000),
            end_ms=round(clip.clip_end_seconds * 1_000),
        ),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f"{target.name}.part"
    partial.write_bytes(cut[0].content)
    partial.replace(target)
