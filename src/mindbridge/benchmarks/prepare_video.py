"""Producing prepared media for the three benchmarks that read `runtime.PreparedVideo`.

Video-MME, Video-MME-v2, and EgoTempo share one shape -- a fixed origin plus ordered
non-overlapping segments -- so the three differ only in which release supplies the file and which
annotation field names it.

Two rules from `staging.py` are load-bearing here and easy to lose:

**The selection is the runner's own.** Each producer parses with the runner's `_parse_arguments`
and narrows with the runner's own selection helper, so Video-MME's duration bands and
Video-MME-v2's group types are applied by the code that defines them. A unit prepared but not run
is a wasted upload; a unit run but not prepared fails the run after it has already paid for the
rest.

**Everything is checked before anything is staged.** Every source file is resolved and its
absence raised before the first upload, for the reason `prepare_m3` re-runs the runner's own
`_validate_subset`: these uploads are gigabytes, and a manifest that dies on the fourth video has
already paid for three.

Where each release's files sit is one named constant below, and each says what fixed it. What
to do when one is absent is not restated here at all: `ensure_media` fetches a media set that can
be fetched and raises the operator's instructions for one that cannot, so `_source` asks it
rather than keeping a second copy of either answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from mindbridge.benchmarks.cli_common import (
    MediaArguments,
    TranscriptSource,
    report,
    select_by_id,
)
from mindbridge.benchmarks.releases import ensure_media
from mindbridge.benchmarks.runtime import (
    PreparedVideo,
    PreparedVideoSegment,
    benchmark_tenant_id,
)
from mindbridge.benchmarks.staging import (
    PrepareRequest,
    Staging,
    key_component,
    staging,
    video_segments,
    within,
    write_manifest,
)
from mindbridge.core import MediaKind

TIMELINE_ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)
"""The origin every manifest here declares.

Each of these clocks is relative to the start of its own unit -- a Video-MME video, an EgoTempo
trimmed clip -- so which instant stands for zero is arbitrary and only has to be the same instant
twice. Fixed rather than the wall clock for the same reason `STAGED_AT` is: a run manifest pins
the prepared manifest's digest, and two preparations of one video have to agree.
"""

VIDEO_MME_MEDIA = ("video-mme", "data")
"""Where a Video-MME source video sits, named by `source_video_id` and not by `video_id`.

Those are two different fields, which is the mistake this constant exists to hold still:
`video_id` is the release's own `001`-`900`, while `source_video_id` is the `videoID` column,
the YouTube ID the files carry. Read out of the central directory of `videos_chunked_20.zip` at
the pinned revision -- `data/zAXbdzvCeV8.mp4` and siblings.
"""

VIDEO_MME_V2_MEDIA = ("video-mme-v2", "videos")
"""Where a Video-MME-v2 source video sits, named by `video_id`.

Read out of the central directories of its 40 volumes at the pinned revision: `videos/001.zip`
holds a flat `001.mp4`-`020.mp4`, `videos/040.zip` holds `781.mp4`-`800.mp4`, and extraction
lands each beside its volume. The annotation carries no second identifier, so `video_id` is also
the only field a file here could be named by.
"""

EGOTEMPO_MEDIA = ("egotempo", "videos")
"""Where an EgoTempo clip sits, named by `clip_id` and already trimmed.

Operator-supplied rather than release content: the pinned Git release is annotations only, and
the videos behind them are Ego4D, released under a signed access agreement that no unattended
download can accept. So this has to match, exactly, the destination `releases.UNOBTAINABLE`
prints when it refuses to fetch them -- that sentence is the only instruction an operator gets,
and it asks for the span `clip_start_seconds..clip_end_seconds` cut out beforehand. Preparation
splits what it finds and does not trim: a clip is one file here, whole.
"""


def prepare_video_mme(request: PrepareRequest) -> None:
    """Cut each selected official video into the segments a Video-MME run ingests."""
    from mindbridge.benchmarks.video_mme import load_video_mme
    from mindbridge.benchmarks.video_mme_cli import _parse_arguments, _select_videos

    arguments = _parse_arguments(list(request.argv), None)
    videos = _select_videos(
        load_video_mme(arguments.dataset_path),
        arguments.video_ids,
        arguments.durations,
        arguments.limit,
    )
    _require_media_only(arguments.transcript_source, benchmark="Video-MME")
    sources = tuple(
        (
            video.video_id,
            _source(
                request.benchmarks_root,
                *VIDEO_MME_MEDIA,
                f"{video.source_video_id}.mp4",
                release="video-mme",
                download=request.download,
                quiet=request.quiet,
            ),
        )
        for video in videos
    )
    target = staging()
    write_manifest(
        arguments.prepared_media_path,
        [
            _prepared_video(
                target,
                video_id=video_id,
                segments=video_segments(source),
                tenant_id=_tenant_id(arguments, video_id, label="Video-MME video ID"),
                media_prefix="video-mme",
                quiet=request.quiet,
            )
            for video_id, source in sources
        ],
    )


def prepare_video_mme_v2(request: PrepareRequest) -> None:
    """Cut each selected group's video into the segments a Video-MME-v2 run ingests.

    Selected by group even though the manifest is per video: a group is the only subset the
    rating is defined over, and `load_video_mme_v2` already refuses a release that splits one
    video across two groups, so one selected group is one prepared video.
    """
    from mindbridge.benchmarks.video_mme_v2 import load_video_mme_v2
    from mindbridge.benchmarks.video_mme_v2_cli import _parse_arguments, _select_groups

    arguments = _parse_arguments(list(request.argv), None)
    groups = _select_groups(
        load_video_mme_v2(arguments.dataset_path),
        arguments.video_ids,
        arguments.group_types,
        arguments.limit,
    )
    _require_media_only(arguments.transcript_source, benchmark="Video-MME-v2")
    sources = tuple(
        (
            group.video_id,
            _source(
                request.benchmarks_root,
                *VIDEO_MME_V2_MEDIA,
                f"{group.video_id}.mp4",
                release="video-mme-v2",
                download=request.download,
                quiet=request.quiet,
            ),
        )
        for group in groups
    )
    target = staging()
    write_manifest(
        arguments.prepared_media_path,
        [
            _prepared_video(
                target,
                video_id=video_id,
                segments=video_segments(source),
                tenant_id=_tenant_id(arguments, video_id, label="Video-MME-v2 video ID"),
                media_prefix="video-mme-v2",
                quiet=request.quiet,
            )
            for video_id, source in sources
        ],
    )


def prepare_egotempo(request: PrepareRequest) -> None:
    """Cut each selected question's official clip into the segments an EgoTempo run ingests.

    One prepared video per clip, not per question: several official questions share a clip, and
    the runner ingests the clip once and answers all of them. Deduplicated in annotation order,
    which is the order `_select_prepared` looks them up in.
    """
    from mindbridge.benchmarks.egotempo import load_egotempo
    from mindbridge.benchmarks.egotempo_cli import _parse_arguments

    arguments = _parse_arguments(list(request.argv), None)
    questions = select_by_id(
        load_egotempo(arguments.dataset_path),
        arguments.question_ids,
        key=lambda question: question.question_id,
        label="EgoTempo question IDs",
        limit=arguments.limit,
    )
    sources = tuple(
        (
            clip_id,
            _source(
                request.benchmarks_root,
                *EGOTEMPO_MEDIA,
                f"{clip_id}.mp4",
                release="egotempo",
                download=request.download,
                narrow=True,
                quiet=request.quiet,
            ),
        )
        for clip_id in dict.fromkeys(question.clip_id for question in questions)
    )
    target = staging()
    write_manifest(
        arguments.prepared_media_path,
        [
            _prepared_video(
                target,
                video_id=clip_id,
                segments=video_segments(source),
                tenant_id=_tenant_id(arguments, clip_id, label="EgoTempo clip ID"),
                media_prefix="egotempo",
                quiet=request.quiet,
            )
            for clip_id, source in sources
        ],
    )


def _prepared_video(
    target: Staging,
    *,
    video_id: str,
    segments: Iterable[tuple[int, int, bytes]],
    tenant_id: str,
    media_prefix: str,
    quiet: bool,
) -> PreparedVideo:
    """Stage one unit's cut segments and describe them on this module's fixed clock.

    Starts accumulate the durations the split declares rather than multiplying the segment index
    by the split length. The two agree while every segment but the last is exactly that long, and
    only the accumulated form stays non-overlapping if that ever stops being true -- which is the
    invariant `PreparedVideo` refuses a manifest for breaking.
    """
    prepared: list[PreparedVideoSegment] = []
    start_ms = 0
    for index, duration_ms, content in segments:
        prepared.append(
            PreparedVideoSegment(
                segment_id=f"{video_id}-{index:04d}",
                start_seconds=start_ms / 1_000,
                duration_ms=duration_ms,
                media_objects=(
                    target.stage(
                        tenant_id=tenant_id,
                        key=f"{index:04d}.mp4",
                        content=content,
                        kind=MediaKind.VIDEO,
                        media_object_id=f"{media_prefix}-{video_id}-{index:04d}",
                        duration_ms=duration_ms,
                    ),
                ),
            )
        )
        start_ms += duration_ms
    report(f"  {video_id}: {len(prepared)} segments -> {tenant_id}", quiet=quiet)
    return PreparedVideo(
        video_id=video_id,
        timeline_origin=TIMELINE_ORIGIN,
        segments=tuple(prepared),
    )


def _tenant_id(arguments: MediaArguments, unit_id: str, *, label: str) -> str:
    """Build the tenant one unit's media has to sit under, refusing an ID that leaves its prefix.

    The unit ID reaches the object key through the tenant, so a release-supplied `a/../b` would
    write outside the prefix the manifest claims.
    """
    return benchmark_tenant_id(
        arguments.tenant_prefix,
        key_component(unit_id, label=label),
        arguments.run_id,
    )


def _require_media_only(transcript_source: TranscriptSource, *, benchmark: str) -> None:
    """Refuse a transcript setting this producer cannot satisfy, before it cuts anything.

    Both Video-MME releases publish separate with- and without-subtitle tables, and the runner
    refuses a manifest that disagrees with the declared source. What it cannot see is a producer
    that quietly staged media only, so this is checked here rather than after the upload.
    """
    if transcript_source != "none":
        raise ValueError(
            f"preparing {benchmark} media produces no transcript, so it cannot serve "
            f"--transcript-source {transcript_source}; prepare with none, or supply a manifest "
            "carrying the released subtitle track"
        )


def _source(
    root: Path,
    *parts: str,
    release: str,
    download: bool = True,
    narrow: bool = False,
    quiet: bool = True,
) -> Path:
    """Resolve one release-named source file, asking `ensure_media` about an absent one.

    Absence is two different problems: a media set that can be fetched may simply not have been,
    and one that is acquired from somewhere other than the release may need a prerequisite this
    machine does not have. Both answers are written in `ensure_media` -- it obtains the first and
    raises the operator's own instructions for the second -- so this asks it rather than keeping a
    second copy that would drift from the destination those instructions name.

    `narrow` asks for this file alone rather than the whole media set, which is what an acquired
    set needs: EgoTempo's clips come out of Ego4D one span at a time, so an unnarrowed call means
    every clip the split names. It is off by default because it cannot be used here at all for the
    two archived releases -- `ensure_media` refuses `only` for those, since no index says which
    multi-gigabyte volume holds which video -- and Video-MME comes through this same helper.
    """
    path = within(root, *parts)
    if not path.exists():
        ensure_media(
            release,
            root=root,
            only=("/".join(parts[1:]),) if narrow else (),
            announce=None if quiet else partial(report, quiet=False),
            download=download,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is absent from the {release} media, which is now on disk; the release's "
                "own layout may have moved"
            )
    return path
