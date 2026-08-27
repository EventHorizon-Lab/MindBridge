"""Cutting MM-Lifelong's sources onto the one clock its four splits are scored against.

`MMLifelongPreparedTimeline` wants `start_seconds` on the official split-wide `total_intervals`
clock, not on any one video's own. That is the whole difficulty here, and it is why this producer
reads the annotation twice: once through the adapter for the questions, and once for the field
the adapter drops.

**Where the clock comes from.** Every question carries its clue spans twice -- `clue_intervals`
(or `clue_interval`, in the month splits) in the source video's own seconds, and
`total_intervals` in the split's, in the same flattened order. So a video's offset is the
difference between the two, and a release that means one thing by it gives the same difference
from every question that cites the video. That agreement is the check: this refuses a split
whose spans imply two different offsets for one video rather than picking one. `mm_lifelong`
itself keeps only the count of these spans, because the runner scores against `total_intervals`
and never needs the source clock; preparing is the one step that does.

A video no question cites has no derivable offset and is therefore not prepared -- the month
splits each cite about two thirds of the release's 23 videos. What is lost is distractors, and
the run manifest's `segment_count` is what records that the timeline was partial.

**Where the media comes from.** All of it is in the release, and `ensure_media` fetches it.
`video_list.txt` names one bilibili source, twenty-two YouTube ones and the EgoLife repository,
which reads like a download list but is a provenance note: the release ships all 189 GB of it
under `videos/` itself. The Week scale in particular is a vendored copy of EgoLife `A1_JAKE`,
already cut into its 30-second slices under `videos/week/day<n>/`, so fetching it a second time
from `lmms-lab/EgoLife` would be the same media on disk twice under two names.

**Audio.** A lifelong corpus is mostly speech, so the clips are cut with `video_segments`,
which is `mindbridge.media.clipping.cut_clips` -- the encoder the product stores evidence with,
and which keeps the source's audio track. Cutting these video-only is a defect this repository
has already shipped once.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from typing import NamedTuple

from mindbridge.benchmarks.cli_common import report, select_by_id
from mindbridge.benchmarks.runtime import benchmark_tenant_id
from mindbridge.benchmarks.staging import (
    SEGMENT_SECONDS,
    PrepareRequest,
    key_component,
    media_duration_ms,
    staging,
    video_segments,
    within,
    write_manifest,
)
from mindbridge.core import MediaKind

MM_LIFELONG_TIMELINE_ORIGIN = datetime(2000, 1, 1, tzinfo=timezone.utc)
"""Where second zero of a split's clock is placed, as `docs/benchmarking.md` documents it.

Arbitrary and fixed, for the reason `STAGED_AT` is: the run manifest pins this file's digest,
so two preparations of one split have to agree.
"""

MM_LIFELONG_RELEASE = "mm-lifelong"
"""The directory MM-Lifelong occupies under `--benchmarks-root`, and its key in `RELEASES`."""

MM_LIFELONG_VIDEO_DIRECTORY = "videos"
"""Where the release keeps its media, under the release directory.

Day and Month are one file per video ID -- `videos/day/0.mp4`, `videos/month/11.mp4` -- and Week
is a directory of pre-cut slices per video ID, `videos/week/day5/DAY5_A1_JAKE_*.mp4`. Read off
the file list of the revision pinned in `releases.py`; the corpus itself is not available to
check against, so a wrong guess is a one-line change here.
"""

MM_LIFELONG_DAY_VIDEO_ID = "0"
"""What the Day split's single source is called, since its spans name no video.

`day/test.json` gives `clue_intervals` as bare spans rather than the `{video_id, intervals}`
groups the other three splits use, because the scale is one video: `videos/day/0.mp4`.
"""

_CLUE_FIELD = {
    "day_test": "clue_intervals",
    "week_test": "clue_intervals",
    "month_train": "clue_interval",
    "month_val": "clue_interval",
}
"""Which field holds a split's source-clock spans, matching `mm_lifelong._clue_count`."""

_SEGMENT_MS = SEGMENT_SECONDS * 1_000


def prepare_mm_lifelong(request: PrepareRequest) -> None:
    """Cut one official split's sources into segments carrying their global start times."""
    from mindbridge.benchmarks.mm_lifelong import load_mm_lifelong
    from mindbridge.benchmarks.mm_lifelong_cli import _parse_arguments
    from mindbridge.benchmarks.mm_lifelong_runner import (
        MMLifelongPreparedSegment,
        MMLifelongPreparedTimeline,
    )
    from mindbridge.benchmarks.releases import ensure_media

    arguments = _parse_arguments(list(request.argv), None)
    questions = select_by_id(
        load_mm_lifelong(arguments.dataset_path, arguments.split),
        arguments.question_indices,
        key=lambda question: question.index,
        label="MM-Lifelong question indices",
        limit=arguments.limit,
    )
    if not questions:
        raise ValueError("MM-Lifelong selection must not be empty")
    wanted = tuple(interval for question in questions for interval in question.reference_intervals)
    offsets = global_offsets(arguments.dataset_path, arguments.split)
    # `--limit` bounds what is prepared as well as what is answered, the same narrowing
    # `mm_lifelong_cli._timeline_for_run` applies to a manifest it is handed. Without it a
    # one-question smoke run downloaded and cut a whole Month scale first.
    scoped = None if arguments.limit is None else wanted
    needed = _needed_videos(offsets, scoped)
    absent = _absent_videos(request.benchmarks_root, arguments.split, needed)
    if absent:
        # On absence rather than eagerly, because `ensure_media` re-derives an acquired media
        # set before it looks at the disk. Narrowed to what this run reads, which is what
        # keeps a one-question smoke run off the release's 189 GB of video.
        ensure_media(
            MM_LIFELONG_RELEASE,
            root=request.benchmarks_root,
            only=_media_patterns(arguments.split, absent),
            download=request.download,
        )
    plan = _plan(request.benchmarks_root, arguments.split, offsets, needed, wanted=scoped)
    _require_coverage(plan, wanted)
    tenant_id = benchmark_tenant_id(arguments.tenant_prefix, arguments.split, arguments.run_id)
    target = staging()
    segments: list[MMLifelongPreparedSegment] = []
    for video_id, cuts in plan.items():
        component = key_component(video_id, label="MM-Lifelong video ID")
        for cut, duration_ms, content in _encode(cuts):
            name = f"{video_id}_{cut.ordinal:05d}"
            segments.append(
                MMLifelongPreparedSegment(
                    segment_id=f"video_{name}",
                    start_seconds=cut.start_seconds,
                    duration_ms=duration_ms,
                    media_objects=(
                        target.stage(
                            tenant_id=tenant_id,
                            key=f"{component}/{cut.ordinal:05d}.mp4",
                            content=content,
                            kind=MediaKind.VIDEO,
                            media_object_id=f"mm_lifelong_{name}",
                            duration_ms=duration_ms,
                        ),
                    ),
                )
            )
        report(f"  {video_id}: {len(cuts)} segments -> {tenant_id}", quiet=request.quiet)
    write_manifest(
        arguments.prepared_media_path,
        MMLifelongPreparedTimeline(
            split=arguments.split,
            timeline_origin=MM_LIFELONG_TIMELINE_ORIGIN,
            # Chronological, which the timeline's own validator requires: the plan is walked in
            # offset order and each video's cuts in local order, so this is already sorted.
            segments=tuple(sorted(segments, key=lambda segment: segment.start_seconds)),
        ),
    )


def global_offsets(dataset_path: Path, split: str) -> dict[str, float]:
    """Where each source video's own second zero sits on the split's clock.

    Derived rather than tabulated, and self-checking: one video's offset is read from every
    question that cites it, and a split whose spans do not agree on it is refused. Returned in
    ascending offset order, which is the order the sources sit on the timeline.
    """
    from mindbridge.benchmarks.mm_lifelong import _INTERVALS

    if split not in _CLUE_FIELD:
        raise ValueError(f"unknown MM-Lifelong split: {split}")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("MM-Lifelong annotations must not be empty")
    candidates: dict[str, set[float]] = {}
    for question in raw:
        if not isinstance(question, dict):
            raise ValueError("MM-Lifelong annotations must be objects")
        totals = _INTERVALS.validate_python(question["total_intervals"])
        local = _local_spans(question, split)
        if len(local) != len(totals):
            raise ValueError(
                "MM-Lifelong clue spans and total_intervals must correspond one for one; "
                f"question {question.get('index')!r} has {len(local)} and {len(totals)}"
            )
        for (video_id, start), (global_start, _) in zip(local, totals, strict=True):
            # Rounded because the release writes hundredths and float subtraction does not:
            # 105839.63 - 13 lands a few ulp off 105826.63 depending on the operands.
            candidates.setdefault(video_id, set()).add(round(global_start - start, 3))
    disagreed = sorted(video for video, seen in candidates.items() if len(seen) != 1)
    if disagreed:
        raise ValueError(
            "MM-Lifelong spans imply more than one global offset for "
            f"{', '.join(disagreed)}; the split's clue and total intervals disagree"
        )
    offsets = {
        # Guarded here rather than where the ID is joined into a path, because it is joined in
        # three places -- the `only` patterns asking for a download, the source directory, and
        # the object key -- and this is the one they all pass through first.
        key_component(video, label="MM-Lifelong video ID"): next(iter(seen))
        for video, seen in candidates.items()
    }
    return dict(sorted(offsets.items(), key=lambda item: item[1]))


def _local_spans(question: dict[str, object], split: str) -> list[tuple[str, float]]:
    """One question's clue spans in their own video's seconds, flattened as `total_intervals` is.

    The Day split names no video because its scale is one; the other three group spans by video
    ID, and the flattening is per group in order, which is what makes the pairing positional.
    """
    from mindbridge.benchmarks.mm_lifelong import _INTERVALS, _SOURCES

    raw = question.get(_CLUE_FIELD[split])
    if raw is None:
        raise ValueError(f"MM-Lifelong {split} questions require {_CLUE_FIELD[split]}")
    if split == "day_test":
        return [(MM_LIFELONG_DAY_VIDEO_ID, start) for start, _ in _INTERVALS.validate_python(raw)]
    return [
        (source.video_id, start)
        for source in _SOURCES.validate_python(raw)
        for start, _ in source.intervals
    ]


class _Cut(NamedTuple):
    """One planned segment, before anything has been encoded."""

    source: Path
    cut_index: int
    """Which segment of `source` this is, which is what `video_segments` counts."""
    ordinal: int
    """Which segment of the whole video ID this is, which is what names it.

    The two differ only for the Week scale, whose video ID is a directory of slices. Counted
    over the unfiltered plan, so a `--limit` run names a segment the same as a full one does.
    """
    start_seconds: float
    duration_ms: int


def _needed_videos(
    offsets: dict[str, float],
    wanted: Sequence[tuple[float, float]] | None,
) -> tuple[str, ...]:
    """Which videos a run reads, decided before any of them is fetched or opened.

    From the offsets alone, so a scoped run downloads one Month video rather than 85 GB of
    them. A video's span is bounded by where the next one starts, which over-estimates wherever
    the split cites nothing for the gap -- the safe direction, since the segments themselves are
    filtered again once their real durations are known.
    """
    bounds = _video_bounds(offsets)
    return tuple(video for video in offsets if wanted is None or _overlaps(bounds[video], wanted))


def _media_patterns(split: str, videos: Sequence[str]) -> tuple[str, ...]:
    """Name each needed video to `ensure_media`, in the release's own layout."""
    scale = _scale(split)
    if scale == "week":
        return tuple(f"{MM_LIFELONG_VIDEO_DIRECTORY}/week/{video}/*" for video in videos)
    return tuple(f"{MM_LIFELONG_VIDEO_DIRECTORY}/{scale}/{video}.mp4" for video in videos)


def _absent_videos(root: Path, split: str, videos: Sequence[str]) -> tuple[str, ...]:
    """Which needed videos are not on disk, which is the only reason to fetch anything.

    A Week video is a directory of slices, so what counts as present is that it holds any; the
    plan reads whichever are there, and a partial day is a partial timeline rather than an error.
    """
    scale = _scale(split)
    directory = within(root, MM_LIFELONG_RELEASE, MM_LIFELONG_VIDEO_DIRECTORY, scale)
    if scale == "week":
        return tuple(video for video in videos if not any((directory / video).glob("*.mp4")))
    return tuple(video for video in videos if not (directory / f"{video}.mp4").exists())


def _plan(
    root: Path,
    split: str,
    offsets: dict[str, float],
    videos: Sequence[str],
    *,
    wanted: Sequence[tuple[float, float]] | None,
) -> dict[str, tuple[_Cut, ...]]:
    """Lay every needed source out on the global clock without cutting anything.

    Durations are read from container headers, so this costs a file open per source and settles
    the two questions worth settling before the encoder runs for hours: whether the media is
    there at all, and whether the timeline it would produce reaches the spans the run will ask
    about.
    """
    planned: dict[str, tuple[_Cut, ...]] = {}
    for video_id in videos:
        offset = offsets[video_id]
        cuts = tuple(
            cut
            for cut in _video_cuts(_sources(root, split, video_id), offset)
            if wanted is None
            or _overlaps((cut.start_seconds, cut.start_seconds + cut.duration_ms / 1_000), wanted)
        )
        if cuts:
            planned[video_id] = cuts
    if not planned:
        raise ValueError("no MM-Lifelong source covers the selected questions' reference intervals")
    return planned


def _video_cuts(sources: tuple[Path, ...], offset: float) -> Iterator[_Cut]:
    """Every segment one video contributes, on the global clock, in order.

    A source shorter than one segment before the first full one is not on the clock. That is the
    Week scale's rule and the release's own: EgoLife's first slice of a day is a partial one --
    17.92 s on day 1 -- and the official offsets are consistent with the clock starting at the
    first whole slice after it, not with that remnant occupying the first eighteen seconds.
    """
    local_ms = 0
    ordinal = 0
    started = False
    for source in sources:
        duration_ms = media_duration_ms(source)
        if not started and duration_ms < _SEGMENT_MS:
            continue
        started = True
        for index in range(math.ceil(duration_ms / _SEGMENT_MS)):
            start_ms = index * _SEGMENT_MS
            yield _Cut(
                source=source,
                cut_index=index,
                ordinal=ordinal,
                start_seconds=offset + (local_ms + start_ms) / 1_000,
                duration_ms=min(start_ms + _SEGMENT_MS, duration_ms) - start_ms,
            )
            ordinal += 1
        local_ms += duration_ms


def _encode(cuts: Sequence[_Cut]) -> Iterator[tuple[_Cut, int, bytes]]:
    """Cut the planned segments, one source decode for all of the segments taken from it.

    ponytail: `video_segments` counts from the start of its source, so the segments before the
    first one wanted are cut and dropped. That is free for the Week scale, whose sources are one
    segment each, and is the cost of a `--limit` run that wants a span late in a Month video.
    Lifting it means an offset on the shared cutter, which more than this producer would use.
    """
    for source, group in groupby(cuts, key=attrgetter("source")):
        wanted = {cut.cut_index: cut for cut in group}
        for index, duration_ms, content in video_segments(source, limit=max(wanted) + 1):
            cut = wanted.get(index)
            if cut is not None:
                yield cut, duration_ms, content


def _scale(split: str) -> str:
    """Which of the release's three scales a split belongs to, and names its media directory."""
    if split not in _CLUE_FIELD:
        raise ValueError(f"unknown MM-Lifelong split: {split}")
    return split.split("_", 1)[0]


def _sources(root: Path, split: str, video_id: str) -> tuple[Path, ...]:
    """The files one video ID is made of, refusing an ID that would leave the corpus."""
    scale = _scale(split)
    component = key_component(video_id, label="MM-Lifelong video ID")
    directory = within(root, MM_LIFELONG_RELEASE, MM_LIFELONG_VIDEO_DIRECTORY, scale)
    if scale != "week":
        source = directory / f"{component}.mp4"
        if not source.exists():
            raise FileNotFoundError(
                f"MM-Lifelong source {source} is absent; it is part of the "
                "MM-Lifelong/MM-Lifelong release and several gigabytes per video"
            )
        return (source,)
    # Sorted by name, which for EgoLife's `DAYn_A1_JAKE_HHMMSSFF` is chronological.
    slices = tuple(sorted((directory / component).glob("*.mp4")))
    if not slices:
        raise FileNotFoundError(
            f"MM-Lifelong Week source {directory / component} holds no slices; the scale is the "
            "EgoLife A1_JAKE media the MM-Lifelong release vendors under videos/week/"
        )
    return slices


def _video_bounds(offsets: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Each video's span on the global clock, bounded by where the next one starts.

    An over-estimate wherever the split cites no video for the gap, which is the safe direction:
    it keeps a source a question might need rather than probing every file to find out.
    """
    ordered = list(offsets.items())
    return {
        video: (offset, ordered[index + 1][1] if index + 1 < len(ordered) else math.inf)
        for index, (video, offset) in enumerate(ordered)
    }


def _overlaps(span: tuple[float, float], wanted: Sequence[tuple[float, float]]) -> bool:
    """Whether a span touches any interval a selected question references."""
    start, end = span
    return any(other_start < end and other_end > start for other_start, other_end in wanted)


def _require_coverage(
    plan: dict[str, tuple[_Cut, ...]],
    wanted: Sequence[tuple[float, float]],
) -> None:
    """Refuse before cutting a timeline `run_mm_lifelong` would reject for being too short.

    The runner reads the end of the last segment as the split's total duration and refuses a
    run whose questions reference anything past it. Finding that out after the encode is the
    expensive way to learn it.
    """
    reachable = max(
        cut.start_seconds + cut.duration_ms / 1_000 for cuts in plan.values() for cut in cuts
    )
    required = max(end for _, end in wanted)
    if required > reachable:
        raise ValueError(
            f"MM-Lifelong sources reach {reachable:.2f}s but the selected questions reference "
            f"{required:.2f}s; the release's media is shorter than its own annotations"
        )
