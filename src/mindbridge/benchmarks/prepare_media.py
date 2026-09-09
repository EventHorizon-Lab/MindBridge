"""Acquire selected benchmark media and prepare causal local video segments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, mkdtemp
from threading import Lock
from typing import TYPE_CHECKING, TypeAlias, TypeVar
from urllib.parse import urlparse

from mindbridge.benchmarks.download import acquire_media
from mindbridge.benchmarks.task_catalog import TaskSpec

if TYPE_CHECKING:
    from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion

_SEGMENT_SECONDS = 30
_VIDEO_SCALE_FILTER = (
    "scale=w='min(640\\,iw)':h='min(360\\,ih)':"
    "force_original_aspect_ratio=decrease:force_divisible_by=2"
)
_VIDEO_FILTER = f"fps=1,{_VIDEO_SCALE_FILTER}"
# Retain the source's real cadence only when the standard 1 fps representation leaves fewer than
# two frames. Qwen's video processor cannot consume a one-frame video, and duplicating a frame
# would invent a temporal observation. A generic prepared-media cache still accepts one real
# frame: consumers with a stricter processor must report that limitation at their own boundary.
_VIDEO_FILTER_NATIVE = _VIDEO_SCALE_FILTER
_MIN_PREPROCESSABLE_VIDEO_FRAMES = 2
_COMPLETE_MARKER = ".complete"
# The concat demuxer has no following packet from which the fps filter can infer
# the final frame's duration. Passing it through keeps N official frames as N seconds.
_OPENEQA_VIDEO_FILTER = f"fps=1:eof_action=pass,{_VIDEO_SCALE_FILTER}"
# OpenEQA episode histories are directories of RGB frames with no published
# video encoding for evaluation -- upstream's `data/frames2videos.py` writes at
# 30 fps into `viewer/static/videos` for its web viewer, not for scoring. One
# frame per second is the rate that loses nothing: `_VIDEO_FILTER` resamples
# every prepared segment to `fps=1`, so encoding the sequence at 1 fps makes
# that resample an identity step and keeps every extracted frame, where 30 fps
# would silently discard 29 of every 30. Raise it only to deliberately thin a
# scene's history.
_OPENEQA_FRAME_RATE = 1
# ponytail: ffmpeg is already multithreaded; bound source-level fan-out at four unless profiling
# on supported hardware demonstrates that a public tuning knob pays for itself.
_PREPARATION_WORKERS = min(4, os.cpu_count() or 1)
_PREPARED_MEDIA_CACHE_VERSION = "ffmpeg-v2"
_VIDEO_SEGMENT_CACHE_KEY_VERSION = "video-segment-v3"
_LEGACY_PREPARED_MEDIA_CACHE_VERSION = "ffmpeg-v1"
_LEGACY_VIDEO_SEGMENT_CACHE_KEY_VERSION = "video-segment-v2"
_T = TypeVar("_T")
_R = TypeVar("_R")
Limit: TypeAlias = int | float | None
ProgressCallback: TypeAlias = Callable[[int, int], None]


def _prepare_many(
    function: Callable[[_T], _R],
    values: Sequence[_T],
    on_progress: ProgressCallback | None = None,
) -> tuple[_R, ...]:
    total = len(values)
    if on_progress is not None:
        on_progress(0, total)
    completed = 0
    progress_lock = Lock()

    def prepare(value: _T) -> _R:
        nonlocal completed
        result = function(value)
        if on_progress is not None:
            with progress_lock:
                completed += 1
                on_progress(completed, total)
        return result

    if len(values) < 2 or _PREPARATION_WORKERS == 1:
        return tuple(map(prepare, values))
    with ThreadPoolExecutor(max_workers=min(_PREPARATION_WORKERS, len(values))) as pool:
        return tuple(pool.map(prepare, values))


def _serialized_announce(callback: Callable[[str], None] | None) -> Callable[[str], None] | None:
    if callback is None:
        return None
    lock = Lock()

    def call(message: str) -> None:
        with lock:
            callback(message)

    return call


def prepare_task_media(
    spec: TaskSpec,
    *,
    root: Path,
    dataset_path: Path,
    media_root: Path | None,
    manifest: Mapping[str, object] | None,
    limit: Limit,
    offset: int,
    download: bool,
    announce: Callable[[str], None] | None = None,
    on_progress: ProgressCallback | None = None,
) -> Mapping[str, object] | None:
    """Make one task's selected media locally usable, returning an auto manifest if needed."""
    source = spec.media_source
    if source is None or _manifest_has_task(manifest, spec.name):
        return None
    announce = _serialized_announce(announce)
    managed_root = spec.media_root(root)
    effective_root = media_root or managed_root
    if effective_root is None:
        return None
    effective_root = effective_root.expanduser().resolve()
    managed = media_root is None
    patterns = _selected_patterns(spec, dataset_path, limit, offset)
    if not patterns:
        raise ValueError(f"{spec.name} produced no selected media")
    unavailable_units: Mapping[str, str] = {}
    if managed:
        unavailable_units = _acquire_selected(
            spec, dataset_path, root, patterns, download, announce
        )
    elif not effective_root.is_dir():
        raise FileNotFoundError(f"{spec.name} media root does not exist: {effective_root}")

    return _prepared_manifest(
        spec,
        dataset_path,
        effective_root,
        root / ".prepared" / _PREPARED_MEDIA_CACHE_VERSION / spec.name,
        root,
        limit,
        offset,
        announce,
        unavailable_units,
        on_progress,
    )


def _prepared_manifest(
    spec: TaskSpec,
    dataset: Path,
    media_root: Path,
    cache: Path,
    root: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
    unavailable_units: Mapping[str, str],
    on_progress: ProgressCallback | None,
) -> Mapping[str, object] | None:
    segment_announce = None if on_progress is not None else announce
    if spec.name in {"m3-bench-robot", "m3-bench-web"}:
        return {
            "units": _m3_manifest(
                dataset,
                media_root,
                cache,
                limit,
                offset,
                segment_announce,
                frozenset(unavailable_units),
                on_progress,
            ),
            **({"unavailable_units": dict(unavailable_units)} if unavailable_units else {}),
        }
    if spec.name in {"video-mme-v2", "egotempo"}:
        return {
            "units": _video_manifest(
                spec.name,
                dataset,
                media_root,
                cache,
                limit,
                offset,
                segment_announce,
                on_progress,
            )
        }
    if spec.name.startswith("mm-lifelong-"):
        return {
            "units": _lifelong_manifest(spec, media_root, cache, segment_announce, on_progress),
        }
    if spec.name.startswith("openeqa-"):
        return {
            "units": _openeqa_manifest(
                spec,
                dataset,
                media_root,
                cache,
                limit,
                offset,
                segment_announce,
                on_progress,
            )
        }
    if spec.name == "supermemory-vqa":
        return {
            "units": _supermemory_manifest(
                dataset,
                media_root,
                root,
                cache,
                limit,
                offset,
                segment_announce,
                on_progress,
            )
        }
    return None


def _acquire_selected(
    spec: TaskSpec,
    dataset: Path,
    root: Path,
    patterns: Sequence[str],
    download: bool,
    announce: Callable[[str], None] | None,
) -> Mapping[str, str]:
    source = spec.media_source
    if source is None:
        return {}
    if source.acquirer == "youtube":
        return _acquire_youtube(spec, dataset, root, patterns, download, announce)
    elif source.acquirer == "ego4d":
        _acquire_ego4d(spec, dataset, root, patterns, download, announce)
    elif source.acquirer in {"open-eqa-hm3d-frames", "scannet"}:
        _acquire_openeqa(spec, root, patterns)
    else:
        acquire_media(
            spec,
            root,
            patterns=patterns,
            download=download,
            allow_missing=spec.name == "supermemory-vqa",
            announce=announce,
        )
    return {}


def _manifest_has_task(manifest: Mapping[str, object] | None, task_name: str) -> bool:
    if manifest is None:
        return False
    tasks = manifest.get("tasks")
    return isinstance(tasks, dict) and task_name in tasks


def _selected_patterns(spec: TaskSpec, dataset: Path, limit: Limit, offset: int) -> tuple[str, ...]:
    source = spec.media_source
    if source is None:
        return ()
    name = spec.name
    if name in {"m3-bench-robot", "m3-bench-web"}:
        from mindbridge.benchmarks.m3_bench import load_m3_bench

        subset = "robot" if name.endswith("robot") else "web"
        return tuple(
            f"videos/{subset}/{_component(video.video_id)}.mp4"
            for video in _selected(load_m3_bench(dataset), limit, offset)
        )
    if name == "video-mme-v2":
        from mindbridge.benchmarks.video_mme_v2 import load_video_mme_v2

        volumes = {
            (int(group.video_id) - 1) // 20 + 1
            for group in _selected(load_video_mme_v2(dataset), limit, offset)
        }
        return tuple(f"videos/{volume:03d}.zip" for volume in sorted(volumes))
    if name == "egotempo":
        from mindbridge.benchmarks.egotempo import load_egotempo

        return tuple(
            f"videos/{clip_id}.mp4"
            for clip_id in dict.fromkeys(
                question.clip_id for question in _selected(load_egotempo(dataset), limit, offset)
            )
        )
    if name.startswith("mm-lifelong-"):
        return (f"videos/{str(spec.variant).split('_', 1)[0]}/*",)
    if name.startswith("openeqa-"):
        from mindbridge.benchmarks.openeqa import FRAME_GLOB, load_openeqa

        split = str(spec.variant)
        return tuple(
            f"data/frames/{split}/{episode}/{FRAME_GLOB}"
            for episode in _selected(
                tuple(
                    dict.fromkeys(
                        question.episode_name for question in load_openeqa(dataset, split=split)
                    )
                ),
                limit,
                offset,
            )
        )
    if name == "supermemory-vqa":
        questions = _supermemory_questions(dataset, limit, offset)
        horizon = max(question.question_ended_at.timestamp() for question in questions)
        starts = _supermemory_starts(dataset, 1)
        video_ids = tuple(
            video_id
            for video_id, started in sorted(starts.items(), key=lambda item: (item[1], item[0]))
            if started < horizon
        )
        return tuple(f"data/video/Person_1/{_component(video_id)}.mp4" for video_id in video_ids)
    return source.patterns


def _m3_manifest(
    dataset: Path,
    media_root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
    unavailable_units: frozenset[str] = frozenset(),
    on_progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    from mindbridge.benchmarks.m3_bench import M3BenchVideo, load_m3_bench

    find_media = _MediaFinder(media_root)

    def prepare(video: M3BenchVideo) -> tuple[str, list[dict[str, object]]]:
        source = find_media(video.video_path, f"{video.video_id}.mp4")
        cutoffs = tuple(question.cutoff_seconds for question in video.questions)
        duration = _duration(source)
        causal_cutoffs = tuple(value for value in cutoffs if value is not None)
        horizon = (
            duration if len(causal_cutoffs) != len(cutoffs) else max(causal_cutoffs, default=0.0)
        )
        boundaries = _grid(min(duration, horizon), causal_cutoffs)
        segments = _segment_video(source, boundaries, cache / _component(video.video_id), announce)
        return (
            video.video_id,
            [
                _path_part(path, f"{video.video_id}-{index:05d}", start, end)
                for index, (start, end, path) in enumerate(segments)
            ],
        )

    videos = tuple(
        video
        for video in _selected(load_m3_bench(dataset), limit, offset)
        if video.video_id not in unavailable_units
    )
    return dict(_prepare_many(prepare, videos, on_progress))


def _video_manifest(
    task_name: str,
    dataset: Path,
    media_root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    find_media = _MediaFinder(media_root)
    sources: tuple[tuple[str, Path], ...]
    if task_name == "video-mme-v2":
        from mindbridge.benchmarks.video_mme_v2 import load_video_mme_v2

        sources = tuple(
            (group.video_id, find_media(f"{group.video_id}.mp4"))
            for group in _selected(load_video_mme_v2(dataset), limit, offset)
        )
    else:
        from mindbridge.benchmarks.egotempo import load_egotempo

        sources = tuple(
            (clip_id, find_media(f"{clip_id}.mp4"))
            for clip_id in dict.fromkeys(
                question.clip_id for question in _selected(load_egotempo(dataset), limit, offset)
            )
        )

    def prepare(item: tuple[str, Path]) -> tuple[str, list[dict[str, object]]]:
        unit_id, source = item
        return unit_id, _video_parts(unit_id, source, cache / _component(unit_id), announce)

    return dict(_prepare_many(prepare, sources, on_progress))


def _lifelong_manifest(
    spec: TaskSpec,
    media_root: Path,
    cache: Path,
    announce: Callable[[str], None] | None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    files = tuple(
        path.resolve()
        for path in sorted(media_root.rglob("*"), key=lambda path: _natural_key(path, media_root))
        if path.is_file() and path.suffix.casefold() in {".mkv", ".mov", ".mp4", ".webm"}
    )
    if not files:
        raise FileNotFoundError(f"{spec.name} has no videos under {media_root}")

    def prepare(source: Path) -> tuple[str, tuple[tuple[float, float, Path], ...]]:
        relative = source.relative_to(media_root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()[:20]
        return relative, _segments(source, cache / key, announce)

    parts: list[dict[str, object]] = []
    timeline = 0.0
    for relative, segments in _prepare_many(prepare, files, on_progress):
        for index, (start, end, path) in enumerate(segments):
            parts.append(
                _path_part(
                    path,
                    f"{relative}-{index:05d}",
                    timeline + start,
                    timeline + end,
                )
            )
        timeline += segments[-1][1]
    return {str(spec.variant): parts}


def _openeqa_manifest(
    spec: TaskSpec,
    dataset: Path,
    media_root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    from mindbridge.benchmarks.openeqa import load_openeqa

    split = str(spec.variant)
    episodes = _selected(
        tuple(
            dict.fromkeys(question.episode_name for question in load_openeqa(dataset, split=split))
        ),
        limit,
        offset,
    )

    def prepare(episode: str) -> tuple[str, list[dict[str, object]]]:
        segments = _openeqa_segments(
            _openeqa_episode(media_root, split, episode),
            cache / _component(episode),
            announce,
        )
        return episode, [
            _path_part(path, f"{episode}-{index:05d}", start, end)
            for index, (start, end, path) in enumerate(segments)
        ]

    return dict(_prepare_many(prepare, episodes, on_progress))


def _openeqa_episode(media_root: Path, split: str, episode: str) -> Path:
    """Locate one episode history under either `frames/<split>` or `frames`."""
    for candidate in (media_root / episode, media_root / split / episode):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"OpenEQA episode history does not exist: {media_root / episode} "
        f"(nor {media_root / split / episode})"
    )


def _openeqa_segments(
    episode: Path,
    cache: Path,
    announce: Callable[[str], None] | None,
) -> tuple[tuple[float, float, Path], ...]:
    """Encode one episode's official frame order directly into cached causal segments."""
    from mindbridge.benchmarks.openeqa import episode_frames

    frames = episode_frames(episode)
    unquotable = tuple(frame for frame in frames if "'" in str(frame) or "\n" in str(frame))
    if unquotable:
        raise ValueError(f"OpenEQA frame path is not safe to encode: {unquotable[0]}")
    boundaries = _grid(len(frames) / _OPENEQA_FRAME_RATE, ())
    key = hashlib.sha256(
        json.dumps(
            [
                "openeqa-segments-v1",
                str(episode.resolve()),
                len(frames),
                frames[0].name,
                frames[-1].name,
                sum(frame.stat().st_size for frame in frames),
                _OPENEQA_FRAME_RATE,
                boundaries,
                _ffmpeg_id(),
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    seconds = 1 / _OPENEQA_FRAME_RATE
    # The concat demuxer takes the frame order this adapter resolved rather than
    # ffmpeg's own globbing. The usual "repeat the final entry" workaround is
    # not applied: `eof_action=pass` preserves the last frame's duration, while
    # repeating the final entry measurably adds a phantom frame (76 encoded from
    # 75 extracted).
    listing = "ffconcat version 1.0\n" + "".join(
        f"file '{frame.resolve()}'\nduration {seconds:.6f}\n" for frame in frames
    )

    def command(working: Path) -> list[str]:
        index = working / "frames.ffconcat"
        index.write_text(listing, encoding="utf-8")
        return [
            _executable("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(index),
            "-r",
            str(_OPENEQA_FRAME_RATE),
            "-t",
            f"{boundaries[-1]:.3f}",
            "-map",
            "0:v:0",
            "-vf",
            _OPENEQA_VIDEO_FILTER,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-map_metadata",
            "-1",
        ]

    return _cached_segments(episode, boundaries, cache, key, command, announce)


def _video_parts(
    unit_id: str,
    source: Path,
    cache: Path,
    announce: Callable[[str], None] | None,
) -> list[dict[str, object]]:
    return [
        _path_part(path, f"{unit_id}-{index:05d}", start, end)
        for index, (start, end, path) in enumerate(_segments(source, cache, announce))
    ]


def _segments(
    source: Path,
    cache: Path,
    announce: Callable[[str], None] | None,
) -> tuple[tuple[float, float, Path], ...]:
    duration = _duration(source)
    return _segment_video(source, _grid(duration, ()), cache, announce)


def _supermemory_manifest(
    dataset: Path,
    media_root: Path,
    root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, list[dict[str, object]]]:
    questions = _supermemory_questions(dataset, limit, offset)
    starts = _supermemory_starts(dataset, 1)
    horizon = max(question.question_ended_at.timestamp() for question in questions)
    video_ids = tuple(
        video_id
        for video_id, started in sorted(starts.items(), key=lambda item: (item[1], item[0]))
        if started < horizon
    )

    def prepare(video_id: str) -> tuple[dict[str, object], ...]:
        component = _component(video_id)
        started = starts[video_id]
        local_horizon = horizon - started
        transcript = _supermemory_transcript(root, video_id)
        question_cuts = tuple(
            question.question_ended_at.timestamp() - started
            for question in questions
            if question.question_video_id == video_id
        )
        source = media_root / "Person_1" / f"{component}.mp4"
        media_end = min(_duration(source), local_horizon) if source.is_file() else 0.0
        transcript_end = max((_number(line["end"]) for line in transcript), default=0.0)
        end = min(local_horizon, max((media_end, transcript_end, *question_cuts, 0.0)))
        if end <= 0:
            return ()
        boundaries = _grid(end, question_cuts)
        prepared: list[dict[str, object]] = []
        if media_end > 0:
            media_boundaries = tuple(value for value in boundaries if value < media_end)
            if not media_boundaries or media_boundaries[-1] != media_end:
                media_boundaries = (*media_boundaries, media_end)
            for index, (start, stop, path) in enumerate(
                _segment_video(source, media_boundaries, cache / component, announce)
            ):
                prepared.append(
                    _path_part(
                        path,
                        f"{video_id}-video-{index:05d}",
                        started + start,
                        started + stop,
                    )
                )
        previous = 0.0
        for index, stop in enumerate(boundaries):
            text = _transcript_text(transcript, previous, stop)
            if text:
                prepared.append(
                    {
                        "text": text,
                        "source_id": f"{video_id}-transcript-{index:05d}",
                        "start_seconds": started + previous,
                        "end_seconds": started + stop,
                    }
                )
            previous = stop
        return tuple(prepared)

    parts = [
        part for prepared in _prepare_many(prepare, video_ids, on_progress) for part in prepared
    ]
    if not parts:
        raise FileNotFoundError("SuperMemory-VQA selected no released video or transcript media")
    parts.sort(key=lambda part: (_number(part["end_seconds"]), str(part["source_id"])))
    return {"subject-1": parts}


def _supermemory_questions(
    dataset: Path, limit: Limit, offset: int
) -> tuple[SuperMemoryQuestion, ...]:
    from mindbridge.benchmarks.supermemory_vqa import load_supermemory_vqa

    questions = tuple(
        question for question in load_supermemory_vqa(dataset) if question.subject == 1
    )
    return tuple(_selected(questions, limit, offset))


def _supermemory_starts(dataset: Path, subject: int) -> dict[str, float]:
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    starts: dict[str, float] = {}
    for question in raw:
        if not isinstance(question, dict) or question.get("subject") != subject:
            continue
        evidence = question.get("question_evidence")
        if not isinstance(evidence, dict):
            continue
        spans = evidence.get("time_spans")
        if isinstance(spans, list):
            for span in spans:
                if isinstance(span, dict):
                    _record_start(starts, span.get("video_id"), span.get("video_start_time_unix"))
        _record_start(starts, evidence.get("video_id"), evidence.get("start_time"))
    return starts


def _record_start(starts: dict[str, float], video_id: object, value: object) -> None:
    if (
        not isinstance(video_id, str)
        or isinstance(value, bool)
        or not isinstance(value, int | float)
    ):
        return
    moment = float(value)
    previous = starts.setdefault(video_id, moment)
    if previous != moment:
        raise ValueError(f"SuperMemory-VQA gives {video_id} inconsistent start times")


def _supermemory_transcript(root: Path, video_id: str) -> tuple[dict[str, object], ...]:
    path = (
        root
        / "supermemory-vqa"
        / "data"
        / "transcripts"
        / "person_1"
        / f"{video_id.lower()}_gemini_aligned_transcript.json"
    )
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("transcript", ()) if isinstance(payload, dict) else ()
    if not isinstance(rows, list):
        raise ValueError(f"invalid SuperMemory-VQA transcript: {path}")
    return tuple(
        row
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("text"), str)
        and bool(str(row["text"]).strip())
        and isinstance(row.get("start"), int | float)
        and isinstance(row.get("end"), int | float)
        and not isinstance(row["start"], bool)
        and not isinstance(row["end"], bool)
        and 0 <= float(row["start"]) <= float(row["end"])
    )


def _transcript_text(lines: Sequence[Mapping[str, object]], start: float, end: float) -> str:
    selected = []
    for line in lines:
        line_start, line_end = _number(line["start"]), _number(line["end"])
        if line_start >= end or line_end <= start:
            continue
        text = str(line["text"]).strip()
        person = line.get("person")
        selected.append(f"{person}: {text}" if isinstance(person, str) and person else text)
    return "\n".join(selected)


def _acquire_youtube(
    spec: TaskSpec,
    dataset: Path,
    root: Path,
    patterns: Sequence[str],
    download: bool,
    announce: Callable[[str], None] | None,
) -> Mapping[str, str]:
    from mindbridge.benchmarks.m3_bench import load_m3_bench

    videos = {video.video_id: video for video in load_m3_bench(dataset)}
    selected = tuple(_component(PurePosixPath(pattern).stem) for pattern in patterns)
    source = spec.media_source
    if source is None:
        raise ValueError(f"{spec.name} has no media source")
    destination = root / source.release / "videos" / "web"
    unusable = tuple(
        video_id
        for video_id in selected
        if (destination / f"{video_id}.mp4").is_file()
        and not _has_audio(destination / f"{video_id}.mp4")
    )
    if unusable:
        raise RuntimeError(
            "M3-Bench web videos have no readable audio stream: " + ", ".join(unusable)
        )
    missing = tuple(
        video_id for video_id in selected if not (destination / f"{video_id}.mp4").is_file()
    )
    if not missing:
        return {}
    if not download:
        raise FileNotFoundError(f"{spec.name} media is missing and --no-download was given")
    command = _yt_dlp_command()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    sleep = _youtube_sleep()
    unavailable: dict[str, str] = {}
    for index, video_id in enumerate(missing, start=1):
        video = videos.get(video_id)
        if announce is not None:
            announce(f"downloading M3-Bench web video {index}/{len(missing)}: {video_id}")
        reason = _acquire_youtube_video(
            video_id,
            None if video is None else video.video_url,
            destination / f"{_component(video_id)}.mp4",
            command,
            sleep,
        )
        if reason is not None:
            unavailable[video_id] = reason
            if announce is not None:
                announce(f"skipping unavailable M3-Bench web video {video_id}: {reason}")
    return unavailable


def _acquire_youtube_video(
    video_id: str,
    video_url: str | None,
    target: Path,
    command: Sequence[str],
    sleep: float,
) -> str | None:
    if video_url is None:
        raise FileNotFoundError(f"M3-Bench web annotation has no URL for {video_id}")
    if urlparse(video_url).scheme not in {"http", "https"}:
        raise ValueError(f"M3-Bench web URL is not HTTP(S): {video_url!r}")
    returncode, output = _download_youtube_video(command, video_url, target, sleep)
    if not returncode and _valid_youtube_video(target):
        return None
    if returncode and _permanently_unavailable_youtube(output):
        target.unlink(missing_ok=True)
        return _yt_dlp_error(output)
    detail = f": {_yt_dlp_error(output)}" if output else ""
    raise RuntimeError(f"yt-dlp could not acquire M3-Bench web video {video_id}{detail}")


def _download_youtube_video(
    command: Sequence[str], url: str, target: Path, sleep: float
) -> tuple[int, str]:
    completed = subprocess.run(
        (
            *command,
            "--output",
            str(target),
            "--format-sort",
            "res:360,ext:mp4:m4a",
            "--merge-output-format",
            "mp4",
            "--remux-video",
            "mp4",
            "--no-playlist",
            "--sleep-interval",
            f"{sleep:g}",
            "--max-sleep-interval",
            f"{sleep * 2:g}",
            "--sleep-requests",
            f"{max(1.0, sleep / 4):g}",
            "--no-progress",
            "--",
            url,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return completed.returncode, output


def _valid_youtube_video(target: Path) -> bool:
    return target.is_file() and bool(target.stat().st_size) and _has_audio(target)


_PERMANENT_YOUTUBE_ERRORS = (
    "private video",
    "video unavailable",
    "has been removed",
    "removed by the uploader",
    "account associated with this video has been terminated",
    "blocked it in your country on copyright grounds",
    "not available in your country",
)


def _permanently_unavailable_youtube(output: str) -> bool:
    normalized = output.casefold()
    return any(message in normalized for message in _PERMANENT_YOUTUBE_ERRORS)


def _yt_dlp_error(output: str) -> str:
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    errors = tuple(line for line in lines if line.casefold().startswith("error:"))
    return (errors or lines or ("unknown yt-dlp error",))[-1]


def _acquire_openeqa(spec: TaskSpec, root: Path, patterns: Sequence[str]) -> None:
    """Accept episode histories an operator already extracted, and fetch nothing.

    There is nothing here to download: HM3D frames are a tarball the upstream
    `data/README.md` links from a third-party host, and ScanNet requires its own
    signed terms of use. What this does do is let frames extracted into the
    catalog's own media root count, so `--media-root` is a convenience rather
    than the only way to run the task.
    """
    source = spec.media_source
    if source is None:
        raise ValueError(f"{spec.name} has no media source")
    release = root / source.release
    missing = tuple(pattern for pattern in patterns if not tuple(release.glob(pattern)))
    if not missing:
        return
    raise FileNotFoundError(
        f"{spec.name} needs {len(missing)} of {len(patterns)} selected episode histories its "
        f"{source.acquirer} acquirer supplies; extract them under {release} or pass "
        f"--media-root {spec.name}=DIR. Absent: {missing[0]}"
    )


def _acquire_ego4d(
    spec: TaskSpec,
    dataset: Path,
    root: Path,
    patterns: Sequence[str],
    download: bool,
    announce: Callable[[str], None] | None,
) -> None:
    from mindbridge.benchmarks.egotempo import load_egotempo

    questions = {question.clip_id: question for question in load_egotempo(dataset)}
    selected = tuple(_component(PurePosixPath(pattern).stem) for pattern in patterns)
    media_source = spec.media_source
    if media_source is None:
        raise ValueError(f"{spec.name} has no media source")
    destination = root / media_source.release / "videos"
    pending = tuple(
        clip_id for clip_id in selected if not (destination / f"{clip_id}.mp4").is_file()
    )
    if not pending:
        return
    if not download:
        raise FileNotFoundError(f"{spec.name} media is missing and --no-download was given")
    unknown = tuple(clip_id for clip_id in pending if clip_id not in questions)
    if unknown:
        raise ValueError(f"unknown EgoTempo clip IDs: {', '.join(unknown)}")
    source_ids = tuple(
        dict.fromkeys(_component(questions[clip_id].source_video_id) for clip_id in pending)
    )
    source_root = root / media_source.release / "ego4d"
    sources = source_root / "v2" / "full_scale"
    absent = tuple(
        video_id for video_id in source_ids if not (sources / f"{video_id}.mp4").is_file()
    )
    if absent:
        command = _ego4d_command()
        profile = os.environ.get("AWS_PROFILE", "default")
        if announce is not None:
            announce(f"downloading {len(absent)} Ego4D full-scale video(s) for EgoTempo")
        completed = subprocess.run(
            (
                *command,
                "--output_directory",
                str(source_root),
                "--datasets",
                "full_scale",
                "--version",
                "v2_1",
                "--aws_profile_name",
                profile,
                "--no-metadata",
                "--yes",
                "--video_uids",
                *absent,
            ),
            check=False,
        )
        still_absent = tuple(
            video_id for video_id in absent if not (sources / f"{video_id}.mp4").is_file()
        )
        if completed.returncode or still_absent:
            raise PermissionError(
                "Ego4D download failed; accept the Ego4D access agreement and configure the "
                f"AWS profile {profile!r} before retrying"
            )
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)

    def cut(clip_id: str) -> None:
        question = questions[clip_id]
        if announce is not None:
            announce(f"cutting EgoTempo clip {clip_id}")
        _trim_video(
            sources / f"{_component(question.source_video_id)}.mp4",
            destination / f"{clip_id}.mp4",
            question.clip_start_seconds,
            question.clip_end_seconds,
        )

    _prepare_many(cut, pending)


class _MediaFinder:
    """Resolve direct media names, indexing a nested tree only if a fallback needs it."""

    def __init__(self, root: Path) -> None:
        self._display_root = root
        self._root = root.resolve()
        self._index: dict[str, tuple[Path, ...]] | None = None
        self._lock = Lock()

    def __call__(self, *names: str) -> Path:
        for name in names:
            supplied = Path(name)
            candidates = tuple(
                candidate
                for candidate in (
                    (self._root / supplied).resolve(),
                    (self._root / supplied.name).resolve(),
                )
                if candidate.is_relative_to(self._root)
            )
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
        stems = {Path(name).stem.casefold() for name in names}
        index = self._fallback_index()
        matches = tuple(path for stem in stems for path in index.get(stem, ()))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one of {', '.join(names)} under {self._display_root}"
            )
        return matches[0]

    def _fallback_index(self) -> dict[str, tuple[Path, ...]]:
        if self._index is None:
            with self._lock:
                if self._index is None:
                    grouped: dict[str, list[Path]] = {}
                    for path in self._root.rglob("*"):
                        if path.is_file() and (resolved := path.resolve()).is_relative_to(
                            self._root
                        ):
                            grouped.setdefault(path.stem.casefold(), []).append(resolved)
                    # ponytail: this is one manifest's snapshot; the next manifest gets a fresh one.
                    self._index = {stem: tuple(paths) for stem, paths in grouped.items()}
        return self._index


def _find_media(root: Path, *names: str) -> Path:
    return _MediaFinder(root)(*names)


def _grid(end: float, forced: Sequence[float]) -> tuple[float, ...]:
    if end <= 0:
        raise ValueError("selected video horizon must be positive")
    values = {
        *(float(value) for value in forced if 0 < value < end),
        *(float(value) for value in range(_SEGMENT_SECONDS, int(end), _SEGMENT_SECONDS)),
        float(end),
    }
    return tuple(sorted(values))


def _segment_video(
    source: Path,
    boundaries: Sequence[float],
    cache: Path,
    announce: Callable[[str], None] | None,
) -> tuple[tuple[float, float, Path], ...]:
    ordered = tuple(sorted(dict.fromkeys(float(value) for value in boundaries)))
    if not ordered or ordered[0] <= 0:
        raise ValueError("video segment boundaries must be positive")
    stat_result = source.stat()

    def cache_key(version: str) -> str:
        return hashlib.sha256(
            json.dumps(
                [
                    version,
                    str(source.resolve()),
                    stat_result.st_size,
                    stat_result.st_mtime_ns,
                    ordered,
                    _ffmpeg_id(),
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:20]

    key = cache_key(_VIDEO_SEGMENT_CACHE_KEY_VERSION)

    def command(_working: Path, video_filter: str = _VIDEO_FILTER) -> list[str]:
        return [
            _executable("ffmpeg"),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            f"{ordered[-1]:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            video_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-tune",
            "zerolatency",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-map_metadata",
            "-1",
        ]

    return _cached_segments(
        source,
        ordered,
        cache,
        key,
        command,
        announce,
        fallback_command=lambda working: command(working, _VIDEO_FILTER_NATIVE),
        legacy_paths=_legacy_segment_paths(
            cache, cache_key(_LEGACY_VIDEO_SEGMENT_CACHE_KEY_VERSION), ordered
        ),
    )


def _cached_segments(
    source: Path,
    boundaries: Sequence[float],
    cache: Path,
    key: str,
    build_command: Callable[[Path], list[str]],
    announce: Callable[[str], None] | None,
    *,
    fallback_command: Callable[[Path], list[str]] | None = None,
    legacy_paths: Sequence[Path] = (),
) -> tuple[tuple[float, float, Path], ...]:
    target = cache / key
    expected = tuple(target / f"segment-{index:05d}.mp4" for index in range(len(boundaries)))
    # Every object below is probed with a full decode before it is trusted, and the marker is
    # written only after all of them passed. A completed entry is then a stat per segment on
    # later runs instead of a decode per segment, which on a thousand-segment corpus is the
    # difference between seconds and an hour before the first question is asked.
    # ponytail: a completed entry corrupted in place is trusted; delete the marker to re-probe.
    complete = target / _COMPLETE_MARKER
    if complete.is_file() and all(path.is_file() and path.stat().st_size for path in expected):
        return _timed_paths(boundaries, expected)
    if all(_has_structural_video(path) for path in expected):
        complete.touch()
        return _timed_paths(boundaries, expected)
    _reuse_valid_legacy(legacy_paths, expected)
    if all(_has_structural_video(path) for path in expected):
        complete.touch()
        return _timed_paths(boundaries, expected)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if announce is not None:
        announce(f"preparing {len(boundaries)} causal segments from {source.name}")
    working = Path(mkdtemp(prefix=f".{key}.", dir=target.parent))
    try:
        produced = _produce_segments(source, boundaries, working, build_command, fallback_command)
        target.mkdir(mode=0o700, exist_ok=True)
        for source_path, target_path in zip(produced, expected, strict=True):
            # A partial cache can contain valid earlier segments beside a failed tail segment.
            # Never overwrite those immutable bytes while repairing only their invalid siblings.
            if not _has_structural_video(target_path):
                os.replace(source_path, target_path)
        if not all(_has_structural_video(path) for path in expected):
            raise RuntimeError(f"video segment cache remained incomplete for {source}")
        complete.touch()
    finally:
        shutil.rmtree(working)
    return _timed_paths(boundaries, expected)


def _reuse_valid_legacy(legacy_paths: Sequence[Path], expected: Sequence[Path]) -> None:
    """Reuse only legacy segments that already meet the processor's two-frame threshold.

    A legacy one-frame object may have lost real source frames to the old fps representation. It
    must therefore take the native-cadence repair path instead of becoming a completed V2 entry.
    """
    if len(legacy_paths) != len(expected):
        return
    expected[0].parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    for legacy_path, target_path in zip(legacy_paths, expected, strict=True):
        if _has_structural_video(target_path) or not _has_preprocessable_video(legacy_path):
            continue
        target_path.unlink(missing_ok=True)
        try:
            os.link(legacy_path, target_path)
        except OSError:
            shutil.copy2(legacy_path, target_path)


def _produce_segments(
    source: Path,
    boundaries: Sequence[float],
    working: Path,
    build_command: Callable[[Path], list[str]],
    fallback_command: Callable[[Path], list[str]] | None,
) -> tuple[Path, ...]:
    normal = working / "normal"
    normal.mkdir()
    output = normal / "segment-%05d.mp4"
    command = _segment_output_command(build_command(normal), boundaries, output)
    _run_ffmpeg(command, source)
    produced = tuple(normal / f"segment-{index:05d}.mp4" for index in range(len(boundaries)))
    if len(produced) == len(boundaries) and all(
        _has_preprocessable_video(path) for path in produced
    ):
        return produced
    if fallback_command is None:
        if all(_has_structural_video(path) for path in produced):
            return produced
        raise RuntimeError(
            f"ffmpeg produced {len(produced)} structurally invalid video segments for {source}"
        )
    native = working / "native"
    native.mkdir()
    native_output = native / "segment-%05d.mp4"
    command = _segment_output_command(fallback_command(native), boundaries, native_output)
    _run_ffmpeg(command, source)
    fallback = tuple(native / f"segment-{index:05d}.mp4" for index in range(len(boundaries)))
    selected = tuple(
        normal_path
        if _has_preprocessable_video(normal_path)
        else fallback_path
        if _has_preprocessable_video(fallback_path)
        else normal_path
        if _has_structural_video(normal_path)
        else fallback_path
        for normal_path, fallback_path in zip(produced, fallback, strict=True)
    )
    if not all(_has_structural_video(path) for path in selected):
        raise RuntimeError(f"ffmpeg produced structurally invalid video segments for {source}")
    return selected


def _segment_output_command(
    command: list[str], boundaries: Sequence[float], output: Path
) -> list[str]:
    cuts = boundaries[:-1]
    if cuts:
        times = ",".join(f"{value:.3f}" for value in cuts)
        command.extend(
            (
                "-force_key_frames",
                times,
                "-segment_times",
                times,
                "-f",
                "segment",
                "-segment_format",
                "mp4",
                "-reset_timestamps",
                "1",
                str(output),
            )
        )
    else:
        # The segment muxer splits an uncut source at its keyframes. A single boundary needs one
        # named output regardless of the source's keyframe cadence.
        command.append(str(output.with_name("segment-00000.mp4")))
    return command


def _legacy_segment_paths(cache: Path, key: str, boundaries: Sequence[float]) -> tuple[Path, ...]:
    """Locate only the prior prepared-media namespace for a matching legacy cache key."""
    parts = cache.parts
    try:
        index = parts.index(_PREPARED_MEDIA_CACHE_VERSION)
    except ValueError:
        return ()
    legacy_root = Path(*parts[:index], _LEGACY_PREPARED_MEDIA_CACHE_VERSION, *parts[index + 1 :])
    return tuple(
        legacy_root / key / f"segment-{offset:05d}.mp4" for offset in range(len(boundaries))
    )


def _timed_paths(
    boundaries: Sequence[float], paths: Sequence[Path]
) -> tuple[tuple[float, float, Path], ...]:
    start = 0.0
    result = []
    for end, path in zip(boundaries, paths, strict=True):
        result.append((start, end, path.resolve()))
        start = end
    return tuple(result)


def _trim_video(source: Path, target: Path, start: float, end: float) -> None:
    if target.is_file() and target.stat().st_size:
        return
    start = max(0.0, start)
    if end <= start:
        raise ValueError("video trim end must follow its start")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            suffix=target.suffix, dir=target.parent, prefix=f".{target.stem}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
        _run_ffmpeg(
            (
                _executable("ffmpeg"),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{end - start:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                _VIDEO_FILTER,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-map_metadata",
                "-1",
                str(temporary),
            ),
            source,
        )
        if not temporary.stat().st_size:
            raise RuntimeError(f"ffmpeg wrote an empty clip for {source}")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_ffmpeg(command: Sequence[str], source: Path) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        reason = " ".join(completed.stderr.split())[-1_000:]
        raise RuntimeError(f"ffmpeg failed for {source}: {reason}")


def _duration(path: Path) -> float:
    completed = subprocess.run(
        (
            _executable("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration = float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"ffprobe could not read a duration from {path}") from error
    if completed.returncode or duration <= 0:
        raise RuntimeError(f"ffprobe could not read a positive duration from {path}")
    return duration


@lru_cache(maxsize=1)
def _ffmpeg_id() -> str:
    executable = _executable("ffmpeg")
    completed = subprocess.run(
        (executable, "-version"), capture_output=True, text=True, check=False
    )
    first = completed.stdout.splitlines()[:1]
    return first[0] if first else executable


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"automatic benchmark video preparation requires {name} on PATH")
    return executable


def _yt_dlp_command() -> tuple[str, ...]:
    if shutil.which("uvx"):
        return ("uvx", "yt-dlp@latest")
    if shutil.which("yt-dlp"):
        return ("yt-dlp",)
    raise FileNotFoundError("automatic M3-Bench web download requires uvx or yt-dlp on PATH")


def _ego4d_command() -> tuple[str, ...]:
    if shutil.which("uvx"):
        return ("uvx", "--from", "ego4d==1.7.3", "ego4d")
    if shutil.which("ego4d"):
        return ("ego4d",)
    raise FileNotFoundError(
        "automatic EgoTempo download requires the official ego4d CLI and approved Ego4D access"
    )


def _youtube_sleep() -> float:
    value = os.environ.get("MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS", "30")
    try:
        seconds = float(value)
    except ValueError as error:
        raise ValueError("MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS must be numeric") from error
    if seconds < 0:
        raise ValueError("MINDBRIDGE_BENCH_YOUTUBE_SLEEP_SECONDS must not be negative")
    return seconds


def _has_audio(path: Path) -> bool:
    completed = subprocess.run(
        (
            _executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _video_frame_count(path: Path) -> int:
    """Return the number of real decoded frames, or zero for a missing or invalid stream."""
    if not path.is_file() or not path.stat().st_size:
        return 0
    completed = subprocess.run(
        (
            _executable("ffprobe"),
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        frame_count = int(completed.stdout.strip())
    except ValueError:
        return 0
    return frame_count if completed.returncode == 0 else 0


def _has_structural_video(path: Path) -> bool:
    """Accept a cache object with at least one real video frame."""
    return _has_preprocessable_video(path) or _video_frame_count(path) == 1


def _has_preprocessable_video(path: Path) -> bool:
    """Identify the two-real-frame minimum required by the WeMM/Qwen video processor."""
    return _video_frame_count(path) >= _MIN_PREPROCESSABLE_VIDEO_FRAMES


def _component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"benchmark media ID is not one path component: {value!r}")
    return value


def _natural_key(path: Path, root: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"([0-9]+)", path.relative_to(root).as_posix())
    )


def _path_part(path: Path, source_id: str, start: float, end: float) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "source_id": source_id,
        "start_seconds": start,
        "end_seconds": end,
    }


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("benchmark media timestamp must be numeric")
    return float(value)


def _selected(values: Sequence[_T], limit: Limit, offset: int) -> Sequence[_T]:
    if limit is None or limit == -1:
        return values[offset:]
    count = max(1, math.ceil(len(values) * limit)) if limit < 1 else int(limit)
    return values[offset : offset + count]
