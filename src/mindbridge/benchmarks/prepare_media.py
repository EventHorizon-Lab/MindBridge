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
from functools import lru_cache
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, mkdtemp
from typing import TYPE_CHECKING, TypeAlias, TypeVar
from urllib.parse import urlparse

from mindbridge.benchmarks.download import acquire_media
from mindbridge.benchmarks.task_catalog import TaskSpec

if TYPE_CHECKING:
    from mindbridge.benchmarks.supermemory_vqa import SuperMemoryQuestion

_SEGMENT_SECONDS = 30
_VIDEO_FILTER = (
    "fps=1,scale=w='min(640\\,iw)':h='min(360\\,ih)':"
    "force_original_aspect_ratio=decrease:force_divisible_by=2"
)
_EGOLIFE_NAME = re.compile(r"^DAY([1-9][0-9]*)_[^_]+_[^_]+_([0-9]{8})$")
_T = TypeVar("_T")
Limit: TypeAlias = int | float | None


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
) -> Mapping[str, object] | None:
    """Make one task's selected media locally usable, returning an auto manifest if needed."""
    source = spec.media_source
    if source is None or _manifest_has_task(manifest, spec.name):
        return None
    managed_root = spec.media_root(root)
    effective_root = media_root or managed_root
    if effective_root is None:
        return None
    effective_root = effective_root.expanduser().resolve()
    managed = media_root is None
    patterns = _selected_patterns(spec, dataset_path, limit, offset)
    if not patterns:
        raise ValueError(f"{spec.name} produced no selected media")
    if managed:
        _acquire_selected(spec, dataset_path, root, patterns, download, announce)
    elif not effective_root.is_dir():
        raise FileNotFoundError(f"{spec.name} media root does not exist: {effective_root}")

    return _prepared_manifest(
        spec,
        dataset_path,
        effective_root,
        root / ".prepared" / "ffmpeg-v1" / spec.name,
        root,
        limit,
        offset,
        announce,
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
) -> Mapping[str, object] | None:
    if spec.name in {"m3-bench-robot", "m3-bench-web"}:
        return {"units": _m3_manifest(dataset, media_root, cache, limit, offset, announce)}
    if spec.name in {"video-mme", "video-mme-v2", "egotempo"}:
        return {
            "units": _video_manifest(spec.name, dataset, media_root, cache, limit, offset, announce)
        }
    if spec.name in {"egolifeqa", "egomemreason"}:
        return {"units": _egolife_manifest(spec.name, dataset, media_root, limit, offset)}
    if spec.name.startswith("mm-lifelong-"):
        return {
            "units": _lifelong_manifest(spec, media_root, cache, announce),
        }
    if spec.name == "supermemory-vqa":
        return {
            "units": _supermemory_manifest(
                dataset, media_root, root, cache, limit, offset, announce
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
) -> None:
    source = spec.media_source
    if source is None:
        return
    if source.acquirer == "youtube":
        _acquire_youtube(spec, dataset, root, patterns, download, announce)
    elif source.acquirer == "ego4d":
        _acquire_ego4d(spec, dataset, root, patterns, download, announce)
    else:
        acquire_media(
            spec,
            root,
            patterns=patterns,
            download=download,
            allow_missing=spec.name == "supermemory-vqa",
            announce=announce,
        )


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
    if name in {"egolifeqa", "egomemreason"}:
        return _egolife_patterns(name, dataset, limit, offset)
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


def _egolife_patterns(task_name: str, dataset: Path, limit: Limit, offset: int) -> tuple[str, ...]:
    if task_name == "egolifeqa":
        from mindbridge.benchmarks.egolife_qa import load_egolife_qa

        questions = _selected(load_egolife_qa(dataset), limit, offset)
        horizon = max(question.query_offset_ms for question in questions)
        return tuple(f"A1_JAKE/DAY{day}/*.mp4" for day in range(1, horizon // 86_400_000 + 2))
    from mindbridge.benchmarks.egomem_reason import load_egomem_reason

    horizons: dict[str, int] = {}
    for question in _selected(load_egomem_reason(dataset), limit, offset):
        horizons[question.identity] = max(
            horizons.get(question.identity, 0), question.query_offset_ms
        )
    return tuple(
        f"{_component(identity)}/DAY{day}/*.mp4"
        for identity, horizon in sorted(horizons.items())
        for day in range(1, horizon // 86_400_000 + 2)
    )


def _m3_manifest(
    dataset: Path,
    media_root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
) -> dict[str, list[dict[str, object]]]:
    from mindbridge.benchmarks.m3_bench import load_m3_bench

    units: dict[str, list[dict[str, object]]] = {}
    for video in _selected(load_m3_bench(dataset), limit, offset):
        source = _find_media(media_root, video.video_path, f"{video.video_id}.mp4")
        cutoffs = tuple(
            float((question.before_clip_index + 1) * _SEGMENT_SECONDS)
            if question.before_clip_index is not None
            else None
            for question in video.questions
        )
        duration = _duration(source)
        causal_cutoffs = tuple(value for value in cutoffs if value is not None)
        horizon = (
            duration if len(causal_cutoffs) != len(cutoffs) else max(causal_cutoffs, default=0.0)
        )
        boundaries = _grid(min(duration, horizon), causal_cutoffs)
        segments = _segment_video(source, boundaries, cache / _component(video.video_id), announce)
        units[video.video_id] = [
            _path_part(path, f"{video.video_id}-{index:05d}", start, end)
            for index, (start, end, path) in enumerate(segments)
        ]
    return units


def _video_manifest(
    task_name: str,
    dataset: Path,
    media_root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
) -> dict[str, list[dict[str, object]]]:
    sources: tuple[tuple[str, Path], ...]
    if task_name == "video-mme":
        from mindbridge.benchmarks.video_mme import load_video_mme

        sources = tuple(
            (
                video.video_id,
                _find_media(media_root, f"{video.source_video_id}.mp4"),
            )
            for video in _selected(load_video_mme(dataset), limit, offset)
        )
    elif task_name == "video-mme-v2":
        from mindbridge.benchmarks.video_mme_v2 import load_video_mme_v2

        sources = tuple(
            (group.video_id, _find_media(media_root, f"{group.video_id}.mp4"))
            for group in _selected(load_video_mme_v2(dataset), limit, offset)
        )
    else:
        from mindbridge.benchmarks.egotempo import load_egotempo

        sources = tuple(
            (clip_id, _find_media(media_root, f"{clip_id}.mp4"))
            for clip_id in dict.fromkeys(
                question.clip_id for question in _selected(load_egotempo(dataset), limit, offset)
            )
        )
    return {
        unit_id: _video_parts(unit_id, source, cache / _component(unit_id), announce)
        for unit_id, source in sources
    }


def _lifelong_manifest(
    spec: TaskSpec,
    media_root: Path,
    cache: Path,
    announce: Callable[[str], None] | None,
) -> dict[str, list[dict[str, object]]]:
    files = tuple(
        path.resolve()
        for path in sorted(media_root.rglob("*"), key=lambda path: _natural_key(path, media_root))
        if path.is_file() and path.suffix.casefold() in {".mkv", ".mov", ".mp4", ".webm"}
    )
    if not files:
        raise FileNotFoundError(f"{spec.name} has no videos under {media_root}")
    parts: list[dict[str, object]] = []
    timeline = 0.0
    for source in files:
        relative = source.relative_to(media_root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()[:20]
        segments = _segments(source, cache / key, announce)
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


def _egolife_manifest(
    task_name: str,
    dataset: Path,
    media_root: Path,
    limit: Limit,
    offset: int,
) -> dict[str, list[dict[str, object]]]:
    horizons: dict[str, int]
    if task_name == "egolifeqa":
        from mindbridge.benchmarks.egolife_qa import load_egolife_qa

        selected = _selected(load_egolife_qa(dataset), limit, offset)
        horizons = {"A1_JAKE": max(question.query_offset_ms for question in selected)}
    else:
        from mindbridge.benchmarks.egomem_reason import load_egomem_reason

        horizons = {}
        for question in _selected(load_egomem_reason(dataset), limit, offset):
            horizons[question.identity] = max(
                horizons.get(question.identity, 0), question.query_offset_ms
            )
    units: dict[str, list[dict[str, object]]] = {}
    for identity, horizon_ms in sorted(horizons.items()):
        _component(identity)
        parts = []
        for path in sorted((media_root / identity).glob("DAY*/*.mp4")):
            start = _egolife_start(path)
            end = start + _SEGMENT_SECONDS
            if end * 1_000 <= horizon_ms:
                parts.append(_path_part(path.resolve(), path.stem, start, end))
        if not parts:
            raise FileNotFoundError(
                f"no complete EgoLife clip for {identity} precedes the selected query horizon"
            )
        units[identity] = parts
    return units


def _supermemory_manifest(
    dataset: Path,
    media_root: Path,
    root: Path,
    cache: Path,
    limit: Limit,
    offset: int,
    announce: Callable[[str], None] | None,
) -> dict[str, list[dict[str, object]]]:
    questions = _supermemory_questions(dataset, limit, offset)
    starts = _supermemory_starts(dataset, 1)
    horizon = max(question.question_ended_at.timestamp() for question in questions)
    video_ids = tuple(
        video_id
        for video_id, started in sorted(starts.items(), key=lambda item: (item[1], item[0]))
        if started < horizon
    )
    parts: list[dict[str, object]] = []
    for video_id in video_ids:
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
            continue
        boundaries = _grid(end, question_cuts)
        if media_end > 0:
            media_boundaries = tuple(value for value in boundaries if value < media_end)
            if not media_boundaries or media_boundaries[-1] != media_end:
                media_boundaries = (*media_boundaries, media_end)
            for index, (start, stop, path) in enumerate(
                _segment_video(source, media_boundaries, cache / component, announce)
            ):
                parts.append(
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
                parts.append(
                    {
                        "text": text,
                        "source_id": f"{video_id}-transcript-{index:05d}",
                        "start_seconds": started + previous,
                        "end_seconds": started + stop,
                    }
                )
            previous = stop
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
) -> None:
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
        return
    if not download:
        raise FileNotFoundError(f"{spec.name} media is missing and --no-download was given")
    command = _yt_dlp_command()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    sleep = _youtube_sleep()
    for index, video_id in enumerate(missing, start=1):
        video = videos.get(video_id)
        if video is None or video.video_url is None:
            raise FileNotFoundError(f"M3-Bench web annotation has no URL for {video_id}")
        if urlparse(video.video_url).scheme not in {"http", "https"}:
            raise ValueError(f"M3-Bench web URL is not HTTP(S): {video.video_url!r}")
        target = destination / f"{_component(video_id)}.mp4"
        if announce is not None:
            announce(f"downloading M3-Bench web video {index}/{len(missing)}: {video_id}")
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
                video.video_url,
            ),
            check=False,
        )
        if (
            completed.returncode
            or not target.is_file()
            or not target.stat().st_size
            or not _has_audio(target)
        ):
            raise RuntimeError(f"yt-dlp could not acquire M3-Bench web video {video_id}")


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
    for clip_id in pending:
        question = questions[clip_id]
        if announce is not None:
            announce(f"cutting EgoTempo clip {clip_id}")
        _trim_video(
            sources / f"{_component(question.source_video_id)}.mp4",
            destination / f"{clip_id}.mp4",
            question.clip_start_seconds,
            question.clip_end_seconds,
        )


def _find_media(root: Path, *names: str) -> Path:
    resolved_root = root.resolve()
    for name in names:
        supplied = Path(name)
        candidates = tuple(
            candidate
            for candidate in (
                (resolved_root / supplied).resolve(),
                (resolved_root / supplied.name).resolve(),
            )
            if candidate.is_relative_to(resolved_root)
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    stems = {Path(name).stem.casefold() for name in names}
    matches = tuple(
        resolved
        for path in resolved_root.rglob("*")
        if path.is_file()
        and path.stem.casefold() in stems
        and (resolved := path.resolve()).is_relative_to(resolved_root)
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one of {', '.join(names)} under {root}")
    return matches[0]


def _egolife_start(path: Path) -> float:
    from mindbridge.benchmarks.egolife_qa import egolife_timecode_offset_ms

    match = _EGOLIFE_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"invalid EgoLife video name: {path.name}")
    return egolife_timecode_offset_ms(int(match.group(1)), match.group(2)) / 1_000


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
    key = hashlib.sha256(
        json.dumps(
            [
                str(source.resolve()),
                stat_result.st_size,
                stat_result.st_mtime_ns,
                ordered,
                _ffmpeg_id(),
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    target = cache / key
    expected = tuple(target / f"segment-{index:05d}.mp4" for index in range(len(ordered)))
    if all(path.is_file() and path.stat().st_size for path in expected):
        return _timed_paths(ordered, expected)
    executable = _executable("ffmpeg")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if announce is not None:
        announce(f"preparing {len(ordered)} causal segments from {source.name}")
    working = Path(mkdtemp(prefix=f".{key}.", dir=target.parent))
    temporary_path: Path | None = working
    try:
        output = working / "segment-%05d.mp4"
        cuts = ordered[:-1]
        command = [
            executable,
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
        ]
        if cuts:
            times = ",".join(f"{value:.3f}" for value in cuts)
            command.extend(("-force_key_frames", times, "-segment_times", times))
        command.extend(
            (
                "-f",
                "segment",
                "-segment_format",
                "mp4",
                "-reset_timestamps",
                "1",
                str(output),
            )
        )
        _run_ffmpeg(command, source)
        produced = tuple(sorted(working.glob("segment-*.mp4")))
        if len(produced) != len(expected) or any(not path.stat().st_size for path in produced):
            raise RuntimeError(
                f"ffmpeg produced {len(produced)} of {len(expected)} segments for {source}"
            )
        target.mkdir(mode=0o700, exist_ok=True)
        for source_path, target_path in zip(produced, expected, strict=True):
            if not target_path.is_file() or not target_path.stat().st_size:
                os.replace(source_path, target_path)
        if not all(path.is_file() and path.stat().st_size for path in expected):
            raise RuntimeError(f"video segment cache remained incomplete for {source}")
    finally:
        if temporary_path is not None:
            shutil.rmtree(temporary_path)
    return _timed_paths(ordered, expected)


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
