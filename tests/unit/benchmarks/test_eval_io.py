"""Small regression checks for resumable benchmark input and response caches."""

from __future__ import annotations

import json
import os
import stat
import zipfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import mindbridge.benchmarks.eval as eval_module
from mindbridge import MindBridgeConfig, Modality
from mindbridge.benchmarks.atm_bench import ATM_BENCH_ADAPTER_VERSION
from mindbridge.benchmarks.download import acquire_media
from mindbridge.benchmarks.eval import _cache_task
from mindbridge.benchmarks.eval_adapters import MediaResolver, load_task
from mindbridge.benchmarks.eval_cache import (
    CachedAnswer,
    DescriptionCache,
    EvidenceInterval,
    ResponseCache,
)
from mindbridge.benchmarks.prepare_media import (
    _OPENEQA_FRAME_RATE,
    _egolife_manifest,
    _lifelong_manifest,
    _m3_manifest,
    _openeqa_episode,
    _openeqa_video,
    _segment_video,
    _selected_patterns,
    prepare_task_media,
)
from mindbridge.benchmarks.task_catalog import TASKS, MediaSource, TaskSpec
from mindbridge.models.base import ModelInput
from mindbridge.types import AssetRef


def _media_spec() -> TaskSpec:
    return TaskSpec(
        "fixture",
        "Fixture",
        "fixture/data.json",
        "v1",
        "owner/annotations",
        "0" * 40,
        media_source=MediaSource("fixture-media", "owner/media", "1" * 40, ("part.zip",)),
    )


def test_archive_download_is_resumable_and_rejects_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = tmp_path / "fixture-media"
    archive = release / "part.zip"

    def snapshot(
        repository: str, revision: str, patterns: tuple[str, ...], destination: Path
    ) -> None:
        assert (repository, revision, patterns) == ("owner/media", "1" * 40, ("part.zip",))
        destination.mkdir()
        with zipfile.ZipFile(destination / "part.zip", "w") as volume:
            volume.writestr("videos/example.mp4", b"video")

    monkeypatch.setattr("mindbridge.benchmarks.download._snapshot", snapshot)
    acquire_media(_media_spec(), tmp_path)
    assert (release / "videos" / "example.mp4").read_bytes() == b"video"

    archive.unlink()
    acquire_media(_media_spec(), tmp_path, download=False)
    with zipfile.ZipFile(archive, "w") as volume:
        volume.writestr("../escape.mp4", b"bad")
    with pytest.raises(ValueError, match="outside its directory"):
        acquire_media(_media_spec(), tmp_path, download=False)
    assert not (tmp_path / "escape.mp4").exists()


def test_release_media_paths_cannot_escape_the_media_root(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")

    with pytest.raises(FileNotFoundError, match="media not found"):
        MediaResolver("fixture", media, None, None).path(str(outside))


def test_m3_timestamp_drives_preparation_without_before_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "robot.json"
    dataset.write_text(
        json.dumps(
            {
                "room": {
                    "video_path": "room.mp4",
                    "qa_list": [
                        {
                            "question": "Where?",
                            "answer": "There.",
                            "question_id": "q1",
                            "type": ["recall"],
                            "timestamp": "00:10",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "media"
    media.mkdir()
    (media / "room.mp4").write_bytes(b"video")
    observed: list[float] = []

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._duration", lambda _path: 60.0)

    def segment(
        _source: Path, boundaries: tuple[float, ...], cache: Path, _announce: object
    ) -> tuple[tuple[float, float, Path], ...]:
        observed.extend(boundaries)
        return ((0.0, boundaries[-1], cache / "segment.mp4"),)

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._segment_video", segment)
    _m3_manifest(dataset, media, tmp_path / "cache", None, 0, None)

    assert observed == [10.0]


def test_egolife_ingests_reencoded_segments_at_the_declared_causal_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "EgoLifeQA_A1_JAKE.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "ID": "q1",
                    "query_time": {"date": "DAY1", "time": "00003000"},
                    "type": "EntityLog",
                    "need_audio": False,
                    "need_name": False,
                    "last_time": False,
                    "question": "Where is it?",
                    "choice_a": "a",
                    "choice_b": "b",
                    "choice_c": "c",
                    "choice_d": "d",
                    "answer": "A",
                }
            ]
        ),
        encoding="utf-8",
    )
    day = tmp_path / "media" / "A1_JAKE" / "DAY1"
    day.mkdir(parents=True)
    source = day / "DAY1_A1_JAKE_00000000.mp4"
    source.write_bytes(b"video")
    requested: list[tuple[Path, tuple[float, ...]]] = []

    def segment(
        source_path: Path, boundaries: tuple[float, ...], cache: Path, _announce: object
    ) -> tuple[tuple[float, float, Path], ...]:
        requested.append((source_path, boundaries))
        prepared = cache / "prepared.mp4"
        prepared.parent.mkdir(parents=True, exist_ok=True)
        prepared.write_bytes(b"reduced")
        return ((0.0, boundaries[-1], prepared),)

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._segment_video", segment)
    units = _egolife_manifest(
        "egolifeqa", dataset, tmp_path / "media", tmp_path / "cache", None, 0, None
    )

    # One re-encode per source clip, and the indexed part is the reduced copy rather than the
    # 20 fps original -- while the causal window still comes from the filename's timecode.
    assert requested == [(source, (30.0,))]
    assert units["A1_JAKE"] == [
        {
            "path": str((tmp_path / "cache" / "A1_JAKE" / "prepared.mp4").resolve()),
            "source_id": "DAY1_A1_JAKE_00000000",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
        }
    ]


def test_evaluation_digest_tracks_media_root_content(tmp_path: Path) -> None:
    dataset = tmp_path / "robot.json"
    dataset.write_text(
        json.dumps(
            {
                "room": {
                    "video_path": "room.mp4",
                    "qa_list": [
                        {
                            "question": "Where?",
                            "answer": "There.",
                            "question_id": "q1",
                            "type": ["recall"],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    media = tmp_path / "media"
    media.mkdir()
    video = media / "room.mp4"
    video.write_bytes(b"first video")
    spec = TASKS["m3-bench-robot"]
    first = load_task(
        spec,
        root=tmp_path,
        dataset_path=dataset,
        media_root=media,
        verify_digest=False,
    )

    video.write_bytes(b"second video")
    second = load_task(
        spec,
        root=tmp_path,
        dataset_path=dataset,
        media_root=media,
        verify_digest=False,
    )

    assert first.evaluation_sha256 != second.evaluation_sha256


@pytest.mark.parametrize("stem", ("20240423_195526_001", "20240423_195526(0)"))
def test_atm_raw_media_filename_supplies_capture_time(tmp_path: Path, stem: str) -> None:
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "question-1",
                    "question": "What was in the photo?",
                    "answer": "A clock.",
                    "qtype": "open_end",
                    "evidence_ids": ["20240423_195526"],
                }
            ]
        ),
        encoding="utf-8",
    )
    emails = tmp_path / "atm-bench/data/raw_memory/email/emails.json"
    emails.parent.mkdir(parents=True)
    emails.write_text(
        json.dumps(
            [
                {
                    "id": "email202404230001",
                    "timestamp": "2024-04-23 10:00:00",
                    "short_summary": "Unrelated email",
                    "detail": "",
                }
            ]
        ),
        encoding="utf-8",
    )
    media = tmp_path / "media"
    media.mkdir()
    (media / f"{stem}.jpg").write_bytes(b"image")

    task = load_task(
        TASKS["atm-bench-main"],
        root=tmp_path,
        dataset_path=dataset,
        media_root=media,
        verify_digest=False,
    )

    assert {
        spec.adapter_version for name, spec in TASKS.items() if name.startswith("atm-bench-")
    } == {ATM_BENCH_ADAPTER_VERSION}
    assert ATM_BENCH_ADAPTER_VERSION == "atm_bench_official_v3"
    assert _cache_task(task) == (
        f"atm-bench-main:{ATM_BENCH_ADAPTER_VERSION}:{task.evaluation_sha256}"
    )
    email = next(item for item in task.units[0].memories if item.source_id.startswith("email"))
    assert email.occurred_at == datetime(2024, 4, 23, 10, tzinfo=timezone.utc)
    captured = next(item for item in task.units[0].memories if item.source_id == stem)
    assert captured.occurred_at == datetime(2024, 4, 23, 19, 55, 26, tzinfo=timezone.utc)


def test_response_cache_merges_run_shards_into_the_shared_cache(tmp_path: Path) -> None:
    answer = CachedAnswer(
        "A",
        0.75,
        ("memory-1",),
        (EvidenceInterval("memory-1", "clip-1", 300.0, 420.0),),
        True,
        "insufficient_evidence",
    )
    first = ResponseCache(tmp_path / "responses", "run-a", "namespace")
    first.put("task", "unit", "question", answer)
    assert first.get("task", "unit", "question") == answer

    second = ResponseCache(tmp_path / "responses", "run-b", "namespace")
    assert second.get("task", "unit", "question") == answer
    second.close()
    first.close()

    assert (tmp_path / "responses" / "cache.db").is_file()
    assert (tmp_path / "responses" / "runs" / "run-a" / "cache.db").is_file()
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "responses" / "cache.db").stat().st_mode) == 0o600


def test_lifelong_preparation_builds_one_ordered_segment_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("first.mp4", "second.mp4"):
        (media / name).write_bytes(b"video")

    def segments(
        source: Path, cache: Path, _announce: object
    ) -> tuple[tuple[float, float, Path], ...]:
        return ((0.0, 12.0, cache / source.name),)

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._segments", segments)
    spec = TaskSpec(
        "mm-lifelong-fixture",
        "MM-Lifelong",
        "fixture.json",
        "v1",
        "owner/repo",
        "0" * 40,
        variant="day_test",
    )

    manifest = _lifelong_manifest(spec, media, tmp_path / "cache", None)
    parts = manifest["day_test"]
    assert [part["start_seconds"] for part in parts] == [0.0, 12.0]
    assert [part["end_seconds"] for part in parts] == [12.0, 24.0]


def test_single_boundary_segmentation_avoids_the_segment_muxer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._ffmpeg_id", lambda: "ffmpeg")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._executable", lambda _name: "ffmpeg")
    commands: list[tuple[str, ...]] = []
    expected_segments = [1]

    def run(command: tuple[str, ...] | list[str], _source: Path) -> None:
        commands.append(tuple(command))
        output = Path(command[-1])
        for index in range(expected_segments[0]):
            Path(str(output).replace("%05d", f"{index:05d}")).write_bytes(b"segment")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._run_ffmpeg", run)

    # Asked for one whole clip, ffmpeg must write one named output. Handing this to the segment
    # muxer without `-segment_times` makes it split at every keyframe instead, which turned one
    # 30 s EgoLife clip into 13 + 17 frames and failed the write.
    prepared = _segment_video(source, (30.0,), tmp_path / "cache", None)

    assert len(prepared) == 1
    assert prepared[0][:2] == (0.0, 30.0)
    assert "segment" not in commands[0]
    assert "-segment_times" not in commands[0]
    assert commands[0][-1].endswith("segment-00000.mp4")

    # With real cuts the segment muxer is still the right tool, and it is told where to cut.
    expected_segments[0] = 2
    _segment_video(source, (10.0, 20.0), tmp_path / "cache", None)

    assert commands[1][commands[1].index("-f") + 1] == "segment"
    assert commands[1][commands[1].index("-segment_times") + 1] == "10.000"
    assert commands[1][-1].endswith("segment-%05d.mp4")


def test_video_segment_cache_repairs_an_interrupted_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._ffmpeg_id", lambda: "ffmpeg")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._executable", lambda _name: "ffmpeg")

    def run(command: tuple[str, ...] | list[str], _source: Path) -> None:
        assert "0:a:0?" in command
        assert command[command.index("-c:a") + 1] == "aac"
        assert command[command.index("-tune") + 1] == "zerolatency"
        output = Path(command[-1])
        for index in range(2):
            Path(str(output).replace("%05d", f"{index:05d}")).write_bytes(b"segment")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._run_ffmpeg", run)
    prepared = _segment_video(source, (1.0, 2.0), tmp_path / "cache", None)
    prepared[1][2].write_bytes(b"")

    repaired = _segment_video(source, (1.0, 2.0), tmp_path / "cache", None)

    assert [part[2].read_bytes() for part in repaired] == [b"segment", b"segment"]


def test_supermemory_download_keeps_the_whole_causal_subject_history(tmp_path: Path) -> None:
    def question(question_id: int, video_id: str, started: int, ended: int) -> dict[str, object]:
        return {
            "question_id": question_id,
            "question": "What happened?",
            "choices": ["This question can not be answered.", "A", "B", "C"],
            "correct_answer": "A",
            "correct_option_index": 1,
            "choice_types": ["incorrect", "correct", "incorrect", "vague"],
            "subject": 1,
            "metadata": {"skill": "memory", "primary_video_id": video_id},
            "video_ids": [video_id],
            "question_evidence": {
                "time_spans": [
                    {
                        "start_time": 0,
                        "end_time": ended,
                        "video_id": video_id,
                        "video_start_time_unix": started,
                    }
                ]
            },
            "is_answerable": True,
        }

    dataset = tmp_path / "all_qa.json"
    dataset.write_text(
        json.dumps(
            [
                question(1, "Person_1_session_later", 1_000, 10),
                question(2, "Person_1_session_earlier", 900, 20),
            ]
        ),
        encoding="utf-8",
    )

    assert _selected_patterns(TASKS["supermemory-vqa"], dataset, 1, 0) == (
        "data/video/Person_1/Person_1_session_earlier.mp4",
        "data/video/Person_1/Person_1_session_later.mp4",
    )


def _openeqa_episode_frames(episode: Path, *names: str) -> Path:
    episode.mkdir(parents=True)
    for name in names:
        (episode / f"{name}-rgb.png").write_bytes(b"frame")
    return episode


def test_openeqa_encodes_episode_frames_at_one_frame_per_second(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    episode = _openeqa_episode_frames(
        tmp_path / "frames" / "hm3d-v0" / "000-hm3d-BFRyYbPCCPE", "00010", "00001", "00002"
    )
    (episode / "00001-depth.png").write_bytes(b"depth")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._ffmpeg_id", lambda: "ffmpeg")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._executable", lambda _name: "ffmpeg")
    commands: list[tuple[str, ...]] = []
    listings: list[str] = []

    def run(command: tuple[str, ...] | list[str], _source: Path) -> None:
        commands.append(tuple(command))
        # Read the concat list while it still exists: it lives in a working
        # directory the encoder removes once the output is in place.
        listings.append(Path(command[command.index("-i") + 1]).read_text(encoding="utf-8"))
        Path(command[-1]).write_bytes(b"episode")

    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._run_ffmpeg", run)

    video = _openeqa_video(episode, tmp_path / "cache", None)

    assert video.read_bytes() == b"episode"
    command = commands[0]
    # Upstream publishes no evaluation encoding; 1 fps is the rate at which the
    # segmenter's own `fps=1` filter becomes an identity step.
    assert _OPENEQA_FRAME_RATE == 1
    assert command[command.index("-r") + 1] == "1"
    # Frame extractions are not guaranteed to have even dimensions, which
    # `yuv420p` requires.
    assert command[command.index("-vf") + 1] == "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    assert command[command.index("-f") + 1] == "concat"

    listing = listings[0]
    entries = [line for line in listing.splitlines() if line.startswith("file ")]
    # The official frame order is `sorted(glob("*-rgb.png"))`, depth maps
    # excluded, and the final entry is not repeated: with `-r` forcing a
    # constant rate its `duration` is honoured, and repeating it measurably
    # encodes 76 frames from 75 extracted.
    assert [Path(entry[6:-1]).name for entry in entries] == [
        "00001-rgb.png",
        "00002-rgb.png",
        "00010-rgb.png",
    ]
    assert listing.count("duration 1.000000") == 3

    # A second call reuses the encode rather than paying for it again.
    assert _openeqa_video(episode, tmp_path / "cache", None) == video
    assert len(commands) == 1

    # Adding a frame changes the cache key, because the extraction changed.
    (episode / "00011-rgb.png").write_bytes(b"frame")
    assert _openeqa_video(episode, tmp_path / "cache", None) != video
    assert len(commands) == 2


def test_openeqa_media_root_accepts_either_frames_layout(tmp_path: Path) -> None:
    split_root = tmp_path / "frames" / "hm3d-v0"
    _openeqa_episode_frames(split_root / "000-hm3d-BFRyYbPCCPE", "00001")

    # `--media-root .../frames/hm3d-v0`, which the catalog's own media path is.
    assert _openeqa_episode(split_root, "hm3d-v0", "000-hm3d-BFRyYbPCCPE").is_dir()
    # `--media-root .../frames`, the layout upstream's own README documents.
    assert _openeqa_episode(tmp_path / "frames", "hm3d-v0", "000-hm3d-BFRyYbPCCPE").is_dir()

    with pytest.raises(FileNotFoundError, match="episode history does not exist"):
        _openeqa_episode(split_root, "hm3d-v0", "001-hm3d-TPhiubUHKcP")


def test_openeqa_refuses_frame_paths_it_cannot_quote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    episode = _openeqa_episode_frames(tmp_path / "it's-a-scene", "00001")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._ffmpeg_id", lambda: "ffmpeg")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._executable", lambda _name: "ffmpeg")

    # An operator-supplied media root reaches ffmpeg's concat list verbatim, so
    # a quote there would silently truncate the frame list.
    with pytest.raises(ValueError, match="not safe to encode"):
        _openeqa_video(episode, tmp_path / "cache", None)


def test_openeqa_selected_patterns_name_the_frames_an_operator_must_supply(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "open-eqa-v0.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "episode_history": "hm3d-v0/000-hm3d-BFRyYbPCCPE",
                    "category": "object recognition",
                    "question": "What is above the TV?",
                    "answer": "Air conditioning unit",
                },
                {
                    "question_id": "q2",
                    "episode_history": "hm3d-v0/000-hm3d-BFRyYbPCCPE",
                    "category": "object localization",
                    "question": "Where is the mirror?",
                    "answer": "Next to the staircase",
                },
                {
                    "question_id": "q3",
                    "episode_history": "hm3d-v0/001-hm3d-TPhiubUHKcP",
                    "category": "spatial understanding",
                    "question": "Is the door open?",
                    "answer": "open",
                },
                {
                    "question_id": "q4",
                    "episode_history": "scannet-v0/002-scannet-scene0709_00",
                    "category": "world knowledge",
                    "question": "What room is this?",
                    "answer": "an office",
                },
            ]
        ),
        encoding="utf-8",
    )

    # Deduplicated per episode, narrowed to the task's split, and bounded by
    # the same episode-counting limit the loader uses.
    assert _selected_patterns(TASKS["openeqa-hm3d"], dataset, 1, 0) == (
        "data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/*-rgb.png",
    )
    assert _selected_patterns(TASKS["openeqa-scannet"], dataset, None, 0) == (
        "data/frames/scannet-v0/002-scannet-scene0709_00/*-rgb.png",
    )

    # Neither split's frames are downloadable from here, so the failure names
    # the acquirer an operator has to satisfy instead of trying a fetch.
    with pytest.raises(ValueError, match="requires its open-eqa-hm3d-frames acquirer"):
        acquire_media(TASKS["openeqa-hm3d"], tmp_path, patterns=("data/frames/hm3d-v0/x/*",))
    with pytest.raises(ValueError, match="requires its scannet acquirer"):
        acquire_media(TASKS["openeqa-scannet"], tmp_path, patterns=("data/frames/scannet-v0/x/*",))


@pytest.mark.parametrize(
    ("task_name", "split", "episodes"),
    (
        ("openeqa-hm3d", "hm3d-v0", ("000-hm3d-BFRyYbPCCPE", "001-hm3d-TPhiubUHKcP")),
        (
            "openeqa-scannet",
            "scannet-v0",
            ("002-scannet-scene0709_00", "142-scannet-scene0653_01"),
        ),
    ),
)
def test_openeqa_accepts_episode_histories_already_in_the_managed_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_name: str,
    split: str,
    episodes: tuple[str, str],
) -> None:
    spec = TASKS[task_name]
    dataset = tmp_path / "open-eqa-v0.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": f"q{index}",
                    "episode_history": f"{split}/{episode}",
                    "category": "object recognition",
                    "question": "What is above the TV?",
                    "answer": "Air conditioning unit",
                }
                for index, episode in enumerate(episodes)
            ]
        ),
        encoding="utf-8",
    )
    frames = tmp_path / "openeqa" / "data" / "frames" / split

    def prepare() -> object:
        return prepare_task_media(
            spec,
            root=tmp_path,
            dataset_path=dataset,
            media_root=None,
            manifest=None,
            limit=None,
            offset=0,
            download=False,
        )

    # Nothing extracted yet. Going through `prepare_task_media` rather than the
    # acquirer directly is the point: an earlier version reached
    # `acquire_media`, which refuses any acquirer-declared source outright, so a
    # correctly extracted tree was rejected too. `FileNotFoundError` rather than
    # that `ValueError` is what proves the dispatch is wired up.
    with pytest.raises(FileNotFoundError, match="acquirer supplies") as absent:
        prepare()
    assert f"--media-root {task_name}=DIR" in str(absent.value)
    assert "2 of 2" in str(absent.value)

    _openeqa_episode_frames(frames / episodes[0], "00001", "00002")

    # A partial extraction is still a failure, and says how much is absent: half
    # a scene set silently scored as a full run is the outcome worth refusing.
    with pytest.raises(FileNotFoundError, match="1 of 2"):
        prepare()

    _openeqa_episode_frames(frames / episodes[1], "00001", "00002")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._ffmpeg_id", lambda: "ffmpeg")
    monkeypatch.setattr("mindbridge.benchmarks.prepare_media._executable", lambda _name: "ffmpeg")
    monkeypatch.setattr(
        "mindbridge.benchmarks.prepare_media._run_ffmpeg",
        lambda command, _source: Path(command[-1]).write_bytes(b"episode"),
    )
    monkeypatch.setattr(
        "mindbridge.benchmarks.prepare_media._segments",
        lambda source, cache, announce: ((0.0, 2.0, source),),
    )

    # Frames extracted into the catalog's own media root are enough on their
    # own, so `--media-root` is a convenience rather than the only way to run.
    manifest = prepare()
    assert isinstance(manifest, dict)
    units = cast(dict[str, object], manifest["units"])
    assert list(units) == list(episodes)


class _CountingDescriber:
    """One describer that reports how many visuals actually reached a model."""

    vision_capabilities = frozenset({Modality.IMAGE})
    vision_model = "caption-model"

    def __init__(self, prefix: str = "caption") -> None:
        self.described: list[str] = []
        self.calls = 0
        self._prefix = prefix

    def describe(self, inputs: Sequence[ModelInput]) -> tuple[str, ...]:
        batch = tuple(inputs)
        self.calls += 1
        digests = tuple(str(value.assets[0].sha256) for value in batch)
        self.described.extend(digests)
        return tuple(f"{self._prefix} for {digest[:4]}" for digest in digests)

    def close(self) -> None:
        return None


def _description_input(tmp_path: Path, digest: str) -> ModelInput:
    path = tmp_path / f"{digest[:8]}.png"
    path.write_bytes(b"png-bytes")
    return ModelInput(
        assets=(
            AssetRef(
                id=digest[:8],
                modality=Modality.IMAGE,
                media_type="image/png",
                size_bytes=path.stat().st_size,
                sha256=digest,
                name=path.name,
                path=path,
            ),
        )
    )


def test_a_cached_description_is_reused_instead_of_described_again(tmp_path: Path) -> None:
    """Two ingests of one corpus must build the same documents.

    The measured generation endpoint returns a different caption for the same image on every
    call even at temperature 0 with a fixed seed, and a caption becomes indexed text, so without
    this an arm cannot be compared with itself across runs.
    """
    first_input = _description_input(tmp_path, "a" * 64)
    second_input = _description_input(tmp_path, "b" * 64)
    backend = _CountingDescriber()
    cache = DescriptionCache(tmp_path / "descriptions.db", backend.vision_model)
    try:
        describer = eval_module._CachedVisionDescriber(cast(Any, backend), cache)

        first = describer.describe((first_input,))
        # A partially cached batch still costs exactly one request, for the misses only.
        second = describer.describe((first_input, second_input))
        third = describer.describe((first_input, second_input))
    finally:
        cache.close()

    assert first == ("caption for aaaa",)
    assert second == ("caption for aaaa", "caption for bbbb")
    assert third == second
    assert backend.calls == 2
    assert backend.described == ["a" * 64, "b" * 64]
    assert (cache.hits, cache.misses) == (3, 2)


def test_a_description_cache_survives_the_process_that_wrote_it(tmp_path: Path) -> None:
    value = _description_input(tmp_path, "c" * 64)
    first_backend = _CountingDescriber()
    cache = DescriptionCache(tmp_path / "descriptions.db", first_backend.vision_model)
    try:
        eval_module._CachedVisionDescriber(cast(Any, first_backend), cache).describe((value,))
    finally:
        cache.close()

    second_backend = _CountingDescriber(prefix="different")
    reopened = DescriptionCache(tmp_path / "descriptions.db", second_backend.vision_model)
    try:
        repeated = eval_module._CachedVisionDescriber(cast(Any, second_backend), reopened).describe(
            (value,)
        )
    finally:
        reopened.close()

    # Same corpus, later run, zero description tokens, and the identical document.
    assert repeated == ("caption for cccc",)
    assert second_backend.calls == 0


def test_two_describer_models_do_not_share_one_cached_caption(tmp_path: Path) -> None:
    # Both sides go through the wrapper, so the test cannot pass by keying differently from the
    # code under test -- an earlier version of this test wrote its own raw key and never
    # exercised the model separation it claimed to.
    value = _description_input(tmp_path, "d" * 64)
    first_backend = _CountingDescriber(prefix="first-model")
    first = DescriptionCache(tmp_path / "descriptions.db", "caption-model")
    try:
        original = eval_module._CachedVisionDescriber(cast(Any, first_backend), first).describe(
            (value,)
        )
    finally:
        first.close()

    second_backend = _CountingDescriber(prefix="second-model")
    second = DescriptionCache(tmp_path / "descriptions.db", "other-caption-model")
    try:
        described = eval_module._CachedVisionDescriber(cast(Any, second_backend), second).describe(
            (value,)
        )
    finally:
        second.close()

    assert original == ("first-model for dddd",)
    assert described == ("second-model for dddd",)
    assert second_backend.calls == 1


def test_the_description_cache_is_opened_only_for_a_configured_vision_slot(
    tmp_path: Path,
) -> None:
    arguments = cast(
        Any, SimpleNamespace(data_root=tmp_path, use_cache=tmp_path / "cache" / "run.db")
    )
    embedding = {"provider": "openai", "model": "tiny-test", "dimension": 4}
    without = MindBridgeConfig.model_validate({"data_dir": tmp_path, "embedding": embedding})
    described = MindBridgeConfig.model_validate(
        {"data_dir": tmp_path, "embedding": embedding, "vision": {"provider": "openai"}}
    )

    assert eval_module._description_cache_path(arguments, None) is None
    assert eval_module._description_cache_path(arguments, without) is None
    # Beside the response cache, and deliberately not namespaced by run: a later run has to read
    # what an earlier one wrote.
    assert eval_module._description_cache_path(arguments, described) == (
        tmp_path / "cache" / "descriptions.db"
    )
    no_response_cache = cast(Any, SimpleNamespace(data_root=tmp_path, use_cache=None))
    assert eval_module._description_cache_path(no_response_cache, described) == (
        tmp_path / "cache" / "descriptions.db"
    )
