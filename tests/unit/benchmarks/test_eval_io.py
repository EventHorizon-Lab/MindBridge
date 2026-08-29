"""Small regression checks for resumable benchmark input and response caches."""

from __future__ import annotations

import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

from mindbridge.benchmarks.download import acquire_media
from mindbridge.benchmarks.eval_adapters import MediaResolver, load_task
from mindbridge.benchmarks.eval_cache import CachedAnswer, EvidenceInterval, ResponseCache
from mindbridge.benchmarks.prepare_media import (
    _lifelong_manifest,
    _m3_manifest,
    _segment_video,
    _selected_patterns,
)
from mindbridge.benchmarks.task_catalog import TASKS, MediaSource, TaskSpec


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


def test_response_cache_merges_run_shards_into_the_shared_cache(tmp_path: Path) -> None:
    answer = CachedAnswer(
        "A",
        0.75,
        ("memory-1",),
        (EvidenceInterval("memory-1", "clip-1", 300.0, 420.0),),
    )
    first = ResponseCache(tmp_path / "responses", "run-a", "namespace")
    first.put("task", "unit", "question", answer)
    assert first.get("task", "unit", "question") == answer
    first.close()

    second = ResponseCache(tmp_path / "responses", "run-b", "namespace")
    assert second.get("task", "unit", "question") == answer
    second.close()

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
