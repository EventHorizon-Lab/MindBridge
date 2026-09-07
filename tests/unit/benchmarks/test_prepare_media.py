"""Regression checks for benchmark media acquisition failures."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import mindbridge.benchmarks.prepare_media as prepare_media
from mindbridge.benchmarks.eval_adapters import load_task
from mindbridge.benchmarks.task_catalog import TASKS


def _m3_annotations(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "private": {
                    "video_url": "https://www.youtube.com/watch?v=private",
                    "video_path": "data/videos/web/private.mp4",
                    "qa_list": [
                        {
                            "question": "Private question?",
                            "answer": "Private answer.",
                            "question_id": "private_Q1",
                            "type": ["recall"],
                            "timestamp": "00:10",
                        }
                    ],
                },
                "public": {
                    "video_url": "https://www.youtube.com/watch?v=public",
                    "video_path": "data/videos/web/public.mp4",
                    "qa_list": [
                        {
                            "question": "Public question?",
                            "answer": "Public answer.",
                            "question_id": "public_Q1",
                            "type": ["recall"],
                            "timestamp": "00:10",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def test_youtube_acquisition_records_permanently_unavailable_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "web.json"
    _m3_annotations(dataset)

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        target = Path(command[command.index("--output") + 1])
        if command[-1].endswith("private"):
            return subprocess.CompletedProcess(command, 1, "", "ERROR: Private video\n")
        target.write_bytes(b"video")
        return subprocess.CompletedProcess(command, 0, "downloaded", "")

    monkeypatch.setattr(prepare_media, "_yt_dlp_command", lambda: ("yt-dlp",))
    monkeypatch.setattr(prepare_media, "_youtube_sleep", lambda: 0.0)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(prepare_media, "_has_audio", lambda _path: True)

    unavailable = prepare_media._acquire_youtube(
        TASKS["m3-bench-web"],
        dataset,
        tmp_path,
        ("videos/web/private.mp4", "videos/web/public.mp4"),
        True,
        None,
    )

    assert unavailable == {"private": "ERROR: Private video"}
    assert not (tmp_path / "m3-bench/videos/web/private.mp4").exists()
    assert (tmp_path / "m3-bench/videos/web/public.mp4").read_bytes() == b"video"


def test_youtube_acquisition_does_not_hide_transient_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "web.json"
    _m3_annotations(dataset)
    monkeypatch.setattr(prepare_media, "_yt_dlp_command", lambda: ("yt-dlp",))
    monkeypatch.setattr(prepare_media, "_youtube_sleep", lambda: 0.0)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ("yt-dlp",), 1, "", "ERROR: connection timed out\n"
        ),
    )

    with pytest.raises(RuntimeError, match="connection timed out"):
        prepare_media._acquire_youtube(
            TASKS["m3-bench-web"],
            dataset,
            tmp_path,
            ("videos/web/private.mp4",),
            True,
            None,
        )


def test_m3_manifest_excludes_and_reports_unavailable_units(tmp_path: Path) -> None:
    dataset = tmp_path / "web.json"
    _m3_annotations(dataset)
    task = load_task(
        TASKS["m3-bench-web"],
        root=tmp_path,
        dataset_path=dataset,
        media_manifest={
            "tasks": {
                "m3-bench-web": {
                    "units": {
                        "public": [
                            {
                                "text": "prepared segment",
                                "start_seconds": 0,
                                "end_seconds": 10,
                            }
                        ]
                    },
                    "unavailable_units": {"private": "ERROR: Private video"},
                }
            }
        },
        verify_digest=False,
    )

    assert tuple(unit.unit_id for unit in task.units) == ("public",)
    assert task.unavailable_units == {"private": "ERROR: Private video"}


def test_m3_manifest_ignores_unavailable_units_outside_selected_slice(tmp_path: Path) -> None:
    dataset = tmp_path / "web.json"
    _m3_annotations(dataset)
    task = load_task(
        TASKS["m3-bench-web"],
        root=tmp_path,
        dataset_path=dataset,
        media_manifest={
            "tasks": {
                "m3-bench-web": {
                    "units": {
                        "public": [
                            {
                                "text": "prepared segment",
                                "start_seconds": 0,
                                "end_seconds": 10,
                            }
                        ]
                    },
                    "unavailable_units": {"private": "ERROR: Private video"},
                }
            }
        },
        offset=1,
        verify_digest=False,
    )

    assert tuple(unit.unit_id for unit in task.units) == ("public",)
    assert task.unavailable_units == {}
