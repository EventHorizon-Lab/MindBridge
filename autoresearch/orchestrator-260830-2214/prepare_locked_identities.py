"""Freeze the offline quality and performance-suite input identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from mindbridge.benchmarks.eval_adapters import LoadedTask, load_task
from mindbridge.benchmarks.m3_bench import load_m3_bench
from mindbridge.benchmarks.prepare_media import prepare_task_media
from mindbridge.benchmarks.task_catalog import TASKS, TaskSpec

E692_ROOT = Path("/home/yons/.codex/worktrees/e692/MindBridge/.benchmarks")
A6B7_ROOT = Path("/home/yons/.codex/worktrees/a6b7/MindBridge/.benchmarks")
M3_LOCK_PATH = Path("/dev/shm/mindbridge-ar-locked-b2d7f2918105/m3-locked.json")
PERFORMANCE_ROOT = Path(__file__).resolve().parent / "locked-inputs" / "performance"
M3_OFFSETS = (5, 14, 20, 45, 67, 73, 85)
M3_UNIT_IDS = (
    "bedroom_06",
    "gym_03",
    "kitchen_05",
    "living_room_07",
    "meeting_room_05",
    "office_05",
    "study_09",
)
M3_QUESTION_COUNT = 87
M3_SOURCE_SHA256 = "f43031bf0216a2ef2e7909f20ecd534e0098da17b63cdf94325a07f1bea372c1"
M3_PAYLOAD_SHA256 = "2183e3bf94163e2af72bedde8a360f66517d14a13dd2e7364e07ece4b4b45447"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object with string keys")
    return cast(dict[str, Any], value)


def _locked_payload(source: Mapping[str, object], unit_ids: Sequence[str]) -> bytes:
    try:
        selected = {unit_id: source[unit_id] for unit_id in unit_ids}
    except KeyError as error:
        raise ValueError(f"M3 source is missing locked unit {error.args[0]}") from None
    return (json.dumps(selected, ensure_ascii=False, indent=2) + "\n").encode()


def _write_locked_m3(source: Path, output: Path) -> dict[str, object]:
    source_bytes = source.read_bytes()
    if _sha256(source_bytes) != M3_SOURCE_SHA256:
        raise ValueError(f"{source}: official M3 source SHA-256 changed")
    videos = load_m3_bench(source)
    if len(videos) != 100:
        raise ValueError(f"expected 100 M3 Robot units, found {len(videos)}")
    selected = tuple(videos[offset] for offset in M3_OFFSETS)
    unit_ids = tuple(video.video_id for video in selected)
    if unit_ids != M3_UNIT_IDS:
        raise ValueError(f"sorted M3 locked unit IDs changed: {unit_ids!r}")
    question_count = sum(len(video.questions) for video in selected)
    if question_count != M3_QUESTION_COUNT:
        raise ValueError(f"expected {M3_QUESTION_COUNT} M3 questions, found {question_count}")

    payload = _locked_payload(_json_object(source), unit_ids)
    if _sha256(payload) != M3_PAYLOAD_SHA256:
        raise ValueError("sorted M3 locked payload SHA-256 changed")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != payload:
            raise ValueError(f"refusing to replace different M3 lock: {output}")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, output)
    return {
        "offsets": M3_OFFSETS,
        "path": str(output.resolve()),
        "payload_sha256": M3_PAYLOAD_SHA256,
        "question_count": M3_QUESTION_COUNT,
        "source_sha256": M3_SOURCE_SHA256,
        "unit_ids": M3_UNIT_IDS,
    }


def _manifest(
    spec: TaskSpec,
    *,
    root: Path,
    dataset: Path,
    media_root: Path,
    limit: int | None,
    offset: int,
) -> Mapping[str, object]:
    prepared = prepare_task_media(
        spec,
        root=root,
        dataset_path=dataset,
        media_root=media_root,
        manifest=None,
        limit=limit,
        offset=offset,
        download=False,
    )
    if prepared is None:
        raise ValueError(f"{spec.name}: expected a prepared media manifest")
    return {"version": 1, "tasks": {spec.name: prepared}}


def _identity(task: LoadedTask, question_count: int) -> dict[str, object]:
    sample_ids = tuple(
        f"{task.spec.name}/{unit.unit_id}/{question.question_id}"
        for unit in task.units
        for question in unit.questions
    )
    if len(sample_ids) != question_count:
        raise ValueError(
            f"{task.spec.name}: expected {question_count} questions, found {len(sample_ids)}"
        )
    return {
        "dataset_sha256": task.dataset_sha256,
        "evaluation_sha256": task.evaluation_sha256,
        "input_sha256": dict(task.input_sha256),
        "ordered_sample_ids_sha256": _sha256("\n".join(sample_ids).encode()),
        "question_count": len(sample_ids),
    }


def _load_identities(
    e692: Path,
    a6b7: Path,
    m3_lock: Path,
    performance_root: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    m3_root = e692 / "m3-catalog"
    m3_spec = TASKS["m3-bench-robot"]
    m3_media = m3_spec.media_root(m3_root)
    if m3_media is None:
        raise AssertionError("M3 Robot catalog is missing its media root")
    m3_quality_manifest = _manifest(
        m3_spec,
        root=m3_root,
        dataset=m3_lock,
        media_root=m3_media,
        limit=None,
        offset=0,
    )
    m3_performance_dataset = performance_root / "m3-robot.json"
    m3_performance_manifest = _manifest(
        m3_spec,
        root=m3_root,
        dataset=m3_performance_dataset,
        media_root=m3_media,
        limit=None,
        offset=0,
    )

    ego_spec = TASKS["egolifeqa"]
    ego_dataset = ego_spec.dataset_path(a6b7)
    ego_media = e692 / "egolife"
    ego_quality_manifest = _manifest(
        ego_spec,
        root=a6b7,
        dataset=ego_dataset,
        media_root=ego_media,
        limit=50,
        offset=50,
    )
    ego_performance_manifest = _manifest(
        ego_spec,
        root=a6b7,
        dataset=ego_dataset,
        media_root=ego_media,
        limit=10,
        offset=50,
    )

    atm_hard = load_task(TASKS["atm-bench-hard"], root=a6b7)
    quality = {
        "locomo-locked": load_task(TASKS["locomo-refined"], root=e692, limit=3, offset=3),
        "atm-hard": atm_hard,
        "gallery-locked": load_task(TASKS["mem-gallery"], root=a6b7, limit=5, offset=5),
        "m3-locked": load_task(
            m3_spec,
            root=m3_root,
            dataset_path=m3_lock,
            media_root=m3_media,
            media_manifest=m3_quality_manifest,
            verify_digest=False,
        ),
        "egolife-q50-99": load_task(
            ego_spec,
            root=a6b7,
            media_root=ego_media,
            media_manifest=ego_quality_manifest,
            limit=50,
            offset=50,
        ),
    }
    performance = {
        "locomo": load_task(
            TASKS["locomo-refined"],
            root=e692,
            dataset_path=performance_root / "locomo-refined.json",
            verify_digest=False,
        ),
        "atm-hard": atm_hard,
        "gallery": load_task(
            TASKS["mem-gallery"],
            root=a6b7,
            dataset_path=performance_root / "mem-gallery",
            media_root=TASKS["mem-gallery"].media_root(a6b7),
            verify_digest=False,
        ),
        "m3": load_task(
            m3_spec,
            root=m3_root,
            dataset_path=m3_performance_dataset,
            media_root=m3_media,
            media_manifest=m3_performance_manifest,
            verify_digest=False,
        ),
        "egolife": load_task(
            ego_spec,
            root=a6b7,
            media_root=ego_media,
            media_manifest=ego_performance_manifest,
            limit=10,
            offset=50,
        ),
    }
    quality_counts = {
        "locomo-locked": 455,
        "atm-hard": 31,
        "gallery-locked": 407,
        "m3-locked": 87,
        "egolife-q50-99": 50,
    }
    performance_counts = {
        "locomo": 20,
        "atm-hard": 31,
        "gallery": 20,
        "m3": 14,
        "egolife": 10,
    }
    return (
        {name: _identity(task, quality_counts[name]) for name, task in quality.items()},
        {name: _identity(task, performance_counts[name]) for name, task in performance.items()},
    )


def _self_check() -> None:
    assert _sha256(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert _locked_payload({"b": 2, "a": 1}, ("a", "b")) == b'{\n  "a": 1,\n  "b": 2\n}\n'
    try:
        _locked_payload({}, ("missing",))
    except ValueError as error:
        assert "missing locked unit" in str(error)
    else:
        raise AssertionError("missing M3 unit was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e692-root", type=Path, default=E692_ROOT)
    parser.add_argument("--a6b7-root", type=Path, default=A6B7_ROOT)
    parser.add_argument("--m3-lock", type=Path, default=M3_LOCK_PATH)
    parser.add_argument("--performance-root", type=Path, default=PERFORMANCE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
        print(0)
        return 0

    e692 = arguments.e692_root.expanduser().resolve()
    a6b7 = arguments.a6b7_root.expanduser().resolve()
    m3_source = TASKS["m3-bench-robot"].dataset_path(e692 / "m3-catalog")
    m3_lock = arguments.m3_lock.expanduser().resolve()
    performance_root = arguments.performance_root.expanduser().resolve()
    m3_lock_identity = _write_locked_m3(m3_source, m3_lock)
    quality_identities, speed_identities = _load_identities(e692, a6b7, m3_lock, performance_root)
    result = {
        "m3_lock": m3_lock_identity,
        "quality_identities": quality_identities,
        "speed_identities": speed_identities,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if output.read_text(encoding="utf-8") != payload:
                raise ValueError(f"refusing to replace different locked identities: {output}")
        else:
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, output)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
