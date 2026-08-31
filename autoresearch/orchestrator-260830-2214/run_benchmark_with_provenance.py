"""Run one frozen benchmark and atomically bind its provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn, TypeVar

_EVAL_PREFIX = ("uv", "run", "--frozen", "python", "-m", "mindbridge.benchmarks.eval")
_CLEAN_PATHS = ("src/mindbridge", "pyproject.toml", "uv.lock")
_REVISION = re.compile(r"[0-9a-f]{40,64}")
_SECRET_ARGUMENT = re.compile(
    r"(?:api[-_]?key|authorization|access[-_]?token|password|secret)\s*[:=]",
    re.IGNORECASE,
)
_SECRET_FLAGS = frozenset(
    {
        "--access-token",
        "--api-key",
        "--authorization",
        "--password",
        "--secret",
    }
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class RepositoryState:
    revision: str
    product_tree: str
    pyproject_blob: str
    uv_lock_blob: str


@dataclass(frozen=True, slots=True)
class GpuState:
    index: int
    name: str
    uuid: str
    driver_version: str
    cuda_visible_devices: str | None


@dataclass(frozen=True, slots=True)
class SequenceState:
    identifier: str
    index: int
    total: int
    label: str
    previous_provenance: Path | None
    previous_sha256: str | None


def _run_text(command: list[str], *, cwd: Path, environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _argv_sha256(command: list[str]) -> str:
    payload = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _repository_state(
    worktree: Path,
    expected_revision: str,
    environment: dict[str, str],
) -> RepositoryState:
    top_level = Path(
        _run_text(["git", "rev-parse", "--show-toplevel"], cwd=worktree, environment=environment)
    ).resolve()
    if top_level != worktree:
        raise ValueError("--worktree must be the repository root")
    revision = _run_text(["git", "rev-parse", "HEAD"], cwd=worktree, environment=environment)
    if revision != expected_revision:
        raise ValueError("worktree HEAD does not match --expected-revision")
    dirty = _run_text(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *_CLEAN_PATHS,
        ],
        cwd=worktree,
        environment=environment,
    )
    if dirty:
        raise ValueError("product source or dependency files are not clean")
    return RepositoryState(
        revision=revision,
        product_tree=_run_text(
            ["git", "rev-parse", "HEAD:src/mindbridge"], cwd=worktree, environment=environment
        ),
        pyproject_blob=_run_text(
            ["git", "rev-parse", "HEAD:pyproject.toml"], cwd=worktree, environment=environment
        ),
        uv_lock_blob=_run_text(
            ["git", "rev-parse", "HEAD:uv.lock"], cwd=worktree, environment=environment
        ),
    )


def _parse_gpu_row(value: str, cuda_visible_devices: str | None) -> GpuState:
    rows = [row for row in csv.reader(value.splitlines(), skipinitialspace=True) if row]
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("exactly one NVIDIA GPU must be visible to the benchmark host")
    raw_index, name, uuid, driver = (item.strip() for item in rows[0])
    try:
        index = int(raw_index)
    except ValueError as error:
        raise ValueError("nvidia-smi returned an invalid GPU index") from error
    if not name or not uuid.startswith(("GPU-", "MIG-")) or not driver:
        raise ValueError("nvidia-smi returned incomplete GPU identity")
    if cuda_visible_devices is not None:
        visible = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
        if len(visible) != 1 or visible[0] == "-1":
            raise ValueError("CUDA_VISIBLE_DEVICES must select exactly one GPU")
        selected = visible[0]
        if selected.isdecimal():
            if int(selected) != index:
                raise ValueError("CUDA_VISIBLE_DEVICES does not match the recorded GPU")
        elif not uuid.startswith(selected):
            raise ValueError("CUDA_VISIBLE_DEVICES UUID does not match the recorded GPU")
    return GpuState(index, name, uuid, driver, cuda_visible_devices)


def _gpu_state(worktree: Path, environment: dict[str, str]) -> GpuState:
    value = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=worktree,
        environment=environment,
    )
    return _parse_gpu_row(value, environment.get("CUDA_VISIBLE_DEVICES"))


def _absolute_without_symlinks(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.resolve() != absolute:
        raise ValueError(f"{label} must not traverse a symbolic link")
    return absolute


def _require_empty_or_missing(path: Path, label: str) -> None:
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a directory or not exist")
    if next(path.iterdir(), None) is not None:
        raise ValueError(f"{label} must be absent or explicitly empty")


def _option_value(command: list[str], option: str) -> str:
    values: list[str] = []
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command):
                raise ValueError(f"benchmark command is missing the value for {option}")
            values.append(command[index + 1])
        elif item.startswith(f"{option}="):
            values.append(item.partition("=")[2])
    if len(values) != 1 or not values[0]:
        raise ValueError(f"benchmark command must provide {option} exactly once")
    return values[0]


def _contains_secret_argument(command: list[str]) -> bool:
    return any(
        item.partition("=")[0].lower().replace("_", "-") in _SECRET_FLAGS
        or _SECRET_ARGUMENT.search(item) is not None
        for item in command
    )


def _validate_command(command: list[str], output_dir: Path, data_root: Path) -> str:
    if tuple(command[: len(_EVAL_PREFIX)]) != _EVAL_PREFIX:
        raise ValueError("benchmark command must start with the frozen MindBridge eval prefix")
    if "--overwrite" in command:
        raise ValueError("locked benchmark commands must not use --overwrite")
    if _contains_secret_argument(command):
        raise ValueError("benchmark credentials must be supplied only through the environment")
    command_output = _absolute_without_symlinks(
        Path(_option_value(command, "--output-path")), "command output path"
    )
    command_data = _absolute_without_symlinks(
        Path(_option_value(command, "--data-root")), "command data root"
    )
    if command_output != output_dir or command_data != data_root:
        raise ValueError("benchmark command paths do not match wrapper paths")
    return _option_value(command, "--run-id")


def _clean_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{label} must be non-empty and contain no control characters")
    return normalized


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _sequence_state(
    identifier: str,
    index: int,
    total: int,
    label: str,
    previous: Path | None,
) -> SequenceState:
    if total <= 0 or index < 0 or index >= total:
        raise ValueError("sequence index must be within the declared positive total")
    identifier = _clean_text(identifier, "sequence id")
    label = _clean_text(label, "sequence label")
    if index == 0:
        if previous is not None:
            raise ValueError("the first sequence entry must not have a predecessor")
        return SequenceState(identifier, index, total, label, None, None)
    if previous is None:
        raise ValueError("non-initial sequence entries require --previous-provenance")
    previous = _absolute_without_symlinks(previous, "previous provenance")
    if not previous.is_file() or previous.is_symlink():
        raise ValueError("previous provenance must be a regular file")
    document = _load_object(previous)
    if document.get("schema_version") != 2:
        raise ValueError("previous provenance has an unsupported schema")
    sequence = document.get("sequence")
    if not isinstance(sequence, dict):
        raise ValueError("previous provenance is missing sequence metadata")
    if (
        sequence.get("id") != identifier
        or sequence.get("index") != index - 1
        or sequence.get("total") != total
    ):
        raise ValueError("previous provenance is not the immediate sequence predecessor")
    return SequenceState(identifier, index, total, label, previous, _sha256(previous))


def _same_repository(left: RepositoryState, right: RepositoryState) -> bool:
    return left == right


def _same_gpu(left: GpuState, right: GpuState) -> bool:
    return left == right


def _artifact_hashes(output_dir: Path, data_root: Path, run_id: str) -> tuple[str, str]:
    result_path = output_dir / "results.json"
    samples_path = output_dir / "samples.jsonl"
    for path in (result_path, samples_path):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"benchmark did not produce a regular {path.name}")
    samples_hash = _sha256(samples_path)
    result = _load_object(result_path)
    if result.get("status") != "completed":
        raise ValueError("benchmark result is not completed")
    if result.get("run_id") != run_id:
        raise ValueError("benchmark result run_id does not match the command")
    result_data = result.get("data_root")
    if not isinstance(result_data, str) or Path(result_data).resolve() != data_root:
        raise ValueError("benchmark result data_root does not match the wrapper")
    if result.get("samples_sha256") != samples_hash:
        raise ValueError("benchmark result does not bind the emitted samples")
    return _sha256(result_path), samples_hash


def _gpu_payload(gpu: GpuState) -> dict[str, object]:
    return {
        "cuda_visible_devices": gpu.cuda_visible_devices,
        "driver_version": gpu.driver_version,
        "index": gpu.index,
        "name": gpu.name,
        "uuid": gpu.uuid,
    }


def _sequence_payload(sequence: SequenceState) -> dict[str, object]:
    return {
        "id": sequence.identifier,
        "index": sequence.index,
        "label": sequence.label,
        "previous_provenance": (
            None if sequence.previous_provenance is None else str(sequence.previous_provenance)
        ),
        "previous_provenance_sha256": sequence.previous_sha256,
        "total": sequence.total,
    }


def _create_attempt_ledger(
    output_dir: Path,
    value: dict[str, Any],
) -> tuple[Path, str]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.parent.resolve() != output_dir.parent:
        raise ValueError("output parent must not traverse a symbolic link")
    path = output_dir.parent / f"{output_dir.name}.attempt.json"
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("an attempt ledger already exists for this output directory") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path, hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("refusing to replace existing run provenance")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("refusing to replace an existing provenance temporary file")
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _iso8601_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one frozen MindBridge benchmark with bound provenance."
    )
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sequence-id")
    parser.add_argument("--sequence-index", type=int)
    parser.add_argument("--sequence-total", type=int)
    parser.add_argument("--sequence-label")
    parser.add_argument("--previous-provenance", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _required(value: _T | None, option: str) -> _T:
    if value is None:
        raise ValueError(f"{option} is required")
    return value


def _fail(message: str) -> NoReturn:
    print(f"provenance wrapper: {message}", file=sys.stderr)
    raise SystemExit(2)


def _self_check() -> None:
    output = Path("/tmp/mindbridge-output")
    data = Path("/tmp/mindbridge-data")
    command = [
        *_EVAL_PREFIX,
        "--tasks",
        "fixture",
        "--run-id",
        "run-1",
        "--output-path",
        str(output),
        "--data-root",
        str(data),
    ]
    assert _validate_command(command, output, data) == "run-1"
    assert len(_argv_sha256(command)) == 64
    gpu = _parse_gpu_row("0, NVIDIA GeForce RTX 5090, GPU-fixture, 999.0", "0")
    assert gpu.name == "NVIDIA GeForce RTX 5090" and gpu.uuid == "GPU-fixture"
    sequence = _sequence_state("suite", 0, 2, "baseline", None)
    assert sequence.previous_sha256 is None
    with tempfile.TemporaryDirectory() as directory:
        attempt_output = Path(directory) / "run"
        ledger, ledger_hash = _create_attempt_ledger(
            attempt_output, {"schema_version": 1, "state": "started"}
        )
        assert ledger.name == "run.attempt.json" and _sha256(ledger) == ledger_hash
        assert _load_object(ledger)["state"] == "started"
        try:
            _create_attempt_ledger(attempt_output, {"schema_version": 1})
        except ValueError:
            pass
        else:
            raise AssertionError("an existing attempt ledger was replaced")
    try:
        _validate_command(["python", "eval.py"], output, data)
    except ValueError:
        pass
    else:
        raise AssertionError("an arbitrary command was accepted")
    try:
        _validate_command([*command, "--api-key", "forbidden"], output, data)
    except ValueError:
        pass
    else:
        raise AssertionError("a credential argument was accepted")


def main() -> int:  # noqa: C901 - the wrapper intentionally has one fail-closed transaction
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.self_check:
        _self_check()
        print(0)
        return 0
    try:
        worktree = _absolute_without_symlinks(
            _required(arguments.worktree, "--worktree"), "worktree"
        )
        if not worktree.is_dir():
            raise ValueError("worktree must be an existing directory")
        expected_revision = str(_required(arguments.expected_revision, "--expected-revision"))
        if _REVISION.fullmatch(expected_revision) is None:
            raise ValueError("expected revision must be a full hexadecimal object ID")
        output_dir = _absolute_without_symlinks(
            _required(arguments.output_dir, "--output-dir"), "output directory"
        )
        data_root = _absolute_without_symlinks(
            _required(arguments.data_root, "--data-root"), "data root"
        )
        if (
            output_dir == data_root
            or output_dir.is_relative_to(data_root)
            or data_root.is_relative_to(output_dir)
        ):
            raise ValueError("output directory and data root must not overlap")
        _require_empty_or_missing(output_dir, "output directory")
        _require_empty_or_missing(data_root, "data root")
        command = list(arguments.command)
        if command[:1] == ["--"]:
            command = command[1:]
        run_id = _validate_command(command, output_dir, data_root)
        sequence = _sequence_state(
            str(_required(arguments.sequence_id, "--sequence-id")),
            int(_required(arguments.sequence_index, "--sequence-index")),
            int(_required(arguments.sequence_total, "--sequence-total")),
            str(_required(arguments.sequence_label, "--sequence-label")),
            arguments.previous_provenance,
        )
        environment = os.environ.copy()
        repository_before = _repository_state(worktree, expected_revision, environment)
        gpu_before = _gpu_state(worktree, environment)
        wrapper_path = Path(__file__).resolve()
        wrapper_hash = _sha256(wrapper_path)
        started_at = _iso8601_now()
        attempt: dict[str, Any] = {
            "argv": command,
            "argv_sha256": _argv_sha256(command),
            "data_root": str(data_root),
            "expected_revision": expected_revision,
            "gpu": _gpu_payload(gpu_before),
            "output_dir": str(output_dir),
            "product_tree": repository_before.product_tree,
            "pyproject_blob": repository_before.pyproject_blob,
            "run_id": run_id,
            "schema_version": 1,
            "sequence": _sequence_payload(sequence),
            "state": "started",
            "started_at": started_at,
            "uv_lock_blob": repository_before.uv_lock_blob,
            "worktree": str(worktree),
            "worktree_revision": repository_before.revision,
            "wrapper_sha256": wrapper_hash,
        }
        attempt_path, attempt_hash = _create_attempt_ledger(output_dir, attempt)
        started_ns = time.monotonic_ns()
        completed = subprocess.run(command, cwd=worktree, env=environment, check=False)
        ended_ns = time.monotonic_ns()
        ended_at = _iso8601_now()
        if _sha256(attempt_path) != attempt_hash:
            raise ValueError("attempt ledger changed while the benchmark was running")
        if _sha256(wrapper_path) != wrapper_hash:
            raise ValueError("provenance wrapper changed while the benchmark was running")
        if completed.returncode:
            print(
                f"benchmark command failed with exit code {completed.returncode}; "
                f"attempt retained at {attempt_path}",
                file=sys.stderr,
            )
            return completed.returncode
        repository_after = _repository_state(worktree, expected_revision, environment)
        if not _same_repository(repository_before, repository_after):
            raise ValueError("repository state changed while the benchmark was running")
        gpu_after = _gpu_state(worktree, environment)
        if not _same_gpu(gpu_before, gpu_after):
            raise ValueError("GPU identity changed while the benchmark was running")
        results_hash, samples_hash = _artifact_hashes(output_dir, data_root, run_id)
        duration_seconds = (ended_ns - started_ns) / 1_000_000_000
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("benchmark duration is invalid")
        sidecar: dict[str, Any] = {
            "attempt_ledger": str(attempt_path),
            "attempt_ledger_sha256": attempt_hash,
            "argv": command,
            "argv_sha256": _argv_sha256(command),
            "data_root": str(data_root),
            "duration_seconds": duration_seconds,
            "ended_at": ended_at,
            "expected_revision": expected_revision,
            "gpu": _gpu_payload(gpu_before),
            "output_dir": str(output_dir),
            "product_tree": repository_before.product_tree,
            "pyproject_blob": repository_before.pyproject_blob,
            "results_sha256": results_hash,
            "run_id": run_id,
            "samples_sha256": samples_hash,
            "schema_version": 2,
            "sequence": _sequence_payload(sequence),
            "started_at": started_at,
            "uv_lock_blob": repository_before.uv_lock_blob,
            "worktree": str(worktree),
            "worktree_revision": repository_before.revision,
            "wrapper_sha256": wrapper_hash,
        }
        provenance_path = output_dir / "run-provenance.json"
        _atomic_write(provenance_path, sidecar)
        print(provenance_path)
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        _fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
