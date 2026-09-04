"""Reproducibility metadata for benchmark result artifacts."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import cast

_NVIDIA_QUERY = (
    "index",
    "name",
    "uuid",
    "memory.total",
    "driver_version",
    "power.limit",
)


def hardware_metadata() -> dict[str, object]:
    """Return client hardware facts that materially affect benchmark timings."""
    gpus = _nvidia_gpus()
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "logical_cores": _available_cpu_count(),
        "ram_total_bytes": _ram_total_bytes(),
        "gpus": gpus,
        # Kept for readers of schema-v10 artifacts while the richer list is adopted.
        "cuda_device_uuids": {
            str(gpu["index"]): gpu["uuid"]
            for gpu in gpus
            if isinstance(gpu.get("index"), int) and isinstance(gpu.get("uuid"), str)
        },
    }


def _available_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def acceleration_runtime_metadata() -> dict[str, object]:
    """Return the installed PyTorch/CUDA runtime, without making CUDA mandatory."""
    try:
        import torch
    except (ImportError, OSError):
        return {
            "torch_version": None,
            "torch_cuda_version": None,
            "torch_cuda_available": False,
            "torch_cudnn_version": None,
        }

    cudnn_version: int | None = None
    with suppress(AttributeError, RuntimeError):
        version = cast(Callable[[], int | None], torch.backends.cudnn.version)
        cudnn_version = version()
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cudnn_version": cudnn_version,
    }


def source_metadata(repository: Path | None = None) -> dict[str, object]:
    """Return the exact Git revision and whether local files differ from it."""
    root = _repository_root() if repository is None else repository
    commit = _git(root, ("rev-parse", "HEAD"))
    status = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git(repository: Path, arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    return result.stdout.strip()


def _nvidia_gpus() -> list[dict[str, object]]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                f"--query-gpu={','.join(_NVIDIA_QUERY)}",
                "--format=csv,noheader,nounits",
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []

    rows: list[dict[str, object]] = []
    for values in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(values) != len(_NVIDIA_QUERY):
            continue
        index = _integer(values[0])
        if index is None:
            continue
        rows.append(
            {
                "index": index,
                "name": _optional_text(values[1]),
                "uuid": _optional_text(values[2]),
                "memory_total_mib": _integer(values[3]),
                "driver_version": _optional_text(values[4]),
                "power_limit_watts": _number(values[5]),
            }
        )
    return rows


def _cpu_model() -> str | None:
    try:
        contents = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return platform.processor() or None
    for line in contents.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() in {"model name", "Hardware", "Processor"}:
            normalized = value.strip()
            if normalized:
                return normalized
    return platform.processor() or None


def _ram_total_bytes() -> int | None:
    try:
        contents = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in contents.splitlines():
        name, separator, value = line.partition(":")
        if separator and name == "MemTotal":
            fields = value.split()
            if len(fields) == 2 and fields[1].casefold() == "kb":
                kibibytes = _integer(fields[0])
                return None if kibibytes is None else kibibytes * 1024
    return None


def _optional_text(value: str) -> str | None:
    normalized = value.strip()
    return None if normalized.casefold() in {"", "n/a", "[n/a]", "not supported"} else normalized


def _integer(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def _number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def unavailable_server_resources(*, base_url: str) -> dict[str, object]:
    """Describe a remote server honestly when the benchmark cannot inspect it."""
    return {
        "scope": "remote_model_server",
        "base_url": base_url,
        "status": "unavailable",
        "reason": "the OpenAI-compatible protocol does not expose server resource telemetry",
    }
