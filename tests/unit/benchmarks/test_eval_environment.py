from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindbridge.benchmarks import eval_environment


def test_hardware_metadata_records_reproducible_gpu_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        "0, NVIDIA GeForce RTX 5090, GPU-one, 32607, 580.178.04, 575.00\n"
        "1, NVIDIA GPU, GPU-two, N/A, 580.178.04, N/A\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    monkeypatch.setattr(eval_environment, "_cpu_model", lambda: "Fixture CPU")
    monkeypatch.setattr(eval_environment, "_ram_total_bytes", lambda: 64 * 1024**3)

    metadata = eval_environment.hardware_metadata()

    assert metadata["cpu_model"] == "Fixture CPU"
    assert metadata["ram_total_bytes"] == 64 * 1024**3
    assert metadata["cuda_device_uuids"] == {"0": "GPU-one", "1": "GPU-two"}
    assert metadata["gpus"] == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 5090",
            "uuid": "GPU-one",
            "memory_total_mib": 32607,
            "driver_version": "580.178.04",
            "power_limit_watts": 575.0,
        },
        {
            "index": 1,
            "name": "NVIDIA GPU",
            "uuid": "GPU-two",
            "memory_total_mib": None,
            "driver_version": "580.178.04",
            "power_limit_watts": None,
        },
    ]


def test_hardware_metadata_handles_missing_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", missing)

    metadata = eval_environment.hardware_metadata()

    assert metadata["gpus"] == []
    assert metadata["cuda_device_uuids"] == {}


def test_source_metadata_distinguishes_clean_dirty_and_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outputs = iter(("a" * 40, " M src/mindbridge/memory.py"))
    monkeypatch.setattr(eval_environment, "_git", lambda *_args: next(outputs))

    assert eval_environment.source_metadata(tmp_path) == {
        "git_commit": "a" * 40,
        "git_dirty": True,
    }

    monkeypatch.setattr(eval_environment, "_git", lambda *_args: None)
    assert eval_environment.source_metadata(tmp_path) == {
        "git_commit": None,
        "git_dirty": None,
    }


def test_unavailable_server_resources_names_scope_and_reason() -> None:
    assert eval_environment.unavailable_server_resources(base_url="https://models.example/v1") == {
        "scope": "remote_model_server",
        "base_url": "https://models.example/v1",
        "status": "unavailable",
        "reason": "the OpenAI-compatible protocol does not expose server resource telemetry",
    }
