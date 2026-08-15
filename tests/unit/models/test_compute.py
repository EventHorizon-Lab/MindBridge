"""Checks for deterministic local model device selection."""

from types import SimpleNamespace

import pytest

from mindbridge.core import ModelUnavailableError
from mindbridge.models import compute


def test_compute_auto_prefers_gpu_and_keeps_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compute,
        "import_module",
        lambda _name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )

    assert compute.select_torch_device() == "cuda"
    assert compute.select_onnx_providers(
        ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    ) == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert compute.select_onnx_providers(
        ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"),
        "tensorrt",
    ) == ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


def test_compute_rejects_an_unavailable_explicit_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        compute,
        "import_module",
        lambda _name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    with pytest.raises(ModelUnavailableError, match="CUDA"):
        compute.select_torch_device("cuda")
    with pytest.raises(ModelUnavailableError, match="cuda provider"):
        compute.select_onnx_providers(("CPUExecutionProvider",), "cuda")


def test_compute_parallelizes_only_when_free_cuda_memory_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda: (9 * 1024**3, 32 * 1024**3),
    )
    monkeypatch.setattr(compute, "import_module", lambda _name: SimpleNamespace(cuda=available))

    assert compute.has_free_cuda_memory(8 * 1024**3)
    assert not compute.has_free_cuda_memory(10 * 1024**3)
