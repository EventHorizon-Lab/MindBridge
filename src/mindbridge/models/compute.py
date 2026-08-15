"""Small device selectors shared by optional local model adapters."""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

from mindbridge.core import ModelUnavailableError


class _Cuda(Protocol):
    def is_available(self) -> bool: ...

    def mem_get_info(self) -> tuple[int, int]: ...


class _Torch(Protocol):
    cuda: _Cuda


def select_torch_device(requested: str | None = None) -> str:
    """Prefer CUDA when available while making explicit overrides fail loudly."""
    device = (requested or "auto").strip().lower()
    if device != "auto" and device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    try:
        torch = cast(_Torch, import_module("torch"))
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if device == "auto":
        return "cuda" if cuda_available else "cpu"
    if device.startswith("cuda") and not cuda_available:
        raise ModelUnavailableError("CUDA was requested but is not available")
    return device


def select_onnx_providers(
    available: tuple[str, ...],
    requested: str | None = None,
) -> tuple[str, ...]:
    """Return an accelerated ONNX provider chain with CPU fallback."""
    device = (requested or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda", "tensorrt"}:
        raise ValueError("ONNX device must be auto, cpu, cuda, or tensorrt")
    preferred = {
        # ponytail: TensorRT needs native-library and engine validation; opt in after deployment
        # calibration instead of letting ONNX Runtime silently fall all the way back to CPU.
        "auto": ("CUDAExecutionProvider",),
        "tensorrt": ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
        "cuda": ("CUDAExecutionProvider",),
        "cpu": (),
    }[device]
    selected = tuple(provider for provider in preferred if provider in available)
    if device != "auto" and device != "cpu" and not selected:
        raise ModelUnavailableError(f"requested ONNX {device} provider is not available")
    if "CPUExecutionProvider" in available:
        selected += ("CPUExecutionProvider",)
    if not selected:
        raise ModelUnavailableError("no supported ONNX execution provider is available")
    return selected


def has_free_cuda_memory(minimum_bytes: int) -> bool:
    """Return whether currently free CUDA memory can safely host parallel model work."""
    if minimum_bytes <= 0:
        raise ValueError("minimum CUDA memory must be positive")
    try:
        torch = cast(_Torch, import_module("torch"))
        return torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= minimum_bytes
    except (ImportError, RuntimeError):
        return False
