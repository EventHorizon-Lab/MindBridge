"""Filesystem isolation for embedded benchmark runs."""

from __future__ import annotations

import base64
from pathlib import Path

_MAX_COMPONENT_BYTES = 240


class BenchmarkRun:
    """Allocate one physical data directory per benchmark unit."""

    def __init__(
        self,
        data_root: str | Path,
        benchmark: str,
        run_id: str,
        *,
        resume: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.benchmark = benchmark
        self.run_id = run_id
        self.resume = resume
        self.relative_layout = Path(
            _safe_component(benchmark, "benchmark"),
            _safe_component(run_id, "run"),
        )
        self.path = self.data_root / self.relative_layout

        self.data_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.mkdir(mode=0o700, exist_ok=True)
        try:
            self.path.mkdir(mode=0o700)
        except FileExistsError:
            if not self.path.is_dir():
                raise NotADirectoryError(
                    f"benchmark run path is not a directory: {self.path}"
                ) from None
            if not resume and any(self.path.iterdir()):
                raise FileExistsError(
                    f"benchmark run directory is not empty; pass resume=True to reuse it: {self.path}"
                ) from None

    def unit_dir(self, unit_id: str) -> Path:
        """Atomically create and return one isolated unit data directory."""
        path = self.path / _safe_component(unit_id, "unit")
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            if self.resume and path.is_dir():
                return path
            raise FileExistsError(f"benchmark unit directory already exists: {path}") from None
        return path


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} identifier must be non-empty and trimmed")
    encoded = base64.b32encode(value.encode("utf-8")).decode("ascii").rstrip("=").lower()
    component = f"{label}-{encoded}"
    if len(component.encode("ascii")) > _MAX_COMPONENT_BYTES:
        raise ValueError(f"{label} identifier is too long for a filesystem directory")
    return component
