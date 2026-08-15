"""Small filesystem primitives shared by reproducible benchmark runners."""

from __future__ import annotations

from pathlib import Path


def sidecar_manifest_path(output_path: Path) -> Path:
    """Return the stable manifest path paired with a prediction artifact."""
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def require_writable_output_pair(output_path: Path, *, overwrite: bool) -> None:
    """Preserve either member of an existing predictions/manifest pair by default."""
    existing = [path for path in (output_path, sidecar_manifest_path(output_path)) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists: {existing[0]}")


def write_text_atomically(path: Path, content: str) -> None:
    """Replace one UTF-8 artifact only after its complete content is durable locally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)
