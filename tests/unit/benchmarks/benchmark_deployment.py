"""The deployment snapshot every benchmark CLI check freezes before inference."""

from __future__ import annotations

import json
from pathlib import Path


def _slot(plugin: str, **config: str) -> dict[str, object]:
    return {
        "plugin": plugin,
        "distribution": "mindbridge",
        "version": "0.1.0",
        "config": config,
    }


def write_deployment_snapshot(directory: Path, *, worker: bool = True) -> Path:
    """Write a secret-free plugin snapshot, with the Worker slots unless a run has no media."""
    snapshot: dict[str, object] = {
        "server_generator": _slot(
            "openai",
            model_id="qwen3.8-max",
            reasoning_effort="low",
        ),
        "server_embedder": _slot("openai", space_id="jina-v5"),
    }
    if worker:
        snapshot["worker_generator"] = _slot("openai")
        snapshot["worker_media_embedder"] = _slot("jina")
        snapshot["worker_text_embedder"] = _slot("openai")
    path = directory / "deployment.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path
