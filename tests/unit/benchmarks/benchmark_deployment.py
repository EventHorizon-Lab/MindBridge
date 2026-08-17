"""The deployment snapshot every benchmark CLI check freezes before inference."""

from __future__ import annotations

import json
from pathlib import Path

SERVER_GENERATOR_REVISION = "answer-fingerprint"
WORKER_GENERATOR_REVISION = "perception-fingerprint"


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
            model_revision=SERVER_GENERATOR_REVISION,
            reasoning_effort="low",
        ),
        "server_embedder": _slot("openai", space_id="jina-v5", space_revision="space-v1"),
    }
    if worker:
        snapshot["worker_generator"] = _slot("openai", model_revision=WORKER_GENERATOR_REVISION)
        snapshot["worker_media_embedder"] = _slot("jina", space_revision="media-v1")
        snapshot["worker_text_embedder"] = _slot("openai", space_revision="text-v1")
    path = directory / "deployment.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path
