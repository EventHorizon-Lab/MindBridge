"""Result artifacts: atomic JSON/JSONL writes, the secret-free config copy, and manifest merging."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import (
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import fields
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast

import yaml
from pydantic import SecretStr

if TYPE_CHECKING:
    pass
from mindbridge import MindBridgeConfig
from mindbridge.benchmarks.eval_config import (
    _CONFIG_FILE,
    _PARTIAL_SAMPLES_FILE,
    _RESULTS_FILE,
    _SAMPLES_FILE,
    _Arguments,
    _JudgeConfig,
    _memory_config_payload,
)
from mindbridge.benchmarks.eval_results import SampleResult
from mindbridge.benchmarks.model_config import (
    DownloadSettings,
    ModelConfig,
    ServerMetricsOverrides,
)


def _write_artifacts(
    arguments: _Arguments,
    samples: Sequence[SampleResult],
    results: Mapping[str, object],
    *,
    config_bytes: bytes | None = None,
) -> None:
    samples_bytes = _jsonl_bytes(sample.json() for sample in samples)
    document = dict(results)
    document["samples_sha256"] = hashlib.sha256(samples_bytes).hexdigest()
    results_bytes = _jsonl_bytes((document,))
    arguments.output_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = [
        (arguments.output_path / _SAMPLES_FILE, samples_bytes),
        (arguments.output_path / _RESULTS_FILE, results_bytes),
    ]
    if config_bytes is not None:
        files.append((arguments.output_path / _CONFIG_FILE, config_bytes))
    _atomic_replace(files)
    # The real samples file now holds everything the crash copy did.
    (arguments.output_path / _PARTIAL_SAMPLES_FILE).unlink(missing_ok=True)


_SECRET_CONFIG_KEYS = frozenset({"api_key", "authorization", "password", "secret", "token"})


_OPAQUE_CONFIG_KEYS = frozenset({"extra_body"})


def _config_artifact(
    arguments: _Arguments,
    config: ModelConfig,
    judge_config: _JudgeConfig,
    memory_config: MindBridgeConfig | None,
    download: DownloadSettings,
    server_metrics: ServerMetricsOverrides,
) -> bytes:
    """Serialize the resolved run configuration without serializing credentials.

    This is a comparison manifest, not a byte-for-byte copy of the input file: flags and
    environment values may override that file, and every OpenAI-compatible block can contain an
    API key. Keeping the resolved, secret-free values is both safer and the only snapshot that
    describes the run that produced the neighboring results.
    """
    product = (
        {
            "embedding": {"provider": "jina-omni"},
            "generation": {
                "provider": "openai",
                "base_url": config.generation_base_url,
                "model": config.generation_model,
                "timeout": config.timeout_seconds,
                "modalities": sorted(modality.value for modality in config.generation_capabilities),
                "min_video_seconds": config.generation_min_video_seconds,
            },
        }
        if memory_config is None
        else _memory_config_payload(memory_config)
    )
    run = {
        item.name: getattr(arguments, item.name)
        for item in fields(_Arguments)
        # These raw strings either duplicate resolved blocks below or, for judge arguments, may
        # themselves contain an API key. The source config path is provenance, not run behavior.
        if item.name not in {"judge_model_args", "memory_config"}
    }
    document = {
        "artifact": {
            "kind": "mindbridge-bench-effective-config",
            "schema_version": 1,
            "credentials": "omitted",
        },
        "product": product,
        "benchmark": {
            "judge": {
                "model": judge_config.model,
                "base_url": judge_config.base_url,
                "timeout_seconds": judge_config.timeout_seconds,
                "concurrency": judge_config.concurrency,
            },
            "download": {
                "benchmarks_root": download.benchmarks_root,
                "data_root": download.data_root,
                "hf_home": download.hf_home,
                "hf_endpoint": download.hf_endpoint,
                "youtube_sleep_seconds": download.youtube_sleep_seconds,
            },
            "server_metrics": server_metrics.model_dump(mode="json"),
            "run": run,
        },
    }
    safe = _secret_free_yaml_value(document)
    return yaml.safe_dump(safe, allow_unicode=True, sort_keys=True).encode("utf-8")


def _secret_free_yaml_value(value: object) -> object:
    """Convert a resolved config tree to safe YAML primitives and omit credential fields."""
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            normalized = name.casefold()
            if normalized in _SECRET_CONFIG_KEYS:
                continue
            # Provider-specific request bodies are intentionally unconstrained and can carry
            # credentials under arbitrary names. Their values cannot be proven safe by a key
            # blacklist, so retain only whether the effective request configured one.
            if normalized in _OPAQUE_CONFIG_KEYS and item:
                result[name] = {"configured": True, "values": "omitted"}
                continue
            result[name] = _secret_free_yaml_value(item)
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return [_secret_free_yaml_value(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_secret_free_yaml_value(item) for item in value]
    if isinstance(value, SecretStr):
        # Defensive fallback for a future credential field whose name is not yet `api_key`.
        return "<redacted>"
    if isinstance(value, str):
        # A gateway URL can carry basic-auth userinfo; `base_url` and the raw `--model_args`
        # string both reach this manifest, so the credential is cut out of the URL itself.
        return _URL_USERINFO.sub("", value)
    return value


_URL_USERINFO = re.compile(r"(?<=://)[^/@\s]+@")


def _atomic_replace(files: Sequence[tuple[Path, bytes]]) -> None:
    temporary: list[tuple[Path, Path]] = []
    try:
        for target, content in files:
            with NamedTemporaryFile(
                mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary.append((Path(stream.name), target))
        for source, target in temporary:
            os.replace(source, target)
    finally:
        for source, _target in temporary:
            source.unlink(missing_ok=True)


def _merged_manifest(
    manifest: Mapping[str, object] | None,
    manifest_directory: Path | None,
    generated: Mapping[str, object],
) -> dict[str, object]:
    payload = cast(
        dict[str, object], _absolute_manifest_paths(dict(manifest or {}), manifest_directory)
    )
    tasks = payload.get("tasks")
    merged = dict(tasks) if isinstance(tasks, dict) else {}
    merged.update(generated)
    payload.update({"version": 1, "tasks": merged})
    return payload


def _absolute_manifest_paths(value: object, directory: Path | None) -> object:
    if isinstance(value, list):
        return [_absolute_manifest_paths(item, directory) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _absolute_manifest_paths(item, directory) for key, item in value.items()}
    path = result.get("path")
    if (
        isinstance(path, str)
        and directory is not None
        and not Path(path).expanduser().is_absolute()
    ):
        result["path"] = str((directory / Path(path).expanduser()).resolve())
    return result


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[object]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)
