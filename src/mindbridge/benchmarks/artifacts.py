"""Small filesystem primitives shared by reproducible benchmark runners."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, JsonValue, field_validator, model_validator

from mindbridge.configuration import copy_plugin_configuration, validate_plugin_name
from mindbridge.contracts import ContractModel, NonEmptyString


class PluginSnapshot(ContractModel):
    """One selected plugin and its reproducible, non-secret configuration."""

    plugin: NonEmptyString
    distribution: NonEmptyString
    version: NonEmptyString
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("plugin", mode="before")
    @classmethod
    def require_canonical_name(cls, value: object) -> object:
        return validate_plugin_name(value)

    @model_validator(mode="after")
    def reject_credentials(self) -> PluginSnapshot:
        copy_plugin_configuration(self.config, "benchmark plugin configuration")
        if _contains_secret(self.config):
            raise ValueError("benchmark plugin configuration must not contain credentials")
        return self


class DeploymentSnapshot(ContractModel):
    """Resolved capability selections used by one benchmark deployment."""

    server_generator: PluginSnapshot
    server_embedder: PluginSnapshot
    worker_generator: PluginSnapshot | None = None
    worker_media_embedder: PluginSnapshot | None = None
    worker_text_embedder: PluginSnapshot | None = None

    @model_validator(mode="after")
    def require_complete_worker(self) -> DeploymentSnapshot:
        worker = (
            self.worker_generator,
            self.worker_media_embedder,
            self.worker_text_embedder,
        )
        if any(item is not None for item in worker) and any(item is None for item in worker):
            raise ValueError("worker plugin snapshots must be provided together")
        return self


@dataclass(frozen=True, slots=True)
class LoadedDeployment:
    """One validated deployment file frozen before a benchmark starts."""

    snapshot: DeploymentSnapshot
    sha256: str


def load_deployment_snapshot(
    path: Path,
    *,
    require_worker: bool = False,
) -> LoadedDeployment:
    """Load the exact secret-free plugin selection recorded with a run."""
    encoded = path.read_bytes()
    snapshot = DeploymentSnapshot.model_validate_json(encoded)
    if require_worker and snapshot.worker_generator is None:
        raise ValueError("raw-media benchmarks require complete worker plugin snapshots")
    return LoadedDeployment(snapshot, hashlib.sha256(encoded).hexdigest())


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


def _contains_secret(value: JsonValue) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(character for character in key.casefold() if character.isalnum())
            if normalized in {
                "accesskey",
                "accesskeyid",
                "apikey",
                "authorization",
                "clientsecret",
                "connectionstring",
                "credential",
                "credentials",
                "password",
                "privatekey",
                "secret",
                "token",
            } or normalized.endswith(
                ("accesskey", "apikey", "password", "privatekey", "secret", "token")
            ):
                return True
            if _contains_secret(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return False
