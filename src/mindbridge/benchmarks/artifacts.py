"""Run identity, selection, and filesystem primitives shared by benchmark runners."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from mindbridge.configuration import copy_plugin_configuration, validate_plugin_name
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex

_Item = TypeVar("_Item")
# Benchmarks number their units either by string ID or by integer index, and the two sort
# differently: a `Hashable` bound would force string ordering onto integers, reporting missing
# EgoMemReason examples as "10, 100, 9". Restricting the variable keeps `sorted` numeric.
_Key = TypeVar("_Key", str, int)


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
    server_reranker: PluginSnapshot | None = None
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


class RunManifestBase(ContractModel):
    """The reproducibility identity every benchmark manifest carries.

    A subclass adds its own `benchmark` literal plus whatever the benchmark alone pins: its
    dataset and evaluator revisions, prepared-media hashes, selected units, and metrics. The
    fields here are the ones that mean the same thing for all nine, so a reader can compare two
    runs of different benchmarks without learning two manifest shapes.
    """

    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    code_revision: NonEmptyString
    deployment: DeploymentSnapshot
    deployment_sha256: Sha256Hex
    answer_prompt_version: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


class MediaRunManifestBase(RunManifestBase):
    """Adds what a run that ingests prepared media pins on top of the shared run identity.

    Eight of the nine benchmarks feed observations through the perception path; LoCoMo is
    text-only and inherits `RunManifestBase` directly. Perception identity stays with each
    benchmark because MEMLENS can run a text-only split where no perception happens at all.
    """

    annotation_sha256: Sha256Hex
    device_id: Identifier
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)


@dataclass(frozen=True, slots=True)
class CommonArguments:
    """The invocation every benchmark CLI accepts, whatever it then does with it."""

    dataset_path: Path
    output_path: Path
    api_base_url: str
    code_revision: str
    deployment_config_path: Path
    run_id: str
    tenant_prefix: str
    recall_limit: int
    request_concurrency: int
    request_timeout_seconds: float
    overwrite: bool


@dataclass(frozen=True, slots=True)
class MediaArguments(CommonArguments):
    """The invocation of a benchmark that ingests prepared media through the Worker."""

    device_id: str
    poll_interval_seconds: float
    processing_timeout_seconds: float


def common_benchmark_parser(*, tenant_prefix: str) -> argparse.ArgumentParser:
    """Build the shared flags as an `argparse` parent, leaving benchmark flags to the caller."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default=tenant_prefix)
    parser.add_argument("--recall-limit", type=int, default=20)
    parser.add_argument("--request-concurrency", type=int, default=4)
    parser.add_argument("--request-timeout-seconds", type=float, default=1_800.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def media_benchmark_parser(
    *, tenant_prefix: str, device_id: str = "benchmark_camera"
) -> argparse.ArgumentParser:
    """Extend the shared parent with the ingest flags every media benchmark accepts."""
    parser = argparse.ArgumentParser(
        add_help=False, parents=[common_benchmark_parser(tenant_prefix=tenant_prefix)]
    )
    parser.add_argument("--device-id", default=device_id)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)
    return parser


def select_by_id(
    items: tuple[_Item, ...],
    requested: Iterable[_Key],
    *,
    key: Callable[[_Item], _Key],
    label: str,
) -> tuple[_Item, ...]:
    """Narrow an official release to the requested units, in the order the release lists them."""
    wanted = tuple(requested)
    if not wanted:
        return items
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"{label} must not contain duplicates")
    unique = set(wanted)
    selected = tuple(item for item in items if key(item) in unique)
    missing = unique - {key(item) for item in selected}
    if missing:
        formatted = ", ".join(str(item) for item in sorted(missing))
        raise ValueError(f"unknown {label}: {formatted}")
    return selected


def sha256_text(content: str) -> str:
    """Hash exactly the UTF-8 bytes an artifact is written with."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def predictions_document(payload: object) -> str:
    """Render a prediction artifact the one way every runner renders it."""
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_run_artifacts(
    output_path: Path,
    predictions: str,
    manifest: RunManifestBase,
) -> None:
    """Write a prediction artifact and the manifest that identifies it, as a pair."""
    write_text_atomically(output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


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
