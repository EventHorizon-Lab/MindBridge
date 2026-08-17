"""Command line scaffolding shared by every reproducible benchmark runner.

Each name here replaces the same declaration repeated once per benchmark CLI. Anything that
differs between benchmarks — dataset selectors, subset validation, per-benchmark manifest counts —
stays in its own runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypedDict, TypeVar

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.sdk import MindBridge

_Item = TypeVar("_Item")
_Id = TypeVar("_Id")


@dataclass(frozen=True, slots=True)
class BenchmarkArguments:
    """Options every reproducible runner accepts."""

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
class MediaBenchmarkArguments(BenchmarkArguments):
    """Adds the options a runner needs when it waits for server-side media processing."""

    device_id: str
    poll_interval_seconds: float
    processing_timeout_seconds: float


class BenchmarkRunManifest(ContractModel):
    """Deployment, code, retrieval, and output identity recorded by every run."""

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


class MediaRunManifest(BenchmarkRunManifest):
    """Adds the identity a run records when it ingests media through the Worker."""

    annotation_sha256: Sha256Hex
    perception_prompt_version: NonEmptyString
    device_id: Identifier
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)


class SharedArgumentValues(TypedDict):
    """Parsed values for the fields declared by `BenchmarkArguments`."""

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


class MediaArgumentValues(TypedDict):
    """Parsed values for the extra fields declared by `MediaBenchmarkArguments`."""

    device_id: str
    poll_interval_seconds: float
    processing_timeout_seconds: float


def benchmark_parser(*, tenant_prefix: str) -> argparse.ArgumentParser:
    """Build a parser carrying the options every runner accepts."""
    parser = argparse.ArgumentParser()
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


def add_media_arguments(parser: argparse.ArgumentParser, *, device_id: str) -> None:
    """Add the options a runner needs while waiting for server-side media processing."""
    parser.add_argument("--device-id", default=device_id)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)


def shared_argument_values(parsed: argparse.Namespace) -> SharedArgumentValues:
    """Map the shared parsed options onto `BenchmarkArguments` field names."""
    return SharedArgumentValues(
        dataset_path=parsed.dataset,
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        code_revision=parsed.code_revision,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        recall_limit=parsed.recall_limit,
        request_concurrency=parsed.request_concurrency,
        request_timeout_seconds=parsed.request_timeout_seconds,
        overwrite=parsed.overwrite,
    )


def media_argument_values(parsed: argparse.Namespace) -> MediaArgumentValues:
    """Map the media processing options onto `MediaBenchmarkArguments` field names."""
    return MediaArgumentValues(
        device_id=parsed.device_id,
        poll_interval_seconds=parsed.poll_interval_seconds,
        processing_timeout_seconds=parsed.processing_timeout_seconds,
    )


@asynccontextmanager
async def connected_memory(arguments: BenchmarkArguments) -> AsyncIterator[MindBridge]:
    """Open the production client a runner measures, and always release it."""
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        yield memory
    finally:
        await memory.close()


class _Serializable(Protocol):
    def model_dump_json(self) -> str: ...

    def model_dump(self, *, mode: str) -> object: ...


def json_predictions(results: tuple[_Serializable, ...]) -> str:
    """Serialize results as one indented JSON array."""
    return (
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def jsonl_predictions(results: tuple[_Serializable, ...]) -> str:
    """Serialize results as one JSON object per line."""
    return "".join(result.model_dump_json() + "\n" for result in results)


def predictions_digest(predictions: str) -> str:
    """Hash the exact prediction bytes a run writes."""
    return hashlib.sha256(predictions.encode("utf-8")).hexdigest()


def completed_now() -> datetime:
    """Stamp manifest completion in UTC."""
    return datetime.now(timezone.utc)


def write_run_artifacts(
    output_path: Path,
    predictions: str,
    manifest: ContractModel,
) -> None:
    """Write predictions and their sidecar manifest, each atomically."""
    write_text_atomically(output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def select_by_id(
    items: tuple[_Item, ...],
    requested_ids: tuple[_Id, ...],
    *,
    identify: Callable[[_Item], _Id],
    label: str,
) -> tuple[_Item, ...]:
    """Restrict official items to a requested subset, rejecting duplicate or unknown IDs."""
    if not requested_ids:
        return items
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError(f"{label} IDs must not contain duplicates")
    requested = set(requested_ids)
    selected = tuple(item for item in items if identify(item) in requested)
    missing = requested - {identify(item) for item in selected}
    if missing:
        raise ValueError(f"unknown {label} IDs: {_readable(missing)}")
    return selected


def index_prepared(
    required_ids: tuple[_Id, ...],
    prepared: tuple[_Item, ...],
    *,
    identify: Callable[[_Item], _Id],
    label: str,
) -> dict[_Id, _Item]:
    """Index prepared media by ID, rejecting a run that is missing any required entry."""
    by_id = {identify(item): item for item in prepared}
    missing = set(required_ids) - by_id.keys()
    if missing:
        raise ValueError(f"missing prepared {label}: {_readable(missing)}")
    return by_id


def _readable(identifiers: set[_Id]) -> str:
    return ", ".join(sorted(str(identifier) for identifier in identifiers))
