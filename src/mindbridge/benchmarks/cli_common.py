"""Argument, manifest, and artifact shapes every reproducible benchmark CLI shares.

Each benchmark CLI owns its dataset selection, its runner invocation, and the counts it
pins. Everything a reproducible run needs regardless of benchmark — where the dataset and
output live, which deployment answered, which code produced it — lives here so a new flag
or manifest field is one edit instead of nine.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    LoadedDeployment,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT
from mindbridge.sdk import MindBridge

_Item = TypeVar("_Item")
_Prepared = TypeVar("_Prepared")
# Benchmarks key prepared units either by string ID or by integer index, and the two sort
# differently: constraining the variable keeps `sorted` numeric so a missing EgoMemReason
# example is reported as "9, 10, 100" rather than "10, 100, 9".
_Key = TypeVar("_Key", str, int)


@dataclass(frozen=True, slots=True)
class CoreArguments:
    """The dataset, deployment, and output identity every benchmark run needs."""

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
class MediaArguments(CoreArguments):
    """Adds the ingest-and-wait knobs a benchmark that uploads media also needs.

    The prepared-media path itself stays with each benchmark: MEMLENS makes it optional
    for its text-only mode, so a required field here would silently break that run.
    """

    device_id: str
    poll_interval_seconds: float
    processing_timeout_seconds: float


def core_parser(*, tenant_prefix: str) -> argparse.ArgumentParser:
    """Build the parser every benchmark CLI starts from."""
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


def add_media_arguments(
    parser: argparse.ArgumentParser,
    *,
    device_id: str,
) -> argparse.ArgumentParser:
    """Add the ingest-and-wait knobs every benchmark that uploads media shares."""
    parser.add_argument("--device-id", default=device_id)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--processing-timeout-seconds", type=float, default=1_800.0)
    return parser


_ArgumentsT = TypeVar("_ArgumentsT", bound=CoreArguments)
_ManifestT = TypeVar("_ManifestT", bound="BenchmarkRunManifest")


def core_arguments(
    arguments_type: type[_ArgumentsT],
    parsed: argparse.Namespace,
    **benchmark_values: object,
) -> _ArgumentsT:
    """Build one benchmark's arguments from the shared flags plus its own."""
    return arguments_type(**_core_values(parsed), **benchmark_values)  # type: ignore[arg-type]


def media_arguments(
    arguments_type: type[_ArgumentsT],
    parsed: argparse.Namespace,
    **benchmark_values: object,
) -> _ArgumentsT:
    """Build one media benchmark's arguments from the shared flags plus its own."""
    values = _core_values(parsed) | {
        "device_id": parsed.device_id,
        "poll_interval_seconds": parsed.poll_interval_seconds,
        "processing_timeout_seconds": parsed.processing_timeout_seconds,
    }
    # A dataclass has no validating constructor to route through, so this one splat is
    # unchecked on purpose. Keeping it here is what spares every CLI its own suppression.
    return arguments_type(**values, **benchmark_values)  # type: ignore[arg-type]


def _core_values(parsed: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_path": parsed.dataset,
        "output_path": parsed.output,
        "api_base_url": parsed.api_base_url,
        "code_revision": parsed.code_revision,
        "deployment_config_path": parsed.deployment_config,
        "run_id": parsed.run_id,
        "tenant_prefix": parsed.tenant_prefix,
        "recall_limit": parsed.recall_limit,
        "request_concurrency": parsed.request_concurrency,
        "request_timeout_seconds": parsed.request_timeout_seconds,
        "overwrite": parsed.overwrite,
    }


class BenchmarkRunManifest(ContractModel):
    """Immutable data, deployment, code, and output identity for one benchmark run.

    Subclasses narrow `benchmark` to their own literal and add the counts they pin.
    Redeclaring an inherited field keeps its position, so `benchmark` stays first in the
    serialized manifest.
    """

    benchmark: NonEmptyString
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


class MediaBenchmarkRunManifest(BenchmarkRunManifest):
    """Adds the ingest identity a benchmark that uploads prepared media also pins."""

    annotation_sha256: Sha256Hex
    device_id: Identifier
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)


def predictions_jsonl(results: Sequence[ContractModel]) -> str:
    """Serialize one result per line in the order the official annotation asked for."""
    return "".join(result.model_dump_json() + "\n" for result in results)


def core_manifest(
    manifest_type: type[_ManifestT],
    arguments: CoreArguments,
    deployment: LoadedDeployment,
    *,
    runner_version: str,
    adapter_version: str,
    predictions: str,
    **benchmark_fields: object,
) -> _ManifestT:
    """Build one benchmark's manifest from the shared identity plus its own counts."""
    shared = _core_manifest_values(
        arguments,
        deployment,
        runner_version=runner_version,
        adapter_version=adapter_version,
        predictions=predictions,
    )
    return manifest_type.model_validate(shared | benchmark_fields)


def media_manifest(
    manifest_type: type[_ManifestT],
    arguments: MediaArguments,
    deployment: LoadedDeployment,
    *,
    runner_version: str,
    adapter_version: str,
    annotation_sha256: str,
    predictions: str,
    **benchmark_fields: object,
) -> _ManifestT:
    """Build one media benchmark's manifest, pinning what it ingested as well."""
    shared = _core_manifest_values(
        arguments,
        deployment,
        runner_version=runner_version,
        adapter_version=adapter_version,
        predictions=predictions,
    ) | {
        "annotation_sha256": annotation_sha256,
        "device_id": arguments.device_id,
        "poll_interval_seconds": arguments.poll_interval_seconds,
        "processing_timeout_seconds": arguments.processing_timeout_seconds,
    }
    return manifest_type.model_validate(shared | benchmark_fields)


def _core_manifest_values(
    arguments: CoreArguments,
    deployment: LoadedDeployment,
    *,
    runner_version: str,
    adapter_version: str,
    predictions: str,
) -> dict[str, object]:
    return {
        "runner_version": runner_version,
        "adapter_version": adapter_version,
        "code_revision": arguments.code_revision,
        "deployment": deployment.snapshot,
        "deployment_sha256": deployment.sha256,
        "answer_prompt_version": ANSWER_FROM_EVIDENCE_PROMPT.version,
        "retrieval_task": EmbedTask.DOCUMENT.value,
        "run_id": arguments.run_id,
        "tenant_prefix": arguments.tenant_prefix,
        "recall_limit": arguments.recall_limit,
        "request_concurrency": arguments.request_concurrency,
        "request_timeout_seconds": arguments.request_timeout_seconds,
        "predictions_sha256": hashlib.sha256(predictions.encode("utf-8")).hexdigest(),
        "completed_at": datetime.now(timezone.utc),
    }


def write_run_artifacts(
    output_path: Path,
    predictions: str,
    manifest: BenchmarkRunManifest,
) -> None:
    """Write the predictions and their sidecar manifest as the run's only outputs."""
    write_text_atomically(output_path, predictions)
    write_text_atomically(
        sidecar_manifest_path(output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


@asynccontextmanager
async def connected_memory(arguments: CoreArguments) -> AsyncIterator[MindBridge]:
    """Open the production client a run measures, and release it however the run ends.

    The API key is read from the environment rather than an argument so a run's recorded
    invocation never carries the credential it used.
    """
    memory = MindBridge.connect(
        base_url=arguments.api_base_url,
        api_key=os.environ.get("MINDBRIDGE_API_KEY"),
        timeout_seconds=arguments.request_timeout_seconds,
    )
    try:
        yield memory
    finally:
        await memory.close()


def select_by_id(
    items: tuple[_Item, ...],
    requested: Iterable[_Key],
    *,
    key: Callable[[_Item], _Key],
    label: str,
) -> tuple[_Item, ...]:
    """Narrow an official release to the requested units, in the order the release lists them.

    `label` names the unit the way the benchmark's own operator does — "EgoMemReason example IDs",
    "MM-Lifelong question indices" — because that text is what someone reads when a run refuses to
    start. Empty `requested` means the whole release, which is how every runner spells "no subset".
    """
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


def index_prepared(
    required: Iterable[_Key],
    prepared: tuple[_Prepared, ...],
    *,
    key: Callable[[_Prepared], _Key],
    label: str,
) -> dict[_Key, _Prepared]:
    """Index prepared media by unit, refusing a run that is missing any required entry.

    Returns the whole index rather than an ordered sequence: callers that need release order
    already hold the units and can read them back in it.
    """
    by_key = {key(item): item for item in prepared}
    missing = set(required) - by_key.keys()
    if missing:
        formatted = ", ".join(str(item) for item in sorted(missing))
        raise ValueError(f"missing prepared {label}: {formatted}")
    return by_key
