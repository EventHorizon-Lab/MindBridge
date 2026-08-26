"""Argument, manifest, and artifact shapes every reproducible benchmark CLI shares.

Each benchmark CLI owns its dataset selection, its runner invocation, and the counts it
pins. Everything a reproducible run needs regardless of benchmark — where the dataset and
output live, which deployment answered, which code produced it — lives here so a new flag
or manifest field is one edit instead of nine.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import AwareDatetime, Field

from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    LoadedDeployment,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.benchmarks.runtime import PreparedVideo
from mindbridge.benchmarks.scoring import (
    SCORING,
    JudgedAnswer,
    ScoringMode,
    build_judge,
    bypass_metrics,
    configured_judge_model,
    judge_answers,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.models import EmbedTask
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT
from mindbridge.sdk import MindBridge

TranscriptSource = Literal["none", "asr", "official_subtitles"]
"""Which official transcript a run ingested, if any.

Both Video-MME releases publish separate with- and without-subtitle columns, so which one a
number belongs in is not recoverable from the number itself.
"""

_Item = TypeVar("_Item")
_Unit = TypeVar("_Unit")
_Out = TypeVar("_Out")
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
    deployment_config_path: Path
    run_id: str
    tenant_prefix: str
    recall_limit: int
    request_concurrency: int
    unit_concurrency: int
    request_timeout_seconds: float
    limit: int | None
    overwrite: bool
    quiet: bool
    predict_only: bool


@dataclass(frozen=True, slots=True)
class MediaArguments(CoreArguments):
    """Adds the ingest-and-wait knobs a benchmark that uploads media also needs.

    The prepared-media path itself stays with each benchmark: MEMLENS makes it optional
    for its text-only mode, so a required field here would silently break that run.
    """

    device_id: str
    poll_interval_seconds: float
    processing_timeout_seconds: float


BENCHMARK_ENVIRONMENT = """environment:
  MINDBRIDGE_API_KEY    bearer token for --api-base-url; read from the environment so a
                        recorded invocation never carries the credential it used"""


def core_parser(
    *,
    tenant_prefix: str,
    prog: str | None = None,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Build the parser every benchmark CLI starts from."""
    parser = build_parser(prog=prog, description=description, epilog=BENCHMARK_ENVIRONMENT)
    parser.add_argument(
        "--dataset", type=Path, required=True, help="official dataset release to replay"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="predictions file to write; its manifest goes beside it as .manifest.json",
    )
    parser.add_argument(
        "--api-base-url", required=True, help="base URL of the deployed MindBridge API to measure"
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        required=True,
        help="JSON description of the deployment that answered, pinned into the manifest",
    )
    parser.add_argument(
        "--run-id", required=True, help="identifier isolating this run's tenants from every other"
    )
    parser.add_argument(
        "--tenant-prefix", default=tenant_prefix, help="prefix for the tenants this run writes to"
    )
    parser.add_argument(
        "--recall-limit", type=int, default=20, help="memories to retrieve per question"
    )
    parser.add_argument(
        "--request-concurrency", type=int, default=4, help="in-flight API requests per unit"
    )
    parser.add_argument(
        "--unit-concurrency",
        type=int,
        default=4,
        help="units of this benchmark run at once; the run holds up to "
        "--unit-concurrency times --request-concurrency requests in flight",
    )
    parser.add_argument(
        "--request-timeout-seconds", type=float, default=1_800.0, help="deadline for one request"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N of this benchmark's own units, for a smoke run; "
        "applied after any explicit selection",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing predictions and manifest"
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="write predictions without scoring them; every metric reports lmms-eval's bypass "
        "sentinel instead, and no judge is contacted",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the progress lines this run writes to stderr",
    )
    return parser


def add_media_arguments(
    parser: argparse.ArgumentParser,
    *,
    device_id: str,
) -> argparse.ArgumentParser:
    """Add the ingest-and-wait knobs every benchmark that uploads media shares."""
    parser.add_argument("--device-id", default=device_id, help="device identity to ingest as")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="delay between processing-status polls",
    )
    parser.add_argument(
        "--processing-timeout-seconds",
        type=float,
        default=1_800.0,
        help="deadline for one observation to finish processing",
    )
    return parser


def add_transcript_source_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Require a run to declare which official transcript, if any, it ingested."""
    parser.add_argument(
        "--transcript-source",
        required=True,
        choices=("none", "asr", "official_subtitles"),
        help="which official transcript this run ingests, if any",
    )
    return parser


def require_declared_transcripts(
    prepared: tuple[PreparedVideo, ...],
    transcript_source: TranscriptSource,
) -> None:
    """Refuse a run whose declared subtitle setting disagrees with its prepared media.

    Shared rather than copied per benchmark: the failure it prevents is a number filed under
    the wrong leaderboard column, which no later check can detect from the artifact alone.
    """
    present = any(
        segment.transcript is not None for video in prepared for segment in video.segments
    )
    if present and transcript_source == "none":
        raise ValueError(
            "prepared media carries transcripts; a run declaring no transcript source would "
            "report a with-subtitles result in the without-subtitles column"
        )
    if not present and transcript_source != "none":
        raise ValueError(
            f"prepared media carries no transcript, so it cannot be {transcript_source}"
        )


def report(message: str, *, quiet: bool) -> None:
    """Write one progress line to stderr so stdout stays the machine-readable artifact."""
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def report_unit(label: str, *, index: int, total: int, quiet: bool) -> None:
    """Announce one finished unit, which is the only progress a long run gives.

    `index` counts units that have completed, not the position of the one being started: units
    overlap, so at any moment several are in flight and no single one of them is "the third".
    """
    report(f"[{index}/{total}] {label}", quiet=quiet)


async def run_units(
    units: Sequence[_Unit],
    *,
    label: Callable[[_Unit], str],
    run: Callable[[_Unit], Awaitable[_Out]],
    unit_concurrency: int,
    quiet: bool,
) -> tuple[_Out, ...]:
    """Run this benchmark's units with `unit_concurrency` of them in flight, in release order.

    Awaiting each unit in turn -- which is what every runner used to do -- capped a whole run at
    `--request-concurrency` in-flight requests, because that is a unit's own budget and only one
    unit was ever spending it. Worse, the cap was not reachable for the whole of a unit either: a
    unit ingests before it answers, and its answer phase touches no Worker at all, so the queue
    the GPUs feed from drained once per unit and again at the end of every unit. Raising
    `--request-concurrency` could not fix it -- past the size of one unit's fan-out the flag
    bought nothing, because the ceiling was the serial loop and not the flag.

    Results come back in the order `units` were given, which several runners require: their
    predictions have to line up with the official annotation's own order.

    A unit that raises ends the run, as the serial `await` did, and the siblings are cancelled
    and waited out before it does. Neither half is what `gather` gives on its own. With
    `return_exceptions=False` it propagates the first exception but leaves the siblings running
    -- they then reach a client `connected_memory` has already closed, from tasks nobody awaits,
    and any unit that had already submitted observations leaves the Worker processing for a run
    that will write nothing. With `return_exceptions=True` every remaining unit runs to
    completion first, which is worse: a run whose first unit fails would ingest the whole corpus
    before saying so. `TaskGroup` does exactly this, and is 3.11; the floor here is 3.10.
    """
    if unit_concurrency <= 0:
        raise ValueError("unit_concurrency must be positive")
    gate = asyncio.Semaphore(unit_concurrency)
    completed = 0

    async def one(unit: _Unit) -> _Out:
        nonlocal completed
        async with gate:
            report(f"starting {label(unit)}", quiet=quiet)
            result = await run(unit)
        completed += 1
        report_unit(label(unit), index=completed, total=len(units), quiet=quiet)
        return result

    tasks = [asyncio.create_task(one(unit)) for unit in units]
    try:
        return tuple(await asyncio.gather(*tasks))
    except BaseException:
        # `BaseException` because an interrupt has to clean up too: a Ctrl-C during a sweep is
        # the likeliest way this path is reached, and leaving units in flight through it is how
        # the summary ends up written while requests are still going out.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
        "deployment_config_path": parsed.deployment_config,
        "run_id": parsed.run_id,
        "tenant_prefix": parsed.tenant_prefix,
        "recall_limit": parsed.recall_limit,
        "request_concurrency": parsed.request_concurrency,
        "unit_concurrency": parsed.unit_concurrency,
        "request_timeout_seconds": parsed.request_timeout_seconds,
        "limit": parsed.limit,
        "overwrite": parsed.overwrite,
        "quiet": parsed.quiet,
        "predict_only": parsed.predict_only,
    }


class ScoringSnapshot(ContractModel):
    """Who scored this run and what they found -- lmms-eval's `metric_list`, recorded per run.

    Every run carries one, including a run nobody scored: `mode` says which of the four ways
    produced the numbers, so a reader never has to infer it from whether a field is present.
    `higher_is_better` travels with them for the same reason lmms-eval's `metric_list` declares
    it -- a bare number does not say which direction is good.
    """

    mode: ScoringMode
    metrics: dict[str, float] = Field(default_factory=dict)
    higher_is_better: dict[str, bool] = Field(default_factory=dict)
    judge_model: NonEmptyString | None = None
    judge_failure_count: int = Field(default=0, ge=0)
    """Answers the judge could not score, floored to 0.0 and counted rather than only logged.

    Upstream logs each one and keeps no tally, so a run whose judge was unreachable is afterwards
    indistinguishable from a run that answered badly. This is the count that separates them; the
    0.0 floor itself is unchanged.
    """


class BenchmarkRunManifest(ContractModel):
    """Immutable data, deployment, and output identity for one benchmark run.

    Subclasses narrow `benchmark` to their own literal and add the counts they pin.
    Redeclaring an inherited field keeps its position, so `benchmark` stays first in the
    serialized manifest.
    """

    benchmark: NonEmptyString
    scoring: ScoringSnapshot
    runner_version: NonEmptyString
    adapter_version: NonEmptyString
    deployment: DeploymentSnapshot
    deployment_sha256: Sha256Hex
    answer_prompt_version: NonEmptyString
    retrieval_task: NonEmptyString
    run_id: Identifier
    tenant_prefix: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    unit_concurrency: int = Field(default=1, gt=0)
    """How many of this benchmark's units were in flight together.

    Defaulted rather than required so a manifest written before units could overlap still parses,
    and to 1 rather than to today's default because that is what those runs actually did.
    """
    request_timeout_seconds: float = Field(gt=0)
    predictions_sha256: Sha256Hex
    completed_at: AwareDatetime


class MediaBenchmarkRunManifest(BenchmarkRunManifest):
    """Adds the ingest identity a benchmark that uploads prepared media also pins."""

    annotation_sha256: Sha256Hex
    device_id: Identifier
    poll_interval_seconds: float = Field(gt=0)
    processing_timeout_seconds: float = Field(gt=0)


def scoring_snapshot(
    benchmark: str,
    arguments: CoreArguments,
    *,
    answers: Sequence[JudgedAnswer] = (),
    metrics: Mapping[str, float] | None = None,
) -> ScoringSnapshot:
    """Score this run the way its benchmark declares, and record who did it.

    One entry point for all four modes, so a runner names its benchmark and hands over either the
    numbers it computed itself or the answers a judge has to read, and never decides the policy.
    `--predict-only` short-circuits every mode, which is what `override_metric("bypass")` does
    upstream: no judge is contacted and no declared metric is computed.
    """
    declared = SCORING[benchmark]
    if arguments.predict_only:
        return ScoringSnapshot(mode="bypass", metrics=bypass_metrics())
    if declared.mode != "judge":
        return ScoringSnapshot(
            mode=declared.mode,
            metrics=dict(metrics or {}),
            higher_is_better=declared.higher_is_better(),
        )
    report(f"judging {len(answers)} answers with {configured_judge_model()}", quiet=arguments.quiet)
    outcome = judge_answers(
        answers,
        judge=build_judge(request_timeout_seconds=arguments.request_timeout_seconds),
        concurrency=arguments.request_concurrency,
    )
    if outcome.failure_count:
        report(
            f"{outcome.failure_count} of {len(answers)} answers scored 0.0 because the judge "
            "could not be read; that is a floor, not a measurement",
            quiet=arguments.quiet,
        )
    return ScoringSnapshot(
        mode="judge",
        metrics={**outcome.metrics, **dict(metrics or {})},
        higher_is_better=declared.higher_is_better(),
        judge_model=configured_judge_model(),
        judge_failure_count=outcome.failure_count,
    )


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
        "deployment": deployment.snapshot,
        "deployment_sha256": deployment.sha256,
        "answer_prompt_version": ANSWER_FROM_EVIDENCE_PROMPT.version,
        "retrieval_task": EmbedTask.DOCUMENT.value,
        "run_id": arguments.run_id,
        "tenant_prefix": arguments.tenant_prefix,
        "recall_limit": arguments.recall_limit,
        "request_concurrency": arguments.request_concurrency,
        "unit_concurrency": arguments.unit_concurrency,
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
    limit: int | None = None,
) -> tuple[_Item, ...]:
    """Narrow an official release to the requested units, in the order the release lists them.

    `label` names the unit the way the benchmark's own operator does — "EgoMemReason example IDs",
    "MM-Lifelong question indices" — because that text is what someone reads when a run refuses to
    start. Empty `requested` means the whole release, which is how every runner spells "no subset".

    `limit` truncates whatever the selection produced, so `--limit` composes with a benchmark's own
    ID flags rather than competing with them. It is deliberately a count of this benchmark's own
    units and not of questions: what a run of one of these costs is dominated by ingesting the
    unit, so limiting questions inside a unit that was ingested anyway saves almost nothing.
    """
    wanted = tuple(requested)
    selected = items
    if wanted:
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"{label} must not contain duplicates")
        unique = set(wanted)
        selected = tuple(item for item in items if key(item) in unique)
        missing = unique - {key(item) for item in selected}
        if missing:
            formatted = ", ".join(str(item) for item in sorted(missing))
            raise ValueError(f"unknown {label}: {formatted}")
    return limit_units(selected, limit, label=label)


def limit_units(
    items: tuple[_Item, ...],
    limit: int | None,
    *,
    label: str,
) -> tuple[_Item, ...]:
    """Keep the first `limit` units of a selection, refusing a limit that would select nothing.

    Separate from `select_by_id` because two benchmarks filter again after selecting by ID —
    Video-MME by duration band, Video-MME-v2 by group type — and a limit applied before that
    filter is a limit on the wrong set: `--limit 2 --duration long` would truncate to the first
    two videos of any band and then quite possibly keep none of them.
    """
    if limit is None:
        return items
    if limit < 1:
        raise ValueError(f"--limit must be a positive count of {label}")
    return items[:limit]


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
