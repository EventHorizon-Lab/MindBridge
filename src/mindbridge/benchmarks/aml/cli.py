"""Run one Agent Memory Leaderboard benchmark against a deployed MindBridge API.

Ties together the six benchmark-neutral loaders, `run_case` (Task 12), and the
six vendored AML scoring pipelines (Task 13). This module -- like every module
under `mindbridge.benchmarks.aml` -- must not import `mindbridge.api`, so it
stays usable against a real deployed server rather than only an in-process one.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import Field

from mindbridge.benchmarks.aml.cases import AmlCase
from mindbridge.benchmarks.aml.driver import (
    EmitFn,
    emit_chat_history,
    emit_context_messages,
    emit_retrieved_context,
    emit_selected,
    eval_user_id,
    row_id,
    run_case,
)
from mindbridge.benchmarks.aml.loaders import (
    beam,
    clbench,
    locomo_refined,
    longmemeval,
    personamem_v1,
    personamem_v2,
)
from mindbridge.benchmarks.artifacts import (
    DeploymentSnapshot,
    LoadedDeployment,
    load_deployment_snapshot,
    sidecar_manifest_path,
    write_text_atomically,
)
from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.benchmarks.cli_common import report, report_unit
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file

# The upstream the vendored pipelines under `benchmarks/aml/` came from, whose
# per-file digests `benchmarks/aml/PINNED.md` records. A constant, not a CLI
# flag: changing it means re-vendoring, not choosing a value per run.
_AML_SOURCE_REPOSITORY = "AML-memory/agent-memory-leaderboard"

# Duplicated from `mindbridge.api.aml_contracts.derive_tenant_id` rather than
# imported: nothing under `mindbridge.benchmarks.aml` may import
# `mindbridge.api` (see module docstring), and this is the one place the CLI
# needs the same mapping, to report it in the run manifest.
_TENANT_DIGEST_CHARACTERS = 32


def _derive_tenant_id(prefix: str, user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:_TENANT_DIGEST_CHARACTERS]}"


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PIPELINES_ROOT = _REPO_ROOT / "benchmarks" / "aml" / "pipelines"


def _one(paths: Sequence[Path]) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected exactly one --dataset path, got {len(paths)}")
    return paths[0]


def _two(paths: Sequence[Path]) -> tuple[Path, Path]:
    if len(paths) != 2:
        raise ValueError(f"expected exactly two --dataset paths, got {len(paths)}")
    return paths[0], paths[1]


def _load_locomo_refined(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    return locomo_refined.load(_one(paths))


def _load_longmemeval(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    return longmemeval.load(_one(paths))


def _load_clbench(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    return clbench.load(_one(paths))


def _load_personamem_v1(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    questions_csv, contexts_jsonl = _two(paths)
    return personamem_v1.load(questions_csv, contexts_jsonl)


def _load_personamem_v2(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    benchmark_csv, data_root = _two(paths)
    return personamem_v2.load(benchmark_csv, data_root)


def _load_beam(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    """Discover every BEAM conversation under the `chats/` root and load each.

    `beam.load` (Task 10) takes one conversation's two files at a time; the
    real corpus nests conversations as
    `chats/{100K,500K,1M,10M}/{conv_id}/chat.json` plus
    `probing_questions/probing_questions.json` alongside it, so the single
    `--dataset` path here is that `chats/` root, walked once.
    """
    chats_root = _one(paths)
    cases: list[AmlCase] = []
    for chat_path in sorted(chats_root.glob("*/*/chat.json")):
        questions_path = chat_path.parent / "probing_questions" / "probing_questions.json"
        cases.extend(beam.load(chat_path, questions_path))
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One benchmark's loader, retrieved-context emitter, and vendored scorer."""

    load: Callable[[Sequence[Path]], tuple[AmlCase, ...]]
    emit: EmitFn
    pipeline: Path


AML_ENVIRONMENT = """environment:
  MINDBRIDGE_AML_API_KEY  bearer token for --api-base-url; required, read from the
                          environment so a recorded invocation never carries it"""


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "locomo-refined": BenchmarkSpec(
        _load_locomo_refined,
        emit_retrieved_context,
        _PIPELINES_ROOT / "locomo-refined" / "pipeline.py",
    ),
    "longmemeval": BenchmarkSpec(
        _load_longmemeval,
        emit_retrieved_context,
        _PIPELINES_ROOT / "longmemeval-s" / "pipeline.py",
    ),
    "beam": BenchmarkSpec(
        _load_beam, emit_retrieved_context, _PIPELINES_ROOT / "beam" / "pipeline.py"
    ),
    "clbench": BenchmarkSpec(
        _load_clbench, emit_selected, _PIPELINES_ROOT / "clbench" / "pipeline.py"
    ),
    "personamem-v1": BenchmarkSpec(
        _load_personamem_v1,
        emit_context_messages,
        _PIPELINES_ROOT / "personamem" / "pipeline_v1.py",
    ),
    "personamem-v2": BenchmarkSpec(
        _load_personamem_v2,
        emit_chat_history,
        _PIPELINES_ROOT / "personamem" / "pipeline_v2.py",
    ),
}


class AmlRunManifest(ContractModel):
    """Pipeline provenance, deployment, and tenant identity for one AML run."""

    benchmark: NonEmptyString
    source_repository: NonEmptyString
    source_sha256: Sha256Hex
    deployment: DeploymentSnapshot
    # Naming matches `mindbridge.benchmarks.cli_common._core_manifest_values`:
    # every other benchmark manifest pins both a deployment and a predictions
    # hash so a reviewer can confirm exactly which deployment and which
    # output rows a manifest describes, not just a count (Cheap 4, final
    # review, 2026-08-17).
    deployment_sha256: Sha256Hex
    run_id: Identifier
    tenant_prefix: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    # Threaded from `--request-timeout-seconds` rather than hardcoded (Cheap
    # 5, final review, 2026-08-17): `driver.run_case` used to hardcode
    # `timeout=600.0` on both `/aml/add` and `/aml/search`, with no operator
    # override and nothing in the manifest recording what was used.
    request_timeout_seconds: float = Field(gt=0)
    # `gt=0`, not `ge=0`: a manifest describing zero questions is a run that
    # never happened (Important 2, task-13 review) -- `main()` also refuses
    # to get this far when a loader produces zero cases (see
    # `_require_nonempty_cases`), but this is the last-resort guarantee that
    # no manifest, however it was built, can claim a run of nothing.
    question_count: int = Field(gt=0)
    # How many of `question_count` carry CL-Bench's `question_unsliced`
    # marker (`loaders/clbench.py`): a question at least
    # `_OVERSIZED_QUESTION_CHARACTERS` long, whether or not the loader found
    # a blank-line break to slice the reference document from the actual
    # query. That both risks an oversized answer prompt and voids the
    # retrieval test for the affected row -- a caveat on this run's score,
    # not a memory-system weakness. Always 0 for every benchmark other than
    # CL-Bench, whose rows never carry the key. Counted from the rows this
    # run actually wrote to `--output` (including rows a resumed run already
    # had on disk), not by re-reading the source dataset.
    oversized_unsliced_question_count: int = Field(ge=0)
    # Computed client-side from `tenant_prefix` and each row's `user_id`
    # (see `_derive_tenant_id`) -- never confirmed against the deployment.
    # The server derives the same mapping independently from its own
    # `MINDBRIDGE_AML_TENANT_PREFIX`; nothing on the AML wire contract
    # carries or checks the prefix, so this map is only accurate if that
    # deployment's environment variable happens to equal the `--tenant-prefix`
    # this run was invoked with. Named `client_derived_...` rather than
    # `tenant_ids` so a reader of the manifest cannot mistake this for
    # something the server confirmed.
    client_derived_tenant_ids: dict[Identifier, Identifier]
    # Matches `_core_manifest_values`'s `predictions_sha256`: a hash of the
    # output file this manifest is the sidecar for, so a reviewer can confirm
    # the two haven't drifted apart.
    predictions_sha256: Sha256Hex
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    benchmark: str
    dataset_paths: tuple[Path, ...]
    output_path: Path
    api_base_url: str
    deployment_config_path: Path
    run_id: str
    tenant_prefix: str
    top_k: int
    concurrency: int
    request_timeout_seconds: float
    quiet: bool


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Load cases, replay them, and record predictions plus a manifest."""
    arguments = _parse_arguments(argv, prog)
    spec = BENCHMARKS[arguments.benchmark]
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    _require_compatible_existing_manifest(
        arguments.output_path,
        benchmark=arguments.benchmark,
        run_id=arguments.run_id,
        deployment=deployment.snapshot,
        recall_limit=arguments.top_k,
    )
    cases = spec.load(arguments.dataset_paths)
    _require_nonempty_cases(
        cases, benchmark=arguments.benchmark, dataset_paths=arguments.dataset_paths
    )
    api_key = os.environ.get("MINDBRIDGE_AML_API_KEY")
    if not api_key:
        raise ValueError("MINDBRIDGE_AML_API_KEY must be set")
    tenant_ids = asyncio.run(_connect_and_run(arguments, spec, cases, api_key))
    _write_manifest(arguments, deployment, tenant_ids)


def _require_compatible_existing_manifest(
    output_path: Path,
    *,
    benchmark: str,
    run_id: str,
    deployment: DeploymentSnapshot,
    recall_limit: int,
) -> None:
    """Refuse to resume into an `--output` whose sidecar manifest disagrees
    with this run's identity (Important 1, task-13 review).

    Row ids (`{case.user_id}#{question.question_id}`, see `_case_ids`) carry
    no `run_id`, `benchmark`, or deployment, so `_run`'s id-based resume
    would otherwise skip every case as already-done -- and `_write_manifest`
    would overwrite the sidecar unconditionally -- when a second invocation
    with a different `--run-id`, `--deployment-config`, or `--top-k` points
    at the same `--output`. That would attribute rows to a run that never
    produced them. Checked before any dataset is loaded or any case is run,
    so a mismatch fails fast. A first run (no sidecar yet) always passes.
    """
    manifest_path = sidecar_manifest_path(output_path)
    if not manifest_path.exists():
        return
    existing = AmlRunManifest.model_validate_json(manifest_path.read_bytes())
    disagreements: list[str] = []
    if existing.benchmark != benchmark:
        disagreements.append(f"benchmark ({existing.benchmark!r} != {benchmark!r})")
    if existing.run_id != run_id:
        disagreements.append(f"run_id ({existing.run_id!r} != {run_id!r})")
    if existing.deployment != deployment:
        disagreements.append("deployment (deployment snapshot differs)")
    if existing.recall_limit != recall_limit:
        disagreements.append(f"recall_limit ({existing.recall_limit} != {recall_limit})")
    if disagreements:
        raise ValueError(
            f"existing manifest at {manifest_path} disagrees with this run: "
            + "; ".join(disagreements)
        )


def _require_nonempty_cases(
    cases: tuple[AmlCase, ...], *, benchmark: str, dataset_paths: tuple[Path, ...]
) -> None:
    """Refuse to write a manifest for a run that replayed zero cases
    (Important 2, task-13 review).

    The realistic trigger is a `--dataset` path pointed one directory too
    high or too low: e.g. BEAM's loader (`chats_root.glob("*/*/chat.json")`)
    silently yields nothing rather than erroring when the root is wrong.
    Without this check the CLI would go on to write a complete-looking
    manifest for zero questions. Naming the resolved dataset path in the
    message is the point: the operator learns their path was wrong instead
    of getting a clean, official-looking empty manifest.
    """
    if cases:
        return
    resolved = ", ".join(str(path.resolve()) for path in dataset_paths)
    raise ValueError(f"{benchmark} loader produced zero cases from --dataset {resolved}")


async def _connect_and_run(
    arguments: _Arguments,
    spec: BenchmarkSpec,
    cases: tuple[AmlCase, ...],
    api_key: str,
) -> dict[str, str]:
    async with httpx.AsyncClient(
        base_url=arguments.api_base_url,
        headers={"Authorization": f"Bearer {api_key}"},
    ) as client:
        return await _run(
            client,
            spec,
            cases,
            run_id=arguments.run_id,
            benchmark=arguments.benchmark,
            top_k=arguments.top_k,
            concurrency=arguments.concurrency,
            output_path=arguments.output_path,
            tenant_prefix=arguments.tenant_prefix,
            request_timeout_seconds=arguments.request_timeout_seconds,
            quiet=arguments.quiet,
        )


async def _run(
    client: httpx.AsyncClient,
    spec: BenchmarkSpec,
    cases: tuple[AmlCase, ...],
    *,
    run_id: str,
    benchmark: str,
    top_k: int,
    concurrency: int,
    output_path: Path,
    tenant_prefix: str,
    request_timeout_seconds: float = 600.0,
    quiet: bool = False,
) -> dict[str, str]:
    """Run every not-yet-finished case and append its rows; resumable by id.

    A case is skipped entirely -- no `/aml/add`, no `/aml/search` -- once every
    row it would produce is already in `output_path`: `run_case` is not
    idempotent (each `/aml/add` call writes new memories), so re-running a
    finished case would duplicate memories server-side, not just output rows.
    One shared semaphore bounds `concurrency` requests in flight, so the bound
    is on load the server actually sees rather than on how that load is grouped
    into cases -- a single-case run parallelises its own chunks and questions
    instead of running strictly serially. A second bound admits only
    `concurrency` cases at a time, which is what keeps the resume guarantee
    below true: `run_case` gathers all of its adds before it issues a single
    search, so starting every case at once puts every case's adds ahead of any
    case's searches in the shared FIFO queue and no case can finish -- and
    therefore no row can be written -- until nearly the whole run's add phase
    is done. A kill before that point left an empty output file even though
    most `/aml/add` writes had landed, and because the rerun re-adds every
    chunk it silently doubled the corpus it then scored against.

    Rows are appended (de-duplicated by id) as each case's task completes, so a
    killed run leaves a prefix a rerun can resume from exactly where it stopped.
    """
    existing_ids = _read_existing_ids(output_path)
    pending = tuple(case for case in cases if not _case_ids(case) <= existing_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    case_slots = asyncio.Semaphore(concurrency)

    async def run_bounded_case(case: AmlCase) -> list[dict[str, object]]:
        async with case_slots:
            return await run_case(
                client,
                case,
                run_id=run_id,
                benchmark=benchmark,
                top_k=top_k,
                emit=spec.emit,
                semaphore=semaphore,
                timeout=request_timeout_seconds,
            )

    report(f"replaying {len(pending)} of {len(cases)} cases", quiet=quiet)
    tasks = [asyncio.create_task(run_bounded_case(case)) for case in pending]
    with output_path.open("a", encoding="utf-8") as handle:
        for index, task in enumerate(asyncio.as_completed(tasks), start=1):
            rows = await task
            report_unit("case complete", index=index, total=len(pending), quiet=quiet)
            for row in rows:
                written_id = str(row["id"])
                if written_id in existing_ids:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing_ids.add(written_id)

    tenant_ids: dict[str, str] = {}
    for case in cases:
        user_id = eval_user_id(run_id, benchmark, case.user_id)
        tenant_ids[user_id] = _derive_tenant_id(tenant_prefix, user_id)
    return tenant_ids


def _case_ids(case: AmlCase) -> set[str]:
    # Delegates to `row_id` (driver.py) rather than reconstructing the id
    # format inline: `run_case` doesn't always write
    # `{case.user_id}#{question.question_id}` -- every loader puts its own
    # "id" in the question payload, which overwrites it. Blocking 2 (final
    # review, 2026-08-17): this function used to hardcode the driver's id
    # format, which never matched what those benchmarks actually wrote, so
    # `_run`'s pending-work check never saw a finished case as done and
    # resumed runs silently re-added and re-searched everything.
    return {row_id(case, question) for question in case.questions}


def _read_output_rows(output_path: Path) -> list[dict[str, object]]:
    if not output_path.exists():
        return []
    rows: list[dict[str, object]] = []
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _read_existing_ids(output_path: Path) -> set[str]:
    return {str(row["id"]) for row in _read_output_rows(output_path)}


def _question_counts(output_path: Path) -> tuple[int, int]:
    """Return `(question_count, oversized_unsliced_question_count)` for this
    run's cumulative output. Reads `output_path` itself -- the rows the run
    actually produced, including any a resumed run already had on disk --
    rather than recomputing anything from the source dataset.
    """
    rows = _read_output_rows(output_path)
    oversized_unsliced = sum(1 for row in rows if row.get("question_unsliced") is True)
    return len(rows), oversized_unsliced


def _write_manifest(
    arguments: _Arguments,
    deployment: LoadedDeployment,
    tenant_ids: dict[str, str],
) -> None:
    question_count, oversized_unsliced_question_count = _question_counts(arguments.output_path)
    manifest = AmlRunManifest(
        benchmark=arguments.benchmark,
        source_repository=_AML_SOURCE_REPOSITORY,
        source_sha256=sha256_file(BENCHMARKS[arguments.benchmark].pipeline),
        deployment=deployment.snapshot,
        deployment_sha256=deployment.sha256,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        recall_limit=arguments.top_k,
        request_concurrency=arguments.concurrency,
        request_timeout_seconds=arguments.request_timeout_seconds,
        question_count=question_count,
        oversized_unsliced_question_count=oversized_unsliced_question_count,
        client_derived_tenant_ids=tenant_ids,
        predictions_sha256=sha256_file(arguments.output_path),
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _positive_int(raw: str) -> int:
    """Argparse `type=` for `--concurrency`: reject non-positive values at
    parse time (Minor 4, task-13 review). `asyncio.Semaphore(0)` blocks
    forever with no output and no error, and the `Field(gt=0)` that would
    otherwise catch it only runs once at manifest-write time -- after the
    run has already hung.
    """
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = build_parser(
        prog=prog,
        description="Run one Agent Memory Leaderboard benchmark against a deployed API.",
        epilog=AML_ENVIRONMENT,
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARKS),
        help="which AML pipeline to replay",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="one per positional argument the benchmark's loader takes, in order",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="JSONL rows to append to; a resumed run keeps the rows already there",
    )
    parser.add_argument(
        "--api-base-url", required=True, help="base URL of the deployed MindBridge API"
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        required=True,
        help="JSON description of the deployment that answered",
    )
    parser.add_argument(
        "--run-id", required=True, help="identifier this run's tenants and manifest are pinned to"
    )
    # No default: the server derives its half of the same mapping from its
    # own `MINDBRIDGE_AML_TENANT_PREFIX` (see `AmlRunManifest`'s
    # `client_derived_tenant_ids` docstring), which nothing on the wire
    # confirms. A default here would let an operator silently inherit a
    # value the deployment may not share; requiring it explicitly forces
    # that choice to be made on purpose (Important 3, task-13 review).
    parser.add_argument(
        "--tenant-prefix",
        required=True,
        help="prefix the deployment agrees on; deliberately has no default",
    )
    parser.add_argument("--top-k", type=int, default=20, help="memories to retrieve per question")
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=4,
        help=(
            "requests in flight across every case; raising it needs "
            "MINDBRIDGE_DATABASE_MAX_POOL_SIZE raised with it (see README)"
        ),
    )
    # Matches `cli_common.core_parser`'s `--request-timeout-seconds`, threaded
    # through to `driver.run_case` instead of that module's hardcoded
    # `timeout=600.0` on every `/aml/add` and `/aml/search` call (Cheap 5,
    # final review, 2026-08-17), and pinned in the manifest like every other
    # benchmark runner already pins it.
    parser.add_argument(
        "--request-timeout-seconds", type=float, default=600.0, help="deadline for one request"
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the progress lines this replay writes to stderr",
    )
    parsed = parser.parse_args(argv)
    return _Arguments(
        benchmark=parsed.benchmark,
        dataset_paths=tuple(parsed.dataset),
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        top_k=parsed.top_k,
        concurrency=parsed.concurrency,
        request_timeout_seconds=parsed.request_timeout_seconds,
        quiet=parsed.quiet,
    )


if __name__ == "__main__":
    main()
