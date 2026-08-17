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
    run_case,
)
from mindbridge.benchmarks.aml.loaders import (
    beam,
    clbench,
    locomo,
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
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file

# The exact pin recorded in `benchmarks/aml/PINNED.md` (Task 13, steps 1-2).
# A constant, not a CLI flag: upgrading it means re-vendoring and re-pinning,
# not choosing a different value per run.
_AML_SOURCE_REPOSITORY = "AML-memory/agent-memory-leaderboard"
_AML_SOURCE_REVISION = "5761ed58502d24153115cbdc010e44957cb18c3a"

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


def _load_locomo(paths: Sequence[Path]) -> tuple[AmlCase, ...]:
    return locomo.load(_one(paths))


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


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "locomo": BenchmarkSpec(
        _load_locomo, emit_retrieved_context, _PIPELINES_ROOT / "locomo-refined" / "pipeline.py"
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
    source_revision: NonEmptyString
    source_sha256: Sha256Hex
    code_revision: NonEmptyString
    deployment: DeploymentSnapshot
    run_id: Identifier
    tenant_prefix: Identifier
    recall_limit: int = Field(gt=0, le=100)
    request_concurrency: int = Field(gt=0)
    tenant_ids: dict[Identifier, Identifier]
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    benchmark: str
    dataset_paths: tuple[Path, ...]
    output_path: Path
    api_base_url: str
    code_revision: str
    deployment_config_path: Path
    run_id: str
    tenant_prefix: str
    top_k: int
    concurrency: int


def main() -> None:
    """Load cases, replay them, and record predictions plus a manifest."""
    arguments = _parse_arguments()
    spec = BENCHMARKS[arguments.benchmark]
    cases = spec.load(arguments.dataset_paths)
    deployment = load_deployment_snapshot(arguments.deployment_config_path)
    api_key = os.environ.get("MINDBRIDGE_AML_API_KEY")
    if not api_key:
        raise SystemExit("MINDBRIDGE_AML_API_KEY must be set")
    tenant_ids = asyncio.run(_connect_and_run(arguments, spec, cases, api_key))
    _write_manifest(arguments, deployment, tenant_ids)


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
) -> dict[str, str]:
    """Run every not-yet-finished case and append its rows; resumable by id.

    A case is skipped entirely -- no `/aml/add`, no `/aml/search` -- once every
    row it would produce is already in `output_path`: `run_case` is not
    idempotent (each `/aml/add` call writes new memories), so re-running a
    finished case would duplicate memories server-side, not just output rows.
    Chunks stay serial inside `run_case`; cases run concurrently up to
    `concurrency`, and rows are appended (de-duplicated by id) as each case's
    task completes, so a killed run leaves a prefix a rerun can resume from
    exactly where it stopped.
    """
    existing_ids = _read_existing_ids(output_path)
    pending = tuple(case for case in cases if not _case_ids(case) <= existing_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(case: AmlCase) -> list[dict[str, object]]:
        async with semaphore:
            return await run_case(
                client,
                case,
                run_id=run_id,
                benchmark=benchmark,
                top_k=top_k,
                emit=spec.emit,
            )

    tasks = [asyncio.create_task(bounded(case)) for case in pending]
    with output_path.open("a", encoding="utf-8") as handle:
        for task in asyncio.as_completed(tasks):
            for row in await task:
                row_id = str(row["id"])
                if row_id in existing_ids:
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing_ids.add(row_id)

    tenant_ids: dict[str, str] = {}
    for case in cases:
        user_id = eval_user_id(run_id, benchmark, case.user_id)
        tenant_ids[user_id] = _derive_tenant_id(tenant_prefix, user_id)
    return tenant_ids


def _case_ids(case: AmlCase) -> set[str]:
    return {f"{case.user_id}#{question.question_id}" for question in case.questions}


def _read_existing_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            ids.add(str(json.loads(stripped)["id"]))
    return ids


def _write_manifest(
    arguments: _Arguments,
    deployment: LoadedDeployment,
    tenant_ids: dict[str, str],
) -> None:
    manifest = AmlRunManifest(
        benchmark=arguments.benchmark,
        source_repository=_AML_SOURCE_REPOSITORY,
        source_revision=_AML_SOURCE_REVISION,
        source_sha256=sha256_file(BENCHMARKS[arguments.benchmark].pipeline),
        code_revision=arguments.code_revision,
        deployment=deployment.snapshot,
        run_id=arguments.run_id,
        tenant_prefix=arguments.tenant_prefix,
        recall_limit=arguments.top_k,
        request_concurrency=arguments.concurrency,
        tenant_ids=tenant_ids,
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(
        sidecar_manifest_path(arguments.output_path),
        manifest.model_dump_json(indent=2) + "\n",
    )


def _parse_arguments() -> _Arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="one per positional argument the benchmark's loader takes, in order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tenant-prefix", default="bench_aml")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parsed = parser.parse_args()
    return _Arguments(
        benchmark=parsed.benchmark,
        dataset_paths=tuple(parsed.dataset),
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        code_revision=parsed.code_revision,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=parsed.tenant_prefix,
        top_k=parsed.top_k,
        concurrency=parsed.concurrency,
    )


if __name__ == "__main__":
    main()
