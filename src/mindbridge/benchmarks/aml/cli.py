"""Run one Agent Memory Leaderboard benchmark against a deployed MindBridge API.

Ties together the six benchmark-neutral loaders, `run_case` (Task 12), and the
six vendored AML scoring pipelines (Task 13). This module -- like every module
under `mindbridge.benchmarks.aml` -- must not import `mindbridge.api`, so it
stays usable against a real deployed server rather than only an in-process one.
"""

from __future__ import annotations

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
from mindbridge.benchmarks.cli_common import core_parser, limit_units, report, report_unit
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex
from mindbridge.file_integrity import sha256_file

# The upstream this benchmark's vendored pipelines come from, recorded in
# `benchmarks/aml/PINNED.md`. A constant, not a CLI flag: changing it means
# re-vendoring, not choosing a different value per run.
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
                          environment so a recorded invocation never carries it
  MINDBRIDGE_AML_TENANT_PREFIX
                          default for --tenant-prefix; the same variable the deployment
                          derives its half of the mapping from, so setting it once makes
                          the two agree by construction"""

TENANT_PREFIX_VARIABLE = "MINDBRIDGE_AML_TENANT_PREFIX"
"""Where `--tenant-prefix` falls back to, and the only default it has.

`--tenant-prefix` deliberately had no default at all (Important 3, task-13 review): the
deployment derives its half of the tenant mapping from its own
`MINDBRIDGE_AML_TENANT_PREFIX`, nothing on the AML wire contract confirms the two agree,
and a literal default here would let an operator silently inherit a value the deployment
does not share. That reasoning is unchanged -- which is why the fallback is *that same
variable* rather than a constant. Unset, with no flag, is still a refusal.

It exists because `mindbridge-bench eval` forwards a fixed set of flags to every task and
`--tenant-prefix` is not one of them, so without a default no AML task could be swept.
Putting the value in the task catalog instead would have been a hardcoded default wearing
a different hat.
"""

SCRIPTMEM_BENCHMARK = "scriptmem"
"""AML's seventh textual benchmark: a `--benchmark` choice that always refuses.

Offered rather than omitted so the refusal below is what an operator gets, and so
`--help` lists the same seven benchmarks AML's own board does.
"""

SCRIPTMEM_UNSUPPORTED = (
    "scriptmem is not runnable offline: its public release ships questions, gold answers "
    "and the scorer, but every `conversation` field in data/public/conversations.jsonl and "
    "data/raw/*.json holds only a {'format_example': ...} placeholder -- the four source "
    "scripts are not distributed, so there is no corpus to ingest and a score would measure "
    "retrieval over synthetic filler. A real AML submission is unaffected; the platform runs "
    "ScriptMem server-side against its own copy."
)
"""Why the seventh AML benchmark is absent, said where an operator meets it.

Named and refused rather than merely missing from `BENCHMARKS`. Left out, `--benchmark
scriptmem` was an argparse "invalid choice" indistinguishable from a typo, and the reason
lived only in `docs/superpowers/specs/2026-08-17-aml-dataset-schemas.md`. An operator who
has the real scripts is not blocked by anything here except a loader; one who does not is
owed the reason rather than a silent gap.
"""


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
    recall_limit: int
    request_concurrency: int
    request_timeout_seconds: float
    limit: int | None
    overwrite: bool
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
        recall_limit=arguments.recall_limit,
        overwrite=arguments.overwrite,
    )
    cases = spec.load(arguments.dataset_paths)
    _require_nonempty_cases(
        cases, benchmark=arguments.benchmark, dataset_paths=arguments.dataset_paths
    )
    # After `_require_nonempty_cases`, so a `--dataset` pointed at the wrong directory is still
    # reported as the wrong path rather than as an empty selection the limit happened to produce.
    cases = limit_units(cases, arguments.limit, label=f"{arguments.benchmark} cases")
    api_key = os.environ.get("MINDBRIDGE_AML_API_KEY")
    if not api_key:
        raise ValueError("MINDBRIDGE_AML_API_KEY must be set")
    # Last thing before the first write, and deliberately after every refusal above. Discarding
    # earlier -- next to the manifest check that motivates it -- meant an `--overwrite` run with
    # an unset API key or a mistyped `--dataset` destroyed the previous run's predictions and
    # then failed without producing any of its own.
    if arguments.overwrite:
        _discard_previous_run(arguments.output_path)
    tenant_ids = asyncio.run(_connect_and_run(arguments, spec, cases, api_key))
    _write_manifest(arguments, deployment, tenant_ids)


def _require_compatible_existing_manifest(
    output_path: Path,
    *,
    benchmark: str,
    run_id: str,
    deployment: DeploymentSnapshot,
    recall_limit: int,
    overwrite: bool = False,
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
    if overwrite and not disagreements:
        # `--overwrite` cannot mean "start this run over": deleting the rows does not unwrite
        # the `/aml/add` calls that produced them, and the tenant is derived from `--run-id`, so
        # re-adding lands in the same tenant and doubles the corpus every later search is scored
        # against. Silently, and only in the score. Resuming is what this case wants, and it is
        # what dropping the flag does.
        raise ValueError(
            f"{manifest_path} already describes this exact run, and --overwrite would re-add "
            "every case into a tenant that already holds its memories, doubling the corpus this "
            "run is then scored against; drop --overwrite to resume where it stopped, or pass a "
            "new --run-id to measure a clean tenant"
        )
    if overwrite:
        # A manifest from a different run, which is what the flag is for.
        return
    if disagreements:
        raise ValueError(
            f"existing manifest at {manifest_path} disagrees with this run: "
            + "; ".join(disagreements)
        )


def _discard_previous_run(output_path: Path) -> None:
    """Throw away an earlier run's rows and manifest so this one starts from nothing.

    Every other runner's `--overwrite` replaces its predictions; this one appends, because a
    resumed AML run must not re-issue `/aml/add` for a case it already finished. Those are not
    in conflict, but only if `--overwrite` deletes rather than truncates-in-place: `_run`
    resumes off the ids already in the file, so leaving the rows there would make an
    "overwriting" run skip exactly the cases it was asked to redo.

    The manifest goes with them. Left behind, `_require_compatible_existing_manifest` would
    still refuse the very run `--overwrite` exists to allow -- a re-run under a new `--run-id`
    into the same output path, which is the whole reason an operator reaches for the flag.
    """
    output_path.unlink(missing_ok=True)
    sidecar_manifest_path(output_path).unlink(missing_ok=True)


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
            top_k=arguments.recall_limit,
            concurrency=arguments.request_concurrency,
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
                # `ensure_ascii=True` here, against this repo's usual
                # `ensure_ascii=False`: `docs/benchmarking.md` feeds these shards to
                # the vendored scorers, which are standalone upstream scripts, and to
                # third-party tooling. A raw U+2028, U+2029 or U+0085 -- the only
                # three `splitlines()` breaks on that `ensure_ascii=False` emits
                # unescaped -- shreds a record mid-string in any reader that splits
                # on Unicode line boundaries. Escaping puts that guarantee on the
                # write side instead of assuming every reader is fixed. It costs
                # about 1% here because all seven AML corpora are 0.00% CJK; the
                # CJK-heavy benchmarks write through their own CLIs, which keep
                # `ensure_ascii=False` and stay readable.
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
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
        recall_limit=arguments.recall_limit,
        request_concurrency=arguments.request_concurrency,
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


MAX_RECALL_LIMIT = 100
"""The largest `--recall-limit` both the wire contract and the manifest accept.

Duplicated from `mindbridge.api.aml_contracts.AmlSearchRequest.top_k` (`ge=1, le=100`) rather
than imported, for the reason in this module's docstring: nothing under
`mindbridge.benchmarks.aml` may import `mindbridge.api`. `AmlRunManifest.recall_limit` below
carries the same bound independently. `tests/unit/benchmarks/aml/test_cli.py` asserts all three
agree, so the copy cannot drift silently.
"""


def _require_usable_recall_limit(value: int) -> int:
    """Reject a `--recall-limit` the deployment would refuse, at parse time.

    Both bounds that would otherwise catch this fire too late to be useful. The server rejects
    every `/aml/search` with `422`, and `AmlRunManifest` raises only once the run is over -- so
    with `--overwrite` the previous run's predictions are already gone, the add phase has already
    written a full corpus server-side, and what remains is an invalid predictions file with no
    sidecar where a finished result used to be. Checked here so nothing is deleted or ingested.
    """
    if not 1 <= value <= MAX_RECALL_LIMIT:
        raise ValueError(f"--recall-limit must be between 1 and {MAX_RECALL_LIMIT}, got {value}")
    return value


def _require_positive_timeout(value: float) -> float:
    """Reject a `--request-timeout-seconds` the manifest would refuse after the run.

    Same failure shape as `_require_usable_recall_limit`: `AmlRunManifest`'s `Field(gt=0)` is the
    only other guard, and it runs last.
    """
    if value <= 0:
        raise ValueError(f"--request-timeout-seconds must be positive, got {value}")
    return value


def _require_positive_concurrency(value: int) -> int:
    """Refuse a non-positive `--request-concurrency` before it can hang the run.

    `asyncio.Semaphore(0)` blocks forever with no output and no error, and the `Field(gt=0)`
    that would otherwise catch it only runs at manifest-write time -- after the run has
    already hung (Minor 4, task-13 review). Checked after parsing rather than as argparse
    `type=`, because the flag is declared once in `core_parser` for every runner and this is
    the only one that deadlocks rather than merely misbehaves on zero.
    """
    if value <= 0:
        raise ValueError(f"--request-concurrency must be a positive integer, got {value}")
    return value


def _require_tenant_prefix(parsed_value: str | None) -> str:
    """Resolve `--tenant-prefix`, or refuse the run naming both ways to supply it."""
    resolved = parsed_value or os.environ.get(TENANT_PREFIX_VARIABLE)
    if not resolved:
        raise ValueError(
            f"--tenant-prefix is required; pass it or set {TENANT_PREFIX_VARIABLE} to the "
            "prefix the deployment derives its own tenants from"
        )
    return resolved


def _require_supported_benchmark(name: str) -> str:
    """Say why ScriptMem is absent, rather than letting it read as a typo."""
    if name == SCRIPTMEM_BENCHMARK:
        raise ValueError(SCRIPTMEM_UNSUPPORTED)
    return name


def _parse_arguments(argv: Sequence[str] | None, prog: str | None) -> _Arguments:
    parser = core_parser(
        # `None`, not "": there is no default to fall back to here, and an empty string is a
        # value -- the shared help formatter would print `(default: )` and the resolution below
        # would be unreachable. `_require_tenant_prefix` supplies the environment fallback or
        # refuses.
        tenant_prefix=None,
        prog=prog,
        description="Run one Agent Memory Leaderboard benchmark against a deployed API.",
        epilog=AML_ENVIRONMENT,
        dataset_action="append",
        dataset_help="one per positional argument the benchmark's loader takes, in order",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted((*BENCHMARKS, SCRIPTMEM_BENCHMARK)),
        help="which AML pipeline to replay",
    )
    parsed = parser.parse_args(argv)
    return _Arguments(
        benchmark=_require_supported_benchmark(parsed.benchmark),
        dataset_paths=tuple(parsed.dataset),
        output_path=parsed.output,
        api_base_url=parsed.api_base_url,
        deployment_config_path=parsed.deployment_config,
        run_id=parsed.run_id,
        tenant_prefix=_require_tenant_prefix(parsed.tenant_prefix),
        recall_limit=_require_usable_recall_limit(parsed.recall_limit),
        request_concurrency=_require_positive_concurrency(parsed.request_concurrency),
        request_timeout_seconds=_require_positive_timeout(parsed.request_timeout_seconds),
        limit=parsed.limit,
        overwrite=parsed.overwrite,
        quiet=parsed.quiet,
    )


if __name__ == "__main__":
    main()
