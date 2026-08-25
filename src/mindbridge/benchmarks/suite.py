"""Run several official benchmarks from one invocation, in the order a suite file lists them.

A suite file exists because a benchmark's name does not determine its invocation. Every runner
needs its own release path, and most need a required choice no sweep can guess -- ATM-Bench's
`--split`, MEMLENS's `--context-window`, M3-Bench's `--subset`, the prepared-media manifest an
operator produced outside MindBridge. The suite records one task per invocation; the sweep
supplies only the flags every runner shares.

The sweep owns two of those per task, so a suite may not set them. `--output` is `--output-dir`
plus the task's own file name. `--run-id` is the sweep's ID plus the task's name, which is not
cosmetic: a tenant is derived from `--tenant-prefix`, the unit ID, and `--run-id`, and a
benchmark run twice in one sweep -- MEMLENS at two context windows, ATM-Bench at both splits --
shares the first two. Without a per-task run ID the second task would write into the first
task's tenants and then answer from its memories.

Tasks run one at a time, and the sweep continues past one that fails, so a benchmark dying four
hours in costs its own result rather than the whole sweep. An interrupt is the exception: it
stops the sweep and still writes the summary. Every outcome lands in `suite-summary.json` beside
the predictions with the exit code and the argv behind it.

Each task's predictions and manifest are written by its runner, unchanged. Nothing here scores
anything -- an official scorer's verdict is attached to one run by `mindbridge-bench score`, and
a sweep summary carrying numbers would be claiming what no runner here measured.
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from mindbridge.benchmarks.artifacts import require_writable_output_pair, write_text_atomically
from mindbridge.benchmarks.cli import INTERRUPT_EXIT_CODE, PROGRAM, RUNNERS, guarded
from mindbridge.benchmarks.cli import parser as build_parser
from mindbridge.benchmarks.cli_common import (
    BENCHMARK_ENVIRONMENT,
    report,
    report_unit,
    select_by_id,
)
from mindbridge.benchmarks.task_catalog import (
    DEFAULT_BENCHMARKS_ROOT,
    listing,
    task_payloads,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex

SUITE_VERSION = "benchmark_suite_v1"
"""Schema identity of the summary this sweep writes, pinned like every other artifact here."""

SUMMARY_FILENAME = "suite-summary.json"
"""What the sweep's own artifact is called inside `--output-dir`."""

SWEEP_OWNED_FLAGS = frozenset({"-o", "--output", "--run-id"})
"""Flags the sweep derives per task.

They are appended last, so argparse would let them win over a suite that set them too. Refusing
the suite outright is the difference between a run whose output went somewhere else and a run
that never started.
"""

_TaskName = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=64),
]
"""A single path component, because a task's name becomes a file name inside `--output-dir`."""


class SuiteTask(ContractModel):
    """One benchmark invocation, named so two parameterisations of it stay distinct."""

    name: _TaskName
    benchmark: NonEmptyString
    arguments: tuple[NonEmptyString, ...] = ()
    """The runner's own flags, verbatim.

    Argv rather than a mapping so a runner's flag is never described twice: a repeatable
    `--video-id`, a bare `--text-only`, and a flag added tomorrow all pass through unchanged, and
    the runner's own parser is what validates them.
    """
    output_name: _TaskName | None = None
    """Overrides the default `<name>.jsonl`, which is wrong for the runners emitting JSON."""

    @model_validator(mode="after")
    def require_a_dispatchable_invocation(self) -> SuiteTask:
        """Reject a task the sweep could only fail on, before any benchmark starts."""
        if self.benchmark not in RUNNERS:
            known = ", ".join(sorted(RUNNERS))
            raise ValueError(f"unknown benchmark: {self.benchmark}; known benchmarks are {known}")
        owned = sorted(
            {
                flag
                for argument in self.arguments
                if (flag := argument.partition("=")[0]) in SWEEP_OWNED_FLAGS
            }
        )
        if owned:
            raise ValueError(
                f"the sweep derives {', '.join(owned)} for every task; remove it from the suite"
            )
        return self

    def predictions_name(self) -> str:
        """Name this task's predictions file inside the sweep's output directory."""
        return self.output_name or f"{self.name}.jsonl"


class BenchmarkSuite(ContractModel):
    """The benchmarks one sweep can run, in the order it runs them."""

    tasks: tuple[SuiteTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_tasks_that_cannot_overwrite_each_other(self) -> BenchmarkSuite:
        """Refuse a suite in which two tasks share a run ID or a predictions file.

        Both are silent when they happen and expensive to detect afterwards: shared run IDs put
        two tasks in one tenant, and a shared file leaves only the last task's predictions with
        every earlier task still reported as having produced a result.
        """
        _reject_repeats((task.name for task in self.tasks), "suite task names must be unique")
        _reject_repeats(
            (task.predictions_name() for task in self.tasks),
            "suite tasks must not share a predictions file",
        )
        return self


class SuiteTaskOutcome(ContractModel):
    """What one task did, where it wrote, and the invocation that produced it."""

    name: Identifier
    benchmark: NonEmptyString
    run_id: Identifier
    output_path: NonEmptyString
    arguments: tuple[NonEmptyString, ...]
    status: Literal["completed", "failed"]
    exit_code: int
    duration_seconds: float = Field(ge=0)


class SuiteRunSummary(ContractModel):
    """One sweep's roster: which benchmarks it ran and which of them produced a result.

    `task_count` is what the sweep selected and `tasks` is what it attempted, so an interrupted
    sweep is the case where the two differ and the remainder never started.
    """

    suite_version: NonEmptyString = SUITE_VERSION
    suite_sha256: Sha256Hex
    run_id: Identifier
    task_count: int = Field(gt=0)
    completed_task_count: int = Field(ge=0)
    failed_task_count: int = Field(ge=0)
    tasks: tuple[SuiteTaskOutcome, ...] = Field(min_length=1)
    started_at: AwareDatetime
    completed_at: AwareDatetime


@dataclass(frozen=True, slots=True)
class _Arguments:
    """The sweep's own flags, plus the ones it forwards to every task verbatim."""

    task_names: tuple[str, ...]
    suite_path: Path | None
    benchmarks_root: Path
    output_dir: Path
    run_id: str
    shared: tuple[str, ...]
    overwrite: bool
    quiet: bool
    dry_run: bool


@dataclass(frozen=True, slots=True)
class _Plan:
    """One task's resolved invocation, fixed before any benchmark runs."""

    task: SuiteTask
    run_id: str
    output_path: Path
    arguments: tuple[str, ...]


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Run every named benchmark in catalog order and record what each of them did."""
    parser = _build_parser(prog)
    parsed = parser.parse_args(argv)
    if parsed.list_tasks:
        print(listing(root=parsed.benchmarks_root))
        return 0
    arguments = _arguments(parser, parsed)
    suite = _load_suite(arguments)
    plans = tuple(_plan(task, arguments) for task in suite.tasks)
    if arguments.dry_run:
        _print_plan(plans)
        return 0
    summary_path = arguments.output_dir / SUMMARY_FILENAME
    _require_writable_outputs(plans, summary_path, overwrite=arguments.overwrite)
    report(f"running {len(plans)} benchmarks", quiet=arguments.quiet)
    started_at = datetime.now(timezone.utc)
    outcomes = _run_plans(plans, quiet=arguments.quiet)
    summary = SuiteRunSummary(
        suite_sha256=hashlib.sha256(suite.model_dump_json().encode("utf-8")).hexdigest(),
        run_id=arguments.run_id,
        task_count=len(plans),
        completed_task_count=sum(outcome.status == "completed" for outcome in outcomes),
        failed_task_count=sum(outcome.status == "failed" for outcome in outcomes),
        tasks=outcomes,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
    )
    write_text_atomically(summary_path, summary.model_dump_json(indent=2) + "\n")
    report(f"wrote {summary_path}", quiet=arguments.quiet)
    return _sweep_exit_code(outcomes)


def _load_suite(arguments: _Arguments) -> BenchmarkSuite:
    """Build the tasks to run, from the catalog or from a suite file.

    Both go through the same model, so a hand-written suite is checked exactly as strictly as a
    catalog entry: an unknown benchmark, a duplicate name, or a flag the sweep owns is refused
    before the first task starts either way.
    """
    if arguments.suite_path is None:
        return BenchmarkSuite.model_validate(
            {"tasks": task_payloads(arguments.task_names, root=arguments.benchmarks_root)}
        )
    suite = BenchmarkSuite.model_validate_json(arguments.suite_path.read_bytes())
    selected = select_by_id(
        suite.tasks,
        arguments.task_names,
        key=lambda task: task.name,
        label="suite task names",
    )
    return BenchmarkSuite(tasks=selected)


def _run_plans(plans: tuple[_Plan, ...], *, quiet: bool) -> tuple[SuiteTaskOutcome, ...]:
    """Run the planned tasks one at a time, stopping only for an interrupt."""
    outcomes: list[SuiteTaskOutcome] = []
    for index, plan in enumerate(plans, start=1):
        report_unit(
            f"{plan.task.benchmark} as {plan.task.name}",
            index=index,
            total=len(plans),
            quiet=quiet,
        )
        outcome = _run_task(plan)
        outcomes.append(outcome)
        report(
            f"{plan.task.name} {outcome.status} in {outcome.duration_seconds:.1f}s",
            quiet=quiet,
        )
        if outcome.exit_code == INTERRUPT_EXIT_CODE:
            report("interrupted; the remaining benchmarks did not start", quiet=quiet)
            break
    return tuple(outcomes)


def _run_task(plan: _Plan) -> SuiteTaskOutcome:
    """Run one benchmark under the dispatcher's own failure contract, and time it.

    `guarded` reduces an exception to an exit code, but not `SystemExit`: a runner whose flags
    argparse rejects raises that, and letting it through would end the sweep at the first
    mistyped task and take the summary of everything before it along.
    """
    _reset_run_timings()
    started = time.monotonic()
    try:
        exit_code = guarded(
            RUNNERS[plan.task.benchmark],
            plan.arguments,
            prog=f"{PROGRAM} {plan.task.benchmark} [{plan.task.name}]",
        )
    except SystemExit as request:
        exit_code = _requested_exit_code(request)
    return SuiteTaskOutcome(
        name=plan.task.name,
        benchmark=plan.task.benchmark,
        run_id=plan.run_id,
        output_path=str(plan.output_path),
        arguments=plan.arguments,
        status=_task_status(plan, exit_code),
        exit_code=exit_code,
        duration_seconds=time.monotonic() - started,
    )


def _task_status(plan: _Plan, exit_code: int) -> Literal["completed", "failed"]:
    """Call a task complete only once its predictions exist, not merely because it exited 0.

    The two can disagree -- a task whose arguments make its runner print help and exit 0 is the
    reachable case -- and a summary reporting a result nobody can open is the one failure this
    artifact exists to make visible.
    """
    if exit_code != 0:
        return "failed"
    if plan.output_path.exists():
        return "completed"
    print(
        f"{PROGRAM}: error: {plan.task.name} exited 0 without writing {plan.output_path}",
        file=sys.stderr,
    )
    return "failed"


def _requested_exit_code(request: SystemExit) -> int:
    """Read the code out of a runner's `SystemExit`, however it chose to raise it."""
    code = request.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(f"{PROGRAM}: error: {code}", file=sys.stderr)
    return 1


def _sweep_exit_code(outcomes: Sequence[SuiteTaskOutcome]) -> int:
    """Report the sweep as interrupted, failed, or complete, in that order of precedence.

    Read off each task's status rather than its code, so a task that exited 0 without leaving
    predictions cannot report a clean sweep from inside a summary that calls it failed.
    """
    if any(outcome.exit_code == INTERRUPT_EXIT_CODE for outcome in outcomes):
        return INTERRUPT_EXIT_CODE
    return 1 if any(outcome.status == "failed" for outcome in outcomes) else 0


def _plan(task: SuiteTask, arguments: _Arguments) -> _Plan:
    """Resolve one task's invocation, with the flags the sweep owns appended last."""
    run_id = f"{arguments.run_id}-{task.name}"
    output_path = arguments.output_dir / task.predictions_name()
    return _Plan(
        task=task,
        run_id=run_id,
        output_path=output_path,
        arguments=(
            *arguments.shared,
            *task.arguments,
            "--output",
            str(output_path),
            "--run-id",
            run_id,
        ),
    )


def _require_writable_outputs(
    plans: Sequence[_Plan],
    summary_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Refuse the whole sweep before it starts if any of its outputs already exist.

    Each runner makes this check for itself, but only once it is running. Finding out at the
    fourth benchmark that the fifth cannot write is what this exists to prevent.
    """
    for plan in plans:
        require_writable_output_pair(plan.output_path, overwrite=overwrite)
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {summary_path}")


def _print_plan(plans: Sequence[_Plan]) -> None:
    """Print the invocation behind each task, which is how a sweep's tenants are learned early.

    Every tenant a run writes to has to be authorized before the API starts, and the run ID the
    tenant is derived from is one the sweep makes up. Printing the argv is what makes that ID
    readable without starting a benchmark to find out.
    """
    for plan in plans:
        command = shlex.join([PROGRAM, plan.task.benchmark, *plan.arguments])
        print(f"{plan.task.name}\t{command}")


def _reset_run_timings() -> None:
    """Start each task's timing summary from zero, or do nothing where telemetry is absent.

    `guarded` logs the accumulated per-operation breakdown after every task, and the accumulator
    is process-wide. Without this the second task's ranked stages would include the first task's
    work, which is the one number about a task that nobody can reconstruct afterwards.
    """
    try:
        from mindbridge.telemetry import reset_timings
    except ImportError:  # pragma: no cover - only a benchmarks install without the API
        return
    reset_timings()


def _reject_repeats(values: Iterable[str], message: str) -> None:
    """Refuse a repeated value, naming every one of them rather than only the first."""
    seen = list(values)
    repeated = sorted({value for value in seen if seen.count(value) > 1})
    if repeated:
        raise ValueError(f"{message}: {', '.join(repeated)}")


def _build_parser(prog: str | None) -> argparse.ArgumentParser:
    """Build the sweep's parser, defaulting everything that has one sensible value.

    Only `--run-id` and the task names have no default. Everything else falls back to the layout
    `docs/benchmarking.md` sets up, which is what makes a smoke run one line. `--recall-limit` and
    its neighbours are deliberately left unset rather than copied from `core_parser`: an unset
    tunable is not forwarded at all, so each runner keeps the default it declares and this command
    cannot pin a stale copy of one.
    """
    parser = build_parser(prog=prog, description=__doc__, epilog=BENCHMARK_ENVIRONMENT)
    parser.add_argument(
        "--tasks",
        action="append",
        default=[],
        metavar="NAME[,NAME...]",
        help="catalog task or group to run; comma-separated and repeatable",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="print every name --tasks accepts, with whether its inputs are present, and exit",
    )
    parser.add_argument(
        "--run-id", help="identifier for this sweep; each task runs under <run-id>-<task name>"
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=DEFAULT_BENCHMARKS_ROOT,
        help="directory the official releases were downloaded into",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="where each task's predictions, manifest, and this sweep's "
        f"{SUMMARY_FILENAME} are written; defaults to <benchmarks-root>/results/<run-id>",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="base URL of the deployed MindBridge API to measure",
    )
    parser.add_argument(
        "--deployment-config",
        type=Path,
        help="JSON description of the deployment that answered, pinned into every manifest; "
        "defaults to <benchmarks-root>/deployment.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N of each benchmark's own units, for a smoke run",
    )
    parser.add_argument(
        "--recall-limit",
        type=int,
        help="memories to retrieve per question, for every task that does not set its own",
    )
    parser.add_argument(
        "--request-concurrency",
        type=int,
        help="in-flight API requests per unit, for every task that does not set its own",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        help="deadline for one request, for every task that does not set its own",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="run the tasks in this JSON file instead of the catalog's; --tasks then narrows it",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing predictions, manifests, and this sweep's summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the invocation each task would run, then exit without running any",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress the progress lines this sweep and its tasks write to stderr",
    )
    return parser


def _arguments(parser: argparse.ArgumentParser, parsed: argparse.Namespace) -> _Arguments:
    """Resolve the parsed flags, including the defaults that depend on other flags."""
    names = tuple(
        name.strip() for group in parsed.tasks for name in group.split(",") if name.strip()
    )
    if not names and parsed.suite is None:
        parser.error("give --tasks NAME[,NAME...], or --suite FILE; --list-tasks prints the names")
    if parsed.run_id is None:
        parser.error("--run-id is required; it is what isolates this sweep's tenants")
    root = parsed.benchmarks_root
    return _Arguments(
        task_names=names,
        suite_path=parsed.suite,
        benchmarks_root=root,
        output_dir=parsed.output_dir or root / "results" / parsed.run_id,
        run_id=parsed.run_id,
        shared=_shared_arguments(parsed, root=root),
        overwrite=parsed.overwrite,
        quiet=parsed.quiet,
        dry_run=parsed.dry_run,
    )


def _shared_arguments(parsed: argparse.Namespace, *, root: Path) -> tuple[str, ...]:
    """Collect the flags forwarded to every task, and only the ones that were given.

    Media knobs are deliberately absent: `--poll-interval-seconds` and its neighbours exist only
    on the runners that ingest media, so forwarding them to every task would make a text-only
    benchmark reject the sweep. A task that needs one carries it in its own arguments.
    """
    shared = [
        "--api-base-url",
        parsed.api_base_url,
        "--deployment-config",
        str(parsed.deployment_config or root / "deployment.json"),
    ]
    for flag, value in (
        ("--limit", parsed.limit),
        ("--recall-limit", parsed.recall_limit),
        ("--request-concurrency", parsed.request_concurrency),
        ("--request-timeout-seconds", parsed.request_timeout_seconds),
    ):
        if value is not None:
            shared.extend((flag, str(value)))
    if parsed.overwrite:
        shared.append("--overwrite")
    if parsed.quiet:
        shared.append("--quiet")
    return tuple(shared)


if __name__ == "__main__":
    raise SystemExit(main())
