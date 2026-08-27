"""Run several official benchmarks from one invocation, in the order a suite file lists them.

A suite file exists because a benchmark's name does not determine its invocation. Every runner
needs its own release path, and most need a required choice no sweep can guess -- ATM-Bench's
`--split`, MEMLENS's `--context-window`, or M3-Bench's `--subset`. Some media tasks also need a
prepared-media manifest: the sweep produces it when a preparer exists and otherwise names the
missing path. The suite records one task per invocation; the sweep supplies only the flags every
runner shares.

The sweep owns two of those per task, so a suite may not set them. `--output` is a directory of
the task's own inside `--output-dir`, so every task's artifacts stay together rather than sharing
one flat directory. `--run-id` is the sweep's ID plus the task's name, which is not
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
a sweep summary carrying numbers would be claiming what no runner here measured. The table this
prints when it finishes is not that claim: it reads the numbers back out of each task's own
manifest and score sidecar and names the source of every row. `--report DIR` prints the same
table for a directory an earlier sweep wrote, so a run scored afterwards can be reported without
being run again.
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
    flag_value,
    report,
    select_by_id,
)
from mindbridge.benchmarks.prepare import PREPARERS
from mindbridge.benchmarks.releases import fetch, missing_inputs, release_for
from mindbridge.benchmarks.report import render, render_directory
from mindbridge.benchmarks.scoring import require_scoring_is_possible
from mindbridge.benchmarks.staging import PrepareRequest
from mindbridge.benchmarks.task_catalog import (
    DEFAULT_BENCHMARKS_ROOT,
    listing,
    task_inputs,
    task_payloads,
)
from mindbridge.contracts import ContractModel, Identifier, NonEmptyString, Sha256Hex

SUITE_VERSION = "benchmark_suite_v1"
"""Schema identity of the summary this sweep writes, pinned like every other artifact here."""

SUMMARY_FILENAME = "suite-summary.json"
"""What the sweep's own artifact is called inside `--output-dir`."""

_RULE_WIDTH = 78
"""How wide a task's banner rule is drawn, short enough to survive an 80-column terminal."""

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
    """Overrides the default `predictions.jsonl`, which is wrong for the runners emitting JSON."""

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

    def predictions_path(self) -> Path:
        """Locate this task's predictions, in a directory of its own under the sweep's.

        A directory per task rather than a file per task because a task writes more than its
        predictions: a sidecar manifest, a prepared-media manifest where it needs one, and a
        score sidecar later. In a flat layout those files only share a name prefix; here
        everything one benchmark produced is one `ls` away.
        """
        return Path(self.name) / (self.output_name or "predictions.jsonl")


class BenchmarkSuite(ContractModel):
    """The benchmarks one sweep can run, in the order it runs them."""

    tasks: tuple[SuiteTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_uniquely_named_tasks(self) -> BenchmarkSuite:
        """Refuse a suite in which two tasks share a name.

        It is silent when it happens and expensive to detect afterwards: a run ID is derived from
        the name, so two tasks called the same thing land in one tenant and answer from each
        other's memories. One check covers the outputs too, now that each task writes inside a
        directory named after it -- distinct names are distinct directories, so two tasks can no
        longer collide on a predictions file however they are spelled.
        """
        _reject_repeats((task.name for task in self.tasks), "suite task names must be unique")
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
    download: bool
    output_dir: Path
    run_id: str
    shared: tuple[str, ...]
    media: tuple[str, ...]
    """Forwarded only to the tasks whose runner declares it accepts them."""
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
    if parsed.report is not None:
        print(render_directory(parsed.report))
        return 0
    arguments = _arguments(parser, parsed)
    suite = _load_suite(arguments)
    plans = tuple(_plan(task, arguments) for task in suite.tasks)
    if arguments.dry_run:
        _print_plan(plans)
        return 0
    # Output check before the download: refusing a sweep costs nothing, and the fetch it would
    # otherwise follow is up to 1.4 GB.
    summary_path = arguments.output_dir / SUMMARY_FILENAME
    _require_writable_outputs(plans, summary_path, overwrite=arguments.overwrite)
    _require_every_task_can_report(plans, predict_only="--predict-only" in arguments.shared)
    _obtain_releases(arguments)
    report(f"running {len(plans)} benchmarks", quiet=arguments.quiet)
    started_at = datetime.now(timezone.utc)
    outcomes = _run_plans(plans, arguments)
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
    # The results table is the run's output, not its progress, so it goes to stdout and `--quiet`
    # leaves it alone. Rendered from the bytes just written rather than from `summary` in memory,
    # so this and a later `--report` on the same directory cannot disagree about the same run.
    print(render(summary_path.read_bytes(), directory=arguments.output_dir))
    return _sweep_exit_code(outcomes)


def _obtain_releases(arguments: _Arguments) -> None:
    """Download the official files the named tasks read, and name what no release supplies.

    Only for catalog tasks: a `--suite` file carries literal paths whose provenance this command
    does not know, and guessing which of its arguments are files would be how it started
    downloading a `--split` value.

    A task whose prepared-media manifest is absent is reported rather than refused. It will fail
    when its turn comes, with its runner naming the file, and refusing the whole sweep for it
    would also refuse the tasks that are ready.

    `--no-download` suppresses the fetch, not the check, and it refuses exactly what the fetch
    would have obtained. Returning early instead meant the flag documented as "fail on an absent
    official release" neither downloaded nor failed: the sweep started and each absent corpus
    surfaced as its own task's failure, hours apart, which is the shape
    `_require_writable_outputs` exists to avoid for outputs. An absent operator artifact is
    still only reported, with or without the flag, for the reason above.
    """
    if arguments.suite_path is not None:
        return
    inputs = task_inputs(arguments.task_names, root=arguments.benchmarks_root)
    for name, paths in inputs.items():
        if not arguments.download:
            absent = missing_inputs(paths)
            refusable = tuple(
                path
                for path in absent
                if release_for(path, root=arguments.benchmarks_root) is not None
            )
            if refusable:
                listed = ", ".join(str(path) for path in refusable)
                raise ValueError(
                    f"{name}: {listed} is absent and --no-download was given; drop the flag to "
                    "fetch what an official release supplies"
                )
            unobtainable = tuple(path for path in absent if path not in refusable)
        else:
            unobtainable = fetch(
                paths,
                root=arguments.benchmarks_root,
                announce=None if arguments.quiet else _announce,
            )
        for path in unobtainable:
            report(
                f"{name}: {path} is not part of any official release; prepare it as "
                "docs/benchmarking.md describes",
                quiet=arguments.quiet,
            )


def _announce(message: str) -> None:
    """Say what is being downloaded, so a multi-gigabyte fetch is never silent."""
    report(message, quiet=False)


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


def _run_plans(
    plans: tuple[_Plan, ...],
    arguments: _Arguments,
) -> tuple[SuiteTaskOutcome, ...]:
    """Run the planned tasks one at a time, stopping only for an interrupt."""
    quiet = arguments.quiet
    outcomes: list[SuiteTaskOutcome] = []
    for index, plan in enumerate(plans, start=1):
        _announce_task(plan, index=index, total=len(plans), quiet=quiet)
        outcome = _run_task_prepared(plan, arguments)
        outcomes.append(outcome)
        report(
            f"{plan.task.name} {outcome.status} in {outcome.duration_seconds:.1f}s",
            quiet=quiet,
        )
        if outcome.exit_code == INTERRUPT_EXIT_CODE:
            report("interrupted; the remaining benchmarks did not start", quiet=quiet)
            break
    return tuple(outcomes)


def _announce_task(plan: _Plan, *, index: int, total: int, quiet: bool) -> None:
    """Mark where one benchmark's output begins, inside a stream of every runner's own progress.

    A sweep's stderr is its tasks' stderr with nothing between them: fifteen runners each counting
    their own units towards fifteen different totals, and a `[3/12]` that belongs to whichever one
    is speaking. The rule is what makes the boundary findable scrolling back through four hours of
    that. The derived run ID goes on it because every tenant it names has to be authorized in the
    deployment before the API starts, and this is where a running sweep says what it derived.
    """
    label = f"[{index}/{total}] {plan.task.name} · {plan.task.benchmark} · {plan.run_id}"
    report("", quiet=quiet)
    report(f"{label} {'─' * max(_RULE_WIDTH - len(label), 3)}", quiet=quiet)


def _run_task_prepared(plan: _Plan, arguments: _Arguments) -> SuiteTaskOutcome:
    """Prepare this task's media if it needs it, then run it.

    A preparation that fails is this task's failure, not the sweep's: it produces the same
    outcome row as a run that died, so the summary says which task could not be staged and the
    remaining tasks still run.
    """
    started = time.monotonic()
    try:
        _prepare_task(plan, arguments)
    except KeyboardInterrupt:
        # Staging is the longest phase outside a runner -- a full decode and re-encode, or
        # thousands of uploads -- so it is where an interrupt is most likely to land. Letting
        # `BaseException` through unwinds `main` and the summary is never written, losing every
        # outcome the sweep had already collected. Reported as this task's interrupt instead, so
        # `_run_plans` breaks and the summary is written exactly as for a runner.
        print(f"{PROGRAM} {plan.task.name}: interrupted while preparing media", file=sys.stderr)
        return _outcome(
            plan,
            exit_code=INTERRUPT_EXIT_CODE,
            duration_seconds=time.monotonic() - started,
        )
    except Exception as error:
        print(f"{PROGRAM} {plan.task.name}: error: {error}", file=sys.stderr)
        return _outcome(plan, exit_code=1, duration_seconds=time.monotonic() - started)
    return _run_task(plan)


def _prepare_task(plan: _Plan, arguments: _Arguments) -> None:
    """Produce this task's prepared media, when a producer exists and it is not already there.

    Per task rather than once per sweep: a manifest names objects under the task's own tenant,
    which the sweep derives from its run ID, so two tasks cannot share one.
    """
    producer = PREPARERS.get(plan.task.benchmark)
    manifest = _prepared_manifest_path(plan)
    if producer is None or manifest is None or manifest.exists():
        return
    report(f"preparing {plan.task.name}", quiet=arguments.quiet)
    producer.produce(
        PrepareRequest(
            argv=plan.arguments,
            benchmarks_root=arguments.benchmarks_root,
            quiet=arguments.quiet,
            download=arguments.download,
        )
    )


def _prepared_manifest_path(plan: _Plan) -> Path | None:
    """The manifest this task reads, read off the argv rather than guessed from the benchmark."""
    for flag in ("--prepared-media", "--prepared-images"):
        value = flag_value(plan.arguments, flag)
        if value is not None:
            return Path(value)
    return None


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
    return _outcome(plan, exit_code=exit_code, duration_seconds=time.monotonic() - started)


def _outcome(plan: _Plan, *, exit_code: int, duration_seconds: float) -> SuiteTaskOutcome:
    """Record what one task did, whether it got as far as running or not."""
    return SuiteTaskOutcome(
        name=plan.task.name,
        benchmark=plan.task.benchmark,
        run_id=plan.run_id,
        output_path=str(plan.output_path),
        arguments=plan.arguments,
        status=_task_status(plan, exit_code),
        exit_code=exit_code,
        duration_seconds=duration_seconds,
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
    output_path = arguments.output_dir / task.predictions_path()
    return _Plan(
        task=task,
        run_id=run_id,
        output_path=output_path,
        arguments=(
            *arguments.shared,
            *(arguments.media if RUNNERS[task.benchmark].media else ()),
            *task.arguments,
            *_prepared_media_arguments(task, arguments),
            "--output",
            str(output_path),
            "--run-id",
            run_id,
        ),
    )


def _prepared_media_arguments(task: SuiteTask, arguments: _Arguments) -> tuple[str, ...]:
    """Point a producer-backed task at the manifest this run will write for it.

    The sweep owns this path for the same reason it owns `--run-id`: a manifest names objects
    under one run's tenant, so a path shared between runs is one a later run cannot read. Left in
    the catalog, an earlier run's file was found on disk, preparation was skipped as
    already-done, and the run failed on a manifest describing somebody else's tenant.

    The run ID is in the file name, not only in the default output directory. `--output-dir`
    given explicitly is the same directory for every run, so without it a second sweep found the
    first one's `prepared.json`, skipped preparation as already-done, and ingested objects under
    a tenant it cannot address -- exactly the failure described above, reintroduced by the flag
    that lets an operator choose where results land.

    A task that already carries the flag keeps it, which is how a hand-written `--suite` entry
    supplies a manifest MindBridge cannot produce.
    """
    producer = PREPARERS.get(task.benchmark)
    if producer is None or flag_value(task.arguments, producer.flag) is not None:
        return ()
    if producer.applies is not None and not producer.applies(task.arguments):
        # Gated here rather than in `_prepare_task`, because this is the call that creates the
        # thing `_prepare_task` keys on: withhold the flag and the manifest path is never
        # appended, `_prepared_manifest_path` answers None, and preparation no-ops for free.
        return ()
    manifest = arguments.output_dir / task.name / f"{arguments.run_id}-prepared.json"
    return (producer.flag, str(manifest))


def _require_writable_outputs(
    plans: Sequence[_Plan],
    summary_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Refuse the whole sweep before it starts if any of its outputs already exist.

    Each runner makes this check for itself, but only once it is running. Finding out at the
    fourth benchmark that the fifth cannot write is what this exists to prevent.

    A resumable runner is exempt, and so is the summary once the sweep contains one. Every other
    runner writes its predictions at the end, so an existing output is a finished result; `aml`
    appends per finished case, so an existing output is a prefix to continue. Refusing it left an
    interrupted `--run-id`-named sweep with no way forward: rerunning hit this, and adding
    `--overwrite` to get past it told the runner to discard the very rows that made resuming
    possible -- then replay them into a tenant that already held their memories. Exempting it
    loses no safety, because `aml` refuses an output whose manifest disagrees with the run about
    benchmark, run id, deployment, or recall limit, which is a stricter test than existence.

    The summary describes the interrupted attempt and is rewritten by the sweep that replaces it.
    """
    resumable = tuple(plan for plan in plans if RUNNERS[plan.task.benchmark].resumable)
    for plan in plans:
        if RUNNERS[plan.task.benchmark].resumable:
            continue
        require_writable_output_pair(plan.output_path, overwrite=overwrite)
    if summary_path.exists() and not overwrite and not resumable:
        raise FileExistsError(f"output already exists: {summary_path}")


def _require_every_task_can_report(plans: Sequence[_Plan], *, predict_only: bool) -> None:
    """Refuse the sweep if a judged task has no judge, before the first one starts.

    Each runner makes this check for itself now, but only once it is running -- and a judged task
    that runs to completion and then cannot score writes nothing, so finding out task by task
    costs every earlier task's predictions. Seventeen of the thirty catalog tasks are judged; the
    eight AML tasks are not, because their number comes from a vendored scorer run afterwards.
    """
    for plan in plans:
        require_scoring_is_possible(plan.task.benchmark, predict_only=predict_only)


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
        "--run-id",
        help="identifier for this sweep; each task runs under <run-id>-<task name>. "
        "Defaults to sweep-<UTC timestamp>, which is unique and therefore isolating",
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
        "--unit-concurrency",
        type=int,
        help="units of one benchmark run at once, for every task that does not set its own",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        help="deadline for one request, for every task that does not set its own",
    )
    parser.add_argument(
        "--device-id",
        help="device identity to ingest as, for every media task that does not set its own",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        help="delay between processing-status polls, for every media task not setting its own",
    )
    parser.add_argument(
        "--processing-timeout-seconds",
        type=float,
        help="deadline for one observation to finish processing, for every media task",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        help="run the tasks in this JSON file instead of the catalog's; --tasks then narrows it",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="fail on an absent official release instead of downloading it first",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing predictions, manifests, and this sweep's summary",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="write every task's predictions without scoring them; no judge is contacted and "
        "each metric reports lmms-eval's bypass sentinel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the invocation each task would run, then exit without running any",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="DIR",
        help=f"print the results table for a directory holding a {SUMMARY_FILENAME}, then exit; "
        "how a run scored after the fact gets its numbers on screen without running again",
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
    root = parsed.benchmarks_root
    run_id = parsed.run_id or _default_run_id()
    return _Arguments(
        task_names=names,
        suite_path=parsed.suite,
        benchmarks_root=root,
        download=parsed.download,
        output_dir=parsed.output_dir or root / "results" / run_id,
        run_id=run_id,
        shared=_shared_arguments(parsed, root=root),
        media=_media_arguments(parsed),
        overwrite=parsed.overwrite,
        quiet=parsed.quiet,
        dry_run=parsed.dry_run,
    )


def _default_run_id() -> str:
    """Name a sweep nobody named, uniquely enough that it isolates its own tenants.

    A tenant is derived from the run ID, so the only thing this has to guarantee is that two
    sweeps never share one; a UTC timestamp does that and sorts, which a random suffix would not.
    It also gives each sweep its own `--output-dir`, so a rerun no longer meets its predecessor's
    files and no longer needs `--overwrite` to get past them.
    """
    return datetime.now(timezone.utc).strftime("sweep-%Y%m%d-%H%M%S")


def _media_arguments(parsed: argparse.Namespace) -> tuple[str, ...]:
    """Collect the ingest-and-wait knobs, for the tasks whose runner declares it takes them.

    Separate from the shared flags because these exist only on the eleven runners that ingest
    media; handing `--device-id` to LoCoMo-Refined is an argparse error that would fail its task.
    `Runner.media` is where that distinction is declared, so a sweep mixing text and media
    benchmarks raises one timeout for the ones it applies to and leaves the other alone.
    """
    media: list[str] = []
    for flag, value in (
        ("--device-id", parsed.device_id),
        ("--poll-interval-seconds", parsed.poll_interval_seconds),
        ("--processing-timeout-seconds", parsed.processing_timeout_seconds),
    ):
        if value is not None:
            media.extend((flag, str(value)))
    return tuple(media)


def _shared_arguments(parsed: argparse.Namespace, *, root: Path) -> tuple[str, ...]:
    """Collect the flags forwarded to every task, and only the ones that were given."""
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
        ("--unit-concurrency", parsed.unit_concurrency),
        ("--request-timeout-seconds", parsed.request_timeout_seconds),
    ):
        if value is not None:
            shared.extend((flag, str(value)))
    if parsed.overwrite:
        shared.append("--overwrite")
    if parsed.predict_only:
        shared.append("--predict-only")
    if parsed.quiet:
        shared.append("--quiet")
    return tuple(shared)


if __name__ == "__main__":
    raise SystemExit(main())
