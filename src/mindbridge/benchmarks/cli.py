"""The `mindbridge-bench` command: one dispatch table over the reproducible runners.

Separate from `mindbridge` on purpose. AGENTS.md keeps this package a leaf — "`benchmarks/`
may only call the public SDK and contracts; no product module may import it" — so the
product entry point cannot address these runners and they cannot borrow its parser. The
plumbing below is therefore a deliberate second copy of the small shape `mindbridge.cli`
defines, which is what that rule costs and all it costs: the codes, the formatter, and the
failure contract are the same, and the two tables never reference each other.

A runner's module is imported only when it runs, so `mindbridge-bench --help` costs nothing
and a runner whose optional extra is absent cannot break the ones whose extras are present.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version

USAGE_EXIT_CODE = 2
"""What an unusable invocation exits with, matching argparse's own convention."""

INTERRUPT_EXIT_CODE = 130
"""What an interrupted run exits with, matching the shell's 128 + SIGINT convention."""

TRACEBACK_VARIABLE = "MINDBRIDGE_TRACEBACK"
"""Set this to keep the full traceback behind a failure instead of the one-line form."""

PROGRAM = "mindbridge-bench"
_DESCRIPTION = "Reproducible MindBridge benchmark runners and their supporting commands."


def installed_version() -> str:
    """Read the distribution version when it is asked for, not when this module loads."""
    return version("mindbridge")


def exit_status_help() -> str:
    """Render the exit-code contract every runner shares, from the constants above."""
    return (
        "exit status:\n"
        "  0    the command completed\n"
        "  1    the command failed; the message names what\n"
        f"  {USAGE_EXIT_CODE}    the invocation was unusable\n"
        f"  {INTERRUPT_EXIT_CODE}  the command was interrupted\n"
        "\n"
        f"Set {TRACEBACK_VARIABLE}=1 to keep the full traceback behind a failure."
    )


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep an epilog's own line breaks while still printing every default worth printing."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """Print a default only where one exists to fall back to."""
        uninformative = action.default is None or action.default == [] or action.default is False
        if action.required or uninformative:
            return action.help
        return super()._get_help_string(action)


@dataclass(frozen=True, slots=True)
class Runner:
    """One dispatchable benchmark command, imported only when it is the one being run."""

    module: str
    summary: str
    extra: str | None = None
    """The optional dependency group this runner needs, named when its import fails."""

    def handler(self) -> Callable[..., object]:
        """Import the module this runner lives in and return its entry point."""
        return import_module(self.module).main  # type: ignore[no-any-return]


RUNNERS: dict[str, Runner] = {
    "locomo-refined": Runner(
        "mindbridge.benchmarks.locomo_refined_cli", "Run official LoCoMo-Refined"
    ),
    "m3": Runner("mindbridge.benchmarks.m3_cli", "Run official M3-Bench"),
    "egolife": Runner("mindbridge.benchmarks.egolife_cli", "Run official EgoLifeQA"),
    "egomem": Runner("mindbridge.benchmarks.egomem_cli", "Run official EgoMemReason"),
    "egotempo": Runner("mindbridge.benchmarks.egotempo_cli", "Run official EgoTempo"),
    "memlens": Runner("mindbridge.benchmarks.memlens_cli", "Run official MemLens"),
    "mm-lifelong": Runner("mindbridge.benchmarks.mm_lifelong_cli", "Run official MM-Lifelong"),
    "supermemory": Runner("mindbridge.benchmarks.supermemory_cli", "Run official SuperMemory VQA"),
    "video-mme": Runner(
        "mindbridge.benchmarks.video_mme_cli",
        "Run official Video-MME",
        extra="benchmarks",
    ),
    "video-mme-v2": Runner(
        "mindbridge.benchmarks.video_mme_v2_cli",
        "Run official Video-MME-v2",
        extra="benchmarks",
    ),
    "aml": Runner("mindbridge.benchmarks.aml.cli", "Replay one offline AML pipeline"),
    "score": Runner(
        "mindbridge.benchmarks.official_score",
        "Record an official scorer's verdict beside a run",
    ),
    "datasets": Runner(
        "mindbridge.benchmarks.dataset_smoke",
        "Check every official release parses and pins its digest",
        extra="benchmarks",
    ),
    "jina": Runner(
        "mindbridge.benchmarks.jina_smoke",
        "Check the local Jina Omni embedder answers",
        extra="cloud-models",
    ),
    "bakeoff": Runner(
        "mindbridge.benchmarks.adapter_bakeoff",
        "Compare candidate adapters on one prepared corpus",
        extra="cloud-models",
    ),
}


def parser(
    *,
    prog: str | None,
    description: str | None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Build the parser shape every benchmark command shares.

    `prog` is what the dispatcher passes down so a runner's usage line reads
    `mindbridge-bench locomo-refined` rather than the module file that happens to hold it.
    """
    sections = (epilog, exit_status_help()) if epilog else (exit_status_help(),)
    built = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog="\n\n".join(sections),
        formatter_class=HelpFormatter,
    )
    built.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"mindbridge {installed_version()}",
        help="show the installed MindBridge version and exit",
    )
    return built


def main(argv: Sequence[str] | None = None) -> int:
    """Route one invocation to the runner it names, or explain the table."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(help_text())
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(f"mindbridge {installed_version()}")
        return 0
    runner = RUNNERS.get(arguments[0])
    if runner is None:
        attempted = arguments[0].strip() or "(none)"
        print(f"{PROGRAM}: unknown benchmark: {attempted}", file=sys.stderr)
        print(help_text(), file=sys.stderr)
        return USAGE_EXIT_CODE
    return guarded(runner, arguments[1:], prog=f"{PROGRAM} {arguments[0]}")


def guarded(runner: Runner, arguments: Sequence[str], *, prog: str) -> int:
    """Import and run one benchmark, reducing any failure to one line and an exit code.

    The import is inside the same guard as the run, so a runner whose extra is absent fails
    the way an incomplete environment does — including for `--help` — and says which extra
    to install instead of printing frames from a missing third-party package.
    """
    _configure_run_observability()
    try:
        handler = runner.handler()
    except ImportError as error:
        _report_failure(prog, error)
        if runner.extra is not None:
            print(f"{prog}: install it with `uv sync --extra {runner.extra}`", file=sys.stderr)
        return 1
    try:
        return invoke(handler, arguments, prog=prog)
    finally:
        # A benchmark run exists to be measured, so its own cost breakdown is part of the
        # result rather than an opt-in. Emitted on failure too: a run that died halfway is
        # exactly when knowing which stage owned the wall clock is worth most.
        _log_run_timings()


def _log_run_timings() -> None:
    """Emit the ranked per-operation summary this run accumulated, when it is reachable."""
    try:
        from mindbridge.telemetry import log_timing_summary
    except ImportError:  # pragma: no cover - only a benchmarks install without the API
        return
    log_timing_summary()


def _configure_run_observability() -> None:
    """Install run logging, or leave the run silent rather than fail on a missing extra.

    Logging only, not the OTLP exporters: the `benchmarks` extra carries the OpenTelemetry
    API and not the SDK or the instrumentation packages, and a measurement run needs its
    timings on stderr rather than a collector. A deployment process configures both.
    """
    try:
        from mindbridge.telemetry import configure_logging
    except ImportError:  # pragma: no cover - only a benchmarks install without the API
        return
    configure_logging(PROGRAM)


def invoke(handler: Callable[..., object], arguments: Sequence[str], *, prog: str) -> int:
    """Run one already-imported runner under the shared failure contract.

    Every failure is one line by default, the API and model errors a long run dies on
    included; `MINDBRIDGE_TRACEBACK=1` is the documented way to get the frames back.
    """
    try:
        result = handler(list(arguments), prog=prog)
    except KeyboardInterrupt:
        print(f"{prog}: interrupted", file=sys.stderr)
        return INTERRUPT_EXIT_CODE
    except Exception as error:
        if os.environ.get(TRACEBACK_VARIABLE, "").strip():
            raise
        _report_failure(prog, error)
        return 1
    # A smoke that ran and failed returns the code it wants; the runners return None.
    return result if isinstance(result, int) else 0


def help_text() -> str:
    """Render the benchmark table, official runners before the supporting commands."""
    return "\n".join(
        [
            f"usage: {PROGRAM} <benchmark> [options]",
            "",
            _DESCRIPTION,
            "",
            f"Run `{PROGRAM} <benchmark> --help` for that command's own flags, defaults,",
            "the environment variables it reads, and its exit status.",
            "",
            "benchmarks:",
            *(f"  {name:<20}{runner.summary}" for name, runner in RUNNERS.items()),
            "",
            "options:",
            "  -h, --help            show this help message and exit",
            "  -V, --version         show the installed MindBridge version and exit",
            "",
            exit_status_help(),
        ]
    )


def _report_failure(prog: str, error: BaseException) -> None:
    """Write one line naming the command and what went wrong."""
    print(f"{prog}: error: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
