"""Minimal dispatcher for the supported local benchmarks."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import version

USAGE_EXIT_CODE = 2
INTERRUPT_EXIT_CODE = 130
PROGRAM = "mindbridge-bench"

RUNNERS: dict[str, tuple[str, str]] = {
    "eval": (
        "mindbridge.benchmarks.eval",
        "Run pinned benchmark tasks with confidence intervals",
    ),
    "local-index": (
        "mindbridge.benchmarks.local_index_benchmark",
        "Measure SQLite and Zvec ingestion and recall",
    ),
    "locomo-refined": (
        "mindbridge.benchmarks.locomo_refined_cli",
        "Run official LoCoMo-Refined",
    ),
}


def installed_version() -> str:
    return version("mindbridge")


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one benchmark without importing any runner for global help."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(help_text())
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(f"mindbridge {installed_version()}")
        return 0

    command = arguments[0]
    runner = RUNNERS.get(command)
    if runner is None:
        attempted = command.strip() or "(none)"
        print(f"{PROGRAM}: unknown benchmark: {attempted}", file=sys.stderr)
        return USAGE_EXIT_CODE

    try:
        handler = import_module(runner[0]).main
        result = handler(arguments[1:], prog=f"{PROGRAM} {command}")
        return result if isinstance(result, int) else 0
    except KeyboardInterrupt:
        print(f"{PROGRAM} {command}: interrupted", file=sys.stderr)
        return INTERRUPT_EXIT_CODE
    except Exception as error:
        message = " ".join(str(error).split()) or type(error).__name__
        print(f"{PROGRAM} {command}: error: {message}", file=sys.stderr)
        return 1


def help_text() -> str:
    return "\n".join(
        (
            f"usage: {PROGRAM} <benchmark> [options]",
            "",
            "benchmarks:",
            *(f"  {name:<18}{runner[1]}" for name, runner in RUNNERS.items()),
            "",
            "options:",
            "  -h, --help        show this help message and exit",
            "  -V, --version     show the installed MindBridge version and exit",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
