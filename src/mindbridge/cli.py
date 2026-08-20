"""The `mindbridge` command: one dispatch table over the deployable entry points.

A subcommand's module is imported only when that subcommand runs, so `mindbridge --help`
costs nothing and a command whose optional extra is absent cannot break the ones whose
extras are installed. Every handler keeps its own parser and its own flags; this module
only routes tokens to it, gives every parser the same `--help` shape through `parser`,
and turns a failure into one line plus a documented exit code.

The benchmark runners are addressed by `mindbridge-bench`, which lives in
`mindbridge.benchmarks.cli` and deliberately does not share this module: AGENTS.md keeps
the evaluation harness a leaf that no product module imports, in either direction.
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

_DESCRIPTION = "Agentic-native embodied memory as a service."


def installed_version() -> str:
    """Read the distribution version when it is asked for, not when this module loads.

    Reading it at import time would make every command that imports this module fail on a
    source tree with no installed metadata, which is a working way to run the CLI.
    """
    return version("mindbridge")


def exit_status_help() -> str:
    """Render the exit-code contract every MindBridge command shares.

    `parser` appends this to whatever a command documents about itself, so the codes are
    written once, beside the constants that implement them, instead of restated as prose in
    every command that prints them.
    """
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
        """Print a default only where one exists to fall back to.

        `ArgumentDefaultsHelpFormatter` alone appends `(default: None)` to every required
        flag, which reads as a value an operator could omit down to, and `(default: False)`
        to every switch, which only repeats what a switch already means.
        """
        uninformative = action.default is None or action.default == [] or action.default is False
        if action.required or uninformative:
            return action.help
        return super()._get_help_string(action)


@dataclass(frozen=True, slots=True)
class Command:
    """One dispatchable entry point, imported only when it is the one being run."""

    module: str
    summary: str
    attribute: str = "main"
    extra: str | None = None
    """The optional dependency group this command needs, named when its import fails."""

    def handler(self) -> Callable[..., object]:
        """Import the module this command lives in and return its callable."""
        return getattr(import_module(self.module), self.attribute)  # type: ignore[no-any-return]


COMMANDS: dict[tuple[str, ...], Command] = {
    ("consolidate",): Command(
        "mindbridge.consolidation_cli",
        "Consolidate one tenant's Episodes, Claims, and Summaries",
        extra="server",
    ),
    ("lifecycle",): Command(
        "mindbridge.lifecycle_cli",
        "Decay and transition one tenant's memory strength",
        extra="server",
    ),
    ("mcp",): Command(
        "mindbridge.server",
        "Serve the deployable MCP server over stdio",
        attribute="run_mcp",
        extra="server",
    ),
    ("config", "check"): Command(
        "mindbridge.config_cli",
        "Report whether one role's configuration is complete",
    ),
    ("edge", "sync"): Command(
        "mindbridge.edge.sync_cli",
        "Drain one edge device's observation outbox once",
        extra="edge",
    ),
}

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(path[0] for path in COMMANDS if len(path) > 1))
"""First tokens that name a group of subcommands rather than a command of their own."""


def parser(
    *,
    prog: str | None,
    description: str | None,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Build the parser shape every MindBridge command shares.

    `prog` is what a dispatcher passes down so a subcommand's usage line reads
    `mindbridge edge sync` rather than the module file that happens to hold it. The
    formatter keeps every documented default visible, and the exit-code contract is
    appended here so no command has to restate it.
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
    """Route one invocation to the command it names, or explain the tree."""
    arguments = _without_separator(list(sys.argv[1:] if argv is None else argv))
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(help_text(COMMANDS, GROUPS))
        return 0
    if arguments[0] in {"-V", "--version"}:
        print(f"mindbridge {installed_version()}")
        return 0
    # Two tokens first so `edge sync` wins over any one-token prefix.
    for length in (2, 1):
        command = COMMANDS.get(tuple(arguments[:length]))
        if command is not None:
            prog = " ".join(("mindbridge", *arguments[:length]))
            return guarded(command, arguments[length:], prog=prog)
    if arguments[0] in GROUPS and _asks_for_help(arguments[1:]):
        print(help_text(COMMANDS, GROUPS, group=arguments[0]))
        return 0
    return report_unknown("mindbridge", arguments, COMMANDS, GROUPS)


def guarded(command: Command, arguments: Sequence[str], *, prog: str) -> int:
    """Import and run one command, reducing any failure to one line and an exit code.

    The import is inside the same guard as the run, so a command whose extra is absent
    fails the way an incomplete environment does — including for `--help` — and says which
    extra to install instead of printing frames from a missing third-party package.
    """
    try:
        handler = command.handler()
    except ImportError as error:
        _report_failure(prog, error)
        if command.extra is not None:
            print(f"{prog}: install it with `uv sync --extra {command.extra}`", file=sys.stderr)
        return 1
    return invoke(handler, arguments, prog=prog)


def invoke(handler: Callable[..., object], arguments: Sequence[str], *, prog: str) -> int:
    """Run one already-imported handler under the shared failure contract.

    Every failure is one line by default, infrastructure and model errors included:
    `MINDBRIDGE_TRACEBACK=1` is the documented way to get the frames back, which beats
    guessing per exception type which failures an operator would rather read as prose.
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
    # A handler that ran and reports its own outcome — a smoke that failed, say — returns
    # the code it wants; every other handler returns None and means success.
    return result if isinstance(result, int) else 0


def consolidate() -> int:
    """Entry point for the documented `mindbridge-consolidate` name."""
    return alias(COMMANDS[("consolidate",)], "mindbridge-consolidate")


def lifecycle() -> int:
    """Entry point for the documented `mindbridge-lifecycle` name."""
    return alias(COMMANDS[("lifecycle",)], "mindbridge-lifecycle")


def mcp() -> int:
    """Entry point for the documented `mindbridge-mcp` name."""
    return alias(COMMANDS[("mcp",)], "mindbridge-mcp")


def alias(command: Command, prog: str) -> int:
    """Run one command under its own binary name, which is what its usage should say."""
    return guarded(command, sys.argv[1:], prog=prog)


def help_text(
    commands: dict[tuple[str, ...], Command],
    groups: Sequence[str],
    *,
    group: str | None = None,
) -> str:
    """Render one dispatch table, or one group of it, the way its names are shaped."""
    selected = {
        path: command
        for path, command in commands.items()
        if group is None or (len(path) > 1 and path[0] == group)
    }
    prog = "mindbridge" if group is None else f"mindbridge {group}"
    lines = [
        f"usage: {prog} <command> [options]",
        "",
        _DESCRIPTION,
        "",
        f"Run `{prog} <command> --help` for that command's own flags, defaults, the",
        "environment variables it reads, and its exit status.",
        "",
        "commands:",
        *(
            command_lines(selected, groups)
            if group is None
            else [f"  {' '.join(path[1:]):<20}{entry.summary}" for path, entry in selected.items()]
        ),
        "",
        "options:",
        "  -h, --help            show this help message and exit",
        "  -V, --version         show the installed MindBridge version and exit",
        "",
        exit_status_help(),
    ]
    return "\n".join(lines)


def command_lines(
    commands: dict[tuple[str, ...], Command],
    groups: Sequence[str],
) -> list[str]:
    """List one command per line, under a heading per group that has one.

    Grouping is driven by `groups` rather than by adjacency in the table, so a command
    appended at the end of the dict still lands under its heading instead of repeating it.
    """
    lines = [
        f"  {path[0]:<20}{command.summary}" for path, command in commands.items() if len(path) == 1
    ]
    for group in groups:
        lines.append(f"  {group}:")
        lines += [
            f"    {' '.join(path[1:]):<18}{command.summary}"
            for path, command in commands.items()
            if len(path) > 1 and path[0] == group
        ]
    return lines


def report_unknown(
    prog: str,
    arguments: Sequence[str],
    commands: dict[tuple[str, ...], Command],
    groups: Sequence[str],
) -> int:
    """Name what was not understood, then show the tree that would have worked."""
    attempted = " ".join(arguments[:2]).strip() or "(none)"
    print(f"{prog}: unknown command: {attempted}", file=sys.stderr)
    print(help_text(commands, groups), file=sys.stderr)
    return USAGE_EXIT_CODE


def _asks_for_help(remainder: Sequence[str]) -> bool:
    """Treat a bare group, or a group asking for help, as a request to list that group."""
    return not remainder or set(remainder) <= {"-h", "--help"}


def _without_separator(arguments: list[str]) -> list[str]:
    """Drop the conventional end-of-flags marker so `mindbridge -- lifecycle` still routes."""
    return arguments[1:] if arguments[:1] == ["--"] else arguments


def _report_failure(prog: str, error: BaseException) -> None:
    """Write one line naming the command and what went wrong."""
    print(f"{prog}: error: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
