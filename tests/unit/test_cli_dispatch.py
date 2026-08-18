"""The two command trees: every entry point answers, every failure has a code.

The per-command parsers are covered where their arguments are; what is checked here is the
contract they share. The parametrized help tests are the ones that would catch a registry
typo, a parser that cannot be constructed, or a subcommand whose usage still names the
module file it happens to live in.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mindbridge import cli, lifecycle_cli
from mindbridge.benchmarks import cli as bench_cli

PRODUCT_PATHS = list(cli.COMMANDS)
BENCH_NAMES = list(bench_cli.RUNNERS)


@pytest.mark.parametrize("path", PRODUCT_PATHS, ids=lambda path: " ".join(path))
def test_every_command_answers_help_under_its_own_name(
    path: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([*path, "--help"])

    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert " ".join(("mindbridge", *path)) in printed
    assert "exit status:" in printed


@pytest.mark.parametrize("name", BENCH_NAMES)
def test_every_benchmark_answers_help_under_its_own_name(
    name: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        bench_cli.main([name, "--help"])

    assert exit_info.value.code == 0
    printed = capsys.readouterr().out
    assert f"mindbridge-bench {name}" in printed
    assert "exit status:" in printed


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_both_version_spellings_work_on_a_subcommand_and_on_a_tree(
    flag: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`-V` is advertised by both trees, so it has to work everywhere `--version` does."""
    expected = f"mindbridge {cli.installed_version()}"

    assert cli.main([flag]) == 0
    assert capsys.readouterr().out.strip() == expected
    assert bench_cli.main([flag]) == 0
    assert capsys.readouterr().out.strip() == expected
    for tree, argv in ((cli, ["lifecycle", flag]), (bench_cli, ["locomo", flag])):
        with pytest.raises(SystemExit) as exit_info:
            tree.main(argv)
        assert exit_info.value.code == 0
        assert capsys.readouterr().out.strip() == expected


def test_each_tree_lists_every_command_it_can_dispatch(capsys: pytest.CaptureFixture[str]) -> None:
    """An invocation with nothing to run explains the surface instead of failing."""
    assert cli.main([]) == 0
    printed = capsys.readouterr().out
    for path in PRODUCT_PATHS:
        assert path[-1] in printed

    assert bench_cli.main([]) == 0
    printed = capsys.readouterr().out
    for name in BENCH_NAMES:
        assert name in printed


@pytest.mark.parametrize("argv", [["edge"], ["edge", "--help"], ["edge", "-h"]])
def test_a_group_lists_its_own_subcommands(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The tree tells the reader to run `<command> --help`, so a group must answer it."""
    assert cli.main(argv) == 0

    printed = capsys.readouterr().out
    assert "usage: mindbridge edge <command>" in printed
    assert "sync" in printed


def test_the_end_of_flags_separator_still_routes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--", "lifecycle", "--help"])

    assert exit_info.value.code == 0
    assert "mindbridge lifecycle" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["nonesuch"], "unknown command: nonesuch"), ([""], "unknown command: (none)")],
)
def test_an_unknown_command_names_it_on_stderr(
    argv: list[str],
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(argv) == cli.USAGE_EXIT_CODE

    streams = capsys.readouterr()
    assert expected in streams.err
    assert streams.out == ""


def test_an_unknown_benchmark_names_it_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert bench_cli.main(["nonesuch"]) == bench_cli.USAGE_EXIT_CODE

    streams = capsys.readouterr()
    assert "unknown benchmark: nonesuch" in streams.err
    assert streams.out == ""


def test_a_configuration_mistake_is_one_line_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """What an operator who forgot a variable sees, instead of a traceback."""

    def handler(argv: Sequence[str], *, prog: str) -> None:
        raise ValueError("MINDBRIDGE_DATABASE_URL must be configured")

    assert cli.invoke(handler, [], prog="mindbridge lifecycle") == 1
    assert capsys.readouterr().err == (
        "mindbridge lifecycle: error: MINDBRIDGE_DATABASE_URL must be configured\n"
    )


def test_an_infrastructure_failure_is_one_line_too(capsys: pytest.CaptureFixture[str]) -> None:
    """A wrong URL or a stale key surfaces as MindBridgeError, a RuntimeError subclass."""

    def handler(argv: Sequence[str], *, prog: str) -> None:
        raise RuntimeError("POST /remember failed with 401")

    assert cli.invoke(handler, [], prog="mindbridge-bench locomo") == 1
    assert capsys.readouterr().err == (
        "mindbridge-bench locomo: error: POST /remember failed with 401\n"
    )


def test_the_traceback_variable_gives_the_frames_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-line form is the default, not the only option."""
    monkeypatch.setenv(cli.TRACEBACK_VARIABLE, "1")

    def handler(argv: Sequence[str], *, prog: str) -> None:
        raise RuntimeError("the pool is gone")

    with pytest.raises(RuntimeError):
        cli.invoke(handler, [], prog="mindbridge lifecycle")


def test_an_interrupt_exits_the_conventional_code(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(argv: Sequence[str], *, prog: str) -> None:
        raise KeyboardInterrupt

    assert cli.invoke(handler, [], prog="mindbridge lifecycle") == cli.INTERRUPT_EXIT_CODE
    assert capsys.readouterr().err == "mindbridge lifecycle: interrupted\n"


def test_a_handler_that_reports_its_own_code_is_forwarded() -> None:
    """`bench jina` returns 1 for a failed smoke; exiting 0 would be silently green."""

    def handler(argv: Sequence[str], *, prog: str) -> int:
        return 1

    assert cli.invoke(handler, [], prog="mindbridge-bench jina") == 1


def test_a_missing_extra_names_the_extra_to_install(capsys: pytest.CaptureFixture[str]) -> None:
    """The import happens inside the guard, so even `--help` explains itself."""
    command = cli.Command("mindbridge.absent_module", "Never installed", extra="server")

    assert cli.guarded(command, ["--help"], prog="mindbridge absent") == 1

    printed = capsys.readouterr().err
    assert "mindbridge absent: error:" in printed
    assert "uv sync --extra server" in printed


def test_dry_run_refuses_to_pretend_it_covers_the_whole_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run qualifies the deletion; alone it would read as "change nothing"."""
    # Removed so a developer with a DSN exported cannot have this reach a real database if
    # the guard under test ever moves below it.
    monkeypatch.delenv("MINDBRIDGE_DATABASE_URL", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        lifecycle_cli.main(["--tenant-id", "tenant_01", "--dry-run"])

    assert exit_info.value.code == cli.USAGE_EXIT_CODE
