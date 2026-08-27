"""Dispatch checks for the intentionally small benchmark command."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

import pytest

from mindbridge.benchmarks import cli


def test_global_help_lists_only_supported_runners_without_importing_them(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def imported(_name: str) -> ModuleType:
        raise AssertionError("global help must not import a runner")

    monkeypatch.setattr(cli, "import_module", imported)

    assert cli.main([]) == 0

    printed = capsys.readouterr().out
    assert set(cli.RUNNERS) == {"eval", "local-index", "locomo-refined"}
    assert "eval" in printed
    assert "local-index" in printed
    assert "locomo-refined" in printed
    for removed in ("m3", "aml", "datasets", "jina", "score"):
        assert removed not in printed


def test_dispatch_lazily_calls_each_supported_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, ...], str | None]] = []
    local = ModuleType("local")
    locomo = ModuleType("locomo")
    evaluation = ModuleType("evaluation")

    def local_main(argv: Sequence[str], *, prog: str | None = None) -> int:
        calls.append(("local-index", tuple(argv), prog))
        return 0

    def locomo_main(argv: Sequence[str], *, prog: str | None = None) -> int:
        calls.append(("locomo-refined", tuple(argv), prog))
        return 0

    def eval_main(argv: Sequence[str], *, prog: str | None = None) -> int:
        calls.append(("eval", tuple(argv), prog))
        return 0

    local.main = local_main  # type: ignore[attr-defined]
    locomo.main = locomo_main  # type: ignore[attr-defined]
    evaluation.main = eval_main  # type: ignore[attr-defined]

    def imported(name: str) -> ModuleType:
        if name.endswith("locomo_refined_cli"):
            return locomo
        return evaluation if name.endswith(".eval") else local

    monkeypatch.setattr(cli, "import_module", imported)

    assert cli.main(["local-index", "--rows", "10"]) == 0
    assert cli.main(["locomo-refined", "--limit", "2"]) == 0
    assert cli.main(["eval", "--tasks", "video-mme"]) == 0
    assert calls == [
        ("local-index", ("--rows", "10"), "mindbridge-bench local-index"),
        (
            "locomo-refined",
            ("--limit", "2"),
            "mindbridge-bench locomo-refined",
        ),
        ("eval", ("--tasks", "video-mme"), "mindbridge-bench eval"),
    ]


def test_dispatch_has_stable_usage_failure_and_interrupt_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = ModuleType("runner")

    def fail(_argv: Sequence[str], *, prog: str | None = None) -> None:
        del prog
        raise RuntimeError("first line\nsecond line")

    module.main = fail  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "import_module", lambda _name: module)

    assert cli.main(["local-index"]) == 1
    assert capsys.readouterr().err == (
        "mindbridge-bench local-index: error: first line second line\n"
    )

    def interrupt(_argv: Sequence[str], *, prog: str | None = None) -> None:
        del prog
        raise KeyboardInterrupt

    module.main = interrupt  # type: ignore[attr-defined]
    assert cli.main(["local-index"]) == cli.INTERRUPT_EXIT_CODE
    assert capsys.readouterr().err == "mindbridge-bench local-index: interrupted\n"

    assert cli.main(["removed"]) == cli.USAGE_EXIT_CODE
    assert "unknown benchmark: removed" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_global_version(
    flag: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "installed_version", lambda: "9.8.7")

    assert cli.main([flag]) == 0
    assert capsys.readouterr().out == "mindbridge 9.8.7\n"
