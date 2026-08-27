"""The hang watchdog is configuration, so the only way to know it works is to hang a run.

Two separate properties have to hold, and a watchdog that has one of them is worse than none:
it consumes the CI slot anyway and reports nothing. `faulthandler_timeout` on its own dumps and
lets the run keep hanging, because `faulthandler_exit_on_timeout` defaults to false. A dump
armed from a session fixture instead of the plugin writes into the unlinked temp file pytest's
fd capture put on descriptor 2, so it reaches nobody, and pytest's own
`pytest_exception_interact` cancels it outright after the first failure of the session.

Run in a subprocess against this repository's real `pyproject.toml`, overriding only the budget,
so what is asserted is the configuration that ships rather than a copy of it.
"""

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _run_hanging_suite(
    tmp_path: Path, *, with_a_prior_failure: bool
) -> subprocess.CompletedProcess[str]:
    body = "import threading\n\n\n"
    if with_a_prior_failure:
        # Ordered before the hang on purpose: the plugin cancels the timer on any failure and
        # only re-arms because it does so per test.
        body += "def test_fails_before_the_hang() -> None:\n    raise AssertionError('deliberate')\n\n\n"
    body += "def test_hangs_forever() -> None:\n    threading.Event().wait()\n"
    probe = tmp_path / "test_hang_probe.py"
    probe.write_text(body, encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            # `-c` pins the child to this repository's configuration; the probe lives outside the
            # tree, which would otherwise make pytest pick a rootdir that has none of it.
            "-c",
            str(REPOSITORY_ROOT / "pyproject.toml"),
            "-o",
            "faulthandler_timeout=2",
            "-p",
            "no:cacheprovider",
            str(probe),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_a_hung_test_ends_the_run_and_names_itself(tmp_path: Path) -> None:
    result = _run_hanging_suite(tmp_path, with_a_prior_failure=False)

    # Exit, not just dump: without `faulthandler_exit_on_timeout` this returns nothing at all
    # because the run is still hanging when the timeout below kills it.
    assert result.returncode != 0
    assert "Timeout (0:00:02)" in result.stderr
    # The stack is the whole point -- a dump that lands in pytest's captured fd names nothing.
    assert "test_hangs_forever" in result.stderr


def test_the_watchdog_survives_an_earlier_failure(tmp_path: Path) -> None:
    """A session-scoped watchdog does not: pytest cancels it on the first failure and never re-arms."""
    result = _run_hanging_suite(tmp_path, with_a_prior_failure=True)

    assert result.returncode != 0
    assert "Timeout (0:00:02)" in result.stderr
    assert "test_hangs_forever" in result.stderr
