"""Console rendering must not allocate resources owned by multiprocessing."""

from __future__ import annotations

import signal
import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.parametrize("exit_mode", ("normal", "atexit", "abrupt", "sigterm"))
def test_console_does_not_create_multiprocessing_resources(exit_mode: str) -> None:
    if exit_mode == "sigterm" and sys.platform == "win32":
        pytest.skip("POSIX signal exit status")
    code = dedent(
        """
        import atexit
        import io
        import logging
        import multiprocessing
        import os
        import signal
        import sys
        from concurrent.futures import ThreadPoolExecutor

        exit_mode = sys.argv[1]

        def report():
            _announce("fixture")
            _configure_logging("INFO")
            logging.getLogger("mindbridge.fixture").warning("fixture log")
            with _progress("fixture", "sample", total=1) as progress:
                progress(1, 1)

        if exit_mode == "atexit":
            # First output after multiprocessing's exit callback has already run.
            atexit.register(report)

        # Exercise named semaphore tracking on Python 3.10-3.13 as well as 3.14.
        multiprocessing.set_start_method("spawn", force=True)
        from multiprocessing import resource_tracker

        register = resource_tracker.register

        def tracked_register(name, kind):
            print("TRACKED_RESOURCE", kind, flush=True)
            register(name, kind)

        resource_tracker.register = tracked_register

        from mindbridge.benchmarks.eval_console import (
            _announce, _configure_logging, _progress,
        )

        if exit_mode != "atexit":
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(report).result()

            class Terminal(io.StringIO):
                def isatty(self):
                    return True

            terminal = Terminal()
            original_stderr = sys.stderr
            sys.stderr = terminal
            try:
                with _progress("terminal fixture", "sample", total=1) as progress:
                    _announce("live fixture")
                    logging.getLogger("mindbridge.fixture").warning("live fixture log")
                    progress(1, 1)
            finally:
                sys.stderr = original_stderr
            print(terminal.getvalue(), file=sys.stderr, flush=True)

        if exit_mode == "abrupt":
            os._exit(73)
        if exit_mode == "sigterm":
            os.kill(os.getpid(), signal.SIGTERM)
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, exit_mode],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    expected = {"normal": 0, "atexit": 0, "abrupt": 73, "sigterm": -signal.SIGTERM}
    assert result.returncode == expected[exit_mode], result.stderr
    assert "fixture log" in result.stderr
    assert "TRACKED_RESOURCE" not in result.stdout, result.stdout
    assert "resource_tracker:" not in result.stderr, result.stderr
