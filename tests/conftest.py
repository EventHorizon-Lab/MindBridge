"""Session-wide watchdog: turn a hung test run into a failure with a thread dump.

`tests/unit` has deadlocked intermittently -- a run that normally finishes in seconds sat for
twenty minutes before someone killed it, and the next run passed. A hang costs a CI slot until
something external notices, and it leaves nothing behind to diagnose, so the two runs that
matter most produce the least evidence.

`faulthandler.dump_traceback_later` is the stdlib answer and needs no plugin: it arms a watchdog
thread that writes every thread's stack to stderr and exits non-zero. The stacks are the point.
The dump that first localised the x264 encoder pools came from attaching to a live hung process,
which only works if a human is watching; this gets the same evidence out of an unattended run.

pytest ships a `faulthandler_timeout` ini option that wraps the same call per test. It did not
fire here when tried against a deliberate deadlock (pytest 9.0.1), so this arms the call directly
rather than depending on behaviour that is not working in this tree.

The budget is per session and deliberately far above any healthy run (`tests/unit` is ~12s,
integration is slower but bounded). Override with MINDBRIDGE_TEST_TIMEOUT_SECONDS; 0 disables.
"""

import faulthandler
import os
from collections.abc import Iterator

import pytest

DEFAULT_TIMEOUT_SECONDS = 900.0


@pytest.fixture(scope="session", autouse=True)
def _fail_on_hang() -> Iterator[None]:
    timeout = float(os.environ.get("MINDBRIDGE_TEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0:
        yield
        return
    # exit=True: a deadlocked interpreter cannot raise, so the watchdog has to end the process
    # itself. It reports as a failed run, which is the whole point of doing this.
    faulthandler.dump_traceback_later(timeout, exit=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
