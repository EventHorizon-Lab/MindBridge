"""Shared fixtures for the benchmark CLI tests."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def restored_logging() -> Iterator[None]:
    """Give pytest back the root handler that `eval`'s `_configure_logging` claims.

    Autouse rather than opt-in because every entry point under test reaches it: `main()`
    configures logging before it does anything else, so a listing or an offline gate leaves the
    process at INFO behind a handler of `eval`'s own. A test file that runs later then reads
    MindBridge's INFO lines out of the stream it was asserting on.
    """
    root = logging.getLogger()
    saved = (root.level, list(root.handlers), list(root.filters))
    try:
        yield
    finally:
        root.setLevel(saved[0])
        root.handlers[:] = saved[1]
        root.filters[:] = saved[2]
