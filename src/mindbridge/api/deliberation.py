"""The periodic control-plane loop a serving process runs on the memory it serves.

`consolidation_candidates()` is the only reader of the `QUERY_FAILURE` rows a failed recall
writes, and one physical `data_dir` has one live owner, so a REST-only or MCP-only deployment
collects that signal and drops it unless the serving process weighs it itself. This module holds
no transport import so that `app.py` and `mcp.py` share one loop instead of running two.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import Protocol, runtime_checkable

from mindbridge.exceptions import MindBridgeError, ValidationError
from mindbridge.types import DeliberationReport, MemoryCapabilities

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class _Deliberating(Protocol):
    """The one control-plane entry point the periodic loop calls, kept off the served surface.

    Separate because the control plane is deliberately not a transport surface: no route and no
    tool dispatches to it, so it is not part of what an adapter serves. It is only what the
    owning process runs on a timer.
    """

    @property
    def capabilities(self) -> MemoryCapabilities: ...

    def deliberate(
        self,
        *,
        limit: int = 32,
        max_rounds: int = 4,
        idle: bool = False,
    ) -> DeliberationReport: ...


def deliberation_lifespan(
    memory: object,
    every_seconds: float,
) -> Callable[[object], AbstractAsyncContextManager[None]]:
    """Run `deliberate()` on an interval for as long as the surface is serving.

    The returned value is a lifespan: FastAPI and MCP both take one, and both ignore what it is
    handed, so the same factory serves either.
    """
    if (
        isinstance(every_seconds, bool)
        or not isinstance(every_seconds, (int, float))
        or not math.isfinite(every_seconds)
        or every_seconds <= 0
    ):
        raise ValidationError("deliberate_every must be a positive number of seconds")
    if not isinstance(memory, _Deliberating) or memory.capabilities.consolidation_model is None:
        raise ValidationError(
            "deliberate_every requires a consolidation backend on the served memory"
        )
    loop_memory = memory

    @asynccontextmanager
    async def lifespan(_served: object) -> AsyncGenerator[None, None]:
        stop = asyncio.Event()
        task = asyncio.create_task(_deliberate_periodically(loop_memory, every_seconds, stop))
        try:
            yield
        finally:
            # A round already inside `to_thread` runs to completion -- a worker thread is not
            # cancellable -- so shutdown asks the loop to stop and waits for it. Cancelling
            # raised into the `await` instead: the thread still ran to the end and still applied
            # its operations, but the task was gone before the line that reports them, so a pass
            # that changed the store on the way out was never accounted for anywhere.
            stop.set()
            await task

    return lifespan


async def _deliberate_periodically(
    memory: _Deliberating,
    every_seconds: float,
    stop: asyncio.Event,
) -> None:
    """Weigh what the served surface recorded, one round per interval, until asked to stop."""
    while True:
        # The interval is waited on the event, so shutdown does not sit out the rest of it.
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), every_seconds)
        if stop.is_set():
            return
        try:
            # Off the event loop: `deliberate()` is a synchronous model round trip, and the
            # requests this process also serves run in the same loop.
            report = await asyncio.to_thread(memory.deliberate)
        except Exception as error:
            # The class and its reason code, never the message: a rejected consolidation reply
            # is quoted back in the `ModelError` it raises, and that quote is model text about
            # stored memories. Raw memory content stays out of a WARNING the way it stays out of
            # every other kernel log line; the traceback is one level down for whoever asked.
            reason = error.reason if isinstance(error, MindBridgeError) else None
            _LOGGER.warning(
                "periodic deliberation failed: %s%s",
                type(error).__name__,
                "" if reason is None else f" ({reason})",
            )
            _LOGGER.debug("periodic deliberation failure detail", exc_info=True)
            continue
        _LOGGER.info(
            "deliberation: rounds=%d weighed=%d applied=%d rejected=%d",
            report.rounds,
            report.weighed,
            report.applied,
            report.rejected,
        )
