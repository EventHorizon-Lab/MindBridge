"""The slow loop running inside the process that serves REST or MCP.

`consolidation_candidates()` is the only reader of the `QUERY_FAILURE` rows the search, answer,
and context surfaces write, and one physical `data_dir` has one live owner -- so a deployment
that only speaks REST or MCP collects that signal and drops it unless the serving process runs
the loop itself. Everything here drives the public surfaces: `create_app` with its `/v1` routes,
and `build_mcp_server` with its tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from _feature_support import CueConsolidator, TinyEmbedder
from fastapi.testclient import TestClient
from mcp import Client

from mindbridge import (
    DeliberationReport,
    Memory,
    MemoryOperation,
    MemoryRecord,
    MemoryTrigger,
    ModelError,
    ValidationError,
)
from mindbridge.api.app import create_app
from mindbridge.api.deliberation import _deliberate_periodically
from mindbridge.api.mcp import build_mcp_server

_QUERY = "where did I leave the blue folder"
_ANSWER = "The blue folder is on the hallway shelf"
_LOOP_LOGGER = "mindbridge.api.deliberation"


class _RecordingConsolidator(CueConsolidator):
    """The shared fake, plus the triggers it was called under."""

    def __init__(self) -> None:
        self.triggers: list[MemoryTrigger] = []

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.triggers.append(trigger)
        return super().consolidate(evidence, trigger=trigger)


def _weighed(caplog: pytest.LogCaptureFixture) -> bool:
    """Whether a round has finished and said so, which is the only externally visible mark."""
    return any("deliberation: rounds=" in message for message in caplog.messages)


def _memory(tmp_path: Path, consolidator: object | None) -> Memory:
    return Memory(
        tmp_path,
        embedder=TinyEmbedder(),
        consolidator=consolidator,  # type: ignore[arg-type]
        minimum_relevance=0,
    )


def test_the_periodic_loop_is_refused_without_a_consolidation_backend(tmp_path: Path) -> None:
    with (
        _memory(tmp_path, None) as memory,
        pytest.raises(ValidationError, match="consolidation backend"),
    ):
        create_app(memory=memory, deliberate_every=0.01)


def test_the_periodic_loop_is_refused_for_a_non_positive_interval(tmp_path: Path) -> None:
    with (
        _memory(tmp_path, _RecordingConsolidator()) as memory,
        pytest.raises(ValidationError, match="positive number of seconds"),
    ):
        create_app(memory=memory, deliberate_every=0)


def test_a_serving_process_consumes_the_query_failures_its_routes_recorded(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P4: the signal REST writes is only readable by the owner, so the owner weighs it."""
    consolidator = _RecordingConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        app = create_app(memory=memory, deliberate_every=0.01)
        with (
            caplog.at_level(logging.INFO, logger=_LOOP_LOGGER),
            TestClient(app) as client,
        ):
            # Two near-equal empty recalls against an empty store: that is the whole of the
            # QUERY_FAILURE signal, and it is written by the route rather than by the SDK.
            for _attempt in range(2):
                response = client.post("/v1/memories/search", json={"query": _QUERY})
                assert response.status_code == 200
                assert response.json()["hits"] == []
            assert client.post("/v1/memories", json={"content": _ANSWER}).status_code == 201

            # Both, and inside the same deadline: exiting on the applied operation alone races
            # the line the round logs after it.
            deadline = time.monotonic() + 30.0
            while not (memory.operations() and _weighed(caplog)) and time.monotonic() < deadline:
                time.sleep(0.05)

        applied = memory.operations()

    assert applied, "the serving process never weighed the query failure it recorded"
    assert MemoryTrigger.QUERY_FAILURE in consolidator.triggers
    assert _weighed(caplog)


class _SlowConsolidator(CueConsolidator):
    """Long enough that shutdown is guaranteed to land inside a round."""

    def __init__(self) -> None:
        self.started = False
        self.finished = False

    def consolidate(
        self,
        evidence: Sequence[MemoryRecord],
        *,
        trigger: MemoryTrigger,
    ) -> tuple[MemoryOperation, ...]:
        self.started = True
        time.sleep(0.5)
        operations = super().consolidate(evidence, trigger=trigger)
        self.finished = True
        return operations


def test_shutdown_lets_the_round_it_interrupts_finish_and_report(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cancelling the loop task abandoned a pass that had already changed the store.

    A round runs in a worker thread, which is not cancellable, so it applied its operations
    either way -- but the task raised out of the `await` before the line that reports what it
    applied, leaving the only account of the pass unwritten. Shutdown asks the loop to stop.
    """
    consolidator = _SlowConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        app = create_app(memory=memory, deliberate_every=0.01)
        with caplog.at_level(logging.INFO, logger=_LOOP_LOGGER), TestClient(app) as client:
            for _attempt in range(2):
                found = client.post("/v1/memories/search", json={"query": _QUERY})
                assert found.status_code == 200
            assert client.post("/v1/memories", json={"content": _ANSWER}).status_code == 201

            deadline = time.monotonic() + 30.0
            while not consolidator.started and time.monotonic() < deadline:
                time.sleep(0.01)
            interrupted = consolidator.started and not consolidator.finished

        assert interrupted, "the round had already finished, so shutdown interrupted nothing"
        assert consolidator.finished
        assert memory.operations()
    # Every round logs the line, including the empty ones this interval fires before anything is
    # recorded, so the assertion is that the round which *applied* something is among them.
    assert [
        message
        for message in caplog.messages
        if "deliberation: rounds=" in message and "applied=0" not in message
    ], "the interrupted round applied its pass and reported nothing"


async def test_a_failed_round_names_its_failure_without_quoting_the_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning carries the failure class and its reason code, and nothing the model wrote.

    An invalid consolidation reply is rejected whole, and the `ModelError` quotes 400 characters
    of it back so the operator can see which key was wrong. That quote is model text about
    stored memories, so `exc_info=True` on the warning put remembered content into every
    operator's log at default level. The traceback moved to DEBUG, where it is asked for.
    """
    quoted = f"consolidation response was invalid (model returned: {_ANSWER})"

    class _Failing:
        def deliberate(self) -> DeliberationReport:
            raise ModelError(quoted, reason="response_invalid")

    stop = asyncio.Event()
    with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
        task = asyncio.create_task(_deliberate_periodically(cast(Any, _Failing()), 0.001, stop))
        for _poll in range(600):
            if any(record.levelno == logging.WARNING for record in caplog.records):
                break
            await asyncio.sleep(0.05)
        stop.set()
        await task

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings, "a failed round said nothing at all"
    assert "ModelError" in warnings[0].getMessage()
    assert "response_invalid" in warnings[0].getMessage()
    assert _ANSWER not in warnings[0].getMessage()
    assert warnings[0].exc_info is None, "the warning still carries the quoted reply"
    assert any(record.levelno == logging.DEBUG and record.exc_info for record in caplog.records)


def test_the_mcp_loop_is_refused_without_a_consolidation_backend(tmp_path: Path) -> None:
    with (
        _memory(tmp_path, None) as memory,
        pytest.raises(ValidationError, match="consolidation backend"),
    ):
        build_mcp_server(memory, deliberate_every=0.01)


def test_the_mcp_loop_is_refused_for_a_non_positive_interval(tmp_path: Path) -> None:
    with (
        _memory(tmp_path, _RecordingConsolidator()) as memory,
        pytest.raises(ValidationError, match="positive number of seconds"),
    ):
        build_mcp_server(memory, deliberate_every=0)


async def test_a_serving_mcp_process_consumes_the_query_failures_its_tools_recorded(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same P4 obligation on the other surface: stdio blocks forever, so it must weigh too."""
    consolidator = _RecordingConsolidator()
    with _memory(tmp_path, consolidator) as memory:
        server = build_mcp_server(memory, deliberate_every=0.01)
        with caplog.at_level(logging.INFO, logger=_LOOP_LOGGER):
            async with Client(server) as client:
                for _attempt in range(2):
                    found = await client.call_tool("search_memories", {"query": _QUERY})
                    assert found.structured_content == {"hits": [], "trace": None}
                await client.call_tool("add_memory", {"content": _ANSWER})

                # Counted rather than timed only because ASYNC110 refuses an awaited sleep in a
                # `while`; 600 * 0.05s is the same 30s deadline the REST test waits.
                for _poll in range(600):
                    if memory.operations() and _weighed(caplog):
                        break
                    await asyncio.sleep(0.05)

        applied = memory.operations()

    assert applied, "the MCP server never weighed the query failure its tools recorded"
    assert MemoryTrigger.QUERY_FAILURE in consolidator.triggers
    assert _weighed(caplog)
