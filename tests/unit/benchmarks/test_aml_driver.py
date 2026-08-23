"""Failure accounting for the AML driver's add phase.

The rest of the driver's contract is covered by `tests/unit/benchmarks/aml/test_driver.py`;
this module owns the one case where a chunk never lands.
"""

import asyncio
import json
from collections.abc import Callable

import httpx

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.aml.driver import emit_retrieved_context, run_case


def _handler(failing_suffix: str) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/aml/add":
            payload = json.loads(request.content)
            if str(payload["request_id"]).endswith(failing_suffix):
                return httpx.Response(503, json={"detail": "add failed"})
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": payload["request_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                },
            )
        return httpx.Response(
            200,
            json={"data": [{"id": "mem_1", "content": "Rob moved to Sweden.", "score": 0.9}]},
        )

    return handle


async def test_run_case_still_searches_when_one_chunk_never_adds() -> None:
    """A chunk the server rejects used to discard every chunk added beside it."""
    transport = httpx.MockTransport(_handler(":chunk-1"))
    case = AmlCase(
        user_id="locomo:conv-0",
        messages=tuple({"role": "user", "content": f"turn {index}"} for index in range(25)),
        questions=(
            AmlQuestion(
                question_id="q0",
                question="Where did Rob move?",
                payload={"gold_answer": "Sweden"},
            ),
        ),
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        rows = await run_case(
            client,
            case,
            run_id="run-1",
            benchmark="locomo",
            top_k=100,
            emit=emit_retrieved_context,
            semaphore=asyncio.Semaphore(8),
        )

    assert len(rows) == 1
    assert rows[0]["id"] == "locomo:conv-0#q0"
    assert "Rob moved to Sweden." in str(rows[0]["retrieved_context"])
    # The row is scored, so it has to say the memory behind it is incomplete.
    assert rows[0]["mindbridge_ingest_failure_count"] == 1
