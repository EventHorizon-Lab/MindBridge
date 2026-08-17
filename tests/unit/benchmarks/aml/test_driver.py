"""AML driver tests."""

from collections.abc import Callable

import httpx
import pytest

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.aml.driver import emit_retrieved_context, run_case


def _handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/aml/add":
            import json

            payload = json.loads(request.content)
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


@pytest.mark.asyncio
async def test_run_case_adds_every_chunk_then_emits_one_row_per_question() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_handler(seen))
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
        )

    assert [request.url.path for request in seen] == ["/aml/add", "/aml/add", "/aml/search"]
    assert len(rows) == 1
    assert rows[0]["id"] == "locomo:conv-0#q0"
    assert rows[0]["question"] == "Where did Rob move?"
    assert rows[0]["gold_answer"] == "Sweden"
    assert "Rob moved to Sweden." in str(rows[0]["retrieved_context"])


@pytest.mark.asyncio
async def test_run_case_fails_loudly_when_add_does_not_echo() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/aml/add":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": "wrong",
                    "user_id": "u",
                    "session_id": "s",
                },
            )
        return httpx.Response(200, json={"data": []})

    case = AmlCase(
        user_id="locomo:conv-0",
        messages=({"role": "user", "content": "hi"},),
        questions=(AmlQuestion(question_id="q0", question="?", payload={}),),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="http://test"
    ) as client:
        with pytest.raises(ValueError, match="did not echo"):
            await run_case(
                client,
                case,
                run_id="run-1",
                benchmark="locomo",
                top_k=10,
                emit=emit_retrieved_context,
            )


@pytest.mark.asyncio
async def test_run_case_raises_before_any_request_when_user_id_would_overflow() -> None:
    seen: list[httpx.Request] = []
    transport = httpx.MockTransport(_handler(seen))
    case = AmlCase(
        user_id="locomo:" + "x" * 250,
        messages=({"role": "user", "content": "hi"},),
        questions=(AmlQuestion(question_id="q0", question="?", payload={}),),
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(ValueError, match="Identifier limit"):
            await run_case(
                client,
                case,
                run_id="run-1",
                benchmark="locomo",
                top_k=10,
                emit=emit_retrieved_context,
            )

    assert seen == []
