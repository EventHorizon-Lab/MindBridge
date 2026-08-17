"""AML driver tests."""

import json
from collections.abc import Callable

import httpx
import pytest

from mindbridge.benchmarks.aml.cases import AmlCase, AmlQuestion
from mindbridge.benchmarks.aml.driver import (
    emit_chat_history,
    emit_context_messages,
    emit_retrieved_context,
    emit_selected,
    run_case,
)


def _handler(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/aml/add":
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

    # Inspect the two `/aml/add` request bodies directly -- asserting only
    # the request *paths* would still pass even if both chunks were sent
    # under one identical request_id, or with their messages scrambled.
    add_bodies = [json.loads(request.content) for request in seen[:2]]
    assert add_bodies[0]["request_id"].endswith("chunk-0")
    assert add_bodies[1]["request_id"].endswith("chunk-1")
    assert add_bodies[0]["request_id"] != add_bodies[1]["request_id"]
    assert add_bodies[0]["user_id"] == add_bodies[1]["user_id"]
    assert add_bodies[0]["session_id"] == add_bodies[1]["session_id"]

    # Messages must arrive in their original order across the chunk
    # boundary: the last message of chunk 0 immediately precedes the first
    # message of chunk 1 in the un-chunked history.
    chunk0_contents = [m["content"] for m in add_bodies[0]["messages"]]
    chunk1_contents = [m["content"] for m in add_bodies[1]["messages"]]
    assert chunk0_contents + chunk1_contents == [f"turn {index}" for index in range(25)]
    last_of_chunk0 = int(chunk0_contents[-1].removeprefix("turn "))
    first_of_chunk1 = int(chunk1_contents[0].removeprefix("turn "))
    assert first_of_chunk1 == last_of_chunk0 + 1


def test_emit_selected_returns_retrieval_selected_with_text_and_created_at() -> None:
    question = AmlQuestion(question_id="q0", question="?", payload={})
    retrieved: list[dict[str, object]] = [
        {"content": "Rob moved to Sweden.", "created_at": "2024-01-01T00:00:00Z"},
        {"content": "Rob likes coffee."},  # /aml/search may omit created_at entirely.
    ]

    result = emit_selected(question, retrieved)

    assert result == {
        "retrieval": {
            "selected": [
                {"text": "Rob moved to Sweden.", "created_at": "2024-01-01T00:00:00Z"},
                {"text": "Rob likes coffee.", "created_at": None},
            ]
        }
    }


def test_emit_context_messages_returns_role_user_content_list() -> None:
    question = AmlQuestion(question_id="q0", question="?", payload={})
    retrieved: list[dict[str, object]] = [
        {"content": "Rob moved to Sweden.", "created_at": "2024-01-01T00:00:00Z"},
        {"content": "Rob likes coffee."},
    ]

    result = emit_context_messages(question, retrieved)

    assert result == {
        "context_messages": [
            {"role": "user", "content": "Rob moved to Sweden."},
            {"role": "user", "content": "Rob likes coffee."},
        ]
    }


def test_emit_chat_history_returns_the_same_shape_under_chat_history() -> None:
    question = AmlQuestion(question_id="q0", question="?", payload={})
    retrieved: list[dict[str, object]] = [
        {"content": "Rob moved to Sweden.", "created_at": "2024-01-01T00:00:00Z"},
        {"content": "Rob likes coffee."},
    ]

    result = emit_chat_history(question, retrieved)

    assert result == {
        "chat_history": [
            {"role": "user", "content": "Rob moved to Sweden."},
            {"role": "user", "content": "Rob likes coffee."},
        ]
    }


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
