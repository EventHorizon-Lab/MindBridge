"""AML driver tests."""

import asyncio
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
            semaphore=asyncio.Semaphore(8),
        )

    assert [request.url.path for request in seen] == ["/aml/add", "/aml/add", "/aml/search"]
    assert sorted(request.url.path for request in seen[:2]) == ["/aml/add", "/aml/add"]
    assert len(rows) == 1
    assert rows[0]["id"] == "locomo:conv-0#q0"
    assert rows[0]["question"] == "Where did Rob move?"
    assert rows[0]["gold_answer"] == "Sweden"
    assert "Rob moved to Sweden." in str(rows[0]["retrieved_context"])

    # Inspect the two `/aml/add` request bodies directly -- asserting only
    # the request *paths* would still pass even if both chunks were sent
    # under one identical request_id, or with their messages scrambled.
    # Chunks are added concurrently, so sort by request_id instead of arrival order.
    add_bodies = sorted(
        (json.loads(request.content) for request in seen[:2]),
        key=lambda body: str(body["request_id"]),
    )
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
                semaphore=asyncio.Semaphore(8),
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
                semaphore=asyncio.Semaphore(8),
            )

    assert seen == []


class _ConcurrencyRecordingServer:
    """An async `/aml` stand-in that records overlap and add/search ordering."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.adds_finished = 0
        self.adds_finished_when_first_search_began: int | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if request.url.path == "/aml/search":
                if self.adds_finished_when_first_search_began is None:
                    self.adds_finished_when_first_search_began = self.adds_finished
                await asyncio.sleep(0)  # yield so overlapping requests can pile up
                return httpx.Response(200, json={"data": []})
            # Adds outlast searches on purpose: a search overlapping the add phase
            # then reliably observes an unfinished add count, so a lost barrier fails
            # this test instead of passing on scheduling luck.
            payload = json.loads(request.content)
            await asyncio.sleep(0.01)
            self.adds_finished += 1
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "request_id": payload["request_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                },
            )
        finally:
            self.in_flight -= 1


def _wide_case(chunks: int, questions: int) -> AmlCase:
    # chunk_messages splits every 20 messages, so 20 * chunks yields `chunks` chunks.
    return AmlCase(
        user_id="locomo:conv-0",
        messages=tuple(
            {"role": "user", "content": f"turn {index}"} for index in range(20 * chunks)
        ),
        questions=tuple(
            AmlQuestion(question_id=f"q{index}", question=f"q{index}?", payload={})
            for index in range(questions)
        ),
    )


@pytest.mark.asyncio
async def test_run_case_searches_only_after_every_add_has_finished() -> None:
    """The add phase is a barrier. A search that overlaps its own case's adds
    reads a partially written memory set and silently under-retrieves, which
    scores as a weak memory system rather than as a broken harness.
    """
    server = _ConcurrencyRecordingServer()
    case = _wide_case(chunks=6, questions=4)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server), base_url="http://test"
    ) as client:
        rows = await run_case(
            client,
            case,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            emit=emit_retrieved_context,
            semaphore=asyncio.Semaphore(8),
        )

    assert len(rows) == 4
    assert server.adds_finished == 6
    assert server.adds_finished_when_first_search_began == 6


@pytest.mark.asyncio
async def test_run_case_never_exceeds_its_request_budget() -> None:
    """One semaphore bounds requests in flight; a case must not burst past it."""
    server = _ConcurrencyRecordingServer()
    case = _wide_case(chunks=12, questions=12)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server), base_url="http://test"
    ) as client:
        await run_case(
            client,
            case,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            emit=emit_retrieved_context,
            semaphore=asyncio.Semaphore(3),
        )

    assert server.max_in_flight == 3


@pytest.mark.asyncio
async def test_run_case_returns_rows_in_question_order() -> None:
    """Questions are searched concurrently; rows must still align with input order."""
    server = _ConcurrencyRecordingServer()
    case = _wide_case(chunks=1, questions=8)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server), base_url="http://test"
    ) as client:
        rows = await run_case(
            client,
            case,
            run_id="run-1",
            benchmark="locomo",
            top_k=10,
            emit=emit_retrieved_context,
            semaphore=asyncio.Semaphore(4),
        )

    assert [row["question"] for row in rows] == [f"q{index}?" for index in range(8)]
