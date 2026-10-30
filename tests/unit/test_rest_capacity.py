"""REST admission precedes body buffering and shares streaming capacity with the SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from mindbridge import (
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Memory,
    Modality,
    ModelInput,
    SearchHit,
)
from mindbridge.api.app import create_app


@pytest.mark.asyncio
async def test_terminal_streams_release_workers_before_slow_network_sends(tmp_path: Path) -> None:
    blocked = 0
    all_blocked, release = asyncio.Event(), asyncio.Event()
    with Memory(tmp_path, embedder=TinyEmbedder(), answerer=_Answerer()) as memory:
        memory.add("reference evidence")
        app = create_app(memory=memory)

        async def request(*, slow: bool) -> list[Message]:
            scope: Scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "method": "POST",
                "path": "/v1/answers/stream",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            }
            received = False
            messages: list[Message] = []

            async def receive() -> Message:
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": b'{"question":"reference evidence"}'}
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

            async def send(message: Message) -> None:
                nonlocal blocked
                messages.append(message)
                if slow and b"event: result" in message.get("body", b""):
                    blocked += 1
                    if blocked == 8:
                        all_blocked.set()
                    await release.wait()

            await app(scope, receive, send)
            return messages

        calls = [asyncio.create_task(request(slow=True)) for _ in range(8)]
        try:
            await asyncio.wait_for(all_blocked.wait(), timeout=5)
            ninth = await asyncio.wait_for(request(slow=False), timeout=2)
            assert ninth[0]["status"] == 200
            assert any(b"event: result" in message.get("body", b"") for message in ninth)
        finally:
            release.set()
            await asyncio.gather(*calls)


class _Answerer:
    generation_capabilities = frozenset({Modality.TEXT})

    def answer(
        self,
        question: ModelInput,
        hits: Sequence[SearchHit],
        *,
        answer_policy: AnswerPolicy = "strict",
        exhaustive: bool = False,
    ) -> AnswerResult:
        return AnswerResult(answer="reference evidence", hits=tuple(hits))

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_http_overload_is_rejected_before_reading_the_body(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        app = create_app(memory=memory)
        received = 0
        full = asyncio.Event()
        release = asyncio.Event()
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "GET",
            "path": "/v1/memories",
            "query_string": b"",
            "headers": [],
        }

        async def delayed_receive() -> Message:
            nonlocal received
            received += 1
            if received == 32:
                full.set()
            await release.wait()
            return {"type": "http.request", "body": b""}

        async def unexpected_receive() -> Message:
            pytest.fail("overloaded request read its body")

        messages: list[Message] = []

        async def send(message: Message) -> None:
            messages.append(message)

        requests = [asyncio.create_task(app(dict(scope), delayed_receive, send)) for _ in range(32)]
        try:
            await asyncio.wait_for(full.wait(), timeout=2)
            await asyncio.wait_for(app(dict(scope), unexpected_receive, send), timeout=1)
            assert messages[0]["status"] == 503
            headers = dict(messages[0]["headers"])
            assert headers[b"retry-after"] == b"1"
            assert json.loads(messages[1]["body"])["reason"] == "capacity_exceeded"
            health_scope = {**scope, "path": "/healthz"}
            await app(health_scope, unexpected_receive, send)
            assert messages[2]["status"] == 200
        finally:
            release.set()
            await asyncio.gather(*requests)


def test_real_operation_overload_has_the_retryable_error_envelope(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder()) as memory:
        with TestClient(create_app(memory=memory)) as client, ExitStack() as operations:
            for _ in range(32):
                operations.enter_context(memory._lifecycle.operation())
            response = client.get("/v1/memories")
            assert response.status_code == 503
            assert response.headers["Retry-After"] == "1"
            assert response.json()["reason"] == "capacity_exceeded"
            assert response.json()["retryable"] is True
        assert memory.list().items == ()


@pytest.mark.asyncio
async def test_rest_and_async_sdk_share_stream_capacity(tmp_path: Path) -> None:
    with Memory(tmp_path, embedder=TinyEmbedder(), answerer=_Answerer()) as memory:
        memory.add("reference evidence")
        facade = AsyncMemory(memory)
        streams = [facade.ask_stream("reference evidence") for _ in range(8)]
        with TestClient(create_app(memory=memory)) as client:
            try:
                for stream in streams:
                    await anext(stream)
                response = client.post(
                    "/v1/answers/stream", json={"question": "reference evidence"}
                )
                assert response.status_code == 503
                assert response.json()["reason"] == "capacity_exceeded"
                assert response.headers["Retry-After"] == "1"
            finally:
                for stream in streams:
                    await stream.aclose()
            response = client.post("/v1/answers/stream", json={"question": "reference evidence"})
            assert response.status_code == 200
            assert "event: result" in response.text
