"""REST admission precedes body buffering and shares streaming capacity with the SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Sequence
from contextlib import ExitStack
from pathlib import Path

import pytest
from _feature_support import TinyEmbedder
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from mindbridge import (
    AnswerChunk,
    AnswerPolicy,
    AnswerResult,
    AsyncMemory,
    Memory,
    Modality,
    ModelInput,
    SearchHit,
)
from mindbridge.api.app import create_app


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
    def chunks() -> Generator[AnswerChunk, None, AnswerResult]:
        yield AnswerChunk(text="answer")
        return AnswerResult(answer="answer")

    with Memory(tmp_path, embedder=TinyEmbedder(), answerer=_Answerer()) as memory:
        memory.add("reference evidence")
        facade = AsyncMemory(memory)
        streams = [facade._pump(chunks()) for _ in range(8)]
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
