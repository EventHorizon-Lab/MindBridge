"""Contract tests for the small asynchronous MindBridge SDK."""

import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from mindbridge import AsyncMindBridge, MindBridgeClientError
from mindbridge.contracts import RecallQuery, RecallRequest


async def test_recall_uses_shared_request_and_response_contracts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/recall"
        assert json.loads(request.content) == {
            "tenant_id": "tenant_01",
            "query": {"text": "Where is the tool?", "media_object_ids": []},
            "filters": {
                "person_ids": [],
                "device_ids": [],
                "memory_types": [],
                "occurred_after": None,
                "occurred_before": None,
            },
            "mode": "answer",
            "limit": 20,
            "include_evidence": True,
        }
        return httpx.Response(
            200,
            json={
                "answer": None,
                "confidence": 0.0,
                "memories": [],
                "evidence": [],
                "trace_id": "trace_01",
            },
        )

    client = _client(respond)
    try:
        result = await client.recall(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="Where is the tool?"),
            )
        )
    finally:
        await client.close()

    assert result.trace_id == "trace_01"


async def test_typed_api_error_preserves_retry_information() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "code": "model_unavailable",
                "message": "memory model is unavailable",
                "trace_id": "trace_failure",
                "issues": [],
            },
        )

    client = _client(respond)
    try:
        with pytest.raises(MindBridgeClientError) as failure:
            await client.recall(
                RecallRequest(
                    tenant_id="tenant_01",
                    query=RecallQuery(text="Where is the tool?"),
                )
            )
    finally:
        await client.close()

    assert failure.value.status_code == 503
    assert failure.value.code == "model_unavailable"
    assert failure.value.trace_id == "trace_failure"


async def test_get_observation_job_uses_tenant_scoped_route() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/jobs/job_01"
        assert request.url.params["tenant_id"] == "tenant_01"
        return httpx.Response(
            200,
            json={
                "job_id": "job_01",
                "observation_id": "observation_01",
                "state": "succeeded",
                "attempt": 1,
                "error_code": None,
                "created_at": "2026-08-11T12:00:00Z",
                "updated_at": "2026-08-11T12:01:00Z",
                "trace_id": "trace_job",
            },
        )

    client = _client(respond)
    try:
        job = await client.get_observation_job("tenant_01", "job_01")
    finally:
        await client.close()

    assert job.state.value == "succeeded"
    assert job.attempt == 1


def _client(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> AsyncMindBridge:
    return AsyncMindBridge(
        httpx.AsyncClient(
            base_url="https://memory.example.test/",
            transport=httpx.MockTransport(handler),
        )
    )
