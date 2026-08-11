"""Contract tests for the small asynchronous MindBridge SDK."""

import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from mindbridge import AsyncMindBridge, MindBridgeClientError
from mindbridge.contracts import FeedbackRequest, ForgetRequest, RecallQuery, RecallRequest
from mindbridge.core import FeedbackType, ForgetTargetType


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


async def test_get_memory_uses_tenant_scoped_route() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/v1/memories/memory_01"
        assert request.url.params["tenant_id"] == "tenant_01"
        return httpx.Response(
            200,
            json={
                "memory_id": "memory_01",
                "memory_type": "episodic",
                "summary": "remembered event",
                "evidence_ids": [],
                "occurred_at": "2026-08-11T12:00:00Z",
                "ended_at": "2026-08-11T12:00:00Z",
                "created_at": "2026-08-11T12:00:00Z",
                "verification_status": "attested",
                "state": "active",
            },
        )

    client = _client(respond)
    try:
        memory = await client.get_memory("tenant_01", "memory_01")
    finally:
        await client.close()

    assert memory.memory_id == "memory_01"


async def test_feedback_uses_shared_contracts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/feedback"
        assert json.loads(request.content)["feedback_type"] == "useful"
        return httpx.Response(
            201,
            json={
                "feedback_id": "feedback_01",
                "feedback_type": "useful",
                "memory_id": "memory_01",
                "corrected_memory_id": None,
                "resulting_state": "strengthened",
                "resulting_strength": 1.5,
                "created_at": "2026-08-11T12:00:00Z",
                "trace_id": "trace_feedback",
            },
        )

    client = _client(respond)
    try:
        receipt = await client.record_feedback(
            FeedbackRequest(
                tenant_id="tenant_01",
                feedback_type=FeedbackType.USEFUL,
                memory_id="memory_01",
            )
        )
    finally:
        await client.close()

    assert receipt.resulting_state is not None
    assert receipt.resulting_strength == 1.5


async def test_forget_and_status_use_shared_contracts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            assert request.url.path == "/v1/forget"
            assert json.loads(request.content)["target_type"] == "memory_record"
            trace_id = "trace_forget"
        else:
            assert request.url.path == "/v1/deletions/tombstone_01"
            assert request.url.params["tenant_id"] == "tenant_01"
            trace_id = "trace_status"
        return httpx.Response(
            200,
            json={
                "tombstone_id": "tombstone_01",
                "target_type": "memory_record",
                "target_id": "memory_01",
                "propagation_state": "complete",
                "requested_at": "2026-08-12T08:00:00Z",
                "completed_at": "2026-08-12T08:00:00Z",
                "error_code": None,
                "trace_id": trace_id,
            },
        )

    client = _client(respond)
    try:
        forgotten = await client.forget(
            ForgetRequest(
                tenant_id="tenant_01",
                target_type=ForgetTargetType.MEMORY_RECORD,
                target_id="memory_01",
            )
        )
        status = await client.get_forget_status("tenant_01", forgotten.tombstone_id)
    finally:
        await client.close()

    assert forgotten.trace_id == "trace_forget"
    assert status.trace_id == "trace_status"


def _client(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> AsyncMindBridge:
    return AsyncMindBridge(
        httpx.AsyncClient(
            base_url="https://memory.example.test/",
            transport=httpx.MockTransport(handler),
        )
    )
