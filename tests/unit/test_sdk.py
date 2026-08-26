"""Contract tests for the small asynchronous MindBridge SDK."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone

import httpx
import pytest

from mindbridge import MindBridge, MindBridgeError
from mindbridge.contracts import (
    DeletionListRequest,
    FeedbackRequest,
    ForgetRequest,
    MemoryWriteStatus,
    RecallQuery,
    RecallRequest,
    RememberRequest,
)
from mindbridge.core import FeedbackType, ForgetTargetType, JobState, MemoryType


async def test_context_manager_closes_the_connection_pool() -> None:
    http_client = httpx.AsyncClient(base_url="https://memory.example.test/")
    memory = MindBridge(http_client)

    async with memory as opened:
        assert opened is memory
        assert not http_client.is_closed

    assert http_client.is_closed


async def test_recall_uses_shared_request_and_response_contracts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/recall"
        assert json.loads(request.content) == {
            "tenant_id": "tenant_01",
            "query": {"text": "Where is the tool?", "media_object_ids": []},
            "memory_ids": [],
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
                "issues": [
                    {
                        "location": ["body", "query"],
                        "message": "Field required",
                        "code": "missing",
                    }
                ],
            },
        )

    client = _client(respond)
    try:
        with pytest.raises(MindBridgeError) as failure:
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
    assert failure.value.issues[0].location == ("body", "query")


async def test_transport_failure_names_the_underlying_cause() -> None:
    """ "MindBridge request failed" alone sent operators to the server log to learn whether the
    deployment was down, the name was wrong, or the request had simply timed out."""

    async def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    client = _client(refuse)
    try:
        with pytest.raises(MindBridgeError) as failure:
            await client.recall(
                RecallRequest(
                    tenant_id="tenant_01",
                    query=RecallQuery(text="Where is the tool?"),
                )
            )
    finally:
        await client.close()

    assert failure.value.code == "transport_error"
    assert "All connection attempts failed" in str(failure.value)
    assert "v1/recall" in str(failure.value)


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
                "memory_ids": ["memory_01"],
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
    assert job.memory_ids == ("memory_01",)


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
                "evidence": [
                    {
                        "evidence_id": "evidence_01",
                        "media_object_id": "media_01",
                        "start_ms": 0,
                        "end_ms": 4_000,
                        "media_url": "https://objects.example.test/media_01",
                        "media_url_expires_at": "2026-08-11T12:05:00Z",
                    }
                ],
                "trace_id": "trace_memory",
            },
        )

    client = _client(respond)
    try:
        memory = await client.get_memory("tenant_01", "memory_01")
    finally:
        await client.close()

    assert memory.memory_id == "memory_01"
    assert memory.evidence[0].evidence_id == "evidence_01"
    assert memory.trace_id == "trace_memory"


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
        elif request.url.path.endswith("tombstone_01"):
            assert request.url.path == "/v1/deletions/tombstone_01"
            assert request.url.params["tenant_id"] == "tenant_01"
            trace_id = "trace_status"
        else:
            assert request.url.path == "/v1/deletions"
            assert request.url.params["limit"] == "1"
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "tombstone_id": "tombstone_01",
                            "target_type": "memory_record",
                            "target_id": "memory_01",
                            "propagation_state": "complete",
                            "requested_at": "2026-08-12T08:00:00Z",
                            "completed_at": "2026-08-12T08:00:00Z",
                            "error_code": None,
                        }
                    ],
                    "next_cursor": None,
                    "trace_id": "trace_page",
                },
            )
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
        page = await client.list_deletions(DeletionListRequest(tenant_id="tenant_01", limit=1))
    finally:
        await client.close()

    assert forgotten.trace_id == "trace_forget"
    assert status.trace_id == "trace_status"
    assert page.items[0].tombstone_id == "tombstone_01"


async def test_stream_observation_job_yields_each_state_with_a_resume_id() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/jobs/job_01/events"
        assert request.url.params["tenant_id"] == "tenant_01"
        assert request.headers["accept"] == "text/event-stream"
        assert "last-event-id" not in request.headers
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                ": keepalive\n\n"
                "id: 100\nevent: job\ndata: " + _job_payload("running", ()) + "\n\n"
                "event: heartbeat\ndata: {}\n\n"
                "id: 200\nevent: job\ndata: " + _job_payload("succeeded", ("memory_01",)) + "\n\n"
            ).encode("utf-8"),
        )

    client = _client(respond)
    try:
        events = [event async for event in client.stream_observation_job("tenant_01", "job_01")]
    finally:
        await client.close()

    assert [event.event_id for event in events] == ["100", "200"]
    assert [event.job.state for event in events] == [JobState.RUNNING, JobState.SUCCEEDED]
    assert events[-1].job.memory_ids == ("memory_01",)


async def test_stream_observation_job_resumes_from_the_supplied_event_id() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["last-event-id"] == "100"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                "id: 200\nevent: job\ndata: " + _job_payload("succeeded", ("memory_01",)) + "\n\n"
            ).encode("utf-8"),
        )

    client = _client(respond)
    try:
        events = [
            event
            async for event in client.stream_observation_job(
                "tenant_01",
                "job_01",
                last_event_id="100",
            )
        ]
    finally:
        await client.close()

    assert [event.event_id for event in events] == ["200"]


async def test_stream_observation_job_raises_a_typed_error_frame() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'event: error\ndata: {"code":"database_unavailable",'
                b'"message":"memory storage is temporarily unavailable",'
                b'"trace_id":"trace_stream","issues":[]}\n\n'
            ),
        )

    client = _client(respond)
    try:
        with pytest.raises(MindBridgeError) as raised:
            async for _event in client.stream_observation_job("tenant_01", "job_01"):
                raise AssertionError("an error frame must not yield a job event")
    finally:
        await client.close()

    assert raised.value.code == "database_unavailable"
    assert raised.value.trace_id == "trace_stream"


async def test_stream_observation_job_reports_a_rejected_request() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "code": "job_not_found",
                "message": "observation processing job does not exist",
                "trace_id": "trace_missing",
                "issues": [],
            },
        )

    client = _client(respond)
    try:
        with pytest.raises(MindBridgeError) as raised:
            async for _event in client.stream_observation_job("tenant_01", "missing"):
                raise AssertionError("a rejected request must not yield a job event")
    finally:
        await client.close()

    assert raised.value.code == "job_not_found"
    assert raised.value.status_code == 404


async def test_remember_many_sends_one_request_for_the_whole_batch() -> None:
    """A caller holding N memories should cost one round trip, not N.

    The kernel has taken a batch all along -- one encoder call instead of N -- while every
    caller-facing surface was single-item.
    """
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        return httpx.Response(
            201,
            json={
                "memories": [
                    _remembered(index, memory["summary"])
                    for index, memory in enumerate(payload["memories"])
                ]
            },
        )

    client = _client(respond)
    try:
        results = await client.remember_many(
            [
                RememberRequest(
                    tenant_id="tenant_01",
                    summary=summary,
                    memory_type=MemoryType.EPISODIC,
                    occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
                )
                for summary in ("first", "second")
            ]
        )
    finally:
        await client.close()

    assert len(requests) == 1
    assert requests[0].url.path == "/v1/memories/batch"
    assert [memory["summary"] for memory in json.loads(requests[0].content)["memories"]] == [
        "first",
        "second",
    ]
    assert tuple(result.summary for result in results) == ("first", "second")
    assert results[0].status is MemoryWriteStatus.CREATED


def _remembered(index: int, summary: str) -> dict[str, object]:
    return {
        "status": "created",
        "memory_id": f"memory_{index:02d}",
        "memory_type": "episodic",
        "summary": summary,
        "evidence_ids": [],
        "occurred_at": "2026-08-11T12:00:00Z",
        "ended_at": "2026-08-11T12:00:00Z",
        "created_at": "2026-08-11T12:00:00Z",
        "verification_status": "attested",
        "state": "active",
        "evidence": [],
        "trace_id": "trace_remember",
    }


def _job_payload(state: str, memory_ids: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "job_id": "job_01",
            "observation_id": "observation_01",
            "state": state,
            "attempt": 1,
            "error_code": None,
            "memory_ids": list(memory_ids),
            "created_at": "2026-08-11T12:00:00Z",
            "updated_at": "2026-08-11T12:00:01Z",
            "trace_id": "trace_stream",
        }
    )


def _client(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> MindBridge:
    return MindBridge(
        httpx.AsyncClient(
            base_url="https://memory.example.test/",
            transport=httpx.MockTransport(handler),
        )
    )


def _retrying_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    delays: list[float],
    retry_attempts: int = 3,
) -> MindBridge:
    """A client whose waits are recorded instead of slept, so retries are testable."""

    async def record(delay: float) -> None:
        delays.append(delay)

    return MindBridge(
        httpx.AsyncClient(
            base_url="https://memory.example.test/",
            transport=httpx.MockTransport(handler),
        ),
        retry_attempts=retry_attempts,
        retry_backoff_seconds=0.5,
        sleep=record,
    )


def _memory_body(**extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "memory_id": "memory_01",
        "memory_type": "semantic",
        "summary": "The tool is on the bench.",
        "evidence_ids": [],
        "occurred_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
        "verification_status": "attested",
        "state": "active",
        "salience": 0.5,
        "strength": 1.0,
        "useful_access_count": 0,
        "positive_feedback_count": 0,
        "negative_feedback_count": 0,
        "last_accessed_at": None,
        "supersedes_memory_id": None,
        "superseded_at": None,
        "evidence": [],
        "trace_id": "trace_01",
    }
    body.update(extra)
    return body


def _remember(idempotency_key: str | None) -> RememberRequest:
    return RememberRequest(
        tenant_id="tenant_01",
        summary="The tool is on the bench.",
        memory_type=MemoryType.SEMANTIC,
        occurred_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        idempotency_key=idempotency_key,
    )


async def test_a_read_survives_a_dependency_that_is_briefly_unavailable() -> None:
    """A 503 from a restarting model used to be final for the caller."""
    statuses = [503, 200]
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status != 200:
            return httpx.Response(
                status,
                json={"code": "model_unavailable", "message": "restarting", "trace_id": "t1"},
            )
        return httpx.Response(200, json=_memory_body())

    async with _retrying_client(handler, delays=delays) as memory:
        result = await memory.get_memory("tenant_01", "memory_01")

    assert result.memory_id == "memory_01"
    assert statuses == [], "the second attempt never happened"
    assert len(delays) == 1
    assert 0.0 <= delays[0] <= 0.5


async def test_a_read_survives_a_dropped_connection() -> None:
    """`httpx.RemoteProtocolError` mid-stream escaped as a terminal transport_error."""
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        return httpx.Response(200, json=_memory_body())

    async with _retrying_client(handler, delays=delays) as memory:
        result = await memory.get_memory("tenant_01", "memory_01")

    assert (attempts, result.memory_id) == (2, "memory_01")


async def test_a_write_with_no_idempotency_key_is_repeated_byte_for_byte() -> None:
    """Omission is the supported default, and the server keys such a write on its content.

    So the retry has to cover it -- gating on a caller-supplied key made the whole feature
    inert on every default call. What makes it safe is that the resend is identical: the same
    body derives the same key, and the server answers `duplicate` with the first outcome
    instead of writing twice.
    """
    statuses = [503, 200]
    bodies: list[bytes] = []
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if statuses.pop(0) != 200:
            return httpx.Response(
                503, json={"code": "model_unavailable", "message": "restarting", "trace_id": "t1"}
            )
        return httpx.Response(200, json=_memory_body(status=MemoryWriteStatus.CREATED.value))

    async with _retrying_client(handler, delays=delays) as memory:
        result = await memory.remember(_remember(None))

    assert (statuses, result.memory_id) == ([], "memory_01")
    assert len(bodies) == 2 and bodies[0] == bodies[1], "the resend must derive the same key"
    assert len(delays) == 1


async def test_a_write_that_names_an_idempotency_key_is_repeated() -> None:
    """The server stores the first outcome under the key, so a repeat cannot double-write."""
    statuses = [503, 200]
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if statuses.pop(0) != 200:
            return httpx.Response(
                503, json={"code": "model_unavailable", "message": "x", "trace_id": "t1"}
            )
        return httpx.Response(200, json=_memory_body(status=MemoryWriteStatus.CREATED.value))

    async with _retrying_client(handler, delays=delays) as memory:
        result = await memory.remember(_remember("write_01"))

    assert (statuses, result.memory_id) == ([], "memory_01")


async def test_repeat_safety_is_decided_by_the_contract_not_by_a_supplied_key() -> None:
    """A write states its idempotency in its schema; one that does not is not repeated.

    That is the property that keeps the gate honest as the API grows -- a new write endpoint
    has to declare `idempotency_key`, whose documented meaning is "omit it and one is derived
    from the content", before the client will send it twice. `RecallRequest` declares none and
    is retried only because `recall()` passes `repeat_is_safe` explicitly, which is a decision
    about one read rather than a hole in the rule.
    """
    from mindbridge.contracts import RememberBatchRequest
    from mindbridge.sdk import _repeat_is_safe

    assert _repeat_is_safe(_remember(None)) is True
    assert _repeat_is_safe(_remember("write_01")) is True
    assert _repeat_is_safe(RememberBatchRequest(memories=(_remember("a"), _remember(None)))) is True
    assert "idempotency_key" not in RecallRequest.model_fields
    assert (
        _repeat_is_safe(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="where is the tool?"))
        )
        is False
    )


async def test_a_rejected_request_is_not_repeated() -> None:
    """A 4xx is a request the server understood; repeating it spends the rejection twice."""
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            404, json={"code": "not_found", "message": "no such memory", "trace_id": "t1"}
        )

    async with _retrying_client(handler, delays=delays) as memory:
        with pytest.raises(MindBridgeError) as raised:
            await memory.get_memory("tenant_01", "memory_01")

    assert (attempts, delays, raised.value.code) == (1, [], "not_found")


async def test_recall_is_repeated_even_though_it_names_no_key() -> None:
    """Recall is the read the outage broke; it opts in explicitly in `recall`."""
    statuses = [503, 200]
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if statuses.pop(0) != 200:
            return httpx.Response(
                503, json={"code": "model_unavailable", "message": "x", "trace_id": "t1"}
            )
        return httpx.Response(
            200,
            json={
                "answer": "On the bench.",
                "confidence": 0.9,
                "memories": [],
                "evidence": [],
                "trace_id": "trace_01",
            },
        )

    async with _retrying_client(handler, delays=delays) as memory:
        result = await memory.recall(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is the tool?"))
        )

    assert (statuses, result.answer) == ([], "On the bench.")


async def test_the_attempt_budget_is_a_ceiling_and_one_attempt_disables_retrying() -> None:
    """A caller running its own retry loop must be able to turn this one off."""
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            503, json={"code": "model_unavailable", "message": "x", "trace_id": "t1"}
        )

    async with _retrying_client(handler, delays=delays, retry_attempts=3) as memory:
        with pytest.raises(MindBridgeError):
            await memory.get_memory("tenant_01", "memory_01")
    budgeted = attempts

    attempts = 0
    single: list[float] = []
    async with _retrying_client(handler, delays=single, retry_attempts=1) as memory:
        with pytest.raises(MindBridgeError):
            await memory.get_memory("tenant_01", "memory_01")

    assert (budgeted, len(delays)) == (3, 2)
    assert (attempts, single) == (1, [])


async def test_the_wait_doubles_its_ceiling_between_attempts() -> None:
    """Full jitter, so one shared outage does not resynchronise every client onto one instant."""
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"code": "model_unavailable", "message": "x", "trace_id": "t1"}
        )

    async with _retrying_client(handler, delays=delays, retry_attempts=4) as memory:
        with pytest.raises(MindBridgeError):
            await memory.get_memory("tenant_01", "memory_01")

    assert len(delays) == 3, "the retry budget was not spent, so the ceilings below prove nothing"
    assert all(delay >= 0.0 for delay in delays)
    assert all(delay <= ceiling for delay, ceiling in zip(delays, (0.5, 1.0, 2.0), strict=True))


async def test_a_client_cannot_be_built_with_an_unusable_retry_budget() -> None:
    with pytest.raises(ValueError, match="retry_attempts"):
        MindBridge(httpx.AsyncClient(base_url="https://x.test/"), retry_attempts=0)
    with pytest.raises(ValueError, match="retry_backoff_seconds"):
        MindBridge(httpx.AsyncClient(base_url="https://x.test/"), retry_backoff_seconds=-1.0)
