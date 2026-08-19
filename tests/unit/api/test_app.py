"""Tests for the thin FastAPI protocol adapter."""

import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    DeletionListRequest,
    DeletionPage,
    DeletionTombstoneView,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    MemoryResult,
    MemoryWriteStatus,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
)
from mindbridge.core import (
    DatabaseUnavailableError,
    DeletionPropagationState,
    DomainInvariantError,
    EnumerationLimitExceededError,
    FeedbackType,
    ForgetTargetNotFoundError,
    ForgetTargetType,
    JobNotFoundError,
    JobState,
    MemoryIntegrityError,
    MemoryNotFoundError,
    MemoryState,
    MemoryType,
    ModelOutputError,
    ModelRequestError,
    ModelUnavailableError,
    ObjectStorageError,
    TaskBrokerError,
    VerificationStatus,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
TENANT_API_KEY = "tenant-01-test-key-00000000000000"
OTHER_TENANT_API_KEY = "other-tenant-test-key-00000000000"


class StubKernel:
    """Protocol stub keeping adapter tests independent from persistence."""

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        return ObservationReceipt(
            observation_id="observation_01",
            processing_job_id="job_process_observation_01",
            idempotency_key=request.idempotency_key or "derived-key",
            status=ObservationStatus.ACCEPTED,
            trace_id="trace_observe",
        )

    async def remember(self, request: RememberRequest) -> RememberResult:
        if request.summary == "invalid":
            raise DomainInvariantError("invalid memory")
        return RememberResult(
            status=MemoryWriteStatus.CREATED,
            memory_id="memory_01",
            memory_type=request.memory_type,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=NOW,
            verification_status=VerificationStatus.UNVERIFIED,
            state=MemoryState.ACTIVE,
            trace_id="trace_remember",
        )

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        return FeedbackReceipt(
            feedback_id="feedback_01",
            feedback_type=request.feedback_type,
            memory_id=request.memory_id,
            corrected_memory_id=None,
            resulting_state=(MemoryState.STRENGTHENED if request.memory_id is not None else None),
            resulting_strength=1.5 if request.memory_id is not None else None,
            created_at=NOW,
            trace_id="trace_feedback",
        )

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        return ForgetReceipt(
            tombstone_id="tombstone_01",
            target_type=request.target_type,
            target_id=request.target_id,
            propagation_state=DeletionPropagationState.COMPLETE,
            requested_at=NOW,
            completed_at=NOW,
            error_code=None,
            trace_id="trace_forget",
        )

    async def get_forget_status(self, tenant_id: str, tombstone_id: str) -> ForgetReceipt:
        if tenant_id != "tenant_01":
            raise ForgetTargetNotFoundError("missing")
        return ForgetReceipt(
            tombstone_id=tombstone_id,
            target_type=ForgetTargetType.MEMORY_RECORD,
            target_id="memory_01",
            propagation_state=DeletionPropagationState.COMPLETE,
            requested_at=NOW,
            completed_at=NOW,
            error_code=None,
            trace_id="trace_forget_status",
        )

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        return DeletionPage(
            items=(
                DeletionTombstoneView(
                    tombstone_id="tombstone_01",
                    target_type=ForgetTargetType.MEMORY_RECORD,
                    target_id="memory_01",
                    propagation_state=DeletionPropagationState.COMPLETE,
                    requested_at=NOW,
                    completed_at=NOW,
                    error_code=None,
                ),
            ),
            next_cursor=None,
            trace_id="trace_deletion_page",
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        return RecallResult(
            answer=None,
            confidence=0.0,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
        )

    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        if job_id != "job_01":
            raise JobNotFoundError("missing")
        return ObservationProcessingJobView(
            job_id=job_id,
            observation_id="observation_01",
            state=JobState.SUCCEEDED,
            attempt=1,
            error_code=None,
            memory_ids=("memory_01",),
            created_at=NOW,
            updated_at=NOW,
            trace_id="trace_job",
        )

    async def watch_observation_job(
        self,
        tenant_id: str,
        job_id: str,
        *,
        after_updated_at: datetime | None = None,
    ) -> AsyncIterator[ObservationProcessingJobView]:
        for state, moment in (
            (JobState.RUNNING, NOW),
            (JobState.SUCCEEDED, NOW + timedelta(seconds=1)),
        ):
            if after_updated_at is not None and moment <= after_updated_at:
                continue
            yield ObservationProcessingJobView(
                job_id=job_id,
                observation_id="observation_01",
                state=state,
                attempt=1,
                error_code=None,
                memory_ids=("memory_01",) if state is JobState.SUCCEEDED else (),
                created_at=NOW,
                updated_at=moment,
                trace_id="trace_job_stream",
            )

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
        if memory_id != "memory_01":
            raise MemoryNotFoundError("missing")
        return MemoryResult(
            memory_id=memory_id,
            memory_type=MemoryType.EPISODIC,
            summary="remembered event",
            evidence_ids=(),
            occurred_at=NOW,
            ended_at=NOW,
            created_at=NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
            trace_id="trace_get_memory",
        )


def test_recall_route_uses_shared_contract() -> None:
    """REST request and response shapes are the same Pydantic contracts."""
    client = _client()

    response = client.post(
        "/v1/recall",
        json={"tenant_id": "tenant_01", "query": {"media_object_ids": ["media_01"]}},
    )

    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace_recall"


def test_feedback_route_uses_shared_contract() -> None:
    response = _client().post(
        "/v1/feedback",
        json={
            "tenant_id": "tenant_01",
            "feedback_type": FeedbackType.USEFUL,
            "memory_id": "memory_01",
        },
    )

    assert response.status_code == 201
    assert response.json()["resulting_state"] == "strengthened"
    assert response.json()["trace_id"] == "trace_feedback"


def test_forget_routes_share_typed_progress_contract() -> None:
    client = _client()
    forgotten = client.post(
        "/v1/forget",
        json={
            "tenant_id": "tenant_01",
            "target_type": "memory_record",
            "target_id": "memory_01",
        },
    )
    status_response = client.get(
        "/v1/deletions/tombstone_01",
        params={"tenant_id": "tenant_01"},
    )
    page = client.get("/v1/deletions", params={"tenant_id": "tenant_01"})

    assert forgotten.status_code == 200
    assert forgotten.json()["propagation_state"] == "complete"
    assert status_response.json()["trace_id"] == "trace_forget_status"
    assert page.json()["items"][0]["tombstone_id"] == "tombstone_01"


def test_job_route_is_tenant_scoped_and_returns_not_found() -> None:
    client = _client()

    found = client.get("/v1/jobs/job_01", params={"tenant_id": "tenant_01"})
    missing = client.get("/v1/jobs/missing", params={"tenant_id": "tenant_01"})

    assert found.status_code == 200
    assert found.json()["state"] == "succeeded"
    assert missing.status_code == 404
    assert missing.json()["code"] == "job_not_found"


def test_job_event_stream_sends_one_complete_view_per_change() -> None:
    client = _client()

    response = client.get("/v1/jobs/job_01/events", params={"tenant_id": "tenant_01"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    events = _sse_events(response.text)
    assert [event["event"] for event in events] == ["job", "job"]
    assert [event["id"] for event in events] == [
        str(_expected_event_id(NOW)),
        str(_expected_event_id(NOW + timedelta(seconds=1))),
    ]
    assert [json.loads(event["data"])["state"] for event in events] == ["running", "succeeded"]
    # Each event carries the whole view, which is what makes resuming from one ID correct.
    assert json.loads(events[-1]["data"])["memory_ids"] == ["memory_01"]


def test_job_event_stream_reports_a_failure_the_contract_has_no_word_for() -> None:
    """A stream that just stops is byte-identical to one that finished.

    `translate_transient_database_errors` re-raises a psycopg error whose sqlstate it does not
    recognise, and only two exception types were caught, so anything else ended the response
    body after the 200 -- and `aiter_lines` completing normally is exactly what a settled job
    looks like. The caller was told the opposite of what happened. `internal_error` is the
    honest answer: it still names the failure and still carries a `trace_id`.
    """

    class BrokenStreamKernel(StubKernel):
        async def watch_observation_job(
            self,
            tenant_id: str,
            job_id: str,
            *,
            after_updated_at: datetime | None = None,
        ) -> AsyncIterator[ObservationProcessingJobView]:
            raise RuntimeError("connection died mid-stream")
            yield  # pragma: no cover - unreachable, present to keep this a generator

    response = _client(BrokenStreamKernel()).get(
        "/v1/jobs/job_01/events", params={"tenant_id": "tenant_01"}
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [event["event"] for event in events] == ["error"]
    assert json.loads(events[0]["data"])["code"] == "internal_error"
    assert json.loads(events[0]["data"])["trace_id"]


def test_job_event_stream_names_a_mapped_failure_as_the_routes_do() -> None:
    """A code the contract does have arrives as that code, not as a generic one."""

    class UnavailableStreamKernel(StubKernel):
        async def watch_observation_job(
            self,
            tenant_id: str,
            job_id: str,
            *,
            after_updated_at: datetime | None = None,
        ) -> AsyncIterator[ObservationProcessingJobView]:
            raise DatabaseUnavailableError("storage went away")
            yield  # pragma: no cover - unreachable, present to keep this a generator

    response = _client(UnavailableStreamKernel()).get(
        "/v1/jobs/job_01/events", params={"tenant_id": "tenant_01"}
    )

    events = _sse_events(response.text)
    assert [event["event"] for event in events] == ["error"]
    assert json.loads(events[0]["data"])["code"] == "database_unavailable"


def test_job_event_stream_resumes_after_the_last_received_event() -> None:
    client = _client()

    response = client.get(
        "/v1/jobs/job_01/events",
        params={"tenant_id": "tenant_01"},
        headers={"Last-Event-ID": str(_expected_event_id(NOW))},
    )

    events = _sse_events(response.text)
    assert [json.loads(event["data"])["state"] for event in events] == ["succeeded"]


def test_job_event_stream_ignores_an_unusable_last_event_id() -> None:
    client = _client()

    response = client.get(
        "/v1/jobs/job_01/events",
        params={"tenant_id": "tenant_01"},
        headers={"Last-Event-ID": "not-a-number"},
    )

    assert response.status_code == 200
    assert len(_sse_events(response.text)) == 2


def test_job_event_stream_rejects_a_missing_or_foreign_job_before_streaming() -> None:
    client = _client()

    missing = client.get("/v1/jobs/missing/events", params={"tenant_id": "tenant_01"})
    foreign = client.get("/v1/jobs/job_01/events", params={"tenant_id": "tenant_02"})

    assert missing.status_code == 404
    assert missing.json()["code"] == "job_not_found"
    assert foreign.status_code == 403


def _sse_events(body: str) -> list[dict[str, str]]:
    """Parse framed events, dropping keepalive comments."""
    events: list[dict[str, str]] = []
    for block in body.split("\n\n"):
        fields = {
            name: value.strip()
            for name, _, value in (line.partition(":") for line in block.splitlines())
            if name
        }
        if "event" in fields:
            events.append(fields)
    return events


def _expected_event_id(moment: datetime) -> int:
    """Pin the wire format independently of the implementation."""
    return (moment - datetime(1970, 1, 1, tzinfo=timezone.utc)) // timedelta(microseconds=1)


def test_memory_routes_are_tenant_scoped_and_traced() -> None:
    client = _client()

    created = client.post(
        "/v1/memories",
        json={
            "tenant_id": "tenant_01",
            "summary": "remembered event",
            "memory_type": "episodic",
            "occurred_at": NOW.isoformat(),
        },
    )
    found = client.get("/v1/memories/memory_01", params={"tenant_id": "tenant_01"})
    missing = client.get("/v1/memories/missing", params={"tenant_id": "tenant_01"})

    assert created.status_code == 201
    assert created.json()["trace_id"] == "trace_remember"
    assert found.status_code == 200
    assert found.json()["memory_id"] == "memory_01"
    assert found.json()["evidence"] == []
    assert found.json()["trace_id"] == "trace_get_memory"
    assert missing.status_code == 404
    assert missing.json()["code"] == "memory_not_found"


def test_validation_errors_have_trace_and_field_location() -> None:
    """Malformed requests fail predictably without entering the kernel."""
    response = _client().post(
        "/v1/recall",
        json={"tenant_id": "tenant_01", "query": {}, "unknown": True},
    )

    body = response.json()
    assert response.status_code == 422
    assert body["trace_id"].startswith("trace_")
    assert {issue["location"][-1] for issue in body["issues"]} == {"query", "unknown"}


def test_api_requires_a_valid_bearer_key() -> None:
    app = build_app(cast(MemoryKernel, StubKernel()), authenticator=_authenticator())
    client = TestClient(app)

    missing = client.post(
        "/v1/recall",
        json={"tenant_id": "tenant_01", "query": {"text": "question"}},
    )
    invalid = client.post(
        "/v1/recall",
        headers={"Authorization": "Bearer invalid"},
        json={"tenant_id": "tenant_01", "query": {"text": "question"}},
    )

    assert missing.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert missing.json()["code"] == "authentication_required"
    assert invalid.status_code == 401
    assert invalid.json()["code"] == "authentication_failed"
    assert "Bearer invalid" not in invalid.text


def test_api_rejects_cross_tenant_requests() -> None:
    client = _client()
    body_response = client.post(
        "/v1/recall",
        json={"tenant_id": "other_tenant", "query": {"text": "question"}},
    )
    query_response = client.get(
        "/v1/memories/memory_01",
        params={"tenant_id": "other_tenant"},
    )

    assert body_response.status_code == 403
    assert query_response.status_code == 403
    assert body_response.json()["code"] == "tenant_access_denied"


def test_domain_errors_use_stable_envelope() -> None:
    """Domain failures remain transport-neutral and machine-readable."""
    response = _client().post(
        "/v1/memories",
        json={
            "tenant_id": "tenant_01",
            "summary": "invalid",
            "memory_type": MemoryType.EPISODIC,
            "occurred_at": NOW.isoformat(),
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "domain_invariant_failed"


def test_enumeration_limit_has_an_actionable_stable_error_code() -> None:
    response = _client(
        FailingKernel(
            EnumerationLimitExceededError(
                "exact enumeration exceeds 1000 candidates; narrow the recall filters"
            )
        )
    ).post(
        "/v1/recall",
        json={"tenant_id": "tenant_01", "query": {"text": "count everything"}},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "enumeration_limit_exceeded"


def test_openapi_exposes_stable_operation_ids() -> None:
    """Agent tooling can derive deterministic operations from OpenAPI."""
    paths = _client().get("/openapi.json").json()["paths"]

    assert paths["/v1/observations"]["post"]["operationId"] == "observe"
    assert paths["/v1/memories"]["post"]["operationId"] == "remember"
    assert paths["/v1/memories/{memory_id}"]["get"]["operationId"] == "getMemory"
    assert paths["/v1/feedback"]["post"]["operationId"] == "recordFeedback"
    assert paths["/v1/forget"]["post"]["operationId"] == "forget"
    assert paths["/v1/deletions/{tombstone_id}"]["get"]["operationId"] == "getForgetStatus"
    assert paths["/v1/deletions"]["get"]["operationId"] == "listDeletions"
    assert paths["/v1/recall"]["post"]["operationId"] == "recall"
    assert paths["/v1/jobs/{job_id}"]["get"]["operationId"] == "getObservationJob"
    assert paths["/v1/jobs/{job_id}/events"]["get"]["operationId"] == "streamObservationJob"
    assert (
        "text/event-stream"
        in paths["/v1/jobs/{job_id}/events"]["get"]["responses"]["200"]["content"]
    )


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (DatabaseUnavailableError("database detail"), 503, "database_unavailable"),
        (ModelUnavailableError("provider detail"), 503, "model_unavailable"),
        (ModelRequestError("provider detail"), 502, "model_request_failed"),
        (ModelOutputError("raw output"), 502, "model_output_invalid"),
        (ObjectStorageError("bucket detail"), 503, "object_storage_unavailable"),
        (TaskBrokerError("redis detail"), 503, "task_broker_unavailable"),
        (MemoryIntegrityError("row detail"), 500, "memory_integrity_failed"),
    ],
)
def test_runtime_errors_use_sanitized_stable_envelopes(
    error: RuntimeError,
    expected_status: int,
    expected_code: str,
) -> None:
    """Dependency details cannot leak through the public Agent contract."""
    response = _client(FailingKernel(error)).post(
        "/v1/recall",
        json={"tenant_id": "tenant_01", "query": {"text": "question"}},
    )

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert str(error) not in response.text


class FailingKernel(StubKernel):
    """Raises one sanitized runtime category from the shared recall route."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def recall(self, request: RecallRequest) -> RecallResult:
        raise self._error


def _client(stub: StubKernel | None = None) -> TestClient:
    kernel = cast(MemoryKernel, stub or StubKernel())
    return TestClient(
        build_app(kernel, authenticator=_authenticator()),
        headers={"Authorization": f"Bearer {TENANT_API_KEY}"},
    )


def _authenticator() -> TenantApiKeyAuthenticator:
    return TenantApiKeyAuthenticator(
        {
            "tenant_01": (TENANT_API_KEY,),
            "other_tenant": (OTHER_TENANT_API_KEY,),
        }
    )
