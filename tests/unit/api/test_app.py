"""Tests for the thin FastAPI protocol adapter."""

from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

from mindbridge.api import create_app
from mindbridge.application import MemoryKernel
from mindbridge.contracts import (
    MemoryView,
    ObservationReceipt,
    ObservationStatus,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import (
    DomainInvariantError,
    MemoryIntegrityError,
    MemoryState,
    MemoryType,
    ModelOutputError,
    ModelUnavailableError,
    ObjectStorageError,
    VerificationStatus,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


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

    async def remember(self, request: RememberRequest) -> MemoryView:
        if request.summary == "invalid":
            raise DomainInvariantError("invalid memory")
        return MemoryView(
            memory_id="memory_01",
            memory_type=request.memory_type,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=request.occurred_at,
            ended_at=request.ended_at or request.occurred_at,
            created_at=NOW,
            verification_status=VerificationStatus.UNVERIFIED,
            state=MemoryState.ACTIVE,
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        return RecallResult(
            answer=None,
            confidence=0.0,
            memories=(),
            evidence=(),
            trace_id="trace_recall",
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


def test_openapi_exposes_stable_operation_ids() -> None:
    """Agent tooling can derive deterministic operations from OpenAPI."""
    paths = _client().get("/openapi.json").json()["paths"]

    assert paths["/v1/observations"]["post"]["operationId"] == "observe"
    assert paths["/v1/memories"]["post"]["operationId"] == "remember"
    assert paths["/v1/recall"]["post"]["operationId"] == "recall"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (ModelUnavailableError("provider detail"), 503, "model_unavailable"),
        (ModelOutputError("raw output"), 502, "model_output_invalid"),
        (ObjectStorageError("bucket detail"), 503, "object_storage_unavailable"),
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

    def __init__(self, error: RuntimeError) -> None:
        self._error = error

    async def recall(self, request: RecallRequest) -> RecallResult:
        raise self._error


def _client(stub: StubKernel | None = None) -> TestClient:
    kernel = cast(MemoryKernel, stub or StubKernel())
    return TestClient(create_app(kernel))
