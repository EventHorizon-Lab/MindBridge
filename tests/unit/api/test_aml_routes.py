"""AML route contract tests."""

import asyncio
import json
from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

from mindbridge.api.aml import _MAX_CONCURRENT_REMEMBERS, AmlSettings
from mindbridge.api.aml_contracts import derive_tenant_id
from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    MemoryResult,
    MemoryView,
    RecallRequest,
    RecallResult,
    RememberRequest,
)
from mindbridge.core import MemoryState, MemoryType, ModelReference, VerificationStatus
from mindbridge.models import GenerateRequest, GenerateResult

_KEY = "aml_test_key_that_is_long_enough_0123456789"
_AUTHENTICATOR_KEY = "throwaway_tenant_api_key_00000000000000000"
_NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


class _StubGenerator:
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            text='{"memories": [{"summary": "Rob moved to Sweden.", "type": "episodic"}]}',
            model_reference=ModelReference(model_id="qwen3.8-max", revision="test"),
        )


class _StubKernel:
    def __init__(self) -> None:
        self.written: list[tuple[str, str]] = []
        self.recalled: list[tuple[str, str, int]] = []

    async def remember(self, request: RememberRequest) -> MemoryResult:
        self.written.append((request.tenant_id, request.summary))
        return MemoryResult(
            memory_id="mem_1",
            memory_type=MemoryType.EPISODIC,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=_NOW,
            ended_at=_NOW,
            created_at=_NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
            trace_id="trace",
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        query_text = request.query.text
        assert query_text is not None
        self.recalled.append((request.tenant_id, query_text, request.limit))
        return RecallResult(
            answer=None,
            confidence=0.0,
            memories=(
                self._memory("mem_1", "Rob moved to Sweden."),
                self._memory("mem_2", "Rob prefers tea."),
            ),
            evidence=(),
            trace_id="trace",
        )

    @staticmethod
    def _memory(memory_id: str, summary: str) -> MemoryView:
        return MemoryView(
            memory_id=memory_id,
            memory_type=MemoryType.SEMANTIC,
            summary=summary,
            evidence_ids=(),
            occurred_at=_NOW,
            ended_at=_NOW,
            created_at=_NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
        )


@pytest.fixture()
def client() -> tuple[TestClient, _StubKernel]:
    kernel = _StubKernel()
    authenticator = TenantApiKeyAuthenticator({"tenant_01": (_AUTHENTICATOR_KEY,)})
    app = build_app(
        cast(MemoryKernel, kernel),
        authenticator=authenticator,
        aml=(AmlSettings(api_key=_KEY, tenant_prefix="bench_aml"), _StubGenerator()),
    )
    return TestClient(app), kernel


def test_add_echoes_all_three_identifiers_byte_for_byte(
    client: tuple[TestClient, _StubKernel],
) -> None:
    http, kernel = client
    request_id = "eval:run-1:locomo_refined:conv-0:chunk-0"
    user_id = "eval:run-1:locomo:conv-0"
    session_id = "eval:run-1:sample:0"
    payload = {
        "request_id": request_id,
        "messages": [{"role": "user", "content": "Rob moved to Sweden.", "timestamp": 1}],
        "user_id": user_id,
        "session_id": session_id,
    }

    response = http.post("/aml/add", json=payload, headers={"Authorization": f"Bearer {_KEY}"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["request_id"] == request_id
    assert body["user_id"] == user_id
    assert body["session_id"] == session_id
    assert kernel.written == [(derive_tenant_id("bench_aml", user_id), "Rob moved to Sweden.")]


def test_search_returns_ranked_items_scoped_to_the_user(
    client: tuple[TestClient, _StubKernel],
) -> None:
    http, kernel = client

    response = http.post(
        "/aml/search",
        json={"query": "Where did Rob move?", "user_id": "eval:run-1:locomo:conv-0", "top_k": 100},
        headers={"Authorization": f"Bearer {_KEY}"},
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == ["mem_1", "mem_2"]
    assert kernel.recalled == [
        (
            derive_tenant_id("bench_aml", "eval:run-1:locomo:conv-0"),
            "Where did Rob move?",
            100,
        )
    ]


def test_different_users_never_share_a_tenant(
    client: tuple[TestClient, _StubKernel],
) -> None:
    http, kernel = client
    for user_id in ("eval:run-1:locomo:conv-0", "eval:run-1:locomo:conv-1"):
        http.post(
            "/aml/search",
            json={"query": "q", "user_id": user_id, "top_k": 10},
            headers={"Authorization": f"Bearer {_KEY}"},
        )

    assert kernel.recalled[0][0] != kernel.recalled[1][0]


def test_missing_credentials_are_rejected(
    client: tuple[TestClient, _StubKernel],
) -> None:
    http, _ = client
    assert (
        http.post("/aml/search", json={"query": "q", "user_id": "u", "top_k": 1}).status_code == 401
    )


def test_wrong_bearer_token_is_rejected(
    client: tuple[TestClient, _StubKernel],
) -> None:
    http, _ = client
    wrong_token = "wrong_token_that_is_long_enough_0123456789"
    response = http.post(
        "/aml/search",
        json={"query": "q", "user_id": "u", "top_k": 1},
        headers={"Authorization": f"Bearer {wrong_token}"},
    )
    assert response.status_code == 401


class _ManyMemoriesGenerator:
    """Extracts `count` memories from one chunk, regardless of its content."""

    def __init__(self, count: int) -> None:
        self._count = count

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        memories = [
            {"summary": f"fact {index}", "type": "semantic"} for index in range(self._count)
        ]
        return GenerateResult(
            text=json.dumps({"memories": memories}),
            model_reference=ModelReference(model_id="qwen3.8-max", revision="test"),
        )


class _ConcurrencyTrackingKernel:
    """Records the highest number of `remember()` calls in flight at once."""

    def __init__(self) -> None:
        self._current = 0
        self.max_seen = 0
        self._lock = asyncio.Lock()

    async def remember(self, request: RememberRequest) -> MemoryResult:
        async with self._lock:
            self._current += 1
            self.max_seen = max(self.max_seen, self._current)
        await asyncio.sleep(0.01)  # hold the slot open so overlapping calls can pile up
        async with self._lock:
            self._current -= 1
        return MemoryResult(
            memory_id="mem",
            memory_type=MemoryType.SEMANTIC,
            summary=request.summary,
            evidence_ids=(),
            occurred_at=_NOW,
            ended_at=_NOW,
            created_at=_NOW,
            verification_status=VerificationStatus.ATTESTED,
            state=MemoryState.ACTIVE,
            trace_id="trace",
        )

    async def recall(self, request: RecallRequest) -> RecallResult:
        raise NotImplementedError


def test_add_bounds_concurrent_remembers() -> None:
    """Cheap 10 (final review, 2026-08-17): a chunk yielding many memories
    used to fan out one concurrent `remember()` per memory with no cap. A
    generator that extracts more memories than `_MAX_CONCURRENT_REMEMBERS`
    must never push more than that many `remember()` calls in flight at once.
    """
    memory_count = _MAX_CONCURRENT_REMEMBERS * 4
    kernel = _ConcurrencyTrackingKernel()
    authenticator = TenantApiKeyAuthenticator({"tenant_01": (_AUTHENTICATOR_KEY,)})
    app = build_app(
        cast(MemoryKernel, kernel),
        authenticator=authenticator,
        aml=(
            AmlSettings(api_key=_KEY, tenant_prefix="bench_aml"),
            _ManyMemoriesGenerator(memory_count),
        ),
    )
    http = TestClient(app)

    response = http.post(
        "/aml/add",
        json={
            "request_id": "r1",
            "messages": [{"role": "user", "content": "hi"}],
            "user_id": "u1",
            "session_id": "s1",
        },
        headers={"Authorization": f"Bearer {_KEY}"},
    )

    assert response.status_code == 200
    assert 1 < kernel.max_seen <= _MAX_CONCURRENT_REMEMBERS
