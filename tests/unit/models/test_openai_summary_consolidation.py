"""Contract tests for evidence-first hierarchical Memory consolidation."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application import ResolvedEvidence, SummaryCandidate, SummaryScope
from mindbridge.core import (
    DomainInvariantError,
    EntityId,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelOutputError,
    ModelReference,
    ObservationId,
    TenantId,
    VerificationStatus,
)
from mindbridge.models import OpenAIOmniSummaryConsolidator, normalize_openai_base_url

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


async def test_summary_consolidator_inspects_native_evidence_and_preserves_revision() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        assert {part["type"] for part in content} >= {"video_url"}
        context = cast(str, content[0]["text"])
        assert "memory_02" in context and "attested" in context
        return _streaming_response(
            {
                "summaries": [
                    {
                        "source_memory_ids": ["memory_01", "memory_02"],
                        "scope": "session",
                        "summary": "A repair was observed and a preference was reported.",
                        "salience": 0.86,
                    }
                ]
            },
            fingerprint="summary-serving-revision-01",
        )

    consolidator = _consolidator(respond)
    candidates, evidence = _candidates()
    try:
        result = await consolidator.propose_summaries(candidates, evidence)
    finally:
        await consolidator.close()

    assert result.summaries[0].source_memory_ids == (
        MemoryId("memory_01"),
        MemoryId("memory_02"),
    )
    assert result.summaries[0].scope is SummaryScope.SESSION
    assert result.model_reference.revision == "summary-serving-revision-01"
    assert result.prompt_version == "consolidate_summaries_v1"


async def test_summary_consolidator_rejects_unknown_memory_and_missing_evidence() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "summaries": [
                    {
                        "source_memory_ids": ["memory_01", "memory_unknown"],
                        "scope": "topic",
                        "summary": "Unsupported",
                        "salience": 0.5,
                    }
                ]
            }
        )

    consolidator = _consolidator(respond)
    candidates, evidence = _candidates()
    try:
        with pytest.raises(ModelOutputError, match="unknown Memory"):
            await consolidator.propose_summaries(candidates, evidence)
        with pytest.raises(DomainInvariantError, match="each exact"):
            await consolidator.propose_summaries(candidates, ())
    finally:
        await consolidator.close()


def _consolidator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIOmniSummaryConsolidator:
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_openai_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAIOmniSummaryConsolidator(client, model_revision="deployment-revision")


def _candidates() -> tuple[tuple[SummaryCandidate, ...], tuple[ResolvedEvidence, ...]]:
    candidates = (
        SummaryCandidate(
            memory=MemoryRecord(
                memory_id=MemoryId("memory_01"),
                tenant_id=TenantId("tenant_01"),
                memory_type=MemoryType.EPISODIC,
                summary="A person repairs a tool.",
                evidence_ids=(EvidenceId("evidence_01"),),
                occurred_at=NOW,
                ended_at=NOW + timedelta(seconds=4),
                created_at=NOW,
                verification_status=VerificationStatus.VERIFIED,
                model_reference=ModelReference(model_id="omni", revision="perception-revision"),
            ),
            entity_ids=(EntityId("tool"),),
        ),
        SummaryCandidate(
            memory=MemoryRecord(
                memory_id=MemoryId("memory_02"),
                tenant_id=TenantId("tenant_01"),
                memory_type=MemoryType.SEMANTIC,
                summary="The user reports preferring the red tool.",
                evidence_ids=(),
                occurred_at=NOW + timedelta(minutes=1),
                ended_at=NOW + timedelta(minutes=1),
                created_at=NOW,
                verification_status=VerificationStatus.ATTESTED,
            ),
            entity_ids=(EntityId("tool"),),
        ),
    )
    return candidates, (_evidence(),)


def _evidence() -> ResolvedEvidence:
    media_id = MediaObjectId("media_01")
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId("evidence_01"),
            tenant_id=TenantId("tenant_01"),
            observation_id=ObservationId("observation_01"),
            media_object_id=media_id,
            start_ms=0,
            end_ms=4_000,
            created_at=NOW,
        ),
        media_object=MediaObject(
            media_object_id=media_id,
            tenant_id=TenantId("tenant_01"),
            kind=MediaKind.VIDEO,
            uri="s3://memory/tenants/tenant_01/clip.mp4",
            sha256="1" * 64,
            size_bytes=100,
            created_at=NOW,
            duration_ms=4_000,
        ),
        media_url="https://objects.example.test/clip.mp4",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _streaming_response(payload: object, *, fingerprint: str | None = None) -> httpx.Response:
    event = {
        "id": "completion_summary_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"content": json.dumps(payload)},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
    )
