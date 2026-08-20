"""Contract tests for the evidence-first Claim pipeline."""

import json
from collections.abc import Callable, Coroutine, Iterator
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.claim_consolidation import ClaimCandidate
from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines import ClaimPipeline
from mindbridge.core import (
    Claim,
    ClaimId,
    ClaimType,
    EntityId,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    ModelReference,
    ObservationId,
    RelationType,
    TenantId,
    VerificationStatus,
)
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


async def test_claim_consolidator_inspects_native_evidence_and_preserves_model_identity() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        system_prompt = cast(str, messages[0]["content"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        assert "same proposition" in system_prompt
        assert "compatible, complementary" in system_prompt
        assert {part["type"] for part in content} >= {"video_url", "input_audio"}
        assert "Propose supported claim" in cast(str, content[-1]["text"])
        assert "claim_04" in cast(str, content[0]["text"])
        return _streaming_response(
            {
                "semantic_claims": [
                    {
                        "source_claim_ids": ["claim_01", "claim_02"],
                        "statement": "The red tool is beside the blue toolbox.",
                        "confidence": 0.94,
                    }
                ],
                "relationships": [
                    {
                        "source_claim_id": "claim_04",
                        "relation_type": "contradicts",
                        "target_claim_id": "claim_03",
                    }
                ],
            },
            fingerprint="claim-serving-01",
        )

    consolidator = _consolidator(respond)
    candidates, evidence = _candidates()
    try:
        result = await consolidator.propose_claims(candidates, evidence)
    finally:
        await consolidator.close()

    assert result.semantic_claims[0].source_claim_ids == (
        ClaimId("claim_01"),
        ClaimId("claim_02"),
    )
    assert result.relationships[0].relation_type is RelationType.CONTRADICTS
    assert result.model_reference.model_id == "qwen3.8-max"
    assert result.prompt_version == "consolidate_claims_v2"


async def test_claim_consolidator_rejects_unknown_and_reversed_relationships() -> None:
    responses: Iterator[object] = iter(
        (
            {
                "semantic_claims": [
                    {
                        "source_claim_ids": ["claim_01", "claim_unknown"],
                        "statement": "Unsupported",
                        "confidence": 0.5,
                    }
                ],
                "relationships": [],
            },
            {
                "semantic_claims": [
                    {
                        "source_claim_ids": ["claim_01", "claim_unknown"],
                        "statement": "Unsupported",
                        "confidence": 0.5,
                    }
                ],
                "relationships": [],
            },
            {
                "semantic_claims": [],
                "relationships": [
                    {
                        "source_claim_id": "claim_01",
                        "relation_type": "supersedes",
                        "target_claim_id": "claim_04",
                    }
                ],
            },
            {
                "semantic_claims": [],
                "relationships": [
                    {
                        "source_claim_id": "claim_01",
                        "relation_type": "supersedes",
                        "target_claim_id": "claim_04",
                    }
                ],
            },
        )
    )

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(next(responses))

    consolidator = _consolidator(respond)
    try:
        with pytest.raises(ModelOutputError, match="unknown claim"):
            await consolidator.propose_claims(*_candidates())
        with pytest.raises(ModelOutputError, match="later claim"):
            await consolidator.propose_claims(*_candidates())
    finally:
        await consolidator.close()


async def test_claim_consolidator_retries_invalid_structure_in_json_mode() -> None:
    responses: Iterator[object] = iter(
        (
            {
                "semantic_claims": [
                    {
                        "source_claim_ids": ["claim_01"],
                        "statement": "Singletons are not consolidation.",
                        "confidence": 0.5,
                    }
                ],
                "relationships": [],
            },
            {"semantic_claims": [], "relationships": []},
        )
    )
    response_formats: list[object] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        response_formats.append(payload.get("response_format"))
        return _streaming_response(next(responses))

    consolidator = _consolidator(respond)
    try:
        result = await consolidator.propose_claims(*_candidates())
    finally:
        await consolidator.close()

    assert result.semantic_claims == ()
    assert response_formats == [None, {"type": "json_object"}]


def _consolidator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> "_ClaimHarness":
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return _ClaimHarness(
        OpenAIGenerator(
            client,
            ModelReference(model_id="qwen3.8-max"),
        )
    )


class _ClaimHarness(ClaimPipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(generator)
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


def _candidates() -> tuple[tuple[ClaimCandidate, ...], tuple[ResolvedEvidence, ...]]:
    candidates = tuple(
        ClaimCandidate(
            claim=Claim(
                claim_id=ClaimId(f"claim_{index:02d}"),
                tenant_id=TenantId("tenant_01"),
                claim_type=ClaimType.STATE,
                statement=statement,
                evidence_ids=(EvidenceId(f"evidence_{index:02d}"),),
                confidence=0.8,
                verification_status=VerificationStatus.VERIFIED,
                valid_from=NOW + timedelta(minutes=index),
                valid_to=None,
                created_at=NOW,
                model_reference=ModelReference(
                    model_id="qwen3.8-max",
                ),
                prompt_version="perceive_events_v3",
            ),
            entity_ids=(EntityId("person_robot_01"), EntityId("red_tool")),
        )
        for index, statement in enumerate(
            (
                "The red tool is beside the blue toolbox.",
                "The red tool remains beside the blue toolbox.",
                "The red tool is no longer beside the blue toolbox.",
                "The red tool remains away from the blue toolbox.",
            ),
            start=1,
        )
    )
    evidence = tuple(
        _evidence(index, kind)
        for index, kind in enumerate(
            (MediaKind.VIDEO, MediaKind.AUDIO, MediaKind.IMAGE, MediaKind.VIDEO),
            start=1,
        )
    )
    return candidates, evidence


def _evidence(index: int, kind: MediaKind) -> ResolvedEvidence:
    suffix = f"{index:02d}"
    media_id = MediaObjectId(f"media_{suffix}")
    extension = {MediaKind.VIDEO: "mp4", MediaKind.AUDIO: "wav", MediaKind.IMAGE: "jpg"}[kind]
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(f"evidence_{suffix}"),
            tenant_id=TenantId("tenant_01"),
            observation_id=ObservationId(f"observation_{suffix}"),
            media_object_id=media_id,
            start_ms=0,
            end_ms=4_000,
            created_at=NOW,
        ),
        media_object=MediaObject(
            media_object_id=media_id,
            tenant_id=TenantId("tenant_01"),
            kind=kind,
            uri=f"s3://memory/tenants/tenant_01/clip_{suffix}.{extension}",
            sha256=f"{index:064x}",
            size_bytes=100,
            created_at=NOW,
            duration_ms=4_000 if kind is not MediaKind.IMAGE else None,
        ),
        media_url=f"https://objects.example.test/clip_{suffix}.{extension}",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _streaming_response(payload: object, *, fingerprint: str | None = None) -> httpx.Response:
    event = {
        "id": "completion_claim_01",
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
