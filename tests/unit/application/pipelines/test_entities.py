"""Contract tests for the pairwise entity adjudication pipeline."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.entity_resolution import EntityCandidate, EntityPair
from mindbridge.application.pipelines.entities import EntityResolutionPipeline
from mindbridge.core import (
    Entity,
    EntityId,
    EntityType,
    EvidenceId,
    ModelOutputError,
    ModelReference,
    TenantId,
)
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)


async def test_adjudication_parses_a_verdict_and_sends_both_records() -> None:
    """Both names reach the judge, and the released prompt's refusal rule travels with them."""
    seen: dict[str, str] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        messages = cast(list[dict[str, object]], payload["messages"])
        seen["system"] = cast(str, messages[0]["content"])
        seen["user"] = json.dumps(messages[1]["content"])
        return _completion(
            '{"same_entity":true,"confidence":0.86,'
            '"discriminating_cue":"same scar above the left eyebrow"}'
        )

    pipeline = _pipeline(respond)
    try:
        verdict = await pipeline.adjudicate(_pair(), ())
    finally:
        await pipeline.close()

    assert verdict.same_entity is True
    assert verdict.confidence == 0.86
    assert "answer false" in seen["system"]
    assert "man in blue denim jacket" in seen["user"]
    assert "man in black t-shirt" in seen["user"]


async def test_a_negative_verdict_survives_parsing() -> None:
    pipeline = _pipeline(
        lambda _request: _async(
            _completion(
                '{"same_entity":false,"confidence":0.91,'
                '"discriminating_cue":"both visible at once at 00:12"}'
            )
        )
    )
    try:
        verdict = await pipeline.adjudicate(_pair(), ())
    finally:
        await pipeline.close()
    assert verdict.same_entity is False
    assert "00:12" in verdict.discriminating_cue


async def test_a_verdict_without_a_cue_is_rejected_rather_than_defaulted() -> None:
    """An unexplained yes is the one output that must never become an edge."""
    pipeline = _pipeline(
        lambda _request: _async(
            _completion('{"same_entity":true,"confidence":0.99,"discriminating_cue":"  "}')
        )
    )
    try:
        with pytest.raises(ModelOutputError):
            await pipeline.adjudicate(_pair(), ())
    finally:
        await pipeline.close()


async def test_an_extra_key_is_rejected() -> None:
    pipeline = _pipeline(
        lambda _request: _async(
            _completion(
                '{"same_entity":true,"confidence":0.9,"discriminating_cue":"cue","merge":true}'
            )
        )
    )
    try:
        with pytest.raises(ModelOutputError):
            await pipeline.adjudicate(_pair(), ())
    finally:
        await pipeline.close()


def _pair() -> EntityPair:
    return EntityPair(
        left=_candidate("entity_a", "man in blue denim jacket"),
        right=_candidate("entity_b", "man in black t-shirt"),
    )


def _candidate(entity_id: str, name: str) -> EntityCandidate:
    return EntityCandidate(
        entity=Entity(
            entity_id=EntityId(entity_id),
            tenant_id=TenantId("tenant_01"),
            entity_type=EntityType.PERSON,
            canonical_name=name,
            created_at=NOW,
        ),
        evidence_ids=(EvidenceId("evidence_1"),),
    )


def _completion(content: str) -> httpx.Response:
    """The generator streams, so the double has to speak server-sent events."""
    event = {
        "id": "completion_entity_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
    )


async def _async(response: httpx.Response) -> httpx.Response:
    return response


class _Harness(EntityResolutionPipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(generator)
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


def _pipeline(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> _Harness:
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return _Harness(
        OpenAIGenerator(client, ModelReference(model_id="qwen3.8-max", revision="rev-1"))
    )
