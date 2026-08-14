"""Contract tests for the evidence-first Episode pipeline."""

import json
from collections.abc import Callable, Coroutine, Iterator
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines import EpisodePipeline
from mindbridge.core import (
    DomainInvariantError,
    Event,
    EventId,
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    ModelReference,
    ObservationId,
    TenantId,
)
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


async def test_episode_consolidator_inspects_native_evidence_and_preserves_revision() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        system_prompt = cast(str, messages[0]["content"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        assert "temporal continuity and a shared goal" in system_prompt
        assert "share only a person, place, object" in system_prompt
        assert {part["type"] for part in content} >= {"video_url", "input_audio"}
        assert "Propose supported episode" in cast(str, content[-1]["text"])
        context = cast(str, content[0]["text"])
        assert "event_01" in context and "evidence_02" in context
        return _streaming_response(
            {
                "episodes": [
                    {
                        "event_ids": ["event_01", "event_02"],
                        "description": "A person retrieves a tool and explains the repair.",
                        "salience": 0.86,
                    }
                ]
            },
            fingerprint="episode-serving-revision-01",
        )

    consolidator = _consolidator(respond)
    events, evidence = _candidates()
    try:
        result = await consolidator.propose_episodes(events, evidence)
    finally:
        await consolidator.close()

    assert result.episodes[0].event_ids == (EventId("event_01"), EventId("event_02"))
    assert result.model_reference.revision == "episode-serving-revision-01"
    assert result.prompt_version == "consolidate_episodes_v2"


async def test_episode_consolidator_rejects_an_unknown_event_id() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "episodes": [
                    {
                        "event_ids": ["event_01", "event_fabricated"],
                        "description": "Unsupported grouping",
                        "salience": 0.5,
                    }
                ]
            }
        )

    consolidator = _consolidator(respond)
    try:
        with pytest.raises(ModelOutputError, match="unknown event"):
            await consolidator.propose_episodes(*_candidates())
    finally:
        await consolidator.close()


async def test_episode_consolidator_requires_every_exact_evidence_span() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response({"episodes": []})

    consolidator = _consolidator(respond)
    events, evidence = _candidates()
    try:
        with pytest.raises(DomainInvariantError, match="each exact"):
            await consolidator.propose_episodes(events, evidence[:1])
    finally:
        await consolidator.close()


async def test_episode_consolidator_retries_invalid_structure_in_json_mode() -> None:
    responses: Iterator[object] = iter(
        (
            {
                "episodes": [
                    {
                        "event_ids": ["event_01"],
                        "description": "A singleton is not an episode.",
                        "salience": 0.5,
                    }
                ]
            },
            {"episodes": []},
        )
    )
    response_formats: list[object] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        response_formats.append(payload.get("response_format"))
        return _streaming_response(next(responses))

    consolidator = _consolidator(respond)
    try:
        result = await consolidator.propose_episodes(*_candidates())
    finally:
        await consolidator.close()

    assert result.episodes == ()
    assert response_formats == [None, {"type": "json_object"}]


def _consolidator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> "_EpisodeHarness":
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=http_client,
        max_retries=0,
    )
    return _EpisodeHarness(
        OpenAIGenerator(
            client,
            ModelReference(model_id="qwen3.8-max", revision="deployment-revision"),
        )
    )


class _EpisodeHarness(EpisodePipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(generator)
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


def _candidates() -> tuple[tuple[Event, ...], tuple[ResolvedEvidence, ...]]:
    events = tuple(
        Event(
            event_id=EventId(f"event_{index:02d}"),
            tenant_id=TenantId("tenant_01"),
            observation_ids=(ObservationId(f"observation_{index:02d}"),),
            evidence_ids=(EvidenceId(f"evidence_{index:02d}"),),
            occurred_at=NOW + timedelta(seconds=index * 5),
            ended_at=NOW + timedelta(seconds=index * 5 + 4),
            description=description,
            salience=0.8,
            created_at=NOW,
            model_reference=ModelReference(model_id="omni", revision="perception-revision"),
            prompt_version="perceive_events_v3",
        )
        for index, description in enumerate(
            ("A person retrieves a tool.", "The person explains the repair."),
            start=1,
        )
    )
    evidence = tuple(
        _evidence(index, kind)
        for index, kind in enumerate((MediaKind.VIDEO, MediaKind.AUDIO), start=1)
    )
    return events, evidence


def _evidence(index: int, kind: MediaKind) -> ResolvedEvidence:
    suffix = f"{index:02d}"
    media_id = MediaObjectId(f"media_{suffix}")
    extension = "mp4" if kind is MediaKind.VIDEO else "wav"
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
            duration_ms=4_000,
        ),
        media_url=f"https://objects.example.test/clip_{suffix}.{extension}",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _streaming_response(payload: object, *, fingerprint: str | None = None) -> httpx.Response:
    event = {
        "id": "completion_episode_01",
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
