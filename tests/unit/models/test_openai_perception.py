"""Contract tests for raw AV event perception."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.core import (
    AnonymousIdentityObservation,
    DeviceId,
    EvidenceId,
    EvidenceSpan,
    IdentityKind,
    MediaKind,
    MediaObject,
    MediaObjectId,
    ModelOutputError,
    ModelReference,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.models.openai_omni import normalize_openai_base_url
from mindbridge.models.openai_perception import OpenAIOmniEventPerceiver

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


async def test_omni_perception_returns_grounded_event_and_provider_revision() -> None:
    """The adapter sends original AV and preserves evidence and model provenance."""

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        system_prompt = cast(str, messages[0]["content"])
        content = cast(list[dict[str, object]], messages[1]["content"])

        assert request.url.path == "/api/v1/chat/completions"
        assert "reasoning_effort" not in payload
        assert "response_format" not in payload
        assert "atomic semantic" in system_prompt
        assert "spoken wording and visible text exactly" in system_prompt
        assert "supplied opaque identity_id" in system_prompt
        assert "visual_bbox_xyxy" in system_prompt
        assert "distinctive appearance" in system_prompt
        assert "truncated sentence" in system_prompt
        assert "cause/effect" in system_prompt
        assert {item["type"] for item in content} >= {"video_url", "input_audio"}
        assert cast(str, content[0]["text"]).startswith("<observation_context>")
        assert cast(str, content[-1]["text"]).startswith("<final_task>")
        assert '"evidence_id":"evidence_video"' in cast(str, content[0]["text"])
        assert '"identity_id":"person_device_01"' in cast(str, content[0]["text"])
        assert '"visual_bbox_xyxy":[0.1,0.2,0.4,0.8]' in cast(str, content[0]["text"])
        assert "embedding" not in cast(str, content[0]["text"])
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 500,
                        "end_ms": 3_500,
                        "description": "A person places a red tool beside a toolbox while speaking.",
                        "salience": 0.82,
                        "evidence_ids": ["evidence_video", "evidence_audio"],
                        "entities": [
                            {
                                "entity_type": "object",
                                "canonical_name": "red tool",
                                "confidence": 0.94,
                                "evidence_ids": ["evidence_video"],
                            },
                            {
                                "entity_type": "object",
                                "canonical_name": "toolbox",
                                "confidence": 0.9,
                                "evidence_ids": ["evidence_video"],
                            },
                        ],
                        "claims": [
                            {
                                "claim_type": "relation",
                                "statement": "The red tool is beside the toolbox.",
                                "confidence": 0.88,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 500,
                                "valid_to_ms": 3_500,
                                "entity_indices": [0, 1],
                            }
                        ],
                    }
                ]
            },
            fingerprint="qwen-serving-revision-01",
        )

    perceiver = _perceiver(respond)
    evidence = (
        _evidence(MediaKind.VIDEO, "clip.mp4", "video"),
        _evidence(MediaKind.AUDIO, "clip.wav", "audio"),
    )
    try:
        result = await perceiver.perceive_events(_observation(), evidence)
    finally:
        await perceiver.close()

    assert result.events[0].start_ms == 500
    assert result.events[0].evidence_ids == (
        EvidenceId("evidence_video"),
        EvidenceId("evidence_audio"),
    )
    assert result.model_reference.revision == "qwen-serving-revision-01"
    assert result.prompt_version == "perceive_events_v9"
    assert [entity.canonical_name for entity in result.events[0].entities] == [
        "red tool",
        "toolbox",
    ]
    assert result.events[0].claims[0].entity_indices == (0, 1)


async def test_omni_perception_retries_invalid_output_once_in_json_mode() -> None:
    calls = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload: dict[str, object] = json.loads(request.content)
        if calls == 1:
            assert "response_format" not in payload
            content = {"events": [{"start_ms": 0}]}
        else:
            assert payload["response_format"] == {"type": "json_object"}
            content = {"events": []}
        return _streaming_response(content)

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert calls == 2
    assert result.events == ()


async def test_omni_perception_rejects_detail_evidence_outside_its_event() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "One visible event",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                        "entities": [
                            {
                                "entity_type": "topic",
                                "canonical_name": "unsupported detail",
                                "confidence": 0.5,
                                "evidence_ids": ["evidence_audio"],
                            }
                        ],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        with pytest.raises(ModelOutputError, match="outside its event"):
            await perceiver.perceive_events(
                _observation(),
                (
                    _evidence(MediaKind.VIDEO, "clip.mp4", "video"),
                    _evidence(MediaKind.AUDIO, "clip.wav", "audio"),
                ),
            )
    finally:
        await perceiver.close()


async def test_omni_perception_rejects_unknown_evidence() -> None:
    """A model cannot fabricate provenance IDs not present in its input."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "Unsupported event",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_fabricated"],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        with pytest.raises(ModelOutputError, match="unknown evidence"):
            await perceiver.perceive_events(
                _observation(),
                (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
            )
    finally:
        await perceiver.close()


async def test_omni_perception_rejects_event_outside_observation() -> None:
    """Provider timestamps cannot extend a memory beyond captured time."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 5_000,
                        "description": "Overlong event",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        with pytest.raises(ModelOutputError, match="exceeds observation"):
            await perceiver.perceive_events(
                _observation(),
                (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
            )
    finally:
        await perceiver.close()


def _perceiver(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIOmniEventPerceiver:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_openai_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=http_client,
        max_retries=0,
    )
    return OpenAIOmniEventPerceiver(client, model_revision="deployment-revision")


def _observation() -> Observation:
    return Observation(
        observation_id=ObservationId("observation_01"),
        tenant_id=TenantId("tenant_01"),
        device_id=DeviceId("device_01"),
        boot_id="boot_01",
        sequence=1,
        sensor=SensorKind.CAMERA,
        media_object_ids=(MediaObjectId("media_video"), MediaObjectId("media_audio")),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=4),
        observed_at=NOW,
        clock_offset_ms=0,
        identity_observations=(
            AnonymousIdentityObservation(
                identity_id="person_device_01",
                kind=IdentityKind.FACE,
                start_ms=500,
                end_ms=3_500,
                confidence=0.91,
                model_reference=ModelReference(
                    model_id="insightface/buffalo_l",
                    revision="1.0.1",
                ),
                visual_bbox_xyxy=(0.1, 0.2, 0.4, 0.8),
            ),
        ),
    )


def _evidence(kind: MediaKind, filename: str, suffix: str) -> ResolvedEvidence:
    media_id = MediaObjectId(f"media_{suffix}")
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(f"evidence_{suffix}"),
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
            kind=kind,
            uri=f"s3://memory/tenants/tenant_01/{filename}",
            sha256="a" * 64,
            size_bytes=100,
            created_at=NOW,
            duration_ms=4_000,
        ),
        media_url=f"https://objects.example.test/{suffix}",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _streaming_response(payload: object, *, fingerprint: str | None = None) -> httpx.Response:
    event = {
        "id": "completion_01",
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
    content = f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=content,
    )
