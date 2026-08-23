"""Contract tests for the multimodal Perception pipeline."""

import json
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge import telemetry
from mindbridge.application.perception import (
    MAX_PERCEIVED_CLAIMS_PER_EVENT,
    MAX_PERCEIVED_ENTITIES_PER_EVENT,
    MAX_PERCEPTION_CLAIMS,
    MAX_PERCEPTION_ENTITIES,
    MAX_PERCEPTION_EVENTS,
    EventPerception,
    ResolvedEvidence,
)
from mindbridge.application.pipelines import PerceptionPipeline
from mindbridge.core import (
    AnonymousIdentityObservation,
    ClaimType,
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
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


async def test_perception_pipeline_returns_grounded_event_and_its_model() -> None:
    """The adapter sends original AV and preserves evidence and model provenance."""

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        system_prompt = cast(str, messages[0]["content"])
        content = cast(list[dict[str, object]], messages[1]["content"])

        assert request.url.path == "/api/v1/chat/completions"
        assert payload["max_tokens"] == 8_192
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
            fingerprint="qwen-serving-fingerprint-01",
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
    assert result.model_reference.model_id == "qwen3.8-max"
    assert result.prompt_version == "perceive_events_v11"
    assert [entity.canonical_name for entity in result.events[0].entities] == [
        "red tool",
        "toolbox",
    ]
    assert result.events[0].claims[0].entity_indices == (0, 1)


async def test_perception_pipeline_retries_invalid_output_once_in_json_mode() -> None:
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


async def test_perception_pipeline_drops_a_detail_whose_evidence_leaves_its_event() -> None:
    """A detail the event's own evidence does not support goes; the event does not go with it."""

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
                            },
                            {
                                "entity_type": "object",
                                "canonical_name": "cup",
                                "confidence": 0.7,
                                "evidence_ids": ["evidence_video"],
                            },
                        ],
                        "claims": [
                            {
                                "claim_type": "state",
                                "statement": "Heard elsewhere in the recording.",
                                "confidence": 0.6,
                                "evidence_ids": ["evidence_audio"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 1_000,
                            }
                        ],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (
                _evidence(MediaKind.VIDEO, "clip.mp4", "video"),
                _evidence(MediaKind.AUDIO, "clip.wav", "audio"),
            ),
        )
    finally:
        await perceiver.close()

    assert [entity.canonical_name for entity in result.events[0].entities] == ["cup"]
    assert result.events[0].claims == ()


async def test_perception_pipeline_drops_the_event_that_fabricated_its_evidence() -> None:
    """A model cannot fabricate provenance IDs, and its other events do not pay for the one that did."""

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
                    },
                    {
                        "start_ms": 1_000,
                        "end_ms": 2_000,
                        "description": "A person sets a cup down.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    },
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert [event.description for event in result.events] == ["A person sets a cup down."]


async def test_perception_pipeline_drops_an_event_reaching_past_the_observation() -> None:
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
                    },
                    {
                        "start_ms": 0,
                        "end_ms": 4_000,
                        "description": "A person sets a cup down.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    },
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert [event.end_ms for event in result.events] == [4_000]


async def test_perception_pipeline_drops_an_unknown_claim_type_and_keeps_the_event() -> None:
    """The enum violation class, which cost 28 of 61 write-path job failures on 2026-08-21.

    The one value the run actually asked for is handled at the enum boundary now, so what this
    covers is the class rather than that instance: any value outside the taxonomy costs its own
    claim and nothing else. It must stay a drop, not a substitution -- `claim_type` reaches a
    database CHECK constraint, so a repaired fifth value would only fail later, after the commit
    was already paid for.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "A person greets a guest.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                        "claims": [
                            {
                                "claim_type": "preference",
                                "statement": "The host prefers tea.",
                                "confidence": 0.6,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 1_000,
                            },
                            {
                                "claim_type": "state",
                                "statement": "A guest is at the door.",
                                "confidence": 0.7,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 1_000,
                            },
                        ],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    claims = result.events[0].claims
    assert [claim.statement for claim in claims] == ["A guest is at the door."]
    assert claims[0].claim_type is ClaimType.STATE


async def test_perception_pipeline_drops_a_claim_whose_validity_leaves_its_event() -> None:
    """Measured once in the 2026-08-21 run, where it cost the whole observation.

    The window is what the evidence covers, so validity reaching past it is not clamped back in.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "A person sets a cup down.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                        "claims": [
                            {
                                "claim_type": "state",
                                "statement": "The cup stays there all afternoon.",
                                "confidence": 0.6,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 3_500,
                            },
                            {
                                "claim_type": "state",
                                "statement": "The cup is on the table.",
                                "confidence": 0.8,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 500,
                                "valid_to_ms": 1_000,
                            },
                        ],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert [claim.valid_to_ms for claim in result.events[0].claims] == [1_000]


async def test_perception_pipeline_repoints_claims_when_an_entity_is_dropped() -> None:
    """Dropping an entity must move the claims that follow it, not re-point them at a neighbour.

    Entities are addressed by position, so leaving the numbering alone would turn a lost detail
    into a wrong memory: the surviving claim would describe whichever entity slid into the gap.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "A person hands over a cup.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                        "entities": [
                            {
                                "entity_type": "spacecraft",
                                "canonical_name": "not a known entity type",
                                "confidence": 0.5,
                                "evidence_ids": ["evidence_video"],
                            },
                            {
                                "entity_type": "person",
                                "canonical_name": "host",
                                "confidence": 0.9,
                                "evidence_ids": ["evidence_video"],
                            },
                            {
                                "entity_type": "object",
                                "canonical_name": "cup",
                                "confidence": 0.9,
                                "evidence_ids": ["evidence_video"],
                            },
                        ],
                        "claims": [
                            {
                                "claim_type": "relation",
                                "statement": "The host is holding the cup.",
                                "confidence": 0.8,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 1_000,
                                "entity_indices": [1, 2],
                            },
                            {
                                "claim_type": "state",
                                "statement": "About the entity that did not survive.",
                                "confidence": 0.4,
                                "evidence_ids": ["evidence_video"],
                                "valid_from_ms": 0,
                                "valid_to_ms": 1_000,
                                "entity_indices": [0],
                            },
                        ],
                    }
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    event = result.events[0]
    assert [entity.canonical_name for entity in event.entities] == ["host", "cup"]
    assert [claim.statement for claim in event.claims] == ["The host is holding the cup."]
    assert event.claims[0].entity_indices == (0, 1)


async def test_perception_pipeline_rejects_output_whose_every_event_was_dropped() -> None:
    """Tolerance stops where there is nothing left: an empty commit would report false success."""

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
        with pytest.raises(ModelOutputError, match="no usable event"):
            await perceiver.perceive_events(
                _observation(),
                (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
            )
    finally:
        await perceiver.close()


def _perceiver(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> "_PerceptionHarness":
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=http_client,
        max_retries=0,
    )
    return _PerceptionHarness(
        OpenAIGenerator(
            client,
            ModelReference(model_id="qwen3.8-max"),
        )
    )


class _PerceptionHarness(PerceptionPipeline):
    def __init__(self, generator: OpenAIGenerator) -> None:
        super().__init__(generator)
        self._owned_generator = generator

    async def close(self) -> None:
        await self._owned_generator.close()


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
                model_reference=ModelReference(model_id="insightface/buffalo_l"),
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


async def test_perception_pipeline_keeps_an_answer_that_carries_an_invented_key() -> None:
    """A key the model padded its answer with must not discard the perception around it.

    Measured on this deployment: forbidding extras threw away every event, entity and claim in
    a clip because the JSON also carried `"start_ms_note": null`. Retrying does not help, since
    the model reproduces the same key -- so the run paid for two full generations, stored
    nothing, and aborted. A repeated evidence id is repaired for the same reason: it says
    nothing a single mention does not.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "start_ms_note": None,
                "events": [
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "A person sets a cup down.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video", "evidence_video"],
                        "commentary": "not a field this pipeline reads",
                    }
                ],
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert len(result.events) == 1
    assert result.events[0].description == "A person sets a cup down."
    assert result.events[0].evidence_ids == ("evidence_video",)


async def test_perception_pipeline_never_accepts_a_renamed_or_inverted_event() -> None:
    """Dropping an element must not shade into accepting it.

    A renamed field leaves the field it replaced missing, and an inverted range is not reordered
    into a valid one: both events go, and neither contributes a repaired version of itself.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(
            {
                "events": [
                    {
                        "start_ms": 0,
                        "finish_ms": 1_000,
                        "description": "Renamed field",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    },
                    {
                        "start_ms": 900,
                        "end_ms": 100,
                        "description": "Inverted range",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    },
                    {
                        "start_ms": 0,
                        "end_ms": 1_000,
                        "description": "A person sets a cup down.",
                        "salience": 0.5,
                        "evidence_ids": ["evidence_video"],
                    },
                ]
            }
        )

    perceiver = _perceiver(respond)
    try:
        result = await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()

    assert [event.description for event in result.events] == ["A person sets a cup down."]


def _event_payload(*, entities: int = 0, claims: int = 0) -> dict[str, object]:
    """One valid event carrying however many valid details a cap test needs."""
    return {
        "start_ms": 0,
        "end_ms": 1_000,
        "description": "A person works through a sequence of small actions.",
        "salience": 0.5,
        "evidence_ids": ["evidence_video"],
        "entities": [
            {
                "entity_type": "object",
                "canonical_name": f"object {index}",
                "confidence": 0.7,
                "evidence_ids": ["evidence_video"],
            }
            for index in range(entities)
        ],
        "claims": [
            {
                "claim_type": "state",
                "statement": f"Something is the case, number {index}.",
                "confidence": 0.7,
                "evidence_ids": ["evidence_video"],
                "valid_from_ms": 0,
                "valid_to_ms": 1_000,
            }
            for index in range(claims)
        ],
    }


async def _perceive(payload: object) -> EventPerception:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _streaming_response(payload)

    perceiver = _perceiver(respond)
    try:
        return await perceiver.perceive_events(
            _observation(),
            (_evidence(MediaKind.VIDEO, "clip.mp4", "video"),),
        )
    finally:
        await perceiver.close()


async def test_a_clip_past_the_claim_total_keeps_the_claims_that_fit() -> None:
    """A processing limit is not a wrong value, so crossing it costs the surplus, not the clip.

    `perceive_events_v11` asks for one event per atomic action -- on the order of 10-25 events per
    clip against the 3.48 measured on 2026-08-21 -- which is what brings these totals within
    reach. They used to raise, which would have turned the density change into a new instance of
    the failure the rest of this module removes.
    """
    result = await _perceive({"events": [_event_payload(claims=60) for _ in range(6)]})

    assert len(result.events) == 6
    assert sum(len(event.claims) for event in result.events) == MAX_PERCEPTION_CLAIMS
    # Spent in event order: the clip keeps its timeline, and the tail loses its claims.
    assert [len(event.claims) for event in result.events] == [60, 60, 60, 60, 16, 0]


async def test_a_clip_past_the_entity_total_keeps_the_entities_that_fit() -> None:
    """And the claims that pointed past them go too, rather than being left pointing at nothing."""
    result = await _perceive(
        {
            "events": [
                {
                    **_event_payload(entities=60, claims=1),
                    "claims": [
                        {
                            "claim_type": "relation",
                            "statement": "About the last entity of this event.",
                            "confidence": 0.7,
                            "evidence_ids": ["evidence_video"],
                            "valid_from_ms": 0,
                            "valid_to_ms": 1_000,
                            "entity_indices": [59],
                        }
                    ],
                }
                for _ in range(6)
            ]
        }
    )

    assert sum(len(event.entities) for event in result.events) == MAX_PERCEPTION_ENTITIES
    assert [len(event.claims) for event in result.events] == [1, 1, 1, 1, 0, 0]


async def test_one_event_past_its_own_detail_limit_keeps_what_fits() -> None:
    """The per-event limit truncates for the same reason the totals do, and by the same rule."""
    result = await _perceive({"events": [_event_payload(entities=70, claims=70)]})

    assert len(result.events) == 1
    assert len(result.events[0].entities) == MAX_PERCEIVED_ENTITIES_PER_EVENT
    assert len(result.events[0].claims) == MAX_PERCEIVED_CLAIMS_PER_EVENT


async def test_a_clip_past_the_event_limit_keeps_the_events_that_fit() -> None:
    result = await _perceive(
        {"events": [_event_payload() for _ in range(MAX_PERCEPTION_EVENTS + 6)]}
    )

    assert len(result.events) == MAX_PERCEPTION_EVENTS


async def test_what_a_cap_discarded_is_recorded_apart_from_what_was_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncation is only honest if it is visible, and the two reasons need telling apart: a wrong
    value is the model's problem, a binding limit is ours."""
    recorded: list[dict[str, str | int | float | bool]] = []
    monkeypatch.setattr(telemetry, "set_current_span_attributes", recorded.append)

    await _perceive({"events": [_event_payload(claims=60) for _ in range(6)]})

    attributes = {key: value for item in recorded for key, value in item.items()}
    assert attributes["mindbridge.perception.over_cap_claim_count"] == 104
    assert attributes["mindbridge.perception.dropped_claim_count"] == 0


async def test_what_was_discarded_reaches_an_operator_with_no_collector(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The span attributes are not a sink on the deployment this fix exists for.

    A span carries them only while it is recording, and the documented recipe is a 0.1 sampler
    against a collector the evaluation did not run. On that box a model emitting one unsupported
    enum per claim looks exactly like a model with little to say: every job succeeds and the
    census is simply low. So the counts go somewhere that needs neither.
    """
    with caplog.at_level(logging.WARNING):
        await _perceive({"events": [_event_payload(claims=60) for _ in range(6)]})

    assert "mindbridge.perception.over_cap_claim_count=104" in caplog.text
    assert "mindbridge.perception.dropped_claim_count=0" in caplog.text
