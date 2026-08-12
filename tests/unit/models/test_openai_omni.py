"""Contract tests for the OpenAI-compatible Omni answer boundary."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.ports import ResolvedQueryMedia
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import (
    EvidenceId,
    EvidenceSpan,
    MediaKind,
    MediaObject,
    MediaObjectId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelOutputError,
    ObservationId,
    TenantId,
    VerificationStatus,
)
from mindbridge.models.openai_omni import OpenAIOmniAnswerer, normalize_openai_base_url

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


async def test_omni_streams_raw_av_and_validates_answer() -> None:
    """The official SDK carries original media rather than text-only captions."""

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        user_content = cast(list[dict[str, object]], messages[1]["content"])
        video = next(item for item in user_content if item["type"] == "video_url")

        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer unit-test-key"
        assert payload["model"] == "qwen3.8-max"
        assert payload["stream"] is True
        assert payload["modalities"] == ["text"]
        assert {item["type"] for item in user_content} >= {
            "image_url",
            "video_url",
            "input_audio",
        }
        assert cast(dict[str, str], video["video_url"])["url"].endswith("media_video")
        assert video["fps"] == 1.0
        assert video["max_pixels"] == 200_704
        assert '"start_ms":1000' in cast(str, user_content[0]["text"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":"The screwdriver is beside the blue toolbox",',
                '"confidence":0.87}',
            ),
        )

    answerer = _answerer(respond)
    assert answerer.model_reference.model_id == "qwen3.8-max"
    assert answerer.model_reference.revision == "deployment-revision"
    evidence = (
        _resolved_evidence(MediaKind.IMAGE, "image.jpg", "media_image", 0),
        _resolved_evidence(MediaKind.VIDEO, "clip.mp4", "media_video", 1_000),
        _resolved_evidence(MediaKind.AUDIO, "meeting.wav", "media_audio", 2_000),
    )
    memory = _memory(tuple(item.evidence_span.evidence_id for item in evidence))

    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is the tool?")),
            (memory,),
            evidence,
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer == "The screwdriver is beside the blue toolbox"
    assert answer.confidence == 0.87


async def test_omni_inspects_native_query_media_before_candidate_evidence() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])

        labels = [item["text"] for item in content if item["type"] == "text"]
        media_parts = [item for item in content if item["type"] != "text"]
        assert labels[1:] == [
            "Query media_object_id=query_image follows.",
            "Query media_object_id=query_video follows.",
            "Query media_object_id=query_audio follows.",
        ]
        assert [item["type"] for item in media_parts] == [
            "image_url",
            "video_url",
            "input_audio",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":"matching moment","confidence":0.9}'),
        )

    query_sources = (
        _resolved_evidence(MediaKind.IMAGE, "query.jpg", "query_image", 0),
        _resolved_evidence(MediaKind.VIDEO, "query.mp4", "query_video", 0),
        _resolved_evidence(MediaKind.AUDIO, "query.wav", "query_audio", 0),
    )
    query_media = tuple(
        ResolvedQueryMedia(
            media_object=item.media_object,
            media_url=item.media_url,
            media_url_expires_at=item.media_url_expires_at,
        )
        for item in query_sources
    )
    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(
                    text="Find a matching moment",
                    media_object_ids=("query_image", "query_video", "query_audio"),
                ),
            ),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (),
            query_media=query_media,
        )
    finally:
        await answerer.close()

    assert answer.answer == "matching moment"


async def test_omni_selects_occurrences_with_schema_validated_candidate_ids() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        assert "exhaustive occurrence-verification" in cast(str, messages[0]["content"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"memory_ids":["memory_01"]}'),
        )

    answerer = _answerer(respond)
    memory = _memory((), verification_status=VerificationStatus.ATTESTED)
    try:
        selected = await answerer.select_occurrences(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="count the tools")),
            (memory,),
            (),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert selected == (memory.memory_id,)


@pytest.mark.parametrize(
    "content",
    [
        '{"memory_ids":["memory_01","memory_01"]}',
        '{"memory_ids":["memory_unknown"]}',
    ],
)
async def test_omni_rejects_invalid_occurrence_selection(content: str) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(content),
        )

    answerer = _answerer(respond)
    try:
        with pytest.raises(ModelOutputError, match="occurrence model"):
            await answerer.select_occurrences(
                RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="count the tools")),
                (_memory((), verification_status=VerificationStatus.ATTESTED),),
                (),
                query_media=(),
            )
    finally:
        await answerer.close()


async def test_omni_rejects_invalid_structured_output() -> None:
    """Provider JSON cannot bypass the answer confidence invariant."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":null,"confidence":0.9}'),
        )

    answerer = _answerer(respond)
    evidence = (_resolved_evidence(MediaKind.IMAGE, "image.jpg", "media_image", 0),)

    try:
        with pytest.raises(ModelOutputError, match="invalid structured output"):
            await answerer.answer(
                RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What happened?")),
                (_memory((evidence[0].evidence_span.evidence_id,)),),
                evidence,
                query_media=(),
            )
    finally:
        await answerer.close()


async def test_omni_uses_attested_source_statement_without_media() -> None:
    """Explicit text is labeled as a report and never promoted to observed evidence."""

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        assert 'marked "attested"' in cast(str, messages[0]["content"])
        user_content = cast(list[dict[str, object]], messages[1]["content"])
        assert '"verification_status":"attested"' in cast(str, user_content[0]["text"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":"Caroline plans to become a counselor",', '"confidence":0.8}'
            ),
        )

    answerer = _answerer(respond)
    memory = _memory((), verification_status=VerificationStatus.ATTESTED)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What is her plan?")),
            (memory,),
            (),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer == "Caroline plans to become a counselor"
    assert answer.confidence == 0.8


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://vlm.example.test/api/v1/chat/completions", "https://vlm.example.test/api/v1"),
        ("https://vlm.example.test/api/v1/embeddings", "https://vlm.example.test/api/v1"),
        ("https://vlm.example.test/api/v1/", "https://vlm.example.test/api/v1"),
    ],
)
def test_normalize_openai_base_url_accepts_root_or_completion_url(
    endpoint: str,
    expected: str,
) -> None:
    """Deployment configuration may use the full endpoint supplied by an operator."""
    assert normalize_openai_base_url(endpoint) == expected


def _answerer(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIOmniAnswerer:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_openai_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=http_client,
        max_retries=0,
    )
    return OpenAIOmniAnswerer(client, model_revision="deployment-revision")


def _resolved_evidence(
    kind: MediaKind,
    filename: str,
    media_object_id: str,
    start_ms: int,
) -> ResolvedEvidence:
    media_id = MediaObjectId(media_object_id)
    return ResolvedEvidence(
        evidence_span=EvidenceSpan(
            evidence_id=EvidenceId(f"evidence_{media_object_id}"),
            tenant_id=TenantId("tenant_01"),
            observation_id=ObservationId(f"observation_{media_object_id}"),
            media_object_id=media_id,
            start_ms=start_ms,
            end_ms=start_ms + 1_000,
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
            duration_ms=10_000,
        ),
        media_url=f"https://objects.example.test/{media_object_id}",
        media_url_expires_at=NOW + timedelta(minutes=5),
    )


def _memory(
    evidence_ids: tuple[EvidenceId, ...],
    *,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=MemoryId("memory_01"),
        tenant_id=TenantId("tenant_01"),
        memory_type=MemoryType.EPISODIC,
        summary="The red screwdriver was left beside the blue toolbox.",
        evidence_ids=evidence_ids,
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=verification_status,
    )


def _completion_stream(*content_parts: str) -> str:
    events = [
        {
            "id": "completion_01",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen3.8-max",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": None,
                }
            ],
        }
        for content in content_parts
    ]
    events.append(
        {
            "id": "completion_01",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen3.8-max",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
