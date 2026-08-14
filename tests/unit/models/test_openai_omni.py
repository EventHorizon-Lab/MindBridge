"""Contract tests for the OpenAI-compatible Omni answer boundary."""

import json
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mindbridge import telemetry
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
from mindbridge.models import openai_chat
from mindbridge.models.openai_media import media_input_span_attributes
from mindbridge.models.openai_omni import OpenAIOmniAnswerer, normalize_openai_base_url

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_media_input_span_attributes_deduplicate_requested_video_work() -> None:
    video = _resolved_evidence(MediaKind.VIDEO, "clip.mp4", "video", 0).media_object

    attributes = media_input_span_attributes(
        (video, video),
        video_frames_per_second=1.5,
    )

    assert attributes == {
        "mindbridge.model.input.media_count": 1,
        "mindbridge.model.input.duration_known_count": 1,
        "mindbridge.model.input.video_seconds": 10.0,
        "mindbridge.model.input.audio_seconds": 0,
        "mindbridge.model.input.estimated_video_frames": 15.0,
    }


async def test_omni_streams_raw_av_and_validates_answer() -> None:
    """The official SDK carries original media rather than text-only captions."""

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        system_prompt = cast(str, messages[0]["content"])
        user_content = cast(list[dict[str, object]], messages[1]["content"])
        video = next(item for item in user_content if item["type"] == "video_url")
        audio = next(item for item in user_content if item["type"] == "input_audio")

        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer unit-test-key"
        assert payload["model"] == "qwen3.8-max"
        assert payload["stream"] is True
        assert payload["modalities"] == ["text"]
        assert "response_format" not in payload
        assert payload["reasoning_effort"] == "low"
        assert 'For yes/no questions, answer "Yes" or "No".' in system_prompt
        assert "answer string is not an evidence report" in system_prompt
        assert "different named person does not support" in system_prompt
        assert "Missing evidence is not evidence of" in system_prompt
        assert "standalone search" in system_prompt
        assert '"cannot be answered" choice is a task answerability option' in system_prompt
        assert "compare candidate occurrence intervals" in system_prompt
        assert {item["type"] for item in user_content} >= {
            "image_url",
            "video_url",
            "input_audio",
        }
        assert cast(dict[str, str], video["video_url"])["url"].endswith("media_video")
        assert video["fps"] == 1.0
        assert video["max_pixels"] == 200_704
        assert cast(dict[str, str], audio["input_audio"]) == {
            "data": "https://objects.example.test/media_audio",
            "format": "wav",
        }
        assert '"start_ms":1000' in cast(str, user_content[0]["text"])
        assert '"ended_at":"2026-08-11T12:00:00+00:00"' in cast(str, user_content[0]["text"])
        assert "Where is the tool?" in cast(str, user_content[-1]["text"])
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
    assert answerer.prompt_version == "answer_from_evidence_v10"
    assert answerer.occurrence_prompt_version == "select_occurrences_v2"
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


async def test_omni_returns_bounded_retrieval_queries_when_evidence_is_missing() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        assert '"attempted_retrieval_queries":["tool location"]' in cast(str, content[0]["text"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":null,"confidence":0.0,',
                '"retrieval_queries":["person_device_01 blue toolbox"]}',
            ),
        )

    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="Where did the person put the tool?"),
            ),
            (),
            (),
            query_media=(),
            attempted_retrieval_queries=("tool location",),
        )
    finally:
        await answerer.close()

    assert answer.answer is None
    assert answer.retrieval_queries == ("person_device_01 blue toolbox",)


async def test_omni_retries_invalid_answer_once_in_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_TRACER", provider.get_tracer("mindbridge-test"))
    ttft_clock = iter((10.0, 10.125, 20.0, 20.25))
    monkeypatch.setattr(openai_chat, "perf_counter", lambda: next(ttft_clock))

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload: dict[str, object] = json.loads(request.content)
        if calls == 1:
            assert "response_format" not in payload
            content = "not json"
        else:
            assert payload["response_format"] == {"type": "json_object"}
            content = '{"answer":"blue toolbox","confidence":0.8}'
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(content, usage=(7, 3)),
        )

    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is it?")),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert calls == 2
    assert answer.answer == "blue toolbox"
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "mindbridge.model.stream_completion"
    ]
    attributes = [span.attributes or {} for span in spans]
    assert [item["mindbridge.model.structured_attempt"] for item in attributes] == [1, 2]
    assert [item["mindbridge.model.json_mode"] for item in attributes] == [False, True]
    assert [item["mindbridge.model.input_tokens"] for item in attributes] == [7, 7]
    assert [item["mindbridge.model.output_tokens"] for item in attributes] == [3, 3]
    assert [item["mindbridge.model.ttft_seconds"] for item in attributes] == pytest.approx(
        [0.125, 0.25]
    )
    answer_span = next(
        span for span in exporter.get_finished_spans() if span.name == "mindbridge.model.answer"
    )
    assert (answer_span.attributes or {})["mindbridge.model.structured_retry_count"] == 1
    assert all(span.parent == answer_span.context for span in spans)
    assert "not json" not in repr(attributes)
    provider.shutdown()


async def test_omni_accepts_one_provider_added_json_code_fence() -> None:
    calls = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('```json\n{"answer":"blue toolbox","confidence":0.8}\n```'),
        )

    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is it?")),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert calls == 1
    assert answer.answer == "blue toolbox"


async def test_omni_keeps_provisional_answer_with_search_queries() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":"somewhere at home","confidence":0.7,',
                '"retrieval_queries":["blue toolbox exact location"]}',
            ),
        )

    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="Where is it?")),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer == "somewhere at home"
    assert answer.confidence == 0.7
    assert answer.retrieval_queries == ("blue toolbox exact location",)


async def test_omni_inspects_native_query_media_before_candidate_evidence() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])

        labels = [item["text"] for item in content if item["type"] == "text"]
        media_parts = [item for item in content if item["type"] != "text"]
        assert labels[1:4] == [
            "Query media_object_id=query_image follows.",
            "Query media_object_id=query_video follows.",
            "Query media_object_id=query_audio follows.",
        ]
        assert "Find a matching moment" in cast(str, labels[-1])
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
        system_prompt = cast(str, messages[0]["content"])
        assert "distinct occurrences" in system_prompt
        assert "same evidence and time represent one occurrence" in system_prompt
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
        assert 'An "attested" summary is an exact caller statement' in cast(
            str, messages[0]["content"]
        )
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
    return OpenAIOmniAnswerer(
        client,
        model_revision="deployment-revision",
        reasoning_effort="low",
    )


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


def _completion_stream(
    *content_parts: str,
    usage: tuple[int, int] | None = None,
) -> str:
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
    if usage is not None:
        input_tokens, output_tokens = usage
        events.append(
            {
                "id": "completion_01",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "qwen3.8-max",
                "choices": [],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
