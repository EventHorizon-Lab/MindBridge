"""Contract tests for the provider-neutral Answer and Occurrence pipelines."""

import json
from collections.abc import Callable, Coroutine
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.application.perception import ResolvedEvidence
from mindbridge.application.pipelines import AnswerPipeline, OccurrencePipeline
from mindbridge.application.pipelines import structured as structured_module
from mindbridge.application.pipelines.answer import QUERY_MEDIA_PART_COST
from mindbridge.application.pipelines.evidence import DEFAULT_MAX_EVIDENCE_MEDIA_PARTS
from mindbridge.application.ports import GeneratedAnswer, ResolvedQueryMedia
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
    ModelReference,
    ObservationId,
    TenantId,
    VerificationStatus,
)
from mindbridge.models import openai as openai_module
from mindbridge.models.openai import OpenAIGenerator, normalize_base_url
from mindbridge.prompts import ANSWER_FROM_EVIDENCE_PROMPT, SELECT_OCCURRENCES_PROMPT

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


async def test_answer_pipeline_streams_raw_av_and_validates_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official SDK carries original media rather than text-only captions."""
    span_attributes: list[dict[str, str | int | float | bool]] = []
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(openai_module, "set_current_span_attributes", span_attributes.append)
    monkeypatch.setattr(openai_module, "perf_counter", lambda: next(clock))

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
    assert answerer.prompt_version == "answer_from_evidence_v12"
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
    assert {
        "mindbridge.model.input.media_count": 3,
        "mindbridge.model.json_mode": False,
    }.items() <= span_attributes[0].items()
    assert {"mindbridge.model.ttft_seconds": 0.25} in span_attributes
    assert {
        "mindbridge.model.input_tokens": 11,
        "mindbridge.model.output_tokens": 3,
        "mindbridge.model.total_tokens": 14,
    } in span_attributes


async def test_answer_pipeline_returns_bounded_queries_when_evidence_is_missing() -> None:
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


async def test_answer_pipeline_retries_invalid_answer_once_in_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    span_attributes: list[dict[str, str | int | float | bool]] = []
    monkeypatch.setattr(
        structured_module,
        "set_current_span_attributes",
        span_attributes.append,
    )

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
            content=_completion_stream(content),
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
    assert {"mindbridge.model.structured_retry_count": 1} in span_attributes


async def test_answer_pipeline_accepts_one_provider_added_json_code_fence() -> None:
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


async def test_answer_pipeline_keeps_provisional_answer_with_search_queries() -> None:
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


async def test_answer_pipeline_inspects_query_media_before_candidate_evidence() -> None:
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


async def test_a_span_signed_to_its_clip_is_announced_with_the_clips_container() -> None:
    """The container format has to describe the bytes actually sent.

    A span is signed to the derived clip the write path cut, which is WAV, while the recording
    it was cut from is M4A. Announcing the source's container declared WAV bytes as m4a on every
    audio span whose recording was not already WAV.
    """

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        audio = next(item for item in content if item["type"] == "input_audio")
        labels = [item["text"] for item in content if item["type"] == "text"]

        assert cast(dict[str, str], audio["input_audio"]) == {
            "data": "https://objects.example.test/clips/deadbeef.wav?sig=x",
            "format": "wav",
        }
        # The source is still the identity the model is asked to cite.
        assert "Source media_object_id=media_audio follows." in labels
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":"they agreed","confidence":0.6}'),
        )

    span = _resolved_evidence(MediaKind.AUDIO, "recording.m4a", "media_audio", 2_000)
    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What was agreed?")),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (_signed_to_clip(span, "deadbeef.wav"),),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer == "they agreed"


async def test_query_media_does_not_suppress_the_clips_cut_from_its_own_source() -> None:
    """Recalling by a stored object must not drop that object's span-scoped clips.

    The exclusion exists so one set of bytes is not attached twice. Once a span is signed to its
    own derived clip, the clip and the source are different bytes, and excluding on the source id
    dropped exactly the cheap, span-scoped bytes the question was about.
    """

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        attached = [
            cast(dict[str, str], item["video_url"])["url"]
            for item in content
            if item["type"] == "video_url"
        ]

        assert attached == [
            "https://objects.example.test/media_video",
            "https://objects.example.test/clips/first.mp4?sig=x",
            "https://objects.example.test/clips/second.mp4?sig=x",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":"twice","confidence":0.5}'),
        )

    span = _resolved_evidence(MediaKind.VIDEO, "recording.mp4", "media_video", 0)
    evidence = tuple(_signed_to_clip(span, name) for name in ("first.mp4", "second.mp4"))
    query_media = (
        ResolvedQueryMedia(
            media_object=span.media_object,
            media_url=span.media_url,
            media_url_expires_at=span.media_url_expires_at,
        ),
    )
    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(
                tenant_id="tenant_01",
                query=RecallQuery(text="How often?", media_object_ids=("media_video",)),
            ),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            evidence,
            query_media=query_media,
        )
    finally:
        await answerer.close()

    assert answer.answer == "twice"


async def test_query_media_and_evidence_share_one_media_part_ceiling() -> None:
    """Both halves of one call are attached against one budget, not one ceiling each.

    Query media carries no derived clip to substitute, so each object is the caller's
    full-resolution upload -- the shape measured at a 60 s gateway timeout with four of them.
    Eight of those plus a full page of evidence is that shape with the evidence still attached.
    """
    attached: list[int] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        attached.append(sum(1 for item in content if item["type"] != "text"))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":"bounded","confidence":0.5}'),
        )

    span = _resolved_evidence(MediaKind.VIDEO, "recording.mp4", "media_video", 0)
    query_media = tuple(
        ResolvedQueryMedia(
            media_object=replace(span.media_object, media_object_id=MediaObjectId(name)),
            media_url=f"https://objects.example.test/query/{name}.mp4",
            media_url_expires_at=span.media_url_expires_at,
        )
        for name in (f"query_{index:02d}" for index in range(8))
    )
    evidence = tuple(_signed_to_clip(span, f"{index:02d}.mp4") for index in range(40))
    answerer = _answerer(respond)
    try:
        for media in (query_media, query_media[:1]):
            await answerer.answer(
                RecallRequest(
                    tenant_id="tenant_01",
                    query=RecallQuery(
                        text="How many?",
                        media_object_ids=tuple(item.media_object.media_object_id for item in media),
                    ),
                ),
                (_memory((), verification_status=VerificationStatus.ATTESTED),),
                evidence,
                query_media=media,
            )
    finally:
        await answerer.close()

    # Eight query objects: three are attached and they have spent the whole budget. One query
    # object: it and sixteen clips, because what it did not spend is still there for evidence.
    assert attached == [
        DEFAULT_MAX_EVIDENCE_MEDIA_PARTS // QUERY_MEDIA_PART_COST,
        1 + DEFAULT_MAX_EVIDENCE_MEDIA_PARTS - QUERY_MEDIA_PART_COST,
    ]


async def test_a_span_that_can_only_offer_the_whole_recording_sends_no_bytes() -> None:
    """Audio cut into several windows has no single clip covering it, and falls back.

    Falling back to the source is complete, but the source is the entire recording: a 70 s span
    of a two-hour file costs more on its own than the whole page of clips beside it, which is the
    timeout this ceiling exists to prevent. The span keeps its place in the prompt's text.
    """

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[1]["content"])
        attached = [item["type"] for item in content if item["type"] != "text"]

        assert attached == ["video_url"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"answer":"partly","confidence":0.4}'),
        )

    recording = replace(
        _resolved_evidence(MediaKind.AUDIO, "recording.m4a", "media_audio", 0),
        media_url="https://objects.example.test/media_audio",
    )
    unsubstituted = replace(
        recording,
        evidence_span=replace(recording.evidence_span, end_ms=70_000),
        media_object=replace(recording.media_object, duration_ms=2 * 60 * 60 * 1_000),
        attached_media_object=replace(recording.media_object, duration_ms=2 * 60 * 60 * 1_000),
    )
    span = _resolved_evidence(MediaKind.VIDEO, "recording.mp4", "media_video", 0)
    answerer = _answerer(respond)
    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What happened?")),
            (_memory((), verification_status=VerificationStatus.ATTESTED),),
            (unsubstituted, _signed_to_clip(span, "covered.mp4")),
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer == "partly"


async def test_occurrence_pipeline_selects_schema_validated_candidate_ids() -> None:
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
        '{"memory_ids":["memory_01","memory_unknown"]}',
        '{"memory_ids":["memory_unknown","memory_01",""],"note":"padded"}',
    ],
)
async def test_occurrence_pipeline_keeps_the_candidates_it_can_verify(content: str) -> None:
    """A repeat, an invented ID, or a padded key costs itself, not the batch's real selections.

    Enumeration verifies memories in batches, so one invented ID used to fail a whole count.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(content),
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


async def test_occurrence_pipeline_still_rejects_a_missing_selection() -> None:
    """`memory_ids` is required: an absent one is not the same statement as an empty one."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream('{"selected":["memory_01"]}'),
        )

    answerer = _answerer(respond)
    try:
        with pytest.raises(ModelOutputError, match="occurrence pipeline"):
            await answerer.select_occurrences(
                RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="count the tools")),
                (_memory((), verification_status=VerificationStatus.ATTESTED),),
                (),
                query_media=(),
            )
    finally:
        await answerer.close()


async def test_answer_pipeline_keeps_an_abstention_whose_confidence_contradicts_it() -> None:
    """The abstention is what the model decided; the confidence beside it is the typo.

    Read-path validation discarded 14 of 55 M3-Bench robot predictions on 2026-08-21, more than
    every wrong answer that benchmark made combined. Zeroing the confidence keeps the abstention
    exactly as given -- including the follow-up queries a second retrieval round needs, which
    used to be thrown away with it.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":null,"confidence":0.9,"retrieval_queries":["blue toolbox"]}'
            ),
        )

    answerer = _answerer(respond)
    evidence = (_resolved_evidence(MediaKind.IMAGE, "image.jpg", "media_image", 0),)

    try:
        answer = await answerer.answer(
            RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What happened?")),
            (_memory((evidence[0].evidence_span.evidence_id,)),),
            evidence,
            query_media=(),
        )
    finally:
        await answerer.close()

    assert answer.answer is None
    assert answer.confidence == 0.0
    assert answer.retrieval_queries == ("blue toolbox",)


async def test_answer_pipeline_keeps_an_answer_that_overran_its_output_contract() -> None:
    """An invented key, a third query, a repeat, a blank, and an unknown ordering hint all drop."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(
                '{"answer":"blue toolbox","confidence":0.8,"reasoning":"because",',
                '"retrieval_queries":["toolbox colour","toolbox colour","",'
                '"where the toolbox is","a third"],',
                '"temporal_order":"chronological"}',
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

    assert answer.answer == "blue toolbox"
    assert answer.retrieval_queries == ("toolbox colour", "where the toolbox is")
    assert answer.temporal_order == "relevance"


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"answer":"blue toolbox","confidence":4}', id="confidence-off-scale"),
        pytest.param('{"response":"blue toolbox","confidence":0.8}', id="renamed-answer"),
        pytest.param('{"answer":"blue toolbox"}', id="missing-confidence"),
    ],
)
async def test_answer_pipeline_still_rejects_what_it_cannot_repair(content: str) -> None:
    """Where the line is: a required field cannot be invented, and a number is not clamped.

    `confidence` on a 1-5 scale is not read as 1.0 -- abstention and its calibration were
    measured as sound, and fabricating a maximum here would be reporting a confidence nobody
    produced. A renamed `answer` leaves the answer missing, which is not an abstention either.
    """

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completion_stream(content),
        )

    answerer = _answerer(respond)
    try:
        with pytest.raises(ModelOutputError, match="invalid structured output"):
            await answerer.answer(
                RecallRequest(tenant_id="tenant_01", query=RecallQuery(text="What happened?")),
                (_memory((), verification_status=VerificationStatus.ATTESTED),),
                (),
                query_media=(),
            )
    finally:
        await answerer.close()


async def test_answer_pipeline_uses_attested_source_statement_without_media() -> None:
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
def test_normalize_base_url_accepts_root_or_completion_url(
    endpoint: str,
    expected: str,
) -> None:
    """Deployment configuration may use the full endpoint supplied by an operator."""
    assert normalize_base_url(endpoint) == expected


class _AnswerHarness:
    def __init__(self, generator: OpenAIGenerator) -> None:
        self._generator = generator
        self._answer = AnswerPipeline(generator)
        self._occurrences = OccurrencePipeline(generator)

    @property
    def model_reference(self) -> ModelReference:
        return ModelReference(model_id="qwen3.8-max")

    @property
    def prompt_version(self) -> str:
        return ANSWER_FROM_EVIDENCE_PROMPT.version

    @property
    def occurrence_prompt_version(self) -> str:
        return SELECT_OCCURRENCES_PROMPT.version

    async def answer(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
        attempted_retrieval_queries: tuple[str, ...] = (),
    ) -> GeneratedAnswer:
        return await self._answer.answer(
            request,
            memories,
            evidence,
            query_media=query_media,
            attempted_retrieval_queries=attempted_retrieval_queries,
        )

    async def select_occurrences(
        self,
        request: RecallRequest,
        memories: tuple[MemoryRecord, ...],
        evidence: tuple[ResolvedEvidence, ...],
        *,
        query_media: tuple[ResolvedQueryMedia, ...],
    ) -> tuple[MemoryId, ...]:
        return await self._occurrences.select_occurrences(
            request,
            memories,
            evidence,
            query_media=query_media,
        )

    async def close(self) -> None:
        await self._generator.close()


def _answerer(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> _AnswerHarness:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
        http_client=http_client,
        max_retries=0,
    )
    return _AnswerHarness(
        OpenAIGenerator(
            client,
            ModelReference(model_id="qwen3.8-max"),
            reasoning_effort="low",
        )
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


def _signed_to_clip(evidence: ResolvedEvidence, name: str) -> ResolvedEvidence:
    """One span as the read path resolves it: signed to the clip the write path cut for it."""
    span = evidence.evidence_span
    return replace(
        evidence,
        media_url=f"https://objects.example.test/clips/{name}?sig=x",
        attached_media_object=replace(
            evidence.media_object,
            media_object_id=MediaObjectId(f"clip_{name}"),
            uri=f"s3://memory/tenants/tenant_01/clips/{name}",
            duration_ms=span.end_ms - span.start_ms,
            derived_from_media_object_id=evidence.media_object.media_object_id,
        ),
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
            "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
        }
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
