"""Focused checks for the official-SDK model adapter."""

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import httpx2 as httpx
import openai
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mindbridge.models.openai_sdk as openai_backend
from mindbridge._telemetry import (
    GEN_AI_FINISH_REASONS,
    GEN_AI_TTFC,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    MODEL_TTFT,
    TOKEN_COMPLETE,
    TOKEN_REPORTED_REQUEST_COUNT,
    TOKEN_TOTAL,
    model_span,
    operation_span,
    token_modality_attribute,
)
from mindbridge.exceptions import ModelError, ModelOutputTruncatedError, ValidationError
from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    GenerationBackend,
    ModelInput,
    TranscriptionBackend,
)
from mindbridge.models.openai_sdk import UNKNOWN_ANSWER, OpenAIModels
from mindbridge.types import AbstentionReason, AssetRef, Modality, SearchHit

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)
ALL_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


def test_text_embedding_keeps_standard_batch_shape_and_restores_order() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "https://sdk.example.test/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer sdk-secret"
        assert payload == {
            "input": ["first", "second"],
            "model": "embed-model",
            "dimensions": 2,
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 5]},
                    {"index": 0, "embedding": [3, 4]},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        first, second = model.embed(
            (ModelInput(text="first"), ModelInput(text="second")), EmbedTask.QUERY
        )

    assert isinstance(model, EmbeddingBackend)
    assert isinstance(model, GenerationBackend)
    assert isinstance(model, TranscriptionBackend)
    assert model.embedding_space == "shared-space-v1"
    assert first == pytest.approx((0.6, 0.8))
    assert second == pytest.approx((0.0, 1.0))


def test_multimodal_embedding_preserves_asset_order_and_inlines_media(
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    video = _asset(tmp_path, "video", Modality.VIDEO, "video/mp4", b"video")
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"audio")

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sample = cast(list[list[dict[str, object]]], payload["input"])[0]
        content = cast(list[dict[str, object]], sample[0]["content"])
        assert [part["type"] for part in content] == [
            "text",
            "image_url",
            "video_url",
            "audio_url",
        ]
        for part, asset in zip(content[1:], (image, video, audio), strict=True):
            kind = cast(str, part["type"])
            path = asset.path
            assert path is not None
            expected = base64.b64encode(path.read_bytes()).decode()
            assert cast(dict[str, str], part[kind])["url"] == (
                f"data:{asset.media_type};base64,{expected}"
            )
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).embed(
            (ModelInput(text="remember", assets=(image, video, audio)),),
            EmbedTask.DOCUMENT,
        )

    assert result[0] == pytest.approx((0.6, 0.8))


def test_answer_maps_native_hit_media_and_abstains_without_hits(tmp_path: Path) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://sdk.example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer sdk-secret"
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        assert [part["type"] for part in content] == ["text", "text", "image_url"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "The cat is on the sofa."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(
        id="memory_1",
        content="",
        score=0.9,
        created_at=NOW,
        assets=(image,),
        modality=Modality.IMAGE,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        abstention = model.answer(ModelInput(text="Unknown?"), ())
        result = model.answer(ModelInput(text="Where is the cat?"), (hit,))

    assert abstention.answer == UNKNOWN_ANSWER
    assert abstention.abstained is True
    assert abstention.abstention_reason is AbstentionReason.NO_EVIDENCE
    assert result.answer == "The cat is on the sofa."
    assert result.hits == (hit,)
    assert len(requests) == 1


def test_answer_marks_the_grounded_unknown_sentinel_as_insufficient_evidence() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": UNKNOWN_ANSWER},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(id="memory_1", content="partial evidence", score=0.9, created_at=NOW)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), (hit,))

    assert result.hits == (hit,)
    assert result.abstained is True
    assert result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_answer_serializes_temporal_and_metadata_evidence() -> None:
    occurred_at = datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc)
    occurred_end = occurred_at + timedelta(minutes=5)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        hit = json.loads(content)["hits"][0]
        assert hit == {
            "memory_id": "memory_1",
            "content": "The red parcel arrived.",
            "score": 0.9,
            "memory_type": "semantic",
            "occurred_at": occurred_at.isoformat(),
            "occurred_end": occurred_end.isoformat(),
            "created_at": NOW.isoformat(),
            "metadata": {"dialog": "delivery", "turn": 7},
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "It arrived on August 26."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(
        id="memory_1",
        content="The red parcel arrived.",
        score=0.9,
        created_at=NOW,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
        metadata={"dialog": "delivery", "turn": 7},
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer = _model(_sdk_client(client)).answer("When did it arrive?", (hit,))

    assert answer.answer == "It arrived on August 26."


def test_answer_can_pin_sampling_for_reproducible_evaluation() -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        requests.append(payload)
        if payload.get("stream"):
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [{"index": 0, "delta": {"content": "Pinned."}, "finish_reason": "stop"}],
            }
            return httpx.Response(
                200,
                content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"index": 0, "message": {"content": "Pinned."}, "finish_reason": "stop"}
                ]
            },
        )

    hit = SearchHit(id="memory_1", content="evidence", score=0.9, created_at=NOW)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        models = _model(
            _sdk_client(client),
            generation_seed=17,
            generation_temperature=0.0,
            generation_max_tokens=512,
            generation_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        answer = models.answer("What?", (hit,))
        streamed = "".join(models.stream_answer(ModelInput(text="What?"), (hit,)))

    assert answer.answer == "Pinned."
    assert streamed == "Pinned."
    # Both request paths share one builder, so neither may drop the caller's provider controls.
    assert len(requests) == 2
    for payload in requests:
        assert payload["seed"] == 17
        assert payload["temperature"] == 0.0
        assert payload["max_tokens"] == 512
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_stream_answer_records_exact_grounding_and_multimodal_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"speech")
    overflow = _asset(tmp_path, "overflow", Modality.AUDIO, "audio/wav", b"second")
    monkeypatch.setattr(openai_backend, "_MAX_INLINE_MODEL_BYTES", 8)

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        chunks = (
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "A greeting"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [
                    {"index": 0, "delta": {"content": " is audible."}, "finish_reason": "stop"}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                    "prompt_tokens_details": {"audio_tokens": 4},
                },
            },
        )
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            content=f"{body}data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    hit = SearchHit(
        id="memory_1",
        content="",
        score=0.9,
        created_at=NOW,
        assets=(audio,),
        modality=Modality.AUDIO,
    )
    dropped = SearchHit(
        id="memory_2",
        content="",
        score=0.8,
        created_at=NOW,
        assets=(overflow,),
        modality=Modality.AUDIO,
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
    ):
        stream = _model(_sdk_client(client)).stream_answer(
            ModelInput(text="What?"),
            (hit, dropped),
        )
        chunks = []
        while True:
            try:
                chunks.append(next(stream))
            except StopIteration as completed:
                grounded = completed.value
                break
    provider.shutdown()

    assert "".join(chunks) == "A greeting is audible."
    assert grounded == (hit,)
    assert grounded.abstention_reason is None
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["gen_ai.usage.input_tokens"] == 20
    assert attributes["gen_ai.usage.output_tokens"] == 3
    assert attributes[token_modality_attribute("input", "audio")] == 4
    assert attributes[token_modality_attribute("input", "text")] == 16
    assert cast(float, attributes[MODEL_TTFT]) >= 0
    assert cast(float, attributes[GEN_AI_TTFC]) >= 0
    assert attributes[GEN_AI_FINISH_REASONS] == ("stop",)


def test_usage_keeps_visual_audio_and_unknown_multimodal_tokens_distinct() -> None:
    exact = openai_backend._model_usage(
        {
            "usage": {
                "input_tokens": 40,
                "output_tokens": 2,
                "total_tokens": 42,
                "input_token_details": {
                    "text_tokens": 10,
                    "image_tokens": 12,
                    "video_tokens": 8,
                    "audio_tokens": 10,
                },
            }
        },
        input_modalities=ALL_MODALITIES,
        output_modalities=frozenset({Modality.TEXT}),
    )
    unresolved = openai_backend._model_usage(
        {"usage": {"prompt_tokens": 20, "total_tokens": 20}},
        input_modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        output_modalities=frozenset(),
    )

    assert exact is not None
    assert exact.input_by_modality == {"text": 10, "image": 12, "video": 8, "audio": 10}
    assert unresolved is not None
    assert unresolved.input_by_modality == {"unattributed": 20}


def test_partial_provider_usage_is_not_reported_as_complete_token_cost() -> None:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with (
        operation_span(tracer, "operation", attributes={}),
        model_span(tracer, "model", attributes={}),
    ):
        openai_backend._record_openai_usage(
            {"usage": {"input_tokens": 5}},
            input_modalities=frozenset({Modality.TEXT}),
            output_modalities=frozenset({Modality.TEXT}),
        )
    provider.shutdown()

    spans = {span.name: span for span in exporter.get_finished_spans()}
    for name in ("model", "operation"):
        attributes = spans[name].attributes
        assert attributes is not None
        assert attributes[TOKEN_COMPLETE] is False
        assert attributes[TOKEN_REPORTED_REQUEST_COUNT] == 0
        assert attributes["gen_ai.usage.input_tokens"] == 5
        assert attributes[token_modality_attribute("input", "text")] == 5
        assert TOKEN_TOTAL not in attributes


def test_answer_rejects_an_oversized_grounding_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit = SearchHit(
        id="memory_1",
        content="large evidence",
        score=0.9,
        created_at=NOW,
    )
    monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_GROUNDED_TEXT_BYTES", 1)
    with httpx.Client() as client:
        model = _model(_sdk_client(client))
        with pytest.raises(ModelError, match="4 MiB"):
            model.answer("What happened?", (hit,))


def test_multimodal_answer_budgets_the_final_text_parts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, "shared", Modality.IMAGE, "image/png", b"image")
    question = ModelInput(text="What?", assets=(image,))
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content="evidence",
            score=0.9,
            created_at=NOW,
            assets=(image,),
            modality=Modality.IMAGE,
        )
        for index in range(10)
    )
    text_parts = openai_backend._answer_text_parts(question, hits)
    actual_bytes = sum(len(part.encode()) for part in text_parts)
    monkeypatch.setattr(
        "mindbridge.models.openai_sdk._MAX_GROUNDED_TEXT_BYTES",
        actual_bytes - 1,
    )

    def unexpected_media_read(_asset: AssetRef) -> str:
        pytest.fail("oversized evidence read media before rejecting the request")

    monkeypatch.setattr(openai_backend, "_asset_data", unexpected_media_read)

    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("oversized evidence reached the model endpoint")

    with (
        httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client,
        pytest.raises(ModelError, match="4 MiB"),
    ):
        _model(_sdk_client(client)).answer(question, hits)


def test_answer_uses_standard_inline_audio_part(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"speech")

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        content = payload["messages"][1]["content"]
        assert content[-1] == {
            "type": "input_audio",
            "input_audio": {"data": "c3BlZWNo", "format": "wav"},
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "A greeting is audible."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(
        id="memory_1",
        content="",
        score=0.9,
        created_at=NOW,
        assets=(audio,),
        modality=Modality.AUDIO,
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer = _model(_sdk_client(client)).answer(ModelInput(text="What?"), (hit,))

    assert answer.answer == "A greeting is audible."


def test_answer_sends_shared_media_once_and_bounds_inline_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, "shared", Modality.IMAGE, "image/png", b"image")
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content="shared image",
            score=0.9,
            created_at=NOW,
            assets=(image,),
            modality=Modality.IMAGE,
        )
        for index in range(2)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content].count("image_url") == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        assert model.answer(ModelInput(text="What?", assets=(image,)), hits).answer == "grounded"
        monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 1)
        with pytest.raises(ModelError, match="64 MiB"):
            model.answer(ModelInput(text="What?", assets=(image,)), hits)


def test_answer_keeps_ranked_media_within_budget_and_retains_overflow_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _asset(tmp_path, "first", Modality.IMAGE, "image/png", b"1234")
    second = _asset(tmp_path, "second", Modality.IMAGE, "image/png", b"5678")
    hits = (
        SearchHit(
            id="memory_1",
            content="first evidence",
            score=0.9,
            created_at=NOW,
            assets=(first,),
            modality=Modality.IMAGE,
        ),
        SearchHit(
            id="memory_2",
            content="second evidence",
            score=0.8,
            created_at=NOW,
            assets=(second,),
            modality=Modality.IMAGE,
        ),
    )
    monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 8)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content] == [
            "text",
            "text",
            "image_url",
            "text",
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), hits)

    assert result.hits[0].assets == (first,)
    assert result.hits[1].assets == ()
    assert result.hits[1].modality is Modality.TEXT


def test_answer_elides_one_oversized_media_item_and_keeps_its_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oversized = _asset(tmp_path, "oversized", Modality.IMAGE, "image/png", b"123456")
    small = _asset(tmp_path, "small", Modality.IMAGE, "image/png", b"123")
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content=f"evidence {index}",
            score=1.0 - index / 10,
            created_at=NOW,
            assets=(asset,),
            modality=Modality.IMAGE,
        )
        for index, asset in enumerate((oversized, small))
    )
    monkeypatch.setattr(openai_backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 28)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content].count("image_url") == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        result = model.answer(ModelInput(text="What?"), hits)
        with pytest.raises(ModelError, match="20 MiB") as failure:
            model.answer(ModelInput(text="What?", assets=(oversized,)), hits)

    assert result.hits[0].assets == ()
    assert result.hits[0].content == "evidence 0"
    assert result.hits[0].modality is Modality.TEXT
    assert result.hits[1].assets == (small,)
    assert failure.value.reason == "payload_too_large"


def test_answer_distinguishes_elided_evidence_from_empty_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oversized = _asset(tmp_path, "oversized", Modality.IMAGE, "image/png", b"123456")
    hit = SearchHit(
        id="memory_1",
        content="",
        score=0.9,
        created_at=NOW,
        assets=(oversized,),
        modality=Modality.IMAGE,
    )
    monkeypatch.setattr(openai_backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 28)

    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        pytest.fail("elided evidence must not call the generation provider")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        model = _model(_sdk_client(client))
        result = model.answer(ModelInput(text="What?"), (hit,))
        stream = model.stream_answer(ModelInput(text="What?"), (hit,))
        assert next(stream) == UNKNOWN_ANSWER
        with pytest.raises(StopIteration) as completed:
            next(stream)

    assert result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE
    assert result.hits == ()
    assert completed.value.value.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


def test_answer_keeps_small_sibling_of_one_oversized_media_item(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    oversized = _asset(tmp_path, "oversized", Modality.IMAGE, "image/png", b"123456")
    small = _asset(tmp_path, "small", Modality.IMAGE, "image/png", b"123")
    hit = SearchHit(
        id="memory_1",
        content="",
        score=0.9,
        created_at=NOW,
        assets=(oversized, small),
        modality=Modality.IMAGE,
    )
    monkeypatch.setattr(openai_backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 28)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content].count("image_url") == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
    ):
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), (hit,))
    provider.shutdown()

    assert result.hits[0].assets == (small,)
    assert result.hits[0].modality is Modality.IMAGE
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[GROUNDING_MEDIA_ELIDED] == 1


def test_answer_caps_grounding_videos_without_dropping_their_text(tmp_path: Path) -> None:
    videos = tuple(
        _asset(tmp_path, f"video_{index}", Modality.VIDEO, "video/mp4", str(index).encode())
        for index in range(3)
    )
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content=f"transcript {index}",
            score=1.0 - index / 10,
            created_at=NOW,
            assets=(video,),
            modality=Modality.VIDEO,
        )
        for index, video in enumerate(videos)
    )

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content].count("video_url") == 2
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client), generation_video_limit=2).answer(
            ModelInput(text="What?"), hits
        )

    assert [len(hit.assets) for hit in result.hits] == [1, 1, 0]
    assert result.hits[2].content == "transcript 2"
    assert result.hits[2].modality is Modality.TEXT


def test_inline_media_budget_bounds_encoded_request_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = _asset(tmp_path, "wire", Modality.IMAGE, "image/png", b"123456789")
    encoded = base64.b64encode(b"123456789").decode("ascii")
    hit = SearchHit(id="memory_1", content="a note", score=0.9, created_at=NOW)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
        assert url.split(",", 1)[1] == encoded
        assert len(encoded) == 12
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        question = ModelInput(text="What?", assets=(image,))
        # 11 admits the nine bytes on disk but not the twelve the request carries.
        monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 11)
        with pytest.raises(ModelError, match="encoded inline model media exceeds 64 MiB"):
            model.answer(question, (hit,))
        monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", len(encoded))
        assert model.answer(question, (hit,)).answer == "grounded"


def test_ranked_media_budget_counts_encoded_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _asset(tmp_path, "first", Modality.IMAGE, "image/png", b"123456")
    second = _asset(tmp_path, "second", Modality.IMAGE, "image/png", b"654321")
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content=f"evidence {index}",
            score=1.0 - index / 10,
            created_at=NOW,
            assets=(asset,),
            modality=Modality.IMAGE,
        )
        for index, asset in enumerate((first, second))
    )
    # Twelve holds both files on disk and only the first once base64 doubles them to eight each.
    monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 12)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in content].count("image_url") == 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), hits)

    assert result.hits[0].assets == (first,)
    assert result.hits[1].assets == ()


def test_grounding_span_counts_evidence_the_budget_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hits = tuple(
        SearchHit(
            id=f"memory_{index}",
            content=content,
            score=1.0 - index / 10,
            created_at=NOW,
            assets=(_asset(tmp_path, f"asset_{index}", Modality.IMAGE, "image/png", b"1234"),),
            modality=Modality.IMAGE,
        )
        for index, content in enumerate(("kept evidence", "elided evidence", ""))
    )
    monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 8)

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "grounded"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
    ):
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), hits)
    provider.shutdown()

    assert len(result.hits) == 2
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[GROUNDING_MEDIA_ELIDED] == 2
    assert attributes[GROUNDING_HITS_DROPPED] == 1


def test_truncated_answer_is_distinguishable_from_a_transport_failure() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "The cat is on the"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    hit = SearchHit(id="memory_1", content="the cat sleeps", score=0.9, created_at=NOW)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
        pytest.raises(ModelOutputTruncatedError, match="generation_max_tokens") as raised,
    ):
        _model(_sdk_client(client), generation_max_tokens=4).answer(
            ModelInput(text="Where is the cat?"), (hit,)
        )
    provider.shutdown()

    assert raised.value.code == "model_output_truncated"
    assert isinstance(raised.value, ModelError)
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[GEN_AI_FINISH_REASONS] == ("length",)


def test_truncated_stream_reports_the_same_error_as_the_buffered_path() -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        chunks = (
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [
                    {"index": 0, "delta": {"content": "The cat is on the"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            },
        )
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            content=f"{body}data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    hit = SearchHit(id="memory_1", content="the cat sleeps", score=0.9, created_at=NOW)
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
        pytest.raises(ModelOutputTruncatedError, match="generation_max_tokens") as raised,
    ):
        model = _model(_sdk_client(client), generation_max_tokens=4)
        tuple(model.stream_answer(ModelInput(text="Where is the cat?"), (hit,)))
    provider.shutdown()

    assert raised.value.code == "model_output_truncated"
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[GEN_AI_FINISH_REASONS] == ("length",)


def test_transcription_uses_its_own_endpoint_key_and_multipart_file(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"speech")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://sdk.example.test/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer sdk-secret"
        assert request.headers["Content-Type"].startswith("multipart/form-data;")
        body = request.read()
        assert b"speech-model" in body
        assert b"audio.wav" in body
        assert b"speech" in body
        return httpx.Response(200, json={"text": "hello there"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).transcribe((audio,))

    assert result == ("hello there",)


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}],
        [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0, 0]}],
    ],
)
def test_embedding_rejects_invalid_response_indices_or_dimensions(
    data: list[dict[str, object]],
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": data})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ModelError, match="invalid"),
    ):
        _model(_sdk_client(client)).embed((ModelInput(text="first"),) * 2)


@pytest.mark.parametrize("network_error", [False, True])
def test_provider_failures_do_not_leak_key_or_body(network_error: bool) -> None:
    secret = "do-not-leak-key"
    private_body = "private memory body"

    def respond(request: httpx.Request) -> httpx.Response:
        if network_error:
            raise httpx.ConnectError(f"{secret}: {private_body}", request=request)
        return httpx.Response(400, text=f"{secret}: {private_body}")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ModelError) as failure,
    ):
        _model(_sdk_client(client, api_key=secret)).embed((private_body,))

    assert secret not in str(failure.value)
    assert private_body not in str(failure.value)


def _raise_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _raise_connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route", request=request)


def _search_hit() -> SearchHit:
    return SearchHit(id="memory_1", content="The toolbox is blue.", score=0.9, created_at=NOW)


@pytest.mark.parametrize(
    ("respond", "expected_reason", "expected_retryable"),
    [
        (
            lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}),
            "auth_failed",
            False,
        ),
        (
            lambda request: httpx.Response(429, json={"error": {"message": "slow down"}}),
            "rate_limited",
            True,
        ),
        # The SDK raises `RateLimitError` for every 429, so exhausted billing is told from a
        # transient burst by the provider's own error code and never invites a retry.
        (
            lambda request: httpx.Response(
                429,
                json={
                    "error": {
                        "message": "You exceeded your current quota",
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                    }
                },
            ),
            "quota_exhausted",
            False,
        ),
        (
            lambda request: httpx.Response(400, json={"error": {"message": "bad input"}}),
            "request_rejected",
            False,
        ),
        (_raise_timeout, "timeout", True),
        (_raise_connect_error, "connection_failed", True),
        (lambda request: httpx.Response(500, json={"error": {"message": "upstream"}}), None, False),
    ],
)
def test_provider_failures_keep_their_cause_and_the_sdk_classification(
    respond: object,
    expected_reason: str | None,
    expected_retryable: bool,
) -> None:
    with (
        httpx.Client(transport=httpx.MockTransport(cast(Any, respond))) as client,
        pytest.raises(ModelError) as failure,
    ):
        _model(_sdk_client(client)).embed((ModelInput(text="remember this"),))

    assert failure.value.reason == expected_reason
    assert failure.value.retryable is expected_retryable
    assert failure.value.stage == "embed"
    # The original provider exception survives as the cause instead of being erased by `from None`.
    assert isinstance(failure.value.__cause__, openai.OpenAIError)


def test_generation_and_transcription_name_their_own_stage(tmp_path: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    audio = _asset(tmp_path, "clip", Modality.AUDIO, "audio/wav", b"wav")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        with pytest.raises(ModelError) as generation:
            model.answer("what happened?", (_search_hit(),))
        with pytest.raises(ModelError) as transcription:
            model.transcribe((audio,))

    assert generation.value.stage == "generate"
    assert transcription.value.stage == "transcribe"
    assert {generation.value.reason, transcription.value.reason} == {"rate_limited"}


def test_backend_and_modality_failures_are_permanent(tmp_path: Path) -> None:
    model = OpenAIModels()
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")

    with pytest.raises(ModelError) as missing_client:
        model.embed((ModelInput(text="remember this"),))
    with pytest.raises(ModelError) as unsupported:
        model.embed((ModelInput(assets=(image,)),))

    assert missing_client.value.reason == "backend_not_configured"
    assert missing_client.value.retryable is False
    assert unsupported.value.reason == "unsupported_modality"
    assert unsupported.value.retryable is False


def test_missing_client_and_unsupported_modality_fail_before_http(tmp_path: Path) -> None:
    model = OpenAIModels()
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    assert model.embed(()) == ()
    with pytest.raises(ModelError, match="embedding SDK client is not configured"):
        model.embed((ModelInput(text="remember this"),))
    with pytest.raises(ModelError, match="does not support: image"):
        model.embed((ModelInput(assets=(image,)),))


def test_generation_controls_reject_invalid_values() -> None:
    with pytest.raises(ValidationError, match="generation_max_tokens"):
        OpenAIModels(generation_max_tokens=0)
    with pytest.raises(ValidationError, match="generation_extra_body"):
        OpenAIModels(generation_extra_body={"": False})
    with pytest.raises(ValidationError, match="generation_video_limit"):
        OpenAIModels(generation_video_limit=0)


def test_adapter_does_not_close_the_caller_owned_sdk_client() -> None:
    with httpx.Client() as transport:
        client = _sdk_client(transport)
        _model(client).close()
        assert client.is_closed() is False


def _model(
    client: OpenAI,
    *,
    generation_seed: int | None = None,
    generation_temperature: float | None = None,
    generation_max_tokens: int | None = None,
    generation_video_limit: int | None = 8,
    generation_extra_body: dict[str, object] | None = None,
) -> OpenAIModels:
    return OpenAIModels(
        client,
        embedding_model="embed-model",
        embedding_space="shared-space-v1",
        generation_model="answer-model",
        transcription_model="speech-model",
        embedding_dimension=2,
        embedding_capabilities=ALL_MODALITIES,
        generation_capabilities=ALL_MODALITIES,
        transcription_capabilities=frozenset({Modality.AUDIO, Modality.VIDEO}),
        generation_seed=generation_seed,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
        generation_video_limit=generation_video_limit,
        generation_extra_body=generation_extra_body,
    )


def _sdk_client(client: httpx.Client, *, api_key: str = "sdk-secret") -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://sdk.example.test/v1",
        http_client=client,
        max_retries=0,
    )


def _asset(
    directory: Path,
    asset_id: str,
    modality: Modality,
    media_type: str,
    data: bytes,
) -> AssetRef:
    path = directory / f"{asset_id}.bin"
    path.write_bytes(data)
    return AssetRef(
        id=asset_id,
        modality=modality,
        media_type=media_type,
        size_bytes=len(data),
        sha256="a" * 64,
        name=f"{asset_id}.wav" if modality is Modality.AUDIO else f"{asset_id}.bin",
        path=path,
    )
