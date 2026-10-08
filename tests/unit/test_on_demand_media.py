"""Bounded native media diagnosis through the public OpenAI adapter."""

import json
from dataclasses import replace
from pathlib import Path

import httpx2 as httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_openai_sdk import _asset, _sdk_client, _search_hit

from mindbridge import OpenAIModels, ValidationError
from mindbridge._telemetry import MODEL_REQUEST_COUNT, TOKEN_TOTAL, model_span, operation_span
from mindbridge.types import EvidenceBasis, MemoryContext, MemoryKind, Modality


@pytest.mark.parametrize(
    "diagnosis,expected",
    [
        ('{"requests":[]}', 0),
        ('{"requests":[{"media_index":1,"reason":"detail_missing"}]}', 1),
        ('{"requests":[{"media_index":1,"reason":"conflicting_evidence"}]}', 1),
        ('{"requests":[{"media_index":99,"reason":"detail_missing"}]}', 0),
        ('{"requests":[{"media_index":true,"reason":"detail_missing"}]}', 0),
        ('{"requests":[{"media_index":1,"reason":"identify_this_person"}]}', 0),
        (
            '{"requests":[{"media_index":0,"reason":"detail_missing"},{"media_index":1,"reason":"detail_missing"}]}',
            0,
        ),
        ("not json", 0),
    ],
)
def test_diagnosis_is_bounded_and_keeps_text_omissions_and_request_cost(
    tmp_path: Path,
    diagnosis: str,
    expected: int,
) -> None:
    assets = tuple(
        _asset(tmp_path, f"picture-{index}", Modality.IMAGE, "image/png", f"png-{index}".encode())
        for index in range(2)
    )
    hit = replace(
        _search_hit(),
        content="Two contrasting descriptions; the label is absent.",
        assets=assets,
        modality=Modality.IMAGE,
    )
    sent: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            assert isinstance(payload["messages"][1]["content"], str)
            assert payload["max_tokens"] <= 512
            assert "data:image" not in request.content.decode()
        else:
            content = payload["messages"][1]["content"]
            parts = content if isinstance(content, list) else []
            images = [part for part in parts if part["type"] == "image_url"]
            assert len(images) == expected
            assert "media_omitted" in str(content)
            assert hit.content in str(content)
            if images:
                assert images[0]["image_url"]["url"].endswith("cG5nLTE=")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": diagnosis if len(sent) == 1 else "grounded answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("media-test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_media_policy="on_demand",
            generation_media_max_items=1,
        )
        with (
            operation_span(tracer, "operation", attributes={}),
            model_span(tracer, "answer", attributes={}),
        ):
            model.answer("What does the missing label say?", (hit,))
    attributes = {span.name: span.attributes for span in exporter.get_finished_spans()}
    provider.shutdown()
    assert len(sent) == 2
    assert attributes["operation"] is not None and attributes["operation"][MODEL_REQUEST_COUNT] == 2
    assert attributes["operation"][TOKEN_TOTAL] == 30
    assert attributes["mindbridge.model.media_diagnosis"] is not None
    assert (
        attributes["mindbridge.model.media_diagnosis"][
            "mindbridge.grounding.media_diagnosis.selected"
        ]
        == expected
    )


def test_text_only_answers_skip_diagnosis() -> None:
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        OpenAIModels(_sdk_client(client), generation_media_policy="on_demand").answer(
            "question", (_search_hit(),)
        )
    assert len(sent) == 1


def test_provider_rejection_does_not_diagnose_again_or_lose_original_omissions(
    tmp_path: Path,
) -> None:
    assets = tuple(
        _asset(tmp_path, f"picture-{i}", Modality.IMAGE, "image/png", b"image") for i in range(2)
    )
    original = _search_hit()
    hit = replace(
        original,
        assets=assets,
        modality=Modality.IMAGE,
        context=MemoryContext(
            kind=MemoryKind.OBSERVATION,
            basis=EvidenceBasis.OBSERVATION,
            confidence=1.0,
            valid_from=None,
            valid_until=None,
            recorded_at=original.created_at,
        ),
    )
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            content = '{"requests":[{"media_index":0,"reason":"detail_missing"}]}'
        elif len(sent) == 2:
            return httpx.Response(
                400, json={"error": {"message": "image input may contain inappropriate content"}}
            )
        else:
            assert "response_format" not in payload
            evidence = json.loads(payload["messages"][1]["content"])["hits"][0]
            assert evidence["media_omitted"] == {"image": 2}
            assert "time order" in payload["messages"][0]["content"]
            content = "answer"
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
            generation_media_policy="on_demand",
        )
        model.answer("question", (hit,), exhaustive=True)
    assert len(sent) == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generation_media_policy": "invalid"},
        {"generation_media_max_items": True},
        {"generation_media_max_items": 0},
        {"generation_media_max_items": 9},
    ],
)
def test_media_controls_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OpenAIModels(**kwargs)  # type: ignore[arg-type]
