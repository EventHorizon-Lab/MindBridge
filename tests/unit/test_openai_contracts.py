"""Provider contracts: reject bad requests and preserve safe failure evidence."""

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx2 as httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_openai_sdk import _asset, _sdk_client, _search_hit

import mindbridge.recipes as recipes
from mindbridge import (
    FormationInput,
    Memory,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ModelError,
    ModelInput,
    ObservationContext,
    OpenAIModels,
    ValidationError,
)
from mindbridge._telemetry import model_span
from mindbridge.api.errors import exception_response
from mindbridge.recall import parse_recall_plan

NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)


def _reply(content: str | None, *, refusal: str | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "provider-snapshot",
            "choices": [
                {
                    "index": 0,
                    "message": {"content": content, "refusal": refusal},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        },
        headers={"x-request-id": "req-success"},
    )


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5-mini-2025-08-07", "o3", "gpt-6-astra"])
def test_reasoning_plans_omit_sampling_and_use_completion_budget(model: str) -> None:
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply('{"shape":"point","steps":[{"op":"similar"}]}')

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(
            _sdk_client(client), generation_model=model, generation_max_tokens=800
        )
        adapter.plan_recall(
            "Where is the toolbox?", reference_at=NOW, corpus_digest="1 text record"
        )
    assert "temperature" not in sent[0]
    assert "seed" not in sent[0]
    assert "max_tokens" not in sent[0]
    assert sent[0]["max_completion_tokens"] == 800


@pytest.mark.parametrize("control", [{"generation_temperature": 0.0}, {"generation_seed": 0}])
def test_unsupported_explicit_reasoning_sampler_fails_before_http(control: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="not support"):
        OpenAIModels(generation_model="gpt-5-mini", **control)


@pytest.mark.parametrize(
    "model", ["gpt-5.1", "gpt-5.1-2025-11-13", "gpt-5.2", "gpt-5.2-2025-12-11"]
)
def test_default_non_reasoning_models_preserve_explicit_temperature(model: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["temperature"] == 0.1
        return _reply("blue")

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIModels(
            _sdk_client(transport), generation_model=model, generation_temperature=0.1
        )
        assert adapter.answer("What color is the toolbox?", (_search_hit(),)).answer == "blue"
    with pytest.raises(ValidationError, match="not support"):
        OpenAIModels(
            generation_model=model,
            generation_temperature=0.1,
            generation_extra_body={"reasoning_effort": "medium"},
        )


def test_reasoning_vision_does_not_inject_a_sampler(tmp_path: Path) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"pixels")

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "temperature" not in body and "seed" not in body
        return _reply('{"descriptions":["Shown: blue toolbox"]}')

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(
            _sdk_client(client), generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        )
        assert adapter.describe((ModelInput(assets=(image,)),)) == ("Shown: blue toolbox",)


@pytest.mark.parametrize("stream", [False, True])
def test_safety_refusal_is_not_retried_or_exposed(stream: bool) -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if not stream:
            return _reply(None, refusal="private refusal body")
        chunks = [
            {"choices": [{"index": 0, "delta": {"refusal": "private refusal body"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_stream=stream)
        with pytest.raises(ModelError) as failure:
            adapter.plan_recall("question", reference_at=NOW, corpus_digest="1 text record")
    assert failure.value.reason == "model_refused"
    assert failure.value.retryable is False
    assert "private" not in str(failure.value)
    assert calls == 1


@pytest.mark.parametrize("stream", [False, True])
def test_answer_refusal_cannot_be_a_successful_partial_answer(stream: bool) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if not stream:
            return _reply("partial answer", refusal="private refusal")
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"refusal": "private refusal", "content": "partial"}}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client))
        with pytest.raises(ModelError) as failure:
            if stream:
                assert list(adapter.stream_answer("question", (_search_hit(),))) == []
            else:
                adapter.answer("question", (_search_hit(),))
    assert failure.value.reason == "model_refused"


@pytest.mark.parametrize(
    "status,code,reason,retryable,http_status",
    [
        (503, "server_is_overloaded", "provider_unavailable", True, 503),
        (500, "server_error", "provider_unavailable", True, 503),
        (429, "organization_spend_limit_exceeded", "quota_exhausted", False, 502),
        (429, "project_spend_limit_exceeded", "quota_exhausted", False, 502),
        (429, "organization_usage_limit_exceeded", "quota_exhausted", False, 502),
    ],
)
def test_provider_failures_publish_recovery_contract(
    status: int, code: str, reason: str, retryable: bool, http_status: int
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"code": code, "message": "private provider detail"}},
            headers={"x-request-id": "req-failure"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client))
        with pytest.raises(ModelError) as failure:
            adapter.answer("question", (_search_hit(),))
    assert failure.value.reason == reason
    assert failure.value.retryable is retryable
    response = exception_response(failure.value)
    assert response.status_code == http_status
    assert "private" not in bytes(response.body).decode()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "code,message,reason",
    [
        ("context_length_exceeded", "private provider detail", "context_length_exceeded"),
        (
            "context_length_exceeded",
            "private input image is too short; context overflow",
            "context_length_exceeded",
        ),
        (
            None,
            "private decoder prompt is longer than the maximum model length of 32",
            "context_length_exceeded",
        ),
        (
            400,
            "private decoder prompt is longer than the maximum model length of 32",
            "context_length_exceeded",
        ),
        (
            "400",
            "private decoder prompt is longer than the maximum model length of 32",
            "context_length_exceeded",
        ),
        (
            "invalid_image",
            "private image constraint; maximum context length is 32",
            "request_rejected",
        ),
        (None, "private invalid input", "request_rejected"),
    ],
)
def test_answer_context_rejection_preserves_evidence_and_does_not_retry(
    tmp_path: Path, stream: bool, code: str | int | None, message: str, reason: str
) -> None:
    calls = 0
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"pixels")
    hit = replace(_search_hit(), assets=(image,), modality=Modality.IMAGE)

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert any(part["type"] == "image_url" for part in body["messages"][1]["content"])
        return httpx.Response(
            400,
            json={"error": {"code": code, "message": message}},
            headers={"x-request-id": "req-context"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(
            _sdk_client(client), generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        )
        with pytest.raises(ModelError) as failure:
            if stream:
                list(adapter.stream_answer("question", (hit,)))
            else:
                adapter.answer("question", (hit,))
    assert calls == 1
    assert failure.value.reason == reason
    assert failure.value.stage == "generate"
    assert failure.value.retryable is False
    assert "private" not in str(failure.value)


@pytest.mark.parametrize("media", [False, True])
def test_answer_text_ceiling_matches_emitted_utf8_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, media: bool
) -> None:
    sent: list[dict[str, Any]] = []
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"pixels")
    hit = replace(
        _search_hit(),
        assets=(image,) if media else (),
        modality=Modality.IMAGE if media else Modality.TEXT,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _reply("blue")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(
            _sdk_client(client), generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        )
        assert adapter.answer("什么颜色?", (hit,)).answer == "blue"
        text_bytes = sum(
            len(part.encode("utf-8"))
            for message in sent[0]["messages"]
            for part in (
                [message["content"]]
                if isinstance(message["content"], str)
                else [part["text"] for part in message["content"] if part["type"] == "text"]
            )
        )
        monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_GROUNDED_TEXT_BYTES", text_bytes)
        assert adapter.answer("什么颜色?", (hit,)).answer == "blue"
        monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_GROUNDED_TEXT_BYTES", text_bytes - 1)
        with pytest.raises(ModelError) as failure:
            adapter.answer("什么颜色?", (hit,))
    assert len(sent) == 2
    assert failure.value.reason == "payload_too_large"


@pytest.mark.parametrize("failed", [False, True])
def test_trace_keeps_request_id_without_provider_bodies(failed: bool) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def respond(request: httpx.Request) -> httpx.Response:
        if failed:
            return httpx.Response(
                503,
                json={"error": {"message": "private body"}},
                headers={"x-request-id": "req-failure"},
            )
        return _reply("blue")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        model_span(provider.get_tracer("contract"), "model", attributes={}),
    ):
        adapter = OpenAIModels(_sdk_client(client))
        if failed:
            with pytest.raises(ModelError):
                adapter.answer("private question", (_search_hit(),))
        else:
            adapter.answer("private question", (_search_hit(),))
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert attrs["gen_ai.request.id"] == ("req-failure" if failed else "req-success")
    assert attrs["mindbridge.model.request_ids"] == (
        ("req-failure",) if failed else ("req-success",)
    )
    assert "private" not in str(attrs)
    provider.shutdown()


def test_strict_formation_uses_schema_but_still_rejects_invented_witnesses() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        format_ = body["response_format"]
        assert format_["type"] == "json_schema"
        assert format_["json_schema"]["strict"] is True
        assert format_["json_schema"]["schema"]["additionalProperties"] is False
        return _reply(
            json.dumps(
                {
                    "items": [
                        {
                            "observation_id": "observation_0",
                            "proposals": [
                                {
                                    "kind": "event",
                                    "content": "The toolbox is blue.",
                                    "confidence": 0.9,
                                    "evidence_observation_ids": ["observation_0", "invented"],
                                }
                            ],
                        }
                    ]
                }
            )
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_response_format="json_schema")
        source = FormationInput(
            "source", ModelInput(text="The toolbox is blue."), ObservationContext()
        )
        assert adapter.form((source,)) == ((),)


def test_json_schema_changes_persisted_recipes() -> None:
    plain = OpenAIModels()
    strict = OpenAIModels(generation_response_format="json_schema")
    assert strict.formation_space != plain.formation_space
    assert strict.vision_space != plain.vision_space
    assert strict.consolidation_recipe != plain.consolidation_recipe


def test_vision_recipe_uses_effective_sampling_controls() -> None:
    implicit = OpenAIModels(generation_model="compatible-model")
    explicit = OpenAIModels(
        generation_model="compatible-model", generation_temperature=0.0, generation_seed=0
    )
    assert implicit.vision_space == explicit.vision_space


def test_invalid_response_format_fails_locally() -> None:
    with pytest.raises(ValidationError, match="generation_response_format"):
        OpenAIModels(generation_response_format=cast(Any, "invalid"))


@pytest.mark.parametrize(
    "reply",
    [
        '{"operations":[{"intent":"unknown","rationale":"private memory body"}]}',
        '{"operations":[{"intent":"correct","private memory body":true}]}',
    ],
)
def test_consolidation_errors_do_not_quote_model_output(reply: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return _reply(reply)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client))
        with pytest.raises(ModelError) as failure:
            adapter.consolidate(
                (MemoryRecord(id="source", content="private memory body", created_at=NOW),),
                trigger=MemoryTrigger.CONTRADICTION,
            )
    assert "private memory body" not in str(failure.value)


@pytest.mark.parametrize("op", ["similar", "match", "window", "neighbors", "entity"])
def test_strict_recall_output_round_trips_through_semantic_parser(op: str) -> None:
    values: dict[str, dict[str, object]] = {
        "similar": {"op": "similar", "query": "toolbox", "k": 12},
        "match": {"op": "match", "terms": ["toolbox"], "any_of": True},
        "window": {"op": "window", "time": ["2026-01-01", "2026-01-02"]},
        "neighbors": {"op": "neighbors", "of": "step:0", "before": 1, "after": 1},
        "entity": {"op": "entity", "name": "Ana"},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        schema = json.loads(request.content)["response_format"]["json_schema"]["schema"]
        step_schema = schema["properties"]["steps"]["items"]
        variants = step_schema.get("anyOf", [step_schema])

        def step(operation: str) -> dict[str, object]:
            variant = next(
                variant for variant in variants if operation in variant["properties"]["op"]["enum"]
            )
            return {key: values[operation].get(key) for key in variant["required"]}

        steps = ([step("similar")] if op == "neighbors" else []) + [step(op)]
        shape = "point" if op == "similar" else "composite"
        return _reply(json.dumps({"shape": shape, "steps": steps}))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        adapter = OpenAIModels(_sdk_client(client), generation_response_format="json_schema")
        payload = adapter.plan_recall("question", reference_at=NOW, corpus_digest="10 text records")
    assert payload is not None
    plan = parse_recall_plan(payload, reference_at=NOW)
    assert plan is not None
    assert plan.steps[-1].op == op


@pytest.mark.parametrize("refused", [False, True])
@pytest.mark.parametrize("iterator_failure", [False, True])
def test_json_stream_close_failure_preserves_usage_and_primary_refusal(
    monkeypatch: pytest.MonkeyPatch, refused: bool, iterator_failure: bool
) -> None:
    class Stream:
        closed = False

        def __iter__(self) -> Iterator[object]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        delta=SimpleNamespace(
                            content=None if refused else '{"shape":"point","steps":[]}',
                            refusal="private refusal" if refused else None,
                        ),
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=4,
                    completion_tokens=3,
                    total_tokens=7,
                ),
            )
            if iterator_failure:
                raise httpx.ReadError("private iterator failure")

        def close(self) -> None:
            self.closed = True
            raise RuntimeError("private cleanup failure")

    stream = Stream()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: _reply("unused"))) as transport,
        model_span(provider.get_tracer("cleanup"), "model", attributes={}),
    ):
        client = _sdk_client(transport)
        monkeypatch.setattr(client.chat.completions, "create", lambda **_kwargs: stream)
        adapter = OpenAIModels(client, generation_stream=True)
        with pytest.raises(ModelError) as failure:
            adapter.plan_recall("question", reference_at=NOW, corpus_digest="1 text record")
    expected_reason = "connection_failed" if iterator_failure else None
    assert failure.value.reason == ("model_refused" if refused else expected_reason)
    assert "private" not in str(failure.value)
    assert stream.closed
    assert (exporter.get_finished_spans()[0].attributes or {})[
        "mindbridge.token_usage.total_tokens"
    ] == 7
    provider.shutdown()


def test_declarative_strict_protocol_reaches_real_planner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schemas = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            count = len(json.loads(request.content)["input"])
            return httpx.Response(
                200,
                json={
                    "data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]
                },
            )
        body = json.loads(request.content)
        if not body.get("stream"):
            schemas.append(body["response_format"]["json_schema"]["name"])
            return _reply('{"shape":"point","steps":[{"op":"similar","query":"toolbox","k":1}]}')
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "blue"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        ]
        return httpx.Response(
            200,
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(recipes, "_openai_client", lambda **_kwargs: _sdk_client(transport))
        with Memory.from_config(
            {
                "data_dir": tmp_path,
                "embedding": {"provider": "openai", "dimension": 2},
                "generation": {"provider": "openai", "response_format": "json_schema"},
                "settings": {"recall_planning": True},
            }
        ) as memory:
            memory.add("The toolbox is blue.")
            assert memory.ask("What color is the toolbox?").answer == "blue"
    assert schemas == ["recall_plan"]


def test_strict_protocol_is_not_downgraded_after_provider_rejection() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["response_format"]["type"] == "json_schema"
        return httpx.Response(400, json={"error": {"message": "Unsupported response format"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIModels(_sdk_client(transport), generation_response_format="json_schema")
        with pytest.raises(ModelError) as failure:
            adapter.plan_recall("question", reference_at=NOW, corpus_digest="1 text record")
    assert failure.value.reason == "request_rejected"
    assert calls == 1


def test_malformed_retry_keeps_all_request_ids() -> None:
    calls = 0
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content = (
            "not JSON"
            if calls == 1
            else '{"items":[{"observation_id":"observation_0","proposals":[]}]}'
        )
        response = _reply(content)
        response.headers["x-request-id"] = f"req-{calls}"
        return response

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        model_span(provider.get_tracer("retry"), "model", attributes={}),
    ):
        adapter = OpenAIModels(_sdk_client(transport))
        assert adapter.form(
            (FormationInput("source", ModelInput(text="blue toolbox"), ObservationContext()),)
        ) == ((),)
    attrs = exporter.get_finished_spans()[0].attributes or {}
    assert attrs["gen_ai.request.id"] == "mixed"
    assert attrs["mindbridge.model.request_ids"] == ("req-1", "req-2")
    assert attrs["mindbridge.token_usage.total_tokens"] == 14
    provider.shutdown()
