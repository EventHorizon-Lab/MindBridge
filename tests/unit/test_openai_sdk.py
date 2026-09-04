"""Focused checks for the official-SDK model adapter."""

import base64
import json
import math
import re
import sys
from array import array
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx2 as httpx
import openai
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mindbridge.memory as memory_module
import mindbridge.models.openai_sdk as openai_backend
from mindbridge._telemetry import (
    FORMATION_PROPOSALS_DROPPED,
    GEN_AI_FINISH_REASONS,
    GEN_AI_TTFC,
    GROUNDING_HITS_DROPPED,
    GROUNDING_MEDIA_ELIDED,
    MODEL_REQUEST_COUNT,
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
    ConsolidationBackend,
    EmbeddingBackend,
    EmbedTask,
    FormationBackend,
    FormationInput,
    GenerationBackend,
    ModelInput,
    TranscriptionBackend,
    VisionDescriptionBackend,
)
from mindbridge.models.openai_sdk import (
    DEFAULT_GENERATION_MODEL,
    UNKNOWN_ANSWER,
    OpenAIModels,
)
from mindbridge.types import (
    AbstentionReason,
    AssetRef,
    EvidenceBasis,
    FormationProposal,
    IdentityClaim,
    MemoryContext,
    MemoryIntent,
    MemoryKind,
    MemoryRecord,
    MemoryTrigger,
    Modality,
    ObservationContext,
    SearchHit,
    SpatialAnchor,
    SpatialContext,
)

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)
ALL_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
# Deliberately not the 16 kHz the adapter demuxes to, so a passthrough cannot look like a resample.
_SOURCE_AUDIO_RATE = 44_100
_VIDEO_MEDIA_TYPES = {".mp4": "video/mp4", ".mkv": "video/x-matroska"}


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


def test_messages_embedding_uses_vllm_shape_without_dimensions(tmp_path: Path) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "https://sdk.example.test/v1/embeddings"
        assert "input" not in payload
        assert payload == {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "remember"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,aW1hZ2U=",
                            },
                        },
                    ],
                }
            ],
            "model": "embed-model",
            "encoding_format": "float",
        }
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            embedding_client=_sdk_client(client),
            embedding_model="embed-model",
            embedding_dimension=2,
            embedding_request_format="messages",
            embedding_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        result = model.embed((ModelInput(text="remember", assets=(image,)),))

    assert model.embedding_space == "embed-model:2:messages-v1:l2-v1"
    assert result[0] == pytest.approx((0.6, 0.8))


def test_embedding_retries_a_context_length_rejection_as_ordered_stills(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL.Image")
    video = _video_asset(tmp_path, "clip", 12)
    requests: list[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        content = cast(
            list[dict[str, object]], json.loads(request.content)["messages"][0]["content"]
        )
        requests.append(content)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "The decoder prompt (length 58344) is longer than the maximum "
                            "model length of 35768."
                        )
                    }
                },
            )
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            embedding_client=_sdk_client(client),
            embedding_model="embed-model",
            embedding_dimension=2,
            embedding_request_format="messages",
            embedding_capabilities=frozenset({Modality.TEXT, Modality.VIDEO}),
        )
        result = model.embed((ModelInput(text="day one", assets=(video,)),))

    # The clip goes once as video and comes back as four ordered stills, and the text is kept.
    assert [part["type"] for part in requests[0]] == ["text", "video_url"]
    assert [part["type"] for part in requests[1]] == ["text", *["image_url"] * 4]
    assert result[0] == pytest.approx((0.6, 0.8))


def test_embedding_does_not_resample_video_for_an_unrelated_bad_request(
    tmp_path: Path,
) -> None:
    video = _asset(tmp_path, "clip", Modality.VIDEO, "video/mp4", b"video")
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(400, json={"error": {"message": "bad input"}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            embedding_client=_sdk_client(client),
            embedding_model="embed-model",
            embedding_dimension=2,
            embedding_request_format="messages",
            embedding_capabilities=frozenset({Modality.TEXT, Modality.VIDEO}),
        )
        with pytest.raises(ModelError) as failure:
            model.embed((ModelInput(text="day one", assets=(video,)),))

    assert requests == 1
    assert failure.value.reason == "request_rejected"


def test_messages_embedding_batches_chat_conversations() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"] == [
            [{"role": "user", "content": [{"type": "text", "text": "first"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "second"}]}],
        ]
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
        model = OpenAIModels(
            embedding_client=_sdk_client(client),
            embedding_dimension=2,
            embedding_request_format="messages",
        )
        first, second = model.embed(("first", "second"))

    assert first == pytest.approx((0.6, 0.8))
    assert second == pytest.approx((0.0, 1.0))


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
        memory = json.loads(content[1]["text"])["memory"]
        assert memory["created_at"] == NOW.isoformat()
        assert not {"score", "occurred_at", "occurred_end"} & memory.keys()
        assert "memory_1" not in request.content.decode()
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


def test_formation_batches_grounded_observations_and_validates_typed_output(
    tmp_path: Path,
) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"wav")

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        serialized = request.content.decode()
        assert "source_text" not in serialized
        assert "source_audio" not in serialized
        assert "microphone" not in serialized
        assert payload["response_format"] == {"type": "json_object"}
        # The per-kind field requirements are pinned by
        # `test_formation_prompt_states_every_field_the_validator_demands`, which checks them
        # against the validator instead of against one sentence's wording.
        assert "cue_modality" in payload["messages"][0]["content"]
        parts = payload["messages"][1]["content"]
        assert [part["type"] for part in parts] == ["text", "text", "input_audio"]
        assert json.loads(parts[0]["text"])["observation"]["observation_id"] == "observation_0"
        assert json.loads(parts[1]["text"])["observation"]["observation_id"] == "observation_1"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": json.dumps(
                                {
                                    "items": [
                                        {
                                            "observation_id": "observation_1",
                                            "proposals": [
                                                {
                                                    "kind": "affect",
                                                    "content": "Sam sounded pleased.",
                                                    "subject": "Sam",
                                                    "value": "pleased",
                                                    "confidence": 0.7,
                                                    "cue_modality": "audio",
                                                    "valence": 0.6,
                                                    "arousal": 0.4,
                                                }
                                            ],
                                        },
                                        {
                                            "observation_id": "observation_0",
                                            "proposals": [
                                                {
                                                    "kind": "state",
                                                    "content": "The lamp is on.",
                                                    "subject": "lamp",
                                                    "predicate": "power",
                                                    "value": "on",
                                                    "confidence": 0.9,
                                                    "valid_from": "2026-08-27T00:00:00Z",
                                                }
                                            ],
                                        },
                                    ]
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    inputs = (
        FormationInput(
            memory_id="source_text",
            content=ModelInput(text="The lamp is on."),
            context=ObservationContext(source_id="user"),
        ),
        FormationInput(
            memory_id="source_audio",
            content=ModelInput(text="Sam laughed.", assets=(audio,)),
            context=ObservationContext(source_id="microphone"),
        ),
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        state, affect = model.form(inputs)

    assert isinstance(model, FormationBackend)
    assert state[0].kind is MemoryKind.STATE
    assert state[0].valid_from == datetime(2026, 8, 27, tzinfo=timezone.utc)
    assert affect[0].kind is MemoryKind.AFFECT
    assert affect[0].cue_modality is Modality.AUDIO


def _vision_model(client: OpenAI, *, capabilities: frozenset[Modality]) -> OpenAIModels:
    return OpenAIModels(
        client,
        generation_model="caption-model",
        generation_capabilities=capabilities,
        embedding_dimension=2,
    )


def _caption_reply(*captions: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"content": json.dumps({"descriptions": list(captions)})},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930},
        },
    )


def test_description_numbers_every_visual_in_one_request_and_sends_video_as_stills(
    tmp_path: Path,
) -> None:
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    clip = _video_asset(tmp_path, "clip", 12)
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _caption_reply("a red bicycle", "a kitchen counter, then an open fridge")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(
            _sdk_client(client),
            capabilities=frozenset({Modality.IMAGE, Modality.VIDEO}),
        )
        captions = model.describe((ModelInput(assets=(picture,)), ModelInput(assets=(clip,))))

    assert isinstance(model, VisionDescriptionBackend)
    assert captions == ("a red bicycle", "a kitchen counter, then an open fridge")
    assert len(seen) == 1
    payload = seen[0]
    assert payload["model"] == "caption-model"
    assert payload["response_format"] == {"type": "json_object"}
    parts = payload["messages"][1]["content"]
    # Ordinals, not identifiers: the reply is matched to inputs by position, and the caption is
    # stored in a searchable document, so nothing naming the asset may enter the prompt.
    assert [part["type"] for part in parts] == [
        "text",
        "image_url",
        "text",
        *("image_url",) * 4,
    ]
    # A visual sent as several stills says so. Without the count, a measured endpoint answered a
    # single four-still clip with four descriptions on every attempt, and the "one string per
    # input" contract then rejected the whole reply -- billed, and no caption stored.
    assert [part["text"] for part in parts if part["type"] == "text"] == [
        "Visual 1:",
        "Visual 2, as 4 ordered stills:",
    ]
    # The video travels as the four ordered stills the generation path samples, never as the file.
    frames = [part["image_url"]["url"] for part in parts[3:]]
    assert all(url.startswith("data:image/jpeg;base64,") for url in frames)
    serialized = json.dumps(payload)
    assert "clip.mp4" not in serialized
    assert str(tmp_path) not in serialized
    assert "video/mp4" not in serialized


def test_description_reports_its_own_token_cost(tmp_path: Path) -> None:
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return _caption_reply("a whiteboard covered in equations")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with (
            operation_span(tracer, "operation", attributes={}),
            model_span(tracer, "model", attributes={}),
        ):
            model.describe((ModelInput(assets=(picture,)),))
    provider.shutdown()

    attributes = {span.name: span.attributes for span in exporter.get_finished_spans()}["model"]
    assert attributes is not None
    assert attributes[TOKEN_TOTAL] == 930
    assert attributes["gen_ai.usage.input_tokens"] == 900
    assert attributes["gen_ai.usage.output_tokens"] == 30
    assert attributes[MODEL_REQUEST_COUNT] == 1
    assert attributes[TOKEN_COMPLETE] is True


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        json.dumps(["a red bicycle"]),
        json.dumps({"captions": ["a red bicycle"]}),
        json.dumps({"descriptions": "a red bicycle"}),
        json.dumps({"descriptions": ["a red bicycle"]}),
        json.dumps({"descriptions": ["a red bicycle", "a fence", "one too many"]}),
        json.dumps({"descriptions": ["a red bicycle", "   "]}),
        json.dumps({"descriptions": ["a red bicycle", 7]}),
    ],
)
def test_description_rejects_output_it_cannot_align_with_its_inputs(
    content: str, tmp_path: Path
) -> None:
    # A caption is written into a memory's document by position. A short, long, or non-string
    # reply silently mislabels one memory with another's contents, so none of it is accepted.
    first = _asset(tmp_path, "first", Modality.IMAGE, "image/png", b"one")
    second = _asset(tmp_path, "second", Modality.IMAGE, "image/png", b"two")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with pytest.raises(ModelError) as failure:
            model.describe((ModelInput(assets=(first,)), ModelInput(assets=(second,))))

    assert failure.value.reason == "response_invalid"
    assert failure.value.stage == "describe"


def test_a_truncated_description_is_reported_as_truncation(tmp_path: Path) -> None:
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")

    def respond(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": '{"descriptions": ["a red bic'},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with pytest.raises(ModelOutputTruncatedError) as failure:
            model.describe((ModelInput(assets=(picture,)),))

    assert failure.value.stage == "describe"


def test_description_refuses_a_modality_the_endpoint_does_not_declare(tmp_path: Path) -> None:
    clip = _video_asset(tmp_path, "clip", 8)

    with httpx.Client() as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with pytest.raises(ModelError, match="does not support: video"):
            model.describe((ModelInput(assets=(clip,)),))
        assert model.describe(()) == ()


def test_description_never_falls_back_to_sending_the_video_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When the frames cannot be decoded the honest outcome is a refusal: uploading the clip
    # itself to derive one index line is a different, much costlier contract.
    clip = _video_asset(tmp_path, "clip", 8)
    monkeypatch.setattr(openai_backend, "_video_frame_urls", lambda *_args, **_kwargs: None)

    with httpx.Client(transport=httpx.MockTransport(_unreachable)) as client:
        model = _vision_model(
            _sdk_client(client),
            capabilities=frozenset({Modality.IMAGE, Modality.VIDEO}),
        )
        with pytest.raises(ModelError) as failure:
            model.describe((ModelInput(assets=(clip,)),))

    assert failure.value.reason == "asset_unavailable"
    assert failure.value.subject == "clip"


def _unreachable(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request should be sent, got {request.url}")


def _malformed_reply() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"index": 0, "message": {"content": '{"descriptions": ['}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 900, "completion_tokens": 5, "total_tokens": 905},
        },
    )


def test_one_malformed_description_reply_is_retried_once(tmp_path: Path) -> None:
    # Measured against the round's endpoint: 3 of 24 single-image calls returned HTTP 200 with
    # truncated JSON, and the same image failed twice then succeeded four times running. The SDK's
    # `max_retries` never sees this, because the transport call succeeded.
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    replies = [_malformed_reply(), _caption_reply("a red bicycle")]
    sent: list[object] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return replies[len(sent) - 1]

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        captions = model.describe((ModelInput(assets=(picture,)),))

    assert captions == ("a red bicycle",)
    assert len(sent) == 2
    # The retry repeats the identical request rather than reshaping it.
    assert sent[0] == sent[1]


def test_a_second_malformed_reply_is_not_retried_again(tmp_path: Path) -> None:
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    sent: list[object] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _malformed_reply()

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with (
            operation_span(tracer, "operation", attributes={}),
            model_span(tracer, "model", attributes={}),
            pytest.raises(ModelError) as failure,
        ):
            model.describe((ModelInput(assets=(picture,)),))
    provider.shutdown()

    assert failure.value.reason == "response_invalid"
    assert len(sent) == 2
    # Both attempts were billed, so both are metered: recording per attempt would report the
    # last one only, and recording on success only would drop the wasted request entirely.
    attributes = {span.name: span.attributes for span in exporter.get_finished_spans()}["model"]
    assert attributes is not None
    assert attributes[TOKEN_TOTAL] == 1_810
    assert attributes[MODEL_REQUEST_COUNT] == 2


def test_a_truncated_description_is_not_retried(tmp_path: Path) -> None:
    # An identical second request cannot clear an output-token limit, so paying for one is waste.
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    sent: list[object] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": '{"descriptions": ["a red bic'},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        with pytest.raises(ModelOutputTruncatedError):
            model.describe((ModelInput(assets=(picture,)),))

    assert len(sent) == 1


def test_description_asks_the_endpoint_for_a_deterministic_sample(tmp_path: Path) -> None:
    # Pinned so the sampler is not a cause of drift. This is not a reproducibility guarantee --
    # the measured endpoint returns a different completion for identical requests anyway -- which
    # is why the benchmark harness caches descriptions by asset digest instead.
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _caption_reply("a red bicycle")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _vision_model(_sdk_client(client), capabilities=frozenset({Modality.IMAGE}))
        first = model.describe((ModelInput(assets=(picture,)),))
        second = model.describe((ModelInput(assets=(picture,)),))

    assert first == second
    assert sent[0]["temperature"] == 0.0
    assert sent[0]["seed"] == 0
    # Byte-identical requests: a caption that reaches a text index must not depend on anything
    # that varies between two calls with the same input.
    assert sent[0] == sent[1]


def test_a_configured_sampler_still_wins_over_the_pinned_default(tmp_path: Path) -> None:
    picture = _asset(tmp_path, "picture", Modality.IMAGE, "image/png", b"png-bytes")
    sent: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _caption_reply("a red bicycle")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(
            _sdk_client(client),
            generation_model="caption-model",
            generation_capabilities=frozenset({Modality.IMAGE}),
            generation_temperature=0.7,
            generation_seed=99,
            embedding_dimension=2,
        )
        model.describe((ModelInput(assets=(picture,)),))

    assert sent[0]["temperature"] == 0.7
    assert sent[0]["seed"] == 99


def test_formation_space_identifies_every_generation_control() -> None:
    baseline = OpenAIModels().formation_space
    variants = {
        OpenAIModels(generation_seed=7).formation_space,
        OpenAIModels(generation_temperature=0.2).formation_space,
        OpenAIModels(generation_max_tokens=128).formation_space,
        OpenAIModels(generation_extra_body={"reasoning": {"effort": "low"}}).formation_space,
        OpenAIModels(
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        ).formation_space,
    }

    assert baseline not in variants
    assert len(variants) == 5
    assert OpenAIModels(generation_seed=7).formation_space in variants


def test_consolidation_numbers_evidence_and_resolves_cited_indices_to_ids() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert "never write a memory identifier" in payload["messages"][0]["content"]
        shown = json.loads(payload["messages"][1]["content"])
        assert shown["trigger"] == "contradiction"
        assert [item["index"] for item in shown["evidence"]] == [0, 1]
        assert [item["memory_id"] for item in shown["evidence"]] == ["raw-1", "derived-1"]
        assert shown["evidence"][1]["context"]["predicate"] == "disposition"
        return _completion(
            {
                "operations": [
                    {
                        "intent": "reinforce",
                        "targets": [1],
                        "evidence": [0],
                        "rationale": "an independent second sighting",
                    },
                    {
                        "intent": "consolidate",
                        "evidence": [0, 1],
                        # Consolidation forgetting: a retirement target the prompt allows only
                        # among this operation's own evidence indices.
                        "targets": [0],
                        "proposal": {
                            "kind": "state",
                            "content": "The lamp is on.",
                            "subject": "lamp",
                            "predicate": "power",
                            "value": "on",
                            "confidence": 0.8,
                        },
                        "rationale": "both sources agree",
                    },
                    {"intent": "forget", "targets": [0], "rationale": "no longer useful"},
                ]
            }
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        assert isinstance(model, ConsolidationBackend)
        assert model.consolidation_model == "answer-model"
        operations = model.consolidate(
            _consolidation_evidence(),
            trigger=MemoryTrigger.CONTRADICTION,
        )

    reinforce, consolidate, forget = operations
    assert reinforce.intent is MemoryIntent.REINFORCE
    assert reinforce.target_ids == ("derived-1",)
    assert reinforce.evidence_ids == ("raw-1",)
    assert reinforce.rationale == "an independent second sighting"
    assert consolidate.intent is MemoryIntent.CONSOLIDATE
    assert consolidate.evidence_ids == ("raw-1", "derived-1")
    assert consolidate.target_ids == ("raw-1",)
    assert consolidate.proposal is not None
    assert consolidate.proposal.kind is MemoryKind.STATE
    assert forget.intent is MemoryIntent.FORGET
    assert forget.target_ids == ("raw-1",)


@pytest.mark.parametrize(
    "operations",
    (
        # An index outside the numbered evidence list fails the whole response.
        [{"intent": "forget", "targets": [7]}],
        [{"intent": "forget", "targets": [-1]}],
        [{"intent": "forget", "targets": ["raw-1"]}],
        # An intent whose shape the kernel could never apply.
        [{"intent": "reinforce", "targets": [0, 1], "evidence": [0]}],
        [{"intent": "consolidate", "evidence": [0]}],
        [{"intent": "shred", "targets": [0]}],
        # Fields outside the operation vocabulary.
        [{"intent": "forget", "targets": [0], "confidence": 0.5}],
        [{"targets": [0]}],
    ),
)
def test_consolidation_refuses_an_ungrounded_or_malformed_operation(
    operations: list[dict[str, object]],
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return _completion({"operations": operations})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        with pytest.raises(ModelError) as failure:
            model.consolidate(_consolidation_evidence(), trigger=MemoryTrigger.MANUAL)

    assert failure.value.reason == "response_invalid"
    assert failure.value.stage == "consolidate"


def test_consolidation_renders_identities_and_resolves_a_naming_claim() -> None:
    """A claim names the person by an identity ID the evidence rendering actually showed.

    Evidence stays cited by index, so no memory identifier is ever written. The identity ID is
    the one identifier the model copies, and it can only copy one it was shown.
    """

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "identify names who a recognized person is" in payload["messages"][0]["content"]
        shown = json.loads(payload["messages"][1]["content"])
        assert shown["evidence"][0]["speaker_ids"] == ["identity_7"]
        assert shown["evidence"][0]["face_identity_ids"] == ["identity_7"]
        assert shown["evidence"][1]["speaker_ids"] == []
        return _completion(
            {
                "operations": [
                    {
                        "intent": "identify",
                        "claim": {
                            "identity_id": "identity_7",
                            "name": "Li",
                            "relationship": "the neighbour",
                        },
                        "evidence": [0],
                        "rationale": "the stranger introduced themselves",
                    }
                ]
            }
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        (identify,) = model.consolidate(_identity_evidence(), trigger=MemoryTrigger.EVIDENCE)

    assert identify.intent is MemoryIntent.IDENTIFY
    assert identify.evidence_ids == ("clip-1",)
    assert identify.target_ids == ()
    assert identify.claim == IdentityClaim(
        identity_id="identity_7",
        name="Li",
        relationship="the neighbour",
    )


@pytest.mark.parametrize(
    "claim",
    (
        # An identity nobody in the shown evidence has is the same miscount as a bad index.
        {"identity_id": "identity_9", "name": "Li"},
        {"identity_id": 7, "name": "Li"},
        {"name": "Li"},
        {"identity_id": "identity_7"},
        {"identity_id": "identity_7", "name": "Li", "confidence": 0.5},
        {"identity_id": "identity_7", "name": "bad\nname"},
    ),
)
def test_consolidation_refuses_a_naming_claim_the_evidence_cannot_ground(
    claim: dict[str, object],
) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return _completion(
            {"operations": [{"intent": "identify", "claim": claim, "evidence": [0]}]}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        with pytest.raises(ModelError) as failure:
            model.consolidate(_identity_evidence(), trigger=MemoryTrigger.MANUAL)

    assert failure.value.reason == "response_invalid"
    assert failure.value.stage == "consolidate"


def _identity_evidence() -> tuple[MemoryRecord, ...]:
    speech = json.dumps(
        {
            "asset_id": "a" * 64,
            "segments": [
                {
                    "start_ms": 0,
                    "end_ms": 900,
                    "text": "I am Li, I live next door",
                    "speaker_id": "identity_7",
                    "speaker_name": None,
                    "identity_score": 0.8,
                }
            ],
        },
        separators=(",", ":"),
    )
    faces = json.dumps(
        {
            "asset_id": "a" * 64,
            "identities": [{"identity_id": "identity_7", "identity_name": None}],
        },
        separators=(",", ":"),
    )
    return (
        MemoryRecord(
            id="clip-1",
            content=(
                "a stranger at the door\n\n"
                f"[speech identities:{'a' * 64}]\n{speech}\n\n"
                f"[face identities:{'a' * 64}]\n{faces}"
            ),
            created_at=NOW,
        ),
        MemoryRecord(id="parcel-1", content="the courier left a parcel", created_at=NOW),
    )


def test_consolidation_carries_declared_evidence_media_as_native_parts(tmp_path: Path) -> None:
    """A native image or audio memory with no description must not reach the slow plane empty."""
    audio = _asset(tmp_path, "audio-1", Modality.AUDIO, "audio/wav", b"RIFFwave-bytes")
    image = _asset(tmp_path, "image-1", Modality.IMAGE, "image/png", b"png-bytes")
    evidence = (
        MemoryRecord(id="raw-1", content="The lamp looks on.", created_at=NOW),
        MemoryRecord(
            id="audio-memory",
            content="",
            created_at=NOW,
            assets=(audio,),
            modality=Modality.AUDIO,
        ),
        MemoryRecord(
            id="image-memory",
            content="",
            created_at=NOW,
            assets=(image,),
            modality=Modality.IMAGE,
        ),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in parts] == [
            "text",
            "text",
            "text",
            "input_audio",
            "text",
            "image_url",
        ]
        assert json.loads(parts[0]["text"]) == {"trigger": "evidence"}
        assert json.loads(parts[2]["text"])["evidence"]["media_order"] == ["media_0"]
        assert json.loads(parts[4]["text"])["evidence"]["media_order"] == ["media_1"]
        assert parts[5]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "content": json.dumps(
                                {"operations": [{"intent": "forget", "targets": [1]}]}
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 3,
                    "total_tokens": 33,
                    "prompt_tokens_details": {
                        "text_tokens": 10,
                        "audio_tokens": 12,
                        "image_tokens": 8,
                    },
                },
            },
        )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        provider.get_tracer("test").start_as_current_span("model"),
    ):
        applied = _model(_sdk_client(client)).consolidate(
            evidence,
            trigger=MemoryTrigger.EVIDENCE,
        )
    provider.shutdown()

    assert applied[0].target_ids == ("audio-memory",)
    # The recorded input modalities are the ones the request actually carried, not text alone.
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[token_modality_attribute("input", "audio")] == 12
    assert attributes[token_modality_attribute("input", "image")] == 8


def test_consolidation_leaves_out_media_the_model_does_not_declare(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "audio-1", Modality.AUDIO, "audio/wav", b"RIFFwave-bytes")
    evidence = (
        MemoryRecord(
            id="audio-memory",
            content="Sam laughed.",
            created_at=NOW,
            assets=(audio,),
            modality=Modality.AUDIO,
        ),
    )

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert isinstance(content, str)
        assert json.loads(content)["evidence"][0]["media_order"] == []
        return _completion({"operations": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client), generation_capabilities=frozenset({Modality.TEXT}))
        assert model.consolidate(evidence, trigger=MemoryTrigger.EVIDENCE) == ()


def test_consolidation_sizes_the_request_by_emitted_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One asset cited by two evidence items is carried twice, so it is charged twice."""
    image = _asset(tmp_path, "image-1", Modality.IMAGE, "image/png", b"png-bytes" * 4)
    evidence = tuple(
        MemoryRecord(
            id=memory_id,
            content="",
            created_at=NOW,
            assets=(image,),
            modality=Modality.IMAGE,
        )
        for memory_id in ("first-memory", "second-memory")
    )

    def respond(request: httpx.Request) -> httpx.Response:
        parts = json.loads(request.content)["messages"][1]["content"]
        assert [part["type"] for part in parts].count("image_url") == 2
        return _completion({"operations": []})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        assert model.consolidate(evidence, trigger=MemoryTrigger.EVIDENCE) == ()
        # 48 encoded bytes each: one copy fits under the ceiling and two do not. Sizing by
        # asset ID reported 48 for a request that carries 96.
        monkeypatch.setattr(openai_backend, "_MAX_INLINE_MODEL_BYTES", 60)
        with pytest.raises(ModelError) as failure:
            model.consolidate(evidence, trigger=MemoryTrigger.EVIDENCE)

    assert failure.value.reason == "payload_too_large"


def test_consolidation_recipe_identifies_every_generation_control() -> None:
    baseline = OpenAIModels().consolidation_recipe
    variants = {
        OpenAIModels(generation_seed=7).consolidation_recipe,
        OpenAIModels(generation_temperature=0.2).consolidation_recipe,
        OpenAIModels(generation_max_tokens=128).consolidation_recipe,
        OpenAIModels(generation_extra_body={"reasoning": {"effort": "low"}}).consolidation_recipe,
    }

    assert baseline not in variants
    assert len(variants) == 4
    # A consolidation recipe is never mistaken for a formation space on the same model.
    assert baseline != OpenAIModels().formation_space


def test_consolidation_without_evidence_calls_no_model() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client))
        assert model.consolidate((), trigger=MemoryTrigger.IDLE) == ()
        with pytest.raises(ValidationError):
            model.consolidate(_consolidation_evidence(), trigger="idle")  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            model.consolidate(["raw-1"], trigger=MemoryTrigger.IDLE)  # type: ignore[list-item]


def _completion(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"content": json.dumps(payload)},
                    "finish_reason": "stop",
                }
            ]
        },
    )


def _consolidation_evidence() -> tuple[MemoryRecord, ...]:
    return (
        MemoryRecord(id="raw-1", content="The lamp looks on.", created_at=NOW),
        MemoryRecord(
            id="derived-1",
            content="Ana is patient",
            created_at=NOW,
            context=MemoryContext(
                kind=MemoryKind.TRAIT,
                basis=EvidenceBasis.MODEL_INFERENCE,
                confidence=0.6,
                valid_from=NOW,
                valid_until=None,
                recorded_at=NOW,
                subject="Ana",
                predicate="disposition",
                value="patient",
                evidence_ids=("raw-0",),
            ),
        ),
    )


def test_vision_space_identifies_the_prompt_as_well_as_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store caches one caption per `vision_space`, so the space must cover the whole recipe.

    The prompt decides what a caption contains and is still being iterated on, while the caption
    itself lands inside a searchable document. A space that tracked only the model would make an
    edited prompt serve captions written under the old one forever, with nothing to notice it.
    """
    baseline = OpenAIModels().vision_space
    assert baseline.startswith(f"{DEFAULT_GENERATION_MODEL}:mindbridge-vision-v1:")
    # Stable for one configuration: two identical compositions must share cached captions.
    assert OpenAIModels().vision_space == baseline
    # Distinct from `formation_space`, which digests the same knobs under a different prompt.
    assert baseline != OpenAIModels().formation_space

    variants = {
        OpenAIModels(generation_model="other-model").vision_space,
        OpenAIModels(generation_seed=7).vision_space,
        OpenAIModels(generation_temperature=0.2).vision_space,
        OpenAIModels(generation_max_tokens=128).vision_space,
        OpenAIModels(generation_extra_body={"reasoning": {"effort": "low"}}).vision_space,
        OpenAIModels(
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE})
        ).vision_space,
    }
    assert baseline not in variants
    assert len(variants) == 6

    monkeypatch.setattr(openai_backend, "_VISION_SYSTEM_PROMPT", "Describe it differently.")
    assert OpenAIModels().vision_space != baseline


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


def test_grounded_prompt_requires_the_marker_the_refusal_meter_reads() -> None:
    """The prompt must demand the same token detection reads, or nothing is ever detected."""
    prompts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        prompts.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": openai_backend._ABSTENTION_MARKER},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(id="memory_1", content="partial evidence", score=0.9, created_at=NOW)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).answer(ModelInput(text="What?"), (hit,))

    assert openai_backend._ABSTENTION_MARKER in prompts[0]
    assert result.abstained is True
    assert result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    "answer",
    [
        "[insufficient_evidence]",
        "[INSUFFICIENT_EVIDENCE]",
        "  insufficient_evidence\n",
        '"[insufficient_evidence]"',
        "根据可用记忆无法回答。[insufficient_evidence]",
        UNKNOWN_ANSWER,
        "I don't know based on the available memories",
        "**I don\u2019t know based on the available memories.**",
        "\nI don't know  based on the available memories! ",
        "I don't know based on the available memories. The hits show a cat, not its location.",
    ],
)
def test_refusals_are_reported_however_the_model_formats_them(answer: str) -> None:
    assert openai_backend._abstention_reason(answer) is AbstentionReason.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    "answer",
    [
        "The cat is on the sofa.",
        "The cat is on the sofa, though I don't know based on the available memories when it moved.",
        "I don't know why the cat moved, but the memories place it on the sofa.",
        "The evidence is insufficient to say when the cat moved, but it is on the sofa.",
        "",
    ],
)
def test_a_hedge_inside_a_real_answer_is_not_reported_as_a_refusal(answer: str) -> None:
    assert openai_backend._abstention_reason(answer) is None


@pytest.mark.parametrize(
    "answer",
    [
        "The runbook says the API returns insufficient_evidence when no memory matches.",
        "Alice logged: status=insufficient_evidence at 09:12, then retried.",
    ],
)
def test_evidence_that_quotes_the_marker_word_is_not_a_refusal(answer: str) -> None:
    """The brackets are what make the marker a machine token rather than an ordinary word.

    Matching the bare word anywhere reported a correct answer as a refusal whenever the corpus
    itself mentioned `insufficient_evidence` -- a runbook line, a logged status -- which corrupts
    the refusal meter in the opposite direction from the exact-equality check it replaced.
    """
    assert openai_backend._abstention_reason(answer) is None


def test_a_refusal_reports_prose_and_not_the_marker_as_its_answer() -> None:
    """`answer` is what a caller shows or speaks; the machine signal is `abstention_reason`."""
    hit = SearchHit(id="memory_1", content="a note", score=0.9, created_at=NOW)
    result = openai_backend._answer_result(openai_backend._ABSTENTION_MARKER, (hit,))

    assert result.answer == UNKNOWN_ANSWER
    assert openai_backend._ABSTENTION_MARKER not in result.answer
    assert result.abstained is True
    assert result.abstention_reason is AbstentionReason.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize(
    ("field", "bounds"),
    [("confidence", ("0", "1")), ("valence", ("-1", "1")), ("arousal", ("0", "1"))],
)
def test_formation_prompt_states_the_range_of_every_bounded_field(
    field: str, bounds: tuple[str, ...]
) -> None:
    """A model that guesses a 1-5 scale loses the whole add(); the prompt must state the range."""
    sentences = openai_backend._FORMATION_SYSTEM_PROMPT.replace("\n", " ").split(". ")
    stated = [
        sentence
        for sentence in sentences
        if field in sentence
        and all(
            re.search(rf"(?<![\d.\-]){re.escape(bound)}(?![\d.])", sentence) for bound in bounds
        )
    ]

    assert stated, f"the formation prompt never states that {field} is in {bounds}"


def test_answer_serializes_temporal_and_metadata_evidence() -> None:
    occurred_at = datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc)
    occurred_end = occurred_at + timedelta(minutes=5)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        hit = json.loads(content)["hits"][0]
        assert hit == {
            "content": "The red parcel arrived.",
            "memory_type": "semantic",
            "occurred_at": occurred_at.isoformat(),
            "occurred_end": occurred_end.isoformat(),
            "metadata": {"dialog": "delivery", "turn": 7},
        }
        assert "memory_1" not in request.content.decode()
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


def test_answer_instructs_the_reader_to_resolve_relative_time() -> None:
    systems: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        systems.append(json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "Two weeks ago, on 20 August 2026."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    hit = SearchHit(
        id="memory_1",
        content="grandpa visited",
        score=0.9,
        created_at=NOW,
        occurred_at=datetime(2026, 8, 20, 9, tzinfo=timezone.utc),
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        _model(_sdk_client(client)).answer("How long ago did grandpa visit?", (hit,))

    assert systems == [openai_backend._GROUNDED_SYSTEM_PROMPT]
    assert systems[0].endswith(
        "Each memory carries the time it happened (`occurred_at`, or `created_at` when the event "
        "time is unknown) and the question carries the reference time it is asked at; resolve "
        "every relative time expression against those timestamps and state the resolved date or "
        "duration explicitly."
    )


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


@pytest.mark.parametrize(
    ("frame_count", "converted", "tight_budget"),
    [(6, True, False), (8, False, False), (6, False, True)],
)
def test_answer_converts_only_videos_below_the_configured_provider_minimum(
    frame_count: int,
    converted: bool,
    tight_budget: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    video = _video_asset(tmp_path, "clip", frame_count)
    hit = SearchHit(
        id="memory_1",
        content="The colored frames are ordered.",
        score=0.9,
        created_at=NOW,
        assets=(video,),
        modality=Modality.VIDEO,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        media = content[2:]
        if converted:
            assert [part["type"] for part in media] == ["image_url"] * 4
            reds = []
            for part in media:
                encoded = part["image_url"]["url"].partition(",")[2]
                with image_module.open(BytesIO(base64.b64decode(encoded))) as image:
                    reds.append(image.convert("RGB").getpixel((0, 0))[0])
            assert reds == sorted(reds)
            assert len(set(reds)) == 4
        else:
            assert [part["type"] for part in media] == ["video_url"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "Ordered."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    original = video.path.read_bytes() if video.path is not None else b""
    if tight_budget:
        monkeypatch.setattr(
            openai_backend, "_MAX_INLINE_MODEL_BYTES", openai_backend._encoded_size(len(original))
        )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = _model(_sdk_client(client), generation_min_video_seconds=2.0)
        prepared = model._answer_request("What happened?", (hit,))
        assert prepared is not None
        assert prepared[2] == frozenset(
            {Modality.TEXT, Modality.IMAGE if converted else Modality.VIDEO}
        )
        result = model.answer("What happened?", (hit,))

    assert result.hits == (hit,)
    assert video.path is not None and video.path.read_bytes() == original


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("image_capable", [False, True])
def test_answer_retries_provider_rejected_short_hit_video_as_text(
    streaming: bool,
    image_capable: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = _asset(tmp_path, "short", Modality.VIDEO, "video/mp4", b"video")
    hit = SearchHit(
        id="memory_1",
        content="The blue toolbox is beside the door.",
        score=0.9,
        created_at=NOW,
        assets=(video,),
        modality=Modality.VIDEO,
    )
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], json.loads(request.content))
        requests.append(payload)
        content = cast(list[dict[str, object]], payload["messages"])[1]["content"]
        if len(requests) == 1:
            assert isinstance(content, list)
            assert any(part["type"] == "video_url" for part in content)
            return httpx.Response(
                400,
                json={"error": {"message": "The video file is too short."}},
            )
        assert isinstance(content, str)
        assert "The blue toolbox is beside the door." in content
        if payload.get("stream"):
            chunk = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "answer-model",
                "choices": [
                    {"index": 0, "delta": {"content": "Beside the door."}, "finish_reason": "stop"}
                ],
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
                    {
                        "index": 0,
                        "message": {"content": "Beside the door."},
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
        if not image_capable:
            monkeypatch.setattr(
                openai_backend,
                "_short_video_frame_urls",
                lambda *_args: pytest.fail("preflight ran without image capability"),
            )
        model = _model(
            _sdk_client(client),
            generation_capabilities=(
                ALL_MODALITIES if image_capable else frozenset({Modality.TEXT, Modality.VIDEO})
            ),
            generation_min_video_seconds=2.0,
        )
        if streaming:
            answer = "".join(model.stream_answer("Where is the toolbox?", (hit,)))
        else:
            result = model.answer("Where is the toolbox?", (hit,))
            answer = result.answer
            assert result.hits[0].assets == ()
            assert result.hits[0].modality is Modality.TEXT
    provider.shutdown()

    assert answer == "Beside the door."
    assert len(requests) == 2
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes[MODEL_REQUEST_COUNT] == 2
    assert attributes[GROUNDING_MEDIA_ELIDED] == 1
    assert attributes[TOKEN_COMPLETE] is False


def test_answer_does_not_elide_video_for_an_unrelated_bad_request(tmp_path: Path) -> None:
    video = _asset(tmp_path, "video", Modality.VIDEO, "video/mp4", b"video")
    hit = SearchHit(
        id="memory_1",
        content="evidence",
        score=0.9,
        created_at=NOW,
        assets=(video,),
        modality=Modality.VIDEO,
    )
    requests = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(400, json={"error": {"message": "bad input"}})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ModelError) as failure,
    ):
        _model(_sdk_client(client)).answer("What happened?", (hit,))

    assert requests == 1
    assert failure.value.reason == "request_rejected"


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


@pytest.mark.parametrize("suffix", [".mp4", ".mkv"])
def test_video_transcription_uploads_the_whole_demuxed_audio_track(
    tmp_path: Path,
    suffix: str,
) -> None:
    """A video's speech must reach the endpoint, in a container the endpoint accepts.

    `/v1/audio/transcriptions` takes flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav or webm, while
    MindBridge also ingests .mkv, .mov, .avi and .ogv, so forwarding the container would drop
    the speech of every video it does not list. Both suffixes are checked because passing only
    .mp4 would also pass if the adapter simply forwarded the file.
    """
    seconds = 2.0
    video = _video_asset(tmp_path, "clip", 8, audio_seconds=seconds, suffix=suffix)
    source = video.path
    assert source is not None
    uploaded: list[bytes] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://sdk.example.test/v1/audio/transcriptions"
        body = request.read()
        assert b'filename="clip.wav"' in body
        assert b"audio/wav" in body
        assert source.read_bytes() not in body
        uploaded.append(_wav_part(body))
        return httpx.Response(200, json={"text": "the kettle is boiling"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = _model(_sdk_client(client)).transcribe((video,))

    assert result == ("the kettle is boiling",)
    rate, channels, samples, peak = _decoded_audio(uploaded[0])
    assert (rate, channels) == (16_000, 1)
    # Measured against an independent decode of the source rather than a nominal duration, so
    # losing even the resampler's final millisecond fails here. A tolerance would not: the same
    # earlier defect that cut multi-window audio to its first 30 seconds would have passed a
    # duration check that only bounded the length from below.
    assert samples == _source_samples(source)
    assert peak > 0


def test_video_without_an_audio_track_costs_no_transcription_request(tmp_path: Path) -> None:
    video = _video_asset(tmp_path, "silent", 8)

    def respond(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("a silent video must not reach the transcription endpoint")

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert _model(_sdk_client(client)).transcribe((video,)) == ("",)


def test_transcription_declares_video_so_a_cloud_route_keeps_a_video_s_speech() -> None:
    assert OpenAIModels().transcription_capabilities == frozenset({Modality.AUDIO, Modality.VIDEO})


def test_video_transcription_fails_before_inference_without_a_demuxer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the demuxer must fail, not silently drop the speech track."""
    video = _video_asset(tmp_path, "clip", 8, audio_seconds=0.5)
    monkeypatch.setitem(sys.modules, "av", None)

    def respond(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("no request may be issued when the audio track cannot be read")

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ModelError, match="av package") as raised,
    ):
        _model(_sdk_client(client)).transcribe((video,))

    assert raised.value.reason == "unsupported_modality"


def test_gpt_transcribe_sends_context_and_names_its_recipe(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"speech")

    def respond(request: httpx.Request) -> httpx.Response:
        body = request.read()
        assert b'name="model"\r\n\r\ngpt-transcribe' in body
        assert b'name="prompt"\r\n\r\nA support call' in body
        assert body.count(b'name="keywords[]"') == 2
        assert b'name="keywords[]"\r\n\r\nMindBridge\r\n' in body
        assert b'name="keywords[]"\r\n\r\nAC-42\r\n' in body
        assert body.count(b'name="languages[]"') == 2
        assert b'name="languages[]"\r\n\r\nen\r\n' in body
        assert b'name="languages[]"\r\n\r\nzh-cn\r\n' in body
        return httpx.Response(200, json={"text": "context-aware transcript"})

    options: dict[str, Any] = {
        "transcription_model": "gpt-transcribe",
        "transcription_prompt": "A support call",
        "transcription_keywords": ("MindBridge", "AC-42"),
        "transcription_languages": ("en", "zh-cn"),
    }
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIModels(transcription_client=_sdk_client(client), **options)
        result = model.transcribe((audio,))

    assert result == ("context-aware transcript",)
    assert model.transcription_space.startswith("gpt-transcribe:asr-v1:")
    assert model.transcription_space == OpenAIModels(**options).transcription_space
    assert (
        model.transcription_space
        != OpenAIModels(
            **{**options, "transcription_keywords": ("MindBridge",)}
        ).transcription_space
    )


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


def test_adapter_controls_reject_invalid_values() -> None:
    with pytest.raises(ValidationError, match="embedding_request_format"):
        OpenAIModels(embedding_request_format=cast(Any, "unknown"))
    with pytest.raises(ValidationError, match="generation_max_tokens"):
        OpenAIModels(generation_max_tokens=0)
    with pytest.raises(ValidationError, match="generation_extra_body"):
        OpenAIModels(generation_extra_body={"": False})
    with pytest.raises(ValidationError, match="generation_min_video_seconds"):
        OpenAIModels(generation_min_video_seconds=float("nan"))
    with pytest.raises(ValidationError, match="generation_video_limit"):
        OpenAIModels(generation_video_limit=0)
    with pytest.raises(ValidationError, match="transcription_prompt"):
        OpenAIModels(transcription_prompt=" ")
    with pytest.raises(ValidationError, match="transcription_keywords"):
        OpenAIModels(transcription_keywords=("invalid\nkeyword",))
    with pytest.raises(ValidationError, match="transcription_languages"):
        OpenAIModels(transcription_languages=cast(Any, "en"))


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
    generation_min_video_seconds: float | None = None,
    generation_video_limit: int | None = 8,
    generation_extra_body: dict[str, object] | None = None,
    generation_capabilities: frozenset[Modality] = ALL_MODALITIES,
) -> OpenAIModels:
    return OpenAIModels(
        client,
        embedding_model="embed-model",
        embedding_space="shared-space-v1",
        generation_model="answer-model",
        transcription_model="speech-model",
        embedding_dimension=2,
        embedding_capabilities=ALL_MODALITIES,
        generation_capabilities=generation_capabilities,
        transcription_capabilities=frozenset({Modality.AUDIO, Modality.VIDEO}),
        generation_seed=generation_seed,
        generation_temperature=generation_temperature,
        generation_max_tokens=generation_max_tokens,
        generation_min_video_seconds=generation_min_video_seconds,
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


def _video_asset(
    directory: Path,
    asset_id: str,
    frame_count: int,
    *,
    audio_seconds: float = 0.0,
    suffix: str = ".mp4",
) -> AssetRef:
    av = pytest.importorskip("av")
    image_module = pytest.importorskip("PIL.Image")
    path = directory / f"{asset_id}{suffix}"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        audio = None
        if audio_seconds:
            audio = container.add_stream("aac", rate=_SOURCE_AUDIO_RATE)
            audio.layout = "mono"
        for index in range(frame_count):
            frame = av.VideoFrame.from_image(image_module.new("RGB", (16, 16), (index * 30, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None:
            for frame in _tone_frames(audio_seconds):
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)
    return AssetRef(
        id=asset_id,
        modality=Modality.VIDEO,
        media_type=_VIDEO_MEDIA_TYPES[suffix],
        size_bytes=path.stat().st_size,
        sha256="a" * 64,
        name=path.name,
        path=path,
    )


def _tone_frames(seconds: float) -> list[Any]:
    """Encode a 440 Hz mono tone as s16 frames, so a decode can tell it from silence."""
    av = pytest.importorskip("av")
    total = int(_SOURCE_AUDIO_RATE * seconds)
    frames = []
    for start in range(0, total, 1024):
        count = min(1024, total - start)
        samples = array(
            "h",
            (
                int(0.5 * 32767 * math.sin(2 * math.pi * 440 * index / _SOURCE_AUDIO_RATE))
                for index in range(start, start + count)
            ),
        )
        frame = av.AudioFrame(format="s16", layout="mono", samples=count)
        frame.sample_rate = _SOURCE_AUDIO_RATE
        frame.pts = start
        frame.time_base = Fraction(1, _SOURCE_AUDIO_RATE)
        frame.planes[0].update(samples.tobytes())
        frames.append(frame)
    return frames


def _source_samples(path: Path) -> int:
    """Decode a container's whole audio track and report its length at the ASR sample rate."""
    av = pytest.importorskip("av")
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        total = sum(frame.samples for frame in container.decode(stream))
        return int(total) * 16_000 // int(stream.rate)


def _wav_part(body: bytes) -> bytes:
    """Slice one WAV file out of a multipart body using its own declared RIFF length."""
    start = body.index(b"RIFF")
    return body[start : start + int.from_bytes(body[start + 4 : start + 8], "little") + 8]


def _decoded_audio(data: bytes) -> tuple[int, int, int, int]:
    """Report a WAV payload's sample rate, channel count, sample count and peak amplitude."""
    av = pytest.importorskip("av")
    with av.open(BytesIO(data)) as container:
        stream = container.streams.audio[0]
        samples = 0
        peak = 0
        for frame in container.decode(stream):
            values = array("h")
            values.frombytes(bytes(frame.planes[0])[: frame.samples * 2])
            samples += frame.samples
            peak = max(peak, max(abs(value) for value in values))
        return stream.rate, len(stream.layout.channels), samples, peak


# The fields `FormationProposal` requires per kind, as the system prompt states them. Keeping the
# table here rather than parsing the prompt makes both directions fail loudly: a validator that
# gains a requirement the prompt never mentions, and a prompt that stops mentioning one.
_FORMATION_REQUIREMENTS = {
    "event": (),
    "entity": ("subject",),
    "affect": ("subject", "value", "cue_modality"),
    "state": ("subject", "predicate", "value"),
    "relation": ("subject", "predicate", "value"),
    "trait": ("subject", "predicate", "value"),
    "response_policy": ("subject", "predicate", "value"),
}


@pytest.mark.parametrize(("kind", "extra"), sorted(_FORMATION_REQUIREMENTS.items()))
def test_formation_prompt_states_every_field_the_validator_demands(
    kind: str, extra: tuple[str, ...]
) -> None:
    # Measured against qwen3.8-flash, a prompt that omitted `entity`'s subject, did not say
    # `content` is required for every kind, and did not say values are strings produced proposals
    # the validator rejected 79% of the time. Naming exactly what is enforced took that to 0%.
    payload: dict[str, object] = {"kind": kind, "content": "a formed memory", "confidence": 0.9}
    for field in extra:
        payload[field] = "text" if field == "cue_modality" else "a value"
    if kind == "affect":
        payload["valence"] = 0.5
        payload["arousal"] = 0.2

    proposal = openai_backend._formation_proposal(payload)

    assert proposal is not None, (
        f"the prompt's stated fields for {kind} do not satisfy the validator"
    )
    # Match this kind's own row in the prompt's table, not the prompt as a whole: every field name
    # appears somewhere for some other kind, so a substring search over the whole text passes even
    # when this kind's row has lost a requirement.
    row = next(
        (
            line
            for line in openai_backend._FORMATION_SYSTEM_PROMPT.splitlines()
            if line.strip().startswith(f"{kind} ") or line.strip().startswith(f"{kind}  ")
        ),
        None,
    )
    assert row is not None, f"the prompt's kind table has no row for {kind}"
    stated = {token.strip(" ,.") for token in row.split("--", 1)[1].split()}
    assert set(extra) <= stated, f"{kind} row omits {sorted(set(extra) - stated)}"


def test_a_malformed_proposal_does_not_discard_its_valid_siblings() -> None:
    # `add` commits the source before formation runs, so raising here failed a write that had
    # already succeeded -- and the retry re-ran formation, failed again, and could never store the
    # memory at all. One badly shaped derived opinion must cost only itself.
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    content = json.dumps(
        {
            "items": [
                {
                    "observation_id": "observation_0",
                    "proposals": [
                        {"kind": "event", "content": "a real event", "confidence": 0.9},
                        {"kind": "entity", "confidence": 1.0},
                        {
                            "kind": "state",
                            "subject": "Dad",
                            "predicate": "is",
                            "value": True,
                            "content": "a boolean value",
                            "confidence": 0.9,
                        },
                    ],
                }
            ]
        }
    )
    inputs = (
        FormationInput(
            memory_id="m0",
            content=ModelInput(text="x"),
            context=ObservationContext(source_id="user"),
        ),
    )

    with provider.get_tracer("test").start_as_current_span("formation"):
        results = openai_backend._formation_results(content, inputs)

    assert [proposal.content for proposal in results[0]] == ["a real event"]
    dropped = [
        span.attributes[FORMATION_PROPOSALS_DROPPED]
        for span in exporter.get_finished_spans()
        if span.attributes is not None and FORMATION_PROPOSALS_DROPPED in span.attributes
    ]
    assert dropped == [2]


@pytest.mark.parametrize(
    ("proposal", "phrase"),
    [
        pytest.param(
            FormationProposal(
                kind=MemoryKind.AFFECT,
                content="The speaker sounded exhausted",
                subject="dad",
                value="exhausted",
                cue_modality=Modality.AUDIO,
                valence=-0.6,
                arousal=0.2,
                confidence=0.8,
            ),
            "one the observation itself carried",
            id="cue_modality",
        ),
        pytest.param(
            FormationProposal(
                kind=MemoryKind.EVENT,
                content="The call happened in the kitchen",
                spatial=SpatialContext(
                    frame_id="house", anchor=SpatialAnchor.OBSERVER, x=1.0, y=2.0
                ),
                confidence=0.7,
            ),
            "reuse that frame_id and anchor unchanged",
            id="spatial",
        ),
    ],
)
def test_formation_prompt_states_every_source_rule_the_validator_enforces(
    proposal: FormationProposal, phrase: str
) -> None:
    # These two rules are enforced in `memory.py`, not by the adapter's shape checks, and a
    # proposal that breaks one is dropped -- so the caller never learns it was formed, and the
    # prompt is the only place the model can learn the rule. Red in both directions: if the
    # prompt drops the sentence, or if the validator drops the rule.
    source = FormationInput(
        memory_id="observation_0",
        content=ModelInput(text="Dad sounded exhausted on the phone."),
        context=ObservationContext(),
    )

    assert memory_module._formation_refusal(proposal, source) is not None
    assert phrase in openai_backend._FORMATION_SYSTEM_PROMPT


def test_embedding_width_mismatch_names_both_widths() -> None:
    with pytest.raises(ModelError) as failure:
        openai_backend._normalized((0.1, 0.2, 0.3), 2)

    assert failure.value.reason == "response_invalid"
    assert "returned 3 values but the configured dimension is 2" in str(failure.value)
