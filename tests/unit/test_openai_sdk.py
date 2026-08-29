"""Focused checks for the official-SDK model adapter."""

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import httpx2 as httpx
import pytest
from openai import OpenAI

import mindbridge.models.openai_sdk as openai_backend
from mindbridge.exceptions import ModelError
from mindbridge.models.base import (
    EmbeddingBackend,
    EmbedTask,
    GenerationBackend,
    ModelInput,
    TranscriptionBackend,
)
from mindbridge.models.openai_sdk import UNKNOWN_ANSWER, OpenAIModels
from mindbridge.types import AssetRef, Modality, SearchHit

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
        assert model.answer(ModelInput(text="Unknown?"), ()).answer == UNKNOWN_ANSWER
        result = model.answer(ModelInput(text="Where is the cat?"), (hit,))

    assert result.answer == "The cat is on the sofa."
    assert result.hits == (hit,)
    assert len(requests) == 1


def test_answer_serializes_temporal_and_metadata_evidence() -> None:
    occurred_at = datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc)
    occurred_end = occurred_at + timedelta(minutes=5)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        hit = json.loads(content)["hits"][0]
        assert hit == {
            "id": "memory_1",
            "content": "The red parcel arrived.",
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
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["seed"] == 17
        assert payload["temperature"] == 0.0
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
        answer = _model(_sdk_client(client), generation_seed=17, generation_temperature=0.0).answer(
            "What?", (hit,)
        )

    assert answer.answer == "Pinned."


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
    monkeypatch.setattr("mindbridge.models.openai_sdk._MAX_INLINE_MODEL_BYTES", 4)

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


def test_missing_client_and_unsupported_modality_fail_before_http(tmp_path: Path) -> None:
    model = OpenAIModels()
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    assert model.embed(()) == ()
    with pytest.raises(ModelError, match="embedding SDK client is not configured"):
        model.embed((ModelInput(text="remember this"),))
    with pytest.raises(ModelError, match="does not support: image"):
        model.embed((ModelInput(assets=(image,)),))


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
