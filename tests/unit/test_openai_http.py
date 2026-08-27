"""Focused checks for the dependency-light default model backend."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest

import mindbridge.models.openai_http as openai_backend
from mindbridge.config import Config
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.base import EmbedTask, ModelBackend, ModelCapabilities, ModelInput
from mindbridge.models.openai_http import UNKNOWN_ANSWER, OpenAIHTTP
from mindbridge.types import AssetRef, Modality, SearchHit

NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)
ALL_MODALITIES = frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})


def test_config_resolves_independent_endpoints_keys_and_explicit_capabilities() -> None:
    config = Config.from_environment(
        {
            "OPENAI_API_KEY": "common",
            "OPENAI_BASE_URL": "https://common.example.test/v1/v1/",
            "MINDBRIDGE_EMBEDDING_API_KEY": "embed-secret",
            "MINDBRIDGE_EMBEDDING_BASE_URL": "https://embed.example.test/api",
            "MINDBRIDGE_GENERATION_BASE_URL": "https://generate.example.test",
            "MINDBRIDGE_TRANSCRIPTION_API_KEY": "speech-secret",
            "MINDBRIDGE_TRANSCRIPTION_BASE_URL": "https://speech.example.test",
            "MINDBRIDGE_EMBEDDING_MODEL": "embed-model",
            "MINDBRIDGE_EMBEDDING_SPACE": "shared-space-v1",
            "MINDBRIDGE_GENERATION_MODEL": "answer-model",
            "MINDBRIDGE_TRANSCRIPTION_MODEL": "speech-model",
            "MINDBRIDGE_TRANSCRIPTION_SPACE": "speech-recipe-v2",
            "MINDBRIDGE_EMBEDDING_DIMENSION": "2",
            "MINDBRIDGE_TIMEOUT_SECONDS": "3.5",
            "MINDBRIDGE_MEDIA_TRANSPORT": "file",
            "MINDBRIDGE_ALLOWED_URL_HOSTS": "media.example.test, CDN.EXAMPLE.TEST. ",
            "MINDBRIDGE_EMBEDDING_MODALITIES": "omni",
            "MINDBRIDGE_GENERATION_MODALITIES": "text,image",
            "MINDBRIDGE_TRANSCRIPTION_MODALITIES": "audio,video",
        }
    )

    assert config.embedding_api_key == "embed-secret"
    assert config.generation_api_key == "common"
    assert config.transcription_api_key == "speech-secret"
    assert config.embedding_base_url == "https://embed.example.test/api/v1"
    assert config.generation_base_url == "https://generate.example.test/v1"
    assert config.transcription_base_url == "https://speech.example.test/v1"
    assert config.embedding_space == "shared-space-v1"
    assert config.transcription_space == "speech-recipe-v2"
    assert config.embedding_dimension == 2
    assert config.timeout_seconds == 3.5
    assert config.media_transport == "file"
    assert config.allowed_url_hosts == {"media.example.test", "cdn.example.test"}
    assert config.capabilities.embedding == ALL_MODALITIES
    assert config.capabilities.generation == {Modality.TEXT, Modality.IMAGE}
    assert config.capabilities.transcription == {Modality.AUDIO, Modality.VIDEO}
    assert "api_key=" not in repr(config)
    assert "embed-secret" not in repr(config)
    with pytest.raises(ValidationError):
        Config.from_environment({"MINDBRIDGE_EMBEDDING_DIMENSION": "many"})
    with pytest.raises(ValidationError, match="hostnames"):
        Config(allowed_url_hosts=frozenset({"https://media.example.test"}))
    with pytest.raises(ValidationError, match="ports"):
        Config(allowed_url_hosts=frozenset({"media.example.test:443"}))
    with pytest.raises(ValidationError, match="atomic"):
        ModelCapabilities(frozenset({Modality.OMNI}), frozenset(), frozenset())


def test_text_embedding_keeps_standard_batch_shape_and_restores_order() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url == "https://embed.example.test/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer embed-secret"
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
        model = OpenAIHTTP(_config(), client=client)
        first, second = model.embed(
            (ModelInput(text="first"), ModelInput(text="second")), EmbedTask.QUERY
        )

    assert isinstance(model, ModelBackend)
    assert model.embedding_space == "shared-space-v1"
    assert first == pytest.approx((0.6, 0.8))
    assert second == pytest.approx((0.0, 1.0))


def test_multimodal_embedding_preserves_asset_order_and_maps_url_parts(
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
            assert cast(dict[str, str], part[kind])["url"] == path.resolve().as_uri()
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenAIHTTP(_config(media_transport="file"), client=client).embed(
            (ModelInput(text="remember", assets=(image, video, audio)),),
            EmbedTask.DOCUMENT,
        )

    assert result[0] == pytest.approx((0.6, 0.8))


def test_answer_maps_native_hit_media_and_abstains_without_hits(tmp_path: Path) -> None:
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url == "https://generate.example.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer generation-secret"
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
        model = OpenAIHTTP(_config(media_transport="file"), client=client)
        assert model.answer(ModelInput(text="Unknown?"), ()).answer == UNKNOWN_ANSWER
        result = model.answer(ModelInput(text="Where is the cat?"), (hit,))

    assert result.answer == "The cat is on the sofa."
    assert result.hits == (hit,)
    assert len(requests) == 1


def test_answer_serializes_temporal_and_metadata_evidence() -> None:
    occurred_at = datetime(2026, 8, 26, 12, 30, tzinfo=timezone.utc)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        hit = json.loads(content)["hits"][0]
        assert hit == {
            "id": "memory_1",
            "content": "The red parcel arrived.",
            "occurred_at": occurred_at.isoformat(),
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
        metadata={"dialog": "delivery", "turn": 7},
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer = OpenAIHTTP(_config(), client=client).answer("When did it arrive?", (hit,))

    assert answer.answer == "It arrived on August 26."


def test_answer_rejects_an_oversized_grounding_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit = SearchHit(
        id="memory_1",
        content="large evidence",
        score=0.9,
        created_at=NOW,
    )
    monkeypatch.setattr("mindbridge.models.openai_http._MAX_GROUNDED_TEXT_BYTES", 1)
    with httpx.Client() as client:
        model = OpenAIHTTP(_config(), client=client)
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
        "mindbridge.models.openai_http._MAX_GROUNDED_TEXT_BYTES",
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
        OpenAIHTTP(_config(media_transport="data"), client=client).answer(question, hits)


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
        answer = OpenAIHTTP(_config(), client=client).answer(ModelInput(text="What?"), (hit,))

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
        model = OpenAIHTTP(_config(), client=client)
        assert model.answer(ModelInput(text="What?", assets=(image,)), hits).answer == "grounded"
        monkeypatch.setattr("mindbridge.models.openai_http._MAX_INLINE_MODEL_BYTES", 1)
        with pytest.raises(ModelError, match="64 MiB"):
            model.answer(ModelInput(text="What?", assets=(image,)), hits)


def test_transcription_uses_its_own_endpoint_key_and_multipart_file(tmp_path: Path) -> None:
    audio = _asset(tmp_path, "audio", Modality.AUDIO, "audio/wav", b"speech")

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://speech.example.test/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer transcription-secret"
        assert request.headers["Content-Type"].startswith("multipart/form-data;")
        body = request.read()
        assert b"speech-model" in body
        assert b"audio.wav" in body
        assert b"speech" in body
        return httpx.Response(200, json={"text": "hello there"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = OpenAIHTTP(_config(), client=client).transcribe((audio,))

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
        OpenAIHTTP(_config(), client=client).embed((ModelInput(text="first"),) * 2)


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
        OpenAIHTTP(_config(embedding_api_key=secret), client=client).embed((private_body,))

    assert secret not in str(failure.value)
    assert private_body not in str(failure.value)


def test_missing_key_and_unsupported_modality_fail_before_http(tmp_path: Path) -> None:
    model = OpenAIHTTP(Config.from_environment({}))
    image = _asset(tmp_path, "image", Modality.IMAGE, "image/png", b"image")
    try:
        assert model.embed(()) == ()
        with pytest.raises(ModelError, match="MINDBRIDGE_EMBEDDING_API_KEY"):
            model.embed((ModelInput(text="remember this"),))
        with pytest.raises(ModelError, match="does not support: image"):
            model.embed((ModelInput(assets=(image,)),))
    finally:
        model.close()


def _config(
    *,
    media_transport: Literal["data", "file"] = "data",
    embedding_api_key: str = "embed-secret",
) -> Config:
    return Config(
        base_url="https://common.example.test",
        embedding_api_key=embedding_api_key,
        embedding_base_url="https://embed.example.test",
        generation_api_key="generation-secret",
        generation_base_url="https://generate.example.test",
        transcription_api_key="transcription-secret",
        transcription_base_url="https://speech.example.test",
        embedding_model="embed-model",
        embedding_space="shared-space-v1",
        generation_model="answer-model",
        transcription_model="speech-model",
        embedding_dimension=2,
        timeout_seconds=5,
        media_transport=media_transport,
        capabilities=ModelCapabilities(
            embedding=ALL_MODALITIES,
            generation=ALL_MODALITIES,
            transcription=frozenset({Modality.AUDIO, Modality.VIDEO}),
        ),
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
