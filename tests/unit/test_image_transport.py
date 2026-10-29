"""Opt-in image transport bounds apply consistently without changing durable sources."""

from __future__ import annotations

import base64
import hashlib
import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx2 as httpx
import pytest
from openai import OpenAI

import mindbridge.configuration as configuration
import mindbridge.models.openai_sdk as backend
from mindbridge import (
    FormationInput,
    Memory,
    MindBridgeConfig,
    Modality,
    ModelInput,
    ObservationContext,
    SearchHit,
)
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.openai_sdk import OpenAIModels
from mindbridge.types import AssetRef

if TYPE_CHECKING:
    from PIL.Image import Image


def _image(
    tmp_path: Path, *, name: str = "picture", size: tuple[int, int] = (300, 200)
) -> AssetRef:
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / f"{name}.png"
    image.frombytes("RGB", size, random.Random(7).randbytes(size[0] * size[1] * 3)).save(path)
    return AssetRef(
        id=name,
        modality=Modality.IMAGE,
        media_type="image/png",
        path=path,
        name=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _client(transport: httpx.Client) -> OpenAI:
    return OpenAI(
        api_key="local-fixture",
        base_url="https://image.example.test/v1",
        http_client=transport,
        max_retries=0,
    )


def _model(
    client: OpenAI,
    *,
    max_pixels: int | None = 4096,
    request_format: Literal["input", "messages"] = "messages",
) -> OpenAIModels:
    return OpenAIModels(
        client,
        embedding_dimension=2,
        embedding_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE}),
        embedding_request_format=request_format,
        image_max_pixels=max_pixels,
    )


def _hit(asset: AssetRef) -> SearchHit:
    return SearchHit(
        id="source-memory",
        content="original observation",
        score=0.9,
        created_at=datetime(2026, 10, 28, tzinfo=timezone.utc),
        modality=Modality.IMAGE,
        assets=(asset,),
    )


def _formation(asset: AssetRef, memory_id: str = "source") -> FormationInput:
    return FormationInput(memory_id, ModelInput(assets=(asset,)), ObservationContext())


def _reply(content: str = "observed") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"content": content}, "finish_reason": "stop"}]},
    )


def _pixels(url: str) -> int:
    image = pytest.importorskip("PIL.Image")
    with image.open(BytesIO(base64.b64decode(url.partition(",")[2]))) as decoded:
        assert decoded.format == "JPEG"
        return int(decoded.width * decoded.height)


@pytest.mark.parametrize("operation", ("embed", "describe", "answer", "form"))
@pytest.mark.parametrize("request_format", ("input", "messages"))
def test_all_image_surfaces_send_bounded_jpeg_and_keep_original_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    request_format: Literal["input", "messages"],
) -> None:
    source = _image(tmp_path)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 20_000)
    assert source.path is not None
    original = source.path.read_bytes()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if operation == "embed":
            content = (
                payload["messages"][0]["content"]
                if request_format == "messages"
                else payload["input"][0][0]["content"]
            )
        else:
            content = payload["messages"][1]["content"]
        urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        assert len(urls) == 1
        assert urls[0].startswith("data:image/jpeg;base64,")
        assert len(urls[0]) <= 20_000
        assert _pixels(urls[0]) <= 4096
        if operation == "embed":
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})
        if operation == "form":
            return _reply(
                json.dumps({"items": [{"observation_id": "observation_0", "proposals": []}]})
            )
        return _reply(
            json.dumps({"descriptions": ["one image"]}) if operation == "describe" else "observed"
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = _model(_client(transport), request_format=request_format)
        if operation == "embed":
            assert model.embed((ModelInput(assets=(source,)),)) == ((1.0, 0.0),)
        elif operation == "describe":
            assert model.describe((ModelInput(assets=(source,)),)) == ("one image",)
        elif operation == "form":
            assert model.form((_formation(source),)) == ((),)
        else:
            result = model.answer(ModelInput(text="What?", assets=(source,)), (_hit(source),))
            assert result.hits[0].assets == (source,)
    assert source.path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() == source.sha256


def test_transcoding_is_deterministic_per_source_and_does_not_reencode_small_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    large, small = _image(tmp_path), _image(tmp_path, name="small", size=(16, 16))
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 20_000)
    solo = backend._image_transport_parts((large,), 4096)
    mixed = backend._image_transport_parts((small, large, large), 4096)
    assert solo == mixed
    assert small.id not in mixed
    assert small.path is not None
    assert backend._bounded_image_url(small, 4096) is None


def test_pixel_bound_applies_even_when_original_bytes_fit(tmp_path: Path) -> None:
    source = _image(tmp_path)
    assert (
        backend._inline_item_size(source, cast(int, source.size_bytes))
        < backend._MAX_INLINE_MODEL_ITEM_BYTES
    )
    url = backend._bounded_image_url(source, 4096)
    assert url is not None and _pixels(url) <= 4096


def test_byte_limit_can_reduce_pixels_further_than_configured_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _image(tmp_path)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 2500)
    url = backend._bounded_image_url(source, 4096)
    assert url is not None and len(url) <= 2500
    assert _pixels(url) < 4096


def test_original_size_jpeg_reencoding_is_available_when_pixels_already_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _image(tmp_path)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 100_000)
    url = backend._bounded_image_url(source, 60_000)
    assert url is not None and len(url) <= 100_000
    assert _pixels(url) == 60_000


def test_same_source_is_converted_once_per_answer_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = pytest.importorskip("PIL.Image")
    source = _image(tmp_path)
    opened = image.open
    calls = 0

    def open_source(path: object) -> Image:
        nonlocal calls
        calls += 1
        return cast("Image", opened(path))

    monkeypatch.setattr(image, "open", open_source)
    with httpx.Client(transport=httpx.MockTransport(lambda _request: _reply())) as transport:
        _model(_client(transport)).answer(
            ModelInput(text="What?", assets=(source,)),
            (_hit(source), replace(_hit(source), id="another-memory")),
        )
    assert calls == 1


def test_image_transport_recipe_separates_default_explicit_and_custom_spaces() -> None:
    default = OpenAIModels(embedding_space="custom")
    first = OpenAIModels(embedding_space="custom", image_max_pixels=4096)
    second = OpenAIModels(embedding_space="custom", image_max_pixels=8192)
    assert default.embedding_space == "custom"
    assert len({default.embedding_space, first.embedding_space, second.embedding_space}) == 3
    assert len({default.vision_space, first.vision_space, second.vision_space}) == 3
    assert len({default.formation_space, first.formation_space, second.formation_space}) == 3
    assert OpenAIModels().vision_space == OpenAIModels(image_max_pixels=None).vision_space
    assert OpenAIModels().formation_space == OpenAIModels(image_max_pixels=None).formation_space


@pytest.mark.parametrize("pixels", (True, 0, -1, 64_000_001, 4096.0))
def test_image_bound_rejects_invalid_constructor_values(pixels: object) -> None:
    with pytest.raises(ValidationError, match="image_max_pixels"):
        OpenAIModels(image_max_pixels=cast(Any, pixels))


@pytest.mark.parametrize("slot", ("embedding", "generation", "vision", "formation"))
def test_config_image_bounds_reach_each_model_role(slot: str) -> None:
    payload: dict[str, object] = {"embedding": {"provider": "openai", "api_key": "EMPTY"}}
    payload[slot] = {"provider": "openai", "api_key": "EMPTY", "image_max_pixels": 4096}
    config = MindBridgeConfig.model_validate(payload)
    model = (
        configuration._build_embedding(config.embedding)
        if slot == "embedding"
        else configuration._build_generation(
            cast(configuration.OpenAIGenerationConfig, config.generation)
        )
        if slot == "generation"
        else configuration._build_formation(
            cast(configuration.OpenAIFormationConfig, config.formation)
        )
        if slot == "formation"
        else configuration._build_vision(cast(configuration.OpenAIVisionConfig, config.vision))
    )
    assert isinstance(model, OpenAIModels)
    assert model._image_max_pixels == 4096
    model.close()


def test_source_pixel_limit_is_checked_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = pytest.importorskip("PIL.Image")
    source = _image(tmp_path)

    @contextmanager
    def oversized_header(_path: object) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(width=8001, height=8001)

    monkeypatch.setattr(image, "open", oversized_header)
    with pytest.raises(ModelError, match="decode bound") as failure:
        backend._bounded_image_url(source, 4096)
    assert failure.value.reason == "payload_too_large"


@pytest.mark.parametrize("operation", ("embed", "describe", "answer", "form"))
def test_image_integrity_changes_are_never_swallowed(tmp_path: Path, operation: str) -> None:
    source = _image(tmp_path)
    assert source.path is not None
    with source.path.open("ab") as output:
        output.write(b"changed")
    with httpx.Client(
        transport=httpx.MockTransport(lambda _request: pytest.fail("changed source was sent"))
    ) as transport:
        model = _model(_client(transport))
        with pytest.raises(ModelError) as failure:
            if operation == "embed":
                model.embed((ModelInput(assets=(source,)),))
            elif operation == "describe":
                model.describe((ModelInput(assets=(source,)),))
            elif operation == "form":
                model.form((_formation(source),))
            else:
                model.answer("What?", (_hit(source),))
    assert failure.value.reason == "asset_changed"


def test_formation_bounds_repeated_transmitted_images_not_only_unique_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _image(tmp_path)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 20_000)
    url = backend._bounded_image_url(source, 4096)
    assert url is not None
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_BYTES", len(url) + 1)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("oversized request was sent")
            )
        ) as transport,
        pytest.raises(ModelError) as failure,
    ):
        _model(_client(transport)).form((_formation(source, "first"), _formation(source, "second")))
    assert failure.value.reason == "payload_too_large"


def test_an_undecodable_image_is_omitted_for_answers_and_refused_for_embedding(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path)
    assert source.path is not None
    source.path.write_bytes(b"x" * cast(int, source.size_bytes))
    with httpx.Client(transport=httpx.MockTransport(lambda _request: _reply())) as transport:
        model = _model(_client(transport))
        result = model.answer("What?", (_hit(source),))
        assert result.hits[0].assets == ()
        with pytest.raises(ModelError) as failure:
            model.embed((ModelInput(assets=(source,)),))
    assert failure.value.reason == "payload_too_large"


def test_opt_in_image_transcoding_keeps_the_durable_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _image(tmp_path)
    assert source.path is not None
    original = source.path.read_bytes()
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 20_000)
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, json={"data": [{"index": 0, "embedding": [1, 0]}]}
                )
            )
        ) as transport,
        Memory(tmp_path / "store", embedder=_model(_client(transport))) as memory,
    ):
        stored = memory.add(source.path)
        durable = memory.get(stored.id).assets[0]
        assert durable.path is not None and durable.path.read_bytes() == original
        assert durable.sha256 == source.sha256
