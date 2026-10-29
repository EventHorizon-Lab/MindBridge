"""Real PyAV sampling and byte-accounted transport preserve original source identity."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from io import BytesIO
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, cast

import httpx2 as httpx
import pytest
from openai import OpenAI
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mindbridge.models.openai_sdk as backend
from mindbridge import Memory, Modality, ModelInput, SearchHit
from mindbridge.evidence import answer_evidence_payloads
from mindbridge.exceptions import ModelError, ValidationError
from mindbridge.models.openai_sdk import OpenAIModels
from mindbridge.types import AssetRef


def _video(tmp_path: Path, *, name: str = "clip", oversized: bool = False) -> AssetRef:
    av = pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / f"{name}.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(80):
            frame = av.VideoFrame.from_image(image.new("RGB", (16, 16), (index * 3, 0, 0)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if oversized:
        with path.open("ab") as output:
            output.write(b"\0" * 40_000)
    return AssetRef(
        id=name,
        modality=Modality.VIDEO,
        media_type="video/mp4",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        name=path.name,
        path=path,
    )


def _reply(answer: str = "observed") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": {"content": answer}, "finish_reason": "stop"}]},
    )


def _client(transport: httpx.Client) -> OpenAI:
    return OpenAI(
        api_key="local-fixture",
        base_url="https://video.example.test/v1",
        http_client=transport,
        max_retries=0,
    )


def _model(
    client: OpenAI,
    *,
    embedding_request_format: Literal["input", "messages"] = "input",
    vision_video_frames: int = 4,
    generation_video_limit: int | None = 8,
) -> OpenAIModels:
    return OpenAIModels(
        client,
        embedding_dimension=2,
        embedding_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
        embedding_request_format=embedding_request_format,
        vision_video_frames=vision_video_frames,
        generation_video_limit=generation_video_limit,
    )


def _hit(video: AssetRef, *, content: str = "source transcript") -> SearchHit:
    return SearchHit(
        id="source-memory",
        content=content,
        score=0.9,
        created_at=datetime(2026, 10, 28, tzinfo=timezone.utc),
        modality=Modality.VIDEO,
        assets=(video,),
    )


@pytest.mark.parametrize("count", (2, 4, 8, 32, 64))
def test_real_video_sampler_preserves_configured_count_and_order(
    tmp_path: Path, count: int
) -> None:
    image = pytest.importorskip("PIL.Image")
    urls = backend._video_frame_urls(_video(tmp_path), count=count)
    assert urls is not None and len(urls) == count
    red = [
        image.open(BytesIO(base64.b64decode(url.partition(",")[2]))).getpixel((8, 8))[0]
        for url in urls
    ]
    assert red[-1] - red[0] > 200
    assert all(left <= right + 3 for left, right in pairwise(red))


def test_frame_count_changes_only_the_nondefault_vision_recipe() -> None:
    default = OpenAIModels()
    assert default.vision_space == OpenAIModels(vision_video_frames=4).vision_space
    assert default.vision_space != OpenAIModels(vision_video_frames=8).vision_space
    assert (
        OpenAIModels(vision_video_frames=8).vision_space
        != OpenAIModels(vision_video_frames=16).vision_space
    )
    assert default.embedding_space == OpenAIModels(vision_video_frames=16).embedding_space


@pytest.mark.parametrize("count", (True, 1, 65, 4.0))
def test_invalid_frame_count_is_rejected(count: object) -> None:
    with pytest.raises(ValidationError, match="vision_video_frames"):
        OpenAIModels(vision_video_frames=cast(Any, count))


def test_vision_sends_the_configured_real_frame_count(tmp_path: Path) -> None:
    clip = _video(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert sum(part["type"] == "image_url" for part in content) == 8
        assert "as 8 ordered stills" in content[0]["text"]
        return _reply(json.dumps({"descriptions": ["one clip"]}))

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = _model(_client(transport), vision_video_frames=8)
        assert model.describe((ModelInput(assets=(clip,)),)) == ("one clip",)


@pytest.mark.parametrize("request_format", ("input", "messages"))
def test_oversized_embedding_samples_only_that_source_and_preserves_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request_format: Literal["input", "messages"]
) -> None:
    small, large = _video(tmp_path, name="small"), _video(tmp_path, name="large", oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content)
        samples = payload["messages"] if request_format == "messages" else payload["input"]
        if request_format == "messages" and isinstance(samples[0], dict):
            samples = [samples]
        kinds = [[part["type"] for part in sample[0]["content"]] for sample in samples]
        assert kinds[0] == ["video_url"]
        if len(kinds) == 2:
            assert kinds[1] == ["image_url"] * 4
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1, 0]} for index in range(len(kinds))]},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        model = _model(_client(transport), embedding_request_format=request_format)
        model.embed((ModelInput(assets=(small,)),))
        model.embed((ModelInput(assets=(small,)), ModelInput(assets=(large,))))
    assert calls == 2
    assert large.path is not None
    assert large.path.stat().st_size == large.size_bytes
    assert hashlib.sha256(large.path.read_bytes()).hexdigest() == large.sha256


@pytest.mark.parametrize("request_format", ("input", "messages"))
@pytest.mark.parametrize("limit,aggregate,items", ((40, 10_000, 1), (100, 100, 1), (100, 250, 2)))
def test_sampled_embeddings_bound_the_actual_parts_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_format: Literal["input", "messages"],
    limit: int,
    aggregate: int,
    items: int,
) -> None:
    clip = _video(tmp_path)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", limit)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_BYTES", aggregate)
    monkeypatch.setattr(
        backend,
        "_video_frame_urls",
        lambda *_args, **_kwargs: ("data:image/jpeg;base64," + "A" * 40,) * 4,
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: pytest.fail("oversized payload was sent")
            )
        ) as transport,
        pytest.raises(ModelError) as failure,
    ):
        _model(_client(transport), embedding_request_format=request_format).embed(
            (ModelInput(assets=(clip,)),) * items
        )
    assert failure.value.reason == "payload_too_large"


def test_undecodable_oversized_embedding_never_sends_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path, oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    monkeypatch.setattr(backend, "_video_frame_urls", lambda *_args, **_kwargs: None)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: pytest.fail("oversized source was sent"))
        ) as transport,
        pytest.raises(ModelError) as failure,
    ):
        _model(_client(transport)).embed((ModelInput(assets=(clip,)),))
    assert failure.value.reason == "payload_too_large"


def test_large_video_answer_prices_frames_reuses_decode_and_keeps_source_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path, oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_BYTES", 30_000)
    decode = backend._video_frame_urls
    calls = 0

    def decoded(
        asset: AssetRef, *, below_seconds: float | None = None, count: int = 4
    ) -> tuple[str, ...] | None:
        nonlocal calls
        calls += 1
        return decode(asset, below_seconds=below_seconds, count=count)

    monkeypatch.setattr(backend, "_video_frame_urls", decoded)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert sum(part["type"] == "image_url" for part in content) == 16
        assert all(part["type"] != "video_url" for part in content)
        labels = [part["text"] for part in content if part["type"] == "text"]
        assert any("16 ordered stills" in label and "motion" in label for label in labels)
        return _reply()

    provider, exporter = TracerProvider(), InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        provider.get_tracer("test").start_as_current_span("answer"),
    ):
        model = _model(_client(transport))
        result = model.answer(ModelInput(text="What?", assets=(clip,)), (_hit(clip),))
    assert result.hits[0].assets == (clip,)
    assert calls == 1
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert attributes["mindbridge.grounding.video_sampled_assets"] == 1
    provider.shutdown()


def test_video_limit_prevents_decoding_unselected_large_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = (
        _video(tmp_path, name="first", oversized=True),
        _video(tmp_path, name="second", oversized=True),
    )
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    decode = backend._video_frame_urls
    seen: list[str] = []

    def decoded(
        asset: AssetRef, *, below_seconds: float | None = None, count: int = 4
    ) -> tuple[str, ...] | None:
        seen.append(asset.id)
        return decode(asset, below_seconds=below_seconds, count=count)

    monkeypatch.setattr(backend, "_video_frame_urls", decoded)
    with httpx.Client(transport=httpx.MockTransport(lambda _request: _reply())) as transport:
        result = _model(_client(transport), generation_video_limit=1).answer(
            "What?", (_hit(first), replace(_hit(second), id="second-memory"))
        )
    assert seen == [first.id]
    assert result.hits[0].assets == (first,)
    assert result.hits[1].assets == ()


def test_undecodable_large_video_keeps_the_existing_text_only_omission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path, oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    monkeypatch.setattr(backend, "_video_frame_urls", lambda *_args, **_kwargs: None)

    def respond(request: httpx.Request) -> httpx.Response:
        content = json.loads(request.content)["messages"][1]["content"]
        assert isinstance(content, str)
        assert json.loads(content)["hits"][0]["media_omitted"] == {"video": 1}
        return _reply()

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        result = _model(_client(transport)).answer("What?", (_hit(clip),))
    assert result.hits[0].assets == ()


def test_short_video_sampling_label_still_respects_the_total_text_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path)
    question, hits = ModelInput(text="What?"), (_hit(clip),)
    texts = backend._answer_text_parts(question, hits, answer_evidence_payloads(hits))
    bare_label_bytes = len(backend._GROUNDED_SYSTEM_PROMPT.encode()) + sum(
        len(text.encode()) for text in (*texts, "attachment 1")
    )
    monkeypatch.setattr(backend, "_MAX_GROUNDED_TEXT_BYTES", bare_label_bytes)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: pytest.fail("oversized text was sent"))
        ) as transport,
        pytest.raises(ModelError, match="4 MiB"),
    ):
        OpenAIModels(
            _client(transport),
            generation_capabilities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            generation_min_video_seconds=11.0,
        ).answer(question, hits)


def test_large_video_source_integrity_is_checked_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path, oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    assert clip.path is not None
    with clip.path.open("ab") as output:
        output.write(b"changed")
    monkeypatch.setattr(
        backend,
        "_video_frame_urls",
        lambda *_args, **_kwargs: pytest.fail("changed source was decoded"),
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda _request: pytest.fail("changed source was sent"))
        ) as transport,
        pytest.raises(ModelError) as failure,
    ):
        _model(_client(transport)).answer("What?", (_hit(clip),))
    assert failure.value.reason == "asset_changed"


def test_oversized_video_ingestion_keeps_the_durable_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = _video(tmp_path, oversized=True)
    monkeypatch.setattr(backend, "_MAX_INLINE_MODEL_ITEM_BYTES", 10_000)
    assert clip.path is not None
    original = clip.path.read_bytes()

    def respond(request: httpx.Request) -> httpx.Response:
        samples = json.loads(request.content)["input"]
        assert [part["type"] for part in samples[0][0]["content"]] == ["image_url"] * 4
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport,
        Memory(tmp_path / "store", embedder=_model(_client(transport))) as memory,
    ):
        stored = memory.add(clip.path)
        durable = memory.get(stored.id).assets[0]
        assert durable.path is not None and durable.path.read_bytes() == original
        assert durable.size_bytes == len(original)
