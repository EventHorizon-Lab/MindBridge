"""Contract tests for the bundled OpenAI-compatible adapters."""

import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI

from mindbridge.core import (
    EmbeddingSpaceReference,
    MediaKind,
    ModelOutputError,
    ModelReference,
    ModelRequestError,
    ModelUnavailableError,
)
from mindbridge.models import (
    EmbedRequest,
    EmbedTask,
    GenerateRequest,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.models.openai import OpenAIEmbedder, OpenAIGenerator, normalize_base_url

MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"


async def test_text_query_uses_typed_embedding_sdk() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert request.url.path == "/api/v1/embeddings"
        assert payload == {
            "input": ["Query: where is the tool?"],
            "model": MODEL_ID,
            "dimensions": 1_024,
            "encoding_format": "float",
        }
        return _embedding_response()

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("where is the tool?"),)),),
                task=EmbedTask.QUERY,
            )
        )
    finally:
        await embedder.close()

    assert result.embeddings[0].values == (1.0,) + (0.0,) * 1_023


async def test_memory_document_uses_jina_document_prompt() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert payload == {
            "input": ["Document: Caroline plans to become a counselor."],
            "model": MODEL_ID,
            "dimensions": 1_024,
            "encoding_format": "float",
        }
        return _embedding_response(model=MODEL_ID)

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(ModelInput((TextPart("Caroline plans to become a counselor."),)),),
                task=EmbedTask.DOCUMENT,
            )
        )
    finally:
        await embedder.close()

    assert result.embeddings[0].values == (1.0,) + (0.0,) * 1_023


async def test_text_document_embedder_batches_and_restores_index_order() -> None:
    second = [0.0, 1.0] + [0.0] * 1_022

    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert payload["input"] == ["Document: first event", "Document: second claim"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": second},
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [1.0] + [0.0] * 1_023,
                    },
                ],
                "model": MODEL_ID,
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    embedder = _embedder(respond)
    try:
        result = await embedder.embed(
            EmbedRequest(
                inputs=(
                    ModelInput((TextPart("first event"),)),
                    ModelInput((TextPart("second claim"),)),
                ),
                task=EmbedTask.DOCUMENT,
            )
        )
    finally:
        await embedder.close()

    assert tuple(item.values for item in result.embeddings) == (
        (1.0,) + (0.0,) * 1_023,
        tuple(second),
    )


async def test_multimodal_query_preserves_native_av_parts() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(list[dict[str, object]], messages[0]["content"])
        assert request.url.path == "/api/v1/embeddings"
        assert payload["model"] == MODEL_ID
        assert payload["dimensions"] == 1_024
        assert content[0] == {"type": "text", "text": "Query: find this moment"}
        assert {item["type"] for item in content} == {
            "text",
            "image_url",
            "video_url",
            "audio_url",
        }
        audio = next(item for item in content if item["type"] == "audio_url")
        assert cast(dict[str, str], audio["audio_url"])["url"].endswith("/media_audio")
        return _embedding_response()

    embedder = _embedder(respond)
    query = ModelInput(
        (
            TextPart("find this moment"),
            *(
                _media_part(kind, suffix)
                for kind, suffix in (
                    (MediaKind.IMAGE, "image"),
                    (MediaKind.VIDEO, "video"),
                    (MediaKind.AUDIO, "audio"),
                )
            ),
        )
    )
    try:
        await embedder.embed(EmbedRequest(inputs=(query,), task=EmbedTask.QUERY))
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    ("model", "embedding", "match"),
    [
        ("wrong-model", [1.0] + [0.0] * 1_023, "model"),
        (MODEL_ID, [1.0, 0.0], "dimension"),
        (MODEL_ID, [0.5] + [0.0] * 1_023, "L2-normalized"),
    ],
)
async def test_invalid_embedding_output_is_rejected(
    model: str,
    embedding: list[float],
    match: str,
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return _embedding_response(model=model, embedding=embedding)

    embedder = _embedder(respond)
    try:
        with pytest.raises(ModelOutputError, match=match):
            await embedder.embed(
                EmbedRequest(
                    inputs=(ModelInput((TextPart("find it"),)),),
                    task=EmbedTask.QUERY,
                )
            )
    finally:
        await embedder.close()


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, ModelRequestError),
        (408, ModelUnavailableError),
        (409, ModelUnavailableError),
        (429, ModelUnavailableError),
        (500, ModelUnavailableError),
    ],
)
async def test_only_transient_provider_failures_are_retryable(
    status_code: int,
    error_type: type[RuntimeError],
) -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "provider detail", "type": "request_error"}},
        )

    embedder = _embedder(respond)
    try:
        with pytest.raises(error_type, match="embedding request failed"):
            await embedder.embed(
                EmbedRequest(
                    inputs=(ModelInput((TextPart("find it"),)),),
                    task=EmbedTask.QUERY,
                )
            )
    finally:
        await embedder.close()


async def test_a_dropped_completion_stream_is_retryable() -> None:
    """A provider that closes a long multimodal response mid-stream raises inside the stream
    iterator, past the SDK error handler, so an observation died permanently on a transient
    network fault instead of being retried."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _truncated_completion_response()

    generator = _generator(respond)
    try:
        with pytest.raises(ModelUnavailableError, match="generation request failed"):
            await generator.generate(
                GenerateRequest(
                    system_prompt="Answer from evidence.",
                    input=ModelInput((TextPart("where is the tool?"),)),
                    max_output_tokens=64,
                )
            )
    finally:
        await generator.close()


async def test_generation_reports_the_configured_model_not_the_serving_fingerprint() -> None:
    """A per-request serving fingerprint must not become the model identity."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return _completion_response("serving-fingerprint-01")

    generator = _generator(respond)
    try:
        result = await generator.generate(
            GenerateRequest(
                system_prompt="Answer from evidence.",
                input=ModelInput((TextPart("where is the tool?"),)),
                max_output_tokens=64,
            )
        )
    finally:
        await generator.close()

    assert result.text == "on the workbench"
    assert result.model_reference == ModelReference(model_id="qwen3.8-max")


def _generator(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIGenerator:
    return OpenAIGenerator(
        AsyncOpenAI(
            api_key="unit-test-key",
            base_url=normalize_base_url("https://vlm.example.test/api/v1/chat/completions"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        ),
        ModelReference(model_id="qwen3.8-max"),
    )


def _completion_response(fingerprint: str) -> httpx.Response:
    event = {
        "id": "completion_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "on the workbench"},
                "finish_reason": "stop",
            }
        ],
    }
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n",
    )


def _truncated_completion_response() -> httpx.Response:
    """One valid chunk, then the peer disappears in the middle of the body."""
    event = {
        "id": "completion_01",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3.8-max",
        "choices": [{"index": 0, "delta": {"content": "on the "}, "finish_reason": None}],
    }

    async def body() -> AsyncIterator[bytes]:
        yield f"data: {json.dumps(event)}\n\n".encode()
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )

    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=body(),
    )


def _embedder(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIEmbedder:
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://embedding.example.test/api/v1/embeddings"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAIEmbedder(
        client,
        ModelReference(model_id=MODEL_ID),
        space_reference=EmbeddingSpaceReference(space_id="jina-space"),
    )


def _media_part(kind: MediaKind, suffix: str) -> MediaPart:
    extension = {MediaKind.IMAGE: "jpg", MediaKind.VIDEO: "mp4", MediaKind.AUDIO: "wav"}[kind]
    return MediaPart(
        kind=kind,
        url=f"https://objects.example.test/media_{suffix}",
        source_uri=f"s3://memory/tenants/tenant_01/query.{extension}",
    )


def _embedding_response(
    *,
    model: str = MODEL_ID,
    embedding: list[float] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding or [1.0] + [0.0] * 1_023,
                }
            ],
            "model": model,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )
