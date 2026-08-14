"""Contract tests for the bundled OpenAI-compatible adapters."""

import json
from collections.abc import Callable, Coroutine
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
from mindbridge.models import EmbedRequest, EmbedTask, MediaPart, ModelInput, TextPart
from mindbridge.models.openai import OpenAIEmbedder, normalize_base_url

QUERY_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DOCUMENT_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"


async def test_text_query_uses_typed_embedding_sdk() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        payload: dict[str, object] = json.loads(request.content)
        assert request.url.path == "/api/v1/embeddings"
        assert payload == {
            "input": ["Query: where is the tool?"],
            "model": QUERY_MODEL_ID,
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
            "model": DOCUMENT_MODEL_ID,
            "dimensions": 1_024,
            "encoding_format": "float",
        }
        return _embedding_response(model=DOCUMENT_MODEL_ID)

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
                "model": DOCUMENT_MODEL_ID,
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
        assert payload["model"] == QUERY_MODEL_ID
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
        (QUERY_MODEL_ID, [1.0, 0.0], "dimension"),
        (QUERY_MODEL_ID, [0.5] + [0.0] * 1_023, "L2-normalized"),
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


def _embedder(
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> OpenAIEmbedder:
    client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://embedding.example.test/api/v1/embeddings"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    document_client = AsyncOpenAI(
        api_key="unit-test-key",
        base_url=normalize_base_url("https://text.example.test/api/v1/embeddings"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAIEmbedder(
        client,
        ModelReference(model_id=QUERY_MODEL_ID, revision="pinned-query-revision"),
        document_client=document_client,
        document_model_reference=ModelReference(
            model_id=DOCUMENT_MODEL_ID,
            revision="pinned-document-revision",
        ),
        space_reference=EmbeddingSpaceReference(space_id="jina-space", revision="v1"),
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
    model: str = QUERY_MODEL_ID,
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
