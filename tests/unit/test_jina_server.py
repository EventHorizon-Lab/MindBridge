"""Contract checks for the SentenceTransformers Jina service."""

import httpx
from anyio import Path as AsyncPath

from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    MediaPart,
)
from mindbridge.core import EmbeddingSpaceReference, MediaKind, ModelReference
from mindbridge.jina_server import create_app


class _FakeEmbedder:
    def __init__(self) -> None:
        self.requests: list[EmbedRequest] = []

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        return EmbeddingSpaceReference("test-space")

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.requests.append(request)
        for input_value in request.inputs:
            for part in input_value.parts:
                if isinstance(part, MediaPart):
                    assert await AsyncPath(part.url).is_file()
        vector = (1.0,) + (0.0,) * 31
        return EmbedResult(
            tuple(
                Embedding(
                    vector,
                    ModelReference("test-model"),
                    self.space_reference,
                )
                for _input in request.inputs
            )
        )


async def test_service_batches_text_and_rejects_the_removed_messages_shape() -> None:
    embedder = _FakeEmbedder()
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        health = await client.get("/health")
        unauthorized = await client.post(
            "/v1/embeddings",
            json={"model": "test-model", "dimensions": 32, "input": ["Query: first"]},
        )
        response = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test-model",
                "dimensions": 32,
                "input": ["Query: first", "Query: second"],
            },
        )
        removed_shape = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "test-model", "dimensions": 32, "messages": []},
        )
        malformed = await client.post(
            "/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
            },
            content="{",
        )

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2
    assert len(embedder.requests) == 1
    assert embedder.requests[0].task is EmbedTask.QUERY
    assert len(embedder.requests[0].inputs) == 2
    assert removed_shape.status_code == 400
    assert malformed.status_code == 400


async def test_service_accepts_omni_input_and_materializes_data_uris() -> None:
    embedder = _FakeEmbedder()
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test-model",
                "dimensions": 32,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Document: a moment"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                            },
                            {
                                "type": "video_url",
                                "video_url": {"url": "data:video/mp4;base64,dmlkZW8="},
                            },
                            {
                                "type": "audio_url",
                                "audio_url": {"url": "data:audio/wav;base64,YXVkaW8="},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert embedder.requests[0].task is EmbedTask.DOCUMENT
    media = [part for part in embedder.requests[0].inputs[0].parts if isinstance(part, MediaPart)]
    assert [part.kind for part in media] == [
        MediaKind.IMAGE,
        MediaKind.VIDEO,
        MediaKind.AUDIO,
    ]
    assert all([not await AsyncPath(part.url).exists() for part in media])
