"""Contract checks for the SentenceTransformers Jina service."""

import asyncio
import base64
import hashlib
import math
import threading
import time

import httpx
import pytest
from anyio import Path as AsyncPath

from mindbridge import jina_server
from mindbridge.application.capabilities import (
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    MediaPart,
    TextPart,
)
from mindbridge.core import EmbeddingSpaceReference, MediaKind, ModelReference
from mindbridge.jina_server import create_app


def _fingerprint(text: str, dimension: int = 32) -> tuple[float, ...]:
    """A unit vector that depends on the input, so a mis-mapped result is visible.

    A fake that answers every input with the same vector cannot fail on the one defect a
    batching layer exists to avoid: handing caller A the embedding computed for caller B. With
    an identical vector, reversing the inputs or slicing every caller the head of the batch
    both stay green -- measured, on the whole suite. The `Embedding` invariant requires L2
    norm 1, hence the normalisation rather than a raw digest.
    """
    digest = hashlib.sha256(text.encode()).digest()
    raw = [byte + 1 for byte in digest[:dimension]]
    norm = math.sqrt(sum(value * value for value in raw))
    return tuple(value / norm for value in raw)


def _input_text(input_value: object) -> str:
    """The text an input carries, media parts included by URL, as the fingerprint's key."""
    parts = []
    for part in input_value.parts:  # type: ignore[attr-defined]
        if isinstance(part, TextPart):
            parts.append(part.text)
        elif isinstance(part, MediaPart):
            parts.append(f"{part.kind.value}:{part.url}")
    return "|".join(parts)


class _FakeEmbedder:
    def __init__(self) -> None:
        self.requests: list[EmbedRequest] = []
        self.media_contents: list[bytes] = []

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        return EmbeddingSpaceReference("test-space")

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.requests.append(request)
        for input_value in request.inputs:
            for part in input_value.parts:
                if isinstance(part, MediaPart):
                    assert await AsyncPath(part.url).is_file()
                    self.media_contents.append(await AsyncPath(part.url).read_bytes())
        return EmbedResult(
            tuple(
                Embedding(
                    _fingerprint(_input_text(input_value)),
                    ModelReference("test-model"),
                    self.space_reference,
                )
                for input_value in request.inputs
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


async def test_service_adaptively_batches_concurrent_requests() -> None:
    embedder = _FakeEmbedder()
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
        batch_wait_ms=50,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            client.post(
                "/v1/embeddings",
                headers={"Authorization": "Bearer test-key"},
                json={"model": "test-model", "dimensions": 32, "input": "Query: first"},
            ),
            client.post(
                "/v1/embeddings",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "test-model",
                    "dimensions": 32,
                    "input": ["Query: second", "Query: third"],
                },
            ),
        )

    assert [len(response.json()["data"]) for response in responses] == [1, 2]
    assert len(embedder.requests) == 1
    assert len(embedder.requests[0].inputs) == 3
    # The point of the batch is that each caller gets back the embedding of the text *it* sent.
    # `Query: ` is the task marker and is stripped, so the embedder sees the bare text.
    # Counts alone cannot see a reversed or head-sliced mapping, which is the whole failure mode.
    assert [point["embedding"] for point in responses[0].json()["data"]] == [
        list(_fingerprint("first"))
    ]
    assert [point["embedding"] for point in responses[1].json()["data"]] == [
        list(_fingerprint("second")),
        list(_fingerprint("third")),
    ]


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


async def test_service_keeps_media_from_separate_messages_distinct() -> None:
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
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,Zmlyc3Q="},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,c2Vjb25k"},
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    assert embedder.media_contents == [b"first", b"second"]


async def test_service_rejects_remote_media_outside_the_allowlist() -> None:
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
                            {
                                "type": "image_url",
                                "image_url": {"url": "http://169.254.169.254/latest/meta-data"},
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "remote media origin is not allowed"
    assert embedder.requests == []


async def test_service_downloads_allowlisted_media_with_a_bounded_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, int, int, str, str]] = []

    class Response:
        status = 200
        length = 6

        def __init__(self) -> None:
            self.headers = {"Content-Type": "image/png"}

        def read(self, amount: int) -> bytes:
            assert amount == jina_server._MAX_MEDIA_BYTES + 1
            return b"remote"

    class Connection:
        def __init__(self, host: str, port: int, *, timeout: int) -> None:
            requests.append((host, port, timeout, "", ""))

        def request(self, method: str, target: str) -> None:
            host, port, timeout, _method, _target = requests[-1]
            requests[-1] = host, port, timeout, method, target

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    embedder = _FakeEmbedder()
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
        media_origins=("https://media.example",),
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
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://media.example/object.png?signature=1"
                                },
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert requests == [("media.example", 443, 30, "GET", "/object.png?signature=1")]
    assert embedder.media_contents == [b"remote"]


async def test_service_materializes_video_inputs_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class Response:
        status = 200
        length = 5

        def __init__(self) -> None:
            self.headers = {"Content-Type": "video/mp4"}

        def read(self, _amount: int) -> bytes:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return b"video"

    class Connection:
        def __init__(self, _host: str, _port: int, *, timeout: int) -> None:
            assert timeout == 30

        def request(self, _method: str, _target: str) -> None:
            return None

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    embedder = _FakeEmbedder()
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
        media_origins=("https://media.example",),
    )
    sample = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "https://media.example/video.mp4"},
                }
            ],
        }
    ]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "test-model",
                "dimensions": 32,
                "input": [sample, sample],
            },
        )

    assert response.status_code == 200
    assert peak == 2


async def test_service_rejects_an_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jina_server, "_MAX_REQUEST_BYTES", 32)
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=_FakeEmbedder(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "test-model", "dimensions": 32, "input": ["Document: value"]},
        )

    assert response.status_code == 413


async def test_service_rejects_oversized_base64_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jina_server, "_MAX_MEDIA_BYTES", 4)

    def unexpected_decode(_encoded: str, *, validate: bool) -> bytes:
        raise AssertionError(f"decoded oversized media with validate={validate}")

    monkeypatch.setattr(base64, "b64decode", unexpected_decode)
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=_FakeEmbedder(),
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
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,MTIzNDU2Nw=="},
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 413


class _PoisonEmbedder(_FakeEmbedder):
    """Fails only on one caller's input, the way a corrupt clip from one tenant does."""

    def __init__(self, poison: str) -> None:
        super().__init__()
        self._poison = poison
        self.batch_sizes: list[int] = []

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        self.batch_sizes.append(len(request.inputs))
        if any(_input_text(value) == self._poison for value in request.inputs):
            raise RuntimeError("model rejected an input")
        return await super().embed(request)


async def test_one_callers_bad_input_does_not_fail_the_others_in_its_batch() -> None:
    """A batch is the server's scheduling choice, so its failures must not be shared.

    Merging requests makes one tenant's corrupt media able to fail up to `--max-batch-inputs`
    unrelated callers, and the retry then re-forms a similar batch. Attribution has to survive
    batching or the feature converts one bad clip into a broad outage.
    """
    embedder = _PoisonEmbedder("poison")
    app = create_app(
        api_key="test-key",
        embedder_config={"model_id": "test-model", "dimension": 32},
        embedder=embedder,
        batch_wait_ms=50,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        healthy, poisoned = await asyncio.gather(
            client.post(
                "/v1/embeddings",
                headers={"Authorization": "Bearer test-key"},
                json={"model": "test-model", "dimensions": 32, "input": "Query: healthy"},
            ),
            client.post(
                "/v1/embeddings",
                headers={"Authorization": "Bearer test-key"},
                json={"model": "test-model", "dimensions": 32, "input": "Query: poison"},
            ),
        )

    # They really were merged -- otherwise this proves nothing about batching.
    assert embedder.batch_sizes[0] == 2
    assert healthy.status_code == 200
    assert [point["embedding"] for point in healthy.json()["data"]] == [
        list(_fingerprint("healthy"))
    ]
    assert poisoned.status_code == 503
