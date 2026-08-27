"""Focused contract checks for the local FastAPI adapter."""

import base64
import inspect
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from mindbridge.api.app import create_app
from mindbridge.exceptions import (
    IndexUnavailableError,
    MemoryNotFoundError,
    ModelError,
    StorageError,
    ValidationError,
)
from mindbridge.types import (
    URL,
    AnswerResult,
    AssetRef,
    Blob,
    ContentInput,
    MemoryRecord,
    Modality,
    Page,
    SearchHit,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
API_KEY = "one-private-api-key"
ASSET = AssetRef(
    id="asset_image",
    modality=Modality.IMAGE,
    media_type="image/png",
    size_bytes=3,
    sha256="a" * 64,
    name="frame.png",
    path=Path("/private/mindbridge/assets/frame.png"),
)


class FakeMemory:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.close_count = 0
        self.failure = failure

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        self._fail()
        copied_metadata = dict(metadata or {})
        self.calls.append(("add", content, occurred_at, copied_metadata))
        return _record(
            "memory_1",
            content if isinstance(content, str) else "Multimodal memory.",
            occurred_at=occurred_at,
            metadata=copied_metadata,
            assets=() if isinstance(content, str) else (ASSET,),
            modality=Modality.TEXT if isinstance(content, str) else Modality.IMAGE,
        )

    def add_many(self, contents: Sequence[ContentInput]) -> tuple[MemoryRecord, ...]:
        self._fail()
        copied = tuple(contents)
        self.calls.append(("add_many", copied))
        return tuple(
            _record(
                f"batch_{index}",
                content if isinstance(content, str) else "Multimodal memory.",
            )
            for index, content in enumerate(copied)
        )

    def search(self, query: ContentInput, *, limit: int = 10) -> tuple[SearchHit, ...]:
        self._fail()
        self.calls.append(("search", query, limit))
        return (_hit(),)

    def ask(self, question: ContentInput, *, limit: int = 10) -> AnswerResult:
        self._fail()
        self.calls.append(("ask", question, limit))
        return AnswerResult(answer="The toolbox is blue.", hits=(_hit(),))

    def get(self, memory_id: str) -> MemoryRecord:
        self._fail()
        self.calls.append(("get", memory_id))
        return _record(memory_id, "The toolbox is blue.")

    def list(self, *, limit: int = 50, cursor: str | None = None) -> Page:
        self._fail()
        self.calls.append(("list", limit, cursor))
        return Page(items=(_record("memory_1", "The toolbox is blue."),), next_cursor="next")

    def delete(self, memory_id: str) -> bool:
        self._fail()
        self.calls.append(("delete", memory_id))
        return True

    def close(self) -> None:
        self.close_count += 1

    def _fail(self) -> None:
        if self.failure is not None:
            raise self.failure


def test_resource_routes_map_the_public_memory_values() -> None:
    memory = FakeMemory()
    app = create_app(memory=memory)

    with TestClient(app) as client:
        health = client.get("/healthz")
        created = client.post(
            "/v1/memories",
            json={
                "content": "  The toolbox is blue.  ",
                "occurred_at": NOW.isoformat(),
                "metadata": {"room": "workshop"},
            },
        )
        batch = client.post(
            "/v1/memories/batch",
            json={"contents": [" first ", "second"]},
        )
        page = client.get("/v1/memories", params={"limit": 2, "cursor": "before"})
        found = client.get("/v1/memories/memory_1")
        searched = client.post(
            "/v1/memories/search",
            json={"query": " toolbox ", "limit": 3},
        )
        answered = client.post(
            "/v1/answers",
            json={"question": " What color is it? ", "limit": 4},
        )
        deleted = client.delete("/v1/memories/memory_1")
        openapi = client.get("/openapi.json").json()

    assert health.json() == {"status": "ok"}
    assert created.status_code == 201
    assert created.json()["content"] == "The toolbox is blue."
    assert created.json()["metadata"] == {"room": "workshop"}
    assert [record["content"] for record in batch.json()["memories"]] == ["first", "second"]
    assert page.json()["next_cursor"] == "next"
    assert found.json()["id"] == "memory_1"
    assert searched.json()["hits"][0]["score"] == 0.9
    assert answered.json()["answer"] == "The toolbox is blue."
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert memory.calls == [
        ("add", "The toolbox is blue.", NOW, {"room": "workshop"}),
        ("add_many", ("first", "second")),
        ("list", 2, "before"),
        ("get", "memory_1"),
        ("search", "toolbox", 3),
        ("ask", "What color is it?", 4),
        ("delete", "memory_1"),
    ]
    assert memory.close_count == 0
    serialized_schema = json.dumps(openapi)
    assert all(name not in serialized_schema for name in ("tenant_id", "user_id", "run_id"))


def test_ordered_content_parts_map_to_public_inputs_without_exposing_asset_paths() -> None:
    memory = FakeMemory()
    image_data = base64.b64encode(b"png").decode()
    audio_data = base64.b64encode(b"wav").decode()

    with TestClient(create_app(memory=memory)) as client:
        created = client.post(
            "/v1/memories",
            json={
                "content": [
                    {"type": "input_text", "text": "  At the station.  "},
                    {
                        "type": "input_image",
                        "image_url": "https://media.example/frame.png",
                    },
                    {
                        "type": "input_file",
                        "file_data": audio_data,
                        "media_type": "audio/wav",
                        "filename": "note.wav",
                    },
                    {
                        "type": "input_file",
                        "file_id": "asset_existing",
                        "media_type": "video/mp4",
                    },
                    {"type": "input_image", "file_id": "asset_existing_image"},
                ]
            },
        )
        searched = client.post(
            "/v1/memories/search",
            json={
                "query": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    }
                ]
            },
        )
        answered = client.post(
            "/v1/answers",
            json={
                "question": [
                    {
                        "type": "input_file",
                        "file_url": "https://media.example/clip.mp4",
                        "media_type": "video/mp4",
                        "filename": "clip.mp4",
                    }
                ]
            },
        )
        openapi = client.get("/openapi.json").json()

    assert created.status_code == 201
    assert searched.status_code == answered.status_code == 200
    added = memory.calls[0][1]
    assert isinstance(added, tuple)
    assert added[0] == "At the station."
    assert added[1] == URL("https://media.example/frame.png", "image/*")
    assert added[2] == Blob(b"wav", "audio/wav", "note.wav")
    assert added[3] == AssetRef(id="asset_existing", media_type="video/mp4")
    assert added[4] == AssetRef(id="asset_existing_image", modality=Modality.IMAGE)
    assert memory.calls[1][1] == (Blob(b"png", "image/png"),)
    assert memory.calls[2][1] == (URL("https://media.example/clip.mp4", "video/mp4", "clip.mp4"),)
    body = created.json()
    assert body["content"] == "Multimodal memory."
    assert body["modality"] == "image"
    assert body["assets"] == [
        {
            "id": "asset_image",
            "modality": "image",
            "media_type": "image/png",
            "size_bytes": 3,
            "sha256": "a" * 64,
            "name": "frame.png",
        }
    ]
    assert "path" not in openapi["components"]["schemas"]["AssetResponse"]["properties"]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "input_text", "text": "memory", "unknown": True}],
        [
            {
                "type": "input_image",
                "image_url": "https://media.example/frame.png",
                "detail": "high",
            }
        ],
        [
            {
                "type": "input_image",
                "image_url": "https://media.example/frame.png",
                "file_id": "asset_image",
            }
        ],
        [{"type": "input_image", "image_url": "http://media.example/frame.png"}],
        [{"type": "input_file", "file_url": "file:///etc/passwd"}],
        [{"type": "input_file", "file_data": "not-base64", "media_type": "audio/wav"}],
        [{"type": "input_file", "file_data": "d2F2"}],
        [{"type": "input_file", "file_id": "asset_pdf", "media_type": "application/pdf"}],
        [{"type": "input_file", "path": "/etc/passwd"}],
    ],
)
def test_content_parts_reject_ambiguous_or_untrusted_sources_before_memory(
    content: object,
) -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory)) as client:
        response = client.post("/v1/memories", json={"content": content})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert memory.calls == []


def test_memory_routes_are_sync_for_fastapi_threadpool_execution() -> None:
    app = create_app(memory=FakeMemory())
    endpoints = [
        route.endpoint
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith("/v1/")
    ]

    assert endpoints
    assert all(not inspect.iscoroutinefunction(endpoint) for endpoint in endpoints)


@pytest.mark.parametrize("field", ["tenant_id", "user_id", "run_id"])
def test_old_isolation_fields_are_rejected_without_echoing_values(field: str) -> None:
    with TestClient(create_app(memory=FakeMemory())) as client:
        response = client.post(
            "/v1/memories",
            json={"content": "safe memory", field: "private-isolation-value"},
        )

    body = response.json()
    assert response.status_code == 422
    assert set(body) == {"code", "message", "trace_id", "issues"}
    assert body["code"] == "validation_error"
    assert body["issues"][0]["location"][-1] == field
    assert body["trace_id"].startswith("trace_")
    assert "private-isolation-value" not in response.text


@pytest.mark.parametrize(
    ("path", "payload", "field"),
    [
        ("/v1/memories", {"content": "   "}, "content"),
        (
            "/v1/memories",
            {"content": "memory", "occurred_at": "2026-08-27T12:00:00"},
            "occurred_at",
        ),
        ("/v1/memories/batch", {"contents": []}, "contents"),
        ("/v1/memories/search", {"query": "memory", "limit": 101}, "limit"),
    ],
)
def test_request_boundaries_are_strict(
    path: str,
    payload: dict[str, object],
    field: str,
) -> None:
    with TestClient(create_app(memory=FakeMemory())) as client:
        response = client.post(path, json=payload)

    assert response.status_code == 422
    assert field in response.json()["issues"][0]["location"]


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_code", "detail_is_public"),
    [
        (ValidationError("content is invalid"), 422, "validation_error", True),
        (MemoryNotFoundError("memory is missing"), 404, "memory_not_found", True),
        (ModelError("provider secret"), 502, "model_error", False),
        (StorageError("database path"), 503, "storage_error", False),
        (IndexUnavailableError("native detail"), 503, "index_unavailable", False),
    ],
)
def test_public_errors_use_one_sanitized_envelope(
    failure: Exception,
    expected_status: int,
    expected_code: str,
    detail_is_public: bool,
) -> None:
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        response = client.get("/v1/memories/memory_1")

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert (str(failure) in response.text) is detail_is_public


def test_unexpected_and_framework_errors_keep_the_flat_envelope() -> None:
    app = create_app(memory=FakeMemory(RuntimeError("private implementation detail")))
    with TestClient(app, raise_server_exceptions=False) as client:
        failed = client.get("/v1/memories/memory_1")
        missing = client.get("/v1/does-not-exist")

    assert failed.status_code == 500
    assert failed.json()["code"] == "internal_error"
    assert "private implementation detail" not in failed.text
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    assert set(missing.json()) == {"code", "message", "trace_id", "issues"}


def test_optional_api_key_protects_v1_but_not_health() -> None:
    app = create_app(memory=FakeMemory(), api_key=API_KEY)
    with TestClient(app) as client:
        health = client.get("/healthz")
        missing = client.get("/v1/memories")
        invalid = client.get(
            "/v1/memories",
            headers={"Authorization": "Bearer wrong"},
        )
        allowed = client.get(
            "/v1/memories",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )

    assert health.status_code == 200
    assert missing.status_code == invalid.status_code == 401
    assert missing.headers["WWW-Authenticate"] == "Bearer"
    assert invalid.json()["code"] == "authentication_error"
    assert API_KEY not in invalid.text
    assert allowed.status_code == 200


def test_authentication_and_size_limits_run_before_body_parsing() -> None:
    memory = FakeMemory()
    app = create_app(memory=memory, api_key=API_KEY)

    with TestClient(app) as client:
        unauthenticated = client.post("/v1/memories", content=b"{")
        oversized = client.post(
            "/v1/memories",
            content=b"",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Length": str(8 * 1024 * 1024 + 1),
            },
        )

    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["code"] == "authentication_error"
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "request_too_large"
    assert memory.calls == []


def test_lifespan_closes_only_a_factory_owned_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = FakeMemory()
    open_count = 0

    def create_memory(*, data_dir: str | Path) -> FakeMemory:
        nonlocal open_count
        assert Path(data_dir) == tmp_path
        open_count += 1
        return owned

    monkeypatch.setattr("mindbridge.memory.Memory", create_memory)
    app = create_app(data_dir=tmp_path)
    assert open_count == 0

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert open_count == 1
        assert owned.close_count == 0

    assert owned.close_count == 1


def _record(
    memory_id: str,
    content: str,
    *,
    occurred_at: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    assets: tuple[AssetRef, ...] = (),
    modality: Modality = Modality.TEXT,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        modality=modality,
        assets=assets,
        created_at=NOW,
        occurred_at=occurred_at,
        metadata=metadata or {},
    )


def _hit() -> SearchHit:
    return SearchHit(
        id="memory_1",
        content="The toolbox is blue.",
        score=0.9,
        modality=Modality.TEXT,
        created_at=NOW,
    )
