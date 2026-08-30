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
    ModelOutputTruncatedError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)
from mindbridge.types import (
    AnswerResult,
    AssetRef,
    Blob,
    ContentInput,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
    SearchHit,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
ENVELOPE_FIELDS = {
    "code",
    "reason",
    "retryable",
    "stage",
    "subject",
    "message",
    "trace_id",
    "issues",
}
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
        self.deleted = True

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        self._fail()
        copied_metadata = dict(metadata or {})
        self.calls.append(("add", content, occurred_at, occurred_end, copied_metadata, memory_type))
        return _record(
            "memory_1",
            content if isinstance(content, str) else "Multimodal memory.",
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=copied_metadata,
            assets=() if isinstance(content, str) else (ASSET,),
            modality=Modality.TEXT if isinstance(content, str) else Modality.IMAGE,
            memory_type=memory_type,
        )

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]:
        self._fail()
        copied = tuple(contents)
        occurrences = tuple(occurred_at or ())
        occurrence_ends = tuple(occurred_end or ())
        metadata_values = tuple(dict(value or {}) for value in metadata or ())
        self.calls.append(
            (
                "add_many",
                copied,
                occurrences,
                occurrence_ends,
                metadata_values,
                memory_type,
            )
        )
        return tuple(
            _record(
                f"batch_{index}",
                content if isinstance(content, str) else "Multimodal memory.",
                occurred_at=occurrences[index] if occurrences else None,
                occurred_end=occurrence_ends[index] if occurrence_ends else None,
                metadata=metadata_values[index] if metadata_values else None,
                memory_type=memory_type,
            )
            for index, content in enumerate(copied)
        )

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> tuple[SearchHit, ...]:
        self._fail()
        self.calls.append(("search", query, limit, memory_type, reference_at))
        return (_hit(),)

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> AnswerResult:
        self._fail()
        self.calls.append(("ask", question, limit, memory_type, reference_at))
        return AnswerResult(answer="The toolbox is blue.", hits=(_hit(),))

    def get(self, memory_id: str) -> MemoryRecord:
        self._fail()
        self.calls.append(("get", memory_id))
        return _record(memory_id, "The toolbox is blue.")

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        self._fail()
        self.calls.append(("list", limit, cursor))
        return Page(items=(_record("memory_1", "The toolbox is blue."),), next_cursor="next")

    def delete(self, memory_id: str) -> bool:
        self._fail()
        self.calls.append(("delete", memory_id))
        return self.deleted

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
                "memory_type": "episodic",
            },
        )
        batch = client.post(
            "/v1/memories/batch",
            json={
                "contents": [" first ", "second"],
                "occurred_at": [NOW.isoformat(), None],
                "occurred_end": [None, None],
                "metadata": [{"room": "workshop"}, None],
                "memory_type": "procedural",
            },
        )
        page = client.get("/v1/memories", params={"limit": 2, "cursor": "before"})
        found = client.get("/v1/memories/memory_1")
        searched = client.post(
            "/v1/memories/search",
            json={
                "query": " toolbox ",
                "limit": 3,
                "memory_type": "episodic",
                "reference_at": NOW.isoformat(),
            },
        )
        answered = client.post(
            "/v1/answers",
            json={
                "question": " What color is it? ",
                "limit": 4,
                "memory_type": "procedural",
                "reference_at": NOW.isoformat(),
            },
        )
        deleted = client.delete("/v1/memories/memory_1")
        openapi = client.get("/openapi.json").json()

    assert health.json() == {"status": "ok"}
    assert created.status_code == 201
    assert created.json()["content"] == "The toolbox is blue."
    assert created.json()["memory_type"] == "episodic"
    assert created.json()["metadata"] == {"room": "workshop"}
    assert [record["content"] for record in batch.json()["memories"]] == ["first", "second"]
    assert {record["memory_type"] for record in batch.json()["memories"]} == {"procedural"}
    assert page.json()["next_cursor"] == "next"
    assert found.json()["id"] == "memory_1"
    assert searched.json()["hits"][0]["score"] == 0.9
    assert answered.json()["answer"] == "The toolbox is blue."
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True}
    assert memory.calls == [
        (
            "add",
            "The toolbox is blue.",
            NOW,
            None,
            {"room": "workshop"},
            MemoryType.EPISODIC,
        ),
        (
            "add_many",
            ("first", "second"),
            (NOW, None),
            (None, None),
            ({"room": "workshop"}, {}),
            MemoryType.PROCEDURAL,
        ),
        ("list", 2, "before"),
        ("get", "memory_1"),
        ("search", "toolbox", 3, MemoryType.EPISODIC, NOW),
        ("ask", "What color is it?", 4, MemoryType.PROCEDURAL, NOW),
        ("delete", "memory_1"),
    ]
    assert memory.close_count == 0
    serialized_schema = json.dumps(openapi)
    assert all(name not in serialized_schema for name in ("tenant_id", "user_id", "run_id"))


def test_ordered_content_parts_map_to_public_inputs_without_exposing_asset_paths() -> None:
    memory = FakeMemory()
    image_data = base64.b64encode(b"png").decode()
    audio_data = base64.b64encode(b"wav").decode()
    video_data = base64.b64encode(b"video").decode()

    with TestClient(create_app(memory=memory)) as client:
        created = client.post(
            "/v1/memories",
            json={
                "content": [
                    {"type": "input_text", "text": "  At the station.  "},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
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
                        "file_url": f"data:video/mp4;base64,{video_data}",
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
    assert added[1] == Blob(b"png", "image/png")
    assert added[2] == Blob(b"wav", "audio/wav", "note.wav")
    assert added[3] == AssetRef(id="asset_existing", media_type="video/mp4")
    assert added[4] == AssetRef(id="asset_existing_image", modality=Modality.IMAGE)
    assert memory.calls[1][1] == (Blob(b"png", "image/png"),)
    assert memory.calls[2][1] == (Blob(b"video", "video/mp4", "clip.mp4"),)
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
    assert set(body) == ENVELOPE_FIELDS
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
    ("failure", "expected_status", "expected_code"),
    [
        (ValidationError("content is invalid"), 422, "validation_error"),
        (MemoryNotFoundError("memory is missing"), 404, "memory_not_found"),
        (SpeakerNotFoundError("speaker is missing"), 404, "speaker_not_found"),
        (ModelError("routing failed"), 502, "model_error"),
        (ModelOutputTruncatedError("answer was cut off"), 502, "model_output_truncated"),
        (StorageError("durable write failed"), 503, "storage_error"),
        (IndexUnavailableError("index rebuild failed"), 503, "index_unavailable"),
    ],
)
def test_public_errors_use_one_envelope_carrying_author_written_messages(
    failure: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        response = client.get("/v1/memories/memory_1")

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert str(failure) in response.text


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_reason", "expected_retryable"),
    [
        (
            ModelError("answer backend is not configured", reason="backend_not_configured"),
            501,
            "backend_not_configured",
            False,
        ),
        (
            ModelError("model cannot embed video", reason="unsupported_modality"),
            422,
            "unsupported_modality",
            False,
        ),
        (ModelError("provider refused", reason="auth_failed"), 502, "auth_failed", False),
        (ModelError("provider is busy", reason="rate_limited"), 503, "rate_limited", True),
        (ModelError("provider timed out", reason="timeout"), 503, "timeout", True),
        (
            StorageError("schema is too new", reason="schema_unsupported"),
            500,
            "schema_unsupported",
            False,
        ),
        (
            StorageError("directory is busy", reason="data_dir_in_use"),
            503,
            "data_dir_in_use",
            True,
        ),
    ],
)
def test_status_follows_whether_the_same_call_can_ever_succeed(
    failure: Exception,
    expected_status: int,
    expected_reason: str,
    expected_retryable: bool,
) -> None:
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        response = client.get("/v1/memories/memory_1")

    body = response.json()
    assert response.status_code == expected_status
    assert body["reason"] == expected_reason
    assert body["retryable"] is expected_retryable
    assert (response.headers.get("Retry-After") is not None) is (
        expected_retryable and expected_status == 503
    )


def test_error_envelope_reports_reason_stage_and_subject() -> None:
    failure = ModelError(
        "local media asset is unavailable",
        reason="asset_unavailable",
        stage="embed",
        subject="asset_image",
    )
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        body = client.get("/v1/memories/memory_1").json()

    assert set(body) == ENVELOPE_FIELDS
    assert body["reason"] == "asset_unavailable"
    assert body["stage"] == "embed"
    assert body["subject"] == "asset_image"
    assert body["retryable"] is False


def test_storage_subjects_stay_out_of_unauthenticated_responses() -> None:
    failure = StorageError(
        "the data directory is already in use by another live MindBridge instance",
        reason="data_dir_in_use",
        stage="open",
        subject="/srv/private/mindbridge",
    )
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        response = client.get("/v1/memories/memory_1")

    assert response.json()["subject"] is None
    assert "/srv/private/mindbridge" not in response.text


def test_list_default_page_size_matches_the_sdk() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        client.get("/v1/memories")

    assert memory.calls == [("list", 100, None)]


def test_delete_reports_whether_the_memory_existed() -> None:
    memory = FakeMemory()
    memory.deleted = False
    with TestClient(create_app(memory=memory)) as client:
        response = client.delete("/v1/memories/memory_1")

    assert response.status_code == 200
    assert response.json() == {"deleted": False}


def test_batch_creation_carries_every_value_the_memory_identity_uses() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/memories/batch",
            json={
                "contents": ["first", "second"],
                "occurred_at": [NOW.isoformat(), None],
                "occurred_end": [None, None],
                "metadata": [{"room": "workshop"}, None],
            },
        )

    assert response.status_code == 201
    assert memory.calls == [
        (
            "add_many",
            ("first", "second"),
            (NOW, None),
            (None, None),
            ({"room": "workshop"}, {}),
            MemoryType.SEMANTIC,
        )
    ]
    assert response.json()["memories"][0]["occurred_at"] == "2026-08-27T12:00:00Z"
    assert response.json()["memories"][0]["metadata"] == {"room": "workshop"}


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
    assert set(missing.json()) == ENVELOPE_FIELDS


def test_size_limit_runs_before_body_parsing() -> None:
    memory = FakeMemory()
    app = create_app(memory=memory)

    with TestClient(app) as client:
        malformed = client.post("/v1/memories", content=b"{")
        oversized = client.post(
            "/v1/memories",
            content=b"",
            headers={"Content-Length": str(8 * 1024 * 1024 + 1)},
        )

    assert malformed.status_code == 422
    assert malformed.json()["code"] == "validation_error"
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "request_too_large"
    assert memory.calls == []


def _record(
    memory_id: str,
    content: str,
    *,
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    assets: tuple[AssetRef, ...] = (),
    modality: Modality = Modality.TEXT,
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        content=content,
        modality=modality,
        memory_type=memory_type,
        assets=assets,
        created_at=NOW,
        occurred_at=occurred_at,
        occurred_end=occurred_end,
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
