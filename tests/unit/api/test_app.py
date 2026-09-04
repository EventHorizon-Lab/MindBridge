"""Focused contract checks for the local FastAPI adapter."""

import base64
import inspect
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from mindbridge.api.app import create_app
from mindbridge.exceptions import (
    IdentityNotFoundError,
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
    ContextBudget,
    ContextBundle,
    ContextConflict,
    ContextUnknown,
    ContextUnknownKind,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryCapabilities,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    PendingCapture,
    ProvisionalActor,
    RetrievalCandidateTrace,
    RetrievalRejection,
    RetrievalScope,
    RetrievalTrace,
    SearchHit,
    SpeakerSegment,
    TracedSearchResult,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
OCCURRED_FROM = datetime(2026, 8, 27, tzinfo=timezone.utc)
OCCURRED_UNTIL = datetime(2026, 8, 28, tzinfo=timezone.utc)
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


# Every modality, because `frozenset` iterates in hash order and enum members hash by identity:
# with five values an unsorted serializer produces the documented order once in 120 runs.
CAPABILITIES = MemoryCapabilities(
    embedding=frozenset(Modality),
    embedding_model="jina-v5-omni",
    embedding_space="space_1",
    embedding_dimension=1024,
    generation=frozenset({Modality.TEXT}),
    generation_model="qwen3-omni",
    speaker_recognition=True,
)


PROVISIONAL = ProvisionalActor(identity_id="identity_2", memory_ids=("memory_2",))
CONFLICT = ContextConflict(
    lineage_id="lineage_1",
    subject="ana",
    predicate="location",
    values=("berlin", "paris"),
    memory_ids=("memory_1", "memory_2"),
)
UNKNOWN = ContextUnknown(
    kind=ContextUnknownKind.BUDGET_EXCLUDED,
    detail="3 candidates did not fit 24 items and 16000 chars",
)
SEGMENT = SpeakerSegment(
    asset_id="b" * 64,
    start_ms=0,
    end_ms=1500,
    text="Where is the toolbox?",
    speaker_id="speaker_1",
    speaker_name="Ann",
    identity_score=0.82,
)
FACE = FaceObservation(
    asset_id="c" * 64,
    bounding_box=(0.1, 0.2, 0.3, 0.4),
    identity_id="identity_1",
    identity_name="Ann",
    identity_score=0.77,
    observed_at_ms=250,
)
ERASURE = IdentityErasure(
    identity_id="identity_1",
    alias_ids=("identity_4",),
    face_exemplars=3,
    voice_exemplars=2,
    face_observations=7,
    speech_segments=11,
)
PROFILE = IdentityProfile(
    identity_id="identity_1",
    name="Ann",
    relationship="daughter",
    confirmed=True,
    evidence_ids=("memory_1",),
)
PENDING = PendingCapture(
    memory_id="memory_1",
    enqueued_at=NOW,
    attempts=1,
    last_error="model timed out",
    awaiting="enrichment",
)


class FakeMemory:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.close_count = 0
        self.failure = failure
        self.deleted = True
        self.declared = CAPABILITIES
        self.profile: IdentityProfile | None = PROFILE
        self.restored: str | None = "identity_2"

    @property
    def capabilities(self) -> MemoryCapabilities:
        return self.declared

    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
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
        context: Sequence[ObservationContext | None] | None = None,
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
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> tuple[SearchHit, ...]:
        self._fail()
        self.calls.append(
            (
                "search",
                query,
                limit,
                memory_type,
                reference_at,
                occurred_from,
                occurred_until,
            )
        )
        return (_hit(),)

    def search_with_trace(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> TracedSearchResult:
        self._fail()
        self.calls.append(
            (
                "search_with_trace",
                query,
                limit,
                memory_type,
                reference_at,
                occurred_from,
                occurred_until,
            )
        )
        return TracedSearchResult(hits=(), trace=_trace())

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> ContextBundle:
        self._fail()
        self.calls.append(("compile", goal, budget, reference_at, scope))
        return _bundle(budget or ContextBudget(), reference_at or NOW)

    def reinforce(self, memory_ids: Sequence[str]) -> int:
        self._fail()
        self.calls.append(("reinforce", tuple(memory_ids)))
        return len(tuple(dict.fromkeys(memory_ids)))

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
    ) -> AnswerResult:
        self._fail()
        self.calls.append(("ask", question, limit, memory_type, reference_at, link_identities))
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

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord:
        self._fail()
        copied_metadata = dict(metadata or {})
        self.calls.append(
            ("capture", content, occurred_at, occurred_end, copied_metadata, memory_type)
        )
        return _record(
            "memory_1",
            content if isinstance(content, str) else "Multimodal memory.",
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=copied_metadata,
            memory_type=memory_type,
        )

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int:
        self._fail()
        self.calls.append(
            ("settle", limit, max_attempts, tuple(memory_ids) if memory_ids else None)
        )
        return 2

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]:
        self._fail()
        self.calls.append(("pending_captures", limit, tuple(memory_ids) if memory_ids else None))
        return (PENDING,)

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        self._fail()
        self.calls.append(("speech", memory_id))
        return (SEGMENT,)

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        self._fail()
        self.calls.append(("faces", memory_id))
        return (FACE,)

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        self._fail()
        self.calls.append(("register_identity", identity_id, name, relationship))

    def identity(self, identity_id: str) -> IdentityProfile | None:
        self._fail()
        self.calls.append(("identity", identity_id))
        return self.profile

    def unlink_identity(self, alias_id: str) -> str | None:
        self._fail()
        self.calls.append(("unlink_identity", alias_id))
        return self.restored

    def forget_identity(self, identity_id: str) -> IdentityErasure:
        self._fail()
        self.calls.append(("forget_identity", identity_id))
        return ERASURE

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
                "occurred_from": OCCURRED_FROM.isoformat(),
                "occurred_until": OCCURRED_UNTIL.isoformat(),
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

    assert health.json()["status"] == "ok"
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
    assert answered.json()["abstained"] is False
    assert answered.json()["abstention_reason"] is None
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
        (
            "search",
            "toolbox",
            3,
            MemoryType.EPISODIC,
            NOW,
            OCCURRED_FROM,
            OCCURRED_UNTIL,
        ),
        ("ask", "What color is it?", 4, MemoryType.PROCEDURAL, NOW, False),
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


def test_search_rejects_a_reversed_occurrence_range_before_memory() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/memories/search",
            json={
                "query": "memory",
                "occurred_from": OCCURRED_UNTIL.isoformat(),
                "occurred_until": OCCURRED_FROM.isoformat(),
            },
        )

    assert response.status_code == 422
    assert "occurred_until must be later than occurred_from" in response.text
    assert memory.calls == []


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


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        # One oversized-input condition, one status, whichever side noticed it. This used to be
        # 413 from the request middleware and 502 from the provider path.
        (ModelError("asset exceeds the request budget", reason="payload_too_large"), 413),
        # Not retryable, so it must not claim to be transient with 503.
        (StorageError("failed to read memory", reason="io_failed"), 500),
    ],
)
def test_one_reason_answers_with_one_status(failure: Exception, expected_status: int) -> None:
    with TestClient(create_app(memory=FakeMemory(failure))) as client:
        response = client.get("/v1/memories/memory_1")

    assert response.status_code == expected_status
    assert response.json()["retryable"] is False
    assert response.headers.get("Retry-After") is None


def test_health_reports_the_live_composition() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["capabilities"] == {
        "embedding": ["audio", "image", "omni", "text", "video"],
        "embedding_model": "jina-v5-omni",
        "embedding_space": "space_1",
        "embedding_dimension": 1024,
        "generation": ["text"],
        "transcription": [],
        "vision": [],
        "face": [],
        "formation": [],
        "generation_model": "qwen3-omni",
        "transcription_space": None,
        "vision_model": None,
        "face_model": None,
        "formation_model": None,
        "consolidation_model": None,
        "speaker_recognition": True,
        "streaming_generation": False,
        # Derived from the backends above: an agent reads which operations it may call instead
        # of inferring the mapping from a modality set.
        "operations": ["ask"],
    }


def test_health_reads_the_injected_memory_rather_than_a_captured_snapshot() -> None:
    memory = FakeMemory()
    memory.declared = MemoryCapabilities(
        embedding=frozenset({Modality.TEXT}),
        embedding_model="text-embedder",
        embedding_space="space_2",
        embedding_dimension=8,
    )

    with TestClient(create_app(memory=memory)) as client:
        body = client.get("/healthz").json()

    assert body["capabilities"]["embedding_space"] == "space_2"
    assert body["capabilities"]["embedding"] == ["text"]
    assert body["capabilities"]["speaker_recognition"] is False


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


def test_the_context_route_returns_the_whole_bundle_without_local_asset_paths() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        response = client.post(
            "/v1/context",
            json={
                "goal": "  What should I bring?  ",
                "budget": {
                    "max_chars": 2_000,
                    "max_items": 8,
                    "memory_types": ["episodic", "semantic"],
                    "min_confidence": 0.5,
                    "freshness_seconds": 3_600,
                    "max_latency_ms": 250,
                },
                "reference_at": NOW.isoformat(),
            },
        )

    assert response.status_code == 200
    assert memory.calls == [
        (
            "compile",
            "What should I bring?",
            ContextBudget(
                max_chars=2_000,
                max_items=8,
                memory_types=frozenset({MemoryType.EPISODIC, MemoryType.SEMANTIC}),
                min_confidence=0.5,
                freshness=timedelta(hours=1),
                max_latency_ms=250,
            ),
            NOW,
            None,
        )
    ]
    bundle = response.json()
    assert bundle["goal"] == "What should I bring?"
    assert bundle["budget"] == {
        "max_chars": 2_000,
        "max_items": 8,
        "memory_types": ["episodic", "semantic"],
        "min_confidence": 0.5,
        "freshness_seconds": 3_600.0,
        "max_latency_ms": 250,
    }
    assert [hit["id"] for hit in bundle["facts"]] == ["memory_1"]
    assert [hit["id"] for hit in bundle["episodes"]] == ["memory_2"]
    # A person the evidence observed whom nobody has named travels beside the ranked hits.
    assert bundle["actors"] == [{"identity_id": "identity_2", "memory_ids": ["memory_2"]}]
    assert bundle["conflicts"] == [
        {
            "lineage_id": "lineage_1",
            "subject": "ana",
            "predicate": "location",
            "values": ["berlin", "paris"],
            "memory_ids": ["memory_1", "memory_2"],
        }
    ]
    assert bundle["unknowns"] == [
        {
            "kind": "budget_excluded",
            "detail": "3 candidates did not fit 24 items and 16000 chars",
        }
    ]
    assert bundle["occurred_from"] == "2026-08-27T00:00:00Z"
    assert bundle["occurred_until"] == "2026-08-28T00:00:00Z"
    assert bundle["frames"] == ["home/map"]
    assert bundle["places"] == ["kitchen"]
    assert bundle["relationships"] == [] and bundle["scene"] == []
    assert bundle["omitted"] == 3
    assert bundle["chars"] == 42
    assert (bundle["elapsed_ms"], bundle["deadline_exceeded"]) == (7, False)
    assert bundle["rendered"].startswith("# Context: What should I bring?")
    # The bundle serializes hits through the same asset model every other route uses.
    assert bundle["episodes"][0]["assets"] == [
        {
            "id": "asset_image",
            "modality": "image",
            "media_type": "image/png",
            "size_bytes": 3,
            "sha256": "a" * 64,
            "name": "frame.png",
        }
    ]
    assert "/private/mindbridge/assets" not in response.text


def test_the_context_route_defaults_to_the_sdk_budget() -> None:
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        response = client.post("/v1/context", json={"goal": "What should I bring?"})

    assert response.status_code == 200
    assert memory.calls == [("compile", "What should I bring?", None, None, None)]
    assert response.json()["budget"] == {
        "max_chars": 16_000,
        "max_items": 24,
        "memory_types": None,
        "min_confidence": 0.0,
        "freshness_seconds": None,
        "max_latency_ms": None,
    }


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


def test_search_keeps_its_default_shape_and_explains_an_empty_result() -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory)) as client:
        plain = client.post("/v1/memories/search", json={"query": "toolbox"})
        explained = client.post("/v1/memories/search", json={"query": "toolbox", "explain": True})

    assert plain.status_code == 200
    assert plain.json()["hits"][0]["id"] == "memory_1"
    assert plain.json()["trace"] is None
    assert explained.status_code == 200
    body = explained.json()
    assert body["hits"] == []
    assert body["trace"]["ambiguous"] is True
    assert body["trace"]["candidate_limit"] == 50
    assert body["trace"]["exhaustive"] is True
    assert body["trace"]["candidates"] == [
        {
            "memory_id": "memory_1",
            "index_ids": ["index_1"],
            "dense_relevance": 0.42,
            "dense_confidence": None,
            "lexical_relevance": None,
            "lexical_rerank_bonus": None,
            "lexical_match": False,
            "gate_relevance": 0.31,
            "base_relevance": None,
            "reinforcement_factor": None,
            "temporal_factor": None,
            "retention_factor": None,
            "final_score": 0.4,
            "rank": None,
            "rejected_by": "minimum_relevance",
        }
    ]
    assert memory.calls == [
        ("search", "toolbox", 10, None, None, None, None),
        ("search_with_trace", "toolbox", 10, None, None, None, None),
    ]


def test_reinforce_route_reaches_the_sdk_and_rejects_an_empty_list() -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory)) as client:
        reinforced = client.post(
            "/v1/memories/reinforce", json={"memory_ids": ["memory_1", "memory_1", "memory_2"]}
        )
        empty = client.post("/v1/memories/reinforce", json={"memory_ids": []})

    assert reinforced.status_code == 200
    assert reinforced.json() == {"reinforced": 2}
    assert memory.calls == [("reinforce", ("memory_1", "memory_1", "memory_2"))]
    assert empty.status_code == 422
    assert empty.json()["code"] == "validation_error"


def test_capture_settle_and_pending_captures_are_always_on() -> None:
    """The fast plane is an ordinary application operation, unlike identity and embodied access."""
    memory = FakeMemory()

    with TestClient(create_app(memory=memory)) as client:
        captured = client.post(
            "/v1/capture",
            json={"content": "The spare key is in the blue toolbox.", "metadata": {"room": "hall"}},
        )
        settled = client.post(
            "/v1/settle",
            json={"limit": 10, "max_attempts": 5, "memory_ids": ["memory_1", "memory_2"]},
        )
        settled_defaults = client.post("/v1/settle", json={})
        pending = client.get(
            "/v1/pending_captures", params={"limit": 5, "memory_ids": ["memory_1"]}
        )

    assert captured.status_code == 201
    assert captured.json()["content"] == "The spare key is in the blue toolbox."
    assert settled.status_code == 200
    assert settled.json() == {"settled": 2}
    assert settled_defaults.json() == {"settled": 2}
    assert pending.status_code == 200
    assert pending.json() == {
        "items": [
            {
                "memory_id": "memory_1",
                "enqueued_at": "2026-08-27T12:00:00Z",
                "attempts": 1,
                "last_error": "model timed out",
                "awaiting": "enrichment",
            }
        ]
    }
    assert memory.calls == [
        (
            "capture",
            "The spare key is in the blue toolbox.",
            None,
            None,
            {"room": "hall"},
            MemoryType.SEMANTIC,
        ),
        ("settle", 10, 5, ("memory_1", "memory_2")),
        ("settle", 100, 3, None),
        ("pending_captures", 5, ("memory_1",)),
    ]


def test_identity_and_embodied_routes_are_off_by_default_and_404_not_403() -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory)) as client:
        speech = client.post("/v1/speech", json={"memory_id": "memory_1"})
        faces = client.post("/v1/faces", json={"memory_id": "memory_1"})
        registered = client.post(
            "/v1/identities", json={"identity_id": "identity_1", "name": "Ann"}
        )
        got = client.get("/v1/identities/identity_1")
        unlinked = client.post("/v1/identities/identity_1/unlink")
        forgotten = client.delete("/v1/identities/identity_1")
        openapi = client.get("/openapi.json").json()

    for response in (speech, faces, registered, got, unlinked, forgotten):
        assert response.status_code == 404
    assert "/v1/speech" not in openapi["paths"]
    assert "/v1/identities" not in openapi["paths"]
    assert memory.calls == []


def test_embodied_routes_dispatch_to_the_sdk_when_enabled() -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory, embodied_operations=True)) as client:
        speech = client.post("/v1/speech", json={"memory_id": "memory_1"})
        faces = client.post("/v1/faces", json={"memory_id": "memory_1"})

    assert speech.status_code == 200
    assert speech.json() == {
        "segments": [
            {
                "asset_id": "b" * 64,
                "start_ms": 0,
                "end_ms": 1500,
                "text": "Where is the toolbox?",
                "speaker_id": "speaker_1",
                "speaker_name": "Ann",
                "identity_score": 0.82,
            }
        ]
    }
    assert faces.status_code == 200
    assert faces.json()["observations"][0]["identity_id"] == "identity_1"
    assert memory.calls == [("speech", "memory_1"), ("faces", "memory_1")]


def test_answer_only_links_identities_when_embodied_operations_is_enabled() -> None:
    """A caller with recall access alone must not acquire merge authority through `/v1/answers`.

    `embodied_operations` gates `analyze_faces`, which commits the corroborated cross-modal
    identity merge; `ask` reaches the same merge through its own face recognition, so REST must
    pass the same switch through as `Memory.ask(..., link_identities=...)` rather than always
    defaulting it on.
    """
    memory = FakeMemory()
    with TestClient(create_app(memory=memory)) as client:
        client.post("/v1/answers", json={"question": "What color is it?"})
    assert memory.calls[-1][-1] is False

    memory = FakeMemory()
    with TestClient(create_app(memory=memory, embodied_operations=True)) as client:
        client.post("/v1/answers", json={"question": "What color is it?"})
    assert memory.calls[-1][-1] is True


def test_embodied_routes_map_memory_not_found() -> None:
    memory = FakeMemory(MemoryNotFoundError("memory is missing"))

    with TestClient(create_app(memory=memory, embodied_operations=True)) as client:
        response = client.post("/v1/speech", json={"memory_id": "memory_1"})

    assert response.status_code == 404
    assert response.json()["code"] == "memory_not_found"


def test_embodied_routes_map_unsupported_modality() -> None:
    memory = FakeMemory(ModelError("no face backend", reason="unsupported_modality"))

    with TestClient(create_app(memory=memory, embodied_operations=True)) as client:
        response = client.post("/v1/faces", json={"memory_id": "memory_1"})

    assert response.status_code == 422
    assert response.json()["reason"] == "unsupported_modality"


def test_identity_routes_dispatch_to_the_sdk_when_enabled() -> None:
    memory = FakeMemory()

    with TestClient(create_app(memory=memory, identity_operations=True)) as client:
        registered = client.post(
            "/v1/identities",
            json={"identity_id": "identity_1", "name": "Ann", "relationship": "daughter"},
        )
        got = client.get("/v1/identities/identity_1")
        unlinked = client.post("/v1/identities/identity_1/unlink")
        forgotten = client.delete("/v1/identities/identity_1")

    assert registered.status_code == 200
    assert registered.json() == {"registered": True}
    assert got.status_code == 200
    assert got.json()["identity"]["name"] == "Ann"
    assert unlinked.status_code == 200
    assert unlinked.json() == {"restored_identity_id": "identity_2"}
    assert forgotten.status_code == 200
    assert forgotten.json()["erasure"]["identity_id"] == "identity_1"
    assert memory.calls == [
        ("register_identity", "identity_1", "Ann", "daughter"),
        ("identity", "identity_1"),
        ("unlink_identity", "identity_1"),
        ("forget_identity", "identity_1"),
    ]


@pytest.mark.parametrize(
    ("name", "relationship"),
    [
        ("Alice Smith", "close friend"),
        ("\u674e\u96f7", "\u670b\u53cb"),
    ],
)
def test_register_identity_accepts_names_with_spaces_and_non_ascii_characters(
    name: str,
    relationship: str,
) -> None:
    """A person's name is plain text, not a memory-id-shaped token; REST must not reject either."""
    memory = FakeMemory()

    with TestClient(create_app(memory=memory, identity_operations=True)) as client:
        response = client.post(
            "/v1/identities",
            json={"identity_id": "identity_1", "name": name, "relationship": relationship},
        )

    assert response.status_code == 200
    assert response.json() == {"registered": True}
    # The SDK receives the name and relationship exactly as sent, unmodified.
    assert memory.calls == [("register_identity", "identity_1", name, relationship)]


def test_get_identity_reports_no_registered_profile_without_failing() -> None:
    memory = FakeMemory()
    memory.profile = None

    with TestClient(create_app(memory=memory, identity_operations=True)) as client:
        response = client.get("/v1/identities/unknown_id")

    assert response.status_code == 200
    assert response.json() == {"identity": None}


def test_identity_routes_map_identity_not_found() -> None:
    memory = FakeMemory(IdentityNotFoundError("identity does not exist: identity_1"))

    with TestClient(create_app(memory=memory, identity_operations=True)) as client:
        registered = client.post(
            "/v1/identities", json={"identity_id": "identity_1", "name": "Ann"}
        )
        forgotten = client.delete("/v1/identities/identity_1")

    assert registered.status_code == 404
    assert registered.json()["code"] == "identity_not_found"
    assert forgotten.status_code == 404
    assert forgotten.json()["code"] == "identity_not_found"


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


def _bundle(budget: ContextBudget, reference_at: datetime) -> ContextBundle:
    """Two populated sections, empty ones, one conflict, and one explicit unknown."""
    return ContextBundle(
        goal="What should I bring?",
        reference_at=reference_at,
        budget=budget,
        actors=(PROVISIONAL,),
        relationships=(),
        scene=(),
        episodes=(_media_hit(),),
        facts=(_hit(),),
        procedures=(),
        affect=(),
        traits=(),
        conflicts=(CONFLICT,),
        unknowns=(UNKNOWN,),
        occurred_from=OCCURRED_FROM,
        occurred_until=OCCURRED_UNTIL,
        frames=("home/map",),
        places=("kitchen",),
        omitted=3,
        chars=42,
        elapsed_ms=7,
        deadline_exceeded=False,
    )


def _media_hit() -> SearchHit:
    """A hit carrying a stored asset, whose local path must never be serialized."""
    return SearchHit(
        id="memory_2",
        content="The toolbox is blue.",
        score=0.7,
        modality=Modality.IMAGE,
        memory_type=MemoryType.EPISODIC,
        assets=(ASSET,),
        created_at=NOW,
    )


def _trace() -> RetrievalTrace:
    return RetrievalTrace(
        candidates=(
            RetrievalCandidateTrace(
                memory_id="memory_1",
                index_ids=("index_1",),
                dense_relevance=0.42,
                gate_relevance=0.31,
                final_score=0.4,
                rejected_by=RetrievalRejection.MINIMUM_RELEVANCE,
            ),
        ),
        candidate_limit=50,
        exhaustive=True,
        ambiguous=True,
    )


def _hit() -> SearchHit:
    return SearchHit(
        id="memory_1",
        content="The toolbox is blue.",
        score=0.9,
        modality=Modality.TEXT,
        created_at=NOW,
    )
