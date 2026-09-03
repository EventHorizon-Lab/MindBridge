"""Focused checks for the local-memory MCP tools."""

import base64
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mindbridge import Memory
from mindbridge.api import content
from mindbridge.api import mcp as mcp_adapter
from mindbridge.api.mcp import build_mcp_server
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
    ContextBudget,
    ContextBundle,
    ContextConflict,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryCapabilities,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
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
PROFILE = IdentityProfile(identity_id="identity_1", name="Ann", relationship="daughter")
ASSET = AssetRef(
    id="asset_image",
    modality=Modality.IMAGE,
    media_type="image/png",
    size_bytes=3,
    sha256="a" * 64,
    name="frame.png",
    path=Path("/private/mindbridge/assets/frame.png"),
)


CAPABILITIES = MemoryCapabilities(
    embedding=frozenset({Modality.TEXT, Modality.IMAGE}),
    embedding_model="jina-v5-omni",
    embedding_space="space_1",
    embedding_dimension=1024,
    generation=frozenset({Modality.TEXT}),
    generation_model="qwen3-omni",
    speaker_recognition=True,
)
CONFLICT = ContextConflict(
    lineage_id="lineage_1",
    subject="ana",
    predicate="location",
    values=("berlin", "paris"),
    memory_ids=("memory_1", "memory_2"),
)


class FakeMemory:
    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.close_count = 0
        self.failure = failure
        self.profile: IdentityProfile | None = PROFILE
        self.restored: str | None = "identity_3"

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
            content=content if isinstance(content, str) else "Multimodal memory.",
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=copied_metadata,
            modality=Modality.TEXT if isinstance(content, str) else Modality.IMAGE,
            assets=() if isinstance(content, str) else (ASSET,),
            memory_type=memory_type,
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

    @property
    def capabilities(self) -> MemoryCapabilities:
        return CAPABILITIES

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
    ) -> AnswerResult:
        self._fail()
        self.calls.append(("ask", question, limit, memory_type, reference_at))
        return AnswerResult(answer="The toolbox is blue.", hits=(_hit(),))

    def get(self, memory_id: str) -> MemoryRecord:
        self._fail()
        self.calls.append(("get", memory_id))
        return _record(memory_id=memory_id)

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        self._fail()
        self.calls.append(("list", limit, cursor))
        return Page(items=(_record(),), next_cursor="next")

    def delete(self, memory_id: str) -> bool:
        self._fail()
        self.calls.append(("delete", memory_id))
        return True

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]:
        self._fail()
        self.calls.append(("speech", memory_id))
        return (SEGMENT,)

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]:
        self._fail()
        self.calls.append(("faces", memory_id))
        return (FACE,)

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None:
        self._fail()
        self.calls.append(("register_speaker", speaker_id, name, relationship))

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


async def test_mcp_publishes_only_the_flat_local_tools() -> None:
    server = build_mcp_server(cast(Memory, FakeMemory()))

    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {
        "add_memory",
        "search_memories",
        "ask_memory",
        "compile_context",
        "get_memory",
        "list_memories",
        "delete_memory",
        "analyze_speech",
        "analyze_faces",
        "register_speaker",
        "register_identity",
        "get_identity",
        "unlink_identity",
        "forget_identity",
        "reinforce_memories",
    }
    assert {name: set(tool.input_schema["properties"]) for name, tool in tools.items()} == {
        "add_memory": {
            "content",
            "occurred_at",
            "occurred_end",
            "metadata",
            "memory_type",
            "context",
        },
        "search_memories": {
            "query",
            "limit",
            "memory_type",
            "reference_at",
            "occurred_from",
            "occurred_until",
            "scope",
            "explain",
        },
        "ask_memory": {"question", "limit", "memory_type", "reference_at", "scope"},
        "compile_context": {"goal", "budget", "reference_at", "scope"},
        "get_memory": {"memory_id"},
        "list_memories": {"limit", "cursor"},
        "delete_memory": {"memory_id"},
        "analyze_speech": {"memory_id"},
        "analyze_faces": {"memory_id"},
        "register_speaker": {"speaker_id", "name", "relationship"},
        "register_identity": {"identity_id", "name", "relationship"},
        "get_identity": {"identity_id"},
        "unlink_identity": {"alias_id"},
        "forget_identity": {"identity_id"},
        "reinforce_memories": {"memory_ids"},
    }
    assert tools["get_memory"].annotations is not None
    assert tools["get_memory"].annotations.read_only_hint is True
    assert tools["list_memories"].annotations is not None
    assert tools["list_memories"].annotations.read_only_hint is True
    assert tools["search_memories"].annotations is not None
    assert tools["search_memories"].annotations.read_only_hint is False
    assert tools["ask_memory"].annotations is not None
    assert tools["ask_memory"].annotations.read_only_hint is False
    assert tools["delete_memory"].annotations is not None
    assert tools["delete_memory"].annotations.destructive_hint is True
    # Compiling reads only, but may cache a transcript, so it cannot claim to be read-only.
    assert tools["compile_context"].annotations is not None
    assert tools["compile_context"].annotations.read_only_hint is False
    assert tools["get_identity"].annotations is not None
    assert tools["get_identity"].annotations.read_only_hint is True
    assert tools["register_speaker"].annotations is not None
    assert tools["register_speaker"].annotations.idempotent_hint is True
    # Reversing a merge discards the alias's evidence, and reinforcement accumulates, so neither
    # may claim to be a safe no-op on retry.
    assert tools["unlink_identity"].annotations is not None
    assert tools["unlink_identity"].annotations.destructive_hint is True
    assert tools["reinforce_memories"].annotations is not None
    assert tools["reinforce_memories"].annotations.read_only_hint is False
    assert tools["reinforce_memories"].annotations.idempotent_hint is False
    # Analysis persists identity evidence, so it cannot be advertised read-only.
    for analysis in ("analyze_speech", "analyze_faces"):
        annotations = tools[analysis].annotations
        assert annotations is not None
        assert annotations.read_only_hint is False
    published = json.dumps({name: tool.input_schema for name, tool in tools.items()})
    assert all(field not in published for field in ("tenant_id", "user_id", "run_id"))


async def test_mcp_returns_structured_results_and_does_not_close_injected_memory() -> None:
    memory = FakeMemory()
    server = build_mcp_server(cast(Memory, memory))

    async with Client(server) as client:
        added = await client.call_tool(
            "add_memory",
            {
                "content": "  The toolbox is blue.  ",
                "occurred_at": NOW.isoformat(),
                "metadata": {"room": "workshop"},
                "memory_type": "episodic",
            },
        )
        searched = await client.call_tool(
            "search_memories",
            {
                "query": " toolbox ",
                "memory_type": "episodic",
                "reference_at": NOW.isoformat(),
                "occurred_from": OCCURRED_FROM.isoformat(),
                "occurred_until": OCCURRED_UNTIL.isoformat(),
            },
        )
        answered = await client.call_tool(
            "ask_memory",
            {
                "question": " What color? ",
                "memory_type": "procedural",
                "reference_at": NOW.isoformat(),
            },
        )
        found = await client.call_tool("get_memory", {"memory_id": "memory_1"})
        page = await client.call_tool("list_memories", {"limit": 7, "cursor": "cursor_1"})
        deleted = await client.call_tool("delete_memory", {"memory_id": "memory_1"})

    assert added.is_error is False
    assert added.structured_content == {
        "id": "memory_1",
        "content": "The toolbox is blue.",
        "modality": "text",
        "memory_type": "episodic",
        "assets": [],
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "occurred_at": NOW.isoformat().replace("+00:00", "Z"),
        "occurred_end": None,
        "metadata": {"room": "workshop"},
        "context": None,
        "place_id": None,
        "forgotten_at": None,
    }
    assert searched.structured_content is not None
    assert searched.structured_content["hits"][0]["score"] == 0.9
    assert answered.structured_content is not None
    assert answered.structured_content["answer"] == "The toolbox is blue."
    assert answered.structured_content["abstained"] is False
    assert answered.structured_content["abstention_reason"] is None
    assert found.structured_content is not None
    assert found.structured_content["id"] == "memory_1"
    assert page.structured_content is not None
    assert page.structured_content["items"][0]["id"] == "memory_1"
    assert page.structured_content["next_cursor"] == "next"
    assert deleted.structured_content == {"deleted": True}
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
            "search",
            "toolbox",
            10,
            MemoryType.EPISODIC,
            NOW,
            OCCURRED_FROM,
            OCCURRED_UNTIL,
        ),
        ("ask", "What color?", 5, MemoryType.PROCEDURAL, NOW),
        ("get", "memory_1"),
        ("list", 7, "cursor_1"),
        ("delete", "memory_1"),
    ]
    assert memory.close_count == 0


async def test_mcp_dispatches_the_embodied_and_identity_operations() -> None:
    """An agent driving a robot can ask who spoke and who was seen, and name them."""
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        speech = await client.call_tool("analyze_speech", {"memory_id": "memory_1"})
        faces = await client.call_tool("analyze_faces", {"memory_id": "memory_1"})
        named_speaker = await client.call_tool(
            "register_speaker",
            {"speaker_id": "speaker_1", "name": "Ann", "relationship": "daughter"},
        )
        named_identity = await client.call_tool(
            "register_identity",
            {"identity_id": "identity_1", "name": "Ann"},
        )
        identity = await client.call_tool("get_identity", {"identity_id": "identity_1"})
        unlinked = await client.call_tool("unlink_identity", {"alias_id": "identity_3"})
        reinforced = await client.call_tool(
            "reinforce_memories",
            {"memory_ids": ["memory_1", "memory_2", "memory_1"]},
        )

    assert speech.structured_content == {
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
    assert faces.structured_content == {
        "observations": [
            {
                "asset_id": "c" * 64,
                "bounding_box": [0.1, 0.2, 0.3, 0.4],
                "identity_id": "identity_1",
                "identity_name": "Ann",
                "identity_score": 0.77,
                "observed_at_ms": 250,
            }
        ]
    }
    assert named_speaker.structured_content == {"registered": True}
    assert named_identity.structured_content == {"registered": True}
    assert identity.structured_content == {
        "identity": {
            "identity_id": "identity_1",
            "name": "Ann",
            "relationship": "daughter",
        }
    }
    assert unlinked.structured_content == {"restored_identity_id": "identity_3"}
    assert reinforced.structured_content == {"reinforced": 2}
    assert memory.calls == [
        ("speech", "memory_1"),
        ("faces", "memory_1"),
        ("register_speaker", "speaker_1", "Ann", "daughter"),
        # An omitted relationship stays omitted rather than clearing the recorded one.
        ("register_identity", "identity_1", "Ann", None),
        ("identity", "identity_1"),
        ("unlink_identity", "identity_3"),
        ("reinforce", ("memory_1", "memory_2", "memory_1")),
    ]


async def test_mcp_reports_an_absent_identity_and_an_irreversible_merge_structurally() -> None:
    """The two "nothing to report" answers are typed nulls, not prose an agent must read."""
    memory = FakeMemory()
    memory.profile = None
    memory.restored = None

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        identity = await client.call_tool("get_identity", {"identity_id": "identity_9"})
        unlinked = await client.call_tool("unlink_identity", {"alias_id": "identity_9"})

    assert identity.is_error is False
    assert identity.structured_content == {"identity": None}
    assert unlinked.is_error is False
    assert unlinked.structured_content == {"restored_identity_id": None}


async def test_mcp_erases_a_person_and_reports_what_was_destroyed() -> None:
    """ "Forget me" is an agent-facing request wherever "who was that" is."""
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        forgotten = await client.call_tool("forget_identity", {"identity_id": "identity_1"})
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert forgotten.structured_content == {
        "erasure": {
            "identity_id": "identity_1",
            "alias_ids": ["identity_4"],
            "face_exemplars": 3,
            "voice_exemplars": 2,
            "face_observations": 7,
            "speech_segments": 11,
        }
    }
    assert memory.calls == [("forget_identity", "identity_1")]
    # A second call reports the person as unknown, so this is destructive and not idempotent.
    annotations = tools["forget_identity"].annotations
    assert annotations is not None
    assert annotations.destructive_hint is True
    assert annotations.idempotent_hint is False


async def test_reinforce_bounds_its_input_before_memory() -> None:
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        empty = await client.call_tool("reinforce_memories", {"memory_ids": []})
        oversize = await client.call_tool(
            "reinforce_memories",
            {"memory_ids": [f"memory_{index}" for index in range(101)]},
        )

    assert _error_envelope(empty)["code"] == "validation_error"
    assert _error_envelope(oversize)["code"] == "validation_error"
    assert memory.calls == []


async def test_mcp_maps_ordered_openai_parts_and_returns_safe_asset_metadata() -> None:
    memory = FakeMemory()
    audio_data = base64.b64encode(b"wav").decode()
    image_data = base64.b64encode(b"png").decode()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        result = await client.call_tool(
            "add_memory",
            {
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

    assert result.is_error is False
    added = memory.calls[0][1]
    assert isinstance(added, tuple)
    assert added == (
        "At the station.",
        Blob(b"png", "image/png"),
        Blob(b"wav", "audio/wav", "note.wav"),
        AssetRef(id="asset_existing", media_type="video/mp4"),
        AssetRef(id="asset_existing_image", modality=Modality.IMAGE),
    )
    assert result.structured_content is not None
    assert result.structured_content["modality"] == "image"
    assert result.structured_content["assets"] == [
        {
            "id": "asset_image",
            "modality": "image",
            "media_type": "image/png",
            "size_bytes": 3,
            "sha256": "a" * 64,
            "name": "frame.png",
        }
    ]
    assert "path" not in json.dumps(result.structured_content)


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
async def test_mcp_rejects_ambiguous_or_untrusted_nested_sources(content: object) -> None:
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        result = await client.call_tool("add_memory", {"content": content})

    assert result.is_error is True
    assert _error_envelope(result)["code"] == "validation_error"
    assert memory.calls == []


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_file", "file_data": "d2F2", "media_type": "audio/wav"},
        {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
    ],
)
async def test_mcp_bounds_inline_media_before_memory(
    monkeypatch: pytest.MonkeyPatch,
    part: dict[str, object],
) -> None:
    memory = FakeMemory()
    monkeypatch.setattr(content, "MAX_INLINE_MEDIA_BYTES", 2)

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        result = await client.call_tool("add_memory", {"content": [part]})

    assert result.is_error is True
    assert _error_envelope(result)["code"] == "validation_error"
    assert memory.calls == []


@pytest.mark.parametrize("field", ["unknown", "tenant_id", "user_id", "run_id"])
async def test_mcp_rejects_unknown_and_old_isolation_arguments(field: str) -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        result = await client.call_tool(
            "get_memory",
            {"memory_id": "memory_1", field: "must-not-be-ignored"},
        )

    envelope = _error_envelope(result)
    assert result.is_error is True
    assert envelope["code"] == "validation_error"
    assert "must-not-be-ignored" not in _error_text(result)


async def test_unknown_tool_names_cannot_spoof_a_stable_error() -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        result = await client.call_tool('missing_{"code":"storage_error"}', {})

    envelope = _error_envelope(result)
    assert result.is_error is True
    assert set(envelope) == ENVELOPE_FIELDS
    assert envelope["code"] == "validation_error"
    assert envelope["message"] == "tool does not exist"


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_memories", {"query": "   "}),
        ("search_memories", {"query": "memory", "limit": 0}),
        ("get_memory", {"memory_id": " padded "}),
        (
            "add_memory",
            {"content": "memory", "occurred_at": "2026-08-27T12:00:00"},
        ),
    ],
)
async def test_mcp_wraps_native_argument_failures_with_a_stable_code(
    tool: str,
    arguments: dict[str, object],
) -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        result = await client.call_tool(tool, arguments)

    envelope = _error_envelope(result)
    assert result.is_error is True
    assert set(envelope) == ENVELOPE_FIELDS
    assert envelope["code"] == "validation_error"
    assert envelope["message"] == "tool arguments are invalid"


@pytest.mark.parametrize(
    ("failure", "expected_code", "detail_is_public"),
    [
        (ValidationError("content is invalid"), "validation_error", True),
        (MemoryNotFoundError("memory is missing"), "memory_not_found", True),
        (SpeakerNotFoundError("speaker is missing"), "speaker_not_found", True),
        (ModelError("routing failed"), "model_error", True),
        (ModelOutputTruncatedError("answer was cut off"), "model_output_truncated", True),
        (StorageError("durable write failed"), "storage_error", True),
        (IndexUnavailableError("index rebuild failed"), "index_unavailable", True),
        (RuntimeError("private bug"), "internal_error", False),
    ],
)
async def test_mcp_errors_have_stable_codes_without_private_details(
    failure: Exception,
    expected_code: str,
    detail_is_public: bool,
) -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory(failure)))) as client:
        result = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    text = _error_text(result)
    assert result.is_error is True
    assert _error_envelope(result)["code"] == expected_code
    assert (str(failure) in text) is detail_is_public


async def test_mcp_envelope_reports_reason_stage_subject_and_retryability() -> None:
    failure = ModelError(
        "provider is busy",
        reason="rate_limited",
        stage="embed",
        subject="asset_image",
    )
    async with Client(build_mcp_server(cast(Memory, FakeMemory(failure)))) as client:
        result = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    envelope = _error_envelope(result)
    assert set(envelope) == ENVELOPE_FIELDS
    assert envelope["code"] == "model_error"
    assert envelope["reason"] == "rate_limited"
    assert envelope["retryable"] is True
    assert envelope["stage"] == "embed"
    assert envelope["subject"] == "asset_image"
    assert cast(str, envelope["trace_id"]).startswith("trace_")


async def test_mcp_names_the_unknown_argument_instead_of_only_rejecting_the_call() -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        result = await client.call_tool(
            "get_memory", {"memory_id": "memory_1", "run_id": "private-value"}
        )

    envelope = _error_envelope(result)
    assert envelope["code"] == "validation_error"
    assert envelope["reason"] == "unknown_field"
    assert envelope["issues"] == [
        {
            "location": ["arguments", "run_id"],
            "message": "Extra inputs are not permitted",
            "type": "extra_forbidden",
        }
    ]
    assert "private-value" not in _error_text(result)


async def test_mcp_never_serializes_a_provider_exception_behind_a_model_error() -> None:
    provider_failure = RuntimeError("sk-live-provider-secret")
    failure = ModelError("embedding request failed", reason="auth_failed", stage="embed")
    failure.__cause__ = provider_failure
    async with Client(build_mcp_server(cast(Memory, FakeMemory(failure)))) as client:
        result = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    assert "sk-live-provider-secret" not in _error_text(result)
    assert _error_envelope(result)["reason"] == "auth_failed"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (MemoryNotFoundError("memory is missing"), "memory_not_found"),
        (ModelError("provider is busy", reason="rate_limited"), "model_error"),
        (StorageError("durable write failed"), "storage_error"),
        (IndexUnavailableError("index rebuild failed"), "index_unavailable"),
        (RuntimeError("private bug"), "internal_error"),
    ],
)
async def test_every_failure_text_is_a_bare_json_envelope(
    failure: Exception,
    expected_code: str,
) -> None:
    """The recoverable codes are exactly the ones an agent must parse to decide to retry."""
    async with Client(build_mcp_server(cast(Memory, FakeMemory(failure)))) as client:
        result = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    text = _error_text(result)
    envelope = json.loads(text)
    assert result.is_error is True
    assert text.lstrip().startswith("{")
    assert set(envelope) == ENVELOPE_FIELDS
    assert envelope["code"] == expected_code


async def test_middleware_and_tool_failures_share_one_parse() -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory(StorageError("no disk"))))) as (
        client
    ):
        from_middleware = await client.call_tool(
            "get_memory", {"memory_id": "memory_1", "run_id": "x"}
        )
        from_tool = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    assert json.loads(_error_text(from_middleware))["code"] == "validation_error"
    assert json.loads(_error_text(from_tool))["code"] == "storage_error"


async def test_search_keeps_its_default_shape_and_explains_an_empty_result() -> None:
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        plain = await client.call_tool("search_memories", {"query": "toolbox"})
        explained = await client.call_tool("search_memories", {"query": "toolbox", "explain": True})

    assert plain.structured_content is not None
    assert plain.structured_content["hits"][0]["id"] == "memory_1"
    assert plain.structured_content["trace"] is None
    assert explained.structured_content is not None
    assert explained.structured_content["hits"] == []
    trace = explained.structured_content["trace"]
    assert trace is not None
    assert trace["ambiguous"] is True
    assert trace["candidate_limit"] == 50
    assert trace["exhaustive"] is True
    assert trace["candidates"] == [
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


async def test_reinforce_reaches_the_sdk_and_deduplicates_ids() -> None:
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        result = await client.call_tool(
            "reinforce_memories", {"memory_ids": ["memory_1", "memory_1", "memory_2"]}
        )
        empty = await client.call_tool("reinforce_memories", {"memory_ids": []})

    assert result.structured_content == {"reinforced": 2}
    assert memory.calls == [("reinforce", ("memory_1", "memory_1", "memory_2"))]
    assert empty.is_error is True
    assert _error_envelope(empty)["code"] == "validation_error"


async def test_the_compile_tool_returns_the_whole_bundle_without_local_asset_paths() -> None:
    memory = FakeMemory()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        compiled = await client.call_tool(
            "compile_context",
            {
                "goal": "  What should I bring?  ",
                "budget": {
                    "max_chars": 2_000,
                    "max_items": 8,
                    "memory_types": ["episodic", "semantic"],
                    "min_confidence": 0.5,
                    "freshness_seconds": 3_600,
                },
                "reference_at": NOW.isoformat(),
            },
        )
        defaulted = await client.call_tool("compile_context", {"goal": "What should I bring?"})

    assert compiled.is_error is False
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
            ),
            NOW,
            None,
        ),
        ("compile", "What should I bring?", None, None, None),
    ]
    bundle = compiled.structured_content
    assert bundle is not None
    assert bundle["goal"] == "What should I bring?"
    assert bundle["budget"] == {
        "max_chars": 2_000,
        "max_items": 8,
        "memory_types": ["episodic", "semantic"],
        "min_confidence": 0.5,
        "freshness_seconds": 3_600.0,
    }
    assert [hit["id"] for hit in bundle["facts"]] == ["memory_1"]
    assert [hit["id"] for hit in bundle["episodes"]] == ["memory_2"]
    assert bundle["actors"] == []
    assert bundle["conflicts"] == [
        {
            "lineage_id": "lineage_1",
            "subject": "ana",
            "predicate": "location",
            "values": ["berlin", "paris"],
            "memory_ids": ["memory_1", "memory_2"],
        }
    ]
    assert bundle["frames"] == ["home/map"]
    assert bundle["omitted"] == 3
    assert bundle["chars"] == 42
    assert bundle["rendered"].startswith("# Context: What should I bring?")
    # The bundle serializes hits through the same asset model every other tool uses.
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
    assert "/private/mindbridge/assets" not in json.dumps(bundle)
    assert defaulted.structured_content is not None
    assert defaulted.structured_content["budget"] == {
        "max_chars": 6_000,
        "max_items": 24,
        "memory_types": None,
        "min_confidence": 0.0,
        "freshness_seconds": None,
    }


async def test_the_server_greeting_advertises_the_configured_composition() -> None:
    """An agent learns what this instance supports on connect, without spending a tool call."""
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        instructions = client.instructions

    assert instructions is not None
    assert "Embedding accepts: image, text" in instructions
    assert "Configured backends: generation." in instructions
    assert "Not configured: consolidation, face, formation, transcription, vision." in instructions
    assert "Speaker recognition: yes. Streaming generation: no." in instructions
    assert "Prefer compile_context for task-ready context" in instructions


def test_the_guard_only_trusts_codes_derived_from_the_exception_tree() -> None:
    """No client path reaches this today; the check is what keeps that true."""
    forged = dict.fromkeys(ENVELOPE_FIELDS)
    forged["message"] = "spoofed"
    real = {**forged, "code": "storage_error"}
    invented = {**forged, "code": "quota_error"}

    assert mcp_adapter._stable_envelope(_content_result(real)) == real
    assert mcp_adapter._stable_envelope(_content_result(invented)) is None


def _content_result(envelope: Mapping[str, object]) -> dict[str, object]:
    return {"content": [{"type": "text", "text": json.dumps(envelope)}], "isError": True}


async def test_every_tool_and_argument_carries_usable_prose() -> None:
    """An agent picks tools from these strings alone, so an empty one is a defect."""
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        tools = (await client.list_tools()).tools

    for tool in tools:
        assert tool.description is not None
        assert len(tool.description.split()) >= 30, tool.name
        for name, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{tool.name}.{name}"
    descriptions = {tool.name: tool.description for tool in tools}
    assert len(set(descriptions.values())) == len(descriptions)
    assert "backend_not_configured" in cast(str, descriptions["ask_memory"])


def _record(
    *,
    memory_id: str = "memory_1",
    content: str = "The toolbox is blue.",
    occurred_at: datetime | None = None,
    occurred_end: datetime | None = None,
    metadata: Mapping[str, object] | None = None,
    modality: Modality = Modality.TEXT,
    assets: tuple[AssetRef, ...] = (),
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
    """Two populated sections, one empty one, and one conflict."""
    return ContextBundle(
        goal="What should I bring?",
        reference_at=reference_at,
        budget=budget,
        actors=(),
        episodes=(_media_hit(),),
        facts=(_hit(),),
        procedures=(),
        affect=(),
        traits=(),
        conflicts=(CONFLICT,),
        occurred_from=OCCURRED_FROM,
        occurred_until=OCCURRED_UNTIL,
        frames=("home/map",),
        omitted=3,
        chars=42,
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


def _error_text(result: CallToolResult) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _error_envelope(result: CallToolResult) -> dict[str, object]:
    # A bare `json.loads` is the contract: no prefix to strip on any failure shape.
    return cast(dict[str, object], json.loads(_error_text(result)))
