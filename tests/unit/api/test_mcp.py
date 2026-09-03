"""Focused checks for the seven local-memory MCP tools."""

import base64
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mindbridge import Memory
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
    MemoryCapabilities,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    RetrievalScope,
    SearchHit,
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
CAPABILITIES = MemoryCapabilities(
    modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.AUDIO}),
    answer=True,
    transcribe=True,
    faces=False,
    describe_vision=False,
    form=True,
    consolidate=False,
    decay=True,
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
        self.forgotten_at: datetime | None = None

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

    def capabilities(self) -> MemoryCapabilities:
        return CAPABILITIES

    def get(self, memory_id: str) -> MemoryRecord:
        self._fail()
        self.calls.append(("get", memory_id))
        return _record(memory_id=memory_id, forgotten_at=self.forgotten_at)

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page:
        self._fail()
        self.calls.append(("list", limit, cursor))
        return Page(items=(_record(),), next_cursor="next")

    def delete(self, memory_id: str) -> bool:
        self._fail()
        self.calls.append(("delete", memory_id))
        return True

    def close(self) -> None:
        self.close_count += 1

    def _fail(self) -> None:
        if self.failure is not None:
            raise self.failure


async def test_mcp_publishes_only_the_seven_flat_local_tools() -> None:
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
        },
        "ask_memory": {"question", "limit", "memory_type", "reference_at", "scope"},
        "compile_context": {"goal", "budget", "reference_at", "scope"},
        "get_memory": {"memory_id"},
        "list_memories": {"limit", "cursor"},
        "delete_memory": {"memory_id"},
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
    assert tools["compile_context"].annotations is not None
    assert tools["compile_context"].annotations.read_only_hint is False
    description = tools["compile_context"].description or ""
    assert "Prefer this tool" in description
    assert "`ask_memory` remains a convenience" in description
    published = json.dumps({name: tool.input_schema for name, tool in tools.items()})
    assert all(field not in published for field in ("tenant_id", "user_id", "run_id"))


async def test_mcp_serializes_the_cognitive_forgetting_state_of_a_record() -> None:
    """Cognitive forgetting is auditable over the wire: `get_memory` still returns the record."""
    memory = FakeMemory()
    memory.forgotten_at = datetime(2026, 9, 3, 9, 0, tzinfo=timezone.utc)
    server = build_mcp_server(cast(Memory, memory))

    async with Client(server) as client:
        found = await client.call_tool("get_memory", {"memory_id": "memory_1"})

    assert found.structured_content is not None
    assert found.structured_content["forgotten_at"] == "2026-09-03T09:00:00Z"


async def test_mcp_instructions_advertise_the_configured_capability_view() -> None:
    async with Client(build_mcp_server(cast(Memory, FakeMemory()))) as client:
        instructions = client.instructions

    assert instructions is not None
    assert "Modalities: audio, image, text." in instructions
    assert "Capabilities: answer, decay, form, transcribe." in instructions
    assert "Unavailable: consolidate, describe_vision, faces." in instructions
    assert "Prefer compile_context for task-ready context" in instructions
    assert "cognitive forgetting" in instructions


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
    monkeypatch.setattr(mcp_adapter, "_MAX_INLINE_MEDIA_BYTES", 2)

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
    forgotten_at: datetime | None = None,
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
        forgotten_at=forgotten_at,
    )


def _hit(
    memory_id: str = "memory_1",
    *,
    score: float = 0.9,
    assets: tuple[AssetRef, ...] = (),
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> SearchHit:
    return SearchHit(
        id=memory_id,
        content="The toolbox is blue.",
        score=score,
        modality=Modality.IMAGE if assets else Modality.TEXT,
        memory_type=memory_type,
        assets=assets,
        created_at=NOW,
    )


def _bundle(budget: ContextBudget, reference_at: datetime) -> ContextBundle:
    return ContextBundle(
        goal="What should I bring?",
        reference_at=reference_at,
        budget=budget,
        actors=(),
        episodes=(_hit("memory_2", score=0.7, assets=(ASSET,), memory_type=MemoryType.EPISODIC),),
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


def _error_text(result: CallToolResult) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _error_envelope(result: CallToolResult) -> dict[str, object]:
    text = _error_text(result)
    return cast(dict[str, object], json.loads(text[text.index("{") :]))
