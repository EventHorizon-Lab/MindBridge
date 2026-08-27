"""Focused checks for the five local-memory MCP tools."""

import base64
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from mindbridge import AsyncMemory, Memory
from mindbridge.api import mcp as mcp_adapter
from mindbridge.api.mcp import build_mcp_server, run_mcp
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
    MemoryType,
    Modality,
    SearchHit,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)
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
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> MemoryRecord:
        self._fail()
        copied_metadata = dict(metadata or {})
        self.calls.append(("add", content, occurred_at, copied_metadata, memory_type))
        return _record(
            content=content if isinstance(content, str) else "Multimodal memory.",
            occurred_at=occurred_at,
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
        return _record(memory_id=memory_id)

    def delete(self, memory_id: str) -> bool:
        self._fail()
        self.calls.append(("delete", memory_id))
        return True

    def close(self) -> None:
        self.close_count += 1

    def _fail(self) -> None:
        if self.failure is not None:
            raise self.failure


async def test_mcp_publishes_only_the_five_flat_local_tools() -> None:
    server = build_mcp_server(cast(Memory, FakeMemory()))

    async with Client(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == {
        "add_memory",
        "search_memories",
        "ask_memory",
        "get_memory",
        "delete_memory",
    }
    assert {name: set(tool.input_schema["properties"]) for name, tool in tools.items()} == {
        "add_memory": {"content", "occurred_at", "metadata", "memory_type"},
        "search_memories": {"query", "limit", "memory_type", "reference_at"},
        "ask_memory": {"question", "limit", "memory_type", "reference_at"},
        "get_memory": {"memory_id"},
        "delete_memory": {"memory_id"},
    }
    assert tools["get_memory"].annotations is not None
    assert tools["get_memory"].annotations.read_only_hint is True
    assert tools["search_memories"].annotations is not None
    assert tools["search_memories"].annotations.read_only_hint is False
    assert tools["ask_memory"].annotations is not None
    assert tools["ask_memory"].annotations.read_only_hint is False
    assert tools["delete_memory"].annotations is not None
    assert tools["delete_memory"].annotations.destructive_hint is True
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
        "metadata": {"room": "workshop"},
    }
    assert searched.structured_content is not None
    assert searched.structured_content["hits"][0]["score"] == 0.9
    assert answered.structured_content is not None
    assert answered.structured_content["answer"] == "The toolbox is blue."
    assert found.structured_content is not None
    assert found.structured_content["id"] == "memory_1"
    assert deleted.structured_content == {"deleted": True}
    assert memory.calls == [
        ("add", "The toolbox is blue.", NOW, {"room": "workshop"}, MemoryType.EPISODIC),
        ("search", "toolbox", 10, MemoryType.EPISODIC, NOW),
        ("ask", "What color?", 5, MemoryType.PROCEDURAL, NOW),
        ("get", "memory_1"),
        ("delete", "memory_1"),
    ]
    assert memory.close_count == 0


async def test_mcp_maps_ordered_openai_parts_and_returns_safe_asset_metadata() -> None:
    memory = FakeMemory()
    audio_data = base64.b64encode(b"wav").decode()

    async with Client(build_mcp_server(cast(Memory, memory))) as client:
        result = await client.call_tool(
            "add_memory",
            {
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

    assert result.is_error is False
    added = memory.calls[0][1]
    assert isinstance(added, tuple)
    assert added == (
        "At the station.",
        URL("https://media.example/frame.png", "image/*"),
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


async def test_mcp_accepts_the_public_async_memory_facade() -> None:
    class FakeAsyncMemory(AsyncMemory):
        def __init__(self) -> None:
            self.memory_ids: list[str] = []

        async def get(self, memory_id: str) -> MemoryRecord:
            self.memory_ids.append(memory_id)
            return _record(memory_id=memory_id)

    memory = FakeAsyncMemory()

    async with Client(build_mcp_server(memory)) as client:
        result = await client.call_tool("get_memory", {"memory_id": "async_memory"})

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["id"] == "async_memory"
    assert memory.memory_ids == ["async_memory"]


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

    assert result.is_error is True
    assert _error_envelope(result) == {
        "code": "validation_error",
        "message": "tool does not exist",
    }


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

    assert result.is_error is True
    assert _error_envelope(result) == {
        "code": "validation_error",
        "message": "tool arguments are invalid",
    }


@pytest.mark.parametrize(
    ("failure", "expected_code", "detail_is_public"),
    [
        (ValidationError("content is invalid"), "validation_error", True),
        (MemoryNotFoundError("memory is missing"), "memory_not_found", True),
        (ModelError("provider secret"), "model_error", False),
        (StorageError("database path"), "storage_error", False),
        (IndexUnavailableError("native detail"), "index_unavailable", False),
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


def test_run_mcp_closes_its_owned_memory_even_when_stdio_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    memory = FakeMemory()

    class StoppedServer:
        def run(self, transport: str) -> None:
            assert transport == "stdio"
            raise RuntimeError("stdio stopped")

    def create_memory(*, data_dir: str | Path) -> FakeMemory:
        assert Path(data_dir) == tmp_path
        return memory

    monkeypatch.setattr(mcp_adapter, "Memory", create_memory)
    monkeypatch.setattr(mcp_adapter, "build_mcp_server", lambda _memory: StoppedServer())

    with pytest.raises(RuntimeError, match="stdio stopped"):
        run_mcp(tmp_path)

    assert memory.close_count == 1


def _record(
    *,
    memory_id: str = "memory_1",
    content: str = "The toolbox is blue.",
    occurred_at: datetime | None = None,
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


def _error_text(result: CallToolResult) -> str:
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _error_envelope(result: CallToolResult) -> dict[str, object]:
    text = _error_text(result)
    return cast(dict[str, object], json.loads(text[text.index("{") :]))
