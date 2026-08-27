"""Minimal MCP tools backed by one local memory boundary."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, Literal, ParamSpec, TypeAlias, TypeVar, cast
from urllib.parse import urlsplit

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

from mindbridge import AsyncMemory, Memory
from mindbridge.exceptions import (
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
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
    SearchHit,
)

_LOGGER = logging.getLogger(__name__)
_MAX_INLINE_MEDIA_BYTES = 8 * 1024 * 1024
_Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=65_536),
]
_Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
_Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
_PartId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
_MediaType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        pattern=r"^[a-z0-9!#$&^_.+-]+/(?:\*|[a-z0-9!#$&^_.+-]+)$",
    ),
]
_Filename = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\\x00-\x1f\x7f]+$",
    ),
]
_Source = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_192)]
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_IDEMPOTENT_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
_DELETE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=True,
    open_world_hint=False,
)
_TOOL_ARGUMENTS = {
    "add_memory": frozenset({"content", "occurred_at", "metadata"}),
    "search_memories": frozenset({"query", "limit"}),
    "ask_memory": frozenset({"question", "limit"}),
    "get_memory": frozenset({"memory_id"}),
    "delete_memory": frozenset({"memory_id"}),
}
_STABLE_ERROR_CODES = frozenset(
    {
        "index_unavailable",
        "internal_error",
        "memory_not_found",
        "mindbridge_error",
        "model_error",
        "speaker_not_found",
        "storage_error",
        "validation_error",
    }
)

_P = ParamSpec("_P")
_T = TypeVar("_T")


class _InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _InputText(_InputModel):
    type: Literal["input_text"]
    text: _Text


class _InputImage(_InputModel):
    type: Literal["input_image"]
    image_url: _Source | None = None
    file_id: _PartId | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> _InputImage:
        _one_source(image_url=self.image_url, file_id=self.file_id)
        if self.image_url is not None:
            _validate_url_or_data(self.image_url)
            if self.image_url.startswith("data:"):
                media_type, _data = _decode_data_url(self.image_url)
                if not media_type.startswith("image/"):
                    raise ValueError("input_image data must have an image media type")
        return self


class _InputFile(_InputModel):
    type: Literal["input_file"]
    file_url: _Source | None = None
    file_data: str | None = None
    file_id: _PartId | None = None
    media_type: _MediaType | None = None
    filename: _Filename | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> _InputFile:
        _one_source(file_url=self.file_url, file_data=self.file_data, file_id=self.file_id)
        if self.media_type is not None and self.media_type.split("/", 1)[0] not in {
            "image",
            "video",
            "audio",
        }:
            raise ValueError("media_type must be image, video, or audio")
        if self.file_url is not None:
            _validate_url_or_data(self.file_url)
            if self.file_url.startswith("data:") and self.media_type is not None:
                embedded_type, _data = _decode_data_url(self.file_url)
                if not _media_type_matches(self.media_type, embedded_type):
                    raise ValueError("media_type contradicts the data URL")
        if self.file_data is not None:
            if self.media_type is None:
                raise ValueError("media_type is required with file_data")
            if self.media_type.endswith("/*"):
                raise ValueError("file_data requires a concrete media_type")
            _decode_base64(self.file_data)
        return self


_InputPart: TypeAlias = Annotated[
    _InputText | _InputImage | _InputFile,
    Field(discriminator="type"),
]
_Parts = Annotated[tuple[_InputPart, ...], Field(min_length=1, max_length=16)]
_Content: TypeAlias = _Text | _Parts


class AssetResult(BaseModel):
    id: str
    modality: Modality
    media_type: str
    size_bytes: int
    sha256: str
    name: str | None = None


class MemoryResult(BaseModel):
    id: str
    content: str
    modality: Modality
    assets: tuple[AssetResult, ...]
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None
    metadata: dict[str, JsonValue]


class SearchHitResult(MemoryResult):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class SearchResult(BaseModel):
    hits: tuple[SearchHitResult, ...]


class AnswerResponse(BaseModel):
    answer: str
    hits: tuple[SearchHitResult, ...]


class DeleteResult(BaseModel):
    deleted: bool


def build_mcp_server(memory: Memory | AsyncMemory) -> MCPServer[None]:
    """Expose five typed agent tools without taking ownership of ``memory``."""
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Fast local memory for agents.",
        version="0.2.0",
        middleware=[cast(ServerMiddleware[Any], _strict_tool_arguments)],
    )

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    async def add_memory(
        content: _Content,
        occurred_at: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> MemoryResult:
        """Store one memory and return its stable record."""
        record = cast(
            MemoryRecord,
            await _invoke(
                memory,
                "add",
                _content_input(content),
                occurred_at=occurred_at,
                metadata=metadata,
            ),
        )
        return _memory_result(record)

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    async def search_memories(query: _Content, limit: _Limit = 10) -> SearchResult:
        """Find the most relevant local memories."""
        hits = cast(
            tuple[SearchHit, ...],
            await _invoke(memory, "search", _content_input(query), limit=limit),
        )
        return SearchResult(hits=tuple(_search_hit_result(hit) for hit in hits))

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    async def ask_memory(question: _Content, limit: _Limit = 5) -> AnswerResponse:
        """Answer only from retrieved local memories."""
        result = cast(
            AnswerResult,
            await _invoke(memory, "ask", _content_input(question), limit=limit),
        )
        return AnswerResponse(
            answer=result.answer,
            hits=tuple(_search_hit_result(hit) for hit in result.hits),
        )

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    async def get_memory(memory_id: _Identifier) -> MemoryResult:
        """Read one memory by its stable identifier."""
        record = cast(MemoryRecord, await _invoke(memory, "get", memory_id))
        return _memory_result(record)

    @server.tool(annotations=_DELETE)
    @_stable_errors
    async def delete_memory(memory_id: _Identifier) -> DeleteResult:
        """Idempotently delete one memory."""
        deleted = cast(bool, await _invoke(memory, "delete", memory_id))
        return DeleteResult(deleted=deleted)

    return server


def run_mcp(data_dir: str | Path = ".mindbridge") -> None:
    """Run the local MCP server over stdio and close its owned memory."""
    memory = Memory(data_dir=data_dir)
    try:
        build_mcp_server(memory).run("stdio")
    finally:
        memory.close()


async def _invoke(
    memory: Memory | AsyncMemory,
    method: str,
    *args: object,
    **kwargs: object,
) -> object:
    operation = getattr(memory, method)
    if isinstance(memory, AsyncMemory):
        return await cast(Callable[..., Awaitable[object]], operation)(*args, **kwargs)
    return await asyncio.to_thread(cast(Callable[..., object], operation), *args, **kwargs)


def _stable_errors(
    operation: Callable[_P, Awaitable[_T]],
) -> Callable[_P, Awaitable[_T]]:
    @wraps(operation)
    async def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return await operation(*args, **kwargs)
        except Exception as error:
            raise ToolError(_error_json(error)) from None

    return guarded


async def _strict_tool_arguments(
    context: ServerRequestContext[Any, Any],
    call_next: CallNext,
) -> HandlerResult:
    params = context.params or {}
    if context.method == "tools/call":
        allowed = _TOOL_ARGUMENTS.get(str(params.get("name", "")))
        if allowed is None:
            return _error_result("validation_error", "tool does not exist")
        arguments = params.get("arguments")
        if isinstance(arguments, Mapping):
            unknown = set(arguments).difference(allowed)
            if unknown:
                return _error_result("validation_error", "tool arguments contain unknown fields")
    result = await call_next(context)
    if (
        context.method == "tools/call"
        and isinstance(result, dict)
        and result.get("isError") is True
        and not _has_error_code(result)
    ):
        return _error_result("validation_error", "tool arguments are invalid")
    return result


def _memory_result(record: MemoryRecord) -> MemoryResult:
    return MemoryResult(
        id=record.id,
        content=record.content,
        modality=record.modality,
        assets=tuple(_asset_result(asset) for asset in record.assets),
        created_at=record.created_at,
        occurred_at=record.occurred_at,
        metadata=cast(dict[str, JsonValue], dict(record.metadata)),
    )


def _search_hit_result(hit: SearchHit) -> SearchHitResult:
    return SearchHitResult(
        id=hit.id,
        content=hit.content,
        modality=hit.modality,
        assets=tuple(_asset_result(asset) for asset in hit.assets),
        score=hit.score,
        created_at=hit.created_at,
        occurred_at=hit.occurred_at,
        metadata=cast(dict[str, JsonValue], dict(hit.metadata)),
    )


def _asset_result(asset: AssetRef) -> AssetResult:
    if (
        asset.modality is None
        or asset.media_type is None
        or asset.size_bytes is None
        or asset.sha256 is None
    ):
        raise ValidationError("stored asset metadata is incomplete")
    return AssetResult(
        id=asset.id,
        modality=asset.modality,
        media_type=asset.media_type,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
        name=asset.name,
    )


def _content_input(content: _Content) -> ContentInput:
    if isinstance(content, str):
        return content
    atoms: list[str | URL | Blob | AssetRef] = []
    for part in content:
        if isinstance(part, _InputText):
            atoms.append(part.text)
        elif isinstance(part, _InputImage):
            if part.file_id is not None:
                atoms.append(AssetRef(id=part.file_id, modality=Modality.IMAGE))
            else:
                atoms.append(
                    _url_or_blob(cast(str, part.image_url), media_type="image/*", name=None)
                )
        elif part.file_id is not None:
            atoms.append(_file_reference(part.file_id, part.media_type))
        elif part.file_data is not None:
            atoms.append(
                Blob(
                    data=_decode_base64(part.file_data),
                    media_type=cast(str, part.media_type),
                    name=part.filename,
                )
            )
        else:
            atoms.append(
                _url_or_blob(
                    cast(str, part.file_url),
                    media_type=part.media_type,
                    name=part.filename,
                )
            )
    return tuple(atoms)


def _file_reference(file_id: str, media_type: str | None) -> AssetRef:
    if media_type is None:
        return AssetRef(id=file_id)
    if media_type.endswith("/*"):
        return AssetRef(id=file_id, modality=Modality(media_type.split("/", 1)[0]))
    return AssetRef(id=file_id, media_type=media_type)


def _url_or_blob(
    value: str,
    *,
    media_type: str | None = None,
    name: str | None,
) -> URL | Blob:
    if value.startswith("data:"):
        embedded_type, data = _decode_data_url(value)
        if media_type is not None and not _media_type_matches(media_type, embedded_type):
            raise ValueError("media_type contradicts the data URL")
        return Blob(data=data, media_type=embedded_type, name=name)
    return URL(value=value, media_type=media_type, name=name)


def _one_source(**sources: object) -> None:
    if sum(source is not None for source in sources.values()) != 1:
        raise ValueError(f"exactly one of {', '.join(sources)} is required")


def _media_type_matches(expected: str, actual: str) -> bool:
    return expected == actual or (
        expected.endswith("/*") and expected.split("/", 1)[0] == actual.split("/", 1)[0]
    )


def _validate_url_or_data(value: str) -> None:
    if value.startswith("data:"):
        _decode_data_url(value)
        return
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("source must be an HTTPS URL without credentials or a fragment")


def _decode_data_url(value: str) -> tuple[str, bytes]:
    header, separator, payload = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ValueError("data URL must contain base64 media bytes")
    media_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if not media_type or "/" not in media_type:
        raise ValueError("data URL must declare a media type")
    return media_type, _decode_base64(payload)


def _decode_base64(value: str) -> bytes:
    maximum_encoded = 4 * ((_MAX_INLINE_MEDIA_BYTES + 2) // 3)
    if len(value) > maximum_encoded:
        raise ValueError(f"inline media must not exceed {_MAX_INLINE_MEDIA_BYTES} bytes")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("file_data must be valid base64") from error
    if len(data) > _MAX_INLINE_MEDIA_BYTES:
        raise ValueError(f"inline media must not exceed {_MAX_INLINE_MEDIA_BYTES} bytes")
    return data


def _error_json(error: Exception) -> str:
    code, message = _error_details(error)
    return json.dumps({"code": code, "message": message}, separators=(",", ":"))


def _error_details(error: Exception) -> tuple[str, str]:
    if isinstance(error, ValidationError):
        return error.code, str(error) or "input is invalid"
    if isinstance(error, MemoryNotFoundError):
        return error.code, str(error) or "memory does not exist"
    if isinstance(error, ModelError):
        return error.code, "model operation failed"
    if isinstance(error, IndexUnavailableError):
        return error.code, "memory index is unavailable"
    if isinstance(error, StorageError):
        return error.code, "memory storage is unavailable"
    if isinstance(error, MindBridgeError):
        return error.code, "memory operation failed"
    _LOGGER.exception("unhandled MCP tool error")
    return "internal_error", "the memory operation failed unexpectedly"


def _error_result(code: str, message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"code": code, "message": message}))],
        is_error=True,
    )


def _has_error_code(result: Mapping[str, object]) -> bool:
    content = result.get("content")
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
            continue
        text = item["text"]
        try:
            envelope: object = json.loads(text[text.index("{") :])
        except (ValueError, json.JSONDecodeError):
            continue
        if (
            isinstance(envelope, dict)
            and set(envelope) == {"code", "message"}
            and envelope.get("code") in _STABLE_ERROR_CODES
            and isinstance(envelope.get("message"), str)
        ):
            return True
    return False
