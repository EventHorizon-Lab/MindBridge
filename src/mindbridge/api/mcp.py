"""Minimal MCP tools backed by one local memory boundary."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from functools import wraps
from typing import Annotated, Any, Literal, ParamSpec, TypeAlias, TypeVar, cast
from uuid import uuid4

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

from mindbridge import Memory
from mindbridge.exceptions import (
    IdentityNotFoundError,
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)
from mindbridge.types import (
    AbstentionReason,
    AssetRef,
    Blob,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextConflict,
    MemoryCapabilities,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    RetrievalScope,
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
_Cursor = Annotated[str, StringConstraints(min_length=1)]
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
_Chars = Annotated[int, Field(strict=True, ge=1, le=65_536)]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
_Seconds = Annotated[float, Field(gt=0.0)]
# Tool defaults are read from the SDK value so the advertised budget cannot drift.
_BUDGET = ContextBudget()
_CAPABILITY_FLAGS: tuple[str, ...] = (
    "answer",
    "transcribe",
    "faces",
    "describe_vision",
    "form",
    "consolidate",
    "decay",
)
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_RETRIEVAL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=False,
)
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
    "add_memory": frozenset(
        {"content", "occurred_at", "occurred_end", "metadata", "memory_type", "context"}
    ),
    "search_memories": frozenset(
        {
            "query",
            "limit",
            "memory_type",
            "reference_at",
            "occurred_from",
            "occurred_until",
            "scope",
        }
    ),
    "ask_memory": frozenset({"question", "limit", "memory_type", "reference_at", "scope"}),
    "compile_context": frozenset({"goal", "budget", "reference_at", "scope"}),
    "get_memory": frozenset({"memory_id"}),
    "list_memories": frozenset({"limit", "cursor"}),
    "delete_memory": frozenset({"memory_id"}),
}
# The error envelope MCP shares with REST.
_ERROR_FIELDS = frozenset(
    {"code", "reason", "retryable", "stage", "subject", "message", "trace_id", "issues"}
)


def _error_codes(root: type[MindBridgeError]) -> frozenset[str]:
    codes = {root.code}
    for subclass in root.__subclasses__():
        codes |= _error_codes(subclass)
    return frozenset(codes)


# Derived rather than listed. A hand-maintained set silently loses every new exception class to the
# middleware's `validation_error` overwrite, which is how `model_output_truncated` was destroyed.
_STABLE_ERROR_CODES = _error_codes(MindBridgeError) | {"internal_error"}

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
            _validate_data_url(self.image_url)
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
            _validate_data_url(self.file_url)
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


class ContextBudgetInput(_InputModel):
    max_chars: _Chars = _BUDGET.max_chars
    max_items: _Limit = _BUDGET.max_items
    memory_types: Annotated[list[MemoryType], Field(min_length=1)] | None = None
    min_confidence: _Confidence = _BUDGET.min_confidence
    freshness_seconds: _Seconds | None = None


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
    memory_type: MemoryType
    assets: tuple[AssetResult, ...]
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None
    occurred_end: AwareDatetime | None
    metadata: dict[str, JsonValue]
    context: MemoryContext | None = None
    forgotten_at: AwareDatetime | None = None


class SearchHitResult(MemoryResult):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class PageResult(BaseModel):
    items: tuple[MemoryResult, ...]
    next_cursor: str | None = None


class SearchResult(BaseModel):
    hits: tuple[SearchHitResult, ...]


class AnswerResponse(BaseModel):
    answer: str
    hits: tuple[SearchHitResult, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class DeleteResult(BaseModel):
    deleted: bool


class ContextBudgetResult(BaseModel):
    max_chars: int
    max_items: int
    memory_types: tuple[MemoryType, ...] | None
    min_confidence: float
    freshness_seconds: float | None


class ContextConflictResult(BaseModel):
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


class ContextBundleResult(BaseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResult
    actors: tuple[SearchHitResult, ...]
    episodes: tuple[SearchHitResult, ...]
    facts: tuple[SearchHitResult, ...]
    procedures: tuple[SearchHitResult, ...]
    affect: tuple[SearchHitResult, ...]
    traits: tuple[SearchHitResult, ...]
    conflicts: tuple[ContextConflictResult, ...]
    occurred_from: AwareDatetime | None
    occurred_until: AwareDatetime | None
    frames: tuple[str, ...]
    omitted: int
    chars: int
    rendered: str


def build_mcp_server(memory: Memory) -> MCPServer[None]:
    """Expose seven typed agent tools without taking ownership of ``memory``."""
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Fast local memory for agents.",
        # Read once at construction: an agent learns the configured capability view at connect
        # time instead of spending a tool call or discovering a missing backend by failing.
        instructions=_instructions(memory.capabilities()),
        version="0.2.0",
        middleware=[cast(ServerMiddleware[Any], _strict_tool_arguments)],
    )

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def add_memory(
        content: _Content,
        occurred_at: AwareDatetime | None = None,
        occurred_end: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryResult:
        """Store one memory and return its stable record."""
        record = memory.add(
            _content_input(content),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )
        return _memory_result(record)

    @server.tool(annotations=_RETRIEVAL)
    @_stable_errors
    def search_memories(
        query: _Content,
        limit: _Limit = 10,
        memory_type: MemoryType | None = None,
        reference_at: AwareDatetime | None = None,
        occurred_from: AwareDatetime | None = None,
        occurred_until: AwareDatetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> SearchResult:
        """Find the most relevant local memories."""
        hits = memory.search(
            _content_input(query),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )
        return SearchResult(hits=tuple(_search_hit_result(hit) for hit in hits))

    @server.tool(annotations=_RETRIEVAL)
    @_stable_errors
    def ask_memory(
        question: _Content,
        limit: _Limit = 5,
        memory_type: MemoryType | None = None,
        reference_at: AwareDatetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> AnswerResponse:
        """Answer only from retrieved local memories."""
        result = memory.ask(
            _content_input(question),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            scope=scope,
        )
        return AnswerResponse(
            answer=result.answer,
            hits=tuple(_search_hit_result(hit) for hit in result.hits),
            abstained=result.abstained,
            abstention_reason=result.abstention_reason,
        )

    @server.tool(annotations=_RETRIEVAL)
    @_stable_errors
    def compile_context(
        goal: _Content,
        budget: ContextBudgetInput | None = None,
        reference_at: AwareDatetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> ContextBundleResult:
        """Compile bounded, task-ready context for a goal.

        Prefer this tool when you need context to act on: it returns actors, facts, episodes,
        procedures, affect cues, traits, unresolved conflicts, and provenance inside one declared
        budget, plus a rendered text view. It selects and structures evidence and never resolves a
        conflict or writes memory. `ask_memory` remains a convenience for one grounded sentence.
        """
        return _bundle_result(
            memory.compile(
                _content_input(goal),
                budget=_context_budget(budget),
                reference_at=reference_at,
                scope=scope,
            )
        )

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def get_memory(memory_id: _Identifier) -> MemoryResult:
        """Read one memory by its stable identifier."""
        return _memory_result(memory.get(memory_id))

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def list_memories(limit: _Limit = 100, cursor: _Cursor | None = None) -> PageResult:
        """List newest memories through an opaque stable cursor."""
        page = memory.list(limit=limit, cursor=cursor)
        return PageResult(
            items=tuple(_memory_result(record) for record in page.items),
            next_cursor=page.next_cursor,
        )

    @server.tool(annotations=_DELETE)
    @_stable_errors
    def delete_memory(memory_id: _Identifier) -> DeleteResult:
        """Idempotently delete one memory."""
        return DeleteResult(deleted=memory.delete(memory_id))

    return server


def _stable_errors(
    operation: Callable[_P, _T],
) -> Callable[_P, _T]:
    @wraps(operation)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
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
            return _error_result("validation_error", "tool does not exist", reason="unknown_field")
        arguments = params.get("arguments")
        if isinstance(arguments, Mapping):
            unknown = set(arguments).difference(allowed)
            if unknown:
                return _error_result(
                    "validation_error",
                    "tool arguments contain unknown fields",
                    reason="unknown_field",
                    issues=[
                        {
                            "location": ["arguments", name],
                            "message": "Extra inputs are not permitted",
                            "type": "extra_forbidden",
                        }
                        for name in sorted(unknown)
                    ],
                )
    result = await call_next(context)
    if (
        context.method == "tools/call"
        and isinstance(result, dict)
        and result.get("isError") is True
        and not _has_error_code(result)
    ):
        return _error_result(
            "validation_error", "tool arguments are invalid", reason="input_invalid"
        )
    return result


def _memory_result(record: MemoryRecord) -> MemoryResult:
    return MemoryResult(
        id=record.id,
        content=record.content,
        modality=record.modality,
        memory_type=record.memory_type,
        assets=tuple(_asset_result(asset) for asset in record.assets),
        created_at=record.created_at,
        occurred_at=record.occurred_at,
        occurred_end=record.occurred_end,
        metadata=cast(dict[str, JsonValue], dict(record.metadata)),
        context=record.context,
        forgotten_at=record.forgotten_at,
    )


def _search_hit_result(hit: SearchHit) -> SearchHitResult:
    return SearchHitResult(
        id=hit.id,
        content=hit.content,
        modality=hit.modality,
        memory_type=hit.memory_type,
        assets=tuple(_asset_result(asset) for asset in hit.assets),
        score=hit.score,
        created_at=hit.created_at,
        occurred_at=hit.occurred_at,
        occurred_end=hit.occurred_end,
        metadata=cast(dict[str, JsonValue], dict(hit.metadata)),
        context=hit.context,
        forgotten_at=hit.forgotten_at,
    )


def _context_budget(budget: ContextBudgetInput | None) -> ContextBudget | None:
    if budget is None:
        return None
    return ContextBudget(
        max_chars=budget.max_chars,
        max_items=budget.max_items,
        memory_types=None if budget.memory_types is None else frozenset(budget.memory_types),
        min_confidence=budget.min_confidence,
        freshness=(
            None
            if budget.freshness_seconds is None
            else timedelta(seconds=budget.freshness_seconds)
        ),
    )


def _bundle_result(bundle: ContextBundle) -> ContextBundleResult:
    return ContextBundleResult(
        goal=bundle.goal,
        reference_at=bundle.reference_at,
        budget=_budget_result(bundle.budget),
        actors=tuple(_search_hit_result(hit) for hit in bundle.actors),
        episodes=tuple(_search_hit_result(hit) for hit in bundle.episodes),
        facts=tuple(_search_hit_result(hit) for hit in bundle.facts),
        procedures=tuple(_search_hit_result(hit) for hit in bundle.procedures),
        affect=tuple(_search_hit_result(hit) for hit in bundle.affect),
        traits=tuple(_search_hit_result(hit) for hit in bundle.traits),
        conflicts=tuple(_conflict_result(conflict) for conflict in bundle.conflicts),
        occurred_from=bundle.occurred_from,
        occurred_until=bundle.occurred_until,
        frames=bundle.frames,
        omitted=bundle.omitted,
        chars=bundle.chars,
        rendered=bundle.render(),
    )


def _budget_result(budget: ContextBudget) -> ContextBudgetResult:
    return ContextBudgetResult(
        max_chars=budget.max_chars,
        max_items=budget.max_items,
        memory_types=None if budget.memory_types is None else tuple(sorted(budget.memory_types)),
        min_confidence=budget.min_confidence,
        freshness_seconds=(None if budget.freshness is None else budget.freshness.total_seconds()),
    )


def _conflict_result(conflict: ContextConflict) -> ContextConflictResult:
    return ContextConflictResult(
        lineage_id=conflict.lineage_id,
        subject=conflict.subject,
        predicate=conflict.predicate,
        values=conflict.values,
        memory_ids=conflict.memory_ids,
    )


def _instructions(capabilities: MemoryCapabilities) -> str:
    """Render the configured capability view an agent reads when it connects."""
    available = [name for name in _CAPABILITY_FLAGS if getattr(capabilities, name)]
    missing = [name for name in _CAPABILITY_FLAGS if not getattr(capabilities, name)]
    return "\n".join(
        (
            "MindBridge is local multimodal memory for one physical data directory.",
            f"Modalities: {_names([modality.value for modality in capabilities.modalities])}.",
            f"Capabilities: {_names(available)}.",
            f"Unavailable: {_names(missing)}.",
            "Prefer compile_context for task-ready context; ask_memory remains a convenience for "
            "one grounded sentence.",
            "Identity naming, cognitive forgetting, consolidation, and operation rollback have no "
            "tool here; they stay with the process that owns this memory.",
        )
    )


def _names(values: Sequence[str]) -> str:
    return ", ".join(sorted(values)) or "none"


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
    atoms: list[str | Blob | AssetRef] = []
    for part in content:
        if isinstance(part, _InputText):
            atoms.append(part.text)
        elif isinstance(part, _InputImage):
            if part.file_id is not None:
                atoms.append(AssetRef(id=part.file_id, modality=Modality.IMAGE))
            else:
                atoms.append(_data_blob(cast(str, part.image_url), media_type="image/*", name=None))
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
                _data_blob(
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


def _data_blob(
    value: str,
    *,
    media_type: str | None = None,
    name: str | None,
) -> Blob:
    embedded_type, data = _decode_data_url(value)
    if media_type is not None and not _media_type_matches(media_type, embedded_type):
        raise ValueError("media_type contradicts the data URL")
    return Blob(data=data, media_type=embedded_type, name=name)


def _one_source(**sources: object) -> None:
    if sum(source is not None for source in sources.values()) != 1:
        raise ValueError(f"exactly one of {', '.join(sources)} is required")


def _media_type_matches(expected: str, actual: str) -> bool:
    return expected == actual or (
        expected.endswith("/*") and expected.split("/", 1)[0] == actual.split("/", 1)[0]
    )


def _validate_data_url(value: str) -> None:
    if not value.startswith("data:"):
        raise ValueError("remote URLs are not accepted; fetch media before calling MindBridge")
    _decode_data_url(value)


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
    return json.dumps(_tool_error(error), separators=(",", ":"))


def _tool_error(error: Exception) -> dict[str, object]:
    if isinstance(error, MindBridgeError):
        return _envelope(
            error.code,
            _error_message(error),
            reason=error.reason,
            retryable=error.retryable,
            stage=error.stage,
            # MCP runs in the owner process as the same user, so a subject naming local state is
            # already visible to the caller. Unauthenticated REST withholds the same value.
            subject=error.subject,
        )
    _LOGGER.exception("unhandled MCP tool error")
    return _envelope(
        "internal_error",
        "the memory operation failed unexpectedly",
        reason="unexpected",
    )


def _error_message(error: MindBridgeError) -> str:
    if isinstance(error, ValidationError):
        return str(error) or "input is invalid"
    if isinstance(error, MemoryNotFoundError):
        return str(error) or "memory does not exist"
    if isinstance(error, SpeakerNotFoundError):
        return str(error) or "speaker does not exist"
    if isinstance(error, IdentityNotFoundError):
        return str(error) or "identity does not exist"
    if isinstance(error, ModelError):
        return str(error) or "model operation failed"
    if isinstance(error, IndexUnavailableError):
        return str(error) or "memory index is unavailable"
    if isinstance(error, StorageError):
        return str(error) or "memory storage is unavailable"
    # An unmapped public error is a MindBridge bug, not caller-actionable detail; REST says the
    # same thing with HTTP 500.
    return "memory operation failed"


def _envelope(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    retryable: bool = False,
    stage: str | None = None,
    subject: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "code": code,
        "reason": reason,
        "retryable": retryable,
        "stage": stage,
        "subject": subject,
        "message": message,
        "trace_id": f"trace_{uuid4().hex}",
        "issues": list(issues),
    }


def _error_result(
    code: str,
    message: str,
    *,
    reason: str | None = None,
    issues: Sequence[Mapping[str, object]] = (),
) -> CallToolResult:
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(_envelope(code, message, reason=reason, issues=issues)),
            )
        ],
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
            and set(envelope) == _ERROR_FIELDS
            and envelope.get("code") in _STABLE_ERROR_CODES
            and isinstance(envelope.get("message"), str)
        ):
            return True
    return False
