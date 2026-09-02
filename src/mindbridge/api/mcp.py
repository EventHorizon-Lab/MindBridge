"""The typed MCP tool surface over one local memory boundary."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Annotated, Any, ParamSpec, TypeVar, cast
from uuid import uuid4

from mcp.server import MCPServer
from mcp.server.context import CallNext, HandlerResult, ServerMiddleware, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import AwareDatetime, BaseModel, Field, JsonValue, StringConstraints

from mindbridge import Memory
from mindbridge.api.content import Content, content_input
from mindbridge.api.errors import error_message
from mindbridge.exceptions import MindBridgeError, ValidationError
from mindbridge.types import (
    AbstentionReason,
    AssetRef,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    RetrievalScope,
    SearchHit,
    SpeakerSegment,
)

_LOGGER = logging.getLogger(__name__)
_Identifier = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
_Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
_Cursor = Annotated[str, StringConstraints(min_length=1)]
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
_NON_IDEMPOTENT_WRITE = ToolAnnotations(
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
# Erasure is destructive and, unlike a delete, cannot be repeated: the second call reports the
# person as unknown, so a retry after a dropped response is not the same call.
_ERASE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=True,
    idempotent_hint=False,
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
    "get_memory": frozenset({"memory_id"}),
    "list_memories": frozenset({"limit", "cursor"}),
    "delete_memory": frozenset({"memory_id"}),
    "analyze_speech": frozenset({"memory_id"}),
    "analyze_faces": frozenset({"memory_id"}),
    "register_speaker": frozenset({"speaker_id", "name", "relationship"}),
    "register_identity": frozenset({"identity_id", "name", "relationship"}),
    "get_identity": frozenset({"identity_id"}),
    "unlink_identity": frozenset({"alias_id"}),
    "forget_identity": frozenset({"identity_id"}),
    "reinforce_memories": frozenset({"memory_ids"}),
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
    place_id: str | None = None


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


# The embodied and identity results carry the public dataclasses directly. Pydantic derives their
# schema from the same definition the SDK returns, so a field cannot drift between the two.
# `forget_identity` wraps rather than returning `IdentityErasure` directly because the MCP SDK
# cannot build a schema for a slotted dataclass as a top-level return type: it reads the slot
# descriptors as defaults, rejects them, and silently publishes no output schema at all.
class SpeechResult(BaseModel):
    segments: tuple[SpeakerSegment, ...]


class FacesResult(BaseModel):
    observations: tuple[FaceObservation, ...]


class IdentityResult(BaseModel):
    identity: IdentityProfile | None


class RegisterResult(BaseModel):
    registered: bool


class UnlinkResult(BaseModel):
    restored_identity_id: str | None


class ForgetResult(BaseModel):
    erasure: IdentityErasure


class ReinforceResult(BaseModel):
    reinforced: int


def build_mcp_server(memory: Memory) -> MCPServer[None]:
    """Expose the agent tool surface without taking ownership of ``memory``.

    Every tool is one call on the injected ``Memory``: the embodied and identity tools are
    reachable here for the same reason the common-path tools are, because this server runs in the
    process that holds it.
    """
    server: MCPServer[None] = MCPServer(
        "mindbridge",
        title="MindBridge Memory",
        description="Fast local memory for agents.",
        version="0.2.0",
        middleware=[cast(ServerMiddleware[Any], _strict_tool_arguments)],
    )

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def add_memory(
        content: Content,
        occurred_at: AwareDatetime | None = None,
        occurred_end: AwareDatetime | None = None,
        metadata: dict[str, JsonValue] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryResult:
        """Store one memory and return its stable record."""
        record = memory.add(
            content_input(content),
            occurred_at=occurred_at,
            occurred_end=occurred_end,
            metadata=metadata,
            memory_type=memory_type,
            context=context,
        )
        return _memory_result(record)

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def search_memories(
        query: Content,
        limit: _Limit = 10,
        memory_type: MemoryType | None = None,
        reference_at: AwareDatetime | None = None,
        occurred_from: AwareDatetime | None = None,
        occurred_until: AwareDatetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> SearchResult:
        """Find the most relevant local memories."""
        hits = memory.search(
            content_input(query),
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )
        return SearchResult(hits=tuple(_search_hit_result(hit) for hit in hits))

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def ask_memory(
        question: Content,
        limit: _Limit = 5,
        memory_type: MemoryType | None = None,
        reference_at: AwareDatetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> AnswerResponse:
        """Answer only from retrieved local memories."""
        result = memory.ask(
            content_input(question),
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

    _register_embodied_tools(server, memory)
    return server


def _register_embodied_tools(server: MCPServer[None], memory: Memory) -> None:
    """Register the embodied and identity tools on an existing server.

    Split from `build_mcp_server` only to keep each registration function readable; the two sets
    dispatch to the same injected memory.
    """

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def analyze_speech(memory_id: _Identifier) -> SpeechResult:
        """Transcribe one stored memory's audio and video, and resolve who spoke.

        Call this to learn what was said in a memory whose content is speech, or which person
        said it. Segments are returned per asset in time order and cover only this one memory's
        audio and video assets; there is no page cursor because the bound is the stored media. A
        memory with no audio or video returns no segments rather than failing.

        Side effects: caches the transcript and records the voice as identity evidence, so an
        identical retry is safe but may resolve a speaker an earlier call left unnamed.
        `speaker_id` is stable and accepted by `register_speaker` and `get_identity`. Fails with
        `model_error` when the configured transcription backend cannot recognize speakers.
        """
        return SpeechResult(segments=memory.speech(memory_id))

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def analyze_faces(memory_id: _Identifier) -> FacesResult:
        """Detect the faces in one stored memory's images and video, and resolve who was seen.

        Call this to learn which people appear in a visual memory. Observations cover only this
        one memory's image and video assets, each with a frame-normalized bounding box and, for
        video, the offset it was observed at; there is no page cursor because the bound is the
        stored media. A memory with no image or video returns no observations rather than failing.

        Side effects: records the face as identity evidence and may bind it to a voice already
        seen with it, so an identical retry is safe but may resolve an identity an earlier call
        left unnamed; `unlink_identity` reverses a binding. `identity_id` is stable and accepted
        by `register_identity` and `get_identity`. Fails with `model_error` when no face backend
        is configured or when it does not support the stored modality.
        """
        return FacesResult(observations=memory.faces(memory_id))

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def register_speaker(
        speaker_id: _Identifier,
        name: _Identifier,
        relationship: _Identifier | None = None,
    ) -> RegisterResult:
        """Name the person behind one `speaker_id` from `analyze_speech`.

        Call this once a conversation establishes who a recognized voice belongs to. Repeating it
        replaces the name, so retrying is safe; omitting `relationship` leaves any recorded
        relationship intact rather than clearing it. Naming a speaker also rewrites the stored
        memories that quote them, so later searches can find the person by name. Fails with
        `speaker_not_found` when the ID is not a recognized speaker.
        """
        memory.register_speaker(speaker_id, name, relationship=relationship)
        return RegisterResult(registered=True)

    @server.tool(annotations=_IDEMPOTENT_WRITE)
    @_stable_errors
    def register_identity(
        identity_id: _Identifier,
        name: _Identifier,
        relationship: _Identifier | None = None,
    ) -> RegisterResult:
        """Name the person behind one `identity_id` from `analyze_faces` or `analyze_speech`.

        Call this for an identity that may have been seen as well as heard; use
        `register_speaker` only for a voice-only speaker ID. Repeating it replaces the name, so
        retrying is safe; omitting `relationship` leaves any recorded relationship intact rather
        than clearing it. Fails with `identity_not_found` when the ID is not a recognized
        identity.
        """
        memory.register_identity(identity_id, name, relationship=relationship)
        return RegisterResult(registered=True)

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def get_identity(identity_id: _Identifier) -> IdentityResult:
        """Read the name and relationship recorded for one identity or speaker ID.

        Call this to turn an `identity_id` or `speaker_id` from a face or speech result into who
        the person is. Merge aliases are followed, so an ID that was bound into another identity
        resolves to that person. Returns `identity: null` when the ID exists but nothing has been
        registered for it; reads nothing else and changes nothing.
        """
        return IdentityResult(identity=memory.identity(identity_id))

    @server.tool(annotations=_DELETE)
    @_stable_errors
    def unlink_identity(alias_id: _Identifier) -> UnlinkResult:
        """Reverse one face-and-voice merge, restoring `alias_id` as its own identity.

        Call this when a person says the system has confused them with someone else. Returns the
        restored identity ID, or `restored_identity_id: null` when the merge cannot be reversed
        because no record names which modality was contributed, and repeating the call is safe
        because a reversed merge stays reversed.

        Side effects: the restored identity keeps no name or relationship and the pair's
        accumulated evidence is reset. This does not suppress the pair: if the same voice and face
        keep appearing together they will be corroborated and merged again.
        """
        return UnlinkResult(restored_identity_id=memory.unlink_identity(alias_id))

    @server.tool(annotations=_ERASE)
    @_stable_errors
    def forget_identity(identity_id: _Identifier) -> ForgetResult:
        """Erase one person: their biometric template, their aliases, and their indexed name.

        Call this when a person asks to be forgotten, and `delete_memory` to forget an event
        instead. Memories, their content and their media survive: a transcript keeps its words
        with the speaker attribution dropped, because forgetting a person is not forgetting the
        evening. The result counts what was destroyed, as the audit record for the request.

        Side effects: this destroys the face and voice templates and cannot be undone, so a later
        encounter mints a fresh unnamed identity for the same person rather than recognizing them
        as forgotten. Retrying is not safe in the sense that matters here: the second call fails
        with `identity_not_found` because the person is already gone, which is not evidence that
        the first call failed.
        """
        return ForgetResult(erasure=memory.forget_identity(identity_id))

    @server.tool(annotations=_NON_IDEMPOTENT_WRITE)
    @_stable_errors
    def reinforce_memories(
        memory_ids: Annotated[tuple[_Identifier, ...], Field(min_length=1, max_length=100)],
    ) -> ReinforceResult:
        """Record that these memories were actually useful, so retrieval favors them later.

        Call this after using retrieved memories to answer or act, with the IDs that were used.
        Duplicate IDs count once and unknown IDs are skipped; `reinforced` reports how many
        existing memories were updated, so a count below the number of IDs sent means the rest do
        not exist.

        Side effects: this is cumulative positive feedback, not a flag, so retrying the same call
        reinforces the same memories again. Send at most 100 IDs per call.
        """
        return ReinforceResult(reinforced=memory.reinforce(memory_ids))


def _stable_errors(
    operation: Callable[_P, _T],
) -> Callable[_P, _T]:
    @wraps(operation)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            raise ToolError(_error_json(error)) from None

    # The MCP SDK publishes `__doc__` verbatim, and CPython 3.13 strips a docstring's common
    # indentation at compile time while earlier versions keep it. Without this, the published
    # tool description -- which is public contract -- would differ by interpreter.
    guarded.__doc__ = None if operation.__doc__ is None else inspect.cleandoc(operation.__doc__)
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


def _error_json(error: Exception) -> str:
    return json.dumps(_tool_error(error), separators=(",", ":"))


def _tool_error(error: Exception) -> dict[str, object]:
    if isinstance(error, MindBridgeError):
        return _envelope(
            error.code,
            error_message(error),
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
