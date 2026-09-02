"""Minimal MCP tools backed by one local memory boundary."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Callable, Mapping, Sequence
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
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    RetrievalScope,
    RetrievalTrace,
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
# Reinforcement accumulates: each call raises `access_count` and moves the ranking factor, so a
# repeated call is not the same as one call and the honest hint is `idempotent_hint=False`.
_COUNTED_WRITE = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
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
            "explain",
        }
    ),
    "ask_memory": frozenset({"question", "limit", "memory_type", "reference_at", "scope"}),
    "get_memory": frozenset({"memory_id"}),
    "list_memories": frozenset({"limit", "cursor"}),
    "delete_memory": frozenset({"memory_id"}),
    "reinforce_memories": frozenset({"memory_ids"}),
}
# Prose the six tool schemas share. MCP publishes these verbatim, so an agent reads them before it
# guesses at frames, units, or media sources.
_CONTENT_DESCRIPTION = (
    "The memory content: a non-blank string, or 1 through 16 ordered parts (`input_text`,"
    " `input_image`, `input_file`). Media must arrive as a base64 `data:` URL or as a `file_id`"
    " already stored in this data directory; remote URLs, local paths, and `file:` URLs are"
    " rejected, so fetch media before calling."
)
_QUERY_DESCRIPTION = (
    "What to look for: a non-blank string, or 1 through 16 ordered parts for a mixed-modal query"
    " (an image plus words). Media follows the same base64 `data:` URL or stored `file_id` rule as"
    " `add_memory`."
)
_MEMORY_TYPE_FILTER_DESCRIPTION = (
    "Restrict results to `semantic`, `episodic`, or `procedural` memories. Null considers every"
    " type; a wrong guess here is a common cause of an empty result."
)
_REFERENCE_AT_DESCRIPTION = (
    "The timezone-aware instant that relative language in the query ('yesterday', 'last week')"
    " is resolved against, and the instant recency is scored from. Defaults to now."
)
_SCOPE_DESCRIPTION = (
    "Optional retrieval filter. `valid_at` selects world validity and `known_at` selects the"
    " transaction version known at that time. `near` and `radius_m` must be supplied together:"
    " `near` is a pose in a named coordinate frame (`frame_id`, `anchor` of observer or subject,"
    " metres for `x`/`y`/`z`, `orientation_xyzw` as a unit quaternion) and `radius_m` is metres."
    " Spatial filtering only matches memories stored with a spatial context in the same frame and"
    " anchor, so it excludes every plain text memory."
)
_MEMORY_ID_DESCRIPTION = (
    "The `id` a previous `add_memory`, `search_memories`, or `list_memories` result returned."
)
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


class SearchHitResult(MemoryResult):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class PageResult(BaseModel):
    items: tuple[MemoryResult, ...]
    next_cursor: str | None = None


class SearchResult(BaseModel):
    hits: tuple[SearchHitResult, ...]
    # Null unless the call asked for it, so the default result keeps the shape agents already read.
    trace: RetrievalTrace | None = None


class AnswerResponse(BaseModel):
    answer: str
    hits: tuple[SearchHitResult, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class DeleteResult(BaseModel):
    deleted: bool


class ReinforceResult(BaseModel):
    reinforced: int


def build_mcp_server(memory: Memory) -> MCPServer[None]:
    """Expose seven typed agent tools without taking ownership of ``memory``."""
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
        content: Annotated[_Content, Field(description=_CONTENT_DESCRIPTION)],
        occurred_at: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "When the remembered event happened, as a timezone-aware timestamp. Omit for a"
                    " timeless fact; without it the memory never matches a search occurrence"
                    " window."
                )
            ),
        ] = None,
        occurred_end: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "End of the remembered interval, timezone-aware. Requires `occurred_at` and"
                    " must be later than it."
                )
            ),
        ] = None,
        metadata: Annotated[
            dict[str, JsonValue] | None,
            Field(
                description=(
                    "Application-owned JSON stored verbatim with the record, at most 262,144 UTF-8"
                    " bytes serialized. It is not searched, and it is not an access-control or"
                    " isolation boundary."
                )
            ),
        ] = None,
        memory_type: Annotated[
            MemoryType,
            Field(
                description=(
                    "`semantic` for a standing fact, `episodic` for something that happened,"
                    " `procedural` for how to do something. Search can filter on it."
                )
            ),
        ] = MemoryType.SEMANTIC,
        context: Annotated[
            ObservationContext | None,
            Field(
                description=(
                    "Typed provenance for this observation: basis, source ID, confidence, validity"
                    " window, and optional spatial pose. Omit it unless the caller actually knows"
                    " these; `scope` filters at search time only match memories that carry them."
                )
            ),
        ] = None,
    ) -> MemoryResult:
        """Store one memory and return its stable record.

        Call this to persist something the agent should be able to recall in a later session.
        Writes durable local state and may call the configured embedding backend. Storage is
        content-addressed, so re-adding identical content returns the existing record instead of a
        duplicate and retrying a call whose response was lost is safe.
        """
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
        query: Annotated[_Content, Field(description=_QUERY_DESCRIPTION)],
        limit: Annotated[
            _Limit,
            Field(
                description=(
                    "Maximum hits to return, 1 through 100. It caps the result; it cannot raise a"
                    " candidate above the relevance gate, so it never turns an empty result into a"
                    " non-empty one."
                )
            ),
        ] = 10,
        memory_type: Annotated[
            MemoryType | None, Field(description=_MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=_REFERENCE_AT_DESCRIPTION)
        ] = None,
        occurred_from: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "Keep only memories whose event interval overlaps at or after this"
                    " timezone-aware instant. Memories stored without `occurred_at` never match."
                )
            ),
        ] = None,
        occurred_until: Annotated[
            AwareDatetime | None,
            Field(
                description=(
                    "Exclusive upper bound of the same half-open overlap filter; with"
                    " `occurred_from` it must be strictly later."
                )
            ),
        ] = None,
        scope: Annotated[RetrievalScope | None, Field(description=_SCOPE_DESCRIPTION)] = None,
        explain: Annotated[
            bool,
            Field(
                description=(
                    "Also return `trace`: every candidate considered, with its score components"
                    " and, when it did not become a hit, the `rejected_by` reason"
                    " (`minimum_relevance`, `ambiguity`, `memory_type`, `occurrence_range`,"
                    " `missing_memory`, `stale_index`, or `limit`). Ask for it when `hits` is"
                    " empty or shorter than expected; `hits` itself is identical either way."
                )
            ),
        ] = False,
    ) -> SearchResult:
        """Rank stored memories against a query and return the matching records.

        Call this to find out what is stored before answering from your own context, or to gather
        evidence you will read yourself; use `ask_memory` when you want MindBridge to write the
        answer. Returns at most `limit` hits, best first, and never invents one: empty `hits` means
        nothing cleared the relevance gate, and `explain=true` reports which stage discarded each
        candidate. Stores no memory and is safe to retry, though it may cache a transcript for
        media it had to transcribe.
        """
        content = _content_input(query)
        if not explain:
            hits = memory.search(
                content,
                limit=limit,
                memory_type=memory_type,
                reference_at=reference_at,
                occurred_from=occurred_from,
                occurred_until=occurred_until,
                scope=scope,
            )
            return SearchResult(hits=tuple(_search_hit_result(hit) for hit in hits))
        traced = memory.search_with_trace(
            content,
            limit=limit,
            memory_type=memory_type,
            reference_at=reference_at,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            scope=scope,
        )
        return SearchResult(
            hits=tuple(_search_hit_result(hit) for hit in traced.hits),
            trace=traced.trace,
        )

    @server.tool(annotations=_RETRIEVAL)
    @_stable_errors
    def ask_memory(
        question: Annotated[
            _Content,
            Field(
                description=(
                    "The question to answer, as a non-blank string or 1 through 16 ordered parts."
                    " It is also the retrieval query, so phrase it the way the memory would be"
                    " worded."
                )
            ),
        ],
        limit: Annotated[
            _Limit,
            Field(
                description=(
                    "Maximum memories read as evidence, 1 through 100. More evidence costs more"
                    " model input and can be truncated by the configured evidence budget."
                )
            ),
        ] = 5,
        memory_type: Annotated[
            MemoryType | None, Field(description=_MEMORY_TYPE_FILTER_DESCRIPTION)
        ] = None,
        reference_at: Annotated[
            AwareDatetime | None, Field(description=_REFERENCE_AT_DESCRIPTION)
        ] = None,
        scope: Annotated[RetrievalScope | None, Field(description=_SCOPE_DESCRIPTION)] = None,
    ) -> AnswerResponse:
        """Answer a question using only the memories retrieved for it.

        Requires a generation backend in the host process: without one every call fails with
        `model_error/backend_not_configured`, so use `search_memories` if you are not sure one is
        configured. Returns the answer with the hits it used; `abstained` is true when the answerer
        reported no usable evidence, and the answer is then a fixed sentence rather than a guess.
        Stores no memory, but each call spends another generation, so retry only on a retryable
        error.
        """
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

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def get_memory(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> MemoryResult:
        """Read one memory by its stable identifier.

        Call this to re-read a record whose ID you already hold, not to discover memories. Fails
        with `memory_not_found` when the ID never existed or was deleted, and that verdict is
        permanent, so retrying an unknown ID never starts working. Changes nothing.
        """
        return _memory_result(memory.get(memory_id))

    @server.tool(annotations=_READ_ONLY)
    @_stable_errors
    def list_memories(
        limit: Annotated[
            _Limit,
            Field(description="Maximum records in this page, 1 through 100."),
        ] = 100,
        cursor: Annotated[
            _Cursor | None,
            Field(
                description=(
                    "The `next_cursor` from the previous page, passed back unchanged. Its contents"
                    " are opaque; a null `next_cursor` means the listing is complete."
                )
            ),
        ] = None,
    ) -> PageResult:
        """Page through every stored memory, newest first.

        Call this to inventory what is stored when you have no query; use `search_memories` when
        you know what you are looking for. Changes nothing and is safe to retry, but a page is a
        snapshot: memories added or deleted while you page can appear twice or not at all.
        """
        page = memory.list(limit=limit, cursor=cursor)
        return PageResult(
            items=tuple(_memory_result(record) for record in page.items),
            next_cursor=page.next_cursor,
        )

    @server.tool(annotations=_DELETE)
    @_stable_errors
    def delete_memory(
        memory_id: Annotated[_Identifier, Field(description=_MEMORY_ID_DESCRIPTION)],
    ) -> DeleteResult:
        """Permanently delete one memory and any media only it referenced.

        Call this when the caller asked for the memory to be forgotten. There is no undo and no
        tombstone to read back. It is idempotent, so retrying is safe: `deleted` reports whether a
        record existed, and a second call on the same ID returns `false`.
        """
        return DeleteResult(deleted=memory.delete(memory_id))

    @server.tool(annotations=_COUNTED_WRITE)
    @_stable_errors
    def reinforce_memories(
        memory_ids: Annotated[
            tuple[_Identifier, ...],
            Field(
                min_length=1,
                max_length=100,
                description=(
                    "IDs of memories that were actually useful, from a previous result. Duplicates"
                    " count once; IDs that no longer exist are skipped."
                ),
            ),
        ],
    ) -> ReinforceResult:
        """Record that these memories were useful, so retrieval favours them later.

        Call this after a memory helped you answer, and only then: it is the sole way to move the
        ranker's reinforcement factor away from its neutral value. It writes durable state and its
        effect accumulates, so two calls count twice; do not repeat one whose outcome you are
        unsure of. Content, timestamps, and metadata are untouched. Returns how many of the
        supplied IDs existed.
        """
        return ReinforceResult(reinforced=memory.reinforce(memory_ids))

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
    ):
        # A tool body signals failure by raising, and the MCP runtime prefixes that message with
        # "Error executing tool <name>: ". Re-emit the envelope on its own so one bare `json.loads`
        # of the text works for every failure, including the recoverable ones an agent parses to
        # decide whether to retry.
        envelope = _stable_envelope(result)
        return _envelope_result(
            envelope
            if envelope is not None
            else _envelope("validation_error", "tool arguments are invalid", reason="input_invalid")
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
    return _envelope_result(_envelope(code, message, reason=reason, issues=issues))


def _envelope_result(envelope: Mapping[str, object]) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(dict(envelope)))],
        is_error=True,
    )


def _stable_envelope(result: Mapping[str, object]) -> dict[str, object] | None:
    """Recover the envelope a tool raised, ignoring any prefix the MCP runtime added to it."""
    content = result.get("content")
    if not isinstance(content, list):
        return None
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
            return envelope
    return None
