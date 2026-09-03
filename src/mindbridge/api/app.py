"""Thin synchronous FastAPI adapter for one local ``Memory`` instance."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, Protocol, TypeAlias, cast

from fastapi import APIRouter, FastAPI, Query, Response, status
from fastapi import Path as PathParameter
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mindbridge.api.errors import error_response, error_responses, register_error_handlers
from mindbridge.types import (
    AbstentionReason,
    AnswerResult,
    AssetRef,
    Blob,
    ContentInput,
    ContextBudget,
    ContextBundle,
    MemoryCapabilities,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    RetrievalScope,
    SearchHit,
)

_MAX_TEXT_CHARACTERS = 65_536
_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_Text = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=_MAX_TEXT_CHARACTERS,
    ),
]
_Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
_MemoryId = Annotated[str, PathParameter(min_length=1)]
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
_Chars = Annotated[int, Field(strict=True, ge=1, le=_MAX_TEXT_CHARACTERS)]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
_Seconds = Annotated[float, Field(gt=0.0)]
# Transport defaults are read from the SDK value so the documented budget cannot drift.
_BUDGET = ContextBudget()


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _InputText(_RequestModel):
    type: Literal["input_text"]
    text: _Text


class _InputImage(_RequestModel):
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


class _InputFile(_RequestModel):
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


class MemoryCreate(_RequestModel):
    content: _Content
    occurred_at: AwareDatetime | None = None
    occurred_end: AwareDatetime | None = None
    metadata: dict[str, JsonValue] | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: ObservationContext | None = None


class MemoryBatchCreate(_RequestModel):
    contents: Annotated[list[_Content], Field(min_length=1, max_length=100)]
    occurred_at: list[AwareDatetime | None] | None = None
    occurred_end: list[AwareDatetime | None] | None = None
    metadata: list[dict[str, JsonValue] | None] | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: list[ObservationContext | None] | None = None


class QueryRequest(_RequestModel):
    query: _Content
    limit: _Limit = 10
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None
    occurred_from: AwareDatetime | None = None
    occurred_until: AwareDatetime | None = None
    scope: RetrievalScope | None = None

    @model_validator(mode="after")
    def validate_occurrence_range(self) -> QueryRequest:
        if (
            self.occurred_from is not None
            and self.occurred_until is not None
            and self.occurred_until <= self.occurred_from
        ):
            raise ValueError("occurred_until must be later than occurred_from")
        return self


class AnswerRequest(_RequestModel):
    question: _Content
    limit: _Limit = 5
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None


class ContextBudgetRequest(_RequestModel):
    max_chars: _Chars = _BUDGET.max_chars
    max_items: _Limit = _BUDGET.max_items
    memory_types: Annotated[list[MemoryType], Field(min_length=1)] | None = None
    min_confidence: _Confidence = _BUDGET.min_confidence
    freshness_seconds: _Seconds | None = None


class ContextRequest(_RequestModel):
    goal: _Content
    budget: ContextBudgetRequest | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None


class _ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class AssetResponse(_ResponseModel):
    id: str
    modality: Modality
    media_type: str
    size_bytes: int
    sha256: str
    name: str | None = None


class MemoryResponse(_ResponseModel):
    id: str
    content: str
    modality: Modality
    memory_type: MemoryType
    assets: tuple[AssetResponse, ...] = ()
    created_at: AwareDatetime
    occurred_at: AwareDatetime | None = None
    occurred_end: AwareDatetime | None = None
    metadata: dict[str, JsonValue]
    context: MemoryContext | None = None
    forgotten_at: AwareDatetime | None = None


class SearchHitResponse(MemoryResponse):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class MemoryBatchResponse(_ResponseModel):
    memories: tuple[MemoryResponse, ...]


class SearchResponse(_ResponseModel):
    hits: tuple[SearchHitResponse, ...]


class AnswerResponse(_ResponseModel):
    answer: str
    hits: tuple[SearchHitResponse, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class ContextBudgetResponse(_ResponseModel):
    max_chars: int
    max_items: int
    memory_types: tuple[MemoryType, ...] | None
    min_confidence: float
    freshness_seconds: float | None


class ContextConflictResponse(_ResponseModel):
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


class ContextBundleResponse(_ResponseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResponse
    actors: tuple[SearchHitResponse, ...]
    episodes: tuple[SearchHitResponse, ...]
    facts: tuple[SearchHitResponse, ...]
    procedures: tuple[SearchHitResponse, ...]
    affect: tuple[SearchHitResponse, ...]
    traits: tuple[SearchHitResponse, ...]
    conflicts: tuple[ContextConflictResponse, ...]
    occurred_from: AwareDatetime | None
    occurred_until: AwareDatetime | None
    frames: tuple[str, ...]
    omitted: int
    chars: int
    rendered: str


class CapabilitiesResponse(_ResponseModel):
    modalities: tuple[Modality, ...]
    answer: bool
    transcribe: bool
    faces: bool
    describe_vision: bool
    form: bool
    consolidate: bool
    decay: bool


class DeleteResponse(_ResponseModel):
    deleted: bool


class PageResponse(_ResponseModel):
    items: tuple[MemoryResponse, ...]
    next_cursor: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class _Memory(Protocol):
    def add(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord: ...

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        occurred_at: Sequence[datetime | None] | None = None,
        occurred_end: Sequence[datetime | None] | None = None,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: Sequence[ObservationContext | None] | None = None,
    ) -> tuple[MemoryRecord, ...]: ...

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
    ) -> tuple[SearchHit, ...]: ...

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> AnswerResult: ...

    def compile(
        self,
        goal: ContentInput,
        *,
        budget: ContextBudget | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
    ) -> ContextBundle: ...

    def capabilities(self) -> MemoryCapabilities: ...

    def get(self, memory_id: str) -> MemoryRecord: ...

    def list(self, *, limit: int = 100, cursor: str | None = None) -> Page: ...

    def delete(self, memory_id: str) -> bool: ...


class _RequestBodyLimit:
    """Bound `/v1` request bodies before framework parsing."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_api_path(str(scope.get("path", ""))):
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        content_length = _content_length(headers)
        if content_length is not None and content_length > _MAX_REQUEST_BODY_BYTES:
            await _request_too_large()(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > _MAX_REQUEST_BODY_BYTES:
                await _request_too_large()(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


def create_app(
    *,
    memory: _Memory,
) -> FastAPI:
    """Create an unauthenticated API over one caller-owned memory instance."""

    def current_service() -> _Memory:
        return memory

    app = FastAPI(title="MindBridge", version="0.2.0")
    register_error_handlers(app)
    app.add_middleware(_RequestBodyLimit)
    router = APIRouter(prefix="/v1")
    standard_statuses = (
        status.HTTP_413_CONTENT_TOO_LARGE,
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    standard_errors = error_responses(*standard_statuses)
    model_errors = error_responses(
        *standard_statuses,
        status.HTTP_501_NOT_IMPLEMENTED,
        status.HTTP_502_BAD_GATEWAY,
    )
    not_found_errors = error_responses(*standard_statuses, status.HTTP_404_NOT_FOUND)

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        operation_id="health",
        responses=error_responses(status.HTTP_500_INTERNAL_SERVER_ERROR),
    )
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post(
        "/memories",
        response_model=MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="createMemory",
        responses=model_errors,
    )
    def create_memory(request: MemoryCreate) -> MemoryResponse:
        record = current_service().add(
            _content_input(request.content),
            occurred_at=request.occurred_at,
            occurred_end=request.occurred_end,
            metadata=request.metadata,
            memory_type=request.memory_type,
            context=request.context,
        )
        return MemoryResponse.model_validate(record)

    @router.post(
        "/memories/batch",
        response_model=MemoryBatchResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="createMemories",
        responses=model_errors,
    )
    def create_memories(request: MemoryBatchCreate) -> MemoryBatchResponse:
        return MemoryBatchResponse.model_validate(
            {
                "memories": current_service().add_many(
                    tuple(_content_input(content) for content in request.contents),
                    occurred_at=request.occurred_at,
                    occurred_end=request.occurred_end,
                    metadata=request.metadata,
                    memory_type=request.memory_type,
                    context=request.context,
                )
            }
        )

    @router.get(
        "/memories",
        response_model=PageResponse,
        operation_id="listMemories",
        responses=standard_errors,
    )
    def list_memories(
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
        cursor: Annotated[str | None, Query(min_length=1)] = None,
    ) -> PageResponse:
        return PageResponse.model_validate(current_service().list(limit=limit, cursor=cursor))

    @router.post(
        "/memories/search",
        response_model=SearchResponse,
        operation_id="searchMemories",
        responses=model_errors,
    )
    def search_memories(request: QueryRequest) -> SearchResponse:
        return SearchResponse.model_validate(
            {
                "hits": current_service().search(
                    _content_input(request.query),
                    limit=request.limit,
                    memory_type=request.memory_type,
                    reference_at=request.reference_at,
                    occurred_from=request.occurred_from,
                    occurred_until=request.occurred_until,
                    scope=request.scope,
                )
            }
        )

    @router.get(
        "/memories/{memory_id}",
        response_model=MemoryResponse,
        operation_id="getMemory",
        responses=not_found_errors,
    )
    def get_memory(memory_id: _MemoryId) -> MemoryResponse:
        return MemoryResponse.model_validate(current_service().get(memory_id))

    @router.delete(
        "/memories/{memory_id}",
        response_model=DeleteResponse,
        operation_id="deleteMemory",
        responses=standard_errors,
    )
    def delete_memory(memory_id: _MemoryId) -> DeleteResponse:
        return DeleteResponse(deleted=current_service().delete(memory_id))

    @router.post(
        "/answers",
        response_model=AnswerResponse,
        operation_id="answer",
        responses=model_errors,
    )
    def answer(request: AnswerRequest) -> AnswerResponse:
        return AnswerResponse.model_validate(
            current_service().ask(
                _content_input(request.question),
                limit=request.limit,
                memory_type=request.memory_type,
                reference_at=request.reference_at,
                scope=request.scope,
            )
        )

    _add_context_routes(
        router,
        current_service,
        model_errors=model_errors,
        standard_errors=standard_errors,
    )
    app.include_router(router)
    return app


def _add_context_routes(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    model_errors: dict[int | str, dict[str, Any]],
    standard_errors: dict[int | str, dict[str, Any]],
) -> None:
    """Register the agent-facing compiler routes on the `/v1` router."""

    @router.post(
        "/context",
        response_model=ContextBundleResponse,
        operation_id="compileContext",
        responses=model_errors,
    )
    def compile_context(request: ContextRequest) -> ContextBundleResponse:
        return _bundle_response(
            current_service().compile(
                _content_input(request.goal),
                budget=_context_budget(request.budget),
                reference_at=request.reference_at,
                scope=request.scope,
            )
        )

    @router.get(
        "/capabilities",
        response_model=CapabilitiesResponse,
        operation_id="capabilities",
        responses=standard_errors,
    )
    def capabilities() -> CapabilitiesResponse:
        reported = current_service().capabilities()
        return CapabilitiesResponse(
            modalities=tuple(sorted(reported.modalities)),
            answer=reported.answer,
            transcribe=reported.transcribe,
            faces=reported.faces,
            describe_vision=reported.describe_vision,
            form=reported.form,
            consolidate=reported.consolidate,
            decay=reported.decay,
        )


def _context_budget(request: ContextBudgetRequest | None) -> ContextBudget | None:
    if request is None:
        return None
    return ContextBudget(
        max_chars=request.max_chars,
        max_items=request.max_items,
        memory_types=None if request.memory_types is None else frozenset(request.memory_types),
        min_confidence=request.min_confidence,
        freshness=(
            None
            if request.freshness_seconds is None
            else timedelta(seconds=request.freshness_seconds)
        ),
    )


def _bundle_response(bundle: ContextBundle) -> ContextBundleResponse:
    return ContextBundleResponse.model_validate(
        {
            "goal": bundle.goal,
            "reference_at": bundle.reference_at,
            "budget": {
                "max_chars": bundle.budget.max_chars,
                "max_items": bundle.budget.max_items,
                "memory_types": (
                    None
                    if bundle.budget.memory_types is None
                    else tuple(sorted(bundle.budget.memory_types))
                ),
                "min_confidence": bundle.budget.min_confidence,
                "freshness_seconds": (
                    None
                    if bundle.budget.freshness is None
                    else bundle.budget.freshness.total_seconds()
                ),
            },
            "actors": bundle.actors,
            "episodes": bundle.episodes,
            "facts": bundle.facts,
            "procedures": bundle.procedures,
            "affect": bundle.affect,
            "traits": bundle.traits,
            "conflicts": bundle.conflicts,
            "occurred_from": bundle.occurred_from,
            "occurred_until": bundle.occurred_until,
            "frames": bundle.frames,
            "omitted": bundle.omitted,
            "chars": bundle.chars,
            "rendered": bundle.render(),
        }
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
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("file_data must be valid base64") from error


def _headers(scope: Scope) -> list[tuple[bytes, bytes]]:
    return list(scope.get("headers", []))


def _one_header(headers: Sequence[tuple[bytes, bytes]], name: bytes) -> bytes | None:
    values = [value for key, value in headers if key.lower() == name]
    return values[0] if len(values) == 1 else None


def _content_length(headers: Sequence[tuple[bytes, bytes]]) -> int | None:
    raw = _one_header(headers, b"content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _is_api_path(path: str) -> bool:
    return path == "/v1" or path.startswith("/v1/")


def _request_too_large() -> Response:
    return error_response(
        status.HTTP_413_CONTENT_TOO_LARGE,
        "request_too_large",
        f"request body must not exceed {_MAX_REQUEST_BODY_BYTES} bytes",
        reason="payload_too_large",
    )
