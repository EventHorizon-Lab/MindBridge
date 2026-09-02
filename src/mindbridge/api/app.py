"""Thin synchronous FastAPI adapter for one local ``Memory`` instance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, FastAPI, Query, Response, status
from fastapi import Path as PathParameter
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mindbridge.api.content import Content, StrictModel, content_input
from mindbridge.api.errors import (
    REASON_STATUS,
    error_response,
    error_responses,
    register_error_handlers,
)
from mindbridge.types import (
    AbstentionReason,
    AnswerResult,
    ContentInput,
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

_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
_MemoryId = Annotated[str, PathParameter(min_length=1)]


class MemoryCreate(StrictModel):
    content: Content
    occurred_at: AwareDatetime | None = None
    occurred_end: AwareDatetime | None = None
    metadata: dict[str, JsonValue] | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: ObservationContext | None = None


class MemoryBatchCreate(StrictModel):
    contents: Annotated[list[Content], Field(min_length=1, max_length=100)]
    occurred_at: list[AwareDatetime | None] | None = None
    occurred_end: list[AwareDatetime | None] | None = None
    metadata: list[dict[str, JsonValue] | None] | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    context: list[ObservationContext | None] | None = None


class QueryRequest(StrictModel):
    query: Content
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


class AnswerRequest(StrictModel):
    question: Content
    limit: _Limit = 5
    memory_type: MemoryType | None = None
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
    place_id: str | None = None


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


class DeleteResponse(_ResponseModel):
    deleted: bool


class PageResponse(_ResponseModel):
    items: tuple[MemoryResponse, ...]
    next_cursor: str | None = None


class CapabilitiesResponse(_ResponseModel):
    """What the composition behind this process can actually do."""

    embedding: tuple[Modality, ...]
    embedding_model: str
    embedding_space: str
    embedding_dimension: int
    generation: tuple[Modality, ...]
    transcription: tuple[Modality, ...]
    vision: tuple[Modality, ...]
    face: tuple[Modality, ...]
    formation: tuple[Modality, ...]
    generation_model: str | None
    transcription_space: str | None
    vision_model: str | None
    face_model: str | None
    formation_model: str | None
    speaker_recognition: bool
    streaming_generation: bool


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    capabilities: CapabilitiesResponse


class _Memory(Protocol):
    @property
    def capabilities(self) -> MemoryCapabilities: ...

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
        return HealthResponse(capabilities=_capabilities_response(current_service().capabilities))

    @router.post(
        "/memories",
        response_model=MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="createMemory",
        responses=model_errors,
    )
    def create_memory(request: MemoryCreate) -> MemoryResponse:
        record = current_service().add(
            content_input(request.content),
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
                    tuple(content_input(content) for content in request.contents),
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
                    content_input(request.query),
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
                content_input(request.question),
                limit=request.limit,
                memory_type=request.memory_type,
                reference_at=request.reference_at,
                scope=request.scope,
            )
        )

    app.include_router(router)
    return app


def _capabilities_response(capabilities: MemoryCapabilities) -> CapabilitiesResponse:
    """Serialize declared capabilities, ordering modality sets so the document is stable."""
    values: dict[str, object] = {}
    for name in CapabilitiesResponse.model_fields:
        value = getattr(capabilities, name)
        values[name] = (
            tuple(sorted(value, key=lambda modality: modality.value))
            if isinstance(value, frozenset)
            else value
        )
    return CapabilitiesResponse.model_validate(values)


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
        # Read from the shared table, so this cannot drift from the provider-side raise site that
        # reports the same reason.
        REASON_STATUS["payload_too_large"],
        "request_too_large",
        f"request body must not exceed {_MAX_REQUEST_BODY_BYTES} bytes",
        reason="payload_too_large",
    )
