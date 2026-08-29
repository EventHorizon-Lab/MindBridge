"""Thin synchronous FastAPI adapter for one local ``Memory`` instance."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Annotated, Literal, Protocol, TypeAlias, cast

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
    AnswerResult,
    AssetRef,
    Blob,
    ContentInput,
    MemoryRecord,
    MemoryType,
    Modality,
    Page,
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
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    memory_type: MemoryType = MemoryType.SEMANTIC


class MemoryBatchCreate(_RequestModel):
    contents: Annotated[list[_Content], Field(min_length=1, max_length=100)]
    memory_type: MemoryType = MemoryType.SEMANTIC


class QueryRequest(_RequestModel):
    query: _Content
    limit: _Limit = 10
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None


class AnswerRequest(_RequestModel):
    question: _Content
    limit: _Limit = 5
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None


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


class SearchHitResponse(MemoryResponse):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class MemoryBatchResponse(_ResponseModel):
    memories: tuple[MemoryResponse, ...]


class SearchResponse(_ResponseModel):
    hits: tuple[SearchHitResponse, ...]


class AnswerResponse(_ResponseModel):
    answer: str
    hits: tuple[SearchHitResponse, ...]


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
    ) -> MemoryRecord: ...

    def add_many(
        self,
        contents: Sequence[ContentInput],
        *,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> tuple[MemoryRecord, ...]: ...

    def search(
        self,
        query: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> tuple[SearchHit, ...]: ...

    def ask(
        self,
        question: ContentInput,
        *,
        limit: int = 10,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
    ) -> AnswerResult: ...

    def get(self, memory_id: str) -> MemoryRecord: ...

    def list(self, *, limit: int = 50, cursor: str | None = None) -> Page: ...

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
    model_errors = error_responses(*standard_statuses, status.HTTP_502_BAD_GATEWAY)
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
                    memory_type=request.memory_type,
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
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
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
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="deleteMemory",
        responses=standard_errors,
    )
    def delete_memory(memory_id: _MemoryId) -> Response:
        current_service().delete(memory_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

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
            )
        )

    app.include_router(router)
    return app


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
    )
