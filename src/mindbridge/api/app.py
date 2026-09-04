"""Thin synchronous FastAPI adapter for one local ``Memory`` instance."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal, Protocol

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

from mindbridge.api.content import (
    MAX_TEXT_CHARACTERS,
    Content,
    StrictModel,
    content_input,
)
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
    ContextBudget,
    ContextBundle,
    ContextUnknownKind,
    MemoryCapabilities,
    MemoryContext,
    MemoryRecord,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    RetrievalScope,
    RetrievalTrace,
    SearchHit,
    TracedSearchResult,
)

_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
_MemoryId = Annotated[str, PathParameter(min_length=1)]
# The same identifier inside a request body, where `PathParameter` does not apply. Constrained
# here rather than only in the SDK so a malformed ID fails validation on every route alike.
_BodyMemoryId = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
_Chars = Annotated[int, Field(strict=True, ge=1, le=MAX_TEXT_CHARACTERS)]
_Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
_Seconds = Annotated[float, Field(gt=0.0)]
_Milliseconds = Annotated[int, Field(strict=True, ge=1)]
# Budget defaults are read from the SDK value, so the published transport default cannot drift
# from `ContextBudget`.
_BUDGET = ContextBudget()


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
    explain: bool = False

    @model_validator(mode="after")
    def validate_occurrence_range(self) -> QueryRequest:
        if (
            self.occurred_from is not None
            and self.occurred_until is not None
            and self.occurred_until <= self.occurred_from
        ):
            raise ValueError("occurred_until must be later than occurred_from")
        return self


class ReinforceRequest(StrictModel):
    memory_ids: Annotated[list[_BodyMemoryId], Field(min_length=1, max_length=100)]


class AnswerRequest(StrictModel):
    question: Content
    limit: _Limit = 5
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None


class ContextBudgetRequest(StrictModel):
    max_chars: _Chars = _BUDGET.max_chars
    max_items: _Limit = _BUDGET.max_items
    memory_types: Annotated[list[MemoryType], Field(min_length=1)] | None = None
    min_confidence: _Confidence = _BUDGET.min_confidence
    # `ContextBudget.freshness` is a timedelta; JSON carries the same bound as seconds.
    freshness_seconds: _Seconds | None = None
    max_latency_ms: _Milliseconds | None = None


class ContextRequest(StrictModel):
    goal: Content
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
    place_id: str | None = None
    forgotten_at: AwareDatetime | None = None


class SearchHitResponse(MemoryResponse):
    score: Annotated[float, Field(ge=0.0, le=1.0)]


class AffectCueResponse(SearchHitResponse):
    """One affect entry of a compiled bundle, with the evidence hop the cue hangs on.

    The observations the cue cites are its own `context.evidence_ids`. `event_ids` are the
    events formed from those same observations: co-occurrence inside one capture, never an
    attributed cause.
    """

    event_ids: tuple[str, ...] = ()


class MemoryBatchResponse(_ResponseModel):
    memories: tuple[MemoryResponse, ...]


class SearchResponse(_ResponseModel):
    hits: tuple[SearchHitResponse, ...]
    # Null unless the request asked for it, so the default response keeps the shape clients read.
    trace: RetrievalTrace | None = None


class AnswerResponse(_ResponseModel):
    answer: str
    hits: tuple[SearchHitResponse, ...]
    abstained: bool
    abstention_reason: AbstentionReason | None


class DeleteResponse(_ResponseModel):
    deleted: bool


class ReinforceResponse(_ResponseModel):
    reinforced: int


class ContextBudgetResponse(_ResponseModel):
    max_chars: int
    max_items: int
    memory_types: tuple[MemoryType, ...] | None
    min_confidence: float
    freshness_seconds: float | None
    max_latency_ms: int | None


class ContextConflictResponse(_ResponseModel):
    lineage_id: str
    subject: str | None
    predicate: str | None
    values: tuple[str, ...]
    memory_ids: tuple[str, ...]


class ProvisionalActorResponse(_ResponseModel):
    identity_id: str
    memory_ids: tuple[str, ...]


class ContextUnknownResponse(_ResponseModel):
    kind: ContextUnknownKind
    detail: str


class ContextBundleResponse(_ResponseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResponse
    # A recognized person in the evidence whom no visible naming assertion names is reported
    # here beside the ranked entity hits, labelled, rather than omitted.
    actors: tuple[SearchHitResponse | ProvisionalActorResponse, ...]
    relationships: tuple[SearchHitResponse, ...]
    scene: tuple[SearchHitResponse, ...]
    episodes: tuple[SearchHitResponse, ...]
    facts: tuple[SearchHitResponse, ...]
    procedures: tuple[SearchHitResponse, ...]
    affect: tuple[AffectCueResponse, ...]
    traits: tuple[SearchHitResponse, ...]
    conflicts: tuple[ContextConflictResponse, ...]
    unknowns: tuple[ContextUnknownResponse, ...]
    occurred_from: AwareDatetime | None
    occurred_until: AwareDatetime | None
    frames: tuple[str, ...]
    places: tuple[str, ...]
    omitted: int
    chars: int
    elapsed_ms: int
    deadline_exceeded: bool
    # The deterministic text of `ContextBundle.render()`, so a caller need not re-derive it.
    rendered: str


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
    consolidation_model: str | None
    speaker_recognition: bool
    streaming_generation: bool
    # Derived from the backends above, not declared: which optional operations this composition
    # can serve. `MemoryCapabilities.document()` is the one place that derivation happens.
    operations: tuple[str, ...]


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
    ) -> TracedSearchResult: ...

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

    def reinforce(self, memory_ids: Sequence[str]) -> int: ...

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
    app = FastAPI(title="MindBridge", version="0.2.0")
    register_error_handlers(app)
    app.add_middleware(_RequestBodyLimit)

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        operation_id="health",
        responses=error_responses(status.HTTP_500_INTERNAL_SERVER_ERROR),
    )
    def health() -> HealthResponse:
        # Read from the injected instance on every call, so a composition swapped behind this
        # process is reported rather than a snapshot taken at construction.
        return HealthResponse(capabilities=_capabilities_response(memory.capabilities))

    app.include_router(_v1_router(memory))
    return app


def _v1_router(memory: _Memory) -> APIRouter:
    """Register every `/v1` route against one caller-owned memory instance."""

    def current_service() -> _Memory:
        return memory

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
        return _search(current_service(), request)

    @router.post(
        "/memories/reinforce",
        response_model=ReinforceResponse,
        operation_id="reinforceMemories",
        responses=standard_errors,
    )
    def reinforce_memories(request: ReinforceRequest) -> ReinforceResponse:
        return ReinforceResponse(reinforced=current_service().reinforce(request.memory_ids))

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

    _add_context_route(router, current_service, responses=model_errors)
    return router


def _add_context_route(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    responses: dict[int | str, dict[str, Any]],
) -> None:
    """Register the compiler route, split out only to keep `_v1_router` readable."""

    @router.post(
        "/context",
        response_model=ContextBundleResponse,
        operation_id="compileContext",
        responses=responses,
    )
    def compile_context(request: ContextRequest) -> ContextBundleResponse:
        return _bundle_response(
            current_service().compile(
                content_input(request.goal),
                budget=_context_budget(request.budget),
                reference_at=request.reference_at,
                scope=request.scope,
            )
        )


def _context_budget(request: ContextBudgetRequest | None) -> ContextBudget | None:
    """Translate the transport budget into the SDK value, which validates every bound."""
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
        max_latency_ms=request.max_latency_ms,
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
                "max_latency_ms": bundle.budget.max_latency_ms,
            },
            "actors": bundle.actors,
            "relationships": bundle.relationships,
            "scene": bundle.scene,
            "episodes": bundle.episodes,
            "facts": bundle.facts,
            "procedures": bundle.procedures,
            "affect": bundle.affect,
            "traits": bundle.traits,
            "conflicts": bundle.conflicts,
            "unknowns": bundle.unknowns,
            "occurred_from": bundle.occurred_from,
            "occurred_until": bundle.occurred_until,
            "frames": bundle.frames,
            "places": bundle.places,
            "omitted": bundle.omitted,
            "chars": bundle.chars,
            "elapsed_ms": bundle.elapsed_ms,
            "deadline_exceeded": bundle.deadline_exceeded,
            "rendered": bundle.render(),
        }
    )


def _search(service: _Memory, request: QueryRequest) -> SearchResponse:
    """Route one search to the traced SDK operation only when the caller asked to see the trace."""
    content = content_input(request.query)
    if not request.explain:
        hits = service.search(
            content,
            limit=request.limit,
            memory_type=request.memory_type,
            reference_at=request.reference_at,
            occurred_from=request.occurred_from,
            occurred_until=request.occurred_until,
            scope=request.scope,
        )
        return SearchResponse.model_validate({"hits": hits})
    traced = service.search_with_trace(
        content,
        limit=request.limit,
        memory_type=request.memory_type,
        reference_at=request.reference_at,
        occurred_from=request.occurred_from,
        occurred_until=request.occurred_until,
        scope=request.scope,
    )
    return SearchResponse.model_validate({"hits": traced.hits, "trace": traced.trace})


def _capabilities_response(capabilities: MemoryCapabilities) -> CapabilitiesResponse:
    """Serialize the one capability document MCP and the CLI publish too."""
    return CapabilitiesResponse.model_validate(capabilities.document())


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
