"""Thin synchronous FastAPI adapter for one local ``Memory`` instance."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from contextvars import Context, copy_context
from datetime import datetime
from time import perf_counter_ns
from typing import Annotated, Any, Literal, Protocol, cast

from fastapi import APIRouter, FastAPI, Query, Request, Response, status
from fastapi import Path as PathParameter
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode, Tracer
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
    Content,
    ContextBudgetInput,
    Limit,
    StrictModel,
    content_input,
    context_budget,
)
from mindbridge.api.errors import (
    REASON_STATUS,
    error_response,
    error_responses,
    exception_response,
    register_error_handlers,
)
from mindbridge.types import (
    AbstentionReason,
    AnswerChunk,
    AnswerResult,
    ConsentState,
    ContentInput,
    ContextBudget,
    ContextBundle,
    ContextUnknownKind,
    ExportBundle,
    FaceObservation,
    IdentityErasure,
    IdentityProfile,
    MemoryCapabilities,
    MemoryContext,
    MemoryIntent,
    MemoryOperationRecord,
    MemoryOutcome,
    MemoryRecord,
    MemoryTrigger,
    MemoryType,
    Modality,
    ObservationContext,
    Page,
    PendingCapture,
    RetentionReport,
    RetrievalScope,
    RetrievalTrace,
    SearchHit,
    SpeakerSegment,
    TracedSearchResult,
)

_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
_MemoryId = Annotated[str, PathParameter(min_length=1)]
# The same identifier inside a request body, where `PathParameter` does not apply. Constrained
# here rather than only in the SDK so a malformed ID fails validation on every route alike.
_BodyMemoryId = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]
# A bounded plain string for person-facing text -- a name or relationship label -- the same shape
# the MCP `register_identity` tool's `name`/`relationship` arguments use. Named separately from
# `_BodyMemoryId` so a future ID-specific tightening of that type cannot silently start rejecting
# names too.
_PersonText = _BodyMemoryId
_HTTP_TTFH = "mindbridge.transport.server_time_to_headers_ms"
_HTTP_TTFB = "mindbridge.transport.server_time_to_first_body_byte_ms"
_HTTP_TOTAL = "mindbridge.transport.server_total_ms"
_STREAM_EXHAUSTED = object()


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
    limit: Limit = 10
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
    limit: Limit = 5
    memory_type: MemoryType | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None


# The shared bounds under the name FastAPI publishes as the OpenAPI component; no fields and no
# docstring of its own, so the schema stays what it was.
class ContextBudgetRequest(ContextBudgetInput):
    pass


class ContextRequest(StrictModel):
    goal: Content
    budget: ContextBudgetRequest | None = None
    reference_at: AwareDatetime | None = None
    scope: RetrievalScope | None = None


class SettleRequest(StrictModel):
    limit: Limit = 100
    max_attempts: Annotated[int, Field(strict=True, ge=1)] = 3
    memory_ids: Annotated[list[_BodyMemoryId], Field(min_length=1, max_length=100)] | None = None


class AnalyzeRequest(StrictModel):
    memory_id: _BodyMemoryId


class IdentityRegisterRequest(StrictModel):
    identity_id: _BodyMemoryId
    name: _PersonText
    relationship: _PersonText | None = None


class ConsentRequest(StrictModel):
    state: ConsentState
    note: _PersonText | None = None


class RetentionRequest(StrictModel):
    dry_run: bool = False


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


class SettleResponse(_ResponseModel):
    settled: int


class PendingCaptureResponse(_ResponseModel):
    memory_id: str
    enqueued_at: AwareDatetime
    attempts: int
    last_error: str | None = None
    awaiting: Literal["enrichment", "formation"]


class PendingCapturesResponse(_ResponseModel):
    items: tuple[PendingCaptureResponse, ...]


class SpeechResponse(_ResponseModel):
    segments: tuple[SpeakerSegment, ...]


class FacesResponse(_ResponseModel):
    observations: tuple[FaceObservation, ...]


class IdentityResponse(_ResponseModel):
    identity: IdentityProfile | None


class RegisterResponse(_ResponseModel):
    registered: bool


class UnlinkResponse(_ResponseModel):
    restored_identity_id: str | None


class ForgetResponse(_ResponseModel):
    erasure: IdentityErasure


class ConsentResponse(_ResponseModel):
    consent: ConsentState | None


class IdentityClaimResponse(_ResponseModel):
    identity_id: str
    name: str
    relationship: str | None


class ConsentClaimResponse(_ResponseModel):
    identity_id: str
    state: ConsentState
    note: str | None


class IdentityChangeResponse(_ResponseModel):
    identity_id: str
    moved_ids: tuple[str, ...]


class MemoryOperationResponse(_ResponseModel):
    """One control-plane log row, flattened exactly as `mindbridge operations` prints it.

    The operation's own fields are lifted beside the log row's because a reader wants one
    record of what was proposed and what it did, not two nested objects to join.
    """

    operation_id: int
    intent: MemoryIntent
    trigger: MemoryTrigger
    evidence_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    claim: IdentityClaimResponse | None
    consent: ConsentClaimResponse | None
    identity: IdentityChangeResponse | None
    rationale: str | None
    model_id: str | None
    recipe: str | None
    created_ids: tuple[str, ...]
    changed_ids: tuple[str, ...]
    forgotten_ids: tuple[str, ...]
    superseded: tuple[tuple[str, int], ...]
    applied_at: datetime
    rolled_back_at: datetime | None
    outcome: MemoryOutcome | None
    outcome_note: str | None


class RecordConsentResponse(_ResponseModel):
    operation: MemoryOperationResponse | None


class ExportResponse(_ResponseModel):
    exported_at: datetime
    identity_id: str | None
    identities: tuple[IdentityProfile, ...]
    records: tuple[MemoryResponse, ...]
    operations: tuple[MemoryOperationResponse, ...]


class RetentionResponse(_ResponseModel):
    dry_run: bool
    media_memory_ids: tuple[str, ...]
    forgotten_memory_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    capture_memory_ids: tuple[str, ...]
    deleted: int


def _operation_response(record: MemoryOperationRecord) -> MemoryOperationResponse:
    operation = record.operation
    return MemoryOperationResponse(
        operation_id=record.operation_id,
        intent=operation.intent,
        trigger=record.trigger,
        evidence_ids=operation.evidence_ids,
        target_ids=operation.target_ids,
        claim=(
            None
            if operation.claim is None
            else IdentityClaimResponse.model_validate(operation.claim)
        ),
        consent=(
            None
            if operation.consent is None
            else ConsentClaimResponse.model_validate(operation.consent)
        ),
        identity=(
            None
            if operation.identity is None
            else IdentityChangeResponse.model_validate(operation.identity)
        ),
        rationale=operation.rationale,
        model_id=record.model_id,
        recipe=record.recipe,
        created_ids=record.created_ids,
        changed_ids=record.changed_ids,
        forgotten_ids=record.forgotten_ids,
        superseded=record.superseded,
        applied_at=record.applied_at,
        rolled_back_at=record.rolled_back_at,
        outcome=record.outcome,
        outcome_note=record.outcome_note,
    )


def _export_response(bundle: ExportBundle) -> ExportResponse:
    return ExportResponse(
        exported_at=bundle.exported_at,
        identity_id=bundle.identity_id,
        identities=bundle.identities,
        records=tuple(MemoryResponse.model_validate(record) for record in bundle.records),
        operations=tuple(_operation_response(record) for record in bundle.operations),
    )


class ContextBudgetResponse(_ResponseModel):
    max_chars: int
    max_items: int
    max_media_items: int | None
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


class NamedActorResponse(_ResponseModel):
    identity_id: str
    name: str
    memory_ids: tuple[str, ...]
    naming_assertion_id: str | None


class ContextUnknownResponse(_ResponseModel):
    kind: ContextUnknownKind
    detail: str


class ContextBundleResponse(_ResponseModel):
    goal: str
    reference_at: AwareDatetime
    budget: ContextBudgetResponse
    # Beside the ranked entity hits: a person a naming assertion names, reached through
    # other evidence, and a recognized person no visible assertion names -- reported rather
    # than omitted either way.
    actors: tuple[SearchHitResponse | NamedActorResponse | ProvisionalActorResponse, ...]
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
        link_identities: bool = True,
    ) -> AnswerResult: ...

    def ask_stream(
        self,
        question: ContentInput,
        *,
        limit: int = 5,
        memory_type: MemoryType | None = None,
        reference_at: datetime | None = None,
        scope: RetrievalScope | None = None,
        link_identities: bool = True,
    ) -> Generator[AnswerChunk, None, AnswerResult]: ...

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

    def capture(
        self,
        content: ContentInput,
        *,
        occurred_at: datetime | None = None,
        occurred_end: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        context: ObservationContext | None = None,
    ) -> MemoryRecord: ...

    def settle(
        self,
        *,
        limit: int = 100,
        max_attempts: int = 3,
        memory_ids: Sequence[str] | None = None,
    ) -> int: ...

    def pending_captures(
        self,
        *,
        limit: int = 100,
        memory_ids: Sequence[str] | None = None,
    ) -> tuple[PendingCapture, ...]: ...

    def speech(self, memory_id: str) -> tuple[SpeakerSegment, ...]: ...

    def faces(self, memory_id: str) -> tuple[FaceObservation, ...]: ...

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
    ) -> None: ...

    def identity(self, identity_id: str) -> IdentityProfile | None: ...

    def unlink_identity(self, alias_id: str) -> str | None: ...

    def forget_identity(self, identity_id: str) -> IdentityErasure: ...

    def record_consent(
        self,
        identity_id: str,
        state: ConsentState,
        *,
        note: str | None = None,
    ) -> MemoryOperationRecord | None: ...

    def consent(self, identity_id: str) -> ConsentState | None: ...

    def export(
        self,
        *,
        identity_id: str | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> ExportBundle: ...

    def apply_retention(self, *, dry_run: bool = False) -> RetentionReport: ...


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


class _RequestTelemetry:
    """Measure the server side of one `/v1` request, including streamed delivery."""

    def __init__(self, app: ASGIApp, *, tracer: Tracer) -> None:
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        if scope["type"] != "http" or not _is_api_path(path):
            await self.app(scope, receive, send)
            return

        started = perf_counter_ns()
        response_started = False
        first_body_sent = False
        response_bytes = 0
        attributes = {
            "mindbridge.span.kind": "transport",
            "http.request.method": str(scope.get("method", "")),
            "mindbridge.transport.response_mode": (
                "streaming" if path == "/v1/answers/stream" else "buffered"
            ),
        }
        with self.tracer.start_as_current_span(
            "mindbridge.http.request",
            kind=SpanKind.SERVER,
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:

            async def observed_send(message: Message) -> None:
                nonlocal response_started, first_body_sent, response_bytes
                now = perf_counter_ns()
                if message["type"] == "http.response.start":
                    response_started = True
                    status_code = int(message["status"])
                    span.set_attribute("http.response.status_code", status_code)
                    span.set_attribute(_HTTP_TTFH, (now - started) / 1_000_000)
                    if status_code >= 500:
                        span.set_status(StatusCode.ERROR)
                elif message["type"] == "http.response.body":
                    body = message.get("body", b"")
                    response_bytes += len(body)
                    if body and not first_body_sent:
                        first_body_sent = True
                        span.set_attribute(_HTTP_TTFB, (now - started) / 1_000_000)
                await send(message)

            try:
                await self.app(scope, receive, observed_send)
            except BaseException:
                span.set_status(StatusCode.ERROR)
                raise
            finally:
                route = scope.get("route")
                route_path = getattr(route, "path", None)
                if isinstance(route_path, str):
                    span.set_attribute("http.route", route_path)
                span.set_attribute(_HTTP_TOTAL, (perf_counter_ns() - started) / 1_000_000)
                span.set_attribute("http.response.body.size", response_bytes)
                span.set_attribute("mindbridge.transport.response_started", response_started)


def create_app(
    *,
    memory: _Memory,
    identity_operations: bool = False,
    embodied_operations: bool = False,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Create an unauthenticated API over one caller-owned memory instance.

    ``identity_operations`` and ``embodied_operations`` mirror the same-named switches on
    ``build_mcp_server``. Both default to off: REST is a network surface, and naming, erasing,
    or analyzing a person is host authority that no route grants unless the host opts in. When a
    switch is off the routes it gates are never registered, so a caller gets 404 rather than 403 --
    it cannot discover through the error what an enabled deployment would offer.
    """
    app = FastAPI(title="MindBridge", version="0.2.0")
    register_error_handlers(app)
    app.add_middleware(_RequestBodyLimit)
    app.add_middleware(
        _RequestTelemetry,
        tracer=tracer or trace.get_tracer("mindbridge.api"),
    )

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

    app.include_router(
        _v1_router(
            memory,
            identity_operations=identity_operations,
            embodied_operations=embodied_operations,
        )
    )
    return app


def _v1_router(  # noqa: C901 - one literal public route registry
    memory: _Memory,
    *,
    identity_operations: bool,
    embodied_operations: bool,
) -> APIRouter:
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
    streaming_errors = {
        **model_errors,
        status.HTTP_200_OK: {
            "description": "SSE answer deltas followed by one grounded result event",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
    }

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
                # Mirrors MCP's own `embodied_operations` switch (`docs/context-os.md`): a
                # caller with recall access alone must not acquire cross-modal merge authority
                # through `ask`, so REST only lets an answer commit that bind when the host has
                # opted in to embodied operations.
                link_identities=embodied_operations,
            )
        )

    @router.post(
        "/answers/stream",
        operation_id="answerStream",
        responses=streaming_errors,
        response_class=StreamingResponse,
    )
    async def answer_stream(request: AnswerRequest, connection: Request) -> StreamingResponse:
        chunks = iter(
            current_service().ask_stream(
                content_input(request.question),
                limit=request.limit,
                memory_type=request.memory_type,
                reference_at=request.reference_at,
                scope=request.scope,
                link_identities=embodied_operations,
            )
        )
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mindbridge-rest-answer")
        context = copy_context()

        try:
            first = await _first_chunk_or_disconnect(
                chunks,
                connection=connection,
                worker=worker,
                context=context,
            )
        except BaseException:
            worker.shutdown(wait=False)
            raise
        if first is _STREAM_EXHAUSTED:
            worker.shutdown(wait=False)
            raise RuntimeError("answer stream ended without a terminal result")
        return _AnswerStreamingResponse(
            first,
            chunks,
            worker=worker,
            context=context,
        )

    _add_context_route(router, current_service, responses=model_errors)
    _add_capture_routes(
        router,
        current_service,
        model_errors=model_errors,
        standard_errors=standard_errors,
    )
    _add_gated_routes(
        router,
        current_service,
        identity_operations=identity_operations,
        embodied_operations=embodied_operations,
        standard_statuses=standard_statuses,
        standard_errors=standard_errors,
        not_found_errors=not_found_errors,
    )
    return router


def _add_gated_routes(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    identity_operations: bool,
    embodied_operations: bool,
    standard_statuses: tuple[int, ...],
    standard_errors: dict[int | str, dict[str, Any]],
    not_found_errors: dict[int | str, dict[str, Any]],
) -> None:
    """Register the opt-in identity and embodied routes, split out to keep `_v1_router` readable."""
    if embodied_operations:
        embodied_errors = error_responses(
            *standard_statuses,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_501_NOT_IMPLEMENTED,
            status.HTTP_502_BAD_GATEWAY,
        )
        _add_embodied_routes(router, current_service, responses=embodied_errors)
    if identity_operations:
        _add_identity_routes(
            router,
            current_service,
            not_found_errors=not_found_errors,
            standard_errors=standard_errors,
        )


def _add_capture_routes(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    model_errors: dict[int | str, dict[str, Any]],
    standard_errors: dict[int | str, dict[str, Any]],
) -> None:
    """Register the fast-capture plane: `capture`, `settle`, and `pending_captures`.

    These are always on, unlike the identity and embodied routes below: they are ordinary
    application operations on the caller's own records, not administrative authority over a
    person.
    """

    @router.post(
        "/capture",
        response_model=MemoryResponse,
        status_code=status.HTTP_201_CREATED,
        operation_id="captureMemory",
        responses=model_errors,
    )
    def capture_memory(request: MemoryCreate) -> MemoryResponse:
        record = current_service().capture(
            content_input(request.content),
            occurred_at=request.occurred_at,
            occurred_end=request.occurred_end,
            metadata=request.metadata,
            memory_type=request.memory_type,
            context=request.context,
        )
        return MemoryResponse.model_validate(record)

    @router.post(
        "/settle",
        response_model=SettleResponse,
        operation_id="settleCaptures",
        responses=model_errors,
    )
    def settle_captures(request: SettleRequest) -> SettleResponse:
        return SettleResponse(
            settled=current_service().settle(
                limit=request.limit,
                max_attempts=request.max_attempts,
                memory_ids=request.memory_ids,
            )
        )

    @router.get(
        "/pending_captures",
        response_model=PendingCapturesResponse,
        operation_id="pendingCaptures",
        responses=standard_errors,
    )
    def pending_captures_route(
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
        memory_ids: Annotated[list[str] | None, Query(min_length=1, max_length=100)] = None,
    ) -> PendingCapturesResponse:
        return PendingCapturesResponse.model_validate(
            {
                "items": current_service().pending_captures(
                    limit=limit,
                    memory_ids=memory_ids,
                )
            }
        )


def _add_embodied_routes(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    responses: dict[int | str, dict[str, Any]],
) -> None:
    """Register `analyze_speech` and `analyze_faces`, gated by the host's `embodied_operations`."""

    @router.post(
        "/speech",
        response_model=SpeechResponse,
        operation_id="analyzeSpeech",
        responses=responses,
    )
    def analyze_speech(request: AnalyzeRequest) -> SpeechResponse:
        return SpeechResponse(segments=current_service().speech(request.memory_id))

    @router.post(
        "/faces",
        response_model=FacesResponse,
        operation_id="analyzeFaces",
        responses=responses,
    )
    def analyze_faces(request: AnalyzeRequest) -> FacesResponse:
        return FacesResponse(observations=current_service().faces(request.memory_id))


def _add_identity_routes(
    router: APIRouter,
    current_service: Callable[[], _Memory],
    *,
    not_found_errors: dict[int | str, dict[str, Any]],
    standard_errors: dict[int | str, dict[str, Any]],
) -> None:
    """Register naming, reading, unlinking, and erasing identities.

    Gated by the host's `identity_operations`: naming and erasing a person is host authority,
    exercised through the caller, exactly as it is on MCP.
    """

    @router.post(
        "/identities",
        response_model=RegisterResponse,
        operation_id="registerIdentity",
        responses=not_found_errors,
    )
    def register_identity(request: IdentityRegisterRequest) -> RegisterResponse:
        current_service().register_identity(
            request.identity_id,
            request.name,
            relationship=request.relationship,
        )
        return RegisterResponse(registered=True)

    @router.get(
        "/identities/{identity_id}",
        response_model=IdentityResponse,
        operation_id="getIdentity",
        responses=standard_errors,
    )
    def get_identity(identity_id: _MemoryId) -> IdentityResponse:
        return IdentityResponse(identity=current_service().identity(identity_id))

    @router.post(
        "/identities/{alias_id}/unlink",
        response_model=UnlinkResponse,
        operation_id="unlinkIdentity",
        responses=standard_errors,
    )
    def unlink_identity(alias_id: _MemoryId) -> UnlinkResponse:
        return UnlinkResponse(restored_identity_id=current_service().unlink_identity(alias_id))

    @router.delete(
        "/identities/{identity_id}",
        response_model=ForgetResponse,
        operation_id="forgetIdentity",
        responses=not_found_errors,
    )
    def forget_identity(identity_id: _MemoryId) -> ForgetResponse:
        return ForgetResponse(erasure=current_service().forget_identity(identity_id))

    @router.post(
        "/identities/{identity_id}/consent",
        response_model=RecordConsentResponse,
        operation_id="recordConsent",
        responses=not_found_errors,
    )
    def record_consent(identity_id: _MemoryId, request: ConsentRequest) -> RecordConsentResponse:
        record = current_service().record_consent(
            identity_id,
            request.state,
            note=request.note,
        )
        return RecordConsentResponse(
            operation=None if record is None else _operation_response(record)
        )

    @router.get(
        "/identities/{identity_id}/consent",
        response_model=ConsentResponse,
        operation_id="getConsent",
        responses=standard_errors,
    )
    def get_consent(identity_id: _MemoryId) -> ConsentResponse:
        return ConsentResponse(consent=current_service().consent(identity_id))

    @router.get(
        "/export",
        response_model=ExportResponse,
        operation_id="exportSubject",
        responses=not_found_errors,
    )
    def export_subject(
        identity_id: Annotated[str | None, Query(min_length=1)] = None,
        memory_ids: Annotated[list[str] | None, Query(min_length=1, max_length=100)] = None,
    ) -> ExportResponse:
        return _export_response(
            current_service().export(identity_id=identity_id, memory_ids=memory_ids)
        )

    @router.post(
        "/retention",
        response_model=RetentionResponse,
        operation_id="applyRetention",
        responses=standard_errors,
    )
    def apply_retention(request: RetentionRequest) -> RetentionResponse:
        return RetentionResponse.model_validate(
            current_service().apply_retention(dry_run=request.dry_run)
        )


async def _first_chunk_or_disconnect(
    stream: Generator[AnswerChunk, None, AnswerResult],
    *,
    connection: Request,
    worker: ThreadPoolExecutor,
    context: Context,
) -> object:
    loop = asyncio.get_running_loop()

    def first_chunk() -> object:
        return next(stream, _STREAM_EXHAUSTED)

    async def wait_for_disconnect() -> None:
        while (await connection.receive()).get("type") != "http.disconnect":
            pass

    pending_chunk = loop.run_in_executor(worker, context.run, first_chunk)
    disconnected = asyncio.create_task(wait_for_disconnect())
    try:
        done, _pending = await asyncio.wait(
            (pending_chunk, disconnected),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnected not in done:
            return pending_chunk.result()
        raise asyncio.CancelledError
    except BaseException:
        # A running synchronous provider call cannot be interrupted safely. Queue close on its
        # pinned worker so disconnects and task cancellation release the response after it returns.
        pending_chunk.add_done_callback(_discard_future_result)
        closing = loop.run_in_executor(worker, context.run, stream.close)
        closing.add_done_callback(_discard_future_result)
        raise
    finally:
        disconnected.cancel()
        with suppress(asyncio.CancelledError):
            await disconnected


def _discard_future_result(future: asyncio.Future[Any]) -> None:
    with suppress(BaseException):
        future.result()


async def _answer_events(
    first: object,
    rest: Generator[AnswerChunk, None, AnswerResult],
    *,
    worker: ThreadPoolExecutor,
    context: Context,
) -> AsyncGenerator[bytes, None]:
    loop = asyncio.get_running_loop()
    chunk = first

    def next_chunk() -> object:
        return next(rest, _STREAM_EXHAUSTED)

    terminal = False
    try:
        while chunk is not _STREAM_EXHAUSTED:
            answer_chunk = cast(AnswerChunk, chunk)
            terminal = answer_chunk.result is not None
            yield _answer_event(answer_chunk)
            chunk = await loop.run_in_executor(worker, context.run, next_chunk)
        if not terminal:
            raise RuntimeError("answer stream ended without a terminal result")
    except Exception as error:
        trace.get_current_span().set_status(StatusCode.ERROR)
        response = exception_response(error)
        yield _sse_event("error", json.loads(bytes(response.body)))


class _AnswerStreamingResponse(StreamingResponse):
    """Own the prefetched synchronous stream for the complete ASGI send lifecycle."""

    def __init__(
        self,
        first: object,
        rest: Generator[AnswerChunk, None, AnswerResult],
        *,
        worker: ThreadPoolExecutor,
        context: Context,
    ) -> None:
        self._events = _answer_events(first, rest, worker=worker, context=context)
        self._rest = rest
        self._worker = worker
        self._context = context
        super().__init__(
            self._events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette sends `http.response.start` before it enters the body iterator. Closing only
            # in `_answer_events` therefore leaks the already-prefetched Memory operation whenever
            # headers or the first body frame cannot be sent.
            try:
                await self._events.aclose()
            finally:
                await _close_answer_stream(self._rest, worker=self._worker, context=self._context)


async def _close_answer_stream(
    stream: Generator[AnswerChunk, None, AnswerResult],
    *,
    worker: ThreadPoolExecutor,
    context: Context,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        closing = loop.run_in_executor(worker, context.run, stream.close)
        closing.add_done_callback(_discard_future_result)
        with suppress(asyncio.CancelledError):
            await asyncio.shield(closing)
    finally:
        worker.shutdown(wait=False)


def _answer_event(chunk: AnswerChunk) -> bytes:
    if chunk.result is None:
        payload: dict[str, object] = {"text": chunk.text}
        event = "delta"
    else:
        payload = AnswerResponse.model_validate(chunk.result).model_dump(mode="json")
        event = "result"
    return _sse_event(event, payload)


def _sse_event(event: str, payload: object) -> bytes:
    return (
        f"event: {event}\ndata: "
        + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + "\n\n"
    ).encode("utf-8")


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
                budget=context_budget(request.budget),
                reference_at=request.reference_at,
                scope=request.scope,
            )
        )


def _bundle_response(bundle: ContextBundle) -> ContextBundleResponse:
    """Publish `ContextBundle.document()`, the same projection MCP and the CLI publish."""
    return ContextBundleResponse.model_validate(bundle.document())


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
