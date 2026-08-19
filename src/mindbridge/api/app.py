"""FastAPI adapter for the shared MindBridge use cases."""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import Lifespan

from mindbridge.api.aml import AmlSettings, register_aml_routes
from mindbridge.api.auth import (
    AuthenticationError,
    TenantApiKeyAuthenticator,
    TenantPrincipal,
    require_tenant,
)
from mindbridge.api.errors import (
    RUNTIME_ERROR_CODES,
    TENANT_ERRORS,
    ErrorCode,
    error_response,
    responses,
)
from mindbridge.api.events import register_job_event_routes
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    DeletionListRequest,
    DeletionPage,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    HealthResponse,
    Identifier,
    MemoryResult,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    RememberResult,
    ValidationIssue,
)
from mindbridge.core import (
    DomainInvariantError,
    EnumerationLimitExceededError,
    IdempotencyConflictError,
)
from mindbridge.models import Generator
from mindbridge.telemetry import current_trace_id

_EMBEDDING_ERRORS: Final[tuple[ErrorCode, ...]] = (
    "model_request_failed",
    "model_output_invalid",
    "model_unavailable",
)
"""What any operation that encodes a vector before answering the caller can return."""

_EVIDENCE_ERRORS: Final[tuple[ErrorCode, ...]] = (
    "memory_integrity_failed",
    "object_storage_unavailable",
)
"""What any operation that resolves and signs evidence before replying can return.

Signing is on the request path, not only in the worker: `_memory_result` reads the evidence
rows and presigns each media object, so a broken object store or a memory pointing at missing
or cross-tenant evidence surfaces to whoever asked, not to a background job.
"""


def build_app(
    kernel: MemoryKernel,
    *,
    authenticator: TenantApiKeyAuthenticator,
    lifespan: Lifespan[FastAPI] | None = None,
    aml: tuple[AmlSettings, Generator] | None = None,
) -> FastAPI:
    """Create a side-effect-free REST adapter around one memory kernel."""
    app = FastAPI(title="MindBridge", version="0.1.0", lifespan=lifespan)
    tenant_authentication = Depends(authenticator)
    _register_request_error_handlers(app)
    _register_runtime_error_handlers(app)
    _register_deletion_routes(app, kernel, authenticator)
    register_job_event_routes(app, kernel, authenticator)

    @app.get("/healthz", response_model=HealthResponse, operation_id="health")
    async def health() -> HealthResponse:
        return HealthResponse(trace_id=current_trace_id())

    @app.post(
        "/v1/observations",
        response_model=ObservationReceipt,
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="observe",
        responses=responses(
            *TENANT_ERRORS,
            "idempotency_conflict",
            "domain_invariant_failed",
            "task_broker_unavailable",
        ),
    )
    async def observe(
        request: ObserveRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> ObservationReceipt:
        require_tenant(principal, request.tenant_id)
        return await kernel.observe(request)

    @app.post(
        "/v1/memories",
        response_model=RememberResult,
        status_code=status.HTTP_201_CREATED,
        operation_id="remember",
        responses=responses(
            *TENANT_ERRORS,
            "idempotency_conflict",
            "domain_invariant_failed",
            *_EMBEDDING_ERRORS,
            *_EVIDENCE_ERRORS,
        ),
    )
    async def remember(
        request: RememberRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> RememberResult:
        require_tenant(principal, request.tenant_id)
        return await kernel.remember(request)

    @app.post(
        "/v1/feedback",
        response_model=FeedbackReceipt,
        status_code=status.HTTP_201_CREATED,
        operation_id="recordFeedback",
        responses=responses(
            *TENANT_ERRORS,
            "memory_not_found",
            "memory_deleted",
            "idempotency_conflict",
            "domain_invariant_failed",
            # A correction writes a new memory version, so it encodes one before replying.
            *_EMBEDDING_ERRORS,
        ),
    )
    async def record_feedback(
        request: FeedbackRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> FeedbackReceipt:
        require_tenant(principal, request.tenant_id)
        return await kernel.record_feedback(request)

    @app.post(
        "/v1/recall",
        response_model=RecallResult,
        operation_id="recall",
        responses=responses(
            *TENANT_ERRORS,
            "enumeration_limit_exceeded",
            *_EMBEDDING_ERRORS,
            *_EVIDENCE_ERRORS,
        ),
    )
    async def recall(
        request: RecallRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> RecallResult:
        require_tenant(principal, request.tenant_id)
        return await kernel.recall(request)

    @app.get(
        "/v1/memories/{memory_id}",
        response_model=MemoryResult,
        operation_id="getMemory",
        responses=responses(
            *TENANT_ERRORS,
            "memory_not_found",
            "memory_deleted",
            *_EVIDENCE_ERRORS,
        ),
    )
    async def get_memory(
        memory_id: Identifier,
        tenant_id: Identifier,
        principal: TenantPrincipal = tenant_authentication,
    ) -> MemoryResult:
        require_tenant(principal, tenant_id)
        return await kernel.get_memory(tenant_id, memory_id)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=ObservationProcessingJobView,
        operation_id="getObservationJob",
        responses=responses(*TENANT_ERRORS, "job_not_found"),
    )
    async def get_observation_job(
        job_id: Identifier,
        tenant_id: Identifier,
        principal: TenantPrincipal = tenant_authentication,
    ) -> ObservationProcessingJobView:
        require_tenant(principal, tenant_id)
        return await kernel.get_observation_job(tenant_id, job_id)

    if aml is not None:
        aml_settings, aml_generator = aml
        register_aml_routes(app, kernel, aml_generator, settings=aml_settings)

    return app


def _register_deletion_routes(
    app: FastAPI,
    kernel: MemoryKernel,
    authenticator: TenantApiKeyAuthenticator,
) -> None:
    """Expose command, status, and edge propagation over one shared use case."""
    tenant_authentication = Depends(authenticator)

    @app.post(
        "/v1/forget",
        response_model=ForgetReceipt,
        operation_id="forget",
        responses=responses(
            *TENANT_ERRORS,
            "forget_target_not_found",
            "idempotency_conflict",
            # Erasing the bytes is part of the command, and a failure to reach them is
            # reported rather than swallowed: the tombstone is marked failed and re-raised.
            "object_storage_unavailable",
        ),
    )
    async def forget(
        request: ForgetRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> ForgetReceipt:
        require_tenant(principal, request.tenant_id)
        return await kernel.forget(request)

    @app.get(
        "/v1/deletions",
        response_model=DeletionPage,
        operation_id="listDeletions",
        # A cursor from another tenant, or one whose tombstone is gone, is a domain failure
        # rather than an empty page: an edge device must not read truncation as completion.
        responses=responses(*TENANT_ERRORS, "domain_invariant_failed"),
    )
    async def list_deletions(
        tenant_id: Identifier,
        cursor: Identifier | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 100,
        principal: TenantPrincipal = tenant_authentication,
    ) -> DeletionPage:
        require_tenant(principal, tenant_id)
        return await kernel.list_deletions(
            DeletionListRequest(tenant_id=tenant_id, cursor=cursor, limit=limit)
        )

    @app.get(
        "/v1/deletions/{tombstone_id}",
        response_model=ForgetReceipt,
        operation_id="getForgetStatus",
        responses=responses(*TENANT_ERRORS, "forget_target_not_found"),
    )
    async def get_forget_status(
        tombstone_id: Identifier,
        tenant_id: Identifier,
        principal: TenantPrincipal = tenant_authentication,
    ) -> ForgetReceipt:
        require_tenant(principal, tenant_id)
        return await kernel.get_forget_status(tenant_id, tombstone_id)


def _register_request_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AuthenticationError)
    async def handle_authentication_error(
        _request: Request,
        error: AuthenticationError,
    ) -> JSONResponse:
        response = error_response(error.code)
        if response.status_code == status.HTTP_401_UNAUTHORIZED:
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        return error_response(
            "request_validation_failed",
            issues=tuple(
                ValidationIssue(
                    location=tuple(str(part) for part in issue["loc"]),
                    message=issue["msg"],
                    code=issue["type"],
                )
                for issue in error.errors()
            ),
        )

    @app.exception_handler(DomainInvariantError)
    async def handle_domain_error(
        _request: Request,
        error: DomainInvariantError,
    ) -> JSONResponse:
        code: ErrorCode
        if isinstance(error, IdempotencyConflictError):
            code = "idempotency_conflict"
        elif isinstance(error, EnumerationLimitExceededError):
            code = "enumeration_limit_exceeded"
        else:
            code = "domain_invariant_failed"
        return error_response(code, message=str(error))


def _register_runtime_error_handlers(app: FastAPI) -> None:
    """Answer every runtime failure from the one table, so none escapes the envelope."""

    async def handle(_request: Request, error: Exception) -> JSONResponse:
        # Starlette dispatches by MRO, so a subclass of a registered error arrives here too.
        code = next(
            RUNTIME_ERROR_CODES[ancestor]
            for ancestor in type(error).__mro__
            if ancestor in RUNTIME_ERROR_CODES
        )
        return error_response(code)

    for exception in RUNTIME_ERROR_CODES:
        app.add_exception_handler(exception, handle)
