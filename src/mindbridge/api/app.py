"""FastAPI adapter for the shared MindBridge use cases."""

from __future__ import annotations

from typing import Annotated

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
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import (
    DeletionListRequest,
    DeletionPage,
    ErrorResponse,
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
    ValidationIssue,
)
from mindbridge.core import (
    DatabaseUnavailableError,
    DomainInvariantError,
    EnumerationLimitExceededError,
    ForgetTargetNotFoundError,
    IdempotencyConflictError,
    JobNotFoundError,
    MemoryDeletedError,
    MemoryIntegrityError,
    MemoryNotFoundError,
    ModelOutputError,
    ModelRequestError,
    ModelUnavailableError,
    ObjectStorageError,
    TaskBrokerError,
)
from mindbridge.models import Generator
from mindbridge.telemetry import current_trace_id


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

    @app.get("/healthz", response_model=HealthResponse, operation_id="health")
    async def health() -> HealthResponse:
        return HealthResponse(trace_id=current_trace_id())

    @app.post(
        "/v1/observations",
        response_model=ObservationReceipt,
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="observe",
    )
    async def observe(
        request: ObserveRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> ObservationReceipt:
        require_tenant(principal, request.tenant_id)
        return await kernel.observe(request)

    @app.post(
        "/v1/memories",
        response_model=MemoryResult,
        status_code=status.HTTP_201_CREATED,
        operation_id="remember",
    )
    async def remember(
        request: RememberRequest,
        principal: TenantPrincipal = tenant_authentication,
    ) -> MemoryResult:
        require_tenant(principal, request.tenant_id)
        return await kernel.remember(request)

    @app.post(
        "/v1/feedback",
        response_model=FeedbackReceipt,
        status_code=status.HTTP_201_CREATED,
        operation_id="recordFeedback",
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
        response = _error_response(error.status_code, code=error.code, message=error.message)
        if error.status_code == status.HTTP_401_UNAUTHORIZED:
            response.headers["WWW-Authenticate"] = "Bearer"
        return response

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        response = ErrorResponse(
            code="request_validation_failed",
            message="request validation failed",
            trace_id=current_trace_id(),
            issues=tuple(
                ValidationIssue(
                    location=tuple(str(part) for part in issue["loc"]),
                    message=issue["msg"],
                    code=issue["type"],
                )
                for issue in error.errors()
            ),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=response.model_dump()
        )

    @app.exception_handler(DomainInvariantError)
    async def handle_domain_error(
        _request: Request,
        error: DomainInvariantError,
    ) -> JSONResponse:
        if isinstance(error, IdempotencyConflictError):
            code = "idempotency_conflict"
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(error, EnumerationLimitExceededError):
            code = "enumeration_limit_exceeded"
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        else:
            code = "domain_invariant_failed"
            status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
        response = ErrorResponse(
            code=code,
            message=str(error),
            trace_id=current_trace_id(),
        )
        return JSONResponse(
            status_code=status_code,
            content=response.model_dump(),
        )


def _register_runtime_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ForgetTargetNotFoundError)
    async def handle_forget_target_not_found(
        _request: Request,
        _error: ForgetTargetNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            code="forget_target_not_found",
            message="forget target or deletion tombstone does not exist",
        )

    @app.exception_handler(MemoryDeletedError)
    async def handle_memory_deleted(
        _request: Request,
        _error: MemoryDeletedError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_410_GONE,
            code="memory_deleted",
            message="memory content was explicitly deleted",
        )

    @app.exception_handler(MemoryNotFoundError)
    async def handle_memory_not_found(
        _request: Request,
        _error: MemoryNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            code="memory_not_found",
            message="memory does not exist",
        )

    @app.exception_handler(JobNotFoundError)
    async def handle_job_not_found(
        _request: Request,
        _error: JobNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            code="job_not_found",
            message="observation processing job does not exist",
        )

    @app.exception_handler(ModelRequestError)
    @app.exception_handler(ModelOutputError)
    async def handle_model_protocol_error(
        _request: Request,
        error: ModelOutputError | ModelRequestError,
    ) -> JSONResponse:
        code, message = {
            ModelOutputError: (
                "model_output_invalid",
                "memory model returned invalid output",
            ),
            ModelRequestError: (
                "model_request_failed",
                "memory model rejected its configured request",
            ),
        }[type(error)]
        return _error_response(
            status.HTTP_502_BAD_GATEWAY,
            code=code,
            message=message,
        )

    @app.exception_handler(DatabaseUnavailableError)
    @app.exception_handler(ModelUnavailableError)
    @app.exception_handler(ObjectStorageError)
    @app.exception_handler(TaskBrokerError)
    async def handle_dependency_unavailable(
        _request: Request,
        error: DatabaseUnavailableError
        | ModelUnavailableError
        | ObjectStorageError
        | TaskBrokerError,
    ) -> JSONResponse:
        code, message = {
            DatabaseUnavailableError: (
                "database_unavailable",
                "memory storage is temporarily unavailable",
            ),
            ModelUnavailableError: ("model_unavailable", "memory model is unavailable"),
            ObjectStorageError: (
                "object_storage_unavailable",
                "evidence media is unavailable",
            ),
            TaskBrokerError: (
                "task_broker_unavailable",
                "observation processing is temporarily unavailable",
            ),
        }[type(error)]
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code=code,
            message=message,
        )

    @app.exception_handler(MemoryIntegrityError)
    async def handle_memory_integrity_error(
        _request: Request,
        _error: MemoryIntegrityError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="memory_integrity_failed",
            message="stored memory is inconsistent",
        )


def _error_response(status_code: int, *, code: str, message: str) -> JSONResponse:
    response = ErrorResponse(code=code, message=message, trace_id=current_trace_id())
    return JSONResponse(status_code=status_code, content=response.model_dump())
