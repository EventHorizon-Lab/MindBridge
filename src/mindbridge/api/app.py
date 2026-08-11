"""FastAPI adapter for the shared MindBridge use cases."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import Lifespan

from mindbridge.application import MemoryKernel
from mindbridge.contracts import (
    ErrorResponse,
    HealthResponse,
    Identifier,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
    ValidationIssue,
)
from mindbridge.core import (
    DomainInvariantError,
    IdempotencyConflictError,
    JobNotFoundError,
    MemoryIntegrityError,
    MemoryNotFoundError,
    ModelOutputError,
    ModelUnavailableError,
    ObjectStorageError,
    TaskBrokerError,
)


def create_app(
    kernel: MemoryKernel,
    *,
    lifespan: Lifespan[FastAPI] | None = None,
) -> FastAPI:
    """Create a side-effect-free REST adapter around one memory kernel."""
    app = FastAPI(title="MindBridge", version="0.1.0", lifespan=lifespan)
    _register_request_error_handlers(app)
    _register_runtime_error_handlers(app)

    @app.get("/healthz", response_model=HealthResponse, operation_id="health")
    async def health() -> HealthResponse:
        return HealthResponse(trace_id=_new_trace_id())

    @app.post(
        "/v1/observations",
        response_model=ObservationReceipt,
        status_code=status.HTTP_202_ACCEPTED,
        operation_id="observe",
    )
    async def observe(request: ObserveRequest) -> ObservationReceipt:
        return await kernel.observe(request)

    @app.post(
        "/v1/memories",
        response_model=MemoryView,
        status_code=status.HTTP_201_CREATED,
        operation_id="remember",
    )
    async def remember(request: RememberRequest) -> MemoryView:
        return await kernel.remember(request)

    @app.post(
        "/v1/recall",
        response_model=RecallResult,
        operation_id="recall",
    )
    async def recall(request: RecallRequest) -> RecallResult:
        return await kernel.recall(request)

    @app.get(
        "/v1/memories/{memory_id}",
        response_model=MemoryView,
        operation_id="getMemory",
    )
    async def get_memory(memory_id: Identifier, tenant_id: Identifier) -> MemoryView:
        return await kernel.get_memory(tenant_id, memory_id)

    @app.get(
        "/v1/jobs/{job_id}",
        response_model=ObservationProcessingJobView,
        operation_id="getObservationJob",
    )
    async def get_observation_job(
        job_id: Identifier,
        tenant_id: Identifier,
    ) -> ObservationProcessingJobView:
        return await kernel.get_observation_job(tenant_id, job_id)

    return app


def _register_request_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        response = ErrorResponse(
            code="request_validation_failed",
            message="request validation failed",
            trace_id=_new_trace_id(),
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
        is_conflict = isinstance(error, IdempotencyConflictError)
        response = ErrorResponse(
            code="idempotency_conflict" if is_conflict else "domain_invariant_failed",
            message=str(error),
            trace_id=_new_trace_id(),
        )
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT
            if is_conflict
            else status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=response.model_dump(),
        )


def _register_runtime_error_handlers(app: FastAPI) -> None:
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

    @app.exception_handler(ModelUnavailableError)
    async def handle_model_unavailable(
        _request: Request,
        _error: ModelUnavailableError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="model_unavailable",
            message="memory model is unavailable",
        )

    @app.exception_handler(ModelOutputError)
    async def handle_model_output_error(
        _request: Request,
        _error: ModelOutputError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_502_BAD_GATEWAY,
            code="model_output_invalid",
            message="memory model returned invalid output",
        )

    @app.exception_handler(ObjectStorageError)
    async def handle_object_storage_error(
        _request: Request,
        _error: ObjectStorageError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="object_storage_unavailable",
            message="evidence media is unavailable",
        )

    @app.exception_handler(TaskBrokerError)
    async def handle_task_broker_error(
        _request: Request,
        _error: TaskBrokerError,
    ) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            code="task_broker_unavailable",
            message="observation processing is temporarily unavailable",
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


def _new_trace_id() -> str:
    return f"trace_{uuid4().hex}"


def _error_response(status_code: int, *, code: str, message: str) -> JSONResponse:
    response = ErrorResponse(code=code, message=message, trace_id=_new_trace_id())
    return JSONResponse(status_code=status_code, content=response.model_dump())
