"""Stable HTTP errors for the small public API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from mindbridge.exceptions import (
    IndexUnavailableError,
    MemoryNotFoundError,
    MindBridgeError,
    ModelError,
    SpeakerNotFoundError,
    StorageError,
    ValidationError,
)

_LOGGER = logging.getLogger(__name__)
_RETRY_AFTER_SECONDS = "1"
# Subjects for these codes describe server state, not caller input, so an unauthenticated client
# gets the code, reason, and stage without the local path or row the failure names.
_PRIVATE_SUBJECT_CODES = frozenset({"index_unavailable", "internal_error", "storage_error"})


class ErrorIssue(BaseModel):
    """One invalid request field."""

    location: tuple[str | int, ...]
    message: str
    type: str


class ErrorEnvelope(BaseModel):
    """The only error shape emitted by the REST API."""

    code: str
    reason: str | None = None
    retryable: bool = False
    stage: str | None = None
    subject: str | None = None
    message: str
    trace_id: str
    issues: tuple[ErrorIssue, ...] = ()


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Expose the shared error envelope in OpenAPI."""
    return {code: {"model": ErrorEnvelope} for code in status_codes}


def register_error_handlers(app: FastAPI) -> None:
    """Make framework, public, and unexpected failures share one envelope."""

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        issues = tuple(
            ErrorIssue(
                location=tuple(issue["loc"]),
                message=issue["msg"],
                type=issue["type"],
            )
            for issue in error.errors()
        )
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            ValidationError.code,
            "request validation failed",
            reason="input_invalid",
            issues=issues,
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        code, message = _http_error(error.status_code)
        return error_response(error.status_code, code, message, headers=error.headers)

    @app.exception_handler(MindBridgeError)
    async def public_error(_request: Request, error: MindBridgeError) -> JSONResponse:
        status_code, message = _public_error(error)
        return error_response(
            status_code,
            error.code,
            message,
            reason=error.reason,
            retryable=error.retryable,
            stage=error.stage,
            subject=None if error.code in _PRIVATE_SUBJECT_CODES else error.subject,
            headers=(
                {"Retry-After": _RETRY_AFTER_SECONDS}
                if status_code == status.HTTP_503_SERVICE_UNAVAILABLE and error.retryable
                else None
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        trace_id = _trace_id()
        _LOGGER.error("unhandled API error", exc_info=error, extra={"trace_id": trace_id})
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "the request failed unexpectedly",
            reason="unexpected",
            trace_id=trace_id,
        )


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    reason: str | None = None,
    retryable: bool = False,
    stage: str | None = None,
    subject: str | None = None,
    issues: tuple[ErrorIssue, ...] = (),
    headers: Mapping[str, str] | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        code=code,
        reason=reason,
        retryable=retryable,
        stage=stage,
        subject=subject,
        message=message,
        trace_id=trace_id or _trace_id(),
        issues=issues,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def _public_error(error: MindBridgeError) -> tuple[int, str]:
    """Map a public failure onto a status that says whether the same call can ever succeed."""
    if isinstance(error, ValidationError):
        return status.HTTP_422_UNPROCESSABLE_CONTENT, str(error) or "input is invalid"
    if isinstance(error, MemoryNotFoundError):
        return status.HTTP_404_NOT_FOUND, str(error) or "memory does not exist"
    if isinstance(error, SpeakerNotFoundError):
        return status.HTTP_404_NOT_FOUND, str(error) or "speaker does not exist"
    if isinstance(error, ModelError):
        return _model_status(error), str(error) or "model operation failed"
    if isinstance(error, IndexUnavailableError):
        return _storage_status(error), str(error) or "memory index is unavailable"
    if isinstance(error, StorageError):
        return _storage_status(error), str(error) or "memory storage is unavailable"
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "memory operation failed"


def _model_status(error: ModelError) -> int:
    if error.reason == "backend_not_configured":
        return status.HTTP_501_NOT_IMPLEMENTED
    if error.reason == "unsupported_modality":
        return status.HTTP_422_UNPROCESSABLE_CONTENT
    return status.HTTP_503_SERVICE_UNAVAILABLE if error.retryable else status.HTTP_502_BAD_GATEWAY


def _storage_status(error: StorageError) -> int:
    if error.reason == "schema_unsupported":
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    return status.HTTP_503_SERVICE_UNAVAILABLE


def _http_error(status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found", "route does not exist"
    if status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        return "method_not_allowed", "method is not allowed for this route"
    return "http_error", "the HTTP request could not be completed"


def _trace_id() -> str:
    return f"trace_{uuid4().hex}"
