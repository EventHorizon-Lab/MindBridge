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

from mindbridge.api.auth import AuthenticationError
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


class ErrorIssue(BaseModel):
    """One invalid request field."""

    location: tuple[str | int, ...]
    message: str
    type: str


class ErrorEnvelope(BaseModel):
    """The only error shape emitted by the REST API."""

    code: str
    message: str
    trace_id: str
    issues: tuple[ErrorIssue, ...] = ()


def error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Expose the shared error envelope in OpenAPI."""
    return {code: {"model": ErrorEnvelope} for code in status_codes}


def register_error_handlers(app: FastAPI) -> None:
    """Make framework, public, and unexpected failures share one envelope."""

    @app.exception_handler(AuthenticationError)
    async def authentication_error(_request: Request, _error: AuthenticationError) -> JSONResponse:
        return error_response(
            status.HTTP_401_UNAUTHORIZED,
            "authentication_error",
            "a valid bearer API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
            issues=issues,
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        code, message = _http_error(error.status_code)
        return error_response(error.status_code, code, message, headers=error.headers)

    @app.exception_handler(MindBridgeError)
    async def public_error(_request: Request, error: MindBridgeError) -> JSONResponse:
        status_code, message = _public_error(error)
        return error_response(status_code, error.code, message)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        trace_id = _trace_id()
        _LOGGER.error("unhandled API error", exc_info=error, extra={"trace_id": trace_id})
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "the request failed unexpectedly",
            trace_id=trace_id,
        )


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    issues: tuple[ErrorIssue, ...] = (),
    headers: Mapping[str, str] | None = None,
    trace_id: str | None = None,
) -> JSONResponse:
    body = ErrorEnvelope(
        code=code,
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
    if isinstance(error, ValidationError):
        return status.HTTP_422_UNPROCESSABLE_CONTENT, str(error) or "input is invalid"
    if isinstance(error, MemoryNotFoundError):
        return status.HTTP_404_NOT_FOUND, str(error) or "memory does not exist"
    if isinstance(error, SpeakerNotFoundError):
        return status.HTTP_404_NOT_FOUND, str(error) or "speaker does not exist"
    if isinstance(error, ModelError):
        return status.HTTP_502_BAD_GATEWAY, "model operation failed"
    if isinstance(error, IndexUnavailableError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "memory index is unavailable"
    if isinstance(error, StorageError):
        return status.HTTP_503_SERVICE_UNAVAILABLE, "memory storage is unavailable"
    return status.HTTP_500_INTERNAL_SERVER_ERROR, "the request failed unexpectedly"


def _http_error(status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found", "route does not exist"
    if status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        return "method_not_allowed", "method is not allowed for this route"
    return "http_error", "the HTTP request could not be completed"


def _trace_id() -> str:
    return f"trace_{uuid4().hex}"
