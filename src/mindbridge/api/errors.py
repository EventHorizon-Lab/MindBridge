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

# Re-exported: the message table lives in a FastAPI-free module so MCP can share it.
from mindbridge.api.messages import MESSAGE_BY_CODE as MESSAGE_BY_CODE
from mindbridge.api.messages import error_message
from mindbridge.exceptions import MindBridgeError, ValidationError

_LOGGER = logging.getLogger(__name__)
_RETRY_AFTER_SECONDS = "1"
# Subjects for these codes describe server state, not caller input, so an unauthenticated client
# gets the code, reason, and stage without the local path or row the failure names.
_PRIVATE_SUBJECT_CODES = frozenset({"index_unavailable", "internal_error", "storage_error"})

# The whole transport mapping, keyed by `reason` alone. Keying on the reason rather than on the
# exception class or the raise site is the contract: one condition gets one status wherever it is
# raised from, so a raise site cannot invent a second answer for a condition already mapped, and
# adding a reason means adding a row here. `MindBridgeError.reason` is a closed vocabulary; a
# reason with no row falls back to its code, which is deliberately coarse.
REASON_STATUS: Mapping[str, int] = {
    "memory_not_found": status.HTTP_404_NOT_FOUND,
    "speaker_not_found": status.HTTP_404_NOT_FOUND,
    "identity_not_found": status.HTTP_404_NOT_FOUND,
    # The request body and one inline media value are the same condition seen from two sides, and
    # both are fixed by sending less, so both say 413 rather than blaming the provider with 502.
    "payload_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
    "input_invalid": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "unknown_field": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "unsupported_modality": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "unexpected": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "schema_unsupported": status.HTTP_500_INTERNAL_SERVER_ERROR,
    # `io_failed` is the coarse label the storage wrapper puts on any failure it cannot classify,
    # programming errors included, and it is deliberately not retryable. 503 would tell a client
    # the opposite, so it says 500. `instance_unusable` is the same shape: a closed or forked
    # instance cannot be revived by the caller repeating the call.
    "io_failed": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "instance_unusable": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "backend_not_configured": status.HTTP_501_NOT_IMPLEMENTED,
    "auth_failed": status.HTTP_502_BAD_GATEWAY,
    "quota_exhausted": status.HTTP_502_BAD_GATEWAY,
    "request_rejected": status.HTTP_502_BAD_GATEWAY,
    "response_invalid": status.HTTP_502_BAD_GATEWAY,
    "output_truncated": status.HTTP_502_BAD_GATEWAY,
    "asset_unavailable": status.HTTP_502_BAD_GATEWAY,
    "asset_changed": status.HTTP_502_BAD_GATEWAY,
    "model_failed": status.HTTP_502_BAD_GATEWAY,
    # Every retryable reason lands here, and `test_every_retryable_reason_says_503` proves it:
    # an agent that reads 503 plus `Retry-After` must not have to also read the reason.
    "connection_failed": status.HTTP_503_SERVICE_UNAVAILABLE,
    "timeout": status.HTTP_503_SERVICE_UNAVAILABLE,
    "rate_limited": status.HTTP_503_SERVICE_UNAVAILABLE,
    "data_dir_in_use": status.HTTP_503_SERVICE_UNAVAILABLE,
    "flush_failed": status.HTTP_503_SERVICE_UNAVAILABLE,
    "index_missing": status.HTTP_503_SERVICE_UNAVAILABLE,
}
# The fallback for the raise sites that pass no reason at all. Coarse on purpose: it says only
# whether the failure is upstream or local, which is all an unclassified raise can honestly say.
STATUS_BY_CODE: Mapping[str, int] = {
    "mindbridge_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "internal_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
    "validation_error": status.HTTP_422_UNPROCESSABLE_CONTENT,
    "memory_not_found": status.HTTP_404_NOT_FOUND,
    "speaker_not_found": status.HTTP_404_NOT_FOUND,
    "identity_not_found": status.HTTP_404_NOT_FOUND,
    "model_error": status.HTTP_502_BAD_GATEWAY,
    "model_output_truncated": status.HTTP_502_BAD_GATEWAY,
    "storage_error": status.HTTP_503_SERVICE_UNAVAILABLE,
    "index_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
}


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
        return exception_response(error)

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, error: Exception) -> JSONResponse:
        return exception_response(error)


def exception_response(error: Exception) -> JSONResponse:
    """Map an operation failure to the stable REST envelope, including during SSE."""
    if isinstance(error, MindBridgeError):
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
    return _public_status(error), error_message(error)


def _public_status(error: MindBridgeError) -> int:
    if error.reason is not None:
        status_code = REASON_STATUS.get(error.reason)
        if status_code is not None:
            return status_code
    return STATUS_BY_CODE.get(error.code, status.HTTP_500_INTERNAL_SERVER_ERROR)


def _http_error(status_code: int) -> tuple[str, str]:
    if status_code == status.HTTP_404_NOT_FOUND:
        return "not_found", "route does not exist"
    if status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
        return "method_not_allowed", "method is not allowed for this route"
    return "http_error", "the HTTP request could not be completed"


def _trace_id() -> str:
    return f"trace_{uuid4().hex}"
