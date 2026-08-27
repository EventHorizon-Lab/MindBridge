"""The one REST error contract: every code, its HTTP status, and what it tells a caller.

`app.py` and `events.py` build their responses from this table, and every route documents
itself from the same entries. A code therefore cannot reach a caller without also reaching
the OpenAPI document, which is what makes that document the contract rather than a partial
description of the success path. Adding an error means adding one row here.

The tables also name the failure for the faces that are not REST: `code_for` is what lets the
job stream and the MCP tools report a failure by the same code REST would, instead of each
deciding for itself. One mapping, three renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

from fastapi import status
from fastapi.responses import JSONResponse

from mindbridge.contracts import ErrorResponse, ValidationIssue
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
    UnsupportedModalityError,
)
from mindbridge.telemetry import current_trace_id

ErrorCode = Literal[
    "authentication_required",
    "authentication_failed",
    "tenant_access_denied",
    "forget_target_not_found",
    "memory_not_found",
    "job_not_found",
    "idempotency_conflict",
    "memory_deleted",
    "request_validation_failed",
    "domain_invariant_failed",
    "enumeration_limit_exceeded",
    "memory_integrity_failed",
    "model_output_invalid",
    "model_request_failed",
    "unsupported_modality_route",
    "database_unavailable",
    "model_unavailable",
    "object_storage_unavailable",
    "task_broker_unavailable",
    "internal_error",
]
"""The closed set of error codes, so a misspelling is a type error rather than a KeyError.

`ERRORS` is keyed by this, and every function that takes a code takes this: `error_response`
runs inside an exception handler, where an unknown key would turn a documented 4xx into an
unhandled 500. `test_every_documented_error_code_is_a_real_code` covers the other direction,
a member with no row.
"""


@dataclass(frozen=True, slots=True)
class ApiError:
    """One error code's status and the sentence a caller can act on.

    `description` is the response `message` wherever the cause carries no detail of its own;
    the handlers that do have one pass it to `error_response` instead.
    """

    status_code: int
    description: str


ERRORS: Final[dict[ErrorCode, ApiError]] = {
    "authentication_required": ApiError(
        status.HTTP_401_UNAUTHORIZED,
        "a valid bearer API key is required",
    ),
    "authentication_failed": ApiError(
        status.HTTP_401_UNAUTHORIZED,
        "the bearer API key is invalid",
    ),
    "tenant_access_denied": ApiError(
        status.HTTP_403_FORBIDDEN,
        "the authenticated tenant cannot access this resource",
    ),
    "forget_target_not_found": ApiError(
        status.HTTP_404_NOT_FOUND,
        "forget target or deletion tombstone does not exist",
    ),
    "memory_not_found": ApiError(
        status.HTTP_404_NOT_FOUND,
        "memory does not exist",
    ),
    "job_not_found": ApiError(
        status.HTTP_404_NOT_FOUND,
        "observation processing job does not exist",
    ),
    "idempotency_conflict": ApiError(
        status.HTTP_409_CONFLICT,
        "the idempotency key already stores different content",
    ),
    "memory_deleted": ApiError(
        status.HTTP_410_GONE,
        "memory content was explicitly deleted",
    ),
    "request_validation_failed": ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "request validation failed",
    ),
    "domain_invariant_failed": ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "the request is well-formed but violates a memory invariant",
    ),
    "enumeration_limit_exceeded": ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "the enumerate scope exceeds the bound; narrow the filters",
    ),
    "memory_integrity_failed": ApiError(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "stored memory is inconsistent",
    ),
    "model_output_invalid": ApiError(
        status.HTTP_502_BAD_GATEWAY,
        "memory model returned invalid output",
    ),
    "model_request_failed": ApiError(
        status.HTTP_502_BAD_GATEWAY,
        "memory model rejected its configured request",
    ),
    "unsupported_modality_route": ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "the configured model route does not support the supplied modalities",
    ),
    "database_unavailable": ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "memory storage is temporarily unavailable",
    ),
    "model_unavailable": ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "memory model is unavailable",
    ),
    "object_storage_unavailable": ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "evidence media is unavailable",
    ),
    "internal_error": ApiError(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "the request failed for a reason the server did not anticipate",
    ),
    "task_broker_unavailable": ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "observation processing is temporarily unavailable",
    ),
}

TENANT_ERRORS: Final[tuple[ErrorCode, ...]] = (
    "authentication_required",
    "authentication_failed",
    "tenant_access_denied",
    "request_validation_failed",
    "database_unavailable",
    "internal_error",
)
"""What every authenticated `/v1` operation can return whatever else it does.

Each route spreads this and then names only the codes its own use case adds, so the document
distinguishes "this operation can 404" from "every operation shares an auth failure".
"""


RUNTIME_ERROR_CODES: Final[dict[type[Exception], ErrorCode]] = {
    ForgetTargetNotFoundError: "forget_target_not_found",
    MemoryNotFoundError: "memory_not_found",
    JobNotFoundError: "job_not_found",
    MemoryDeletedError: "memory_deleted",
    MemoryIntegrityError: "memory_integrity_failed",
    ModelOutputError: "model_output_invalid",
    UnsupportedModalityError: "unsupported_modality_route",
    ModelRequestError: "model_request_failed",
    DatabaseUnavailableError: "database_unavailable",
    ModelUnavailableError: "model_unavailable",
    ObjectStorageError: "object_storage_unavailable",
    TaskBrokerError: "task_broker_unavailable",
}
"""Every failure the app answers without reading anything off the exception itself.

One row registers the handler and names the code, so an error class and its published code
cannot be added apart. Subclasses are covered: Starlette dispatches by MRO and the handler
looks the code up the same way.
"""


DOMAIN_INVARIANT_ERROR_CODES: Final[dict[type[Exception], ErrorCode]] = {
    IdempotencyConflictError: "idempotency_conflict",
    EnumerationLimitExceededError: "enumeration_limit_exceeded",
    DomainInvariantError: "domain_invariant_failed",
}
"""The failures whose message is the caller's, so the handler passes `str(error)` through."""

_ALL_ERROR_CODES: Final[dict[type[Exception], ErrorCode]] = {
    **RUNTIME_ERROR_CODES,
    **DOMAIN_INVARIANT_ERROR_CODES,
}


def code_for(error: BaseException) -> ErrorCode | None:
    """Name the code this failure reports, or None if no row in the contract claims it.

    Walks the MRO for the same reason the REST handler does -- a subclass of a mapped error is
    still that error -- and is shared so the streaming and MCP faces cannot drift from REST on
    what a failure is called. None means "nothing here has a word for this", which each face
    answers in its own terms rather than inventing a code.
    """
    return next(
        (
            _ALL_ERROR_CODES[ancestor]
            for ancestor in type(error).__mro__
            if ancestor in _ALL_ERROR_CODES
        ),
        None,
    )


def error_body(
    code: ErrorCode,
    *,
    message: str | None = None,
    issues: tuple[ValidationIssue, ...] = (),
) -> ErrorResponse:
    """Build the envelope every face sends, so only the wrapper around it differs."""
    return ErrorResponse(
        code=code,
        message=message or ERRORS[code].description,
        trace_id=current_trace_id(),
        issues=issues,
    )


def responses(*codes: ErrorCode) -> dict[int | str, dict[str, Any]]:
    """Document the exact codes one operation returns, grouped under their shared statuses.

    Returning `ErrorResponse` as the model for each status is what puts the envelope into
    `components.schemas`; without it a generated client parses no error at all.
    """
    grouped: dict[int, list[ErrorCode]] = {}
    for code in codes:
        grouped.setdefault(ERRORS[code].status_code, []).append(code)
    return {
        status_code: {
            "model": ErrorResponse,
            "description": "\n\n".join(f"`{code}` — {ERRORS[code].description}" for code in group),
        }
        for status_code, group in sorted(grouped.items())
    }


def error_response(
    code: ErrorCode,
    *,
    message: str | None = None,
    issues: tuple[ValidationIssue, ...] = (),
) -> JSONResponse:
    """Render one error, taking its status from the same row the OpenAPI document reads."""
    body = error_body(code, message=message, issues=issues)
    return JSONResponse(status_code=ERRORS[code].status_code, content=body.model_dump())
