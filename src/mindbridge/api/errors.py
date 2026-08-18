"""The one REST error contract: every code, its HTTP status, and what it tells a caller.

`app.py` and `events.py` build their responses from this table, and every route documents
itself from the same entries. A code therefore cannot reach a caller without also reaching
the OpenAPI document, which is what makes that document the contract rather than a partial
description of the success path. Adding an error means adding one row here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from fastapi import status
from fastapi.responses import JSONResponse

from mindbridge.contracts import ErrorResponse, ValidationIssue
from mindbridge.telemetry import current_trace_id


@dataclass(frozen=True, slots=True)
class ApiError:
    """One error code's status and the sentence a caller can act on.

    `description` is the response `message` wherever the cause carries no detail of its own;
    the handlers that do have one pass it to `error_response` instead.
    """

    status_code: int
    description: str


ERRORS: Final[dict[str, ApiError]] = {
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
    "task_broker_unavailable": ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "observation processing is temporarily unavailable",
    ),
}

TENANT_ERRORS: Final[tuple[str, ...]] = (
    "authentication_required",
    "authentication_failed",
    "tenant_access_denied",
    "request_validation_failed",
    "database_unavailable",
)
"""What every authenticated `/v1` operation can return whatever else it does.

Each route spreads this and then names only the codes its own use case adds, so the document
distinguishes "this operation can 404" from "every operation shares an auth failure".
"""


def responses(*codes: str) -> dict[int | str, dict[str, Any]]:
    """Document the exact codes one operation returns, grouped under their shared statuses.

    Returning `ErrorResponse` as the model for each status is what puts the envelope into
    `components.schemas`; without it a generated client parses no error at all.
    """
    grouped: dict[int, list[str]] = {}
    for code in codes:
        grouped.setdefault(ERRORS[code].status_code, []).append(code)
    return {
        status_code: {
            "model": ErrorResponse,
            "description": "".join(
                f"`{code}` — {ERRORS[code].description}\n\n" for code in group
            ).strip(),
        }
        for status_code, group in sorted(grouped.items())
    }


def error_response(
    code: str,
    *,
    message: str | None = None,
    issues: tuple[ValidationIssue, ...] = (),
) -> JSONResponse:
    """Render one error, taking its status from the same row the OpenAPI document reads."""
    error = ERRORS[code]
    body = ErrorResponse(
        code=code,
        message=message or error.description,
        trace_id=current_trace_id(),
        issues=issues,
    )
    return JSONResponse(status_code=error.status_code, content=body.model_dump())
