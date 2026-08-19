"""Server-sent events carrying observation processing progress."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse

from mindbridge.api.auth import TenantApiKeyAuthenticator, TenantPrincipal, require_tenant
from mindbridge.api.errors import TENANT_ERRORS, code_for, error_body, responses
from mindbridge.application.kernel import MemoryKernel
from mindbridge.contracts import Identifier, ObservationProcessingJobView

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_KEEPALIVE_SECONDS = 15.0
_KEEPALIVE_FRAME = ": keepalive\n\n"


def register_job_event_routes(
    app: FastAPI,
    kernel: MemoryKernel,
    authenticator: TenantApiKeyAuthenticator,
) -> None:
    """Expose observation processing progress as one resumable event stream."""
    tenant_authentication = Depends(authenticator)

    @app.get(
        "/v1/jobs/{job_id}/events",
        operation_id="streamObservationJob",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": (
                    "One `job` event per observed change, each carrying the complete job view. "
                    "Reconnect with `Last-Event-ID` to skip states already received. After the "
                    "stream opens, a failure arrives as an `error` event carrying this same "
                    "`ErrorResponse` envelope rather than as a status code."
                ),
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            **responses(*TENANT_ERRORS, "job_not_found", "internal_error"),
        },
    )
    async def stream_observation_job(
        job_id: Identifier,
        tenant_id: Identifier,
        request: Request,
        principal: TenantPrincipal = tenant_authentication,
    ) -> StreamingResponse:
        require_tenant(principal, tenant_id)
        # Resolve authorization and existence before the 200 and its streaming headers are sent;
        # afterwards the status can no longer report a missing or forbidden job.
        await kernel.get_observation_job(tenant_id, job_id)
        return StreamingResponse(
            _job_events(
                kernel,
                tenant_id,
                job_id,
                _requested_resume_point(request.headers.get("last-event-id")),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )


async def _job_events(
    kernel: MemoryKernel,
    tenant_id: str,
    job_id: str,
    after_updated_at: datetime | None,
) -> AsyncIterator[str]:
    """Frame each job view, keeping an idle connection alive between changes."""
    views = kernel.watch_observation_job(
        tenant_id,
        job_id,
        after_updated_at=after_updated_at,
    )
    try:
        while True:
            upcoming = asyncio.ensure_future(views.__anext__())
            try:
                while not upcoming.done():
                    done, _still_running = await asyncio.wait(
                        {upcoming}, timeout=_KEEPALIVE_SECONDS
                    )
                    if not done:
                        yield _KEEPALIVE_FRAME
                view = upcoming.result()
            except StopAsyncIteration:
                return
            except Exception as error:
                # Every failure, not the two that were foreseen. A silent close is
                # byte-identical to normal completion for the client -- `aiter_lines` just
                # ends -- so anything escaping here without a frame was read as "the job
                # settled". `translate_transient_database_errors` deliberately re-raises a
                # psycopg error whose sqlstate it does not recognise, and a malformed row
                # raises `MemoryIntegrityError`; neither was caught, and both told the caller
                # the opposite of what happened. `Exception`, not `BaseException`, so
                # cancellation still propagates.
                # Distinguish a broken stream from a settled job; a silent close would look
                # identical to normal completion.
                yield _error_frame(error)
                return
            finally:
                upcoming.cancel()
            yield _job_frame(view)
    finally:
        await views.aclose()


def _job_frame(view: ObservationProcessingJobView) -> str:
    return f"id: {_event_id(view.updated_at)}\nevent: job\ndata: {view.model_dump_json()}\n\n"


def _error_frame(error: Exception) -> str:
    """Frame the same code and sentence the non-streaming routes would have returned.

    Read through the same mapping the handlers use, so a stream cannot report a failure by a
    name the rest of the API stopped using -- a divergence nothing would catch until a client
    was already connected. `internal_error` is the honest answer for a failure the contract
    has no word for: the caller still learns the stream broke and still gets a `trace_id`,
    which is strictly more than the silence this replaced.
    """
    body = error_body(code_for(error) or "internal_error")
    return f"event: error\ndata: {body.model_dump_json()}\n\n"


def _event_id(moment: datetime) -> int:
    """Microseconds since the epoch, the only value on a job row that always increases."""
    return (moment - _EPOCH) // timedelta(microseconds=1)


def _requested_resume_point(last_event_id: str | None) -> datetime | None:
    """Read a resume point, ignoring an unusable one rather than failing the stream.

    A client only ever echoes an ID this server produced, so a malformed value costs at most one
    duplicate event; every event carries complete state.
    """
    if last_event_id is None:
        return None
    try:
        microseconds = int(last_event_id.strip())
        if microseconds < 0:
            return None
        return _EPOCH + timedelta(microseconds=microseconds)
    except (ValueError, OverflowError):
        return None
