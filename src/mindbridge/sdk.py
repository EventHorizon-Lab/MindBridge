"""Small asynchronous Python client for the stable MindBridge REST contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from mindbridge.contracts import (
    ContractModel,
    DeletionListRequest,
    DeletionPage,
    ErrorResponse,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    MemoryResult,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)

_Response = TypeVar("_Response", bound=ContractModel)


class MindBridgeError(RuntimeError):
    """A transport failure or typed error returned by MindBridge."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        trace_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.trace_id = trace_id


@dataclass(frozen=True, slots=True)
class ObservationJobEvent:
    """One streamed job state and the opaque ID that resumes after it."""

    event_id: str
    job: ObservationProcessingJobView


class MindBridge:
    """Call observe, remember, and recall through one typed asynchronous client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> MindBridge:
        return self

    async def __aexit__(self, *_error: object) -> None:
        await self.close()

    @classmethod
    def connect(
        cls,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> MindBridge:
        """Create a client for use with `async with` or an explicit `close()`."""
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if api_key is not None and not api_key.strip():
            raise ValueError("api_key must not be blank when provided")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key is not None else None
        return cls(
            httpx.AsyncClient(
                base_url=base_url.rstrip("/") + "/",
                headers=headers,
                timeout=timeout_seconds,
            )
        )

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        """Submit one observation through the production API."""
        return await self._post("v1/observations", request, ObservationReceipt)

    async def remember(self, request: RememberRequest) -> MemoryResult:
        """Retain one explicit memory through the production API."""
        return await self._post("v1/memories", request, MemoryResult)

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record one useful, wrong, missing, or correction signal."""
        return await self._post("v1/feedback", request, FeedbackReceipt)

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        """Explicitly erase one exact memory or source observation."""
        return await self._post("v1/forget", request, ForgetReceipt)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Recall memories and grounded evidence through the production API."""
        return await self._post("v1/recall", request, RecallResult)

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryResult:
        """Read one tenant-owned memory."""
        return await self._request(
            "GET",
            f"v1/memories/{quote(memory_id, safe='')}",
            MemoryResult,
            params={"tenant_id": tenant_id},
        )

    async def get_forget_status(
        self,
        tenant_id: str,
        tombstone_id: str,
    ) -> ForgetReceipt:
        """Read durable deletion propagation state."""
        return await self._request(
            "GET",
            f"v1/deletions/{quote(tombstone_id, safe='')}",
            ForgetReceipt,
            params={"tenant_id": tenant_id},
        )

    async def list_deletions(self, request: DeletionListRequest) -> DeletionPage:
        """List one stable page of tenant deletion barriers."""
        parameters = {"tenant_id": request.tenant_id, "limit": str(request.limit)}
        if request.cursor is not None:
            parameters["cursor"] = request.cursor
        return await self._request(
            "GET",
            "v1/deletions",
            DeletionPage,
            params=parameters,
        )

    async def get_observation_job(
        self,
        tenant_id: str,
        job_id: str,
    ) -> ObservationProcessingJobView:
        """Read one durable observation processing state."""
        return await self._request(
            "GET",
            f"v1/jobs/{quote(job_id, safe='')}",
            ObservationProcessingJobView,
            params={"tenant_id": tenant_id},
        )

    async def stream_observation_job(
        self,
        tenant_id: str,
        job_id: str,
        *,
        last_event_id: str | None = None,
    ) -> AsyncIterator[ObservationJobEvent]:
        """Follow one job's progress, resuming after `last_event_id` when it is supplied.

        Every event carries a complete job view, so resuming needs only the last ID received
        rather than a replayed history. The stream ends when the attempt settles or the server
        closes its window; reconnecting is the caller's decision.
        """
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        try:
            async with self._client.stream(
                "GET",
                f"v1/jobs/{quote(job_id, safe='')}/events",
                params={"tenant_id": tenant_id},
                headers=headers,
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise _api_error(response)
                frame: dict[str, str] = {}
                async for line in response.aiter_lines():
                    if line.startswith(":"):
                        continue
                    if line:
                        name, _, value = line.partition(":")
                        frame[name] = value.strip()
                        continue
                    event = _streamed_job_event(frame)
                    frame = {}
                    if event is not None:
                        yield event
        except httpx.HTTPError as error:
            raise MindBridgeError(
                f"MindBridge job event stream failed: {type(error).__name__}: {error}",
                code="transport_error",
            ) from error

    async def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        await self._client.aclose()

    async def _post(
        self,
        path: str,
        request: ContractModel,
        response_type: type[_Response],
    ) -> _Response:
        return await self._request("POST", path, response_type, request=request)

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        response_type: type[_Response],
        *,
        request: ContractModel | None = None,
        params: Mapping[str, str] | None = None,
    ) -> _Response:
        try:
            response = await self._client.request(
                method,
                path,
                json=request.model_dump(mode="json") if request is not None else None,
                params=params,
            )
        except httpx.HTTPError as error:
            # The cause is the only thing that separates "wrong address", "nothing listening",
            # and "timed out", and none of it reaches a caller who cannot read the server log.
            raise MindBridgeError(
                f"MindBridge {method} {path} failed: {type(error).__name__}: {error}",
                code="transport_error",
            ) from error
        if not response.is_success:
            raise _api_error(response)
        try:
            return response_type.model_validate_json(response.content)
        except ValidationError as error:
            raise MindBridgeError(
                "MindBridge returned an invalid response",
                code="invalid_response",
                status_code=response.status_code,
            ) from error


def _streamed_job_event(frame: Mapping[str, str]) -> ObservationJobEvent | None:
    """Read one framed event, ignoring event types this client version does not know."""
    event = frame.get("event")
    if event is None:
        return None
    if event == "error":
        raise _streamed_error(frame.get("data", ""))
    if event != "job":
        return None
    try:
        return ObservationJobEvent(
            event_id=frame.get("id", ""),
            job=ObservationProcessingJobView.model_validate_json(frame.get("data", "")),
        )
    except ValidationError as error:
        raise MindBridgeError(
            "MindBridge returned an invalid response",
            code="invalid_response",
        ) from error


def _streamed_error(payload: str) -> MindBridgeError:
    try:
        error = ErrorResponse.model_validate_json(payload)
    except ValidationError:
        return MindBridgeError("MindBridge stream reported an error", code="stream_error")
    return MindBridgeError(error.message, code=error.code, trace_id=error.trace_id)


def _api_error(response: httpx.Response) -> MindBridgeError:
    try:
        error = ErrorResponse.model_validate_json(response.content)
    except ValidationError:
        return MindBridgeError(
            "MindBridge request was rejected",
            code="http_error",
            status_code=response.status_code,
        )
    return MindBridgeError(
        error.message,
        code=error.code,
        status_code=response.status_code,
        trace_id=error.trace_id,
    )
