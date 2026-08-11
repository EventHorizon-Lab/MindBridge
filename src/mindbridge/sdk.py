"""Small asynchronous Python client for the stable MindBridge REST contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeVar
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from mindbridge.contracts import (
    ContractModel,
    ErrorResponse,
    FeedbackReceipt,
    FeedbackRequest,
    ForgetReceipt,
    ForgetRequest,
    MemoryView,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberRequest,
)

_Response = TypeVar("_Response", bound=ContractModel)


class MindBridgeClientError(RuntimeError):
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


class AsyncMindBridge:
    """Call observe, remember, and recall through one typed asynchronous client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @classmethod
    def connect(
        cls,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> AsyncMindBridge:
        """Create a client that owns its connection pool until `close()` is called."""
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

    async def remember(self, request: RememberRequest) -> MemoryView:
        """Retain one explicit memory through the production API."""
        return await self._post("v1/memories", request, MemoryView)

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record one useful, wrong, missing, or correction signal."""
        return await self._post("v1/feedback", request, FeedbackReceipt)

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        """Explicitly erase one exact memory or source observation."""
        return await self._post("v1/forget", request, ForgetReceipt)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Recall memories and grounded evidence through the production API."""
        return await self._post("v1/recall", request, RecallResult)

    async def get_memory(self, tenant_id: str, memory_id: str) -> MemoryView:
        """Read one tenant-owned memory."""
        return await self._request(
            "GET",
            f"v1/memories/{quote(memory_id, safe='')}",
            MemoryView,
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
            raise MindBridgeClientError(
                "MindBridge request failed",
                code="transport_error",
            ) from error
        if not response.is_success:
            raise _api_error(response)
        try:
            return response_type.model_validate_json(response.content)
        except ValidationError as error:
            raise MindBridgeClientError(
                "MindBridge returned an invalid response",
                code="invalid_response",
                status_code=response.status_code,
            ) from error


def _api_error(response: httpx.Response) -> MindBridgeClientError:
    try:
        error = ErrorResponse.model_validate_json(response.content)
    except ValidationError:
        return MindBridgeClientError(
            "MindBridge request was rejected",
            code="http_error",
            status_code=response.status_code,
        )
    return MindBridgeClientError(
        error.message,
        code=error.code,
        status_code=response.status_code,
        trace_id=error.trace_id,
    )
