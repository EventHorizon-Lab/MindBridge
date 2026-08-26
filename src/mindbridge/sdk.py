"""Small asynchronous Python client for the stable MindBridge REST contract."""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
    MediaObjectInput,
    MediaUploadRequest,
    MediaUploadTicket,
    MemoryResult,
    ObservationProcessingJobView,
    ObservationReceipt,
    ObserveRequest,
    RecallRequest,
    RecallResult,
    RememberBatchRequest,
    RememberBatchResult,
    RememberRequest,
    RememberResult,
    ValidationIssue,
)
from mindbridge.core import MediaKind, SensorKind, media_kind_for_suffix
from mindbridge.file_integrity import sha256_file

_Response = TypeVar("_Response", bound=ContractModel)

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.5

_UPLOAD_CHUNK_BYTES = 1024 * 1024
"""How much of a file is held in memory at once, on the way to its digest and to storage."""

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
"""Statuses that describe a dependency being briefly unable, not a request being refused.

`503` is the one that was costing whole runs: a `model_unavailable` raised while an embedder or
generator restarts reached the caller as a terminal error, so a read that the server would have
served a second later was recorded as an infrastructure failure. `429`, `502` and `504` are the
same shape. Every other status is deliberately absent -- a 4xx is a request the server understood
and rejected, and repeating it only spends the rejection again.
"""


def _repeat_is_safe(request: ContractModel | None) -> bool:
    """Whether re-sending this request cannot durably duplicate what it may already have done.

    A retry is only ever safe against a *duplicate*, never against a slow server: the first
    attempt may have been applied and only its response lost, so the question is what a second
    application would do.

    A request with no body is a read and repeating it changes nothing. A write is safe when its
    contract carries `idempotency_key` at all, because that field's documented meaning is "omit
    it and one is derived from the content": `observe` keys on (tenant, device, boot, sequence)
    and `remember`, `feedback` and `forget` on a digest of the request minus the key itself, so
    an identical resend answers `duplicate` with the first outcome rather than writing twice.
    This function is deliberately asked of the *type* rather than of the value. Requiring a
    caller-supplied key would have made the retry inert on every default call -- omission is the
    supported default, and the payload here is serialised once outside the retry loop, so a
    repeat is byte-identical and lands on the same derived key. A batch is safe only if every
    member of it is, since the server applies them individually.

    A write whose contract has no such field is not retried, which is what keeps this honest as
    the API grows: a new endpoint has to state its idempotency before the client will repeat it.
    """
    if request is None:
        return True
    if isinstance(request, RememberBatchRequest):
        return all(_repeat_is_safe(item) for item in request.memories)
    return "idempotency_key" in type(request).model_fields


class MindBridgeError(RuntimeError):
    """A transport failure or typed error returned by MindBridge."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        trace_id: str | None = None,
        issues: tuple[ValidationIssue, ...] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.trace_id = trace_id
        self.issues = issues


@dataclass(frozen=True, slots=True)
class ObservationJobEvent:
    """One streamed job state and the opaque ID that resumes after it."""

    event_id: str
    job: ObservationProcessingJobView


class MindBridge:
    """Call observe, remember, and recall through one typed asynchronous client."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        upload_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Wrap one HTTP client, plus a second one for bytes that do not go to MindBridge.

        `upload_client` sends presigned uploads, and exists because they must not carry
        `client`'s `Authorization` header: a presigned URL already carries credentials of its
        own in its query string, object storage refuses a request that arrives with two, and
        the API key has no business at a storage endpoint in the first place. One is created on
        first use when none is given, and `close()` releases whichever pools this client owns.
        """
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must not be negative")
        self._client = client
        self._retry_attempts = retry_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._sleep = sleep
        self._upload_client = upload_client

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
        retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> MindBridge:
        """Create a client for use with `async with` or an explicit `close()`.

        `retry_attempts=1` turns retrying off, for a caller that runs its own.
        """
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
            ),
            retry_attempts=retry_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    async def observe(self, request: ObserveRequest) -> ObservationReceipt:
        """Submit one observation through the production API."""
        return await self._post("v1/observations", request, ObservationReceipt)

    async def create_media_upload(self, request: MediaUploadRequest) -> MediaUploadTicket:
        """Ask where to send one file's bytes, without being lent any storage credentials.

        Repeated even though `MediaUploadRequest` names no `idempotency_key`, and the exception
        is argued rather than assumed: signing writes nothing down. A repeat that the server
        already answered costs one more short-lived URL for a key the digest had already fixed,
        never a second copy of anything -- so unlike the writes `_repeat_is_safe` guards, there
        is no duplicate for it to risk.
        """
        return await self._post("v1/media/uploads", request, MediaUploadTicket, repeat_is_safe=True)

    async def observe_file(
        self,
        path: str | Path,
        *,
        tenant_id: str,
        kind: MediaKind | None = None,
        occurred_at: datetime | None = None,
        ended_at: datetime | None = None,
        device_id: str = "sdk",
        boot_id: str = "sdk",
        sequence: int | None = None,
        sensor: SensorKind | None = None,
        idempotency_key: str | None = None,
    ) -> ObservationReceipt:
        """Store one local file as evidence and observe it, in one call.

        Reads the file twice -- once to digest it, once to send it -- and never holds more than
        a chunk of it, so the size of the file is a matter for the network rather than for
        memory. The digest names the object, so everything derived below is a function of the
        bytes: observing the same unchanged file again answers `duplicate` rather than storing
        a second copy, and a call that fails after the upload can simply be made again.

        `kind` is read from the file's extension, and passing one overrides that reading: an
        extension the declared kind contradicts is left off the object key rather than argued
        with, since the key is addressed by the digest and needs no extension to be read back.
        It has to be passed for a container this API does not recognize.

        The timestamps default to the file's modification time, which is stable across a repeat
        where `now` would not be. Pass `occurred_at` and `ended_at` together for anything with a
        duration -- `ended_at` is read against `occurred_at`, not against the file -- because
        nothing here decodes the media: left alone, the observation covers an instant and the
        evidence span registered for it is empty.
        """
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        implied_kind = media_kind_for_suffix(suffix)
        resolved_kind = kind or implied_kind
        if resolved_kind is None:
            raise ValueError(
                f"cannot infer a media kind from {suffix or 'a file with no extension'!r}; "
                "pass kind= to say what it holds"
            )
        # One stat, so the length that gets signed and the time that gets recorded describe the
        # same file. Hashing a multi-gigabyte one is seconds of blocking work besides, and
        # neither belongs on the loop that is also serving whatever else this client is doing.
        status = await asyncio.to_thread(file_path.stat)
        size_bytes = status.st_size
        sha256 = await asyncio.to_thread(sha256_file, file_path)
        ticket = await self.create_media_upload(
            MediaUploadRequest(
                tenant_id=tenant_id,
                sha256=sha256,
                size_bytes=size_bytes,
                # The extension travels only while it agrees with the kind being declared. One
                # this API has no opinion about is arbitrary text on the way to a signature,
                # and one that contradicts the declared kind would make the observation this
                # is uploading for unacceptable -- while a key addressed by its digest reads
                # back the same with no extension at all.
                suffix=suffix if resolved_kind is implied_kind else None,
            )
        )
        await self._upload(ticket.upload_url, file_path, size_bytes)
        started_at = occurred_at or datetime.fromtimestamp(status.st_mtime, tz=timezone.utc)
        return await self.observe(
            ObserveRequest(
                tenant_id=tenant_id,
                device_id=device_id,
                boot_id=boot_id,
                # Derived from the digest so two different files cannot collide onto one
                # observation: the ID the server derives is a function of exactly
                # (tenant, device, boot, sequence), and a fixed default would make every file
                # after the first a conflicting write against the first one's key.
                sequence=int(sha256[:15], 16) if sequence is None else sequence,
                sensor=sensor
                or (
                    SensorKind.MICROPHONE if resolved_kind is MediaKind.AUDIO else SensorKind.CAMERA
                ),
                media_objects=(
                    MediaObjectInput(
                        media_object_id=sha256,
                        kind=resolved_kind,
                        uri=ticket.uri,
                        sha256=sha256,
                        size_bytes=size_bytes,
                        created_at=started_at,
                        # Left unknown rather than guessed: nothing here decoded the file, and
                        # claiming the observation's own span as a playable length is a lie
                        # about the media that the evidence span already covers honestly.
                        duration_ms=None,
                    ),
                ),
                occurred_at=started_at,
                ended_at=ended_at or started_at,
                observed_at=started_at,
                idempotency_key=idempotency_key,
            )
        )

    async def remember(self, request: RememberRequest) -> RememberResult:
        """Retain one explicit memory, reporting whether this call is what created it."""
        return await self._post("v1/memories", request, RememberResult)

    async def remember_many(
        self,
        requests: Sequence[RememberRequest],
    ) -> tuple[RememberResult, ...]:
        """Retain up to 100 memories in one call, and so in one encoder round trip.

        A caller already holding N memories should hand over all N: the server encodes the
        whole batch in one request to its embedder instead of N, which is the difference
        between one round trip and N for the same work. Results come back in request order,
        each with its own `created` or `duplicate` status.
        """
        result = await self._post(
            "v1/memories/batch",
            RememberBatchRequest(memories=tuple(requests)),
            RememberBatchResult,
        )
        return result.memories

    async def record_feedback(self, request: FeedbackRequest) -> FeedbackReceipt:
        """Record one useful, wrong, missing, or correction signal."""
        return await self._post("v1/feedback", request, FeedbackReceipt)

    async def forget(self, request: ForgetRequest) -> ForgetReceipt:
        """Explicitly erase one exact memory or source observation."""
        return await self._post("v1/forget", request, ForgetReceipt)

    async def recall(self, request: RecallRequest) -> RecallResult:
        """Recall memories and grounded evidence through the production API.

        Retried even though `RecallRequest` names no idempotency key, because this is the read a
        briefly-unavailable model used to fail outright. The duplicate it risks is one extra
        recorded access against the memories it returns, which feeds strength and decay -- a
        bounded, directionally-honest cost, against losing the answer entirely.
        """
        return await self._post("v1/recall", request, RecallResult, repeat_is_safe=True)

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
        """Release the underlying HTTP connection pools."""
        await self._client.aclose()
        if self._upload_client is not None:
            await self._upload_client.aclose()

    async def _upload(self, upload_url: str, path: Path, size_bytes: int) -> None:
        """Send one file's bytes to the signed URL that was minted for exactly them.

        Not retried, and that is a choice rather than an omission: the PUT itself is safe to
        repeat -- the key is fixed by the digest, so a second attempt overwrites the same
        object -- but the URL is short-lived and a caller that still holds the file gets a
        correct retry for free by calling `observe_file` again, which re-signs and re-derives
        the same key and the same observation. A second retry loop here would only be able to
        replay one signature.

        `Content-Length` is set explicitly because the signature covers it: httpx would
        otherwise stream a chunked body, which is neither the length that was signed nor
        something an S3 PUT accepts.
        """
        if self._upload_client is None:
            self._upload_client = httpx.AsyncClient(timeout=self._client.timeout)
        try:
            response = await self._upload_client.put(
                upload_url,
                content=_file_chunks(path),
                headers={"Content-Length": str(size_bytes)},
            )
        except httpx.HTTPError as error:
            raise MindBridgeError(
                f"MindBridge media upload failed: {type(error).__name__}: {error}",
                code="transport_error",
            ) from error
        if not response.is_success:
            # The body is object storage's, not MindBridge's, so it is not parsed as the shared
            # error envelope; the status is what a caller can act on.
            raise MindBridgeError(
                "MindBridge media upload was rejected by object storage",
                code="media_upload_failed",
                status_code=response.status_code,
            )

    async def _post(
        self,
        path: str,
        request: ContractModel,
        response_type: type[_Response],
        *,
        repeat_is_safe: bool | None = None,
    ) -> _Response:
        return await self._request(
            "POST",
            path,
            response_type,
            request=request,
            repeat_is_safe=repeat_is_safe,
        )

    async def _request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        response_type: type[_Response],
        *,
        request: ContractModel | None = None,
        params: Mapping[str, str] | None = None,
        repeat_is_safe: bool | None = None,
    ) -> _Response:
        """Send one request, repeating it only while repeating it is safe and can help.

        There used to be no retry at all: every non-2xx became a terminal `MindBridgeError` and
        every transport error a terminal `transport_error`, so a 503 from a model that was
        restarting was final for the caller. A retry here is not a substitute for the server's own
        durability -- an observation is still made durable before its job runs -- it just stops a
        caller losing a read to an outage measured in seconds.

        `repeat_is_safe` defaults to what the request itself permits; a caller that knows better
        than the request passes it explicitly.
        """
        may_repeat = _repeat_is_safe(request) if repeat_is_safe is None else repeat_is_safe
        payload = request.model_dump(mode="json") if request is not None else None
        attempt = 0
        while True:
            attempt += 1
            last = not may_repeat or attempt >= self._retry_attempts
            try:
                response = await self._client.request(method, path, json=payload, params=params)
            except httpx.HTTPError as error:
                if last:
                    # The cause is the only thing that separates "wrong address", "nothing
                    # listening", and "timed out", and none of it reaches a caller who cannot
                    # read the server log.
                    raise MindBridgeError(
                        f"MindBridge {method} {path} failed: {type(error).__name__}: {error}",
                        code="transport_error",
                    ) from error
            else:
                if response.is_success:
                    return _decoded(response, response_type)
                if last or response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise _api_error(response)
            await self._backoff(attempt)

    async def _backoff(self, attempt: int) -> None:
        """Wait a uniformly random share of a doubling ceiling before repeating.

        Full jitter rather than a fixed delay because the failure this exists for is shared: the
        clients that see one model outage all see it at once, and a fixed backoff would have them
        all return together and re-create the load they are waiting out.
        """
        ceiling = self._retry_backoff_seconds * 2 ** (attempt - 1)
        await self._sleep(random.uniform(0.0, ceiling))


async def _file_chunks(path: Path) -> AsyncIterator[bytes]:
    """Stream one file a chunk at a time, off the event loop."""
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, _UPLOAD_CHUNK_BYTES):
            yield chunk


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
    return MindBridgeError(
        error.message,
        code=error.code,
        trace_id=error.trace_id,
        issues=error.issues,
    )


def _decoded(response: httpx.Response, response_type: type[_Response]) -> _Response:
    """Parse one successful body, keeping the status on the error a bad body raises."""
    try:
        return response_type.model_validate_json(response.content)
    except ValidationError as error:
        raise MindBridgeError(
            "MindBridge returned an invalid response",
            code="invalid_response",
            status_code=response.status_code,
        ) from error


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
        issues=error.issues,
    )
