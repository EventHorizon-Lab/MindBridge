"""Authenticated serving for local SentenceTransformers embedding models."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hmac
import http.client
import json
import os
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from mindbridge.application.capabilities import (
    Embedder,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.cli import parser as build_parser
from mindbridge.configuration import require_environment_value
from mindbridge.core import EmbeddingSpaceReference, MediaKind, ModelRequestError
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
)
from mindbridge.models.plugins import close_model, load_embedder

_MAX_MEDIA_BYTES = 64 * 1024 * 1024
_MAX_REQUEST_BYTES = 96 * 1024 * 1024
_MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_BATCH_INPUTS = 32
_DEFAULT_BATCH_WAIT_MS = 2.0
_DEFAULT_MEDIA_IO_CONCURRENCY = 8
_MediaOrigin = tuple[str, str, int]
_MEDIA_TYPES = {
    "image_url": MediaKind.IMAGE,
    "video_url": MediaKind.VIDEO,
    "audio_url": MediaKind.AUDIO,
}
_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "video/mp4": ".mp4",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
}


@dataclass(slots=True)
class _PendingEmbed:
    request: EmbedRequest
    result: asyncio.Future[EmbedResult]


class _AdaptiveBatchingEmbedder:
    """Coalesce concurrent requests while keeping query and document lanes independent."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        max_batch_inputs: int,
        batch_wait_ms: float,
    ) -> None:
        if max_batch_inputs <= 0:
            raise ValueError("max batch inputs must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch wait must not be negative")
        self._embedder = embedder
        self._max_batch_inputs = max_batch_inputs
        self._batch_wait_seconds = batch_wait_ms / 1_000
        self._queues = {task: deque[_PendingEmbed]() for task in EmbedTask}
        self._workers: dict[EmbedTask, asyncio.Task[None]] = {}

    @property
    def space_reference(self) -> EmbeddingSpaceReference:
        return self._embedder.space_reference

    async def embed(self, request: EmbedRequest) -> EmbedResult:
        if not request.inputs:
            return await self._embedder.embed(request)
        result: asyncio.Future[EmbedResult] = asyncio.get_running_loop().create_future()
        self._queues[request.task].append(_PendingEmbed(request, result))
        worker = self._workers.get(request.task)
        if worker is None or worker.done():
            self._workers[request.task] = asyncio.create_task(self._serve(request.task))
        return await result

    async def _serve(self, task: EmbedTask) -> None:
        queue = self._queues[task]
        try:
            while queue:
                batch = self._take_batch(queue)
                input_count = sum(len(item.request.inputs) for item in batch)
                if input_count < self._max_batch_inputs and not queue and self._batch_wait_seconds:
                    await asyncio.sleep(self._batch_wait_seconds)
                    batch.extend(
                        self._take_batch(queue, available=self._max_batch_inputs - input_count)
                    )
                batch = [item for item in batch if not item.result.cancelled()]
                if batch:
                    await self._embed_batch(task, batch)
        finally:
            self._workers.pop(task, None)

    def _take_batch(
        self,
        queue: deque[_PendingEmbed],
        *,
        available: int | None = None,
    ) -> list[_PendingEmbed]:
        remaining = self._max_batch_inputs if available is None else available
        batch: list[_PendingEmbed] = []
        while queue:
            pending = queue[0]
            if pending.result.cancelled():
                queue.popleft()
                continue
            size = len(pending.request.inputs)
            if size > remaining and (batch or available is not None):
                break
            batch.append(queue.popleft())
            remaining -= size
            if remaining <= 0:
                break
        return batch

    async def _embed_batch(self, task: EmbedTask, batch: list[_PendingEmbed]) -> None:
        inputs = tuple(input_value for item in batch for input_value in item.request.inputs)
        try:
            result = await self._embedder.embed(EmbedRequest(inputs=inputs, task=task))
            if len(result.embeddings) != len(inputs):
                raise RuntimeError("embedder returned the wrong adaptive batch size")
        except asyncio.CancelledError:
            for item in batch:
                item.result.cancel()
            raise
        except Exception as error:
            # A batch is this server's scheduling artefact, not something any caller asked for,
            # so one caller's input must not fail the others. Attribute the failure by re-running
            # each request alone; a single-input batch has nobody else to blame and stops the
            # recursion. This also recovers the common batch-wide case -- an OOM caused by the
            # merged size -- because the retries are small enough to fit.
            if len(batch) == 1:
                item = batch[0]
                if not item.result.done():
                    item.result.set_exception(error)
                return
            for item in batch:
                await self._embed_batch(task, [item])
            return
        offset = 0
        for item in batch:
            size = len(item.request.inputs)
            if not item.result.done():
                item.result.set_result(EmbedResult(result.embeddings[offset : offset + size]))
            offset += size


def _reject_unusable_options(
    *,
    api_key: str,
    media_io_concurrency: int,
    max_batch_inputs: int,
    batch_wait_ms: float,
) -> None:
    """Reject the flags before `lifespan` loads the model.

    The batching embedder validates its own two arguments, but `lifespan` builds it only after
    `load_embedder` has put several GiB of weights on the GPU -- so `--max-batch-inputs 0` used
    to download and load a model and then die. Checked here, `create_app` refuses first.
    """
    if not api_key.strip():
        raise ValueError("embedding API key must not be empty")
    if media_io_concurrency <= 0:
        raise ValueError("media I/O concurrency must be positive")
    if max_batch_inputs <= 0:
        raise ValueError("maximum batch inputs must be positive")
    if batch_wait_ms < 0:
        raise ValueError("batch wait must not be negative")


def create_app(
    *,
    api_key: str,
    embedder_config: Mapping[str, object] | None = None,
    embedder: Embedder | None = None,
    media_origins: Sequence[str] = (),
    max_batch_inputs: int = _DEFAULT_MAX_BATCH_INPUTS,
    batch_wait_ms: float = _DEFAULT_BATCH_WAIT_MS,
    media_io_concurrency: int = _DEFAULT_MEDIA_IO_CONCURRENCY,
) -> FastAPI:
    """Build one single-model service; injection keeps contract tests model-free."""
    _reject_unusable_options(
        api_key=api_key,
        media_io_concurrency=media_io_concurrency,
        max_batch_inputs=max_batch_inputs,
        batch_wait_ms=batch_wait_ms,
    )
    config = dict(embedder_config or {})
    allowed_media_origins = _allowed_media_origins(media_origins)

    def serving(loaded: Embedder) -> _AdaptiveBatchingEmbedder:
        return _AdaptiveBatchingEmbedder(
            loaded,
            max_batch_inputs=max_batch_inputs,
            batch_wait_ms=batch_wait_ms,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Both names resolve to the same factory. The older one is used because entry points
        # are written at install time: a GPU host that pulls this source and restarts without
        # re-running `uv sync` would otherwise get "unknown embedder plugin" from a command
        # that worked yesterday.
        loaded = embedder or load_embedder("jina", config)
        app.state.embedder = serving(loaded)
        try:
            yield
        finally:
            if embedder is None:
                await close_model(loaded)

    app = FastAPI(title="MindBridge SentenceTransformers embedding service", lifespan=lifespan)
    app.state.media_io_slots = asyncio.Semaphore(media_io_concurrency)
    if embedder is not None:
        app.state.embedder = serving(embedder)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        _authorize(request, api_key)
        return {
            "object": "list",
            "data": [{"id": _model_id(config), "object": "model"}],
        }

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> dict[str, object]:
        _authorize(request, api_key)
        payload = await _request_json(request)
        return await _embedding_response(
            payload,
            cast(Embedder, request.app.state.embedder),
            config,
            allowed_media_origins,
            cast(asyncio.Semaphore, request.app.state.media_io_slots),
        )

    return app


async def _embedding_response(
    payload: object,
    embedder: Embedder,
    config: Mapping[str, object],
    allowed_media_origins: frozenset[_MediaOrigin],
    media_io_slots: asyncio.Semaphore,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")
    expected_model = _model_id(config)
    if payload.get("model") != expected_model:
        raise HTTPException(status_code=404, detail="unknown model")
    dimension = _dimension(config)
    if payload.get("dimensions", dimension) != dimension:
        raise HTTPException(status_code=400, detail="unsupported embedding dimension")
    if payload.get("encoding_format", "float") != "float":
        raise HTTPException(status_code=400, detail="only float encoding is supported")
    try:
        vectors = await _embed(
            embedder,
            payload.get("input"),
            allowed_media_origins,
            media_io_slots,
        )
    except HTTPException:
        raise
    except ModelRequestError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail="embedding failed") from error
    return {
        "object": "list",
        "model": expected_model,
        "data": [
            {"object": "embedding", "index": index, "embedding": list(vector)}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


async def _embed(
    embedder: Embedder,
    raw_input: object,
    allowed_media_origins: frozenset[_MediaOrigin],
    media_io_slots: asyncio.Semaphore,
) -> tuple[tuple[float, ...], ...]:
    with tempfile.TemporaryDirectory(prefix="mindbridge-sentence-transformers-") as temp_dir:
        directory = Path(temp_dir)
        parsed = tuple(
            await asyncio.gather(
                *(
                    _materialize_input(
                        sample,
                        directory,
                        index,
                        allowed_media_origins,
                        media_io_slots,
                    )
                    for index, sample in enumerate(_samples(raw_input))
                )
            )
        )
        tasks = {task for task, _input in parsed}
        if len(tasks) != 1:
            raise HTTPException(status_code=400, detail="one request cannot mix embedding tasks")
        result = await embedder.embed(
            EmbedRequest(
                inputs=tuple(model_input for _task, model_input in parsed),
                task=parsed[0][0],
            )
        )
        return tuple(item.values for item in result.embeddings)


async def _materialize_input(
    sample: object,
    directory: Path,
    sample_index: int,
    allowed_media_origins: frozenset[_MediaOrigin],
    media_io_slots: asyncio.Semaphore,
) -> tuple[EmbedTask, ModelInput]:
    async with media_io_slots:
        return await asyncio.to_thread(
            _model_input,
            sample,
            directory,
            sample_index,
            allowed_media_origins,
        )


def _samples(raw_input: object) -> list[object]:
    if isinstance(raw_input, str):
        return [raw_input]
    if not isinstance(raw_input, list) or not raw_input:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if all(isinstance(item, str) for item in raw_input):
        return list(raw_input)
    if all(isinstance(item, dict) and "role" in item for item in raw_input):
        return [raw_input]
    if all(isinstance(item, list) for item in raw_input):
        return list(raw_input)
    raise HTTPException(status_code=400, detail="unsupported input shape")


def _model_input(
    sample: object,
    directory: Path,
    sample_index: int,
    allowed_media_origins: frozenset[_MediaOrigin],
) -> tuple[EmbedTask, ModelInput]:
    media: list[MediaPart]
    if isinstance(sample, str):
        text, media = sample, []
    elif isinstance(sample, list):
        text, media = _message_parts(sample, directory, sample_index, allowed_media_origins)
    else:
        raise HTTPException(status_code=400, detail="unsupported input sample")

    task, text = _task_and_text(text)
    parts = ([TextPart(text)] if text else []) + media
    if not parts:
        raise HTTPException(status_code=400, detail="input has no encodable content")
    return task, ModelInput(tuple(parts))


def _message_parts(
    messages: list[object],
    directory: Path,
    sample_index: int,
    allowed_media_origins: frozenset[_MediaOrigin],
) -> tuple[str, list[MediaPart]]:
    texts: list[str] = []
    media: list[MediaPart] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="invalid message")
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="invalid message content")
        for part in content:
            text, media_part = _message_part(
                part,
                directory,
                sample_index,
                len(media),
                allowed_media_origins,
            )
            if text is not None:
                texts.append(text)
            if media_part is not None:
                media.append(media_part)
    return "\n".join(value for value in texts if value), media


def _message_part(
    part: object,
    directory: Path,
    sample_index: int,
    part_index: int,
    allowed_media_origins: frozenset[_MediaOrigin],
) -> tuple[str | None, MediaPart | None]:
    if not isinstance(part, dict):
        raise HTTPException(status_code=400, detail="invalid content part")
    kind = part.get("type")
    if kind == "text":
        value = part.get("text")
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail="invalid text part")
        return value, None
    if kind not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported part: {kind}")
    typed_kind = cast(str, kind)
    url = _content_url(part, typed_kind)
    return None, MediaPart(
        _MEDIA_TYPES[typed_kind],
        _materialize_media(
            url,
            directory,
            sample_index,
            part_index,
            allowed_media_origins,
        ),
    )


def _task_and_text(text: str) -> tuple[EmbedTask, str]:
    for prefix, task in (("Query:", EmbedTask.QUERY), ("Document:", EmbedTask.DOCUMENT)):
        if text.startswith(prefix):
            return task, text[len(prefix) :].lstrip()
    return EmbedTask.DOCUMENT, text


def _content_url(part: dict[str, Any], kind: str) -> str:
    value = part.get(kind)
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=400, detail=f"invalid {kind} part")
    return value


def _materialize_media(
    url: str,
    directory: Path,
    sample_index: int,
    part_index: int,
    allowed_media_origins: frozenset[_MediaOrigin],
) -> str:
    if not url.startswith("data:"):
        data, mime = _download_media(url, allowed_media_origins)
        return _write_media(data, mime, directory, sample_index, part_index)
    try:
        header, encoded = url.split(",", 1)
        mime = header[5:].split(";", 1)[0]
        if ";base64" not in header:
            raise ValueError
        if len(encoded) > 4 * ((_MAX_MEDIA_BYTES + 2) // 3):
            raise HTTPException(status_code=413, detail="media item exceeds 64 MiB")
        data = base64.b64decode(encoded, validate=True)
    except HTTPException:
        raise
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="invalid media data URI") from error
    if len(data) > _MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="media item exceeds 64 MiB")
    return _write_media(data, mime, directory, sample_index, part_index)


def _write_media(
    data: bytes,
    mime: str,
    directory: Path,
    sample_index: int,
    part_index: int,
) -> str:
    path = directory / f"media-{sample_index}-{part_index}{_SUFFIXES.get(mime, '.bin')}"
    path.write_bytes(data)
    return str(path)


def _allowed_media_origins(values: Sequence[str]) -> frozenset[_MediaOrigin]:
    origins: set[_MediaOrigin] = set()
    for value in values:
        parsed = urlsplit(value)
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("media origins must not contain a path, query, or fragment")
        try:
            origins.add(_remote_origin(parsed))
        except HTTPException as error:
            raise ValueError("media origins must be HTTP(S) origins without credentials") from error
    return frozenset(origins)


def _remote_origin(parsed: SplitResult) -> _MediaOrigin:
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise HTTPException(status_code=400, detail="invalid remote media URL")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="invalid remote media URL") from error
    return parsed.scheme, parsed.hostname.lower(), port


def _download_media(
    url: str,
    allowed_media_origins: frozenset[_MediaOrigin],
) -> tuple[bytes, str]:
    parsed = urlsplit(url)
    origin = _remote_origin(parsed)
    if origin not in allowed_media_origins:
        raise HTTPException(status_code=400, detail="remote media origin is not allowed")
    scheme, hostname, port = origin
    connection_type = (
        http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    )
    connection = connection_type(hostname, port, timeout=_MEDIA_DOWNLOAD_TIMEOUT_SECONDS)
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    try:
        connection.request("GET", target)
        response = connection.getresponse()
        if response.status != 200:
            raise HTTPException(status_code=502, detail="remote media download failed")
        if response.length is not None and response.length > _MAX_MEDIA_BYTES:
            raise HTTPException(status_code=413, detail="media item exceeds 64 MiB")
        data = response.read(_MAX_MEDIA_BYTES + 1)
        mime = response.headers.get("Content-Type", "application/octet-stream").split(";", 1)[0]
    except HTTPException:
        raise
    except (OSError, http.client.HTTPException) as error:
        raise HTTPException(status_code=502, detail="remote media download failed") from error
    finally:
        connection.close()
    if len(data) > _MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="media item exceeds 64 MiB")
    return data, mime


async def _request_json(request: Request) -> object:
    declared = request.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > _MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="request body is too large")
        except ValueError as error:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from error
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="request body is too large")
        body.extend(chunk)
    try:
        return json.loads(body)
    except (ValueError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=400, detail="request body must be JSON") from error


def _authorize(request: Request, api_key: str) -> None:
    supplied = request.headers.get("Authorization", "")
    if not hmac.compare_digest(supplied, f"Bearer {api_key}"):
        raise HTTPException(
            status_code=401,
            detail="invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _model_id(config: Mapping[str, object]) -> str:
    return str(config.get("model_id", DEFAULT_EMBEDDER_MODEL_ID))


def _dimension(config: Mapping[str, object]) -> int:
    value = config.get("dimension", DEFAULT_EMBEDDING_DIMENSION)
    if type(value) is not int or value <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    return value


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    """Load one SentenceTransformers model and serve the embedding contract."""
    options = _parser(prog).parse_args(argv)
    api_key = require_environment_value(os.environ, "MINDBRIDGE_EMBEDDER_API_KEY")
    config: dict[str, object] = {
        "model_id": options.model_id,
        "space_id": options.embedding_space_id,
        "dimension": options.embedding_dimension,
        "device": options.device,
        "max_concurrency": options.max_concurrency,
    }
    if options.model_revision is not None:
        config["model_revision"] = options.model_revision
    if options.trust_remote_code is not None:
        config["trust_remote_code"] = options.trust_remote_code
    uvicorn.run(
        create_app(
            api_key=api_key,
            embedder_config=config,
            media_origins=options.media_origin,
            max_batch_inputs=options.max_batch_inputs,
            batch_wait_ms=options.batch_wait_ms,
            media_io_concurrency=options.media_io_concurrency,
        ),
        host=options.host,
        port=options.port,
        workers=1,
    )


def _parser(prog: str | None) -> argparse.ArgumentParser:
    built = build_parser(
        prog=prog,
        description="Serve a SentenceTransformers embedding model (Jina Omni by default).",
        epilog=(
            "environment:\n  MINDBRIDGE_EMBEDDER_API_KEY  bearer token required by /v1/* routes"
        ),
    )
    built.add_argument("--host", default="127.0.0.1")
    built.add_argument("--port", type=int, default=8002)
    built.add_argument("--device", default="cuda")
    built.add_argument("--model-id", default=DEFAULT_EMBEDDER_MODEL_ID)
    built.add_argument("--model-revision")
    built.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "allow model repository code; requires --model-revision, because the opt-in "
            "executes that repository's Python on every worker start. Enabled by default "
            "only for bundled Jina Omni, which resolves its own pin"
        ),
    )
    built.add_argument(
        "--embedding-space-id",
        default=DEFAULT_EMBEDDING_SPACE.space_id,
        help="new models or revisions must use a new space and re-encode existing vectors",
    )
    built.add_argument(
        "--embedding-dimension",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSION,
    )
    built.add_argument("--max-concurrency", type=int, default=1)
    built.add_argument("--max-batch-inputs", type=int, default=_DEFAULT_MAX_BATCH_INPUTS)
    built.add_argument("--batch-wait-ms", type=float, default=_DEFAULT_BATCH_WAIT_MS)
    built.add_argument(
        "--media-io-concurrency",
        type=int,
        default=_DEFAULT_MEDIA_IO_CONCURRENCY,
    )
    built.add_argument(
        "--media-origin",
        action="append",
        default=[],
        metavar="URL",
        help="HTTP(S) origin allowed for remote media; repeat for multiple object stores",
    )
    return built
