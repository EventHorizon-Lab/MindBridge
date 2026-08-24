"""Authenticated SentenceTransformers serving for the bundled Jina Omni model."""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import os
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Request

from mindbridge.application.capabilities import (
    Embedder,
    EmbedRequest,
    EmbedTask,
    MediaPart,
    ModelInput,
    TextPart,
)
from mindbridge.cli import parser as build_parser
from mindbridge.configuration import require_environment_value
from mindbridge.core import MediaKind
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_SPACE,
)
from mindbridge.models.plugins import close_model, load_embedder

_MAX_MEDIA_BYTES = 64 * 1024 * 1024
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


def create_app(
    *,
    api_key: str,
    embedder_config: Mapping[str, object] | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build one single-model service; injection keeps contract tests model-free."""
    if not api_key.strip():
        raise ValueError("embedding API key must not be empty")
    config = dict(embedder_config or {})

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        loaded = embedder or load_embedder("jina", config)
        app.state.embedder = loaded
        try:
            yield
        finally:
            if embedder is None:
                await close_model(loaded)

    app = FastAPI(title="MindBridge Jina Omni embedding service", lifespan=lifespan)
    if embedder is not None:
        app.state.embedder = embedder

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
        try:
            payload = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="request body must be JSON") from error
        return await _embedding_response(
            payload, cast(Embedder, request.app.state.embedder), config
        )

    return app


async def _embedding_response(
    payload: object,
    embedder: Embedder,
    config: Mapping[str, object],
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
        vectors = await _embed(embedder, payload.get("input"))
    except HTTPException:
        raise
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


async def _embed(embedder: Embedder, raw_input: object) -> tuple[tuple[float, ...], ...]:
    with tempfile.TemporaryDirectory(prefix="mindbridge-jina-") as temp_dir:
        directory = Path(temp_dir)
        parsed = tuple(
            _model_input(sample, directory, index)
            for index, sample in enumerate(_samples(raw_input))
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
    sample: object, directory: Path, sample_index: int
) -> tuple[EmbedTask, ModelInput]:
    media: list[MediaPart]
    if isinstance(sample, str):
        text, media = sample, []
    elif isinstance(sample, list):
        text, media = _message_parts(sample, directory, sample_index)
    else:
        raise HTTPException(status_code=400, detail="unsupported input sample")

    task, text = _task_and_text(text)
    parts = ([TextPart(text)] if text else []) + media
    if not parts:
        raise HTTPException(status_code=400, detail="input has no encodable content")
    return task, ModelInput(tuple(parts))


def _message_parts(
    messages: list[object], directory: Path, sample_index: int
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
        for part_index, part in enumerate(content):
            text, media_part = _message_part(part, directory, sample_index, part_index)
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
        _materialize_media(url, directory, sample_index, part_index),
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


def _materialize_media(url: str, directory: Path, sample_index: int, part_index: int) -> str:
    if not url.startswith("data:"):
        return url
    try:
        header, encoded = url.split(",", 1)
        mime = header[5:].split(";", 1)[0]
        if ";base64" not in header:
            raise ValueError
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise HTTPException(status_code=400, detail="invalid media data URI") from error
    if len(data) > _MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="media item exceeds 64 MiB")
    path = directory / f"media-{sample_index}-{part_index}{_SUFFIXES.get(mime, '.bin')}"
    path.write_bytes(data)
    return str(path)


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
    """Load one pinned model and serve the embedding contract."""
    options = _parser(prog).parse_args(argv)
    api_key = require_environment_value(os.environ, "MINDBRIDGE_EMBEDDER_API_KEY")
    config: dict[str, object] = {
        "model_id": options.model_id,
        "space_id": DEFAULT_EMBEDDING_SPACE.space_id,
        "dimension": DEFAULT_EMBEDDING_DIMENSION,
        "device": options.device,
        "max_concurrency": options.max_concurrency,
    }
    uvicorn.run(
        create_app(api_key=api_key, embedder_config=config),
        host=options.host,
        port=options.port,
        workers=1,
    )


def _parser(prog: str | None) -> argparse.ArgumentParser:
    built = build_parser(
        prog=prog,
        description="Serve Jina v5 Omni with SentenceTransformers.",
        epilog=(
            "environment:\n  MINDBRIDGE_EMBEDDER_API_KEY  bearer token required by /v1/* routes"
        ),
    )
    built.add_argument("--host", default="127.0.0.1")
    built.add_argument("--port", type=int, default=8002)
    built.add_argument("--device", default="cuda")
    built.add_argument("--model-id", default=DEFAULT_EMBEDDER_MODEL_ID)
    built.add_argument("--max-concurrency", type=int, default=1)
    return built
