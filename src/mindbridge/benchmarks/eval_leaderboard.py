"""Participant-hosted Agent Memory Leaderboard Add/Search benchmark adapter."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import Field, StringConstraints, field_validator, model_validator

from mindbridge import AsyncMemory, Blob, MemoryRecord, MindBridgeConfig, SearchHit
from mindbridge.api.content import StrictModel
from mindbridge.api.errors import register_error_handlers
from mindbridge.benchmarks.eval_artifacts import _atomic_replace, _json_bytes
from mindbridge.benchmarks.eval_config import DEFAULT_CONFIG_SECTIONS, _read_config_document
from mindbridge.infrastructure.local._lock import DataDirectoryLock
from mindbridge.types import ContentInput

_Text = Annotated[str, StringConstraints(strict=True, min_length=1, pattern=r"\S")]
_IMAGE_BYTES = 10 * 1024 * 1024
_TOTAL_IMAGE_BYTES = 30 * 1024 * 1024
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _image(url: str) -> Blob:
    header, separator, encoded = url.partition(",")
    if not separator or header not in {
        "data:image/jpeg;base64",
        "data:image/png;base64",
        "data:image/webp;base64",
    }:
        raise ValueError("images require an inline JPEG, PNG, or WebP Base64 Data URI")
    if len(encoded) > 4 * ((_IMAGE_BYTES + 2) // 3):
        raise ValueError("an image must not exceed 10 MiB decoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("invalid image Base64") from None
    if not data or len(data) > _IMAGE_BYTES:
        raise ValueError("an image must contain between 1 byte and 10 MiB decoded")
    return Blob(data=data, media_type=header[5:-7])


class _TextPart(StrictModel):
    type: Literal["text"]
    text: _Text


class _ImageURL(StrictModel):
    url: _Text

    @field_validator("url")
    @classmethod
    def validate_image(cls, value: str) -> str:
        _image(value)
        return value


class _ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: _ImageURL


_Part: TypeAlias = Annotated[_TextPart | _ImagePart, Field(discriminator="type")]
_Content: TypeAlias = _Text | Annotated[list[_Part], Field(min_length=1)]


def _input(content: _Content) -> ContentInput:
    if isinstance(content, str):
        return content
    return tuple(
        part.text if isinstance(part, _TextPart) else _image(part.image_url.url) for part in content
    )


def _image_bytes(content: _Content) -> int:
    return (
        0
        if isinstance(content, str)
        else sum(
            len(_image(part.image_url.url).data) for part in content if isinstance(part, _ImagePart)
        )
    )


def _timestamp(value: int | None) -> datetime | None:
    return None if value is None else _EPOCH + timedelta(milliseconds=value)


class _Message(StrictModel):
    role: Literal["user", "assistant"]
    content: _Content
    timestamp: Annotated[int, Field(strict=True)] | None = None

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: int | None) -> int | None:
        try:
            _timestamp(value)
        except OverflowError:
            raise ValueError("timestamp is outside the supported datetime range") from None
        return value


class AddRequest(StrictModel):
    request_id: _Text
    messages: Annotated[list[_Message], Field(min_length=1)]
    user_id: _Text
    session_id: _Text

    @model_validator(mode="after")
    def validate_media_total(self) -> AddRequest:
        if sum(_image_bytes(message.content) for message in self.messages) > _TOTAL_IMAGE_BYTES:
            raise ValueError("Add images must not exceed 30 MiB decoded in total")
        return self


class SearchRequest(StrictModel):
    query: _Content
    options: list[_Text] | None = None
    user_id: _Text
    top_k: Annotated[int, Field(strict=True, ge=1, le=100)]

    @model_validator(mode="after")
    def validate_media_total(self) -> SearchRequest:
        if _image_bytes(self.query) > _TOTAL_IMAGE_BYTES:
            raise ValueError("Search images must not exceed 30 MiB decoded in total")
        return self


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _write_receipt(receipt: Path, fingerprint: str, *, complete: bool) -> None:
    _atomic_replace([(receipt, _json_bytes({"fingerprint": fingerprint, "complete": complete}))])


def _prepare_add(receipt: Path, request: AddRequest) -> tuple[str, bool]:
    _directory(receipt.parent)
    fingerprint = hashlib.sha256(_json_bytes(request.model_dump(mode="json"))).hexdigest()
    prior = json.loads(receipt.read_bytes()) if receipt.exists() else None
    if prior is not None and prior["fingerprint"] != fingerprint:
        raise HTTPException(422, "request_id was already used with a different payload")
    complete = prior is not None and bool(prior["complete"])
    if not complete:
        _write_receipt(receipt, fingerprint, complete=False)
    return fingerprint, complete


def _persist_sources(
    directory: Path, records: Sequence[MemoryRecord], messages: Sequence[_Message]
) -> None:
    _directory(directory)
    _atomic_replace(
        [
            (
                directory / f"{record.id}.json",
                _json_bytes(message.model_dump(mode="json")["content"]),
            )
            for record, message in zip(records, messages, strict=True)
        ]
    )


def _fallback_content(hit: SearchHit) -> object:
    if not hit.assets:
        return hit.content
    parts: list[dict[str, object]] = []
    if hit.content.strip():
        parts.append({"type": "text", "text": hit.content})
    for asset in hit.assets:
        if asset.path is None or asset.media_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(
                422, "retrieved media cannot be represented by the AML image contract"
            )
        data = asset.path.read_bytes()
        if len(data) > _IMAGE_BYTES:
            raise HTTPException(422, "retrieved image exceeds 10 MiB decoded")
        parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{asset.media_type};base64,{base64.b64encode(data).decode('ascii')}"
                },
            }
        )
    return parts


def create_app(  # noqa: C901 - Add and Search share authorization and per-user ownership
    *,
    data_root: Path,
    memory_factory: Callable[[Path], AsyncMemory],
    api_key: str | None,
    data_root_lock: DataDirectoryLock | None = None,
) -> FastAPI:
    """Serve the benchmark contract using one physically isolated store per user ID."""
    root = data_root.expanduser().resolve()
    locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        owner = DataDirectoryLock(root) if data_root_lock is None else data_root_lock
        try:
            yield
        finally:
            owner.close()

    app = FastAPI(title="MindBridge Agent Memory Leaderboard adapter", lifespan=lifespan)
    register_error_handlers(app)

    def authorize(request: Request) -> None:
        if api_key is None:
            return
        authorization = request.headers.get("authorization", "")
        scheme, _, credential = authorization.partition(" ")
        supplied = (
            credential
            if scheme.casefold() in {"token", "bearer"}
            else request.headers.get("x-api-key", "")
        )
        if not hmac.compare_digest(supplied.encode("utf-8"), api_key.encode("utf-8")):
            raise HTTPException(401, "invalid memory system key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/add", dependencies=[Depends(authorize)])
    async def add(request: AddRequest) -> dict[str, object]:
        user = _digest(request.user_id)
        async with locks.setdefault(user, asyncio.Lock()):
            receipt = root / "requests" / user / f"{_digest(request.request_id)}.json"
            fingerprint, complete = await asyncio.to_thread(_prepare_add, receipt, request)
            if not complete:
                memory = await asyncio.to_thread(memory_factory, root / "users" / user)
                async with memory:
                    records = await memory.add_many(
                        [_input(message.content) for message in request.messages],
                        occurred_at=[_timestamp(message.timestamp) for message in request.messages],
                        metadata=[
                            {
                                "request_id": request.request_id,
                                "session_id": request.session_id,
                                "role": message.role,
                                "message_index": index,
                            }
                            for index, message in enumerate(request.messages)
                        ],
                    )
                    await asyncio.to_thread(
                        _persist_sources, root / "sources" / user, records, request.messages
                    )
                await asyncio.to_thread(_write_receipt, receipt, fingerprint, complete=True)
        return {
            "success": True,
            "request_id": request.request_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

    @app.post("/search", dependencies=[Depends(authorize)])
    async def search(request: SearchRequest) -> dict[str, object]:
        user = _digest(request.user_id)
        async with locks.setdefault(user, asyncio.Lock()):
            directory = root / "users" / user
            if not directory.exists():
                return {"data": []}
            memory = await asyncio.to_thread(memory_factory, directory)
            async with memory:
                hits = await memory.search(_input(request.query), limit=request.top_k)
                rows: list[dict[str, object]] = []
                image_bytes = 0
                for hit in hits:
                    source = root / "sources" / user / f"{hit.id}.json"
                    content = (
                        json.loads(source.read_bytes())
                        if source.exists()
                        else _fallback_content(hit)
                    )
                    if isinstance(content, list):
                        image_bytes += sum(
                            len(_image(part["image_url"]["url"]).data)
                            for part in content
                            if part["type"] == "image_url"
                        )
                    if image_bytes > _TOTAL_IMAGE_BYTES:
                        raise HTTPException(422, "Search response images exceed 30 MiB decoded")
                    rows.append(
                        {
                            "id": hit.id,
                            "content": content,
                            "score": hit.score,
                            "created_at": (hit.occurred_at or hit.created_at).isoformat(),
                        }
                    )
        return {"data": rows}

    return app


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    """Start a single-process AML server with the existing evaluation backend pool."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="MINDBRIDGE_AML_API_KEY")
    parser.add_argument(
        "--public-smoke", action="store_true", help="allow unauthenticated public smoke"
    )
    arguments = parser.parse_args(argv)
    api_key = os.environ.get(arguments.api_key_env)
    if not arguments.public_smoke and not api_key:
        parser.error(f"set {arguments.api_key_env}, or use --public-smoke for public smoke only")
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        values = dict(_read_config_document(arguments.config.expanduser().resolve()))
        values.pop("benchmark", None)
        values.setdefault("embedding", dict(DEFAULT_CONFIG_SECTIONS["embedding"]))
        config = MindBridgeConfig.model_validate(values).model_copy(update={"generation": None})
    except ValueError as error:
        parser.error(str(error))
    try:
        import uvicorn
    except ImportError:
        parser.error("serving requires uvicorn and mindbridge[benchmarks,server]")
    from mindbridge.benchmarks.eval import _BackendPool
    from mindbridge.benchmarks.model_config import ModelConfig

    with ExitStack() as cleanup:
        root = arguments.data_root.expanduser().resolve()
        owner = DataDirectoryLock(root)
        cleanup.callback(owner.close)
        pool = _BackendPool(
            ModelConfig(),
            device=None,
            batch_size=64,
            needs_speech=False,
            seed=0,
            memory_config=config,
        )
        cleanup.callback(pool.close)
        uvicorn.run(
            create_app(
                data_root=root, memory_factory=pool.memory, api_key=api_key, data_root_lock=owner
            ),
            host=arguments.host,
            port=arguments.port,
            access_log=False,
        )
    return 0
