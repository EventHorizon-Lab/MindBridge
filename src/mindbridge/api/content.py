"""The one strict content-part contract REST and MCP both accept.

REST and MCP validate the same ordered `input_text`, `input_image`, and `input_file` union and
normalize it into the same public `ContentInput`. This module is that decoder; the transports
only wrap it in their own request and tool models.

`_InputText`, `_InputImage`, and `_InputFile` keep their leading underscore because FastAPI names
OpenAPI schema components after the class: renaming one is a published REST contract change, not a
style fix.
"""

from __future__ import annotations

import base64
import binascii
from datetime import timedelta
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mindbridge.types import AssetRef, Blob, ContentInput, ContextBudget, MemoryType, Modality

MAX_TEXT_CHARACTERS = 65_536
# Bounds one decoded inline media value before any model or storage work. REST additionally caps
# the whole request body, which is why the two surfaces can share one decoder limit.
MAX_INLINE_MEDIA_BYTES = 8 * 1024 * 1024
MAX_CONTENT_PARTS = 16

Text = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_TEXT_CHARACTERS,
    ),
]
PartId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
MediaType = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        pattern=r"^[a-z0-9!#$&^_.+-]+/(?:\*|[a-z0-9!#$&^_.+-]+)$",
    ),
]
Filename = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\\x00-\x1f\x7f]+$",
    ),
]
Source = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_192)]
# The numeric bounds the request and tool models share. `strict=True` keeps a float out of an
# integer field, which JSON would otherwise round into a silently different request.
Limit = Annotated[int, Field(strict=True, ge=1, le=100)]
Chars = Annotated[int, Field(strict=True, ge=1, le=MAX_TEXT_CHARACTERS)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Seconds = Annotated[float, Field(gt=0.0)]
Milliseconds = Annotated[int, Field(strict=True, ge=1)]
# The published compilation defaults are the SDK value's, never a second copy of the numbers.
_BUDGET = ContextBudget()


class StrictModel(BaseModel):
    """Reject every field the public contract does not name."""

    model_config = ConfigDict(extra="forbid")


class _InputText(StrictModel):
    type: Literal["input_text"]
    text: Text


class _InputImage(StrictModel):
    type: Literal["input_image"]
    image_url: Source | None = None
    file_id: PartId | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> _InputImage:
        _one_source(image_url=self.image_url, file_id=self.file_id)
        if self.image_url is not None:
            _validate_data_url(self.image_url)
            if self.image_url.startswith("data:"):
                media_type, _data = _decode_data_url(self.image_url)
                if not media_type.startswith("image/"):
                    raise ValueError("input_image data must have an image media type")
        return self


class _InputFile(StrictModel):
    type: Literal["input_file"]
    file_url: Source | None = None
    file_data: str | None = None
    file_id: PartId | None = None
    media_type: MediaType | None = None
    filename: Filename | None = None

    @model_validator(mode="after")
    def require_one_source(self) -> _InputFile:
        _one_source(file_url=self.file_url, file_data=self.file_data, file_id=self.file_id)
        if self.media_type is not None and self.media_type.split("/", 1)[0] not in {
            "image",
            "video",
            "audio",
        }:
            raise ValueError("media_type must be image, video, or audio")
        if self.file_url is not None:
            _validate_data_url(self.file_url)
            if self.file_url.startswith("data:") and self.media_type is not None:
                embedded_type, _data = _decode_data_url(self.file_url)
                if not _media_type_matches(self.media_type, embedded_type):
                    raise ValueError("media_type contradicts the data URL")
        if self.file_data is not None:
            if self.media_type is None:
                raise ValueError("media_type is required with file_data")
            if self.media_type.endswith("/*"):
                raise ValueError("file_data requires a concrete media_type")
            decode_base64(self.file_data)
        return self


InputPart: TypeAlias = Annotated[
    _InputText | _InputImage | _InputFile,
    Field(discriminator="type"),
]
Parts = Annotated[tuple[InputPart, ...], Field(min_length=1, max_length=MAX_CONTENT_PARTS)]
Content: TypeAlias = Text | Parts


def content_input(content: Content) -> ContentInput:
    """Normalize validated transport parts into one public ordered content value."""
    if isinstance(content, str):
        return content
    atoms: list[str | Blob | AssetRef] = []
    for part in content:
        if isinstance(part, _InputText):
            atoms.append(part.text)
        elif isinstance(part, _InputImage):
            if part.file_id is not None:
                atoms.append(AssetRef(id=part.file_id, modality=Modality.IMAGE))
            else:
                atoms.append(_data_blob(cast(str, part.image_url), media_type="image/*", name=None))
        elif part.file_id is not None:
            atoms.append(_file_reference(part.file_id, part.media_type))
        elif part.file_data is not None:
            atoms.append(
                Blob(
                    data=decode_base64(part.file_data),
                    media_type=cast(str, part.media_type),
                    name=part.filename,
                )
            )
        else:
            atoms.append(
                _data_blob(
                    cast(str, part.file_url),
                    media_type=part.media_type,
                    name=part.filename,
                )
            )
    return tuple(atoms)


# The compilation bounds both transports accept. REST publishes it as `ContextBudgetRequest` and
# MCP under this name, so REST subclasses rather than reuses the class: each SDK names its schema
# component after the class, and renaming one is a published contract change. No docstring, for
# the same reason -- it would become a schema `description` neither surface publishes today.
class ContextBudgetInput(StrictModel):
    max_chars: Chars = _BUDGET.max_chars
    max_items: Limit = _BUDGET.max_items
    memory_types: Annotated[list[MemoryType], Field(min_length=1)] | None = None
    min_confidence: Confidence = _BUDGET.min_confidence
    # `ContextBudget.freshness` is a timedelta; JSON carries the same bound as seconds.
    freshness_seconds: Seconds | None = None
    max_latency_ms: Milliseconds | None = None


def context_budget(request: ContextBudgetInput | None) -> ContextBudget | None:
    """Translate the transport budget into the SDK value, which validates every bound."""
    if request is None:
        return None
    return ContextBudget(
        max_chars=request.max_chars,
        max_items=request.max_items,
        memory_types=None if request.memory_types is None else frozenset(request.memory_types),
        min_confidence=request.min_confidence,
        freshness=(
            None
            if request.freshness_seconds is None
            else timedelta(seconds=request.freshness_seconds)
        ),
        max_latency_ms=request.max_latency_ms,
    )


def decode_base64(value: str) -> bytes:
    """Decode one bounded inline media value, rejecting oversize input before decoding."""
    maximum_encoded = 4 * ((MAX_INLINE_MEDIA_BYTES + 2) // 3)
    if len(value) > maximum_encoded:
        raise ValueError(f"inline media must not exceed {MAX_INLINE_MEDIA_BYTES} bytes")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("file_data must be valid base64") from error
    if len(data) > MAX_INLINE_MEDIA_BYTES:
        raise ValueError(f"inline media must not exceed {MAX_INLINE_MEDIA_BYTES} bytes")
    return data


def _file_reference(file_id: str, media_type: str | None) -> AssetRef:
    if media_type is None:
        return AssetRef(id=file_id)
    if media_type.endswith("/*"):
        return AssetRef(id=file_id, modality=Modality(media_type.split("/", 1)[0]))
    return AssetRef(id=file_id, media_type=media_type)


def _data_blob(
    value: str,
    *,
    media_type: str | None = None,
    name: str | None,
) -> Blob:
    embedded_type, data = _decode_data_url(value)
    if media_type is not None and not _media_type_matches(media_type, embedded_type):
        raise ValueError("media_type contradicts the data URL")
    return Blob(data=data, media_type=embedded_type, name=name)


def _one_source(**sources: object) -> None:
    if sum(source is not None for source in sources.values()) != 1:
        raise ValueError(f"exactly one of {', '.join(sources)} is required")


def _media_type_matches(expected: str, actual: str) -> bool:
    return expected == actual or (
        expected.endswith("/*") and expected.split("/", 1)[0] == actual.split("/", 1)[0]
    )


def _validate_data_url(value: str) -> None:
    if not value.startswith("data:"):
        raise ValueError("remote URLs are not accepted; fetch media before calling MindBridge")
    _decode_data_url(value)


def _decode_data_url(value: str) -> tuple[str, bytes]:
    header, separator, payload = value.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ValueError("data URL must contain base64 media bytes")
    media_type = header.removeprefix("data:").removesuffix(";base64").lower()
    if not media_type or "/" not in media_type:
        raise ValueError("data URL must declare a media type")
    return media_type, decode_base64(payload)
