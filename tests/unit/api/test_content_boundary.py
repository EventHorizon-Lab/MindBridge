"""Transport media is normalized once and remains stable through nested validation."""

import base64

import pytest
from pydantic import TypeAdapter, ValidationError

from mindbridge.api.app import MemoryCreate
from mindbridge.api.content import InputPart, content_input
from mindbridge.exceptions import ValidationError as ContentValidationError
from mindbridge.types import Blob


@pytest.mark.parametrize(
    ("part", "expected"),
    [
        (
            {"type": "input_image", "image_url": "data:image/png;base64,cG5n"},
            Blob(b"png", "image/png"),
        ),
        (
            {
                "type": "input_file",
                "file_url": "data:audio/wav;base64,d2F2",
                "media_type": "audio/*",
            },
            Blob(b"wav", "audio/wav"),
        ),
        (
            {"type": "input_file", "file_data": "d2F2", "media_type": "audio/wav"},
            Blob(b"wav", "audio/wav"),
        ),
    ],
)
def test_media_decoded_once_through_nested_validation_and_conversion(
    monkeypatch: pytest.MonkeyPatch, part: dict[str, str], expected: Blob
) -> None:
    original = base64.b64decode
    calls = 0

    def counting_decode(value: str, *, validate: bool = False) -> bytes:
        nonlocal calls
        calls += 1
        return original(value, validate=validate)

    monkeypatch.setattr(base64, "b64decode", counting_decode)
    validated: InputPart = TypeAdapter(InputPart).validate_python(part)
    request = MemoryCreate(content=(validated,))
    first = content_input(request.content)
    second = content_input(request.content)
    assert first == (expected,)
    assert second == (expected,)
    assert first[0] is second[0]
    assert calls == 1
    assert request.model_dump(mode="json")["content"][0] == validated.model_dump(mode="json")


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image", "image_url": "https://example.com/a.png"},
        {"type": "input_image", "image_url": "data:audio/wav;base64,d2F2"},
        {"type": "input_file", "file_url": "data:audio/wav;base64,d2F2", "media_type": "image/png"},
        {"type": "input_file", "file_data": "!", "media_type": "audio/wav"},
        {"type": "input_file", "file_data": "d2F2", "file_id": "a", "media_type": "audio/wav"},
    ],
)
def test_invalid_media_is_rejected_at_transport_boundary(part: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(InputPart).validate_python(part)


def test_validated_media_source_cannot_be_reassigned() -> None:
    part: InputPart = TypeAdapter(InputPart).validate_python(
        {"type": "input_file", "file_data": "d2F2", "media_type": "audio/wav"}
    )
    with pytest.raises(ValidationError):
        part.file_data = "cG5n"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("part", "message"),
    [
        (
            {"type": "input_image", "image_url": "data:image/png;base64,"},
            "Blob data must be non-empty bytes",
        ),
        (
            {"type": "input_file", "file_url": "data:application/pdf;base64,cGRm"},
            "media_type must be an image, video, or audio media type",
        ),
    ],
)
def test_blob_errors_keep_their_conversion_boundary(part: dict[str, str], message: str) -> None:
    validated: InputPart = TypeAdapter(InputPart).validate_python(part)
    request = MemoryCreate(content=(validated,))
    with pytest.raises(ContentValidationError, match=message):
        content_input(request.content)
