"""Public model capabilities and plugin discovery."""

from mindbridge.application.capabilities import (
    ALL_MEDIA_KINDS,
    Embedder,
    Embedding,
    EmbedRequest,
    EmbedResult,
    EmbedTask,
    GenerateRequest,
    GenerateResult,
    Generator,
    InputPart,
    MediaPart,
    ModelInput,
    TextPart,
    declared_supported_media_kinds,
    require_supported_media,
)
from mindbridge.models.plugins import load_embedder, load_generator

__all__ = [
    "ALL_MEDIA_KINDS",
    "EmbedRequest",
    "EmbedResult",
    "EmbedTask",
    "Embedder",
    "Embedding",
    "GenerateRequest",
    "GenerateResult",
    "Generator",
    "InputPart",
    "MediaPart",
    "ModelInput",
    "TextPart",
    "declared_supported_media_kinds",
    "load_embedder",
    "load_generator",
    "require_supported_media",
]
