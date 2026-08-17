"""Public model capabilities and plugin discovery."""

from mindbridge.application.capabilities import (
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
)
from mindbridge.models.plugins import load_embedder, load_generator

__all__ = [
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
    "load_embedder",
    "load_generator",
]
