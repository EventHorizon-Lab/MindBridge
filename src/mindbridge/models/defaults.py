"""Deployable model defaults without importing any provider SDK."""

from collections.abc import Mapping
from typing import Annotated

from pydantic import AfterValidator

from mindbridge.configuration import (
    PluginInteger,
    optional_environment_value,
    require_environment_value,
)
from mindbridge.core import DEFAULT_EMBEDDING_DIMENSION, EmbeddingSpaceReference

__all__ = [
    "DEFAULT_EMBEDDER_MODEL_ID",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_SPACE",
    "DEFAULT_GENERATOR_MODEL_ID",
    "DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS",
    "MATRYOSHKA_DIMENSIONS",
    "MatryoshkaDimension",
    "embedding_dimension_from_environment",
    "jina_media_embedder_config",
    "openai_embedder_config",
    "openai_generator_config",
    "require_matryoshka_dimension",
]

DEFAULT_GENERATOR_MODEL_ID = "qwen3.8-max"
DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS = 1_800.0
"""Deadline the bundled generator applies when a supplied config does not name one.

The Worker sizes its Celery budget from this, so the two have to be the same number. They
were not: the plugin defaulted to 1800 while the Worker assumed 780, and a deployment that
supplied MINDBRIDGE_GENERATOR_CONFIG_JSON without this key got a model client allowed 1800s
inside a task killed at 1080 -- the disagreement the derived budget exists to remove.
"""
DEFAULT_EMBEDDER_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_EMBEDDING_SPACE = EmbeddingSpaceReference(
    space_id="jinaai/jina-embeddings-v5-omni-small-retrieval-1024",
)

# Jina v5 trains these Matryoshka prefixes; any other width is an untrained
# truncation, so a deployment may pick from this set but not invent a size.
MATRYOSHKA_DIMENSIONS = (32, 64, 128, 256, 512, 768, 1_024)


def require_matryoshka_dimension(dimension: int) -> int:
    """Reject vector widths the encoder was never trained to truncate to."""
    if dimension not in MATRYOSHKA_DIMENSIONS:
        supported = ", ".join(str(value) for value in MATRYOSHKA_DIMENSIONS)
        raise ValueError(f"embedding dimension must be one of {supported}")
    return dimension


MatryoshkaDimension = Annotated[PluginInteger, AfterValidator(require_matryoshka_dimension)]


def embedding_dimension_from_environment(source: Mapping[str, str]) -> int:
    """One deployment-wide vector width shared by the index and every encoder."""
    return require_matryoshka_dimension(
        int(source.get("MINDBRIDGE_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))
    )


# Every process that loads a bundled plugin builds its fallback configuration here, so each
# variable is read in exactly one place. Three copies of these builders had already drifted
# apart — only one of them applied a request deadline — and per-process differences now arrive
# as arguments instead of as another variable.
#
# These builders cover credentials and model identity only: the values a deployment cannot start
# without. A plugin's remaining optional settings are reachable through its `*_CONFIG_JSON`
# object rather than one environment variable each, which is what keeps this surface bounded as
# plugins gain knobs.
def openai_generator_config(
    source: Mapping[str, str],
    *,
    request_timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Read the bundled OpenAI generator contract shared by every deployable process."""
    config: dict[str, object] = {
        "api_key": require_environment_value(source, "MINDBRIDGE_GENERATOR_API_KEY"),
        "endpoint": require_environment_value(source, "MINDBRIDGE_GENERATOR_ENDPOINT"),
        "model_id": source.get("MINDBRIDGE_GENERATOR_MODEL_ID", DEFAULT_GENERATOR_MODEL_ID),
    }
    if request_timeout_seconds is not None:
        config["request_timeout_seconds"] = request_timeout_seconds
    return config


def openai_embedder_config(
    source: Mapping[str, str],
    *,
    request_timeout_seconds: float | None = None,
) -> dict[str, object]:
    """Read the bundled OpenAI text encoder contract for querying, ingest, and consolidation."""
    config: dict[str, object] = {
        "api_key": require_environment_value(source, "MINDBRIDGE_EMBEDDER_API_KEY"),
        "endpoint": require_environment_value(source, "MINDBRIDGE_EMBEDDER_ENDPOINT"),
        "model_id": source.get("MINDBRIDGE_EMBEDDER_MODEL_ID", DEFAULT_EMBEDDER_MODEL_ID),
        **_embedding_space_config(source),
    }
    if request_timeout_seconds is not None:
        config["request_timeout_seconds"] = request_timeout_seconds
    return config


def jina_media_embedder_config(source: Mapping[str, str]) -> dict[str, object]:
    """Read the bundled local Jina contract for the Worker's image, video, and audio encoder."""
    config: dict[str, object] = {
        "model_id": source.get("MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID", DEFAULT_EMBEDDER_MODEL_ID),
        **_embedding_space_config(source),
    }
    device = optional_environment_value(source, "MINDBRIDGE_MEDIA_EMBEDDER_DEVICE")
    if device is not None:
        config["device"] = device
    return config


def _embedding_space_config(source: Mapping[str, str]) -> dict[str, object]:
    """Name the one search space and vector width every encoder in a deployment must share."""
    return {
        "space_id": source.get("MINDBRIDGE_EMBEDDING_SPACE_ID", DEFAULT_EMBEDDING_SPACE.space_id),
        "dimension": embedding_dimension_from_environment(source),
    }
