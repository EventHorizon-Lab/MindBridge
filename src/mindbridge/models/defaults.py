"""Deployable model defaults without importing any provider SDK."""

from collections.abc import Mapping

from mindbridge.core import DEFAULT_EMBEDDING_DIMENSION, EmbeddingSpaceReference

__all__ = [
    "DEFAULT_EMBEDDER_MODEL_ID",
    "DEFAULT_EMBEDDER_REVISION",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_SPACE",
    "DEFAULT_GENERATOR_MODEL_ID",
    "MATRYOSHKA_DIMENSIONS",
    "embedding_dimension_from_environment",
    "require_matryoshka_dimension",
]

DEFAULT_GENERATOR_MODEL_ID = "qwen3.8-max"
DEFAULT_EMBEDDER_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_EMBEDDER_REVISION = "12949877f0092093f366c6450340011320152a05"
DEFAULT_EMBEDDING_SPACE = EmbeddingSpaceReference(
    space_id="jinaai/jina-embeddings-v5-omni-small-retrieval-1024",
    revision="omni@12949877f0092093f366c6450340011320152a05",
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


def embedding_dimension_from_environment(source: Mapping[str, str]) -> int:
    """One deployment-wide vector width shared by the index and every encoder."""
    return require_matryoshka_dimension(
        int(source.get("MINDBRIDGE_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))
    )
