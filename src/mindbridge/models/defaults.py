"""Deployable model defaults without importing any provider SDK."""

from mindbridge.core import EmbeddingSpaceReference

DEFAULT_GENERATOR_MODEL_ID = "qwen3.8-max"
DEFAULT_MEDIA_EMBEDDER_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
DEFAULT_MEDIA_EMBEDDER_REVISION = "12949877f0092093f366c6450340011320152a05"
DEFAULT_TEXT_EMBEDDER_MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"
DEFAULT_TEXT_EMBEDDER_REVISION = "6856e76bb72982e58de0620458a4e8b3614da340"
DEFAULT_EMBEDDING_DIMENSION = 1_024
DEFAULT_EMBEDDING_SPACE = EmbeddingSpaceReference(
    space_id="jinaai/jina-embeddings-v5-small-retrieval-1024",
    revision=(
        "omni@12949877f0092093f366c6450340011320152a05+"
        "text@6856e76bb72982e58de0620458a4e8b3614da340"
    ),
)
