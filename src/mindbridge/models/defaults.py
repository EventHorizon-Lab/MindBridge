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
    "DEFAULT_EMBEDDER_REVISION",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_SPACE",
    "DEFAULT_GENERATOR_MODEL_ID",
    "DEFAULT_GENERATOR_REQUEST_TIMEOUT_SECONDS",
    "EmbeddingDimension",
    "embedding_dimension_from_environment",
    "openai_embedder_config",
    "openai_generator_config",
    "require_distinct_embedding_space",
    "require_embedding_dimension",
    "sentence_transformers_media_embedder_config",
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
DEFAULT_EMBEDDER_REVISION = "12949877f0092093f366c6450340011320152a05"
"""Upstream commit the local Jina encoder loads, and executes remote code from.

Migration 0021 dropped `model_revision` from every table because the recorded value was
never compared against what the provider served. That argument is about a column. This
constant is a loader argument: `snapshot_download(revision=...)` and `code_revision` are
resolved by the Hub against a content-addressed commit, so this is the one place the pin
was enforced rather than filed. Without it a worker restart resolves the repository's
default branch, which under `trust_remote_code=True` changes both the weights and the
Python being executed, with no configuration change and no signal -- and because
`space_id` does not vary with the checkout, nothing downstream can see that it happened.
"""
DEFAULT_EMBEDDING_SPACE = EmbeddingSpaceReference(
    space_id="jinaai/jina-embeddings-v5-omni-small-retrieval-1024",
)


def embedder_revision_for(model_id: str, revision: str | None) -> str | None:
    """The commit to load `model_id` at: an explicit pin, else the bundled pin only if it fits.

    `DEFAULT_EMBEDDER_REVISION` is a commit of `DEFAULT_EMBEDDER_MODEL_ID`. Applying it to a
    repository an operator named instead resolves a sha that repository does not contain, so
    `snapshot_download` raises `RevisionNotFoundError` naming a value that appears nowhere in
    their configuration. Every path that turns a model id into a download resolves the pin
    here, so naming another repository cannot inherit a pin belonging to this one, and the
    bundled model cannot lose its pin by a caller forgetting to pass it.
    """
    if revision is not None:
        return revision
    return DEFAULT_EMBEDDER_REVISION if model_id == DEFAULT_EMBEDDER_MODEL_ID else None


def require_embedding_dimension(dimension: int) -> int:
    """Require a usable vector width; the selected model validates whether it can produce it."""
    if dimension <= 0:
        raise ValueError("embedding dimension must be positive")
    return dimension


EmbeddingDimension = Annotated[PluginInteger, AfterValidator(require_embedding_dimension)]


def embedding_dimension_from_environment(source: Mapping[str, str]) -> int:
    """One deployment-wide vector width shared by the index and every encoder."""
    return require_embedding_dimension(
        int(source.get("MINDBRIDGE_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))
    )


def require_distinct_embedding_space(
    model_id: str,
    space_id: str,
    *,
    model_revision: str | None = None,
) -> None:
    """Refuse to write another encoder or revision into the bundled Jina vector space."""
    revision = embedder_revision_for(model_id, model_revision)
    if space_id == DEFAULT_EMBEDDING_SPACE.space_id and (
        model_id != DEFAULT_EMBEDDER_MODEL_ID or revision != DEFAULT_EMBEDDER_REVISION
    ):
        raise ValueError("a non-default embedding model or revision requires a new space_id")


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


def sentence_transformers_media_embedder_config(
    source: Mapping[str, str],
) -> dict[str, object]:
    """Read the local SentenceTransformers contract for the Worker's media encoder."""
    config: dict[str, object] = {
        "model_id": source.get("MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID", DEFAULT_EMBEDDER_MODEL_ID),
        **_embedding_space_config(source),
    }
    # Only the local encoder takes a revision, because only the local encoder downloads
    # anything. The OpenAI-shaped encoder talks to an endpoint that resolves its own model,
    # so a revision there would be the unread record 0021 removed. Absent, the pin is
    # resolved from the model id rather than defaulted here, so overriding only
    # MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID does not pin another repository to this one's commit.
    revision = optional_environment_value(source, "MINDBRIDGE_MEDIA_EMBEDDER_MODEL_REVISION")
    if revision is not None:
        config["model_revision"] = revision
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
