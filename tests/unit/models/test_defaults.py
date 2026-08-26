"""Tests for the one place every process reads its bundled plugin contract."""

import pytest

from mindbridge.models import jina as sentence_transformers_models
from mindbridge.models import openai as openai_models
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDER_MODEL_ID,
    DEFAULT_EMBEDDING_SPACE,
    openai_embedder_config,
    openai_generator_config,
    require_distinct_embedding_space,
    sentence_transformers_media_embedder_config,
)

ENVIRONMENT = {
    "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
    "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
    "MINDBRIDGE_EMBEDDER_API_KEY": "embedder-secret",
    "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://embedder.example.test/v1",
    "MINDBRIDGE_EMBEDDING_SPACE_ID": "jina-v5",
    "MINDBRIDGE_EMBEDDING_DIMENSION": "512",
}


def test_every_builder_produces_keys_its_plugin_accepts() -> None:
    """extra="forbid" turns a builder that invents a key into a startup failure, so pin them."""
    openai_models._GeneratorConfig.model_validate(openai_generator_config(ENVIRONMENT))
    openai_models._EmbedderConfig.model_validate(openai_embedder_config(ENVIRONMENT))
    sentence_transformers_models._EmbedderConfig.model_validate(
        sentence_transformers_media_embedder_config(ENVIRONMENT)
    )


def test_text_and_media_encoders_agree_on_one_search_space() -> None:
    """Recall returns nothing when two encoders in one deployment disagree about the space."""
    text = openai_embedder_config(ENVIRONMENT)
    media = sentence_transformers_media_embedder_config(ENVIRONMENT)

    for key in ("space_id", "dimension"):
        assert text[key] == media[key]
    assert text["dimension"] == 512


def test_optional_settings_are_omitted_rather_than_sent_as_none() -> None:
    """An absent optional setting must leave the plugin's own default in place."""
    assert "request_timeout_seconds" not in openai_generator_config(ENVIRONMENT)
    assert "device" not in sentence_transformers_media_embedder_config(ENVIRONMENT)

    with_deadline = openai_generator_config(ENVIRONMENT, request_timeout_seconds=780.0)

    assert with_deadline["request_timeout_seconds"] == 780.0


def test_bundled_fallback_covers_credentials_and_identity_only() -> None:
    """Optional plugin tuning belongs in `*_CONFIG_JSON`, not in one variable per knob."""
    tuning = {
        "reasoning_effort",
        "max_retries",
        "video_frames_per_second",
        "video_max_pixels",
    }

    assert not tuning & set(openai_generator_config(ENVIRONMENT))
    assert not tuning & set(openai_embedder_config(ENVIRONMENT))


def test_another_model_or_revision_cannot_reuse_the_default_vector_space() -> None:
    require_distinct_embedding_space(DEFAULT_EMBEDDER_MODEL_ID, DEFAULT_EMBEDDING_SPACE.space_id)

    with pytest.raises(ValueError, match="new space_id"):
        require_distinct_embedding_space(
            "Qwen/Qwen3-VL-Embedding-2B", DEFAULT_EMBEDDING_SPACE.space_id
        )
    with pytest.raises(ValueError, match="new space_id"):
        require_distinct_embedding_space(
            DEFAULT_EMBEDDER_MODEL_ID,
            DEFAULT_EMBEDDING_SPACE.space_id,
            model_revision="different-revision",
        )
