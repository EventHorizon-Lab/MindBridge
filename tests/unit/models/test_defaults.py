"""Tests for the one place every process reads its bundled plugin contract."""

from mindbridge.models import jina as jina_models
from mindbridge.models import openai as openai_models
from mindbridge.models.defaults import (
    jina_media_embedder_config,
    openai_embedder_config,
    openai_generator_config,
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
    jina_models._EmbedderConfig.model_validate(jina_media_embedder_config(ENVIRONMENT))


def test_text_and_media_encoders_agree_on_one_search_space() -> None:
    """Recall returns nothing when two encoders in one deployment disagree about the space."""
    text = openai_embedder_config(ENVIRONMENT)
    media = jina_media_embedder_config(ENVIRONMENT)

    for key in ("space_id", "dimension"):
        assert text[key] == media[key]
    assert text["dimension"] == 512


def test_optional_settings_are_omitted_rather_than_sent_as_none() -> None:
    """An absent optional setting must leave the plugin's own default in place."""
    assert "request_timeout_seconds" not in openai_generator_config(ENVIRONMENT)
    assert "device" not in jina_media_embedder_config(ENVIRONMENT)

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
