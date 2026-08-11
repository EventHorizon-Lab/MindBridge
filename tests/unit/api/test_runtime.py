"""Tests for explicit production process configuration."""

import pytest

from mindbridge.api import RuntimeSettings, create_production_app


def test_runtime_settings_use_documented_defaults_and_redact_key() -> None:
    """A valid deployment config is concise and safe to inspect in diagnostics."""
    settings = RuntimeSettings.from_environment(
        {
            "MINDBRIDGE_DATABASE_URL": (
                "postgresql://mindbridge:database-secret@postgres/mindbridge"
            ),
            "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
            "MINDBRIDGE_TASK_BROKER_URL": "redis://:broker-secret@redis:6379/0",
            "MINDBRIDGE_VLM_API_KEY": "secret-unit-test-key",
            "MINDBRIDGE_VLM_ENDPOINT": "https://vlm.example.test/api/v1/chat/completions",
            "MINDBRIDGE_EMBEDDING_API_KEY": "secret-embedding-key",
            "MINDBRIDGE_EMBEDDING_ENDPOINT": "https://embedding.example.test/v1/embeddings",
            "MINDBRIDGE_TENANT_API_KEYS_JSON": (
                '{"tenant_01":["tenant-api-key-000000000000000000"]}'
            ),
        }
    )

    assert settings.object_storage_endpoint_url is None
    assert settings.object_storage_region == "us-east-1"
    assert settings.vlm_model_id == "qwen3.8-max"
    assert settings.embedding_model_id == "jinaai/jina-embeddings-v5-omni-small-retrieval"
    assert settings.embedding_model_revision == "12949877f0092093f366c6450340011320152a05"
    assert settings.minimum_embedding_similarity == 0.0
    assert settings.tenant_api_keys_json is not None
    assert "secret-unit-test-key" not in repr(settings)
    assert "secret-embedding-key" not in repr(settings)
    assert "broker-secret" not in repr(settings)
    assert "database-secret" not in repr(settings)
    assert "tenant-api-key" not in repr(settings)


def test_runtime_settings_fail_fast_on_missing_required_value() -> None:
    """The process cannot start in a partially wired state."""
    with pytest.raises(ValueError, match="MINDBRIDGE_DATABASE_URL"):
        RuntimeSettings.from_environment({})


def test_runtime_settings_reject_empty_direct_configuration() -> None:
    """Programmatic composition has the same validation as environment loading."""
    with pytest.raises(ValueError, match="object_storage_bucket"):
        RuntimeSettings(
            database_url="postgresql://mindbridge@postgres/mindbridge",
            object_storage_bucket=" ",
            task_broker_url="redis://redis:6379/0",
            vlm_api_key="unit-test-key",
            vlm_endpoint="https://vlm.example.test/v1",
            embedding_api_key="unit-test-embedding-key",
            embedding_endpoint="https://embedding.example.test/v1",
        )


def test_runtime_settings_reject_invalid_embedding_similarity() -> None:
    with pytest.raises(ValueError, match="minimum_embedding_similarity"):
        RuntimeSettings(
            database_url="postgresql://mindbridge@postgres/mindbridge",
            object_storage_bucket="memory",
            task_broker_url="redis://redis:6379/0",
            vlm_api_key="unit-test-key",
            vlm_endpoint="https://vlm.example.test/v1",
            embedding_api_key="unit-test-embedding-key",
            embedding_endpoint="https://embedding.example.test/v1",
            minimum_embedding_similarity=float("nan"),
        )


def test_production_rest_fails_closed_without_tenant_credentials() -> None:
    settings = RuntimeSettings(
        database_url="postgresql://mindbridge@postgres/mindbridge",
        object_storage_bucket="memory",
        task_broker_url="redis://redis:6379/0",
        vlm_api_key="unit-test-key",
        vlm_endpoint="https://vlm.example.test/v1",
        embedding_api_key="unit-test-embedding-key",
        embedding_endpoint="https://embedding.example.test/v1",
    )

    with pytest.raises(ValueError, match="MINDBRIDGE_TENANT_API_KEYS_JSON"):
        create_production_app(settings)
