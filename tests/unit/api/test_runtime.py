"""Tests for explicit production process configuration."""

import pytest

from mindbridge.api import RuntimeSettings


def test_runtime_settings_use_documented_defaults_and_redact_key() -> None:
    """A valid deployment config is concise and safe to inspect in diagnostics."""
    settings = RuntimeSettings.from_environment(
        {
            "MINDBRIDGE_DATABASE_URL": "postgresql://mindbridge@postgres/mindbridge",
            "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
            "MINDBRIDGE_VLM_API_KEY": "secret-unit-test-key",
            "MINDBRIDGE_VLM_ENDPOINT": "https://vlm.example.test/api/v1/chat/completions",
        }
    )

    assert settings.object_storage_endpoint_url is None
    assert settings.object_storage_region == "us-east-1"
    assert settings.vlm_model_id == "qwen3.8-max"
    assert "secret-unit-test-key" not in repr(settings)


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
            vlm_api_key="unit-test-key",
            vlm_endpoint="https://vlm.example.test/v1",
        )
