"""Tests for environment parsing shared by deployable processes."""

import re
from pathlib import Path

import pytest

from mindbridge.configuration import (
    _configuration_document,
    copy_plugin_configuration,
    optional_environment_value,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
    variable_name,
)


def test_environment_values_are_validated_without_transformation() -> None:
    environ = {"REQUIRED": "secret-value", "BLANK": " "}

    assert require_environment_value(environ, "REQUIRED") == "secret-value"
    assert optional_environment_value(environ, "MISSING") is None
    assert optional_environment_value(environ, "BLANK") is None
    with pytest.raises(ValueError, match="REQUIRED_MISSING"):
        require_environment_value(environ, "REQUIRED_MISSING")
    assert validate_plugin_name("openai") == "openai"
    with pytest.raises(ValueError, match="trimmed lowercase"):
        validate_plugin_name("OpenAI")


def test_explicit_plugin_json_is_authoritative_and_direct_config_is_json_only() -> None:
    def unused_default() -> dict[str, object]:
        raise AssertionError("an explicit plugin config must not evaluate its fallback")

    assert plugin_configuration(
        {"PLUGIN_JSON": '{"model":"selected"}'},
        "PLUGIN_JSON",
        unused_default,
    ) == {"model": "selected"}
    with pytest.raises(ValueError, match="JSON values"):
        copy_plugin_configuration({"model": object()}, "plugin_config")
    with pytest.raises(ValueError, match="valid JSON"):
        plugin_configuration({"PLUGIN_JSON": '{"temperature":NaN}'}, "PLUGIN_JSON")
    with pytest.raises(ValueError, match="JSON values"):
        copy_plugin_configuration({"temperature": float("inf")}, "plugin_config")


def test_file_keys_derive_their_variable_names() -> None:
    assert variable_name("max_pool_size", "database") == "MINDBRIDGE_DATABASE_MAX_POOL_SIZE"
    assert variable_name("bucket", "object_storage") == "MINDBRIDGE_OBJECT_STORAGE_BUCKET"
    assert variable_name("model_id", "media_embedder") == "MINDBRIDGE_MEDIA_EMBEDDER_MODEL_ID"
    assert (
        variable_name("minimum_embedding_similarity") == "MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY"
    )


def test_configuration_file_is_found_only_where_it_is_named(tmp_path: Path) -> None:
    named = tmp_path / "named.toml"
    named.write_text("[database]\nmax_pool_size = 8\n", encoding="utf-8")

    assert _configuration_document({"MINDBRIDGE_CONFIG_FILE": str(named)}, None) == {
        "database": {"max_pool_size": 8}
    }
    assert _configuration_document({}, named) == {"database": {"max_pool_size": 8}}
    # No file named and none in the working directory is not an error.
    assert _configuration_document({}, tmp_path / "absent.toml") is None
    # The message must name the path, so an operator sees which one it looked for.
    absent = tmp_path / "absent.toml"
    with pytest.raises(ValueError, match=f"names {re.escape(str(absent))}, which is not a file"):
        _configuration_document({"MINDBRIDGE_CONFIG_FILE": str(absent)}, None)


def test_a_malformed_configuration_file_names_itself(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[database\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain valid TOML"):
        _configuration_document({}, broken)
