"""Tests for environment parsing shared by deployable processes."""

import pytest

from mindbridge.configuration import (
    copy_plugin_configuration,
    optional_environment_value,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
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
