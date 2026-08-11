"""Tests for environment parsing shared by deployable processes."""

import pytest

from mindbridge.configuration import optional_environment_value, require_environment_value


def test_environment_values_are_validated_without_transformation() -> None:
    environ = {"REQUIRED": "secret-value", "BLANK": " "}

    assert require_environment_value(environ, "REQUIRED") == "secret-value"
    assert optional_environment_value(environ, "MISSING") is None
    assert optional_environment_value(environ, "BLANK") is None
    with pytest.raises(ValueError, match="REQUIRED_MISSING"):
        require_environment_value(environ, "REQUIRED_MISSING")
