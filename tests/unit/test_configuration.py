"""Tests for environment parsing shared by deployable processes."""

import json
import re
from pathlib import Path

import pytest

from mindbridge.configuration import (
    CREDENTIAL_VARIABLES,
    KNOWN_SCALAR_KEYS,
    TOP_LEVEL_KEYS,
    _configuration_document,
    _flattened_plugins,
    _flattened_scalars,
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


def test_scalar_sections_and_top_level_keys_flatten_to_their_variables() -> None:
    document: dict[str, object] = {
        "minimum_embedding_similarity": 0.25,
        "database": {"max_pool_size": 32},
        "object_storage": {"bucket": "mindbridge-media", "endpoint_url": "http://minio:9000"},
        "embedding": {"dimension": 1024, "space_id": "jina-v5"},
    }

    assert _flattened_scalars(document) == {
        "MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY": "0.25",
        "MINDBRIDGE_DATABASE_MAX_POOL_SIZE": "32",
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "mindbridge-media",
        "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL": "http://minio:9000",
        "MINDBRIDGE_EMBEDDING_DIMENSION": "1024",
        "MINDBRIDGE_EMBEDDING_SPACE_ID": "jina-v5",
    }


def test_a_plugin_section_contributes_no_scalar_of_its_own() -> None:
    # A plugin section's body is one config object; only its `plugin` selector is a scalar.
    flattened = _flattened_scalars({"generator": {"plugin": "openai", "model_id": "qwen3.8-max"}})

    assert flattened == {"MINDBRIDGE_GENERATOR_PLUGIN": "openai"}


def test_every_credential_is_refused_inside_the_file() -> None:
    # One synthetic document per credential: a committed file will never carry one, so a guard
    # tested only against real fixtures would run zero times and still report success.
    refused: dict[str, dict[str, object]] = {
        "MINDBRIDGE_DATABASE_URL": {"database": {"url": "postgresql://u:p@h/d"}},
        "MINDBRIDGE_TASK_BROKER_URL": {"task": {"broker_url": "redis://h:6379/0"}},
        "MINDBRIDGE_GENERATOR_API_KEY": {"generator": {"api_key": "sk-secret"}},
        "MINDBRIDGE_EMBEDDER_API_KEY": {"embedder": {"api_key": "sk-secret"}},
        "MINDBRIDGE_TENANT_API_KEYS_JSON": {"tenant": {"api_keys_json": "{}"}},
        "MINDBRIDGE_API_KEY": {"api_key": "sk-secret"},
        "MINDBRIDGE_AML_API_KEY": {"aml": {"api_key": "sk-secret"}},
    }
    assert set(refused) == set(CREDENTIAL_VARIABLES), "a credential gained no rejection test"

    for variable, document in refused.items():
        with pytest.raises(ValueError, match=f"{variable} is a credential"):
            _flattened_scalars(document)


def test_the_file_rejects_shapes_no_variable_could_carry() -> None:
    with pytest.raises(ValueError, match="must not nest"):
        _flattened_scalars({"database": {"pool": {"max_size": 4}}})
    with pytest.raises(ValueError, match="must be text, a number, or a boolean"):
        _flattened_scalars({"object_storage": {"bucket": ["a", "b"]}})
    with pytest.raises(ValueError, match="not a known configuration section"):
        _flattened_scalars({"databse": {"max_pool_size": 8}})
    with pytest.raises(ValueError, match="not a known key"):
        _flattened_scalars({"database": {"max_poool_size": 8}})
    with pytest.raises(ValueError, match="not a known top-level"):
        _flattened_scalars({"minimum_similarity": 0.5})


def test_the_known_keys_cannot_fall_behind_what_the_code_reads() -> None:
    # KNOWN_SCALAR_KEYS is the one table in the loader, so it needs a guard outside itself.
    # Every MINDBRIDGE_<SECTION>_* name the source reads must be derivable from it, or a
    # variable exists that no file key can reach.
    read: set[str] = set()
    for module in Path("src").rglob("*.py"):
        read |= set(re.findall(r"MINDBRIDGE_[A-Z0-9_]+", module.read_text(encoding="utf-8")))

    for section, keys in KNOWN_SCALAR_KEYS.items():
        prefix = f"MINDBRIDGE_{section.upper()}_"
        # Bare `MINDBRIDGE_<SECTION>_` hits are glob notation inside comments, not variables.
        reachable = {name for name in read if name.startswith(prefix) and name != prefix}
        derived = {variable_name(key, section) for key in keys}
        assert reachable - CREDENTIAL_VARIABLES == derived, f"[{section}] drifted"

    for key in TOP_LEVEL_KEYS:
        assert variable_name(key) in read, f"{key} configures no variable the code reads"


def test_a_plugin_section_becomes_its_config_object() -> None:
    flattened = _flattened_plugins(
        {"generator": {"plugin": "openai", "endpoint": "https://g/v1", "model_id": "qwen3.8-max"}},
        {"MINDBRIDGE_GENERATOR_API_KEY": "sk-live"},
    )

    assert json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"]) == {
        "api_key": "sk-live",
        "endpoint": "https://g/v1",
        "model_id": "qwen3.8-max",
    }
    assert "MINDBRIDGE_GENERATOR_PLUGIN" not in flattened


def test_an_environment_override_is_read_in_the_type_the_file_declared() -> None:
    flattened = _flattened_plugins(
        {"generator": {"request_timeout_seconds": 1800, "max_retries": 2, "model_id": "a"}},
        {
            "MINDBRIDGE_GENERATOR_REQUEST_TIMEOUT_SECONDS": "900",
            "MINDBRIDGE_GENERATOR_MAX_RETRIES": "5",
            "MINDBRIDGE_GENERATOR_MODEL_ID": "b",
        },
    )
    generator = json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"])

    # Strict pydantic fields reject "900" and "5" as surely as they reject a missing key.
    assert generator == {"request_timeout_seconds": 900, "max_retries": 5, "model_id": "b"}
    assert isinstance(generator["request_timeout_seconds"], int)


def test_a_boolean_override_is_not_read_as_a_non_empty_string() -> None:
    flattened = _flattened_plugins(
        {"media_sampling": {"generation_proxy": True}},
        {"MINDBRIDGE_MEDIA_SAMPLING_GENERATION_PROXY": "false"},
    )

    # bool("false") is True; bool must be matched before int, which it subclasses.
    assert json.loads(flattened["MINDBRIDGE_MEDIA_SAMPLING_CONFIG_JSON"]) == {
        "generation_proxy": False
    }


def test_the_embedding_space_reaches_every_encoder_from_one_place() -> None:
    document: dict[str, object] = {
        "embedding": {"dimension": 1024, "space_id": "jina-v5", "space_revision": "omni@abc"},
        "embedder": {"endpoint": "https://e/v1"},
        "media_embedder": {"device": "cuda:0"},
        "generator": {"endpoint": "https://g/v1"},
    }
    flattened = _flattened_plugins(document, {})

    for section in ("embedder", "media_embedder"):
        encoded = json.loads(flattened[f"MINDBRIDGE_{section.upper()}_CONFIG_JSON"])
        assert encoded["space_id"] == "jina-v5"
        assert encoded["space_revision"] == "omni@abc"
        assert encoded["dimension"] == 1024
    # The generator shares no embedding space.
    assert "space_id" not in json.loads(flattened["MINDBRIDGE_GENERATOR_CONFIG_JSON"])


def test_an_override_of_a_key_the_file_omits_arrives_as_text() -> None:
    flattened = _flattened_plugins(
        {"embedder": {"endpoint": "https://e/v1"}},
        {"MINDBRIDGE_EMBEDDER_MODEL_REVISION": "abc123"},
    )

    assert json.loads(flattened["MINDBRIDGE_EMBEDDER_CONFIG_JSON"])["model_revision"] == "abc123"
