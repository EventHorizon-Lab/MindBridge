"""Tests for explicit server composition."""

import pytest

import mindbridge.api.runtime as runtime_module
from mindbridge.server import Settings, create_app


def test_settings_use_deployable_defaults_and_redact_credentials() -> None:
    settings = Settings.from_environment(_environment())

    assert settings.generator_plugin == "openai"
    assert settings.embedder_plugin == "openai"
    assert settings.generator_config["model_id"] == "qwen3.8-max"
    assert settings.generator_config["model_revision"] == "deployment-revision"
    assert settings.generator_config["reasoning_effort"] == "low"
    assert settings.embedder_config["model_id"] == (
        "jinaai/jina-embeddings-v5-omni-small-retrieval"
    )
    assert settings.embedder_config["document_model_id"] == (
        "jinaai/jina-embeddings-v5-text-small-retrieval"
    )
    assert settings.minimum_embedding_similarity == 0.0
    assert "secret" not in repr(settings)


def test_explicit_plugin_json_does_not_require_bundled_provider_variables() -> None:
    settings = Settings.from_environment(
        {
            "MINDBRIDGE_DATABASE_URL": "postgresql://mindbridge@postgres/mindbridge",
            "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
            "MINDBRIDGE_TASK_BROKER_URL": "redis://redis:6379/0",
            "MINDBRIDGE_GENERATOR_PLUGIN": "anthropic",
            "MINDBRIDGE_GENERATOR_CONFIG_JSON": '{"model":"claude"}',
            "MINDBRIDGE_EMBEDDER_PLUGIN": "custom",
            "MINDBRIDGE_EMBEDDER_CONFIG_JSON": '{"model":"local"}',
        }
    )

    assert settings.generator_config == {"model": "claude"}
    assert settings.embedder_config == {"model": "local"}


def test_settings_require_generator_revision_for_bundled_default() -> None:
    environment = dict(_environment())
    del environment["MINDBRIDGE_GENERATOR_MODEL_REVISION"]

    with pytest.raises(ValueError, match="MINDBRIDGE_GENERATOR_MODEL_REVISION"):
        Settings.from_environment(environment)


def test_settings_reject_empty_direct_configuration() -> None:
    with pytest.raises(ValueError, match="object_storage_bucket"):
        _settings(object_storage_bucket=" ")


def test_settings_reject_invalid_plugin_name() -> None:
    with pytest.raises(ValueError, match="generator_plugin"):
        _settings(generator_plugin="OpenAI")


def test_settings_reject_invalid_embedding_similarity() -> None:
    with pytest.raises(ValueError, match="minimum_embedding_similarity"):
        _settings(minimum_embedding_similarity=float("nan"))


def test_rest_fails_closed_without_tenant_credentials() -> None:
    with pytest.raises(ValueError, match="MINDBRIDGE_TENANT_API_KEYS_JSON"):
        create_app(_settings())


async def test_runtime_closes_every_model_and_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[str] = []

    class Model:
        def __init__(self, name: str, *, falsey: bool = False) -> None:
            self.name = name
            self.falsey = falsey

        def __bool__(self) -> bool:
            return not self.falsey

        async def close(self) -> None:
            closed.append(self.name)

    class Store:
        async def open(self) -> None:
            return None

        async def close(self) -> None:
            closed.append("store")

    generator = Model("generator")
    embedder = Model("embedder", falsey=True)
    monkeypatch.setattr(runtime_module, "load_generator", lambda *_arguments: generator)
    monkeypatch.setattr(runtime_module, "load_embedder", lambda *_arguments: embedder)
    monkeypatch.setattr(runtime_module, "PostgresMemoryStore", lambda *_arguments: Store())
    monkeypatch.setattr(runtime_module, "S3MediaAccess", lambda *_arguments, **_options: object())
    monkeypatch.setattr(runtime_module, "create_task_queue", lambda *_arguments: object())
    monkeypatch.setattr(
        runtime_module,
        "CeleryObservationJobPublisher",
        lambda *_arguments: object(),
    )
    monkeypatch.setattr(runtime_module, "MemoryKernel", lambda *_arguments, **_options: object())

    runtime = runtime_module._build_runtime(_settings())
    async with runtime.open():
        pass

    assert closed == ["store", "embedder", "generator"]


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://mindbridge@postgres/mindbridge",
        "object_storage_bucket": "memory",
        "task_broker_url": "redis://redis:6379/0",
        "generator_config": {"model": "test"},
        "embedder_config": {"model": "test"},
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


def _environment() -> dict[str, str]:
    return {
        "MINDBRIDGE_DATABASE_URL": ("postgresql://mindbridge:database-secret@postgres/mindbridge"),
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://:broker-secret@redis:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
        "MINDBRIDGE_GENERATOR_MODEL_REVISION": "deployment-revision",
        "MINDBRIDGE_GENERATOR_REASONING_EFFORT": "low",
        "MINDBRIDGE_EMBEDDER_API_KEY": "query-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://query.example.test/v1",
        "MINDBRIDGE_EMBEDDER_DOCUMENT_API_KEY": "document-secret",
        "MINDBRIDGE_EMBEDDER_DOCUMENT_ENDPOINT": "https://document.example.test/v1",
        "MINDBRIDGE_TENANT_API_KEYS_JSON": ('{"tenant_01":["tenant-api-key-000000000000000000"]}'),
    }
