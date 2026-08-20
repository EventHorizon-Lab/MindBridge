"""Tests for explicit server composition."""

import pytest

import mindbridge.api.runtime as runtime_module
import mindbridge.consolidation_cli as consolidation_cli
import mindbridge.lifecycle_cli as lifecycle_cli
import mindbridge.worker as worker
from mindbridge.core import EmbeddedObjectType, EmbeddingSpaceReference
from mindbridge.infrastructure.postgres import (
    DEFAULT_DATABASE_MAX_POOL_SIZE,
    resolve_database_max_pool_size,
)
from mindbridge.server import ObjectStorageEnvironment, Settings, create_app

SPACE = EmbeddingSpaceReference(space_id="jina-v5")


def test_settings_use_deployable_defaults_and_redact_credentials() -> None:
    settings = Settings.from_environment(_environment())

    assert settings.generator_plugin == "openai"
    assert settings.embedder_plugin == "openai"
    assert settings.generator_config["model_id"] == "qwen3.8-max"
    assert settings.embedder_config["model_id"] == (
        "jinaai/jina-embeddings-v5-omni-small-retrieval"
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


def test_settings_require_generator_credentials_for_bundled_default() -> None:
    environment = dict(_environment())
    del environment["MINDBRIDGE_GENERATOR_API_KEY"]

    with pytest.raises(ValueError, match="MINDBRIDGE_GENERATOR_API_KEY"):
        Settings.from_environment(environment)


def test_settings_reject_empty_direct_configuration() -> None:
    with pytest.raises(ValueError, match="database_url"):
        _settings(database_url=" ")


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
        space_reference = SPACE

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
    monkeypatch.setattr(
        runtime_module, "PostgresMemoryStore", lambda *_arguments, **_options: Store()
    )
    monkeypatch.setattr(runtime_module, "S3MediaAccess", lambda *_arguments, **_options: object())
    monkeypatch.setattr(
        runtime_module, "create_task_queue", lambda *_arguments, **_keywords: object()
    )
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


async def test_startup_rejects_a_space_no_stored_vector_can_match() -> None:
    """An unreachable tenant must abort startup instead of degrading recall to empty results."""
    asked: list[str] = []

    class Probe:
        async def unreachable_embedded_object_types(
            self,
            tenant_id: str,
            space_reference: EmbeddingSpaceReference,
        ) -> tuple[EmbeddedObjectType, ...]:
            asked.append(tenant_id)
            if tenant_id != "tenant_02":
                return ()
            return (EmbeddedObjectType.EVIDENCE_SPAN, EmbeddedObjectType.CLAIM)

    await runtime_module._require_reachable_embedding_space(Probe(), ("tenant_01",), SPACE)
    with pytest.raises(ValueError, match=r"tenant_02.*evidence_span, claim"):
        await runtime_module._require_reachable_embedding_space(
            Probe(),
            ("tenant_01", "tenant_02"),
            SPACE,
        )

    assert asked == ["tenant_01", "tenant_01", "tenant_02"]


async def test_runtime_probes_every_configured_tenant_on_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected Embedder's own space is what startup compares against stored vectors."""
    probed: list[tuple[str, EmbeddingSpaceReference]] = []

    class Embedder:
        space_reference = SPACE

    class Store:
        async def open(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def unreachable_embedded_object_types(
            self,
            tenant_id: str,
            space_reference: EmbeddingSpaceReference,
        ) -> tuple[EmbeddedObjectType, ...]:
            probed.append((tenant_id, space_reference))
            return ()

    monkeypatch.setattr(runtime_module, "load_generator", lambda *_arguments: object())
    monkeypatch.setattr(runtime_module, "load_embedder", lambda *_arguments: Embedder())
    monkeypatch.setattr(
        runtime_module, "PostgresMemoryStore", lambda *_arguments, **_options: Store()
    )
    monkeypatch.setattr(runtime_module, "S3MediaAccess", lambda *_arguments, **_options: object())
    monkeypatch.setattr(
        runtime_module, "create_task_queue", lambda *_arguments, **_keywords: object()
    )
    monkeypatch.setattr(
        runtime_module,
        "CeleryObservationJobPublisher",
        lambda *_arguments: object(),
    )
    monkeypatch.setattr(runtime_module, "MemoryKernel", lambda *_arguments, **_options: object())

    runtime = runtime_module._build_runtime(
        _settings(
            tenant_api_keys_json=(
                '{"tenant_02":["tenant-api-key-000000000000000000"],'
                '"tenant_01":["tenant-api-key-111111111111111111"]}'
            )
        )
    )
    async with runtime.open():
        pass

    assert probed == [("tenant_01", SPACE), ("tenant_02", SPACE)]


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "database_url": "postgresql://mindbridge@postgres/mindbridge",
        "object_storage": ObjectStorageEnvironment(bucket="memory"),
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
        "MINDBRIDGE_EMBEDDER_API_KEY": "query-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://query.example.test/v1",
        "MINDBRIDGE_TENANT_API_KEYS_JSON": ('{"tenant_01":["tenant-api-key-000000000000000000"]}'),
    }


def test_every_process_that_opens_a_pool_reads_one_ceiling() -> None:
    """The variable governs the server's total, so one process honouring it is not enough."""
    # The API, the worker, and the two sweeps share a PostgreSQL `max_connections`, so a
    # deployment lowering the ceiling has to lower all four. Previously only Settings read the
    # variable and the other three silently kept the default.
    assert resolve_database_max_pool_size({}) == DEFAULT_DATABASE_MAX_POOL_SIZE
    assert resolve_database_max_pool_size({"MINDBRIDGE_DATABASE_MAX_POOL_SIZE": "8"}) == 8
    assert (
        Settings.from_environment(
            {**_environment(), "MINDBRIDGE_DATABASE_MAX_POOL_SIZE": "8"}
        ).database_max_pool_size
        == 8
    )
    with pytest.raises(ValueError, match="MINDBRIDGE_DATABASE_MAX_POOL_SIZE"):
        resolve_database_max_pool_size({"MINDBRIDGE_DATABASE_MAX_POOL_SIZE": "0"})
    for module in (worker, consolidation_cli, lifecycle_cli):
        assert "resolve_database_max_pool_size" in module.__dict__
