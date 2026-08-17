"""Environment-backed composition for the deployable MindBridge server."""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, cast

from fastapi import FastAPI
from mcp.server import MCPServer

from mindbridge.api.aml import AmlSettings
from mindbridge.api.app import build_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.api.mcp import build_mcp_server
from mindbridge.application.kernel import MemoryKernel
from mindbridge.application.pipelines import AnswerPipeline, OccurrencePipeline
from mindbridge.configuration import (
    copy_plugin_configuration,
    optional_environment_value,
    plugin_configuration,
    require_environment_value,
    validate_plugin_name,
)
from mindbridge.core import EmbeddedObjectType, EmbeddingSpaceReference, TenantId
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import (
    ObjectStorageEnvironment,
    S3MediaAccess,
    object_storage_from_environment,
)
from mindbridge.infrastructure.task_queue import (
    CeleryObservationJobPublisher,
    create_task_queue,
)
from mindbridge.models import Generator
from mindbridge.models.defaults import (
    DEFAULT_EMBEDDING_DIMENSION,
    embedding_dimension_from_environment,
    openai_embedder_config,
    openai_generator_config,
)
from mindbridge.models.plugins import (
    close_model,
    load_embedder,
    load_generator,
    load_reranker,
)
from mindbridge.telemetry import configure_telemetry, instrument_fastapi


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated infrastructure and directly selected model plugin configuration."""

    database_url: str = field(repr=False)
    object_storage: ObjectStorageEnvironment
    task_broker_url: str = field(repr=False)
    generator_config: Mapping[str, object] = field(repr=False)
    embedder_config: Mapping[str, object] = field(repr=False)
    generator_plugin: str = "openai"
    embedder_plugin: str = "openai"
    reranker_plugin: str | None = None
    reranker_config: Mapping[str, object] = field(default_factory=dict, repr=False)
    minimum_embedding_similarity: float = 0.0
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    tenant_api_keys_json: str | None = field(default=None, repr=False)
    aml_api_key: str | None = field(default=None, repr=False)
    aml_tenant_prefix: str = "bench_aml"

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("task_broker_url", self.task_broker_url),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        for name, value in (
            ("generator_plugin", self.generator_plugin),
            ("embedder_plugin", self.embedder_plugin),
        ):
            validate_plugin_name(value, name)
        if self.reranker_plugin is not None:
            validate_plugin_name(self.reranker_plugin, "reranker_plugin")
        elif self.reranker_config:
            raise ValueError("reranker_config requires reranker_plugin")
        object.__setattr__(
            self,
            "generator_config",
            copy_plugin_configuration(self.generator_config, "generator_config"),
        )
        object.__setattr__(
            self,
            "embedder_config",
            copy_plugin_configuration(self.embedder_config, "embedder_config"),
        )
        object.__setattr__(
            self,
            "reranker_config",
            copy_plugin_configuration(self.reranker_config, "reranker_config"),
        )
        if self.tenant_api_keys_json is not None and not self.tenant_api_keys_json.strip():
            raise ValueError("tenant_api_keys_json must not be empty when provided")
        if (
            not math.isfinite(self.minimum_embedding_similarity)
            or not -1.0 <= self.minimum_embedding_similarity <= 1.0
        ):
            raise ValueError("minimum_embedding_similarity must be between -1 and 1")

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Read the documented deployment contract and fail before startup."""
        source = os.environ if environ is None else environ
        generator_plugin = source.get("MINDBRIDGE_GENERATOR_PLUGIN", "openai")
        embedder_plugin = source.get("MINDBRIDGE_EMBEDDER_PLUGIN", "openai")
        reranker_plugin = optional_environment_value(source, "MINDBRIDGE_RERANKER_PLUGIN")
        generator_config = plugin_configuration(
            source,
            "MINDBRIDGE_GENERATOR_CONFIG_JSON",
            (lambda: openai_generator_config(source)) if generator_plugin == "openai" else None,
        )
        embedder_config = plugin_configuration(
            source,
            "MINDBRIDGE_EMBEDDER_CONFIG_JSON",
            (lambda: openai_embedder_config(source)) if embedder_plugin == "openai" else None,
        )
        reranker_config = (
            plugin_configuration(source, "MINDBRIDGE_RERANKER_CONFIG_JSON", lambda: {})
            if reranker_plugin is not None
            else {}
        )
        return cls(
            database_url=require_environment_value(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage=object_storage_from_environment(source),
            task_broker_url=require_environment_value(source, "MINDBRIDGE_TASK_BROKER_URL"),
            generator_plugin=generator_plugin,
            generator_config=generator_config,
            embedder_plugin=embedder_plugin,
            embedder_config=embedder_config,
            reranker_plugin=reranker_plugin,
            reranker_config=reranker_config,
            minimum_embedding_similarity=float(
                source.get("MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY", "0.0")
            ),
            embedding_dimension=embedding_dimension_from_environment(source),
            tenant_api_keys_json=optional_environment_value(
                source, "MINDBRIDGE_TENANT_API_KEYS_JSON"
            ),
            aml_api_key=optional_environment_value(source, "MINDBRIDGE_AML_API_KEY"),
            aml_tenant_prefix=source.get("MINDBRIDGE_AML_TENANT_PREFIX", "bench_aml"),
        )


class _EmbeddingSpaceProbe(Protocol):
    async def unreachable_embedded_object_types(
        self,
        tenant_id: TenantId,
        space_reference: EmbeddingSpaceReference,
    ) -> tuple[EmbeddedObjectType, ...]: ...


async def _require_reachable_embedding_space(
    probe: _EmbeddingSpaceProbe,
    tenant_ids: tuple[str, ...],
    space_reference: EmbeddingSpaceReference,
) -> None:
    """Refuse to serve an Embedder that cannot reach what a configured tenant already stored."""
    for tenant_id in tenant_ids:
        stranded = await probe.unreachable_embedded_object_types(
            TenantId(tenant_id),
            space_reference,
        )
        if stranded:
            names = ", ".join(object_type.value for object_type in stranded)
            raise ValueError(
                f"tenant {tenant_id!r} stores {names} vectors outside the selected space "
                f"{space_reference}; recall would return nothing instead of failing"
            )


@dataclass(frozen=True, slots=True)
class _Runtime:
    kernel: MemoryKernel
    store: PostgresMemoryStore
    models: tuple[object, ...]
    generator: Generator
    embedding_space: EmbeddingSpaceReference
    tenant_ids: tuple[str, ...]

    @asynccontextmanager
    async def open(self) -> AsyncIterator[None]:
        async with AsyncExitStack() as resources:
            for model in self.models:
                resources.push_async_callback(close_model, model)
            await self.store.open()
            resources.push_async_callback(self.store.close)
            await _require_reachable_embedding_space(
                self.store,
                self.tenant_ids,
                self.embedding_space,
            )
            yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the deployable REST application."""
    resolved = settings or Settings.from_environment()
    if resolved.tenant_api_keys_json is None:
        raise ValueError("MINDBRIDGE_TENANT_API_KEYS_JSON must be configured for the REST API")
    authenticator = TenantApiKeyAuthenticator.from_json(resolved.tenant_api_keys_json)
    runtime = _build_runtime(resolved)
    telemetry = configure_telemetry("mindbridge-api")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with runtime.open():
            yield

    app = build_app(
        runtime.kernel,
        authenticator=authenticator,
        lifespan=lifespan,
        aml=(
            (
                AmlSettings(
                    api_key=resolved.aml_api_key,
                    tenant_prefix=resolved.aml_tenant_prefix,
                ),
                runtime.generator,
            )
            if resolved.aml_api_key is not None
            else None
        ),
    )
    instrument_fastapi(app, telemetry)
    return app


def create_mcp_server(settings: Settings | None = None) -> MCPServer[None]:
    """Create the deployable MCP server."""
    runtime = _build_runtime(settings or Settings.from_environment())
    configure_telemetry("mindbridge-mcp")

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> AsyncIterator[None]:
        async with runtime.open():
            yield

    return build_mcp_server(runtime.kernel, lifespan=lifespan)


def run_mcp() -> None:
    """Run the deployable MCP server over stdio."""
    create_mcp_server().run(transport="stdio")


def _build_runtime(settings: Settings) -> _Runtime:
    store = PostgresMemoryStore(
        settings.database_url, embedding_dimension=settings.embedding_dimension
    )
    media_access = S3MediaAccess(settings.object_storage)
    generator = load_generator(settings.generator_plugin, settings.generator_config)
    embedder = load_embedder(settings.embedder_plugin, settings.embedder_config)
    reranker = (
        load_reranker(settings.reranker_plugin, settings.reranker_config)
        if settings.reranker_plugin is not None
        else None
    )
    publisher = CeleryObservationJobPublisher(create_task_queue(settings.task_broker_url))
    kernel = MemoryKernel(
        store,
        AnswerPipeline(generator),
        OccurrencePipeline(generator),
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=publisher,
        embedder=embedder,
        reranker=reranker,
        minimum_embedding_similarity=settings.minimum_embedding_similarity,
    )
    models = cast(
        tuple[object, ...],
        (generator, embedder, *((reranker,) if reranker is not None else ())),
    )
    tenant_ids = (
        TenantApiKeyAuthenticator.from_json(settings.tenant_api_keys_json).tenant_ids
        if settings.tenant_api_keys_json is not None
        else ()
    )
    return _Runtime(kernel, store, models, generator, embedder.space_reference, tenant_ids)
