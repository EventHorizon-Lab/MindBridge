"""Environment-backed composition for the deployable MindBridge API."""

from __future__ import annotations

import math
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import cast

from fastapi import FastAPI
from mcp.server import MCPServer
from openai.types.shared import ReasoningEffort

from mindbridge.api.app import create_app
from mindbridge.api.auth import TenantApiKeyAuthenticator
from mindbridge.api.mcp import create_mcp_server
from mindbridge.application.kernel import MemoryKernel
from mindbridge.configuration import optional_environment_value, require_environment_value
from mindbridge.core import EmbeddingSpaceReference
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.s3 import S3MediaAccess
from mindbridge.infrastructure.task_queue import (
    CeleryObservationJobPublisher,
    create_task_queue,
)
from mindbridge.models.jina import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
    DEFAULT_JINA_RETRIEVAL_SPACE,
    DEFAULT_JINA_TEXT_MODEL_ID,
    DEFAULT_JINA_TEXT_REVISION,
)
from mindbridge.models.openai_chat import REASONING_EFFORT_VALUES
from mindbridge.models.openai_embeddings import OpenAIJinaEmbedder
from mindbridge.models.openai_omni import (
    DEFAULT_OMNI_MODEL_ID,
    OpenAIOmniAnswerer,
)
from mindbridge.telemetry import configure_telemetry, instrument_fastapi


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated process configuration with redacted credentials."""

    database_url: str = field(repr=False)
    object_storage_bucket: str
    vlm_api_key: str = field(repr=False)
    vlm_endpoint: str
    vlm_model_revision: str
    embedding_api_key: str = field(repr=False)
    embedding_endpoint: str
    text_embedding_api_key: str = field(repr=False)
    text_embedding_endpoint: str
    task_broker_url: str = field(repr=False)
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "us-east-1"
    vlm_model_id: str = DEFAULT_OMNI_MODEL_ID
    embedding_model_id: str = DEFAULT_JINA_OMNI_MODEL_ID
    embedding_model_revision: str = DEFAULT_JINA_OMNI_REVISION
    text_embedding_model_id: str = DEFAULT_JINA_TEXT_MODEL_ID
    text_embedding_model_revision: str = DEFAULT_JINA_TEXT_REVISION
    embedding_space_id: str = DEFAULT_JINA_RETRIEVAL_SPACE.space_id
    embedding_space_revision: str = DEFAULT_JINA_RETRIEVAL_SPACE.revision
    minimum_embedding_similarity: float = 0.0
    answer_reasoning_effort: ReasoningEffort = None
    tenant_api_keys_json: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("object_storage_bucket", self.object_storage_bucket),
            ("object_storage_region", self.object_storage_region),
            ("task_broker_url", self.task_broker_url),
            ("vlm_api_key", self.vlm_api_key),
            ("vlm_endpoint", self.vlm_endpoint),
            ("vlm_model_id", self.vlm_model_id),
            ("vlm_model_revision", self.vlm_model_revision),
            ("embedding_api_key", self.embedding_api_key),
            ("embedding_endpoint", self.embedding_endpoint),
            ("embedding_model_id", self.embedding_model_id),
            ("embedding_model_revision", self.embedding_model_revision),
            ("text_embedding_api_key", self.text_embedding_api_key),
            ("text_embedding_endpoint", self.text_embedding_endpoint),
            ("text_embedding_model_id", self.text_embedding_model_id),
            ("text_embedding_model_revision", self.text_embedding_model_revision),
            ("embedding_space_id", self.embedding_space_id),
            ("embedding_space_revision", self.embedding_space_revision),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            self.object_storage_endpoint_url is not None
            and not self.object_storage_endpoint_url.strip()
        ):
            raise ValueError("object_storage_endpoint_url must not be empty when provided")
        if self.tenant_api_keys_json is not None and not self.tenant_api_keys_json.strip():
            raise ValueError("tenant_api_keys_json must not be empty when provided")
        if (
            self.answer_reasoning_effort is not None
            and self.answer_reasoning_effort not in REASONING_EFFORT_VALUES
        ):
            raise ValueError("answer_reasoning_effort is not supported")
        if (
            not math.isfinite(self.minimum_embedding_similarity)
            or not -1.0 <= self.minimum_embedding_similarity <= 1.0
        ):
            raise ValueError("minimum_embedding_similarity must be between -1 and 1")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        """Read only documented variables and fail before starting on omissions."""
        source = os.environ if environ is None else environ
        return cls(
            database_url=require_environment_value(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage_bucket=require_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"
            ),
            object_storage_endpoint_url=optional_environment_value(
                source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"
            ),
            object_storage_region=source.get("MINDBRIDGE_OBJECT_STORAGE_REGION", "us-east-1"),
            task_broker_url=require_environment_value(source, "MINDBRIDGE_TASK_BROKER_URL"),
            vlm_api_key=require_environment_value(source, "MINDBRIDGE_VLM_API_KEY"),
            vlm_endpoint=require_environment_value(source, "MINDBRIDGE_VLM_ENDPOINT"),
            vlm_model_id=source.get("MINDBRIDGE_VLM_MODEL_ID", DEFAULT_OMNI_MODEL_ID),
            vlm_model_revision=require_environment_value(source, "MINDBRIDGE_VLM_MODEL_REVISION"),
            embedding_api_key=require_environment_value(source, "MINDBRIDGE_EMBEDDING_API_KEY"),
            embedding_endpoint=require_environment_value(source, "MINDBRIDGE_EMBEDDING_ENDPOINT"),
            embedding_model_id=source.get(
                "MINDBRIDGE_EMBEDDING_MODEL_ID", DEFAULT_JINA_OMNI_MODEL_ID
            ),
            embedding_model_revision=source.get(
                "MINDBRIDGE_EMBEDDING_MODEL_REVISION", DEFAULT_JINA_OMNI_REVISION
            ),
            text_embedding_api_key=require_environment_value(
                source, "MINDBRIDGE_TEXT_EMBEDDING_API_KEY"
            ),
            text_embedding_endpoint=require_environment_value(
                source, "MINDBRIDGE_TEXT_EMBEDDING_ENDPOINT"
            ),
            text_embedding_model_id=source.get(
                "MINDBRIDGE_TEXT_EMBEDDING_MODEL_ID", DEFAULT_JINA_TEXT_MODEL_ID
            ),
            text_embedding_model_revision=source.get(
                "MINDBRIDGE_TEXT_EMBEDDING_MODEL_REVISION", DEFAULT_JINA_TEXT_REVISION
            ),
            embedding_space_id=source.get(
                "MINDBRIDGE_EMBEDDING_SPACE_ID", DEFAULT_JINA_RETRIEVAL_SPACE.space_id
            ),
            embedding_space_revision=source.get(
                "MINDBRIDGE_EMBEDDING_SPACE_REVISION", DEFAULT_JINA_RETRIEVAL_SPACE.revision
            ),
            minimum_embedding_similarity=float(
                source.get("MINDBRIDGE_MINIMUM_EMBEDDING_SIMILARITY", "0.0")
            ),
            answer_reasoning_effort=cast(
                ReasoningEffort,
                optional_environment_value(source, "MINDBRIDGE_ANSWER_REASONING_EFFORT"),
            ),
            tenant_api_keys_json=optional_environment_value(
                source, "MINDBRIDGE_TENANT_API_KEYS_JSON"
            ),
        )


@dataclass(frozen=True, slots=True)
class _ProductionRuntime:
    kernel: MemoryKernel
    store: PostgresMemoryStore
    answerer: OpenAIOmniAnswerer
    recall_embedder: OpenAIJinaEmbedder

    @asynccontextmanager
    async def open(self) -> AsyncIterator[None]:
        """Open and close every process-owned network resource exactly once."""
        async with AsyncExitStack() as resources:
            await self.store.open()
            resources.push_async_callback(self.store.close)
            resources.push_async_callback(self.answerer.close)
            resources.push_async_callback(self.recall_embedder.close)
            yield


def create_production_app(settings: RuntimeSettings | None = None) -> FastAPI:
    """Wire PostgreSQL, S3-compatible media, and Omni into one API process."""
    resolved_settings = settings or RuntimeSettings.from_environment()
    if resolved_settings.tenant_api_keys_json is None:
        raise ValueError("MINDBRIDGE_TENANT_API_KEYS_JSON must be configured for the REST API")
    authenticator = TenantApiKeyAuthenticator.from_json(resolved_settings.tenant_api_keys_json)
    runtime = _build_runtime(resolved_settings)
    telemetry = configure_telemetry("mindbridge-api")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with runtime.open():
            yield

    app = create_app(runtime.kernel, authenticator=authenticator, lifespan=lifespan)
    instrument_fastapi(app, telemetry)
    return app


def create_production_mcp_server(
    settings: RuntimeSettings | None = None,
) -> MCPServer[None]:
    """Wire the production kernel into the official MCP server."""
    runtime = _build_runtime(settings or RuntimeSettings.from_environment())
    configure_telemetry("mindbridge-mcp")

    @asynccontextmanager
    async def lifespan(_server: MCPServer[None]) -> AsyncIterator[None]:
        async with runtime.open():
            yield

    return create_mcp_server(runtime.kernel, lifespan=lifespan)


def run_production_mcp() -> None:
    """Run the production MCP server over the official stdio transport."""
    create_production_mcp_server().run(transport="stdio")


def _build_runtime(settings: RuntimeSettings) -> _ProductionRuntime:
    store = PostgresMemoryStore(settings.database_url)
    media_access = S3MediaAccess(
        settings.object_storage_bucket,
        endpoint_url=settings.object_storage_endpoint_url,
        region_name=settings.object_storage_region,
    )
    answerer = OpenAIOmniAnswerer.connect(
        api_key=settings.vlm_api_key,
        endpoint=settings.vlm_endpoint,
        model_id=settings.vlm_model_id,
        model_revision=settings.vlm_model_revision,
        reasoning_effort=settings.answer_reasoning_effort,
    )
    recall_embedder = OpenAIJinaEmbedder.connect(
        api_key=settings.embedding_api_key,
        endpoint=settings.embedding_endpoint,
        model_id=settings.embedding_model_id,
        model_revision=settings.embedding_model_revision,
        document_api_key=settings.text_embedding_api_key,
        document_endpoint=settings.text_embedding_endpoint,
        document_model_id=settings.text_embedding_model_id,
        document_model_revision=settings.text_embedding_model_revision,
        space_reference=EmbeddingSpaceReference(
            space_id=settings.embedding_space_id,
            revision=settings.embedding_space_revision,
        ),
    )
    job_publisher = CeleryObservationJobPublisher(create_task_queue(settings.task_broker_url))
    kernel = MemoryKernel(
        store,
        answerer,
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=job_publisher,
        recall_embedder=recall_embedder,
        minimum_embedding_similarity=settings.minimum_embedding_similarity,
    )
    return _ProductionRuntime(kernel, store, answerer, recall_embedder)
