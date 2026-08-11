"""Environment-backed composition for the deployable MindBridge API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI

from mindbridge.api.app import create_app
from mindbridge.application import MemoryKernel
from mindbridge.infrastructure import (
    CeleryObservationJobPublisher,
    PostgresMemoryStore,
    S3MediaAccess,
    create_task_queue,
)
from mindbridge.models import (
    DEFAULT_JINA_OMNI_MODEL_ID,
    DEFAULT_JINA_OMNI_REVISION,
    DEFAULT_OMNI_MODEL_ID,
    OpenAIJinaEmbedder,
    OpenAIOmniAnswerer,
)


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Validated process configuration with redacted credentials."""

    database_url: str = field(repr=False)
    object_storage_bucket: str
    vlm_api_key: str = field(repr=False)
    vlm_endpoint: str
    embedding_api_key: str = field(repr=False)
    embedding_endpoint: str
    task_broker_url: str = field(repr=False)
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "us-east-1"
    vlm_model_id: str = DEFAULT_OMNI_MODEL_ID
    embedding_model_id: str = DEFAULT_JINA_OMNI_MODEL_ID
    embedding_model_revision: str = DEFAULT_JINA_OMNI_REVISION

    def __post_init__(self) -> None:
        for name, value in (
            ("database_url", self.database_url),
            ("object_storage_bucket", self.object_storage_bucket),
            ("object_storage_region", self.object_storage_region),
            ("task_broker_url", self.task_broker_url),
            ("vlm_api_key", self.vlm_api_key),
            ("vlm_endpoint", self.vlm_endpoint),
            ("vlm_model_id", self.vlm_model_id),
            ("embedding_api_key", self.embedding_api_key),
            ("embedding_endpoint", self.embedding_endpoint),
            ("embedding_model_id", self.embedding_model_id),
            ("embedding_model_revision", self.embedding_model_revision),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if (
            self.object_storage_endpoint_url is not None
            and not self.object_storage_endpoint_url.strip()
        ):
            raise ValueError("object_storage_endpoint_url must not be empty when provided")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeSettings:
        """Read only documented variables and fail before starting on omissions."""
        source = os.environ if environ is None else environ
        return cls(
            database_url=_required(source, "MINDBRIDGE_DATABASE_URL"),
            object_storage_bucket=_required(source, "MINDBRIDGE_OBJECT_STORAGE_BUCKET"),
            object_storage_endpoint_url=_optional(source, "MINDBRIDGE_OBJECT_STORAGE_ENDPOINT_URL"),
            object_storage_region=source.get("MINDBRIDGE_OBJECT_STORAGE_REGION", "us-east-1"),
            task_broker_url=_required(source, "MINDBRIDGE_TASK_BROKER_URL"),
            vlm_api_key=_required(source, "MINDBRIDGE_VLM_API_KEY"),
            vlm_endpoint=_required(source, "MINDBRIDGE_VLM_ENDPOINT"),
            vlm_model_id=source.get("MINDBRIDGE_VLM_MODEL_ID", DEFAULT_OMNI_MODEL_ID),
            embedding_api_key=_required(source, "MINDBRIDGE_EMBEDDING_API_KEY"),
            embedding_endpoint=_required(source, "MINDBRIDGE_EMBEDDING_ENDPOINT"),
            embedding_model_id=source.get(
                "MINDBRIDGE_EMBEDDING_MODEL_ID", DEFAULT_JINA_OMNI_MODEL_ID
            ),
            embedding_model_revision=source.get(
                "MINDBRIDGE_EMBEDDING_MODEL_REVISION", DEFAULT_JINA_OMNI_REVISION
            ),
        )


def create_production_app(settings: RuntimeSettings | None = None) -> FastAPI:
    """Wire PostgreSQL, S3-compatible media, and Omni into one API process."""
    runtime = settings or RuntimeSettings.from_environment()
    store = PostgresMemoryStore(runtime.database_url)
    media_access = S3MediaAccess(
        runtime.object_storage_bucket,
        endpoint_url=runtime.object_storage_endpoint_url,
        region_name=runtime.object_storage_region,
    )
    answerer = OpenAIOmniAnswerer.connect(
        api_key=runtime.vlm_api_key,
        endpoint=runtime.vlm_endpoint,
        model_id=runtime.vlm_model_id,
    )
    recall_embedder = OpenAIJinaEmbedder.connect(
        api_key=runtime.embedding_api_key,
        endpoint=runtime.embedding_endpoint,
        model_id=runtime.embedding_model_id,
        model_revision=runtime.embedding_model_revision,
    )
    job_publisher = CeleryObservationJobPublisher(create_task_queue(runtime.task_broker_url))
    kernel = MemoryKernel(
        store,
        answerer,
        embedding_index=store,
        media_deleter=media_access,
        media_url_signer=media_access,
        observation_job_publisher=job_publisher,
        recall_embedder=recall_embedder,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as resources:
            await store.open()
            resources.push_async_callback(store.close)
            resources.push_async_callback(answerer.close)
            resources.push_async_callback(recall_embedder.close)
            yield

    return create_app(kernel, lifespan=lifespan)


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value if value is not None and value.strip() else None
