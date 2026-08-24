"""Integration check for the deployable FastAPI composition root."""

import os

import pytest
from fastapi.testclient import TestClient
from mcp import Client
from psycopg import connect

from mindbridge.server import (
    ObjectStorageEnvironment,
    Settings,
    create_app,
    create_mcp_server,
)

DATABASE_URL = os.getenv("MINDBRIDGE_TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        DATABASE_URL is None,
        reason="MINDBRIDGE_TEST_DATABASE_URL is not configured",
    ),
]


def test_production_app_opens_and_closes_runtime_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented Uvicorn factory has a complete, working lifespan."""
    assert DATABASE_URL is not None
    with connect(DATABASE_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    settings = _settings(tenant_api_keys_json='{"tenant_01":["tenant-api-key-000000000000000000"]}')

    with TestClient(create_app(settings)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_production_mcp_opens_and_closes_runtime_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented stdio server uses the same complete production lifespan."""
    assert DATABASE_URL is not None
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    settings = _settings()

    async with Client(create_mcp_server(settings)) as client:
        tools = await client.list_tools()

    assert {tool.name for tool in tools.tools} >= {"memory_observe", "memory_recall"}


def _settings(*, tenant_api_keys_json: str | None = None) -> Settings:
    assert DATABASE_URL is not None
    return Settings(
        database_url=DATABASE_URL,
        object_storage=ObjectStorageEnvironment(
            bucket="memory",
            endpoint_url="https://objects.example.test",
        ),
        task_broker_url="memory://",
        generator_config={
            "api_key": "unit-test-generator-key",
            "endpoint": "https://generator.example.test/v1",
            "model_id": "qwen3.8-max",
        },
        embedder_config={
            "api_key": "unit-test-query-key",
            "endpoint": "https://query.example.test/v1",
            "model_id": "jina-omni",
            "space_id": "jina-v5",
        },
        tenant_api_keys_json=tenant_api_keys_json,
    )
