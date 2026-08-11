"""Integration check for the deployable FastAPI composition root."""

import os

import pytest
from fastapi.testclient import TestClient
from psycopg import connect

from mindbridge.api import RuntimeSettings, create_production_app

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
    settings = RuntimeSettings(
        database_url=DATABASE_URL,
        object_storage_bucket="memory",
        object_storage_endpoint_url="https://objects.example.test",
        task_broker_url="memory://",
        vlm_api_key="unit-test-vlm-key",
        vlm_endpoint="https://vlm.example.test/api/v1/chat/completions",
        embedding_api_key="unit-test-embedding-key",
        embedding_endpoint="https://embedding.example.test/api/v1/embeddings",
    )

    with TestClient(create_production_app(settings)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
