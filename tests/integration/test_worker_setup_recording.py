"""Integration check that a Worker setup failure lands on the row it would have stranded."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from psycopg import AsyncConnection

import mindbridge.worker as worker_module
from mindbridge.application.capabilities import Embedder
from mindbridge.application.observation_processing import ObservationBatch
from mindbridge.core import (
    DeviceId,
    JobId,
    MediaKind,
    MediaObject,
    MediaObjectId,
    Observation,
    ObservationId,
    SensorKind,
    TenantId,
)
from mindbridge.infrastructure.postgres import PostgresMemoryStore
from mindbridge.infrastructure.task_queue import ObservationProcessingTaskMessage
from mindbridge.worker import WorkerSettings

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_worker_setup")


async def test_an_encoder_that_cannot_load_leaves_a_recorded_row_not_a_pending_one(
    store: PostgresMemoryStore,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row an out-of-memory error stranded is the whole defect, so the table has to show it.

    `task_acks_late` acks and discards the message of a task that raised, and the encoder load
    ran before `ProcessObservation` could claim the row, so the 2026-08-21 evaluation turned 479
    CUDA out-of-memory errors into ~17 `failed` rows and ~318 rows left `pending` with nothing
    queued to deliver them. Only the ledger itself distinguishes the two outcomes.
    """
    observation_id, job_id = await _write_observation(store)

    def load(_plugin: str, _config: Mapping[str, object]) -> Embedder:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker_module, "load_embedder", load)
    settings = WorkerSettings.from_environment(_environment(database_url))
    message = ObservationProcessingTaskMessage(
        tenant_id=TENANT_ID,
        observation_id=observation_id,
        job_id=job_id,
    )

    def deliver() -> BaseException | None:
        """One synchronous Celery delivery, off this test's event loop as a prefork child is."""
        worker_module._dispose_worker_runtime()
        try:
            worker_module.run_observation_processing(settings, message)
        except BaseException as error:
            return error
        finally:
            worker_module._dispose_worker_runtime()
        return None

    assert await _job_state(database_url, job_id) == ("pending", 0, None)
    failure = await asyncio.to_thread(deliver)

    assert isinstance(failure, RuntimeError)
    assert str(failure) == "CUDA out of memory"

    # Recorded, attributed to this delivery's attempt, and named for what actually happened:
    # `mindbridge jobs --include-failed --republish` can act on it, which a `pending` row with no
    # message and no error code gives an operator no reason to believe.
    assert await _job_state(database_url, job_id) == ("failed", 1, "worker_setup_failed")


async def _write_observation(store: PostgresMemoryStore) -> tuple[ObservationId, JobId]:
    """Enqueue one real observation, which is what puts a `pending` job row in the ledger."""
    observation_id = ObservationId("observation_setup_01")
    media_object_id = MediaObjectId("media_setup_01")
    result = await store.write_observation(
        ObservationBatch(
            media_objects=(
                MediaObject(
                    media_object_id=media_object_id,
                    tenant_id=TENANT_ID,
                    kind=MediaKind.VIDEO,
                    uri="s3://memory/tenant_worker_setup/clip_01.mp4",
                    sha256=f"{7:064x}",
                    size_bytes=100,
                    created_at=NOW,
                    duration_ms=4_000,
                ),
            ),
            observation=Observation(
                observation_id=observation_id,
                tenant_id=TENANT_ID,
                device_id=DeviceId("device_01"),
                boot_id="boot_01",
                sequence=1,
                sensor=SensorKind.CAMERA,
                media_object_ids=(media_object_id,),
                occurred_at=NOW,
                ended_at=NOW + timedelta(seconds=4),
                observed_at=NOW,
                clock_offset_ms=0,
            ),
            evidence_spans=(),
        ),
        idempotency_key="observe_setup_01",
        content_digest=f"{107:064x}",
    )
    return observation_id, result.processing_job_id


async def _job_state(
    database_url: str,
    job_id: JobId,
) -> tuple[str, int, str | None]:
    connection = await AsyncConnection.connect(database_url)
    async with connection:
        row = await (
            await connection.execute(
                """
                SELECT state, attempt, error_code FROM jobs
                WHERE tenant_id = %s AND job_id = %s
                """,
                (TENANT_ID, job_id),
            )
        ).fetchone()
    return cast(tuple[str, int, str | None], row)


def _environment(database_url: str) -> Mapping[str, str]:
    """The Worker's own contract, pointed at the disposable database."""
    return {
        "MINDBRIDGE_DATABASE_URL": database_url,
        "MINDBRIDGE_OBJECT_STORAGE_BUCKET": "memory",
        "MINDBRIDGE_TASK_BROKER_URL": "redis://redis:6379/0",
        "MINDBRIDGE_GENERATOR_API_KEY": "generator-secret",
        "MINDBRIDGE_GENERATOR_ENDPOINT": "https://generator.example.test/v1",
        "MINDBRIDGE_EMBEDDER_API_KEY": "text-embedding-secret",
        "MINDBRIDGE_EMBEDDER_ENDPOINT": "https://text.example.test/v1",
    }
