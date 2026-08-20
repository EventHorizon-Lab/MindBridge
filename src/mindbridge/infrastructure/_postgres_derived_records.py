"""Shared idempotent PostgreSQL writes for derived graph records."""

from __future__ import annotations

import hashlib
import json
from typing import cast

from mindbridge.core import DomainInvariantError, Event, MemoryRecord
from mindbridge.infrastructure._postgres_types import DatabaseConnection


async def write_event(connection: DatabaseConnection, event: Event) -> bool:
    """Insert one deterministic Event and its provenance links."""
    content_digest = event_content_digest(event)
    cursor = await connection.execute(
        """
        INSERT INTO events (
            tenant_id, event_id, parent_event_id, hierarchy_level, description, salience,
            status, occurred_at, ended_at, model_id, prompt_version,
            content_digest, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING event_id
        """,
        (
            event.tenant_id,
            event.event_id,
            event.parent_event_id,
            event.hierarchy_level.value,
            event.description,
            event.salience,
            event.status.value,
            event.occurred_at,
            event.ended_at,
            event.model_reference.model_id,
            event.prompt_version,
            content_digest,
            event.created_at,
        ),
    )
    created = await cursor.fetchone() is not None
    if not created:
        row = await (
            await connection.execute(
                """
                SELECT content_digest, hierarchy_level FROM events
                WHERE tenant_id = %s AND event_id = %s
                """,
                (event.tenant_id, event.event_id),
            )
        ).fetchone()
        if row is None or cast(tuple[str, str], row) != (
            content_digest,
            event.hierarchy_level.value,
        ):
            raise DomainInvariantError("event identifier already stores different content")
    await _write_event_links(connection, event)
    return created


def event_content_digest(event: Event) -> str:
    """Hash immutable Event content while excluding mutable hierarchy state."""
    return _digest(
        {
            "event_id": event.event_id,
            "tenant_id": event.tenant_id,
            "observation_ids": event.observation_ids,
            "evidence_ids": event.evidence_ids,
            "occurred_at": event.occurred_at.isoformat(),
            "ended_at": event.ended_at.isoformat(),
            "description": event.description,
            "salience": event.salience,
            "model_id": event.model_reference.model_id,
            "prompt_version": event.prompt_version,
            "created_at": event.created_at.isoformat(),
        }
    )


def derived_memory_content_digest(memory: MemoryRecord) -> str:
    """Hash one model-derived MemoryRecord for deterministic retries."""
    return _digest(
        {
            "memory_id": memory.memory_id,
            "tenant_id": memory.tenant_id,
            "memory_type": memory.memory_type.value,
            "summary": memory.summary,
            "evidence_ids": memory.evidence_ids,
            "occurred_at": memory.occurred_at.isoformat(),
            "ended_at": memory.ended_at.isoformat(),
            "created_at": memory.created_at.isoformat(),
            "verification_status": memory.verification_status.value,
            "state": memory.state.value,
            "salience": memory.salience,
            "supersedes_memory_id": memory.supersedes_memory_id,
            "model_id": (
                memory.model_reference.model_id if memory.model_reference is not None else None
            ),
        }
    )


async def _write_event_links(connection: DatabaseConnection, event: Event) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            """
            INSERT INTO event_observations (tenant_id, event_id, observation_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
            """,
            (
                (event.tenant_id, event.event_id, observation_id)
                for observation_id in event.observation_ids
            ),
        )
        await cursor.executemany(
            """
            INSERT INTO event_evidence (tenant_id, event_id, evidence_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
            """,
            ((event.tenant_id, event.event_id, evidence_id) for evidence_id in event.evidence_ids),
        )


def _digest(value: dict[str, object]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
