"""Atomic speech-document updates for candidate evidence naming changes."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from mindbridge.infrastructure.local.store._codec import (
    memory_from_row,
    optional_row_text,
    prepare_write_batch,
    row_text,
)
from mindbridge.infrastructure.local.store._corroboration import independent_evidence_enabled
from mindbridge.infrastructure.local.store._identity import dependent_naming_identities
from mindbridge.infrastructure.local.store.records import (
    _read_memory_assets,
    replace_memory_embeddings,
)
from mindbridge.infrastructure.local.store.rows import StoredEmbedding, StoredMemory

NamingProjectionFactory = Callable[
    [Mapping[str, str | None], Sequence[StoredMemory]],
    tuple[tuple[StoredMemory, ...], tuple[StoredEmbedding, ...]],
]


def naming_snapshot(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    identity_ids: Sequence[str] = (),
) -> dict[str, str | None]:
    """Remember candidate names before links or versions can remove dependencies."""
    if not independent_evidence_enabled(connection):
        return {}
    affected = {*dependent_naming_identities(connection, memory_ids), *identity_ids}
    return _names(connection, sorted(affected))


def _names(connection: sqlite3.Connection, identity_ids: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for identity_id in identity_ids:
        row = connection.execute(
            "SELECT name FROM identities WHERE identity_id = ?", (identity_id,)
        ).fetchone()
        if row is not None:
            result[identity_id] = optional_row_text(row, "name")
    return result


def refresh_naming_documents(
    connection: sqlite3.Connection,
    before: Mapping[str, str | None],
    factory: NamingProjectionFactory | None,
) -> None:
    """Rebuild changed names inside their authoritative mutation transaction.

    The callback does read/model preparation only. Candidate naming changes hold
    this transaction during embedding so a model failure rolls back evidence,
    registry names, speech documents, embeddings, and outbox rows together.
    """
    if factory is None or not before:
        return
    changed = {
        identity_id: name
        for identity_id, name in _names(connection, tuple(before)).items()
        if name != before[identity_id]
    }
    if not changed:
        return
    memory_ids = sorted(
        {
            row_text(row, "memory_id")
            for identity_id in changed
            for row in connection.execute(
                """
                SELECT DISTINCT ma.memory_id FROM speech_segments AS s
                JOIN memory_assets AS ma ON ma.asset_id = s.asset_id
                JOIN memory_records AS m ON m.memory_id = ma.memory_id
                WHERE s.speaker_id = ?
                """,
                (identity_id,),
            ).fetchall()
        }
    )
    assets = _read_memory_assets(connection, memory_ids)
    current = {
        memory_id: memory_from_row(
            connection.execute(
                "SELECT * FROM memory_records WHERE memory_id = ?", (memory_id,)
            ).fetchone(),
            assets=assets[memory_id],
            context=None,
        )
        for memory_id in memory_ids
    }
    memories, embeddings = factory(changed, tuple(current.values()))
    # Only text and vectors belong to this callback. The transaction may already
    # have retired, forgotten, or narrowed a row: never reapply old semantics.
    updated = tuple(
        replace(current[memory.memory_id], content=memory.content, updated_at=memory.updated_at)
        for memory in memories
    )
    prepared, vectors, by_memory = prepare_write_batch(updated, embeddings)
    replace_memory_embeddings(connection, prepared, vectors, by_memory)
