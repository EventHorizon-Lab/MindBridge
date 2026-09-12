"""Which identities a memory is in, asked in both directions with one rule."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from mindbridge.infrastructure.local.store._codec import SQLITE_PARAMETER_BATCH, row_text

# Which identities one memory is "in": the records whose semantic subject is that identity, plus
# the records whose media carries a diarised speech segment or a face observation of them. This
# direction answers the search index projection, one document at a time. A merge re-points every
# one of these rows onto the surviving identity, so only canonical IDs ever appear here.
IDENTITY_MEMORY_SQL = """
    SELECT memory_id AS memory_id, ms.identity_id AS identity_id
    FROM memory_semantics AS ms
    WHERE ms.identity_id IS NOT NULL AND ({predicate})
    UNION
    SELECT memory_id, ss.speaker_id
    FROM memory_assets AS ma
    JOIN speech_segments AS ss ON ss.asset_id = ma.asset_id
    WHERE ss.speaker_id IS NOT NULL AND ({predicate})
    UNION
    SELECT memory_id, fo.identity_id
    FROM memory_assets AS ma
    JOIN face_observations AS fo ON fo.asset_id = ma.asset_id
    WHERE {predicate}
"""


# The same membership asked the other way round -- which memories one identity is in -- resolved
# once from three bound copies of the canonical ID. Correlating the projection statement against
# the row being filtered instead made SQLite re-run its three-branch UNION for every candidate
# `memory_records` row, and for every embedding in the store on a merge. The two directions are
# one membership rule; `test_identity_membership_sql_agrees_in_both_directions` pins that.
IDENTITY_MEMORIES_SQL = """
    SELECT ms.memory_id AS memory_id
    FROM memory_semantics AS ms
    WHERE ms.identity_id = ?
    UNION
    SELECT ma.memory_id
    FROM memory_assets AS ma
    JOIN speech_segments AS ss ON ss.asset_id = ma.asset_id
    WHERE ss.speaker_id = ?
    UNION
    SELECT ma.memory_id
    FROM memory_assets AS ma
    JOIN face_observations AS fo ON fo.asset_id = ma.asset_id
    WHERE fo.identity_id = ?
"""


IDENTITY_SCOPE_CLAUSE = f"AND memory_records.memory_id IN ({IDENTITY_MEMORIES_SQL})"


def identity_projections(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Project every identity each memory is about, for the search index to filter on."""
    projections: dict[str, list[str]] = {}
    for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        statement = IDENTITY_MEMORY_SQL.format(predicate=f"memory_id IN ({placeholders})")
        for row in connection.execute(statement, (*batch, *batch, *batch)).fetchall():
            projections.setdefault(row_text(row, "memory_id"), []).append(
                row_text(row, "identity_id")
            )
    return {memory_id: tuple(sorted(values)) for memory_id, values in projections.items()}
