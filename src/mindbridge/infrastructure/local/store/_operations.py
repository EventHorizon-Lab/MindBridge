"""The append-only memory-operation log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    optional_datetime_from_row,
    optional_row_text,
    parse_datetime,
    row_text,
)
from mindbridge.infrastructure.local.store.rows import (
    StoredEvidenceClauseChange,
    StoredOperation,
    require_identifier,
)


def active_operation_id(connection: sqlite3.Connection, operation_key: str) -> int | None:
    row = connection.execute(
        """
        SELECT operation_id FROM memory_operations
        WHERE operation_key = ? AND rolled_back_at IS NULL
        """,
        (operation_key,),
    ).fetchone()
    return None if row is None else int(row["operation_id"])


def later_operation_depends_on(
    connection: sqlite3.Connection,
    operation_id: int,
    memory_ids: Sequence[str],
) -> bool:
    """Return whether a later standing operation names one of these outputs."""
    ids = tuple(dict.fromkeys(memory_ids))
    for memory_id in ids:
        require_identifier(memory_id, "memory_id")
    for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH - 1):
        batch = ids[offset : offset + SQLITE_PARAMETER_BATCH - 1]
        placeholders = ", ".join("?" for _memory_id in batch)
        if (
            connection.execute(
                f"""
            SELECT 1
            FROM memory_operations AS o
            JOIN json_tree(json_array(
                json_extract(o.operation_json, '$.evidence_ids'),
                json_extract(o.operation_json, '$.target_ids'),
                json_extract(o.effects_json, '$.created_ids'),
                json_extract(o.effects_json, '$.changed_ids'),
                json_extract(o.effects_json, '$.activated_ids'),
                json_extract(o.effects_json, '$.linked'),
                json_extract(o.effects_json, '$.superseded')
            )) AS j
            WHERE o.operation_id > ? AND o.rolled_back_at IS NULL
              AND j.type = 'text' AND j.value IN ({placeholders})
            LIMIT 1
            """,
                (operation_id, *batch),
            ).fetchone()
            is not None
        ):
            return True
    return False


def insert_operation(connection: sqlite3.Connection, operation: StoredOperation) -> int:
    cursor = connection.execute(
        """
        INSERT INTO memory_operations (
            operation_key, intent, trigger, model_id, recipe,
            operation_json, effects_json, applied_at, rolled_back_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            operation.operation_key,
            operation.intent,
            operation.trigger,
            operation.model_id,
            operation.recipe,
            operation.operation_json,
            json.dumps(
                {
                    "created_ids": list(operation.created_ids),
                    "changed_ids": list(operation.changed_ids),
                    "activated_ids": list(operation.activated_ids),
                    "forgotten_ids": list(operation.forgotten_ids),
                    "linked": [list(pair) for pair in operation.linked],
                    "linked_clauses": [list(pair) for pair in operation.linked_clauses],
                    "clause_changes": [
                        {
                            "memory_id": change.memory_id,
                            "clause_id": change.clause_id,
                            "previous_active": change.previous_active,
                            "previous_confidence": change.previous_confidence,
                            "applied_confidence": change.applied_confidence,
                            "applied_recorded_at": datetime_text(change.applied_recorded_at),
                            "applied_version": change.applied_version,
                        }
                        for change in operation.clause_changes
                    ],
                    "superseded": [list(pair) for pair in operation.superseded],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            datetime_text(operation.applied_at),
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("failed to log a memory operation")
    return int(cursor.lastrowid)


def operation_from_row(row: sqlite3.Row) -> StoredOperation:
    effects = json.loads(row_text(row, "effects_json"))
    if not isinstance(effects, dict):
        raise ValueError("logged operation effects must encode an object")
    return StoredOperation(
        operation_id=int(row["operation_id"]),
        operation_key=row_text(row, "operation_key"),
        intent=row_text(row, "intent"),
        trigger=row_text(row, "trigger"),
        model_id=optional_row_text(row, "model_id"),
        recipe=optional_row_text(row, "recipe"),
        operation_json=row_text(row, "operation_json"),
        created_ids=tuple(effects.get("created_ids") or ()),
        changed_ids=tuple(effects.get("changed_ids") or ()),
        activated_ids=tuple(effects.get("activated_ids") or ()),
        forgotten_ids=tuple(effects.get("forgotten_ids") or ()),
        linked=tuple((pair[0], pair[1]) for pair in effects.get("linked") or ()),
        linked_clauses=tuple((pair[0], pair[1]) for pair in effects.get("linked_clauses") or ()),
        clause_changes=tuple(
            StoredEvidenceClauseChange(
                memory_id=change["memory_id"],
                clause_id=change["clause_id"],
                previous_active=change["previous_active"],
                previous_confidence=change.get("previous_confidence"),
                applied_confidence=change["applied_confidence"],
                applied_recorded_at=parse_datetime(change["applied_recorded_at"]),
                applied_version=int(change["applied_version"]),
            )
            for change in effects.get("clause_changes") or ()
        ),
        superseded=tuple((pair[0], int(pair[1])) for pair in effects.get("superseded") or ()),
        applied_at=parse_datetime(row_text(row, "applied_at")),
        rolled_back_at=optional_datetime_from_row(row, "rolled_back_at"),
        outcome=optional_row_text(row, "outcome"),
        outcome_note=optional_row_text(row, "outcome_note"),
    )
