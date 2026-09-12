"""Typed lineage: evidence clauses, bitemporal versions, and their reconciliation.

Every function takes an open connection and runs inside the caller's transaction. Derived
confidence and visibility are recomputed from the remaining independent evidence rather than
stored as independent facts, so retiring or losing a source never destroys the observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    optional_datetime_from_row,
    optional_datetime_text,
    optional_row_text,
    parse_datetime,
    row_text,
)
from mindbridge.infrastructure.local.store._outbox import queue_memory_embeddings
from mindbridge.infrastructure.local.store.errors import StaleOperationError
from mindbridge.infrastructure.local.store.rows import (
    StoredEvidenceClauseChange,
    StoredMemory,
    canonical_object_json,
    require_identifier,
)
from mindbridge.types import EvidenceBasis, MemoryContext, MemoryKind


def write_memory_context(
    connection: sqlite3.Connection,
    memory_id: str,
    context: MemoryContext,
    *,
    transaction_memory_ids: set[str] | None = None,
    superseded: list[tuple[str, int]] | None = None,
    write_evidence: bool = True,
    recorded_at: datetime | None = None,
) -> None:
    context_recorded_at = context.recorded_at if recorded_at is None else recorded_at
    existing = connection.execute(
        "SELECT lineage_id FROM memory_semantics WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if existing is not None and write_evidence:
        for source_memory_id in context.evidence_ids:
            add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=context.confidence,
                recorded_at=context_recorded_at,
            )
        if not version_retired(connection, memory_id):
            return
        # Every version of this claim is retired: it was superseded or rolled back, and asserting
        # it again is a fresh claim rather than a repeat of a standing one. Restating it as the
        # next version reconciles the lineage the same way a first assertion does, so renaming a
        # person back, or re-registering a name after `rollback`, lands instead of doing nothing.

    lineage_id = (
        row_text(existing, "lineage_id")
        if existing is not None
        else context.lineage_id or memory_id
    )
    recorded_at = _next_lineage_transaction_time(
        connection,
        context_recorded_at,
        lineage_id=lineage_id,
        kind=context.kind.value,
        transaction_memory_ids=transaction_memory_ids,
    )
    valid_from = context.valid_from
    valid_until = context.valid_until
    spatial = context.spatial
    orientation = None if spatial is None else spatial.orientation_xyzw
    connection.execute(
        """
        INSERT OR IGNORE INTO memory_semantics (
            memory_id, lineage_id, kind, basis, source_id,
            subject, predicate, value, model_id, recipe, identity_id,
            cue_modality, valence, arousal,
            spatial_frame_id, spatial_anchor, spatial_x, spatial_y, spatial_z,
            spatial_qx, spatial_qy, spatial_qz, spatial_qw, spatial_uncertainty_m
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            memory_id,
            lineage_id,
            context.kind.value,
            context.basis.value,
            context.source_id,
            context.subject,
            context.predicate,
            context.value,
            context.model_id,
            context.recipe,
            context.identity_id,
            None if context.cue_modality is None else context.cue_modality.value,
            context.valence,
            context.arousal,
            None if spatial is None else spatial.frame_id,
            None if spatial is None else spatial.anchor.value,
            None if spatial is None else spatial.x,
            None if spatial is None else spatial.y,
            None if spatial is None else spatial.z,
            None if orientation is None else orientation[0],
            None if orientation is None else orientation[1],
            None if orientation is None else orientation[2],
            None if orientation is None else orientation[3],
            None if spatial is None else spatial.position_uncertainty_m,
        ),
    )
    if existing is None and write_evidence:
        for source_memory_id in context.evidence_ids:
            add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=context.confidence,
                recorded_at=recorded_at,
            )

    # Explicit formation clauses are inserted immediately after this semantic row because their
    # parent FK requires it to exist first. Evaluate the pending clause as one assessment here so
    # a STATE performs its lineage reconciliation in the same atomic formation transaction. The
    # later clause insertion recomputes the same projection from durable rows. A joint witness is
    # still one assessment, so it does not clear the two-group TRAIT/naming threshold.
    evidence_count = (
        1
        if not write_evidence and context.evidence_ids
        else _evidence_summary(connection, memory_id)[0]
    )
    # Decided against the lineage as it stands, before this write retires anything: an assertion
    # nobody can see must not be what displaces the standing one, and the explicit statement that
    # suppresses a guess only counts while it is still unretired.
    visible = semantic_visibility(
        connection,
        memory_id=memory_id,
        lineage_id=lineage_id,
        kind=context.kind.value,
        basis=context.basis.value,
        identity_id=context.identity_id,
        evidence_count=evidence_count,
        valid_from=valid_from,
        valid_until=valid_until,
    )

    supersedes_id = context.supersedes_id
    # A state changes, an asserted trait replaces the last one, and naming a person supersedes
    # whatever they were called before -- so the retracted name stops answering to active reads
    # instead of sitting beside the current one. Everything else accumulates evidence instead,
    # as does a claim that is not visible: it waits beside the standing one for corroboration.
    reconcile_lineage = visible and (
        context.kind is MemoryKind.STATE
        or (context.kind is MemoryKind.TRAIT and context.basis is EvidenceBasis.USER_STATEMENT)
        or (context.kind is MemoryKind.ENTITY and context.identity_id is not None)
    )
    if reconcile_lineage:
        recorded_at, supersedes_id = _retire_displaced_lineage(
            connection,
            memory_id,
            lineage_id=lineage_id,
            kind=context.kind.value,
            recorded_at=recorded_at,
            supersedes_id=supersedes_id,
            valid_from=valid_from,
            valid_until=valid_until,
            transaction_memory_ids=transaction_memory_ids,
            superseded=superseded,
        )

    latest = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM memory_versions WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if latest is None:
        raise RuntimeError("failed to allocate a memory version")
    connection.execute(
        """
        INSERT INTO memory_versions (
            memory_id, version, confidence, valid_from, valid_until,
            recorded_at, retired_at, visible, supersedes_id
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            memory_id,
            int(latest["version"]) + 1,
            context.confidence,
            optional_datetime_text(valid_from),
            optional_datetime_text(valid_until),
            datetime_text(recorded_at),
            int(visible),
            supersedes_id,
        ),
    )


def _retire_displaced_lineage(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    lineage_id: str,
    kind: str,
    recorded_at: datetime,
    supersedes_id: str | None,
    valid_from: datetime | None,
    valid_until: datetime | None,
    transaction_memory_ids: set[str] | None,
    superseded: list[tuple[str, int]] | None,
) -> tuple[datetime, str | None]:
    """Retire every standing version this claim displaces; return its time and what it replaced.

    A version whose validity only partly overlaps is carried forward over the interval that
    survives, so retiring the whole of it never silently drops the part the new claim says
    nothing about.
    """
    old_rows = connection.execute(
        """
        SELECT
            s.memory_id, v.version, v.confidence, v.valid_from, v.valid_until,
            v.recorded_at, v.visible, v.supersedes_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        WHERE s.lineage_id = ? AND s.kind = ?
          AND s.memory_id <> ? AND v.retired_at IS NULL
        ORDER BY v.recorded_at, s.memory_id, v.version
        """,
        (lineage_id, kind, memory_id),
    ).fetchall()
    for old in old_rows:
        old_from = optional_datetime_from_row(old, "valid_from")
        old_until = optional_datetime_from_row(old, "valid_until")
        old_recorded = parse_datetime(row_text(old, "recorded_at"))
        # Independent assertions in one storage batch remain conflicting. Wall-clock equality
        # alone cannot identify a batch on low-resolution or frozen clocks.
        if (
            transaction_memory_ids is not None
            and row_text(old, "memory_id") in transaction_memory_ids
        ) or not _intervals_overlap(valid_from, valid_until, old_from, old_until):
            continue
        tx_time = max(recorded_at, old_recorded + timedelta(microseconds=1))
        recorded_at = tx_time
        connection.execute(
            """
            UPDATE memory_versions
            SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (datetime_text(tx_time), row_text(old, "memory_id"), int(old["version"])),
        )
        if superseded is not None:
            superseded.append((row_text(old, "memory_id"), int(old["version"])))
        if valid_from is not None and (old_from is None or old_from < valid_from):
            _carry_memory_version(
                connection,
                old,
                valid_from=old_from,
                valid_until=valid_from,
                recorded_at=tx_time,
            )
        if valid_until is not None and (old_until is None or valid_until < old_until):
            _carry_memory_version(
                connection,
                old,
                valid_from=valid_until,
                valid_until=old_until,
                recorded_at=tx_time,
            )
        supersedes_id = supersedes_id or row_text(old, "memory_id")
    return recorded_at, supersedes_id


def _intervals_overlap(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> bool:
    return not (
        (left_until is not None and right_from is not None and left_until <= right_from)
        or (right_until is not None and left_from is not None and right_until <= left_from)
    )


def _carry_memory_version(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    valid_from: datetime | None,
    valid_until: datetime | None,
    recorded_at: datetime,
) -> None:
    memory_id = row_text(row, "memory_id")
    latest = connection.execute(
        "SELECT COALESCE(MAX(version), 0) AS version FROM memory_versions WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    if latest is None:
        raise RuntimeError("failed to allocate a memory version")
    connection.execute(
        """
        INSERT INTO memory_versions (
            memory_id, version, confidence, valid_from, valid_until,
            recorded_at, retired_at, visible, supersedes_id
        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            memory_id,
            int(latest["version"]) + 1,
            float(row["confidence"]),
            optional_datetime_text(valid_from),
            optional_datetime_text(valid_until),
            datetime_text(recorded_at),
            int(row["visible"]),
            optional_row_text(row, "supersedes_id"),
        ),
    )


def _next_semantic_transaction_time(
    connection: sqlite3.Connection,
    proposed: datetime,
    memory_ids: Iterable[str],
) -> datetime:
    unique_ids = tuple(dict.fromkeys(memory_ids))
    latest = proposed
    for offset in range(0, len(unique_ids), SQLITE_PARAMETER_BATCH):
        batch = unique_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        for table in ("memory_versions", "memory_evidence"):
            row = connection.execute(
                f"""
                SELECT MAX(recorded_at) AS recorded_at, MAX(retired_at) AS retired_at
                FROM {table} WHERE memory_id IN ({placeholders})
                """,
                batch,
            ).fetchone()
            if row is None:
                continue
            for field in ("recorded_at", "retired_at"):
                value = optional_row_text(row, field)
                if value is not None:
                    latest = max(latest, parse_datetime(value) + timedelta(microseconds=1))
    return latest


def _next_lineage_transaction_time(
    connection: sqlite3.Connection,
    proposed: datetime,
    *,
    lineage_id: str,
    kind: str,
    transaction_memory_ids: set[str] | None,
) -> datetime:
    current_batch = transaction_memory_ids or set()
    memory_ids = tuple(
        row_text(row, "memory_id")
        for row in connection.execute(
            "SELECT memory_id FROM memory_semantics WHERE lineage_id = ? AND kind = ?",
            (lineage_id, kind),
        ).fetchall()
        if row_text(row, "memory_id") not in current_batch
    )
    return _next_semantic_transaction_time(connection, proposed, memory_ids)


def memory_evidence_linked(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
) -> bool:
    """Return whether this record already cites this source in an unretired evidence row."""
    return (
        connection.execute(
            """
            SELECT 1 FROM memory_evidence
            WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
            """,
            (memory_id, source_memory_id),
        ).fetchone()
        is not None
    )


def _evidence_clause_id(source_memory_ids: Sequence[str]) -> str:
    """Stable identity for one conjunction; caller preserves declaration order separately."""
    canonical = tuple(sorted(source_memory_ids))
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"mindbridge-evidence-clause-v1:{payload}".encode()).hexdigest()


def add_evidence_clause(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_ids: Sequence[str],
    *,
    confidence: float,
    recorded_at: datetime,
) -> StoredEvidenceClauseChange | None:
    """Persist one complete support conjunction and project its members into flat evidence."""
    members = tuple(source_memory_ids)
    if not members or len(set(members)) != len(members):
        raise ValueError("evidence clause members must be non-empty and unique")
    for source_memory_id in members:
        require_identifier(source_memory_id, "source_memory_id")
    clause_id = _evidence_clause_id(members)
    row = connection.execute(
        """
        SELECT confidence, recorded_at, retired_at FROM memory_evidence_clauses
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is not None and row["retired_at"] is None and float(row["confidence"]) == confidence:
        return None
    tx_time = _next_clause_transaction_time(connection, memory_id, clause_id, recorded_at)
    previous_active = row is not None and row["retired_at"] is None
    previous_confidence = None if row is None else float(row["confidence"])
    version_row = connection.execute(
        """
        SELECT COALESCE(MAX(version), 0) AS version
        FROM memory_evidence_clause_versions
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if version_row is None:
        raise RuntimeError("failed to allocate an evidence clause version")
    version = int(version_row["version"]) + 1
    if previous_active:
        connection.execute(
            """
            UPDATE memory_evidence_clause_versions SET retired_at = ?
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (datetime_text(tx_time), memory_id, clause_id),
        )
    if row is None:
        connection.execute(
            """
            INSERT INTO memory_evidence_clauses (
                memory_id, clause_id, member_count, confidence, recorded_at, retired_at
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (memory_id, clause_id, len(members), confidence, datetime_text(tx_time)),
        )
        connection.executemany(
            """
            INSERT INTO memory_evidence_clause_members (
                memory_id, clause_id, source_memory_id, position
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (memory_id, clause_id, source_memory_id, position)
                for position, source_memory_id in enumerate(members)
            ),
        )
    else:
        connection.execute(
            """
            UPDATE memory_evidence_clauses
            SET retired_at = NULL, confidence = ?, recorded_at = ?
            WHERE memory_id = ? AND clause_id = ?
            """,
            (confidence, datetime_text(tx_time), memory_id, clause_id),
        )
    connection.execute(
        """
        INSERT INTO memory_evidence_clause_versions (
            memory_id, clause_id, version, confidence, recorded_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, NULL)
        """,
        (memory_id, clause_id, version, confidence, datetime_text(tx_time)),
    )
    _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
    refresh_evidence_projection(connection, memory_id, tx_time)
    restamp_dependent_evidence(connection, (memory_id,), tx_time)
    return StoredEvidenceClauseChange(
        memory_id=memory_id,
        clause_id=clause_id,
        previous_active=previous_active,
        previous_confidence=previous_confidence,
        applied_confidence=confidence,
        applied_recorded_at=tx_time,
        applied_version=version,
    )


def _next_clause_transaction_time(
    connection: sqlite3.Connection,
    memory_id: str,
    clause_id: str,
    proposed: datetime,
) -> datetime:
    """Return a time strictly after this clause's latest state transition."""
    tx_time = _next_semantic_transaction_time(connection, proposed, (memory_id,))
    row = connection.execute(
        """
        SELECT MAX(recorded_at) AS recorded_at, MAX(retired_at) AS retired_at
        FROM memory_evidence_clause_versions
        WHERE memory_id = ? AND clause_id = ?
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is None:
        return tx_time
    for field in ("recorded_at", "retired_at"):
        value = optional_datetime_from_row(row, field)
        if value is not None:
            tx_time = max(tx_time, value + timedelta(microseconds=1))
    return tx_time


def _rebuild_flat_evidence_from_clauses(
    connection: sqlite3.Connection,
    memory_id: str,
    *,
    changed_at: datetime,
) -> None:
    """Make legacy flat links the exact union of complete active support clauses."""
    rows = connection.execute(
        """
        SELECT m.source_memory_id, MAX(c.confidence) AS confidence
        FROM memory_evidence_clauses AS c
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
        WHERE c.memory_id = ? AND c.retired_at IS NULL
        GROUP BY m.source_memory_id
        HAVING COUNT(*) > 0
        ORDER BY MIN(m.position), m.source_memory_id
        """,
        (memory_id,),
    ).fetchall()
    wanted = {row_text(row, "source_memory_id"): float(row["confidence"]) for row in rows}
    active = {
        row_text(row, "source_memory_id")
        for row in connection.execute(
            """
            SELECT source_memory_id FROM memory_evidence
            WHERE memory_id = ? AND retired_at IS NULL
            """,
            (memory_id,),
        ).fetchall()
    }
    retired = active - set(wanted)
    if retired:
        connection.execute(
            f"""
            UPDATE memory_evidence SET retired_at = ?
            WHERE memory_id = ? AND retired_at IS NULL
              AND source_memory_id IN ({", ".join("?" for _ in retired)})
            """,
            (datetime_text(changed_at), memory_id, *sorted(retired)),
        )
    for source_memory_id, confidence in wanted.items():
        if source_memory_id not in active:
            _insert_memory_evidence(
                connection,
                memory_id,
                source_memory_id,
                confidence=confidence,
                recorded_at=changed_at,
            )
        else:
            connection.execute(
                """
                UPDATE memory_evidence SET confidence = ?
                WHERE memory_id = ? AND source_memory_id = ? AND retired_at IS NULL
                """,
                (confidence, memory_id, source_memory_id),
            )


def retire_evidence_clauses_for_source(
    connection: sqlite3.Connection,
    source_memory_id: str,
    *,
    retired_at: datetime,
) -> dict[str, datetime]:
    """Retire every conjunction containing a withdrawn source; never shorten it."""
    rows = connection.execute(
        """
        SELECT c.memory_id, c.clause_id
        FROM memory_evidence_clauses AS c
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
        WHERE m.source_memory_id = ? AND c.retired_at IS NULL
        ORDER BY c.memory_id, c.clause_id
        """,
        (source_memory_id,),
    ).fetchall()
    if not rows:
        return {}
    changed: dict[str, datetime] = {}
    for row in rows:
        memory_id = row_text(row, "memory_id")
        changed_at = _retire_active_evidence_clause(
            connection,
            memory_id,
            row_text(row, "clause_id"),
            retired_at=retired_at,
        )
        if changed_at is not None:
            changed[memory_id] = max(changed.get(memory_id, changed_at), changed_at)
    for memory_id, changed_at in changed.items():
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=changed_at)
    return changed


def active_evidence_dependent_closure(
    connection: sqlite3.Connection,
    source_memory_id: str,
) -> tuple[str, ...]:
    """Return active derived dependents reachable from one source before it is withdrawn."""
    closure: dict[str, None] = {}
    frontier = [source_memory_id]
    while frontier:
        source_id = frontier.pop()
        for row in connection.execute(
            """
            SELECT DISTINCT c.memory_id
            FROM memory_evidence_clauses AS c
            JOIN memory_evidence_clause_members AS m
              ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
            WHERE m.source_memory_id = ? AND c.retired_at IS NULL
            ORDER BY c.memory_id
            """,
            (source_id,),
        ).fetchall():
            dependent_id = row_text(row, "memory_id")
            if dependent_id == source_memory_id or dependent_id in closure:
                continue
            closure[dependent_id] = None
            frontier.append(dependent_id)
    return tuple(closure)


def deletion_cascade(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return existing roots followed by unsupported records in their reverse closure."""
    selected = tuple(
        memory_id
        for memory_id in dict.fromkeys(memory_ids)
        if connection.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        is not None
    )
    selected_set = set(selected)
    affected: dict[str, None] = {}
    for memory_id in selected:
        for dependent_id in active_evidence_dependent_closure(connection, memory_id):
            if dependent_id not in selected_set:
                affected.setdefault(dependent_id, None)
    grounded = grounded_affected_memory_ids(
        connection,
        tuple(affected),
        excluding=selected,
    )
    return (
        *selected,
        *(memory_id for memory_id in affected if memory_id not in grounded),
    )


def grounded_affected_memory_ids(  # noqa: C901 - bounded support fixed point
    connection: sqlite3.Connection,
    affected: Sequence[str],
    *,
    excluding: Sequence[str],
) -> frozenset[str]:
    """Solve support only inside one withdrawal's dependent subgraph.

    Existing visible support outside the subgraph is unchanged and therefore acts as a seed.
    Inside it, observations and host assertions seed themselves; every other record needs one
    complete clause whose members are already grounded. A mutual-support cycle cannot seed itself.
    """
    affected_set = set(affected)
    excluded = set(excluding)
    grounded = {
        memory_id
        for memory_id in affected
        if memory_id not in excluded and _is_evidence_root(connection, memory_id)
    }
    clauses: dict[str, dict[str, set[str]]] = {}
    for memory_id in affected:
        if memory_id in excluded:
            continue
        for row in connection.execute(
            """
            SELECT c.clause_id, m.source_memory_id
            FROM memory_evidence_clauses AS c
            JOIN memory_evidence_clause_members AS m
              ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
            WHERE c.memory_id = ? AND c.retired_at IS NULL
            ORDER BY c.clause_id, m.position
            """,
            (memory_id,),
        ).fetchall():
            clauses.setdefault(memory_id, {}).setdefault(row_text(row, "clause_id"), set()).add(
                row_text(row, "source_memory_id")
            )
    outside_grounded: dict[str, bool] = {}

    def member_is_grounded(source_id: str) -> bool:
        if source_id in excluded:
            return False
        if source_id in affected_set:
            return source_id in grounded
        if source_id not in outside_grounded:
            # A record outside the withdrawal's subgraph keeps whatever support it had: a hidden
            # trait still below its visibility threshold is supported, not unsupported, so it
            # must keep grounding the alternatives that cite it.
            outside_grounded[source_id] = _is_evidence_root(connection, source_id) or (
                connection.execute(
                    """
                    SELECT 1 FROM memory_versions AS v
                    WHERE v.memory_id = ? AND v.retired_at IS NULL
                      AND (
                        v.visible = 1
                        OR EXISTS (
                            SELECT 1 FROM memory_evidence_clauses AS c
                            WHERE c.memory_id = v.memory_id AND c.retired_at IS NULL
                        )
                      )
                    """,
                    (source_id,),
                ).fetchone()
                is not None
            )
        return outside_grounded[source_id]

    moved = True
    while moved:
        moved = False
        for memory_id, alternatives in clauses.items():
            if memory_id in grounded:
                continue
            if any(
                members and all(member_is_grounded(source_id) for source_id in members)
                for members in alternatives.values()
            ):
                grounded.add(memory_id)
                moved = True
    return frozenset(grounded)


def _is_evidence_root(connection: sqlite3.Connection, memory_id: str) -> bool:
    """Return whether a record stands without another memory's support."""
    row = connection.execute(
        """
        SELECT s.kind, s.basis
        FROM memory_records AS r
        LEFT JOIN memory_semantics AS s ON s.memory_id = r.memory_id
        WHERE r.memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if row is None:
        return False
    return (
        row["kind"] is None
        or row_text(row, "kind") == MemoryKind.OBSERVATION.value
        or row_text(row, "basis")
        in {EvidenceBasis.USER_STATEMENT.value, EvidenceBasis.RESPONSE_FEEDBACK.value}
    )


def _retire_active_evidence_clause(
    connection: sqlite3.Connection,
    memory_id: str,
    clause_id: str,
    *,
    retired_at: datetime,
) -> datetime | None:
    """Close one current clause interval and mirror that state on the clause projection."""
    row = connection.execute(
        """
        SELECT 1 FROM memory_evidence_clauses
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (memory_id, clause_id),
    ).fetchone()
    if row is None:
        return None
    tx_time = _next_clause_transaction_time(connection, memory_id, clause_id, retired_at)
    connection.execute(
        """
        UPDATE memory_evidence_clause_versions SET retired_at = ?
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (datetime_text(tx_time), memory_id, clause_id),
    )
    connection.execute(
        """
        UPDATE memory_evidence_clauses SET retired_at = ?
        WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
        """,
        (datetime_text(tx_time), memory_id, clause_id),
    )
    return tx_time


# Independence is counted per capture: a raw observation resolves to the `source_id` its
# `ObservationContext` carried, falling back to the observation record itself when the caller
# supplied none. A derived source (an affect cue, say) inherits the group its own evidence
# already resolves to, so two observations of one capture -- and every cue formed from them --
# stay one group and cannot corroborate each other into a visible trait. A source resolving to
# several groups keeps its own identity: nothing that traces back to a single capture can then
# reach two groups. `MIN = MAX` says "exactly one distinct group" without the temp B-tree a
# `COUNT(DISTINCT ...)` would build, and holds because `source_group_id` is `TEXT NOT NULL`.
SOURCE_GROUP_QUERY = """
    SELECT COALESCE(
        (
            SELECT MIN(d.source_group_id)
            FROM memory_evidence AS d
            WHERE d.memory_id = r.memory_id AND d.retired_at IS NULL
            HAVING MIN(d.source_group_id) = MAX(d.source_group_id)
        ),
        s.source_id,
        r.memory_id
    ) AS source_group_id
    FROM memory_records AS r
    LEFT JOIN memory_semantics AS s
      ON s.memory_id = r.memory_id AND s.kind = 'observation'
    WHERE r.memory_id = ?
"""


def restamp_dependent_evidence(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    changed_at: datetime,
) -> None:
    """Re-resolve the inherited group on every row citing a record whose own evidence moved.

    `source_group_id` is copied from the cited source when the citing row is written, so
    reinforcing, rolling back, or cascading a delete through that source later leaves the rows
    that cite it stamped with a group the source no longer resolves to -- and a dependent trait
    keeps counting corroboration that is gone. Restamp the whole reachable citation subgraph in
    the same transaction and reproject what changed.
    """
    closure: dict[str, None] = {}
    frontier = list(dict.fromkeys(memory_ids))
    while frontier:
        source_memory_id = frontier.pop()
        if source_memory_id in closure:
            continue
        closure[source_memory_id] = None
        frontier.extend(
            row_text(row, "memory_id")
            for row in connection.execute(
                """
                SELECT memory_id FROM memory_evidence
                WHERE source_memory_id = ? AND retired_at IS NULL
                """,
                (source_memory_id,),
            ).fetchall()
        )
    # A sweep in arbitrary order can restamp a record before one of its own sources settles, so
    # sweep until nothing moves. The citation graph is acyclic, so each sweep settles at least
    # one more record and the bound is only there to stop a corrupted cycle from spinning.
    for _sweep in range(len(closure)):
        moved = False
        for source_memory_id in closure:
            row = connection.execute(SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
            if row is None:
                continue
            group = row_text(row, "source_group_id")
            dependent_ids = tuple(
                dict.fromkeys(
                    row_text(dependent, "memory_id")
                    for dependent in connection.execute(
                        """
                        SELECT memory_id FROM memory_evidence
                        WHERE source_memory_id = ?
                          AND retired_at IS NULL AND source_group_id <> ?
                        """,
                        (source_memory_id, group),
                    ).fetchall()
                )
            )
            if not dependent_ids:
                continue
            moved = True
            connection.execute(
                """
                UPDATE memory_evidence SET source_group_id = ?
                WHERE source_memory_id = ? AND retired_at IS NULL
                """,
                (group, source_memory_id),
            )
            for dependent_id in dependent_ids:
                refresh_evidence_projection(
                    connection,
                    dependent_id,
                    _next_semantic_transaction_time(connection, changed_at, (dependent_id,)),
                )
        if not moved:
            break


def _insert_memory_evidence(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
    *,
    confidence: float,
    recorded_at: datetime,
) -> bool:
    if memory_evidence_linked(connection, memory_id, source_memory_id):
        return False
    source = connection.execute(SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
    if source is None:
        raise sqlite3.IntegrityError("evidence source memory does not exist")
    position_row = connection.execute(
        """
        SELECT COALESCE(MAX(position) + 1, 0) AS position
        FROM memory_evidence
        WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    if position_row is None:
        raise RuntimeError("failed to allocate an evidence position")
    connection.execute(
        """
        INSERT INTO memory_evidence (
            memory_id, source_memory_id, source_group_id, position,
            confidence, recorded_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, ?, NULL)
        """,
        (
            memory_id,
            source_memory_id,
            row_text(source, "source_group_id"),
            int(position_row["position"]),
            confidence,
            datetime_text(recorded_at),
        ),
    )
    # This record's own active evidence just changed, so whatever cites it may now inherit a
    # different group.
    restamp_dependent_evidence(connection, (memory_id,), recorded_at)
    return True


def add_memory_evidence(
    connection: sqlite3.Connection,
    memory_id: str,
    source_memory_id: str,
    *,
    confidence: float,
    recorded_at: datetime,
) -> bool:
    previous_count, previous_confidence = _evidence_summary(connection, memory_id)
    tx_time = _next_semantic_transaction_time(connection, recorded_at, (memory_id,))
    if not _insert_memory_evidence(
        connection,
        memory_id,
        source_memory_id,
        confidence=confidence,
        recorded_at=tx_time,
    ):
        return False
    evidence_count, combined = _evidence_summary(connection, memory_id)
    if evidence_count == previous_count and combined == previous_confidence:
        return True
    refresh_evidence_projection(connection, memory_id, tx_time)
    return True


def require_active_memories(connection: sqlite3.Connection, memory_ids: Sequence[str]) -> None:
    """Fail the transaction unless every named memory still exists and is still un-forgotten."""
    for memory_id in dict.fromkeys(memory_ids):
        require_identifier(memory_id, "memory_id")
        row = connection.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ? AND forgotten_at IS NULL",
            (memory_id,),
        ).fetchone()
        if row is None:
            raise StaleOperationError(f"{memory_id} is no longer active")


def version_retired(connection: sqlite3.Connection, memory_id: str) -> bool:
    """Return whether a typed record exists whose every version has been retired."""
    row = connection.execute(
        """
        SELECT COUNT(*) AS total, COUNT(retired_at) AS retired
        FROM memory_versions WHERE memory_id = ?
        """,
        (memory_id,),
    ).fetchone()
    return row is not None and bool(int(row["total"])) and int(row["total"]) == int(row["retired"])


def require_unretired_memories(connection: sqlite3.Connection, memory_ids: Sequence[str]) -> None:
    """Fail the transaction unless every named memory's current version still stands.

    Deliberately narrower than `require_active_memories`: it says nothing about `forgotten_at`, and it
    never looks at `visible`, because a hidden inferred `TRAIT` is a legitimate `REINFORCE`
    target. A record with no typed version at all -- a plain observation -- passes.
    """
    for memory_id in dict.fromkeys(memory_ids):
        require_identifier(memory_id, "memory_id")
        if version_retired(connection, memory_id):
            raise StaleOperationError(f"{memory_id} has no current version")


def require_every(changed: Sequence[str], requested: Sequence[str], effect: str) -> None:
    """Fail the transaction unless every requested target actually took the effect."""
    if set(changed) != set(requested):
        missing = sorted(set(requested) - set(changed))
        raise StaleOperationError(f"could not {effect} {', '.join(missing)}")


def set_forgotten(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    forgotten_at: datetime | None,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id in dict.fromkeys(memory_ids):
        cursor = connection.execute(
            """
            UPDATE memory_records
            SET forgotten_at = ?
            WHERE memory_id = ? AND (forgotten_at IS NULL) != (? IS NULL)
            """,
            (
                None if forgotten_at is None else datetime_text(forgotten_at),
                memory_id,
                None if forgotten_at is None else 1,
            ),
        )
        if cursor.rowcount:
            changed.append(memory_id)
    return tuple(changed)


def refresh_multi_source_projections(
    connection: sqlite3.Connection,
    memories: Sequence[StoredMemory],
    *,
    changed_at: datetime,
) -> None:
    """Project confidence and visibility for records written with several sources at once.

    A record inserted with all of its evidence rows already present never reaches the per-row
    projection in `add_memory_evidence`. Formation cites one source and needs nothing here;
    consolidation needs it for the noisy-OR confidence and trait visibility.
    """
    for memory in memories:
        context = memory.context
        if context is not None and len(context.evidence_ids) > 1:
            refresh_evidence_projection(connection, memory.memory_id, changed_at)


def asserted_confidence(connection: sqlite3.Connection, memory_id: str) -> float:
    """Return the confidence one source lends this assertion, not the noisy-OR projection.

    Version 1 never changes, so reinforcing the same record twice combines equal independent
    support instead of compounding whatever the last projection happened to be.
    """
    row = connection.execute(
        "SELECT confidence FROM memory_versions WHERE memory_id = ? ORDER BY version LIMIT 1",
        (memory_id,),
    ).fetchone()
    return 1.0 if row is None else float(row["confidence"])


def retire_memory_versions(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id in dict.fromkeys(memory_ids):
        tx_time = _next_semantic_transaction_time(connection, retired_at, (memory_id,))
        cursor = connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND retired_at IS NULL
            """,
            (datetime_text(tx_time), memory_id),
        )
        if cursor.rowcount:
            changed.append(memory_id)
    return tuple(changed)


def restore_memory_versions(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str | tuple[str, int]],
    *,
    recorded_at: datetime,
) -> None:
    """Carry a retired version forward again, by memory ID or by exact `(id, version)` pair.

    A `CORRECT` retires whatever version was current, so naming the record is enough. A lineage
    supersession may also have split the record's remaining validity into carried versions, so
    the operation row names the exact version it retired and that one is restored.
    """
    for entry in dict.fromkeys(memory_ids):
        memory_id, version = (entry, None) if isinstance(entry, str) else entry
        row = connection.execute(
            f"""
            SELECT memory_id, version, confidence, valid_from, valid_until,
                   recorded_at, visible, supersedes_id
            FROM memory_versions
            WHERE memory_id = ?{"" if version is None else " AND version = ?"}
            ORDER BY version DESC LIMIT 1
            """,
            (memory_id,) if version is None else (memory_id, version),
        ).fetchone()
        if row is None:
            continue
        _carry_memory_version(
            connection,
            row,
            valid_from=optional_datetime_from_row(row, "valid_from"),
            valid_until=optional_datetime_from_row(row, "valid_until"),
            recorded_at=_next_semantic_transaction_time(connection, recorded_at, (memory_id,)),
        )


def current_evidence_ids(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[str, ...]:
    """Return the memories currently cited as evidence for one derived record."""
    return tuple(
        row_text(row, "source_memory_id")
        for row in connection.execute(
            """
            SELECT source_memory_id FROM memory_evidence
            WHERE memory_id = ? AND retired_at IS NULL
            ORDER BY position
            """,
            (memory_id,),
        ).fetchall()
    )


def retire_memory_evidence(
    connection: sqlite3.Connection,
    pairs: Sequence[tuple[str, str]],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    changed: list[str] = []
    for memory_id, source_memory_id in pairs:
        # Control-plane reinforcement has always been a singleton source link. Retire that
        # exact legacy clause, never every alternative that happens to mention the same source.
        tx_time = _retire_active_evidence_clause(
            connection,
            memory_id,
            _evidence_clause_id((source_memory_id,)),
            retired_at=retired_at,
        )
        if tx_time is None:
            continue
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
        changed.append(memory_id)
    for memory_id in dict.fromkeys(changed):
        refresh_evidence_projection(
            connection,
            memory_id,
            _next_semantic_transaction_time(connection, retired_at, (memory_id,)),
        )
    restamp_dependent_evidence(connection, tuple(dict.fromkeys(changed)), retired_at)
    return tuple(dict.fromkeys(changed))


def retire_evidence_clauses(
    connection: sqlite3.Connection,
    clauses: Sequence[tuple[str, str]],
    *,
    retired_at: datetime,
) -> tuple[str, ...]:
    """Retire precisely the clauses an operation introduced, preserving alternatives."""
    changed: list[str] = []
    for memory_id, clause_id in clauses:
        tx_time = _retire_active_evidence_clause(
            connection,
            memory_id,
            clause_id,
            retired_at=retired_at,
        )
        if tx_time is None:
            continue
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=tx_time)
        changed.append(memory_id)
    for memory_id in dict.fromkeys(changed):
        refresh_evidence_projection(
            connection,
            memory_id,
            _next_semantic_transaction_time(connection, retired_at, (memory_id,)),
        )
    restamp_dependent_evidence(connection, tuple(dict.fromkeys(changed)), retired_at)
    return tuple(dict.fromkeys(changed))


def evidence_clause_changes_are_current(
    connection: sqlite3.Connection,
    changes: Sequence[StoredEvidenceClauseChange],
) -> bool:
    """Require the operation's version or an explicit inverse chain restoring that version.

    Confidence equality is insufficient: an unrelated A -> B -> A sequence is still a later
    mutation. An inverse version records the exact predecessor it restored, so rolling back the
    newer operation makes the older one reversible again without weakening the ABA guard.
    """
    for change in changes:
        # Deleting the output this operation created cascaded its clause and every version of
        # it. `reverse_evidence_clause_changes` treats that deletion as the complete inverse;
        # the currency guard has to agree, or the operation can never be rolled back.
        if (
            connection.execute(
                """
                SELECT 1 FROM memory_evidence_clauses
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            is None
        ):
            continue
        applied = connection.execute(
            """
            SELECT confidence, recorded_at
            FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND version = ?
            """,
            (change.memory_id, change.clause_id, change.applied_version),
        ).fetchone()
        if (
            applied is None
            or float(applied["confidence"]) != change.applied_confidence
            or parse_datetime(row_text(applied, "recorded_at")) != change.applied_recorded_at
        ):
            return False
        current = connection.execute(
            """
            SELECT version, confidence, restores_version
            FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (change.memory_id, change.clause_id),
        ).fetchone()
        if current is None or float(current["confidence"]) != change.applied_confidence:
            return False
        version = int(current["version"])
        visited: set[int] = set()
        while version != change.applied_version:
            if version in visited:
                return False
            visited.add(version)
            row = connection.execute(
                """
                SELECT restores_version FROM memory_evidence_clause_versions
                WHERE memory_id = ? AND clause_id = ? AND version = ?
                """,
                (change.memory_id, change.clause_id, version),
            ).fetchone()
            if row is None or row["restores_version"] is None:
                return False
            version = int(row["restores_version"])
    return True


def reverse_evidence_clause_changes(
    connection: sqlite3.Connection,
    changes: Sequence[StoredEvidenceClauseChange],
    *,
    reversed_at: datetime,
) -> tuple[str, ...]:
    """Append the inverse clause state at rollback time without rewriting earlier intervals."""
    changed: dict[str, datetime] = {}
    for change in changes:
        # Deleting an output created by this operation cascades its clauses first. That deletion
        # is already the complete inverse for the output and leaves nothing here to restate.
        if (
            connection.execute(
                """
                SELECT 1 FROM memory_evidence_clauses
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            is None
        ):
            continue
        tx_time = _next_clause_transaction_time(
            connection,
            change.memory_id,
            change.clause_id,
            reversed_at,
        )
        current = connection.execute(
            """
            SELECT version FROM memory_evidence_clause_versions
            WHERE memory_id = ? AND clause_id = ? AND retired_at IS NULL
            """,
            (change.memory_id, change.clause_id),
        ).fetchone()
        if current is None:
            raise RuntimeError("a reversible clause change requires an active current version")
        connection.execute(
            """
            UPDATE memory_evidence_clause_versions SET retired_at = ?
            WHERE memory_id = ? AND clause_id = ? AND version = ? AND retired_at IS NULL
            """,
            (
                datetime_text(tx_time),
                change.memory_id,
                change.clause_id,
                int(current["version"]),
            ),
        )
        if change.previous_active:
            if change.previous_confidence is None:
                raise RuntimeError("an active previous clause requires a confidence")
            latest = connection.execute(
                """
                SELECT MAX(version) AS version FROM memory_evidence_clause_versions
                WHERE memory_id = ? AND clause_id = ?
                """,
                (change.memory_id, change.clause_id),
            ).fetchone()
            if latest is None or latest["version"] is None:
                raise RuntimeError("failed to allocate an inverse clause version")
            version = int(latest["version"]) + 1
            connection.execute(
                """
                INSERT INTO memory_evidence_clause_versions (
                    memory_id, clause_id, version, confidence, recorded_at, retired_at,
                    restores_version
                ) VALUES (?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    change.memory_id,
                    change.clause_id,
                    version,
                    change.previous_confidence,
                    datetime_text(tx_time),
                    change.applied_version - 1,
                ),
            )
            connection.execute(
                """
                UPDATE memory_evidence_clauses
                SET confidence = ?, recorded_at = ?, retired_at = NULL
                WHERE memory_id = ? AND clause_id = ?
                """,
                (
                    change.previous_confidence,
                    datetime_text(tx_time),
                    change.memory_id,
                    change.clause_id,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE memory_evidence_clauses SET retired_at = ?
                WHERE memory_id = ? AND clause_id = ?
                """,
                (datetime_text(tx_time), change.memory_id, change.clause_id),
            )
        changed[change.memory_id] = max(changed.get(change.memory_id, tx_time), tx_time)
    for memory_id, changed_at in changed.items():
        _rebuild_flat_evidence_from_clauses(connection, memory_id, changed_at=changed_at)
        refresh_evidence_projection(connection, memory_id, changed_at)
    restamp_dependent_evidence(connection, tuple(changed), reversed_at)
    return tuple(changed)


def validate_formation_links(
    sources: Sequence[str],
    forget_ids: Sequence[str],
    evidence: Sequence[tuple[str, str, float]],
) -> None:
    """Check every identifier and confidence a formation commit is about to write."""
    for source_memory_id in sources:
        require_identifier(source_memory_id, "source_memory_id")
    for memory_id in forget_ids:
        require_identifier(memory_id, "forget_id")
    for memory_id, source_memory_id, confidence in evidence:
        require_identifier(memory_id, "memory_id")
        require_identifier(source_memory_id, "source_memory_id")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")


def _evidence_summary(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[int, float]:
    # Historic singleton clauses retain the capture-group noisy-OR projection. A multi-member
    # clause is one model assessment, however many sources it names: treat all such assessments
    # as one conservative alternative rather than inventing independent votes from its members.
    rows = connection.execute(
        """
        SELECT MAX(c.confidence) AS confidence
        FROM memory_evidence AS e
        JOIN memory_evidence_clauses AS c
          ON c.memory_id = e.memory_id AND c.retired_at IS NULL AND c.member_count = 1
        JOIN memory_evidence_clause_members AS m
          ON m.memory_id = c.memory_id AND m.clause_id = c.clause_id
             AND m.source_memory_id = e.source_memory_id
        WHERE e.memory_id = ? AND e.retired_at IS NULL
        GROUP BY e.source_group_id
        """,
        (memory_id,),
    ).fetchall()
    combined = 0.0
    for row in rows:
        combined = 1.0 - (1.0 - combined) * (1.0 - float(row["confidence"]))
    joint = connection.execute(
        """
        SELECT MAX(confidence) AS confidence
        FROM memory_evidence_clauses
        WHERE memory_id = ? AND retired_at IS NULL AND member_count > 1
        """,
        (memory_id,),
    ).fetchone()
    joint_confidence = (
        0.0 if joint is None or joint["confidence"] is None else float(joint["confidence"])
    )
    return max(len(rows), 1 if joint_confidence else 0), max(combined, joint_confidence)


def semantic_visibility(
    connection: sqlite3.Connection,
    *,
    memory_id: str,
    lineage_id: str,
    kind: str,
    basis: str,
    identity_id: str | None,
    evidence_count: int,
    valid_from: datetime | None,
    valid_until: datetime | None,
    explicit_intervals: Sequence[tuple[datetime | None, datetime | None]] | None = None,
) -> bool:
    if basis in {
        EvidenceBasis.USER_STATEMENT.value,
        EvidenceBasis.RESPONSE_FEEDBACK.value,
    }:
        return True
    if kind != MemoryKind.OBSERVATION.value and evidence_count == 0:
        return False
    # A naming assertion bound to an identity is corroborated like an inferred trait: what an
    # agent claims somebody is called stays hidden, and so stays out of the projected
    # `identities.name`, until two independent evidence groups support it.
    if kind != MemoryKind.TRAIT.value and not (
        kind == MemoryKind.ENTITY.value and identity_id is not None
    ):
        return True
    if evidence_count < 2:
        return False
    if explicit_intervals is None:
        rows = connection.execute(
            """
            SELECT v.valid_from, v.valid_until
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            WHERE s.lineage_id = ? AND s.kind = ? AND s.basis = ?
              AND s.memory_id <> ? AND v.retired_at IS NULL
            """,
            (
                lineage_id,
                kind,
                EvidenceBasis.USER_STATEMENT.value,
                memory_id,
            ),
        ).fetchall()
        explicit_intervals = tuple(
            (
                optional_datetime_from_row(row, "valid_from"),
                optional_datetime_from_row(row, "valid_until"),
            )
            for row in rows
        )
    return not any(
        _intervals_overlap(valid_from, valid_until, explicit_from, explicit_until)
        for explicit_from, explicit_until in explicit_intervals
    )


def _refresh_inherited_columns(connection: sqlite3.Connection, memory_id: str) -> None:
    """Re-derive the place and the metadata one derived record inherits from its live evidence.

    The kernel clears both columns when two sources disagree, and evidence is not permanent: a
    deleted source or a rolled-back consolidation retires it. Recomputing them here, with the
    kernel's rule -- every remaining source agrees, or nothing -- keeps a record whose survivors
    all stand in one room from staying unscoped and out of every place-filtered read.

    A naming assertion -- a bound `ENTITY` -- inherits neither column from the clips that show
    the person, so it is left alone: who somebody is does not stop being true in another room.
    """
    rows = connection.execute(
        """
        SELECT r.place_id, r.metadata_json
        FROM memory_evidence AS e
        JOIN memory_records AS r ON r.memory_id = e.source_memory_id
        WHERE e.memory_id = ? AND e.retired_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM memory_semantics AS s
              WHERE s.memory_id = e.memory_id AND s.kind = ? AND s.identity_id IS NOT NULL
          )
        """,
        (memory_id, MemoryKind.ENTITY.value),
    ).fetchall()
    if not rows:
        # Nothing supports the record any more, or nothing may reach it. The caller removes an
        # unsupported record; clearing its columns first would only alter the row it deletes.
        return
    places = {optional_row_text(row, "place_id") for row in rows}
    tags = {canonical_object_json(row_text(row, "metadata_json")) for row in rows}
    place_id = places.pop() if len(places) == 1 else None
    current = connection.execute(
        "SELECT place_id FROM memory_records WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()
    connection.execute(
        "UPDATE memory_records SET place_id = ?, metadata_json = ? WHERE memory_id = ?",
        (place_id, tags.pop() if len(tags) == 1 else "{}", memory_id),
    )
    # The search index carries `place_id` as a filter field, so a record that regains or loses
    # its inherited place has to be reprojected -- this UPDATE bypasses `write_memory`, which
    # is where every other place change is noticed.
    if current is not None and optional_row_text(current, "place_id") != place_id:
        queue_memory_embeddings(connection, memory_id, exclude=set())


def refresh_evidence_projection(
    connection: sqlite3.Connection,
    memory_id: str,
    changed_at: datetime,
) -> None:
    _refresh_inherited_columns(connection, memory_id)
    evidence_count, confidence = _evidence_summary(connection, memory_id)
    rows = connection.execute(
        """
        SELECT v.*, s.lineage_id, s.kind, s.basis, s.identity_id
        FROM memory_versions AS v
        JOIN memory_semantics AS s ON s.memory_id = v.memory_id
        WHERE v.memory_id = ? AND v.retired_at IS NULL
        ORDER BY v.version
        """,
        (memory_id,),
    ).fetchall()
    for row in rows:
        recorded_at = parse_datetime(row_text(row, "recorded_at"))
        tx_time = max(changed_at, recorded_at + timedelta(microseconds=1))
        connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (datetime_text(tx_time), memory_id, int(row["version"])),
        )
        visible = semantic_visibility(
            connection,
            memory_id=memory_id,
            lineage_id=row_text(row, "lineage_id"),
            kind=row_text(row, "kind"),
            basis=row_text(row, "basis"),
            identity_id=optional_row_text(row, "identity_id"),
            evidence_count=evidence_count,
            valid_from=optional_datetime_from_row(row, "valid_from"),
            valid_until=optional_datetime_from_row(row, "valid_until"),
        )
        _carry_memory_version(
            connection,
            row,
            valid_from=optional_datetime_from_row(row, "valid_from"),
            valid_until=optional_datetime_from_row(row, "valid_until"),
            recorded_at=tx_time,
        )
        latest = connection.execute(
            """
            SELECT MAX(version) AS version FROM memory_versions WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        if latest is None:
            raise RuntimeError("failed to update evidence projection")
        connection.execute(
            """
            UPDATE memory_versions SET confidence = ?, visible = ?
            WHERE memory_id = ? AND version = ?
            """,
            (confidence, int(visible), memory_id, int(latest["version"])),
        )


def rebuild_reconciled_lineage(  # noqa: C901 - replay order is the state contract
    connection: sqlite3.Connection,
    lineage_id: str,
    kind: str,
    *,
    changed_at: datetime,
) -> None:
    assertions = connection.execute(
        """
        SELECT v.*, s.kind, s.basis, s.identity_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id AND v.version = 1
        WHERE s.lineage_id = ? AND s.kind = ?
          AND EXISTS (
              SELECT 1 FROM memory_evidence AS e
              WHERE e.memory_id = s.memory_id AND e.retired_at IS NULL
          )
        ORDER BY v.recorded_at, s.memory_id
        """,
        (lineage_id, kind),
    ).fetchall()
    current = connection.execute(
        """
        SELECT v.*
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        WHERE s.lineage_id = ? AND s.kind = ? AND v.retired_at IS NULL
        ORDER BY s.memory_id, v.version
        """,
        (lineage_id, kind),
    ).fetchall()
    tx_time = changed_at
    for row in current:
        tx_time = max(
            tx_time,
            parse_datetime(row_text(row, "recorded_at")) + timedelta(microseconds=1),
        )
    for row in current:
        connection.execute(
            """
            UPDATE memory_versions SET retired_at = ?
            WHERE memory_id = ? AND version = ? AND retired_at IS NULL
            """,
            (
                datetime_text(tx_time),
                row_text(row, "memory_id"),
                int(row["version"]),
            ),
        )

    # ponytail: replay is O(assertions²); add a compacted assertion ledger only if long-lived
    # lineages make deletion latency measurable.
    segments: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
    offset = 0
    while offset < len(assertions):
        recorded_at = row_text(assertions[offset], "recorded_at")
        group: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
        while (
            offset < len(assertions) and row_text(assertions[offset], "recorded_at") == recorded_at
        ):
            row = assertions[offset]
            group.append(
                (
                    row,
                    optional_datetime_from_row(row, "valid_from"),
                    optional_datetime_from_row(row, "valid_until"),
                )
            )
            offset += 1
        for cut_owner, cut_from, cut_until in group:
            if (
                kind == MemoryKind.TRAIT.value
                and row_text(cut_owner, "basis") != EvidenceBasis.USER_STATEMENT.value
            ):
                continue
            remaining: list[tuple[sqlite3.Row, datetime | None, datetime | None]] = []
            for owner, valid_from, valid_until in segments:
                if not _intervals_overlap(valid_from, valid_until, cut_from, cut_until):
                    remaining.append((owner, valid_from, valid_until))
                    continue
                if cut_from is not None and (valid_from is None or valid_from < cut_from):
                    remaining.append((owner, valid_from, cut_from))
                if cut_until is not None and (valid_until is None or cut_until < valid_until):
                    remaining.append((owner, cut_until, valid_until))
            segments = remaining
        segments.extend(group)

    explicit_intervals = tuple(
        (valid_from, valid_until)
        for row, valid_from, valid_until in segments
        if row_text(row, "basis") == EvidenceBasis.USER_STATEMENT.value
    )
    summaries: dict[str, tuple[int, float]] = {}
    for row, valid_from, valid_until in segments:
        memory_id = row_text(row, "memory_id")
        evidence_count, confidence = summaries.setdefault(
            memory_id,
            _evidence_summary(connection, memory_id),
        )
        visible = semantic_visibility(
            connection,
            memory_id=memory_id,
            lineage_id=lineage_id,
            kind=kind,
            basis=row_text(row, "basis"),
            identity_id=optional_row_text(row, "identity_id"),
            evidence_count=evidence_count,
            valid_from=valid_from,
            valid_until=valid_until,
            explicit_intervals=explicit_intervals,
        )
        _carry_memory_version(
            connection,
            row,
            valid_from=valid_from,
            valid_until=valid_until,
            recorded_at=tx_time,
        )
        latest = connection.execute(
            "SELECT MAX(version) AS version FROM memory_versions WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if latest is None:
            raise RuntimeError("failed to rebuild a reconciled memory lineage")
        connection.execute(
            """
            UPDATE memory_versions SET confidence = ?, visible = ?
            WHERE memory_id = ? AND version = ?
            """,
            (confidence, int(visible), memory_id, int(latest["version"])),
        )
