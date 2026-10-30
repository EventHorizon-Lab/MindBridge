"""Authoritative scoped selection of memory rows: bitemporal, spatial, and identity axes."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Collection, Sequence
from datetime import datetime, timezone

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    optional_datetime_from_row,
    optional_row_text,
    parse_datetime,
    row_text,
)
from mindbridge.infrastructure.local.store._identity import resolve_identity_id
from mindbridge.infrastructure.local.store._membership import IDENTITY_SCOPE_CLAUSE
from mindbridge.infrastructure.local.store.rows import require_aware
from mindbridge.types import (
    EvidenceBasis,
    MemoryContext,
    MemoryKind,
    Modality,
    SpatialAnchor,
    SpatialContext,
)


def require_scope_axes(
    *,
    valid_at: datetime | None,
    known_at: datetime | None,
    near: SpatialContext | None,
    radius_m: float | None,
) -> None:
    """Validate the scope axes a hydration reads under, for callers that may not reach one."""
    if (near is None) != (radius_m is None):
        raise ValueError("near and radius_m must be supplied together")
    if near is not None and not isinstance(near, SpatialContext):
        raise ValueError("near must be a SpatialContext")
    if radius_m is not None and (
        isinstance(radius_m, bool)
        or not isinstance(radius_m, int | float)
        or not math.isfinite(float(radius_m))
        or radius_m < 0
    ):
        raise ValueError("radius_m must be a non-negative finite number")
    if valid_at is not None:
        require_aware(valid_at, "valid_at")
    if known_at is not None:
        require_aware(known_at, "known_at")


def memory_in_scope(
    row: sqlite3.Row,
    *,
    active_only: bool,
    known_at: datetime | None,
    near: SpatialContext | None,
    semantic_ids: frozenset[str],
    scoped_ids: Collection[str],
) -> bool:
    """Decide whether one `memory_records` row survives the requested scope.

    `row` needs only `memory_id`, `created_at`, and `forgotten_at`, so both the scoped hydration
    and the scoped count reach this one predicate.
    """
    if not active_only:
        return True
    # Forgetting is a column, not a deletion: the row stays readable by ID and drops out of every
    # active slate, whatever its validity interval or pose says.
    if row["forgotten_at"] is not None:
        return False
    memory_id = row_text(row, "memory_id")
    if memory_id in scoped_ids:
        return True
    # A record with no `memory_semantics` row declares no validity interval, so it is valid at
    # every `valid_at` exactly like a version row whose `valid_from` and `valid_until` are NULL;
    # only recorded time can exclude it, and there `created_at` is the honest bound. `near` is
    # separate and spatial: a record with no pose is not at any location, so it drops just as a
    # semantic row without a pose does in the selection pass.
    return (
        near is None
        and memory_id not in semantic_ids
        and (known_at is None or parse_datetime(row_text(row, "created_at")) <= known_at)
    )


def version_time_clause(
    *,
    valid_at: datetime | None,
    known_at: datetime | None,
    active_only: bool,
    alias: str = "v",
) -> tuple[str, tuple[object, ...]]:
    """Temporal eligibility of an internal version alias, shared by selection and recall."""
    if not active_only:
        return (
            ("", ())
            if known_at is None
            else (f"AND {alias}.recorded_at <= ?", (datetime_text(known_at),))
        )
    known = datetime_text(known_at or datetime.now(timezone.utc))
    valid = datetime_text(valid_at or datetime.now(timezone.utc))
    return (
        f"AND {alias}.recorded_at <= ? AND ({alias}.retired_at IS NULL OR {alias}.retired_at > ?) "
        f"AND ({alias}.valid_from IS NULL OR {alias}.valid_from <= ?) "
        f"AND ({alias}.valid_until IS NULL OR {alias}.valid_until > ?) AND {alias}.visible = 1",
        (known, known, valid, valid),
    )


def memory_time_clause(
    *, valid_at: datetime | None, known_at: datetime | None
) -> tuple[str, tuple[object, ...]]:
    """Select active records before LIMIT, including untyped records known at the cutoff."""
    version_clause, parameters = version_time_clause(
        valid_at=valid_at, known_at=known_at, active_only=True
    )
    raw_clause = "" if known_at is None else "AND memory_records.created_at <= ?"
    raw_parameters: tuple[object, ...] = () if known_at is None else (datetime_text(known_at),)
    return (
        "AND ((NOT EXISTS (SELECT 1 FROM memory_semantics AS s "
        f"WHERE s.memory_id = memory_records.memory_id) {raw_clause}) "
        "OR EXISTS (SELECT 1 FROM memory_versions AS v "
        "WHERE v.memory_id = memory_records.memory_id "
        f"{version_clause}))",
        (*raw_parameters, *parameters),
    )


def select_memory_contexts(  # noqa: C901 - one authoritative bitemporal selection pass
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    near: SpatialContext | None = None,
    radius_m: float | None = None,
    active_only: bool = False,
) -> tuple[dict[str, tuple[sqlite3.Row, SpatialContext | None]], frozenset[str]]:
    """Choose the one current version row per memory and apply every scope predicate.

    Shared by `read_memories` and `count_memories` so a scoped count cannot drift from the
    scoped hydration it predicts.
    """
    if not memory_ids:
        return {}, frozenset()
    require_scope_axes(valid_at=valid_at, known_at=known_at, near=near, radius_m=radius_m)
    version_clause, version_parameters = version_time_clause(
        valid_at=valid_at, known_at=known_at, active_only=active_only, alias="candidate"
    )

    rows: list[sqlite3.Row] = []
    for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        rows.extend(
            connection.execute(
                f"""
                SELECT
                    s.memory_id AS semantic_memory_id,
                    s.lineage_id, s.kind, s.basis, s.source_id,
                    s.subject, s.predicate, s.value, s.model_id, s.recipe,
                    s.identity_id, s.cue_modality, s.valence, s.arousal,
                    s.spatial_frame_id, s.spatial_anchor,
                    s.spatial_x, s.spatial_y, s.spatial_z,
                    s.spatial_qx, s.spatial_qy, s.spatial_qz, s.spatial_qw,
                    s.spatial_uncertainty_m,
                    v.version, v.confidence, v.valid_from, v.valid_until,
                    v.recorded_at, v.retired_at, v.visible, v.supersedes_id
                FROM memory_semantics AS s
                LEFT JOIN memory_versions AS v
                  ON v.memory_id = s.memory_id AND v.version = (
                      SELECT candidate.version FROM memory_versions AS candidate
                      WHERE candidate.memory_id = s.memory_id {version_clause}
                      ORDER BY candidate.recorded_at DESC, candidate.version DESC
                      LIMIT 1
                  )
                WHERE s.memory_id IN ({placeholders})
                ORDER BY s.memory_id
                """,
                (*version_parameters, *batch),
            ).fetchall()
        )

    semantic_ids = frozenset(row_text(row, "semantic_memory_id") for row in rows)
    selected: dict[str, sqlite3.Row] = {}
    for row in rows:
        memory_id = row_text(row, "semantic_memory_id")
        if row["version"] is None:
            continue
        selected[memory_id] = row

    scoped: dict[str, tuple[sqlite3.Row, SpatialContext | None]] = {}
    for memory_id, row in selected.items():
        spatial: SpatialContext | None = None
        if row["spatial_frame_id"] is not None:
            orientation = None
            if row["spatial_qx"] is not None:
                orientation = (
                    float(row["spatial_qx"]),
                    float(row["spatial_qy"]),
                    float(row["spatial_qz"]),
                    float(row["spatial_qw"]),
                )
            spatial = SpatialContext(
                frame_id=row_text(row, "spatial_frame_id"),
                anchor=SpatialAnchor(row_text(row, "spatial_anchor")),
                x=float(row["spatial_x"]),
                y=float(row["spatial_y"]),
                z=float(row["spatial_z"]),
                orientation_xyzw=orientation,
                position_uncertainty_m=(
                    None
                    if row["spatial_uncertainty_m"] is None
                    else float(row["spatial_uncertainty_m"])
                ),
            )
        if near is not None:
            assert radius_m is not None
            if (
                spatial is None
                or spatial.frame_id != near.frame_id
                or spatial.anchor is not near.anchor
            ):
                continue
            distance = math.sqrt(
                (spatial.x - near.x) ** 2 + (spatial.y - near.y) ** 2 + (spatial.z - near.z) ** 2
            )
            tolerance = radius_m
            tolerance += spatial.position_uncertainty_m or 0.0
            tolerance += near.position_uncertainty_m or 0.0
            if distance > tolerance:
                continue
        scoped[memory_id] = (row, spatial)
    return scoped, semantic_ids


def read_memory_contexts(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
    near: SpatialContext | None = None,
    radius_m: float | None = None,
    active_only: bool = False,
) -> tuple[dict[str, MemoryContext], frozenset[str]]:
    scoped, semantic_ids = select_memory_contexts(
        connection,
        memory_ids,
        valid_at=valid_at,
        known_at=known_at,
        near=near,
        radius_m=radius_m,
        active_only=active_only,
    )
    evidence = _read_memory_evidence(connection, tuple(scoped), known_at=known_at)
    contexts: dict[str, MemoryContext] = {}
    for memory_id, (row, spatial) in scoped.items():
        retired_at = optional_datetime_from_row(row, "retired_at")
        if known_at is not None and retired_at is not None and known_at < retired_at:
            retired_at = None
        contexts[memory_id] = MemoryContext(
            kind=MemoryKind(row_text(row, "kind")),
            basis=EvidenceBasis(row_text(row, "basis")),
            confidence=float(row["confidence"]),
            valid_from=optional_datetime_from_row(row, "valid_from"),
            valid_until=optional_datetime_from_row(row, "valid_until"),
            recorded_at=parse_datetime(row_text(row, "recorded_at")),
            visible=bool(int(row["visible"])),
            retired_at=retired_at,
            lineage_id=row_text(row, "lineage_id"),
            source_id=optional_row_text(row, "source_id"),
            subject=optional_row_text(row, "subject"),
            predicate=optional_row_text(row, "predicate"),
            value=optional_row_text(row, "value"),
            evidence_ids=tuple(evidence.get(memory_id, ())),
            supersedes_id=optional_row_text(row, "supersedes_id"),
            model_id=optional_row_text(row, "model_id"),
            recipe=optional_row_text(row, "recipe"),
            identity_id=optional_row_text(row, "identity_id"),
            spatial=spatial,
            cue_modality=(
                None if row["cue_modality"] is None else Modality(row_text(row, "cue_modality"))
            ),
            valence=None if row["valence"] is None else float(row["valence"]),
            arousal=None if row["arousal"] is None else float(row["arousal"]),
        )
    return contexts, semantic_ids


def _read_memory_evidence(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
    *,
    known_at: datetime | None,
) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {}
    evidence_time = (
        "retired_at IS NULL"
        if known_at is None
        else "recorded_at <= ? AND (retired_at IS NULL OR retired_at > ?)"
    )
    for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
        batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        parameters: tuple[object, ...] = tuple(batch)
        if known_at is not None:
            known_text = datetime_text(known_at)
            parameters += (known_text, known_text)
        for row in connection.execute(
            f"""
            SELECT memory_id, source_memory_id
            FROM memory_evidence
            WHERE memory_id IN ({placeholders}) AND {evidence_time}
            ORDER BY memory_id, position
            """,
            parameters,
        ).fetchall():
            evidence.setdefault(row_text(row, "memory_id"), []).append(
                row_text(row, "source_memory_id")
            )
    return evidence


def identity_scope(
    connection: sqlite3.Connection,
    identity_id: str | None,
) -> tuple[str | None, tuple[object, ...]]:
    """Return the SQL clause and parameters that scope `memory_records` to one identity.

    A `None` clause means the requested identity does not exist, so nothing is in scope.
    """
    if identity_id is None:
        return "", ()
    resolved_id = resolve_identity_id(connection, identity_id)
    if resolved_id is None:
        return None, ()
    return IDENTITY_SCOPE_CLAUSE, (resolved_id, resolved_id, resolved_id)
