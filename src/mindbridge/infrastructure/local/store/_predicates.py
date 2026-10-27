"""Persist and validate exact predicate resolution for the current schema."""

from __future__ import annotations

import sqlite3

from mindbridge.infrastructure.local.store.errors import UnsupportedSchemaError
from mindbridge.kernel.predicates import (
    PREDICATE_REGISTRY_VERSION,
    claim_lineage_id,
    resolve_predicate,
)


def _validate_semantic_columns(connection: sqlite3.Connection) -> None:
    names = (
        "memory_id",
        "lineage_id",
        "kind",
        "basis",
        "source_id",
        "subject",
        "predicate",
        "value",
        "model_id",
        "recipe",
        "identity_id",
        "cue_modality",
        "valence",
        "arousal",
        "spatial_frame_id",
        "spatial_anchor",
        "spatial_x",
        "spatial_y",
        "spatial_z",
        "spatial_qx",
        "spatial_qy",
        "spatial_qz",
        "spatial_qw",
        "spatial_uncertainty_m",
        "predicate_key",
        "predicate_registry",
        "predicate_match",
    )
    reals = {
        "valence",
        "arousal",
        "spatial_x",
        "spatial_y",
        "spatial_z",
        "spatial_qx",
        "spatial_qy",
        "spatial_qz",
        "spatial_qw",
        "spatial_uncertainty_m",
    }
    expected = {
        name: (
            "REAL" if name in reals else "TEXT",
            int(name in {"lineage_id", "kind", "basis"}),
            int(name == "memory_id"),
        )
        for name in names
    }
    actual = {
        str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
        for row in connection.execute("PRAGMA table_info(memory_semantics)")
    }
    if actual != expected:
        raise UnsupportedSchemaError("local schema has invalid predicate semantics columns")


def predicate_binding(
    lineage_id: str,
    *,
    kind: str,
    basis: str,
    predicate: str | None,
    value: str | None,
    subject: str | None,
    identity_id: str | None,
    frame_id: str | None,
    anchor: str | None,
) -> tuple[str | None, str]:
    resolution = resolve_predicate(predicate, kind=kind, basis=basis, value=value)
    canonical = claim_lineage_id(
        kind=kind,
        predicate=predicate,
        subject=subject,
        identity_id=identity_id,
        frame_id=frame_id,
        anchor=anchor,
        resolution=resolution,
    )
    if lineage_id != canonical:
        # A caller's custom lineage is authority too; an apparent alias is no permission to
        # reinterpret it. Leave the resolved key empty and identify the custom binding explicitly.
        return None, "custom"
    return resolution.key, resolution.match


def validate_predicate_schema(connection: sqlite3.Connection) -> None:
    _validate_semantic_columns(connection)
    if connection.execute(
        "SELECT 1 FROM memory_semantics WHERE predicate_registry IS NULL OR predicate_registry <> ? "
        "OR predicate_match IS NULL OR predicate_match NOT IN "
        "('canonical', 'alias', 'literal', 'ineligible', 'custom') LIMIT 1",
        (PREDICATE_REGISTRY_VERSION,),
    ).fetchone():
        raise UnsupportedSchemaError("local schema has unsupported predicate provenance")
