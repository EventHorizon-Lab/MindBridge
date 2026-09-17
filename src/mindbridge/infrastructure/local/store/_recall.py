"""Predicate reads behind recall programs: folding, windows, and corpus-order neighbours."""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Sequence
from datetime import datetime

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    row_text,
)
from mindbridge.infrastructure.local.store._membership import (
    IDENTITY_MEMORIES_SQL,
    identity_projections,
)
from mindbridge.infrastructure.local.store.rows import (
    MEMORY_MODALITIES,
    MEMORY_TYPES,
    require_aware,
)

# The event time the recall primitives order and window by. A record with no stated `occurred_at`
# is placed at the time it was stored, which is the same substitution the answer prompt tells the
# reader about, so one timeline covers a corpus that dates some records and not others.
RECALL_EVENT_TIME = "COALESCE(occurred_at, created_at)"


# NFKC casefold as a SQL function, so a substring predicate folds both sides the same way.
# SQLite's own `lower()` is ASCII-only, and a term that differs from the content only by an
# accent or a full-width digit would silently match nothing.
RECALL_FOLD_FUNCTION = "mindbridge_fold"


# One primitive call reads at most this many rows whatever the caller asks for. It bounds the
# hydration behind it, not the scan: a substring predicate reads the table either way.
RECALL_MAX_ROWS = 500


# The longest term a match predicate accepts. A folded substring this long is a paragraph rather
# than a term, and a plan that asks for one is asking `similar` to do a set read's work.
RECALL_MAX_TERM_CHARS = 200


def recall_fold(value: str) -> str:
    """Fold text the one way the recall primitives compare it: NFKC, case-folded."""
    return unicodedata.normalize("NFKC", value).casefold()


def recall_terms(terms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes)):
        raise ValueError("terms must be a sequence of strings")
    folded: list[str] = []
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ValueError("every term must be non-empty text")
        if len(term) > RECALL_MAX_TERM_CHARS:
            raise ValueError(f"a term must be at most {RECALL_MAX_TERM_CHARS} characters")
        folded.append(recall_fold(term.strip()))
    return tuple(dict.fromkeys(folded))


def require_recall_window(
    occurred_from: datetime | None,
    occurred_until: datetime | None,
    *,
    require_both: bool,
) -> None:
    if require_both and (occurred_from is None or occurred_until is None):
        raise ValueError("occurred_from and occurred_until are required")
    if occurred_from is not None:
        require_aware(occurred_from, "occurred_from")
    if occurred_until is not None:
        require_aware(occurred_until, "occurred_until")
    if occurred_from is not None and occurred_until is not None and occurred_until <= occurred_from:
        raise ValueError("occurred_until must be later than occurred_from")


def require_recall_filters(modality: str | None, memory_type: str | None, max_rows: int) -> None:
    if modality is not None and modality not in MEMORY_MODALITIES:
        raise ValueError("modality is invalid")
    if memory_type is not None and memory_type not in MEMORY_TYPES:
        raise ValueError("memory_type is invalid")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int):
        raise ValueError(f"max_rows must be between 1 and {RECALL_MAX_ROWS}")
    if not 1 <= max_rows <= RECALL_MAX_ROWS:
        raise ValueError(f"max_rows must be between 1 and {RECALL_MAX_ROWS}")


def require_recall_neighbors(before: int, after: int, max_rows: int) -> None:
    for name, value in (("before", before), ("after", after)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 50:
            raise ValueError(f"{name} must be between 0 and 50")
    if before == 0 and after == 0:
        raise ValueError("before and after must not both be zero")
    require_recall_filters(None, None, max_rows)


def recall_time_clause(
    occurred_from: datetime | None,
    occurred_until: datetime | None,
) -> tuple[str, tuple[object, ...]]:
    """Bound the effective event interval, start inclusive and end exclusive.

    `embedding_ids_in_range` bounds the same interval but drops rows with no `occurred_at`,
    because an index pass has an unfiltered fallback behind it. These primitives are the answer
    themselves, so a record with no stated event time is placed at its `created_at` rather than
    left out of every window.
    """
    clauses: list[str] = []
    parameters: list[object] = []
    if occurred_from is not None:
        start = datetime_text(occurred_from)
        clauses.append(
            f"""AND (
                (occurred_end IS NOT NULL AND occurred_end > ?)
                OR (occurred_end IS NULL AND {RECALL_EVENT_TIME} >= ?)
            )"""
        )
        parameters.extend((start, start))
    if occurred_until is not None:
        clauses.append(f"AND {RECALL_EVENT_TIME} < ?")
        parameters.append(datetime_text(occurred_until))
    return " ".join(clauses), tuple(parameters)


def recall_term_clause(terms: Sequence[str], *, any_of: bool) -> tuple[str, tuple[object, ...]]:
    if not terms:
        return "", ()
    if not isinstance(any_of, bool):
        raise ValueError("any_of must be a boolean")
    joiner = " OR " if any_of else " AND "
    tests = joiner.join(f"instr({RECALL_FOLD_FUNCTION}(content), ?) > 0" for _term in terms)
    return f"AND ({tests})", tuple(terms)


def neighbor_ids(
    connection: sqlite3.Connection,
    position: sqlite3.Row,
    *,
    before: int,
    after: int,
    place_id: str | None,
    identity_clause: str,
    identity_parameters: tuple[object, ...],
) -> tuple[tuple[str, int, str], ...]:
    """Read the rows immediately around one anchor, each with its own order key."""
    event_time = position["event_time"]
    row_position = position["row_position"]
    place_clause = "" if place_id is None else "AND place_id = ?"
    place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
    found: list[tuple[str, int, str]] = []
    for comparison, order, count in (("<", "DESC", before), (">", "ASC", after)):
        if count == 0:
            continue
        rows = connection.execute(
            f"""
            SELECT memory_id, {RECALL_EVENT_TIME} AS event_time, rowid AS row_position
            FROM memory_records
            WHERE forgotten_at IS NULL
              AND (
                  {RECALL_EVENT_TIME} {comparison} ?
                  OR ({RECALL_EVENT_TIME} = ? AND rowid {comparison} ?)
              )
              {place_clause}
              {identity_clause}
            ORDER BY {RECALL_EVENT_TIME} {order}, rowid {order}
            LIMIT ?
            """,
            (event_time, event_time, row_position, *place_parameters, *identity_parameters, count),
        ).fetchall()
        found.extend(
            (row_text(row, "event_time"), int(row["row_position"]), row_text(row, "memory_id"))
            for row in rows
        )
    return tuple(found)


# The structural edges a related read may follow. Every one is an authoritative column the kernel
# wrote, never a similarity: two records share a capture, a person, a room, a lineage, or an
# evidence link, or they do not. That is what makes expansion free -- the edges are already in
# SQLite, so recovering the rest of a question's support costs one indexed read and no model call.
RELATED_EDGES: tuple[str, ...] = ("capture", "identity", "place", "lineage", "evidence")


# Anchor rows to the keys one edge groups by, and those keys back to the memories that carry
# them. `identity` and `evidence` are not key lookups and are handled in `related_memory_ids`.
_EDGE_KEYS: dict[str, str] = {
    "capture": (
        "SELECT DISTINCT source_id AS edge_key FROM memory_semantics "
        "WHERE memory_id IN ({ids}) AND source_id IS NOT NULL"
    ),
    "lineage": (
        "SELECT DISTINCT lineage_id AS edge_key FROM memory_semantics WHERE memory_id IN ({ids})"
    ),
    "place": (
        "SELECT DISTINCT place_id AS edge_key FROM memory_records "
        "WHERE memory_id IN ({ids}) AND place_id IS NOT NULL"
    ),
}


# ponytail: `memory_semantics.source_id` carries no index, so the capture edge is a scan of the
# semantic table -- fine at the sizes a companion store reaches, wrong for a million-record one.
# `CREATE INDEX memory_semantics_source_idx ON memory_semantics (source_id, memory_id) WHERE
# source_id IS NOT NULL` is the upgrade, and it needs a schema version.
_EDGE_MEMBERS: dict[str, str] = {
    "capture": "SELECT memory_id FROM memory_semantics WHERE source_id IN ({keys})",
    "lineage": "SELECT memory_id FROM memory_semantics WHERE lineage_id IN ({keys})",
    "place": "SELECT memory_id FROM memory_records WHERE place_id IN ({keys})",
}


def require_related_edges(edges: Sequence[str]) -> tuple[str, ...]:
    if isinstance(edges, (str, bytes)):
        raise ValueError("edges must be a sequence of edge names")
    chosen = tuple(dict.fromkeys(edges))
    unknown = tuple(edge for edge in chosen if edge not in RELATED_EDGES)
    if not chosen or unknown:
        raise ValueError(f"edges must be chosen from {', '.join(RELATED_EDGES)}")
    return chosen


def related_memory_ids(
    connection: sqlite3.Connection,
    anchors: Sequence[str],
    *,
    edges: Sequence[str],
    ceiling: int | None,
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Return the memories each edge links the anchors to, and the edges that matched too much.

    An edge is dropped whole when it links to more than `ceiling` records, and it is reported
    rather than trimmed. This is the per-edge form of the bound recall programs already apply to
    a predicate: on a two-speaker transcript every record is about both speakers, so the identity
    edge selects the corpus and says nothing about the question -- while the capture edge beside
    it stays selective. Gating the reads together would let the first discard the second.
    """
    found: dict[str, tuple[str, ...]] = {}
    dropped: list[str] = []
    for edge in edges:
        if edge == "identity":
            keys = tuple(
                dict.fromkeys(
                    identity_id
                    for identities in identity_projections(connection, anchors).values()
                    for identity_id in identities
                )
            )
            members = tuple(
                row_text(row, "memory_id")
                for key in keys
                for row in connection.execute(IDENTITY_MEMORIES_SQL, (key, key, key))
            )
        elif edge == "evidence":
            members = tuple(
                row_text(row, "memory_id")
                for statement in (
                    "SELECT memory_id FROM memory_evidence "
                    "WHERE source_memory_id IN ({ids}) AND retired_at IS NULL",
                    "SELECT source_memory_id AS memory_id FROM memory_evidence "
                    "WHERE memory_id IN ({ids}) AND retired_at IS NULL",
                )
                for row in _rows_for_ids(connection, statement, anchors)
            )
        else:
            keys = tuple(
                dict.fromkeys(
                    row_text(row, "edge_key")
                    for row in _rows_for_ids(connection, _EDGE_KEYS[edge], anchors)
                )
            )
            members = tuple(
                row_text(row, "memory_id")
                for row in _rows_for_ids(connection, _EDGE_MEMBERS[edge], keys, key="keys")
            )
        linked = tuple(dict.fromkeys(members))
        if ceiling is not None and len(linked) > ceiling:
            dropped.append(edge)
            continue
        found[edge] = linked
    return found, tuple(dropped)


def _rows_for_ids(
    connection: sqlite3.Connection,
    statement: str,
    identifiers: Sequence[str],
    *,
    key: str = "ids",
) -> tuple[sqlite3.Row, ...]:
    """Run one `IN (...)` statement over an identifier set too large for a single bind."""
    rows: list[sqlite3.Row] = []
    for offset in range(0, len(identifiers), SQLITE_PARAMETER_BATCH):
        batch = identifiers[offset : offset + SQLITE_PARAMETER_BATCH]
        if not batch:
            continue
        placeholders = ", ".join("?" for _identifier in batch)
        rows.extend(connection.execute(statement.format(**{key: placeholders}), tuple(batch)))
    return tuple(rows)
