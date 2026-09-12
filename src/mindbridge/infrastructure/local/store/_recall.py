"""Predicate reads behind recall programs: folding, windows, and corpus-order neighbours."""

from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Sequence
from datetime import datetime

from mindbridge.infrastructure.local.store._codec import datetime_text, row_text
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
