"""Consolidation candidate discovery for the memory control plane."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from itertools import groupby

from mindbridge.infrastructure.local.store._codec import (
    optional_datetime_from_row,
    parse_datetime,
    row_text,
)
from mindbridge.infrastructure.local.store.rows import StoredCandidate
from mindbridge.types import EvidenceBasis, MemoryKind

# Claims whose write contract gives one lineage one standing value. Relations are accumulating,
# as are model-inferred traits; only a person's own trait assertion has the single-value contract.
# Keep the complete SQL predicate in one place so all functional scans agree about which claims
# can participate in a conflict.
FUNCTIONAL_CLAIM_SQL = "(s.kind = ? OR (s.kind = ? AND s.basis = ?))"


FUNCTIONAL_CLAIM_PARAMETERS = (
    MemoryKind.STATE.value,
    MemoryKind.TRAIT.value,
    EvidenceBasis.USER_STATEMENT.value,
)


def weighed_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when anything last weighed it: an operation or a deliberation.

    Both halves are needed. An applied operation proves the record was considered *and* acted
    on; a deliberation row proves only that it was considered, which is exactly the case an
    operation log cannot record -- the backend proposed nothing, or the kernel refused
    everything. Without the second half a zero-yield candidate returns every round and is paid
    for every round.
    """
    return merge_weighed(
        consumed_at_by_memory(connection),
        deliberated_at_by_memory(connection),
    )


def merge_weighed(
    consumed: dict[str, datetime],
    deliberated: Mapping[str, datetime],
) -> dict[str, datetime]:
    """Keep the newer of the two marks per memory, mutating and returning the first."""
    for memory_id, deliberated_at in deliberated.items():
        previous = consumed.get(memory_id)
        if previous is None or deliberated_at > previous:
            consumed[memory_id] = deliberated_at
    return consumed


def deliberated_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when a deliberation last had it in its evidence set."""
    return {
        row_text(row, "memory_id"): parse_datetime(row_text(row, "weighed_at"))
        for row in connection.execute(
            """
            SELECT m.memory_id AS memory_id, MAX(d.weighed_at) AS weighed_at
            FROM memory_deliberation_memories AS m
            JOIN memory_deliberations AS d ON d.deliberation_id = m.deliberation_id
            GROUP BY m.memory_id
            """
        ).fetchall()
    }


def consumed_at_by_memory(connection: sqlite3.Connection) -> dict[str, datetime]:
    """Return, per memory, when a still-standing operation last consumed or produced it.

    The log stores IDs inside its two JSON documents rather than in columns, so this reads them
    back with `json_tree`. A rolled-back operation consumed nothing that still stands.
    """
    # ponytail: one full scan of the standing operation log per call, JSON-parsed row by row,
    # because both consumers need the map before they pick their candidate windows. Narrow it to
    # `j.value IN (...)` over those windows -- which means computing them first -- once a store's
    # log is long enough for the scan to show up next to the two window queries.
    return {
        row_text(row, "memory_id"): parse_datetime(row_text(row, "consumed_at"))
        for row in connection.execute(
            """
            SELECT j.value AS memory_id, MAX(o.applied_at) AS consumed_at
            FROM memory_operations AS o
            JOIN json_tree(json_array(
                     json_extract(o.operation_json, '$.evidence_ids'),
                     json_extract(o.operation_json, '$.target_ids'),
                     json_extract(o.effects_json, '$.created_ids'),
                     json_extract(o.effects_json, '$.changed_ids')
                 )) AS j
            WHERE o.rolled_back_at IS NULL AND j.type = 'text'
            GROUP BY j.value
            """
        ).fetchall()
    }


def evidence_candidates(
    connection: sqlite3.Connection,
    consumed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Derived records that gained independent evidence nothing has weighed yet."""
    # ponytail: over-fetch the newest window and filter, rather than push the consumed-time
    # comparison into SQL. Raise the factor only if a store whose newest records are all already
    # deliberated measurably under-fills the window.
    newest = connection.execute(
        """
        SELECT e.memory_id AS memory_id, MAX(e.recorded_at) AS newest
        FROM memory_evidence AS e
        JOIN memory_records AS r ON r.memory_id = e.memory_id
        WHERE e.retired_at IS NULL AND r.forgotten_at IS NULL
        GROUP BY e.memory_id
        ORDER BY newest DESC, e.memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    supported = {
        row_text(row, "memory_id"): parse_datetime(row_text(row, "newest")) for row in newest
    }
    due = {
        memory_id: consumed.get(memory_id)
        for memory_id, latest in supported.items()
        if memory_id not in consumed or latest > consumed[memory_id]
    }
    if not due:
        return []
    placeholders = ", ".join("?" for _memory_id in due)
    fresh: dict[str, list[str]] = {memory_id: [] for memory_id in due}
    # A forgotten source is dropped here, not later: `consolidate()` reads the shown records with
    # `active_only=True`, so naming one would inflate `evidence_count` with an ID the deliberation
    # never sees. A candidate left with no remaining sources falls out of the `if sources` guard.
    for row in connection.execute(
        f"""
        SELECT e.memory_id AS memory_id, e.source_memory_id AS source_memory_id,
               e.recorded_at AS recorded_at
        FROM memory_evidence AS e
        JOIN memory_records AS s
          ON s.memory_id = e.source_memory_id AND s.forgotten_at IS NULL
        WHERE e.retired_at IS NULL AND e.memory_id IN ({placeholders})
        ORDER BY e.memory_id, e.position
        """,
        tuple(due),
    ).fetchall():
        memory_id = row_text(row, "memory_id")
        threshold = due[memory_id]
        if threshold is None or parse_datetime(row_text(row, "recorded_at")) > threshold:
            fresh[memory_id].append(row_text(row, "source_memory_id"))
    return [
        StoredCandidate(
            trigger="evidence",
            memory_ids=(memory_id, *sources),
            evidence_count=len(sources),
        )
        for memory_id, sources in fresh.items()
        if sources
    ][:limit]


def contradiction_candidates(
    connection: sqlite3.Connection,
    weighed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Lineages whose current visible claims disagree, the way the compiler reports them.

    Retiring one side through `CORRECT` is what clears a contradiction, but a deliberation the
    model could not resolve clears nothing -- so the lineage is dropped while nothing about it
    has changed since it was last weighed, and returns as soon as a further claim is recorded in
    it. The signal is the newest transaction time among the disagreeing claims.
    """
    entries: dict[str, list[tuple[str, str, datetime]]] = {}
    # A correlated overlap predicate made SQLite re-discover the same conflicting peers for each
    # member (and then do it again while hydrating members).  Claims in a lineage are instead
    # consumed together.  The two sweeps retain only the best two endpoints for distinct values:
    # an interval participates iff an earlier or later interval with another value overlaps it.
    rows = connection.execute(
        f"""
        SELECT s.lineage_id AS lineage_id, s.memory_id AS memory_id, s.value AS value,
               v.recorded_at AS recorded_at, v.valid_from AS valid_from,
               v.valid_until AS valid_until
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
          AND s.lineage_id IS NOT NULL AND s.value IS NOT NULL AND {FUNCTIONAL_CLAIM_SQL}
        ORDER BY s.lineage_id, s.memory_id
        """,
        FUNCTIONAL_CLAIM_PARAMETERS,
    )
    for lineage_id, lineage_rows in groupby(rows, key=lambda row: row_text(row, "lineage_id")):
        members = _overlapping_functional_members(lineage_rows)
        if members:
            entries[lineage_id] = members
            # The old query applies the lineage limit before due-state filtering.  Stop here so a
            # previously weighed earlier lineage continues to consume a slot in exactly the same way.
            if len(entries) == limit:
                break
    candidates: list[StoredCandidate] = []
    for lineage_entries in entries.values():
        memory_ids = tuple(entry[0] for entry in lineage_entries)
        signal_at = max(entry[2] for entry in lineage_entries)
        if _is_due(signal_at, memory_ids, weighed):
            candidates.append(
                StoredCandidate(
                    trigger="contradiction",
                    memory_ids=memory_ids,
                    evidence_count=len({entry[1] for entry in lineage_entries}),
                )
            )
    return candidates


def _overlapping_functional_members(
    rows: Iterable[sqlite3.Row],
) -> list[tuple[str, str, datetime]]:
    """Return only members with a differently-valued half-open overlap in one lineage."""
    intervals = [
        (
            row_text(row, "memory_id"),
            row_text(row, "value"),
            parse_datetime(row_text(row, "recorded_at")),
            optional_datetime_from_row(row, "valid_from"),
            optional_datetime_from_row(row, "valid_until"),
        )
        for row in rows
    ]
    intervals.sort(key=lambda entry: (entry[3] is not None, entry[3], entry[0]))
    overlapping_ids: set[str] = set()

    # Each entry is (value, endpoint).  An endpoint of None means unbounded: -infinity for a
    # start and +infinity for an end.  Keeping two different values is sufficient because a
    # query excludes only its own value; a discarded value cannot become extremal without a later
    # row of that value re-entering the two slots.
    latest_ends: list[tuple[str, datetime | None]] = []
    for memory_id, value, _recorded_at, valid_from, valid_until in intervals:
        other_end = _other_extremum(latest_ends, value)
        if other_end is not None and _end_after(other_end[1], valid_from):
            overlapping_ids.add(memory_id)
        _update_endpoint_extrema(latest_ends, value, valid_until, latest=True)

    earliest_starts: list[tuple[str, datetime | None]] = []
    for memory_id, value, _recorded_at, valid_from, valid_until in reversed(intervals):
        other_start = _other_extremum(earliest_starts, value)
        if other_start is not None and _end_after(valid_until, other_start[1]):
            overlapping_ids.add(memory_id)
        _update_endpoint_extrema(earliest_starts, value, valid_from, latest=False)

    members = [
        (memory_id, value, recorded_at)
        for memory_id, value, recorded_at, _valid_from, _valid_until in intervals
        if memory_id in overlapping_ids
    ]
    return sorted(members, key=lambda entry: entry[0])


def _other_extremum(
    extrema: Sequence[tuple[str, datetime | None]], value: str
) -> tuple[str, datetime | None] | None:
    for other_value, endpoint in extrema:
        if other_value != value:
            return other_value, endpoint
    return None


def _end_after(end: datetime | None, start: datetime | None) -> bool:
    return end is None or start is None or end > start


def _update_endpoint_extrema(
    extrema: list[tuple[str, datetime | None]],
    value: str,
    endpoint: datetime | None,
    *,
    latest: bool,
) -> None:
    """Update two distinct value extrema in constant time."""
    for index, (known_value, known_endpoint) in enumerate(extrema):
        if known_value == value:
            if _endpoint_precedes(known_endpoint, endpoint, latest=latest):
                extrema[index] = (value, endpoint)
            break
    else:
        extrema.append((value, endpoint))
    extrema.sort(key=lambda item: _endpoint_sort_key(item[1], latest=latest), reverse=latest)
    del extrema[2:]


def _endpoint_precedes(left: datetime | None, right: datetime | None, *, latest: bool) -> bool:
    if latest:
        return _endpoint_sort_key(left, latest=True) < _endpoint_sort_key(right, latest=True)
    return _endpoint_sort_key(left, latest=False) > _endpoint_sort_key(right, latest=False)


def _endpoint_sort_key(endpoint: datetime | None, *, latest: bool) -> tuple[bool, datetime | None]:
    # For latest ends, None sorts after all timestamps; for earliest starts, before them.
    return (endpoint is None if latest else endpoint is not None, endpoint)


def _is_due(
    signal_at: datetime, memory_ids: Sequence[str], weighed: Mapping[str, datetime]
) -> bool:
    """Report whether a group's own signal is newer than anything that has weighed it."""
    marks = [weighed[memory_id] for memory_id in memory_ids if memory_id in weighed]
    return not marks or signal_at > max(marks)


def pressure_candidates(
    connection: sqlite3.Connection,
    weighed: Mapping[str, datetime],
    limit: int,
    record_budget: int | None,
) -> list[StoredCandidate]:
    """Oldest never-weighed records, while the store holds more than its configured budget.

    Pressure has no signal time of its own -- the condition is a level, not an event -- so
    "already weighed" here means weighed at all rather than weighed since some moment. Once
    everything in the store has been weighed once, pressure derives nothing further: a
    deliberation that keeps re-proposing the same forgetting is the disease this marker exists to
    cure, and the honest answer is that there is no new work.
    """
    if record_budget is None:
        return []
    active = int(
        connection.execute(
            "SELECT COUNT(*) AS records FROM memory_records WHERE forgotten_at IS NULL"
        ).fetchone()["records"]
    )
    if active <= record_budget:
        return []
    rows = connection.execute(
        """
        SELECT memory_id
        FROM memory_records
        WHERE forgotten_at IS NULL
        ORDER BY COALESCE(last_accessed_at, created_at), memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    over = active - record_budget
    return [
        StoredCandidate(
            trigger="pressure",
            memory_ids=(memory_id,),
            evidence_count=over,
        )
        for memory_id in (row_text(row, "memory_id") for row in rows)
        if memory_id not in weighed
    ][:limit]


def idle_candidates(
    connection: sqlite3.Connection,
    weighed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Derived lineages nothing has ever weighed, oldest first.

    Only reached when the operator declares the window. The kernel never infers idleness from a
    clock: whether the device is idle or charging is the host's knowledge, not the store's.
    """
    rows = connection.execute(
        """
        SELECT s.memory_id AS memory_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE v.retired_at IS NULL AND r.forgotten_at IS NULL
        ORDER BY v.recorded_at, s.memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    return [
        StoredCandidate(trigger="idle", memory_ids=(memory_id,), evidence_count=1)
        for memory_id in (row_text(row, "memory_id") for row in rows)
        if memory_id not in weighed
    ][:limit]


def feedback_candidates(
    connection: sqlite3.Connection,
    consumed: Mapping[str, datetime],
    limit: int,
) -> list[StoredCandidate]:
    """Records whose recall was confirmed since a standing operation last saw them.

    `reinforce_memories` is one writer of `last_accessed_at`; under the default
    `reinforce_on_answer`, an `ask()` answer citing a record is the other.
    """
    rows = connection.execute(
        """
        SELECT memory_id, last_accessed_at, access_count
        FROM memory_records
        WHERE last_accessed_at IS NOT NULL AND forgotten_at IS NULL
        ORDER BY last_accessed_at DESC, memory_id
        LIMIT ?
        """,
        (min(1_000, limit * 4),),
    ).fetchall()
    candidates: list[StoredCandidate] = []
    for row in rows:
        memory_id = row_text(row, "memory_id")
        threshold = consumed.get(memory_id)
        accessed = parse_datetime(row_text(row, "last_accessed_at"))
        count = int(row["access_count"])
        if count > 0 and (threshold is None or accessed > threshold):
            candidates.append(
                StoredCandidate(
                    trigger="feedback",
                    memory_ids=(memory_id,),
                    evidence_count=count,
                )
            )
    return candidates[:limit]
