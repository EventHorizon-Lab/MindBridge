"""Recall primitives: substring match, event-time windows, neighbours, and the corpus digest."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from mindbridge.infrastructure.local.store._codec import optional_datetime_from_row, row_text
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._identity import canonical_subject, resolve_identity_id
from mindbridge.infrastructure.local.store._recall import (
    RECALL_EVENT_TIME,
    neighbor_ids,
    recall_term_clause,
    recall_terms,
    recall_time_clause,
    require_recall_filters,
    require_recall_neighbors,
    require_recall_window,
)
from mindbridge.infrastructure.local.store._selection import identity_scope
from mindbridge.infrastructure.local.store.records import MemoryRecords
from mindbridge.infrastructure.local.store.rows import (
    RecallDigest,
    RecallRead,
    StoredMemory,
    require_identifier,
)
from mindbridge.types import SpatialContext


class RecallReads:
    """Predicate reads that answer by completeness rather than by rank."""

    def __init__(
        self,
        *,
        connections: Connections,
        records: MemoryRecords,
    ) -> None:
        self._connections = connections
        self._records = records
        self._digest: tuple[int, RecallDigest] | None = None

    def match_memories(
        self,
        terms: Sequence[str] = (),
        *,
        any_of: bool = True,
        occurred_from: datetime | None = None,
        occurred_until: datetime | None = None,
        modality: str | None = None,
        memory_type: str | None = None,
        max_rows: int = 200,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> RecallRead:
        """Return every active memory whose content contains the terms, oldest first.

        This is the exhaustive counterpart to search: the answer is defined by the predicate and
        not by a relevance order, so a caller can say "these are all of them" up to `max_rows`.
        Matching is an NFKC case-folded substring over `content`, which already carries
        transcripts, visual descriptions and speech prose, so a media memory is reachable by the
        words its own record holds. No terms means the predicate is the time window alone.

        The scope arguments are `read_memories`'s, and hydration goes through it, so bitemporal,
        place, identity and metric scope have exactly one implementation. Because that read runs
        after `max_rows` has already been applied, the returned `RecallRead` carries how many IDs
        were selected: a caller can only tell a truncated set from a complete one from that count.
        """
        folded = recall_terms(terms)
        require_recall_window(occurred_from, occurred_until, require_both=False)
        require_recall_filters(modality, memory_type, max_rows)
        selected = self._recall_memory_ids(
            folded,
            any_of=any_of,
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            modality=modality,
            memory_type=memory_type,
            max_rows=max_rows,
            place_id=place_id,
            identity_id=identity_id,
        )
        return RecallRead(
            self._hydrate_recall(
                selected,
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                place_id=place_id,
                identity_id=identity_id,
            ),
            selected=len(selected),
        )

    def memories_in_window(
        self,
        *,
        occurred_from: datetime,
        occurred_until: datetime,
        modality: str | None = None,
        memory_type: str | None = None,
        max_rows: int = 200,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> RecallRead:
        """Return every active memory whose event time overlaps the window, oldest first.

        The window is half-open, start inclusive, and it is required: a window primitive with no
        bounds is the whole store, which is `list_memories`. A memory with no `occurred_at` is
        placed by its `created_at`, the same substitution the answer prompt describes, so a
        corpus that never states event times still has a timeline.
        """
        require_recall_window(occurred_from, occurred_until, require_both=True)
        return self.match_memories(
            (),
            occurred_from=occurred_from,
            occurred_until=occurred_until,
            modality=modality,
            memory_type=memory_type,
            max_rows=max_rows,
            valid_at=valid_at,
            known_at=known_at,
            near=near,
            radius_m=radius_m,
            place_id=place_id,
            identity_id=identity_id,
        )

    def neighbor_memories(
        self,
        memory_ids: Sequence[str],
        *,
        before: int = 1,
        after: int = 1,
        max_rows: int = 200,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> RecallRead:
        """Return the active memories adjacent to the anchors in corpus order, oldest first.

        Corpus order is `(effective event time, rowid)`. `memory_records` is a rowid table and
        every adapter writes one turn per `add`, in turn order, so the rowid breaks ties between
        records that share a timestamp by the order they arrived -- which is what "what did I say
        just before that" asks for. The anchors themselves are not returned: the step that found
        them already holds them.
        """
        anchors = tuple(dict.fromkeys(memory_ids))
        for memory_id in anchors:
            require_identifier(memory_id, "memory_id")
        require_recall_neighbors(before, after, max_rows)
        if not anchors:
            return RecallRead()
        # ponytail: two statements per anchor, each an index-less ordered scan bounded by
        # `before`/`after`. Anchors are bounded by the caller's plan; if a plan ever wants
        # hundreds, a single window-function query over the whole ordered set is the upgrade.
        found: list[tuple[str, int, str]] = []
        with self._connections.read_transaction() as connection:
            identity_clause, identity_parameters = identity_scope(connection, identity_id)
            if identity_clause is None:
                return RecallRead()
            for memory_id in anchors:
                position = connection.execute(
                    f"""
                    SELECT {RECALL_EVENT_TIME} AS event_time, rowid AS row_position
                    FROM memory_records
                    WHERE memory_id = ? AND forgotten_at IS NULL
                    """,
                    (memory_id,),
                ).fetchone()
                if position is None:
                    continue
                found.extend(
                    neighbor_ids(
                        connection,
                        position,
                        before=before,
                        after=after,
                        place_id=place_id,
                        identity_clause=identity_clause,
                        identity_parameters=identity_parameters,
                    )
                )
        anchor_ids = set(anchors)
        ordered = tuple(
            dict.fromkeys(
                memory_id
                for _event_time, _row_position, memory_id in sorted(found)
                if memory_id not in anchor_ids
            )
        )
        selected = ordered[:max_rows]
        return RecallRead(
            self._hydrate_recall(
                selected,
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                place_id=place_id,
                identity_id=identity_id,
            ),
            selected=len(selected),
        )

    def identity_id_for_name(self, name: str) -> str | None:
        """Resolve a display name to the canonical identity ID that carries it.

        The comparison is the NFKC casefold the rest of the kernel uses, and the lowest matching
        ID wins so two identities registered under one name resolve deterministically. A name no
        identity was registered under returns None, which leaves the caller to fall back on
        matching the name as text.
        """
        wanted = canonical_subject(name)
        if wanted is None:
            return None
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT identity_id, name
                FROM identities
                WHERE name IS NOT NULL
                ORDER BY identity_id
                """
            ).fetchall()
            for row in rows:
                if canonical_subject(row_text(row, "name")) == wanted:
                    return resolve_identity_id(connection, row_text(row, "identity_id"))
        return None

    def recall_digest(self) -> RecallDigest:
        """Summarize the active corpus, recomputed only after a commit.

        Three aggregate reads, none of them touching content or media: a count, the span of the
        effective event time, and what modalities and named identities exist at all.
        """
        cached = self._digest
        generation = self._connections.write_generation
        if cached is not None and cached[0] == generation:
            return cached[1]
        with self._connections.read_transaction() as connection:
            span = connection.execute(
                f"""
                SELECT COUNT(*) AS records,
                       MIN({RECALL_EVENT_TIME}) AS earliest,
                       MAX(COALESCE(occurred_end, occurred_at, created_at)) AS latest
                FROM memory_records
                WHERE forgotten_at IS NULL
                """
            ).fetchone()
            modalities = connection.execute(
                """
                SELECT DISTINCT modality
                FROM memory_records
                WHERE forgotten_at IS NULL
                ORDER BY modality
                """
            ).fetchall()
            named = connection.execute(
                "SELECT COUNT(*) AS named FROM identities WHERE name IS NOT NULL"
            ).fetchone()
        digest = RecallDigest(
            records=0 if span is None else int(span["records"]),
            earliest=None if span is None else optional_datetime_from_row(span, "earliest"),
            latest=None if span is None else optional_datetime_from_row(span, "latest"),
            modalities=tuple(row_text(row, "modality") for row in modalities),
            named_identities=0 if named is None else int(named["named"]),
        )
        self._digest = (generation, digest)
        return digest

    def _recall_memory_ids(
        self,
        terms: tuple[str, ...],
        *,
        any_of: bool,
        occurred_from: datetime | None,
        occurred_until: datetime | None,
        modality: str | None,
        memory_type: str | None,
        max_rows: int,
        place_id: str | None,
        identity_id: str | None,
    ) -> tuple[str, ...]:
        """Select the bounded, chronologically ordered ID set one recall predicate defines."""
        clauses: list[str] = []
        parameters: list[object] = []
        time_clause, time_parameters = recall_time_clause(occurred_from, occurred_until)
        clauses.append(time_clause)
        parameters.extend(time_parameters)
        for column, value in (("modality", modality), ("memory_type", memory_type)):
            if value is not None:
                clauses.append(f"AND {column} = ?")
                parameters.append(value)
        if place_id is not None:
            clauses.append("AND place_id = ?")
            parameters.append(place_id)
        term_clause, term_parameters = recall_term_clause(terms, any_of=any_of)
        clauses.append(term_clause)
        parameters.extend(term_parameters)
        with self._connections.read_transaction() as connection:
            identity_clause, identity_parameters = identity_scope(connection, identity_id)
            if identity_clause is None:
                return ()
            rows = connection.execute(
                f"""
                SELECT memory_id
                FROM memory_records
                WHERE forgotten_at IS NULL
                {" ".join(clauses)}
                {identity_clause}
                ORDER BY {RECALL_EVENT_TIME}, rowid
                LIMIT ?
                """,
                (*parameters, *identity_parameters, max_rows),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)

    def _hydrate_recall(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None,
        known_at: datetime | None,
        near: SpatialContext | None,
        radius_m: float | None,
        place_id: str | None,
        identity_id: str | None,
    ) -> tuple[StoredMemory, ...]:
        """Hydrate a selected ID set through the one authoritative scoped read."""
        if not memory_ids:
            return ()
        return self._records.read_memories(
            memory_ids,
            valid_at=valid_at,
            known_at=known_at,
            near=near,
            radius_m=radius_m,
            place_id=place_id,
            identity_id=identity_id,
            active_only=True,
        )
