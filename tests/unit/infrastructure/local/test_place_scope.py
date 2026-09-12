"""The symbolic place axis: one nullable `place_id` and an indexed equality over it.

MindBridge already stores metric pose -- a frame id, a position, a quaternion and an uncertainty --
and already scopes by radius. What it cannot express is the predicate a household query actually
uses ("in the kitchen"), which is also the only spatial label a robot can supply when it cannot
localise metrically. That is a symbolic equality, and an equality is the one spatial predicate
SQLite indexes cheaply.

These tests pin the four things that make it worth having rather than the metric version: the
value round-trips and stays optional, a scoped hydration filters in SQL and is planned onto
`memory_records_place_idx` rather than degrading into the post-hoc Python filter the metric radius
scope is forced to be, relabelling a room reprojects the memory into the search index (which now
carries `place_id` as a pushed-down filter field), and an existing store gains the column without
losing a memory.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.store._connections import Connections

_NOW = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
_SPACE = "place-probe:2"
_TASK = "retrieval.document"


def _memory(
    memory_id: str,
    content: str,
    *,
    place_id: str | None = None,
    memory_type: str = "semantic",
) -> StoredMemory:
    return StoredMemory(
        memory_id=memory_id,
        content=content,
        metadata_json="{}",
        created_at=_NOW,
        updated_at=_NOW,
        occurred_at=_NOW,
        memory_type=memory_type,
        place_id=place_id,
    )


def _embedding(memory_id: str, *, object_part: int = 0) -> StoredEmbedding:
    return StoredEmbedding(
        embedding_id=f"{memory_id}#{object_part}",
        memory_id=memory_id,
        values=(0.6, 0.8),
        model_id="place-probe",
        space_id=_SPACE,
        task=_TASK,
        created_at=_NOW,
        object_part=object_part,
        normalized=True,
    )


def test_place_id_round_trips_and_stays_optional(tmp_path: Path) -> None:
    """A place is a label a robot may or may not have, so absence is a first-class value."""
    with LocalStore(tmp_path) as store:
        store.records.write_memory(
            _memory("labelled", "the blue inhaler is in the top drawer", place_id="kitchen")
        )
        store.records.write_memory(_memory("unlabelled", "someone mentioned Thursday"))

        labelled = store.records.read_memory("labelled")
        unlabelled = store.records.read_memory("unlabelled")
        assert labelled is not None and labelled.place_id == "kitchen"
        assert unlabelled is not None and unlabelled.place_id is None

        # The batch read and the listing hydrate the same column, not just the single read.
        assert [
            memory.place_id for memory in store.records.read_memories(("labelled", "unlabelled"))
        ] == [
            "kitchen",
            None,
        ]
        assert {memory.memory_id: memory.place_id for memory in store.records.list_memories()} == {
            "labelled": "kitchen",
            "unlabelled": None,
        }


def test_a_place_label_must_be_real_text(tmp_path: Path) -> None:
    """`None` means "unknown"; an empty or untrimmed label would be a second, silent spelling."""
    with pytest.raises(ValueError, match="place_id"):
        _memory("blank", "text", place_id="")
    with pytest.raises(ValueError, match="place_id"):
        _memory("padded", "text", place_id=" kitchen ")
    with (
        LocalStore(tmp_path) as store,
        closing(sqlite3.connect(store.database_path)) as connection,
        pytest.raises(sqlite3.IntegrityError),
    ):
        connection.execute(
            """
            INSERT INTO memory_records (
                memory_id, content, modality, memory_type, metadata_json,
                created_at, updated_at, place_id
            ) VALUES ('raw', 'text', 'text', 'semantic', '{}', ?, ?, '   ')
            """,
            (_NOW.isoformat(), _NOW.isoformat()),
        )


def test_hydrating_a_candidate_slate_scopes_it_by_place(tmp_path: Path) -> None:
    """`read_memories(place_id=...)` is where the search path's other scope axes already plug in.

    The filter is SQL rather than Python so a scoped hydration reads only the rows it returns, and
    it composes with the bitemporal and metric arguments instead of shadowing them.
    """
    with LocalStore(tmp_path) as store:
        store.records.write_memory(_memory("kitchen-1", "the kettle is on", place_id="kitchen"))
        store.records.write_memory(
            _memory("kitchen-2", "the top drawer is open", place_id="kitchen")
        )
        store.records.write_memory(_memory("garden-1", "the hose is coiled", place_id="garden"))
        store.records.write_memory(_memory("nowhere", "someone said Thursday"))
        slate = ("garden-1", "kitchen-2", "nowhere", "kitchen-1")

        # No place scope hydrates the whole slate, in the caller's ranking.
        assert [memory.memory_id for memory in store.records.read_memories(slate)] == list(slate)

        # A place scope keeps the ranking and drops everything else, unlabelled rows included.
        assert [
            memory.memory_id for memory in store.records.read_memories(slate, place_id="kitchen")
        ] == [
            "kitchen-2",
            "kitchen-1",
        ]
        assert [
            memory.memory_id for memory in store.records.read_memories(slate, place_id="garden")
        ] == ["garden-1"]
        assert store.records.read_memories(slate, place_id="attic") == ()

        # A blank label is a caller error, not "every memory".
        with pytest.raises(ValueError, match="place_id"):
            store.records.read_memories(slate, place_id="")


def test_the_query_a_scoped_hydration_actually_runs_uses_the_place_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan claim has to be made about the statement the store issues, not one a test wrote.

    A trace callback captures the real SQL `read_memories(place_id=...)` runs, and that captured
    text is what gets planned. `(place_id, memory_id)` in that column order lets SQLite probe the
    composite on both terms at once, so a candidate that is not at the place costs one index probe
    and no table read -- and it picks that plan without `ANALYZE`, which no store has ever run.

    The callback is armed on `_open_connection` rather than on `sqlite3.connect`, and before the
    store exists rather than around the one call: the store pools its connections, so the read
    under test runs on a connection opened much earlier and a callback installed just beforehand
    would observe nothing.
    """
    statements: list[str] = []
    real_open = Connections._open_connection

    def tracing_open(store: Connections, *, secure_delete: bool = False) -> sqlite3.Connection:
        connection = real_open(store, secure_delete=secure_delete)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(Connections, "_open_connection", tracing_open)
    with LocalStore(tmp_path) as store:
        for index in range(6):
            place = "kitchen" if index == 0 else None
            store.records.write_memory(
                _memory(f"m-{index}", f"observation {index}", place_id=place)
            )
        slate = tuple(f"m-{index}" for index in range(6))
        statements.clear()
        scoped = store.records.read_memories(slate, place_id="kitchen")

        assert [memory.memory_id for memory in scoped] == ["m-0"]
        # The trace callback reports expanded SQL, so the captured statement needs no bindings
        # and carries the place predicate as a literal.
        hydrations = [
            statement
            for statement in statements
            if "FROM memory_records" in statement and "place_id = 'kitchen'" in statement
        ]
        assert len(hydrations) == 1, statements

        with closing(sqlite3.connect(store.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            plan = "\n".join(
                str(row["detail"])
                for row in connection.execute(f"EXPLAIN QUERY PLAN {hydrations[0]}")
            )

    assert "memory_records_place_idx (place_id=? AND memory_id=?)" in plan
    assert "SCAN" not in plan


def test_changing_a_place_label_requeues_the_search_index(tmp_path: Path) -> None:
    """Zvec carries `place_id` as a filter field, so relabelling a room reprojects the memory."""
    with LocalStore(tmp_path) as store:
        store.records.write_memory(_memory("relabelled", "the kettle is on", place_id="kitchen"))
        store.index.write_embedding(_embedding("relabelled"))
        store.index.acknowledge_index_operations(store.index.pending_index_operations())
        assert store.index.pending_index_operations() == ()

        store.records.write_memory(
            _memory("relabelled", "the kettle is on", place_id="utility room")
        )
        reread = store.records.read_memory("relabelled")
        assert reread is not None and reread.place_id == "utility room"
        assert [operation.embedding_id for operation in store.index.pending_index_operations()] == [
            "relabelled#0"
        ]
        store.index.acknowledge_index_operations(store.index.pending_index_operations())

        # Rewriting the same label is not a change, so it still costs nothing.
        store.records.write_memory(
            _memory("relabelled", "the kettle is on", place_id="utility room")
        )
        assert store.index.pending_index_operations() == ()
