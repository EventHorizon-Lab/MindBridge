"""The symbolic place axis: one nullable `place_id` and an indexed equality over it.

MindBridge already stores metric pose -- a frame id, a position, a quaternion and an uncertainty --
and already scopes by radius. What it cannot express is the predicate a household query actually
uses ("in the kitchen"), which is also the only spatial label a robot can supply when it cannot
localise metrically. That is a symbolic equality, and an equality is the one spatial predicate
SQLite indexes cheaply.

These tests pin the four things that make it worth having rather than the metric version: the
value round-trips and stays optional, a scoped hydration filters in SQL and is planned onto
`memory_records_place_idx` rather than degrading into the post-hoc Python filter the metric radius
scope is forced to be, relabelling a room costs no reindex, and an existing store gains the column
without losing a memory.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredEmbedding, StoredMemory
from mindbridge.infrastructure.local.store import _SCHEMA_VERSION

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
        store.write_memory(
            _memory("labelled", "the blue inhaler is in the top drawer", place_id="kitchen")
        )
        store.write_memory(_memory("unlabelled", "someone mentioned Thursday"))

        labelled = store.read_memory("labelled")
        unlabelled = store.read_memory("unlabelled")
        assert labelled is not None and labelled.place_id == "kitchen"
        assert unlabelled is not None and unlabelled.place_id is None

        # The batch read and the listing hydrate the same column, not just the single read.
        assert [memory.place_id for memory in store.read_memories(("labelled", "unlabelled"))] == [
            "kitchen",
            None,
        ]
        assert {memory.memory_id: memory.place_id for memory in store.list_memories()} == {
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
        store.write_memory(_memory("kitchen-1", "the kettle is on", place_id="kitchen"))
        store.write_memory(_memory("kitchen-2", "the top drawer is open", place_id="kitchen"))
        store.write_memory(_memory("garden-1", "the hose is coiled", place_id="garden"))
        store.write_memory(_memory("nowhere", "someone said Thursday"))
        slate = ("garden-1", "kitchen-2", "nowhere", "kitchen-1")

        # No place scope hydrates the whole slate, in the caller's ranking.
        assert [memory.memory_id for memory in store.read_memories(slate)] == list(slate)

        # A place scope keeps the ranking and drops everything else, unlabelled rows included.
        assert [memory.memory_id for memory in store.read_memories(slate, place_id="kitchen")] == [
            "kitchen-2",
            "kitchen-1",
        ]
        assert [memory.memory_id for memory in store.read_memories(slate, place_id="garden")] == [
            "garden-1"
        ]
        assert store.read_memories(slate, place_id="attic") == ()

        # A blank label is a caller error, not "every memory".
        with pytest.raises(ValueError, match="place_id"):
            store.read_memories(slate, place_id="")


def test_the_query_a_scoped_hydration_actually_runs_uses_the_place_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plan claim has to be made about the statement the store issues, not one a test wrote.

    A trace callback captures the real SQL `read_memories(place_id=...)` runs, and that captured
    text is what gets planned. `(place_id, memory_id)` in that column order lets SQLite probe the
    composite on both terms at once, so a candidate that is not at the place costs one index probe
    and no table read -- and it picks that plan without `ANALYZE`, which no store has ever run.
    """
    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(
        database: str | Path,
        *,
        timeout: float = 5.0,
        isolation_level: Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None = None,
    ) -> sqlite3.Connection:
        connection = real_connect(database, timeout=timeout, isolation_level=isolation_level)
        connection.set_trace_callback(statements.append)
        return connection

    with LocalStore(tmp_path) as store:
        for index in range(6):
            place = "kitchen" if index == 0 else None
            store.write_memory(_memory(f"m-{index}", f"observation {index}", place_id=place))
        slate = tuple(f"m-{index}" for index in range(6))
        monkeypatch.setattr(sqlite3, "connect", tracing_connect)
        scoped = store.read_memories(slate, place_id="kitchen")
        monkeypatch.setattr(sqlite3, "connect", real_connect)

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


def test_changing_only_a_place_label_does_not_requeue_the_search_index(tmp_path: Path) -> None:
    """Zvec carries no place, so relabelling a room must not cost a reindex of that memory."""
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory("relabelled", "the kettle is on", place_id="kitchen"))
        store.write_embedding(_embedding("relabelled"))
        store.acknowledge_index_operations(store.pending_index_operations())
        assert store.pending_index_operations() == ()

        store.write_memory(_memory("relabelled", "the kettle is on", place_id="utility room"))
        reread = store.read_memory("relabelled")
        assert reread is not None and reread.place_id == "utility room"
        assert store.pending_index_operations() == ()

        # Content still is indexed text, so changing that must still requeue.
        store.write_memory(_memory("relabelled", "the kettle is off", place_id="utility room"))
        assert [operation.embedding_id for operation in store.pending_index_operations()] == [
            "relabelled#0"
        ]


@pytest.mark.skipif(sqlite3.sqlite_version_info < (3, 35), reason="DROP COLUMN needs SQLite 3.35")
def test_a_store_without_place_id_gains_the_column_and_keeps_its_memories(
    tmp_path: Path,
) -> None:
    """The migration is additive: an existing store opens, keeps every row, and gains the axis."""
    with LocalStore(tmp_path) as store:
        store.write_memory(_memory("preserved", "A red tool is in drawer two", place_id="workshop"))
    with closing(sqlite3.connect(tmp_path / "state.sqlite3")) as connection:
        connection.executescript(
            """
            DROP INDEX memory_records_place_idx;
            ALTER TABLE memory_records DROP COLUMN place_id;
            PRAGMA user_version = 9;
            """
        )

    with LocalStore(tmp_path) as store:
        preserved = store.read_memory("preserved")
        store.write_memory(_memory("after", "the kettle is on", place_id="kitchen"))
        after = store.read_memories(("preserved", "after"), place_id="kitchen")
        with closing(sqlite3.connect(store.database_path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(memory_records)")}
            indexes = {row[1] for row in connection.execute("PRAGMA index_list(memory_records)")}
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            migrated_plan = "\n".join(
                str(row[3])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN "
                    "SELECT content FROM memory_records WHERE memory_id IN (?, ?) AND place_id = ?",
                    ("preserved", "after", "kitchen"),
                )
            )

    assert preserved is not None and preserved.content == "A red tool is in drawer two"
    # The dropped label is gone with the dropped column; the axis, not the value, is restored.
    assert preserved.place_id is None
    assert "place_id" in columns
    assert version == _SCHEMA_VERSION
    # The restored index is usable straight away, on rows written after the migration. Without
    # this the column alone would come back and every upgraded store would silently scan.
    assert [memory.memory_id for memory in after] == ["after"]
    assert "memory_records_place_idx" in indexes
    assert "memory_records_place_idx (place_id=? AND memory_id=?)" in migrated_plan
