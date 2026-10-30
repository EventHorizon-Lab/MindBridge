"""Temporal eligibility must precede a recall predicate's row bound."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge import EvidenceBasis, MemoryContext, MemoryKind
from mindbridge.infrastructure.local import LocalStore, StoredMemory
from mindbridge.infrastructure.local.store import datetime_text
from mindbridge.infrastructure.local.store._selection import read_memory_contexts


@pytest.mark.parametrize("window", [False, True])
@pytest.mark.parametrize("historical", [False, True])
def test_expired_records_do_not_hide_valid_records_behind_the_limit(
    tmp_path: Path, window: bool, historical: bool
) -> None:
    reference = datetime.now(timezone.utc)
    start = reference - timedelta(days=10)
    with LocalStore(tmp_path) as store:
        rows = tuple(
            StoredMemory(
                memory_id=f"kettle-{index}",
                content=f"Kettle delivery {index}",
                metadata_json="{}",
                occurred_at=start + timedelta(minutes=index),
                created_at=start,
                updated_at=start,
                context=MemoryContext(
                    kind=MemoryKind.OBSERVATION,
                    basis=EvidenceBasis.OBSERVATION,
                    confidence=1.0,
                    recorded_at=start,
                    valid_from=start,
                    valid_until=reference - timedelta(days=1) if index < 4 else None,
                ),
            )
            for index in range(6)
        )
        store.records.write_memories(rows)
        result = (
            store.recall.memories_in_window(
                occurred_from=start,
                occurred_until=reference,
                max_rows=3,
                valid_at=reference if historical else None,
            )
            if window
            else store.recall.match_memories(
                ("delivery",), max_rows=3, valid_at=reference if historical else None
            )
        )

        assert tuple(row.memory_id for row in result) == ("kettle-4", "kettle-5")
        assert result.selected == 2


def test_backdated_records_unknown_at_the_cutoff_cannot_fill_the_window(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        store.records.write_memories(
            tuple(
                StoredMemory(
                    memory_id=f"kettle-{index}",
                    content=f"Kettle delivery {index}",
                    metadata_json="{}",
                    occurred_at=start + timedelta(minutes=index),
                    created_at=start + timedelta(days=3) if index < 4 else start,
                    updated_at=start + timedelta(days=3) if index < 4 else start,
                )
                for index in range(6)
            )
        )
        result = store.recall.match_memories(
            ("delivery",), max_rows=3, known_at=start + timedelta(days=1)
        )

        assert tuple(row.memory_id for row in result) == ("kettle-4", "kettle-5")
        assert result.selected == 2


@pytest.mark.parametrize("active_only", [False, True])
@pytest.mark.parametrize("historical", [False, True])
def test_context_hydration_decodes_only_the_selected_version(
    tmp_path: Path, active_only: bool, historical: bool
) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(
            StoredMemory(
                memory_id="kettle",
                content="Kettle delivery",
                metadata_json="{}",
                created_at=start,
                updated_at=start,
                context=MemoryContext(
                    kind=MemoryKind.OBSERVATION,
                    basis=EvidenceBasis.OBSERVATION,
                    confidence=1 / 16,
                    recorded_at=start,
                    valid_from=None,
                    valid_until=None,
                ),
            )
        )
        with store._connections.transaction() as connection:
            connection.execute(
                "UPDATE memory_versions SET retired_at = ?",
                (datetime_text(start + timedelta(seconds=1)),),
            )
            connection.executemany(
                "INSERT INTO memory_versions (memory_id, version, confidence, "
                "recorded_at, retired_at, visible) VALUES ('kettle', ?, ?, ?, ?, 1)",
                (
                    (
                        version,
                        version / 16,
                        datetime_text(start + timedelta(seconds=version - 1)),
                        None
                        if version == 16
                        else datetime_text(start + timedelta(seconds=version)),
                    )
                    for version in range(2, 17)
                ),
            )
        decoded = 0

        def count_rows(cursor: sqlite3.Cursor, values: tuple[object, ...]) -> sqlite3.Row:
            nonlocal decoded
            if cursor.description[0][0] == "semantic_memory_id":
                decoded += 1
            return sqlite3.Row(cursor, values)

        with store._connections.read_transaction() as connection:
            original = connection.row_factory
            connection.row_factory = count_rows
            try:
                contexts, semantic_ids = read_memory_contexts(
                    connection,
                    ("kettle",),
                    active_only=active_only,
                    known_at=start + timedelta(seconds=7) if historical else None,
                )
            finally:
                connection.row_factory = original

        assert decoded == 1
        assert semantic_ids == {"kettle"}
        assert contexts["kettle"].confidence == (0.5 if historical else 1.0)
        assert contexts["kettle"].recorded_at == start + timedelta(seconds=7 if historical else 15)


@pytest.mark.parametrize("exclusion", ["expired", "future", "unknown", "hidden", "retired"])
def test_a_typed_record_without_an_eligible_version_cannot_become_untyped(
    tmp_path: Path, exclusion: str
) -> None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=2)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(
            StoredMemory(
                memory_id="kettle",
                content="A typed observation",
                metadata_json="{}",
                created_at=start,
                updated_at=start,
                context=MemoryContext(
                    kind=MemoryKind.OBSERVATION,
                    basis=EvidenceBasis.OBSERVATION,
                    confidence=0.5,
                    recorded_at=now + timedelta(days=1) if exclusion == "unknown" else start,
                    valid_from=now + timedelta(days=1) if exclusion == "future" else start,
                    valid_until=now - timedelta(days=1) if exclusion == "expired" else None,
                ),
            )
        )
        if exclusion in {"hidden", "retired"}:
            with store._connections.transaction() as connection:
                if exclusion == "hidden":
                    connection.execute("UPDATE memory_versions SET visible = 0")
                else:
                    connection.execute(
                        "UPDATE memory_versions SET retired_at = ?", (datetime_text(now),)
                    )

        assert (
            store.records.read_memories(("kettle",), active_only=True, valid_at=now, known_at=now)
            == ()
        )
        stored = store.records.read_memory("kettle")
        assert stored is not None and stored.context is not None
