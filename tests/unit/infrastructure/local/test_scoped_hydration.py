"""SQL-filtered records must not incur asset or semantic hydration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mindbridge.infrastructure.local import LocalStore, StoredAsset, StoredMemory


@pytest.mark.parametrize("place_id", (None, "selected", "absent"))
def test_scoped_hydration_reads_related_rows_only_for_selected_memories(
    tmp_path: Path, place_id: str | None
) -> None:
    now = datetime(2026, 10, 27, tzinfo=timezone.utc)
    memories = tuple(
        StoredMemory(
            memory_id=f"memory-{index}",
            content=f"source {index}",
            metadata_json="{}",
            created_at=now,
            updated_at=now,
            place_id="selected" if index == 0 else "excluded",
            modality="image",
            assets=(
                StoredAsset(
                    asset_id=f"{index:064x}",
                    modality="image",
                    mime_type="image/png",
                    size_bytes=1,
                    sha256=f"{index:064x}",
                    relative_path=f"assets/00/{index:064x}",
                    created_at=now,
                ),
            ),
        )
        for index in range(4)
    )
    statements: list[str] = []
    with LocalStore(tmp_path) as store:
        store.records.write_memories(memories)
        with store._connections.connection() as connection:
            connection.set_trace_callback(statements.append)
        result = store.records.read_memories(
            (*tuple(memory.memory_id for memory in reversed(memories)), "missing"),
            place_id=place_id,
        )
        expected = (
            tuple(reversed(memories))
            if place_id is None
            else (memories[0],)
            if place_id == "selected"
            else ()
        )
        assert result == expected
        related_reads = [
            statement
            for statement in statements
            if "FROM memory_assets AS ma" in statement or "FROM memory_semantics" in statement
        ]
        assert all("missing" not in statement for statement in related_reads)
        if place_id is not None:
            for memory_id in ("memory-1", "memory-2", "memory-3"):
                assert all(memory_id not in statement for statement in related_reads)
        if place_id == "absent":
            assert related_reads == []
