"""Proof ancestry and materialized evidence must share a read snapshot."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

import mindbridge.infrastructure.local.store.semantics as semantics_module
from mindbridge.infrastructure.local import LocalStore, StoredMemory
from mindbridge.infrastructure.local.store._corroboration import _load_nodes
from mindbridge.kernel.corroboration import EvidenceNode


def test_delivery_evidence_keeps_snapshot_during_concurrent_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 10, 8, tzinfo=timezone.utc)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(
            StoredMemory(
                memory_id="root",
                content="Original supporting observation.",
                metadata_json="{}",
                created_at=now,
                updated_at=now,
            )
        )

        def load_then_delete(
            connection: sqlite3.Connection, memory_id: str
        ) -> tuple[dict[str, EvidenceNode], bool]:
            nodes, truncated = _load_nodes(connection, memory_id)
            assert store.records.delete_memory(memory_id)
            return nodes, truncated

        monkeypatch.setattr(semantics_module, "_load_nodes", load_then_delete)
        nodes, records, truncated = store.semantics.delivery_evidence("root")
        assert tuple(nodes) == ("root",)
        assert not truncated
        assert tuple(record.content for record in records) == ("Original supporting observation.",)
        assert store.records.read_memory("root") is None
