"""Temporal eligibility must be applied before the bounded background window."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mindbridge.infrastructure.local import LocalStore, StoredMemory
from mindbridge.types import EvidenceBasis, MemoryContext, MemoryKind


def test_unknown_semantic_version_cannot_hide_an_older_eligible_history(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    known_before = start + timedelta(days=2)
    with LocalStore(tmp_path) as store:
        store.records.write_memory(
            StoredMemory(
                memory_id="older",
                content="An earlier observation.",
                metadata_json="{}",
                created_at=start,
                updated_at=start,
            )
        )
        store.records.write_memory(
            StoredMemory(
                memory_id="later-claim",
                content="A claim not yet recorded at the cutoff.",
                metadata_json="{}",
                created_at=start + timedelta(days=1),
                updated_at=start + timedelta(days=3),
                context=MemoryContext(
                    kind=MemoryKind.EVENT,
                    basis=EvidenceBasis.MODEL_INFERENCE,
                    confidence=0.8,
                    valid_from=None,
                    valid_until=None,
                    recorded_at=start + timedelta(days=3),
                ),
            )
        )
        background = store.records.formation_history(
            known_before=known_before, exclude_ids=(), limit=1
        )
        assert tuple(record.memory_id for record in background) == ("older",)
        assert background[0].context is None
