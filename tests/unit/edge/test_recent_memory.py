"""Offline recent-memory cache checks for edge devices."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mindbridge.contracts import (
    DeletionPage,
    DeletionTombstoneView,
    EvidenceView,
    MemoryResult,
    ObservationReceipt,
    ObservationStatus,
)
from mindbridge.core import (
    DeletionPropagationState,
    ForgetTargetType,
    MemoryIntegrityError,
    MemoryState,
    MemoryType,
    VerificationStatus,
    derive_observation_id,
)
from mindbridge.edge import (
    SQLiteDeletionInbox,
    SQLiteObservationOutbox,
    SQLiteRecentMemory,
    enqueue_captured_media,
)

NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)


def test_recent_memory_survives_restart_uses_local_evidence_and_expires(
    tmp_path: Path,
) -> None:
    current_time = [NOW]

    def clock() -> datetime:
        return current_time[0]

    database_path, media_path, outbox, receipt, memory = _queued_memory(tmp_path, clock=clock)
    recent = SQLiteRecentMemory(
        database_path,
        retention=timedelta(hours=1),
        clock=clock,
    )

    with pytest.raises(MemoryIntegrityError, match="do not match"):
        recent.cache_job_memories(
            "tenant_01",
            receipt.observation_id,
            receipt.processing_job_id,
            ("memory_wrong",),
            (memory,),
        )
    assert len(outbox.pending_processing_jobs()) == 1

    recent.cache_job_memories(
        "tenant_01",
        receipt.observation_id,
        receipt.processing_job_id,
        (memory.memory_id,),
        (memory,),
    )

    assert outbox.pending_processing_jobs() == ()
    cached = SQLiteRecentMemory(database_path, clock=clock).get_memory(
        "tenant_01", memory.memory_id
    )
    assert cached is not None
    assert cached.evidence[0].media_url == media_path.as_uri()
    assert cached.evidence[0].media_url_expires_at == NOW + timedelta(hours=1)

    current_time[0] = NOW + timedelta(hours=1, seconds=1)
    assert recent.list_memories("tenant_01") == ()


def test_tombstones_remove_recent_memory_and_pending_jobs(tmp_path: Path) -> None:
    database_path, media_path, outbox, receipt, memory = _queued_memory(tmp_path)
    recent = SQLiteRecentMemory(database_path, clock=lambda: NOW)
    recent.cache_job_memories(
        "tenant_01",
        receipt.observation_id,
        receipt.processing_job_id,
        (memory.memory_id,),
        (memory,),
    )
    inbox = SQLiteDeletionInbox(database_path, clock=lambda: NOW)

    inbox.apply_page(
        "tenant_01",
        _deletion_page(ForgetTargetType.MEMORY_RECORD, memory.memory_id, "tombstone_01"),
    )

    assert recent.get_memory("tenant_01", memory.memory_id) is None
    assert media_path.exists()

    _, observation_media, _, observation_receipt, observation_memory = _queued_memory(
        tmp_path,
        sequence=8,
    )
    recent.cache_job_memories(
        "tenant_01",
        observation_receipt.observation_id,
        observation_receipt.processing_job_id,
        (observation_memory.memory_id,),
        (observation_memory,),
    )

    inbox.apply_page(
        "tenant_01",
        _deletion_page(
            ForgetTargetType.OBSERVATION,
            observation_receipt.observation_id,
            "tombstone_02",
        ),
    )

    assert recent.get_memory("tenant_01", observation_memory.memory_id) is None
    assert not observation_media.exists()

    _, pending_media, _, pending_receipt, _ = _queued_memory(tmp_path, sequence=9)
    assert len(outbox.pending_processing_jobs()) == 1

    inbox.apply_page(
        "tenant_01",
        _deletion_page(
            ForgetTargetType.OBSERVATION,
            pending_receipt.observation_id,
            "tombstone_03",
        ),
    )

    assert outbox.pending_processing_jobs() == ()
    assert not pending_media.exists()


def _queued_memory(
    tmp_path: Path,
    *,
    sequence: int = 7,
    clock: Callable[[], datetime] | None = None,
) -> tuple[Path, Path, SQLiteObservationOutbox, ObservationReceipt, MemoryResult]:
    database_path = tmp_path / "edge.db"
    media_path = tmp_path / f"clip_{sequence}.mp4"
    media_path.write_bytes(b"video")
    effective_clock = clock or (lambda: NOW)
    outbox = SQLiteObservationOutbox(database_path, clock=effective_clock)
    request = enqueue_captured_media(
        outbox,
        media_path,
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=sequence,
        bucket="memory",
        occurred_at=NOW + timedelta(minutes=sequence),
        ended_at=NOW + timedelta(minutes=sequence, seconds=30),
        observed_at=NOW + timedelta(minutes=sequence, seconds=30),
    )
    item = outbox.next_pending()
    assert item is not None
    observation_id = derive_observation_id("tenant_01", "camera_01", "boot_01", sequence)
    receipt = ObservationReceipt(
        observation_id=observation_id,
        processing_job_id=f"job_process_{observation_id}",
        idempotency_key=request.idempotency_key or "unexpected",
        status=ObservationStatus.ACCEPTED,
        trace_id=f"trace_observe_{sequence}",
    )
    outbox.acknowledge(item, receipt)
    memory = MemoryResult(
        memory_id=f"memory_{sequence}",
        memory_type=MemoryType.EPISODIC,
        summary="A person placed a red tool beside the toolbox.",
        evidence_ids=(f"evidence_{sequence}",),
        occurred_at=request.occurred_at,
        ended_at=request.ended_at,
        created_at=NOW,
        verification_status=VerificationStatus.VERIFIED,
        state=MemoryState.ACTIVE,
        evidence=(
            EvidenceView(
                evidence_id=f"evidence_{sequence}",
                media_object_id=request.media_objects[0].media_object_id,
                start_ms=0,
                end_ms=30_000,
                media_url="https://objects.example.test/signed.mp4",
                media_url_expires_at=NOW + timedelta(minutes=5),
            ),
        ),
        trace_id=f"trace_memory_{sequence}",
    )
    return database_path, media_path, outbox, receipt, memory


def _deletion_page(
    target_type: ForgetTargetType,
    target_id: str,
    tombstone_id: str,
) -> DeletionPage:
    return DeletionPage(
        items=(
            DeletionTombstoneView(
                tombstone_id=tombstone_id,
                target_type=target_type,
                target_id=target_id,
                propagation_state=DeletionPropagationState.COMPLETE,
                requested_at=NOW,
                completed_at=NOW,
                error_code=None,
            ),
        ),
        next_cursor=None,
        trace_id=f"trace_{tombstone_id}",
    )
