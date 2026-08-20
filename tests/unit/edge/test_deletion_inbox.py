"""Edge tombstone durability and local evidence erasure checks."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mindbridge.contracts import (
    DeletionPage,
    DeletionTombstoneView,
    MediaObjectInput,
    ObserveRequest,
)
from mindbridge.core import (
    DeletionPropagationState,
    ForgetTargetType,
    IdentityKind,
    MediaKind,
    MemoryDeletedError,
    ModelReference,
    SensorKind,
    derive_observation_id,
)
from mindbridge.edge import (
    EdgeMediaFile,
    FaceVoiceAssociationEvidence,
    LocalIdentitySample,
    SQLiteDeletionInbox,
    SQLiteIdentityMemory,
    SQLiteObservationOutbox,
)

NOW = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)


def test_deletion_inbox_requires_private_file_backed_storage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="file-backed"):
        SQLiteDeletionInbox(Path(":memory:"))

    database_path = tmp_path / "nested" / "edge.db"
    SQLiteDeletionInbox(database_path)

    assert database_path.stat().st_mode & 0o777 == 0o600


def test_tombstone_erasure_rolls_back_until_local_media_can_be_deleted(
    tmp_path: Path,
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")
    database_path = tmp_path / "edge.db"
    outbox = SQLiteObservationOutbox(database_path, clock=lambda: NOW)
    inbox = SQLiteDeletionInbox(database_path, clock=lambda: NOW)
    request = _request()
    observation_id = derive_observation_id(
        request.tenant_id,
        request.device_id,
        request.boot_id,
        request.sequence,
    )
    identity_memory = SQLiteIdentityMemory(
        database_path,
        device_id=request.device_id,
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    identity = identity_memory.recognize_and_remember(
        LocalIdentitySample(
            tenant_id=request.tenant_id,
            kind=IdentityKind.FACE,
            source_observation_id=observation_id,
            sample_id="face_track_01",
            embedding=(1.0, 0.0),
            model_reference=ModelReference(model_id="insightface/buffalo_l"),
        ),
        minimum_similarity=0.8,
    )
    voice = identity_memory.recognize_and_remember(
        LocalIdentitySample(
            tenant_id=request.tenant_id,
            kind=IdentityKind.VOICE,
            source_observation_id="voice_observation",
            sample_id="voice_track_01",
            embedding=(1.0, 0.0),
            model_reference=ModelReference(model_id="3d-speaker/eres2netv2"),
        ),
        minimum_similarity=0.8,
    )
    association_model = ModelReference(model_id="lr-asd")
    identity_memory.record_face_voice_evidence(
        FaceVoiceAssociationEvidence(
            tenant_id=request.tenant_id,
            source_observation_id="association_observation",
            evidence_id="asd_01",
            face_identity_id=identity.identity_id,
            voice_identity_id=voice.identity_id,
            start_ms=0,
            end_ms=1_000,
            confidence=0.95,
            model_reference=association_model,
        )
    )
    assert (
        identity_memory.resolve_identity(
            request.tenant_id,
            voice,
            association_model_reference=association_model,
            minimum_observations=1,
            minimum_duration_ms=500,
            minimum_confidence=0.9,
            minimum_margin=0.0,
        ).identity_id
        == identity.identity_id
    )
    media_files = (
        EdgeMediaFile(
            media_object_id="media_01",
            local_path=media_path,
            content_type="video/mp4",
        ),
    )
    outbox.enqueue(request, media_files)
    page = _deletion_page(request)

    media_path.unlink()
    media_path.mkdir()
    with pytest.raises(IsADirectoryError):
        inbox.apply_page(request.tenant_id, page)
    assert inbox.read_cursor(request.tenant_id) is None
    assert outbox.pending_count() == 1

    media_path.rmdir()
    media_path.write_bytes(b"video")
    assert inbox.apply_page(request.tenant_id, page) == 1
    assert inbox.apply_page(request.tenant_id, page) == 1
    assert inbox.read_cursor(request.tenant_id) == "tombstone_01"
    assert outbox.pending_count() == 0
    assert (
        identity_memory.resolve_identity(
            request.tenant_id,
            voice,
            association_model_reference=association_model,
            minimum_observations=1,
            minimum_duration_ms=500,
            minimum_confidence=0.9,
            minimum_margin=0.0,
        ).identity_id
        == voice.identity_id
    )
    assert identity_memory.forget_identity(request.tenant_id, identity.identity_id) == 0
    assert not media_path.exists()
    with pytest.raises(MemoryDeletedError):
        outbox.enqueue(request, media_files)


def _request() -> ObserveRequest:
    return ObserveRequest(
        tenant_id="tenant_01",
        device_id="camera_01",
        boot_id="boot_01",
        sequence=7,
        sensor=SensorKind.CAMERA,
        media_objects=(
            MediaObjectInput(
                media_object_id="media_01",
                kind=MediaKind.VIDEO,
                uri="s3://memory/tenants/tenant_01/media_01.mp4",
                sha256="00" * 32,
                size_bytes=5,
                created_at=NOW,
                duration_ms=30_000,
            ),
        ),
        occurred_at=NOW,
        ended_at=NOW + timedelta(seconds=30),
        observed_at=NOW + timedelta(seconds=30),
    )


def _deletion_page(request: ObserveRequest) -> DeletionPage:
    observation_id = derive_observation_id(
        request.tenant_id,
        request.device_id,
        request.boot_id,
        request.sequence,
    )
    return DeletionPage(
        items=(
            DeletionTombstoneView(
                tombstone_id="tombstone_01",
                target_type=ForgetTargetType.OBSERVATION,
                target_id=observation_id,
                propagation_state=DeletionPropagationState.COMPLETE,
                requested_at=NOW,
                completed_at=NOW,
                error_code=None,
            ),
        ),
        next_cursor=None,
        trace_id="trace_deletion_page",
    )
