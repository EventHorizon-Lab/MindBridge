"""Checks for encrypted, bounded, and forgettable device identity learning."""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from mindbridge.core import (
    IdempotencyConflictError,
    IdentityKind,
    MemoryIntegrityError,
    ModelReference,
)
from mindbridge.edge import (
    FaceVoiceAssociationEvidence,
    LocalIdentityMatch,
    LocalIdentitySample,
    SQLiteIdentityMemory,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
MODEL = ModelReference(model_id="insightface/buffalo_l")
VOICE_MODEL = ModelReference(model_id="3d-speaker/eres2netv2")
ASSOCIATION_MODEL = ModelReference(model_id="lr-asd")


def test_identity_memory_matches_learns_and_encrypts_samples(tmp_path: Path) -> None:
    database_path = tmp_path / "edge.sqlite3"
    key = AESGCM.generate_key(bit_length=256)
    memory = SQLiteIdentityMemory(
        database_path,
        device_id="robot_01",
        encryption_key=key,
        maximum_samples_per_identity=2,
        clock=lambda: NOW,
    )

    first = memory.recognize_and_remember(_sample("sample_01", (1.0, 0.0)), minimum_similarity=0.8)
    second_sample = _sample("sample_02", (0.99, 0.01))
    second = memory.recognize_and_remember(second_sample, minimum_similarity=0.8)
    third = memory.recognize_and_remember(_sample("sample_03", (1.0, 0.0)), minimum_similarity=0.8)
    retry = memory.recognize_and_remember(second_sample, minimum_similarity=0.8)
    stranger = memory.recognize_and_remember(
        _sample("sample_04", (0.0, 1.0)), minimum_similarity=0.8
    )

    assert first.enrolled_new
    assert second.identity_id == first.identity_id
    assert third.identity_id == first.identity_id
    assert retry.identity_id == first.identity_id
    assert not second.enrolled_new
    assert stranger.identity_id != first.identity_id
    cloud_identity = second.to_observation_input(
        start_ms=100,
        end_ms=900,
        visual_bbox_xyxy=(0.1, 0.2, 0.4, 0.8),
    )
    assert cloud_identity.identity_id == first.identity_id
    assert cloud_identity.visual_bbox_xyxy == (0.1, 0.2, 0.4, 0.8)
    assert "embedding" not in cloud_identity.model_dump()
    assert database_path.stat().st_mode & 0o777 == 0o600
    with closing(sqlite3.connect(database_path)) as connection, connection:
        retained = connection.execute(
            "SELECT encrypted_embedding FROM edge_identity_templates WHERE identity_id = ?",
            (first.identity_id,),
        ).fetchall()
    assert len(retained) == 2
    assert all(b"embedding" not in row[0] for row in retained)


def test_identity_memory_refuses_an_ambiguous_biometric_merge(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    # 0.3 apart: two people, not one face twice. Both then score just over the 0.78 gate
    # for the third sample and land inside the margin, so neither one clearly wins it.
    first = memory.recognize_and_remember(_sample("first", (1.0, 0.0, 0.0)), minimum_similarity=0.9)
    second = memory.recognize_and_remember(
        _sample("second", (0.3, 0.953939, 0.0)), minimum_similarity=0.9
    )

    ambiguous = memory.recognize_and_remember(
        _sample("ambiguous", (0.80, 0.5766, 0.1658)),
        minimum_similarity=0.78,
        minimum_margin=0.02,
    )

    assert first.identity_id != second.identity_id
    assert ambiguous.enrolled_new
    assert ambiguous.identity_id not in {first.identity_id, second.identity_id}


def test_identity_memory_binds_a_sample_tied_between_two_views_of_one_face(
    tmp_path: Path,
) -> None:
    """A tie between fragments of one identity must not mint a third fragment."""
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    # 0.99 apart: one face from two angles, split only because enrolment ran strict.
    first = memory.recognize_and_remember(_sample("front", (1.0, 0.0)), minimum_similarity=0.995)
    second = memory.recognize_and_remember(
        _sample("angled", (0.99, 0.1411)), minimum_similarity=0.995
    )

    between = memory.recognize_and_remember(
        _sample("between", (0.99749, 0.07073)),
        minimum_similarity=0.9,
        minimum_margin=0.02,
    )

    assert first.identity_id != second.identity_id
    assert not between.enrolled_new
    assert between.identity_id in {first.identity_id, second.identity_id}


def test_identity_samples_are_idempotent_and_bound_to_the_device_key(tmp_path: Path) -> None:
    database_path = tmp_path / "edge.sqlite3"
    memory = SQLiteIdentityMemory(
        database_path,
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    sample = _sample("sample_01", (1.0, 0.0))
    memory.recognize_and_remember(sample, minimum_similarity=0.8)

    with pytest.raises(IdempotencyConflictError):
        memory.recognize_and_remember(_sample("sample_01", (0.0, 1.0)), minimum_similarity=0.8)
    with pytest.raises(IdempotencyConflictError):
        memory.recognize_and_remember(
            LocalIdentitySample(
                tenant_id="tenant_01",
                kind=IdentityKind.FACE,
                source_observation_id="observation_02",
                sample_id="sample_01",
                embedding=(1.0, 0.0),
                model_reference=MODEL,
            ),
            minimum_similarity=0.8,
        )
    wrong_key_memory = SQLiteIdentityMemory(
        database_path,
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    with pytest.raises(MemoryIntegrityError, match="cannot be decrypted"):
        wrong_key_memory.recognize_and_remember(
            _sample("sample_02", (1.0, 0.0)), minimum_similarity=0.8
        )


def test_identity_memory_forgets_observation_samples(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    match = memory.recognize_and_remember(_sample("sample_01", (1.0, 0.0)), minimum_similarity=0.8)

    assert memory.forget_observation("tenant_01", "observation_01") == 1
    assert memory.forget_identity("tenant_01", match.identity_id) == 0


def test_face_voice_association_requires_repeatable_unambiguous_evidence(tmp_path: Path) -> None:
    memory = SQLiteIdentityMemory(
        tmp_path / "edge.sqlite3",
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    face = memory.recognize_and_remember(
        _sample("face_01", (1.0, 0.0), observation_id="face_enrollment"),
        minimum_similarity=0.8,
    )
    competing_face = memory.recognize_and_remember(
        _sample("face_02", (0.0, 1.0), observation_id="other_face_enrollment"),
        minimum_similarity=0.8,
    )
    voice = memory.recognize_and_remember(
        _sample(
            "voice_01",
            (1.0, 0.0),
            kind=IdentityKind.VOICE,
            observation_id="voice_enrollment",
            model_reference=VOICE_MODEL,
        ),
        minimum_similarity=0.8,
    )
    first = _association_evidence("asd_01", "observation_01", face.identity_id, voice.identity_id)

    memory.record_face_voice_evidence(first)
    assert _resolve(memory, voice).identity_id == voice.identity_id
    memory.record_face_voice_evidence(first)
    with pytest.raises(IdempotencyConflictError):
        memory.record_face_voice_evidence(
            _association_evidence(
                "asd_01",
                "observation_01",
                face.identity_id,
                voice.identity_id,
                confidence=0.91,
            ),
        )

    memory.record_face_voice_evidence(
        _association_evidence("asd_02", "observation_02", face.identity_id, voice.identity_id),
    )

    resolved = _resolve(memory, voice)
    assert resolved.identity_id == face.identity_id
    assert resolved.to_observation_input(start_ms=0, end_ms=1_000).kind is IdentityKind.VOICE

    memory.record_face_voice_evidence(
        _association_evidence(
            "asd_03",
            "observation_03",
            competing_face.identity_id,
            voice.identity_id,
            confidence=0.93,
        ),
    )
    memory.record_face_voice_evidence(
        _association_evidence(
            "asd_04",
            "observation_04",
            competing_face.identity_id,
            voice.identity_id,
            confidence=0.93,
        ),
    )

    assert _resolve(memory, voice).identity_id == voice.identity_id
    memory.record_face_voice_evidence(
        _association_evidence(
            "asd_tie",
            "observation_tie",
            competing_face.identity_id,
            voice.identity_id,
            confidence=1.0,
            duration_ms=40,
        ),
    )
    assert _resolve(memory, voice, minimum_margin=0.0).identity_id == voice.identity_id
    memory.forget_observation("tenant_01", "observation_03")
    memory.forget_observation("tenant_01", "observation_04")
    assert _resolve(memory, voice).identity_id == face.identity_id
    memory.forget_observation("tenant_01", "observation_02")
    assert _resolve(memory, voice).identity_id == voice.identity_id
    memory.record_face_voice_evidence(
        _association_evidence("asd_05", "observation_05", face.identity_id, voice.identity_id),
    )
    assert _resolve(memory, voice).identity_id == face.identity_id
    assert memory.forget_observation("tenant_01", "face_enrollment") == 1
    assert _resolve(memory, voice).identity_id == voice.identity_id


def test_a_device_upgraded_from_a_revision_keyed_store_can_still_learn(tmp_path: Path) -> None:
    """A device carrying the old schema must not meet a NOT NULL column that no writer fills."""
    database_path = tmp_path / "edge.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE edge_identity_templates (
                tenant_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                identity_id TEXT NOT NULL,
                source_observation_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                nonce BLOB NOT NULL,
                encrypted_embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    tenant_id, device_id, kind, model_id, model_revision, sample_id
                )
            );
            CREATE INDEX edge_identity_match_idx
                ON edge_identity_templates (
                    tenant_id, device_id, kind, model_id, model_revision, dimension, identity_id
                );
            CREATE TABLE edge_face_voice_evidence (
                tenant_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                association_model_id TEXT NOT NULL,
                association_model_revision TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                source_observation_id TEXT NOT NULL,
                face_identity_id TEXT NOT NULL,
                voice_identity_id TEXT NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (
                    tenant_id, device_id, association_model_id,
                    association_model_revision, evidence_id
                )
            );
            """
        )

    memory = SQLiteIdentityMemory(
        database_path,
        device_id="robot_01",
        encryption_key=AESGCM.generate_key(bit_length=256),
        clock=lambda: NOW,
    )
    match = memory.recognize_and_remember(_sample("sample_01", (1.0, 0.0)), minimum_similarity=0.8)

    assert match.enrolled_new is True
    with sqlite3.connect(database_path) as connection:
        for table in ("edge_identity_templates", "edge_face_voice_evidence"):
            columns = {
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert not {name for name in columns if "revision" in name}


def _sample(
    sample_id: str,
    embedding: tuple[float, ...],
    *,
    kind: IdentityKind = IdentityKind.FACE,
    observation_id: str = "observation_01",
    model_reference: ModelReference = MODEL,
) -> LocalIdentitySample:
    return LocalIdentitySample(
        tenant_id="tenant_01",
        kind=kind,
        source_observation_id=observation_id,
        sample_id=sample_id,
        embedding=embedding,
        model_reference=model_reference,
    )


def _association_evidence(
    evidence_id: str,
    observation_id: str,
    face_identity_id: str,
    voice_identity_id: str,
    *,
    confidence: float = 0.95,
    duration_ms: int = 1_000,
) -> FaceVoiceAssociationEvidence:
    return FaceVoiceAssociationEvidence(
        tenant_id="tenant_01",
        source_observation_id=observation_id,
        evidence_id=evidence_id,
        face_identity_id=face_identity_id,
        voice_identity_id=voice_identity_id,
        start_ms=0,
        end_ms=duration_ms,
        confidence=confidence,
        model_reference=ASSOCIATION_MODEL,
    )


def _resolve(
    memory: SQLiteIdentityMemory,
    match: LocalIdentityMatch,
    *,
    minimum_margin: float = 0.1,
) -> LocalIdentityMatch:
    return memory.resolve_identity(
        "tenant_01",
        match,
        association_model_reference=ASSOCIATION_MODEL,
        minimum_observations=2,
        minimum_duration_ms=1_500,
        minimum_confidence=0.9,
        minimum_margin=minimum_margin,
    )
