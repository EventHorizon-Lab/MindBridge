"""Checks for encrypted, bounded, and forgettable device identity learning."""

import sqlite3
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
from mindbridge.edge import LocalIdentitySample, SQLiteIdentityMemory

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)
MODEL = ModelReference(model_id="insightface/buffalo_l", revision="1.0.1")


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
    cloud_identity = second.to_observation_input(start_ms=100, end_ms=900)
    assert cloud_identity.identity_id == first.identity_id
    assert "embedding" not in cloud_identity.model_dump()
    assert database_path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(database_path) as connection:
        retained = connection.execute(
            "SELECT encrypted_embedding FROM edge_identity_templates WHERE identity_id = ?",
            (first.identity_id,),
        ).fetchall()
    assert len(retained) == 2
    assert all(b"embedding" not in row[0] for row in retained)


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


def _sample(sample_id: str, embedding: tuple[float, ...]) -> LocalIdentitySample:
    return LocalIdentitySample(
        tenant_id="tenant_01",
        kind=IdentityKind.FACE,
        source_observation_id="observation_01",
        sample_id=sample_id,
        embedding=embedding,
        model_reference=MODEL,
    )
