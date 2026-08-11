"""Encrypted device-local face and voice identity memory."""

from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypeAlias, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    IdempotencyConflictError,
    IdentityKind,
    MemoryIntegrityError,
    ModelReference,
    derive_stable_id,
)
from mindbridge.edge.identity_schema import initialize_identity_tables

_NONCE_BYTES = 12
_AES_256_KEY_BYTES = 32
_DEFAULT_MAXIMUM_SAMPLES = 32
_TemplateRow: TypeAlias = sqlite3.Row


@dataclass(frozen=True, slots=True)
class LocalIdentitySample:
    """One model-versioned biometric embedding that never leaves its device."""

    tenant_id: str
    kind: IdentityKind
    source_observation_id: str
    sample_id: str
    embedding: tuple[float, ...]
    model_reference: ModelReference

    def __post_init__(self) -> None:
        _require_non_empty(
            tenant_id=self.tenant_id,
            source_observation_id=self.source_observation_id,
            sample_id=self.sample_id,
        )
        _normalized_embedding(self.embedding)


@dataclass(frozen=True, slots=True)
class LocalIdentityMatch:
    """One anonymous device-domain identity match safe to send to the cloud."""

    identity_id: str
    kind: IdentityKind
    confidence: float
    model_reference: ModelReference
    enrolled_new: bool

    def __post_init__(self) -> None:
        if not self.identity_id.strip():
            raise ValueError("identity_id must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("identity confidence must be between 0 and 1")

    def to_observation_input(
        self,
        *,
        start_ms: int,
        end_ms: int,
    ) -> IdentityObservationInput:
        """Create the cloud-safe interval without exposing the local embedding."""
        return IdentityObservationInput(
            identity_id=self.identity_id,
            kind=self.kind,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=self.confidence,
            model_id=self.model_reference.model_id,
            model_revision=self.model_reference.revision,
        )


class _EncryptedSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    tenant_id: str
    device_id: str
    kind: IdentityKind
    identity_id: str
    source_observation_id: str
    sample_id: str
    model_id: str
    model_revision: str
    embedding: Annotated[tuple[float, ...], Field(min_length=1)]


class SQLiteIdentityMemory:
    """Remember encrypted embeddings and learn from bounded observation samples."""

    def __init__(
        self,
        database_path: Path,
        *,
        device_id: str,
        encryption_key: bytes,
        maximum_samples_per_identity: int = _DEFAULT_MAXIMUM_SAMPLES,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if str(database_path) == ":memory:":
            raise ValueError("edge identity memory must use a file-backed SQLite database")
        if not device_id.strip():
            raise ValueError("device_id must not be empty")
        if len(encryption_key) != _AES_256_KEY_BYTES:
            raise ValueError("edge identity encryption_key must contain exactly 32 bytes")
        if maximum_samples_per_identity <= 0:
            raise ValueError("maximum_samples_per_identity must be positive")
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = database_path
        self._device_id = device_id
        self._cipher = AESGCM(encryption_key)
        self._maximum_samples_per_identity = maximum_samples_per_identity
        self._clock = clock or _utc_now
        with self._connect() as connection:
            initialize_identity_tables(connection)
        os.chmod(database_path, 0o600)

    def recognize_and_remember(
        self,
        sample: LocalIdentitySample,
        *,
        minimum_similarity: float,
    ) -> LocalIdentityMatch:
        """Match one sample and retain it as bounded, forgettable device learning."""
        if not math.isfinite(minimum_similarity) or not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        sample = replace(sample, embedding=_normalized_embedding(sample.embedding))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._find_sample(connection, sample)
            if existing is not None:
                payload = self._decrypt(existing)
                if (
                    payload.source_observation_id != sample.source_observation_id
                    or _cosine_similarity(sample.embedding, payload.embedding) < 1.0 - 1e-6
                ):
                    raise IdempotencyConflictError(
                        "edge identity sample was reused with different embedding content"
                    )
                return self._match(existing, confidence=1.0, enrolled_new=False)

            # ponytail: device-local identities use a linear scan; add FAISS after measured need.
            best_match = self._best_match(connection, sample)
            if best_match is None or best_match[1] < minimum_similarity:
                enrolled_new = True
                identity_id = _new_identity_id(self._device_id, sample)
                confidence = 1.0
            else:
                enrolled_new = False
                identity_id = best_match[0]
                confidence = max(0.0, min(1.0, best_match[1]))
            self._insert_sample(connection, sample, identity_id)
            self._prune_oldest_samples(connection, sample, identity_id)
        return LocalIdentityMatch(
            identity_id=identity_id,
            kind=sample.kind,
            confidence=confidence,
            model_reference=sample.model_reference,
            enrolled_new=enrolled_new,
        )

    def forget_identity(self, tenant_id: str, identity_id: str) -> int:
        """Delete every local biometric sample for one anonymous identity."""
        _require_non_empty(tenant_id=tenant_id, identity_id=identity_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM edge_identity_templates
                WHERE tenant_id = ? AND device_id = ? AND identity_id = ?
                """,
                (tenant_id, self._device_id, identity_id),
            )
        return cursor.rowcount

    def forget_observation(self, tenant_id: str, observation_id: str) -> int:
        """Delete identity samples learned from one forgotten observation."""
        _require_non_empty(tenant_id=tenant_id, observation_id=observation_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM edge_identity_templates
                WHERE tenant_id = ? AND device_id = ? AND source_observation_id = ?
                """,
                (tenant_id, self._device_id, observation_id),
            )
        return cursor.rowcount

    def _best_match(
        self,
        connection: sqlite3.Connection,
        sample: LocalIdentitySample,
    ) -> tuple[str, float] | None:
        best: tuple[str, float] | None = None
        for row in connection.execute(
            """
            SELECT * FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND kind = ?
              AND model_id = ? AND model_revision = ? AND dimension = ?
            ORDER BY identity_id, created_at, sample_id
            """,
            (*self._sample_scope(sample), len(sample.embedding)),
        ):
            similarity = _cosine_similarity(sample.embedding, self._decrypt(row).embedding)
            if best is None or similarity > best[1]:
                best = (str(row["identity_id"]), similarity)
        return best

    def _find_sample(
        self,
        connection: sqlite3.Connection,
        sample: LocalIdentitySample,
    ) -> _TemplateRow | None:
        return cast(
            _TemplateRow | None,
            connection.execute(
                """
                SELECT * FROM edge_identity_templates
                WHERE tenant_id = ? AND device_id = ? AND kind = ?
                  AND model_id = ? AND model_revision = ? AND sample_id = ?
                """,
                (*self._sample_scope(sample), sample.sample_id),
            ).fetchone(),
        )

    def _insert_sample(
        self,
        connection: sqlite3.Connection,
        sample: LocalIdentitySample,
        identity_id: str,
    ) -> None:
        payload = _EncryptedSample(
            tenant_id=sample.tenant_id,
            device_id=self._device_id,
            kind=sample.kind,
            identity_id=identity_id,
            source_observation_id=sample.source_observation_id,
            sample_id=sample.sample_id,
            model_id=sample.model_reference.model_id,
            model_revision=sample.model_reference.revision,
            embedding=sample.embedding,
        )
        nonce = os.urandom(_NONCE_BYTES)
        connection.execute(
            """
            INSERT INTO edge_identity_templates (
                tenant_id, device_id, kind, identity_id, source_observation_id,
                sample_id, model_id, model_revision, dimension, nonce,
                encrypted_embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.tenant_id,
                self._device_id,
                sample.kind.value,
                identity_id,
                sample.source_observation_id,
                sample.sample_id,
                sample.model_reference.model_id,
                sample.model_reference.revision,
                len(sample.embedding),
                nonce,
                self._cipher.encrypt(nonce, payload.model_dump_json().encode(), None),
                self._now().isoformat(),
            ),
        )

    def _decrypt(self, row: _TemplateRow) -> _EncryptedSample:
        try:
            payload = _EncryptedSample.model_validate_json(
                self._cipher.decrypt(bytes(row["nonce"]), bytes(row["encrypted_embedding"]), None)
            )
        except (InvalidTag, ValidationError) as error:
            raise MemoryIntegrityError("edge identity template cannot be decrypted") from error
        if not _payload_matches_row(payload, row):
            raise MemoryIntegrityError("edge identity template metadata changed")
        return payload

    def _prune_oldest_samples(
        self,
        connection: sqlite3.Connection,
        sample: LocalIdentitySample,
        identity_id: str,
    ) -> None:
        old_samples = connection.execute(
            """
            SELECT sample_id FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND kind = ?
              AND model_id = ? AND model_revision = ? AND identity_id = ?
            ORDER BY created_at DESC, sample_id DESC
            LIMIT -1 OFFSET ?
            """,
            (*self._sample_scope(sample), identity_id, self._maximum_samples_per_identity),
        ).fetchall()
        connection.executemany(
            """
            DELETE FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND kind = ?
              AND model_id = ? AND model_revision = ? AND sample_id = ?
            """,
            ((*self._sample_scope(sample), row["sample_id"]) for row in old_samples),
        )

    def _sample_scope(self, sample: LocalIdentitySample) -> tuple[str, str, str, str, str]:
        return (
            sample.tenant_id,
            self._device_id,
            sample.kind.value,
            sample.model_reference.model_id,
            sample.model_reference.revision,
        )

    def _match(
        self,
        row: _TemplateRow,
        *,
        confidence: float,
        enrolled_new: bool,
    ) -> LocalIdentityMatch:
        return LocalIdentityMatch(
            identity_id=str(row["identity_id"]),
            kind=IdentityKind(row["kind"]),
            confidence=confidence,
            model_reference=ModelReference(
                model_id=str(row["model_id"]),
                revision=str(row["model_revision"]),
            ),
            enrolled_new=enrolled_new,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("edge identity clock must return a timezone-aware datetime")
        return now


def _normalized_embedding(embedding: tuple[float, ...]) -> tuple[float, ...]:
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("identity embedding must contain finite values")
    magnitude = math.sqrt(math.fsum(value * value for value in embedding))
    if magnitude == 0.0:
        raise ValueError("identity embedding must not be a zero vector")
    return tuple(value / magnitude for value in embedding)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise MemoryIntegrityError("edge identity template dimension changed")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _payload_matches_row(payload: _EncryptedSample, row: _TemplateRow) -> bool:
    return (
        payload.tenant_id,
        payload.device_id,
        payload.kind.value,
        payload.identity_id,
        payload.source_observation_id,
        payload.sample_id,
        payload.model_id,
        payload.model_revision,
        len(payload.embedding),
    ) == (
        row["tenant_id"],
        row["device_id"],
        row["kind"],
        row["identity_id"],
        row["source_observation_id"],
        row["sample_id"],
        row["model_id"],
        row["model_revision"],
        row["dimension"],
    )


def _new_identity_id(device_id: str, sample: LocalIdentitySample) -> str:
    prefix = "person" if sample.kind is IdentityKind.FACE else "speaker"
    return derive_stable_id(
        prefix,
        sample.tenant_id,
        device_id,
        sample.model_reference.model_id,
        sample.model_reference.revision,
        sample.sample_id,
    )


def _require_non_empty(**values: str) -> None:
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"{name} must not be empty")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
