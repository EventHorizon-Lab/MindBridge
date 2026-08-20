"""Encrypted device-local face and voice identity memory."""

from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypeAlias, cast

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mindbridge.contracts import IdentityObservationInput
from mindbridge.core import (
    IdempotencyConflictError,
    IdentityKind,
    IdentityScope,
    MemoryIntegrityError,
    ModelReference,
    derive_stable_id,
    utc_now,
)
from mindbridge.edge._sqlite import connect as sqlite_connect
from mindbridge.edge.identity_schema import initialize_identity_tables

_NONCE_BYTES = 12
_AES_256_KEY_BYTES = 32
_DEFAULT_MAXIMUM_SAMPLES = 32
_IDENTITY_PROTOTYPE_SAMPLES = 3
_MAXIMUM_ASSOCIATION_EVIDENCE_PER_PAIR = 64
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
        transcript: str | None = None,
        visual_bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> IdentityObservationInput:
        """Create the cloud-safe interval without exposing the local embedding."""
        return IdentityObservationInput(
            identity_id=self.identity_id,
            kind=self.kind,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=self.confidence,
            model_id=self.model_reference.model_id,
            scope=IdentityScope.DEVICE,
            transcript=transcript,
            visual_bbox_xyxy=visual_bbox_xyxy,
        )


@dataclass(frozen=True, slots=True)
class FaceVoiceAssociationEvidence:
    """One unambiguous active-speaker interval linking local face and voice IDs."""

    tenant_id: str
    source_observation_id: str
    evidence_id: str
    face_identity_id: str
    voice_identity_id: str
    start_ms: int
    end_ms: int
    confidence: float
    model_reference: ModelReference

    def __post_init__(self) -> None:
        _require_non_empty(
            tenant_id=self.tenant_id,
            source_observation_id=self.source_observation_id,
            evidence_id=self.evidence_id,
            face_identity_id=self.face_identity_id,
            voice_identity_id=self.voice_identity_id,
        )
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("face/voice evidence time range must have positive duration")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("face/voice evidence confidence must be between 0 and 1")


class _EncryptedSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    tenant_id: str
    device_id: str
    kind: IdentityKind
    identity_id: str
    source_observation_id: str
    sample_id: str
    model_id: str
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
        self._clock = clock or utc_now
        with self._connect() as connection:
            initialize_identity_tables(connection)
        os.chmod(database_path, 0o600)

    def recognize_and_remember(
        self,
        sample: LocalIdentitySample,
        *,
        minimum_similarity: float,
        minimum_margin: float = 0.0,
    ) -> LocalIdentityMatch:
        """Match one sample only when one bounded identity prototype clearly wins."""
        if not math.isfinite(minimum_similarity) or not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0 and 1")
        if not math.isfinite(minimum_margin) or not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be between 0 and 1")
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
            ranked_matches, templates = self._rank_matches(connection, sample)
            best_match = ranked_matches[0] if ranked_matches else None
            ambiguous = (
                len(ranked_matches) > 1
                and best_match is not None
                and best_match[1] - ranked_matches[1][1] < minimum_margin
                # A runner-up that would itself be accepted into the winner is the same face
                # from another angle, not a rival. Enrolling a third identity there resolves
                # nothing and leaves one more near-duplicate for the next sample to tie
                # against, so the margin only guards against a genuinely distinct rival.
                and _identities_are_distinct(
                    templates[best_match[0]],
                    templates[ranked_matches[1][0]],
                    minimum_similarity,
                )
            )
            if best_match is None or best_match[1] < minimum_similarity or ambiguous:
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

    def record_face_voice_evidence(self, evidence: FaceVoiceAssociationEvidence) -> None:
        """Retain one idempotent ASD interval without prematurely merging identities."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._require_identity_kind(
                connection,
                tenant_id=evidence.tenant_id,
                identity_id=evidence.face_identity_id,
                kind=IdentityKind.FACE,
            )
            self._require_identity_kind(
                connection,
                tenant_id=evidence.tenant_id,
                identity_id=evidence.voice_identity_id,
                kind=IdentityKind.VOICE,
            )
            existing = self._find_association_evidence(connection, evidence)
            if existing is not None:
                if not _association_evidence_matches_row(evidence, existing):
                    raise IdempotencyConflictError(
                        "face/voice evidence ID was reused with different content"
                    )
            else:
                self._insert_association_evidence(connection, evidence)
                self._prune_association_evidence(connection, evidence)

    def resolve_identity(
        self,
        tenant_id: str,
        match: LocalIdentityMatch,
        *,
        association_model_reference: ModelReference,
        minimum_observations: int,
        minimum_duration_ms: int,
        minimum_confidence: float,
        minimum_margin: float,
    ) -> LocalIdentityMatch:
        """Map a voice to a face only when both are each other's clear best match."""
        _require_non_empty(tenant_id=tenant_id)
        _validate_association_thresholds(
            minimum_observations=minimum_observations,
            minimum_duration_ms=minimum_duration_ms,
            minimum_confidence=minimum_confidence,
            minimum_margin=minimum_margin,
        )
        if match.kind is IdentityKind.FACE:
            return match
        with self._connect() as connection:
            # ponytail: bounded local evidence is scored on read; persist links if profiling demands it.
            rows = connection.execute(
                """
                SELECT face_identity_id, voice_identity_id,
                       SUM((end_ms - start_ms) * confidence) AS score
                FROM edge_face_voice_evidence
                WHERE tenant_id = ? AND device_id = ?
                  AND association_model_id = ?
                GROUP BY face_identity_id, voice_identity_id
                HAVING COUNT(DISTINCT source_observation_id) >= ?
                   AND SUM(end_ms - start_ms) >= ?
                   AND SUM((end_ms - start_ms) * confidence)
                       / SUM(end_ms - start_ms) >= ?
                ORDER BY score DESC, face_identity_id, voice_identity_id
                """,
                (
                    tenant_id,
                    self._device_id,
                    association_model_reference.model_id,
                    minimum_observations,
                    minimum_duration_ms,
                    minimum_confidence,
                ),
            ).fetchall()
        voice_candidates = [row for row in rows if row["voice_identity_id"] == match.identity_id]
        if not voice_candidates:
            return match
        winner = voice_candidates[0]
        face_candidates = [
            row for row in rows if row["face_identity_id"] == winner["face_identity_id"]
        ]
        if not (
            _is_clear_winner(winner, voice_candidates, minimum_margin)
            and _is_clear_winner(winner, face_candidates, minimum_margin)
        ):
            return match
        return replace(match, identity_id=str(winner["face_identity_id"]))

    def forget_identity(self, tenant_id: str, identity_id: str) -> int:
        """Delete every local biometric sample for one anonymous identity."""
        _require_non_empty(tenant_id=tenant_id, identity_id=identity_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM edge_identity_templates
                WHERE tenant_id = ? AND device_id = ? AND identity_id = ?
                """,
                (tenant_id, self._device_id, identity_id),
            )
            connection.execute(
                """
                DELETE FROM edge_face_voice_evidence
                WHERE tenant_id = ? AND device_id = ?
                  AND (face_identity_id = ? OR voice_identity_id = ?)
                """,
                (tenant_id, self._device_id, identity_id, identity_id),
            )
        return cursor.rowcount

    def forget_observation(self, tenant_id: str, observation_id: str) -> int:
        """Delete identity samples learned from one forgotten observation."""
        _require_non_empty(tenant_id=tenant_id, observation_id=observation_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM edge_identity_templates
                WHERE tenant_id = ? AND device_id = ? AND source_observation_id = ?
                """,
                (tenant_id, self._device_id, observation_id),
            )
            connection.execute(
                """
                DELETE FROM edge_face_voice_evidence
                WHERE tenant_id = ? AND device_id = ?
                  AND (
                    source_observation_id = ?
                    OR NOT EXISTS (
                        SELECT 1 FROM edge_identity_templates AS face
                        WHERE face.tenant_id = edge_face_voice_evidence.tenant_id
                          AND face.device_id = edge_face_voice_evidence.device_id
                          AND face.identity_id = edge_face_voice_evidence.face_identity_id
                          AND face.kind = 'face'
                    )
                    OR NOT EXISTS (
                        SELECT 1 FROM edge_identity_templates AS voice
                        WHERE voice.tenant_id = edge_face_voice_evidence.tenant_id
                          AND voice.device_id = edge_face_voice_evidence.device_id
                          AND voice.identity_id = edge_face_voice_evidence.voice_identity_id
                          AND voice.kind = 'voice'
                    )
                  )
                """,
                (tenant_id, self._device_id, observation_id),
            )
        return cursor.rowcount

    def _require_identity_kind(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        identity_id: str,
        kind: IdentityKind,
    ) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND identity_id = ? AND kind = ?
            LIMIT 1
            """,
            (tenant_id, self._device_id, identity_id, kind.value),
        ).fetchone()
        if row is None:
            raise ValueError(f"{kind.value}_identity_id is not enrolled on this device")

    def _find_association_evidence(
        self,
        connection: sqlite3.Connection,
        evidence: FaceVoiceAssociationEvidence,
    ) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            connection.execute(
                """
                SELECT * FROM edge_face_voice_evidence
                WHERE tenant_id = ? AND device_id = ?
                  AND association_model_id = ?
                  AND evidence_id = ?
                """,
                (*self._association_scope(evidence), evidence.evidence_id),
            ).fetchone(),
        )

    def _insert_association_evidence(
        self,
        connection: sqlite3.Connection,
        evidence: FaceVoiceAssociationEvidence,
    ) -> None:
        connection.execute(
            """
            INSERT INTO edge_face_voice_evidence (
                tenant_id, device_id, association_model_id,
                evidence_id, source_observation_id, face_identity_id, voice_identity_id,
                start_ms, end_ms, confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *self._association_scope(evidence),
                evidence.evidence_id,
                evidence.source_observation_id,
                evidence.face_identity_id,
                evidence.voice_identity_id,
                evidence.start_ms,
                evidence.end_ms,
                evidence.confidence,
                self._now().isoformat(),
            ),
        )

    def _prune_association_evidence(
        self,
        connection: sqlite3.Connection,
        evidence: FaceVoiceAssociationEvidence,
    ) -> None:
        old_evidence = connection.execute(
            """
            SELECT evidence_id FROM edge_face_voice_evidence
            WHERE tenant_id = ? AND device_id = ?
              AND association_model_id = ?
              AND face_identity_id = ? AND voice_identity_id = ?
            ORDER BY created_at DESC, evidence_id DESC
            LIMIT -1 OFFSET ?
            """,
            (
                *self._association_scope(evidence),
                evidence.face_identity_id,
                evidence.voice_identity_id,
                _MAXIMUM_ASSOCIATION_EVIDENCE_PER_PAIR,
            ),
        ).fetchall()
        connection.executemany(
            """
            DELETE FROM edge_face_voice_evidence
            WHERE tenant_id = ? AND device_id = ?
              AND association_model_id = ?
              AND evidence_id = ?
            """,
            ((*self._association_scope(evidence), row["evidence_id"]) for row in old_evidence),
        )

    def _association_scope(
        self,
        evidence: FaceVoiceAssociationEvidence,
    ) -> tuple[str, str, str]:
        return (
            evidence.tenant_id,
            self._device_id,
            evidence.model_reference.model_id,
        )

    def _rank_matches(
        self,
        connection: sqlite3.Connection,
        sample: LocalIdentitySample,
    ) -> tuple[tuple[tuple[str, float], ...], dict[str, tuple[tuple[float, ...], ...]]]:
        """Score every identity and keep the templates the scan already decrypted."""
        scores_by_identity: dict[str, list[float]] = {}
        templates_by_identity: dict[str, list[tuple[float, ...]]] = {}
        for row in connection.execute(
            """
            SELECT * FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND kind = ?
              AND model_id = ? AND dimension = ?
            ORDER BY identity_id, created_at, sample_id
            """,
            (*self._sample_scope(sample), len(sample.embedding)),
        ):
            identity_id = str(row["identity_id"])
            embedding = self._decrypt(row).embedding
            scores_by_identity.setdefault(identity_id, []).append(
                _cosine_similarity(sample.embedding, embedding)
            )
            templates_by_identity.setdefault(identity_id, []).append(embedding)
        ranked = []
        for identity_id, scores in scores_by_identity.items():
            strongest = sorted(scores, reverse=True)[:_IDENTITY_PROTOTYPE_SAMPLES]
            ranked.append((identity_id, math.fsum(strongest) / len(strongest)))
        return (
            tuple(sorted(ranked, key=lambda item: (-item[1], item[0]))),
            {identity: tuple(values) for identity, values in templates_by_identity.items()},
        )

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
                  AND model_id = ? AND sample_id = ?
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
            embedding=sample.embedding,
        )
        nonce = os.urandom(_NONCE_BYTES)
        connection.execute(
            """
            INSERT INTO edge_identity_templates (
                tenant_id, device_id, kind, identity_id, source_observation_id,
                sample_id, model_id, dimension, nonce,
                encrypted_embedding, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample.tenant_id,
                self._device_id,
                sample.kind.value,
                identity_id,
                sample.source_observation_id,
                sample.sample_id,
                sample.model_reference.model_id,
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
              AND model_id = ? AND identity_id = ?
            ORDER BY created_at DESC, sample_id DESC
            LIMIT -1 OFFSET ?
            """,
            (*self._sample_scope(sample), identity_id, self._maximum_samples_per_identity),
        ).fetchall()
        connection.executemany(
            """
            DELETE FROM edge_identity_templates
            WHERE tenant_id = ? AND device_id = ? AND kind = ?
              AND model_id = ? AND sample_id = ?
            """,
            ((*self._sample_scope(sample), row["sample_id"]) for row in old_samples),
        )

    def _sample_scope(self, sample: LocalIdentitySample) -> tuple[str, str, str, str]:
        return (
            sample.tenant_id,
            self._device_id,
            sample.kind.value,
            sample.model_reference.model_id,
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
            model_reference=ModelReference(model_id=str(row["model_id"])),
            enrolled_new=enrolled_new,
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite_connect(self._database_path)

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("edge identity clock must return a timezone-aware datetime")
        return now


def _validate_association_thresholds(
    *,
    minimum_observations: int,
    minimum_duration_ms: int,
    minimum_confidence: float,
    minimum_margin: float,
) -> None:
    if minimum_observations <= 0:
        raise ValueError("minimum_observations must be positive")
    if minimum_duration_ms <= 0:
        raise ValueError("minimum_duration_ms must be positive")
    if not math.isfinite(minimum_confidence) or not 0.0 < minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be greater than 0 and at most 1")
    if not math.isfinite(minimum_margin) or not 0.0 <= minimum_margin <= 1.0:
        raise ValueError("minimum_margin must be between 0 and 1")


def _association_evidence_matches_row(
    evidence: FaceVoiceAssociationEvidence,
    row: sqlite3.Row,
) -> bool:
    return (
        evidence.tenant_id,
        evidence.source_observation_id,
        evidence.evidence_id,
        evidence.face_identity_id,
        evidence.voice_identity_id,
        evidence.start_ms,
        evidence.end_ms,
        evidence.confidence,
        evidence.model_reference.model_id,
    ) == (
        row["tenant_id"],
        row["source_observation_id"],
        row["evidence_id"],
        row["face_identity_id"],
        row["voice_identity_id"],
        row["start_ms"],
        row["end_ms"],
        row["confidence"],
        row["association_model_id"],
    )


def _identities_are_distinct(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    minimum_similarity: float,
) -> bool:
    """Report whether the closest templates of two identities still fail to match.

    Aggregated like `_rank_matches` scores a sample, so one outlier template pair cannot
    declare two people the same identity and switch the ambiguity guard off.
    """
    similarities = sorted(
        (_cosine_similarity(one, other) for one in left for other in right),
        reverse=True,
    )[:_IDENTITY_PROTOTYPE_SAMPLES]
    if not similarities:
        return True
    return math.fsum(similarities) / len(similarities) < minimum_similarity


def _is_clear_winner(
    candidate: sqlite3.Row,
    ranked_candidates: list[sqlite3.Row],
    minimum_margin: float,
) -> bool:
    winner = ranked_candidates[0]
    if (
        winner["face_identity_id"],
        winner["voice_identity_id"],
    ) != (
        candidate["face_identity_id"],
        candidate["voice_identity_id"],
    ):
        return False
    if len(ranked_candidates) == 1:
        return True
    score = float(candidate["score"])
    runner_up_score = float(ranked_candidates[1]["score"])
    return score > runner_up_score and (score - runner_up_score) / score >= minimum_margin


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
        len(payload.embedding),
    ) == (
        row["tenant_id"],
        row["device_id"],
        row["kind"],
        row["identity_id"],
        row["source_observation_id"],
        row["sample_id"],
        row["model_id"],
        row["dimension"],
    )


def _new_identity_id(device_id: str, sample: LocalIdentitySample) -> str:
    prefix = "person" if sample.kind is IdentityKind.FACE else "speaker"
    return derive_stable_id(
        prefix,
        sample.tenant_id,
        device_id,
        sample.model_reference.model_id,
        sample.sample_id,
    )


def _require_non_empty(**values: str) -> None:
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
