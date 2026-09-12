"""Media descriptors and the analyses cached on them: transcripts, descriptions, speech, faces."""

from __future__ import annotations

import math
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    asset_from_row,
    datetime_text,
    normalized_vector,
    optional_datetime_text,
    row_blob,
    row_text,
    unpack_vector,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._identity import (
    FACE_EXEMPLAR_LIMIT,
    VOICE_EXEMPLAR_LIMIT,
    accepted_identity,
    delete_unobserved_identities,
    resolve_identity_id,
    restrained_identities,
    write_identity_exemplars,
)
from mindbridge.infrastructure.local.store._outbox import queue_asset_identity_projection
from mindbridge.infrastructure.local.store.rows import (
    IdentityChange,
    IdentityExemplarState,
    IdentityState,
    SpeechRollback,
    StoredAsset,
    require_aware,
    require_identifier,
    validated_identity_modality,
    validated_sha256,
)
from mindbridge.models.base import FaceAnalysis, SpeechAnalysis
from mindbridge.types import FaceObservation, SpeakerSegment


def write_asset(connection: sqlite3.Connection, asset: StoredAsset) -> None:
    connection.execute(
        """
        INSERT INTO media_assets (
            asset_id, modality, mime_type, size_bytes, sha256,
            relative_path, name, transcript, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (asset_id) DO NOTHING
        """,
        (
            asset.asset_id,
            asset.modality,
            asset.mime_type,
            asset.size_bytes,
            asset.sha256,
            asset.relative_path,
            asset.name,
            asset.transcript,
            datetime_text(asset.created_at),
        ),
    )
    row = connection.execute(
        """
        SELECT asset_id, modality, mime_type, size_bytes, sha256,
               relative_path, name, transcript, created_at
        FROM media_assets
        WHERE asset_id = ?
        """,
        (asset.asset_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("asset upsert did not produce a row")
    stored = asset_from_row(row)
    immutable = (
        "modality",
        "mime_type",
        "size_bytes",
        "sha256",
        "relative_path",
    )
    if any(getattr(stored, field) != getattr(asset, field) for field in immutable):
        raise ValueError(f"asset {asset.asset_id!r} conflicts with stored metadata")
    name = stored.name if stored.name is not None else asset.name
    transcript = asset.transcript if asset.transcript is not None else stored.transcript
    if name != stored.name or transcript != stored.transcript:
        connection.execute(
            "UPDATE media_assets SET name = ?, transcript = ? WHERE asset_id = ?",
            (name, transcript, asset.asset_id),
        )


def read_unreferenced_assets(
    connection: sqlite3.Connection,
    asset_ids: Sequence[str],
) -> tuple[StoredAsset, ...]:
    if not asset_ids:
        return ()
    found: dict[str, StoredAsset] = {}
    for offset in range(0, len(asset_ids), SQLITE_PARAMETER_BATCH):
        batch = asset_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _asset_id in batch)
        rows = connection.execute(
            f"""
            SELECT a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                   a.relative_path, a.name, a.transcript, a.created_at
            FROM media_assets AS a
            WHERE a.asset_id IN ({placeholders})
              AND NOT EXISTS (
                  SELECT 1 FROM memory_assets AS ma WHERE ma.asset_id = a.asset_id
              )
            """,
            tuple(batch),
        ).fetchall()
        found.update((row_text(row, "asset_id"), asset_from_row(row)) for row in rows)
    return tuple(found[asset_id] for asset_id in asset_ids if asset_id in found)


def _read_speech(
    connection: sqlite3.Connection,
    asset_id: str,
) -> tuple[SpeakerSegment, ...]:
    rows = connection.execute(
        """
        SELECT s.start_ms, s.end_ms, s.transcript, s.speaker_id,
               i.name AS speaker_name, s.identity_score
        FROM speech_segments AS s
        LEFT JOIN identities AS i ON i.identity_id = s.speaker_id
        WHERE s.asset_id = ?
        ORDER BY s.position
        """,
        (asset_id,),
    ).fetchall()
    return tuple(
        SpeakerSegment(
            asset_id=asset_id,
            start_ms=int(row["start_ms"]),
            end_ms=int(row["end_ms"]),
            text=row_text(row, "transcript"),
            speaker_id=(None if row["speaker_id"] is None else row_text(row, "speaker_id")),
            speaker_name=(None if row["speaker_name"] is None else row_text(row, "speaker_name")),
            identity_score=(
                None if row["identity_score"] is None else float(row["identity_score"])
            ),
        )
        for row in rows
    )


def _read_faces(
    connection: sqlite3.Connection,
    asset_id: str,
) -> tuple[FaceObservation, ...]:
    rows = connection.execute(
        """
        SELECT f.observed_at_ms, f.box_x, f.box_y, f.box_width, f.box_height,
               f.identity_id, i.name AS identity_name, f.identity_score
        FROM face_observations AS f
        JOIN identities AS i ON i.identity_id = f.identity_id
        WHERE f.asset_id = ?
        ORDER BY f.position
        """,
        (asset_id,),
    ).fetchall()
    return tuple(
        FaceObservation(
            asset_id=asset_id,
            observed_at_ms=(None if row["observed_at_ms"] is None else int(row["observed_at_ms"])),
            bounding_box=(
                float(row["box_x"]),
                float(row["box_y"]),
                float(row["box_width"]),
                float(row["box_height"]),
            ),
            identity_id=row_text(row, "identity_id"),
            identity_name=(
                None if row["identity_name"] is None else row_text(row, "identity_name")
            ),
            identity_score=(
                None if row["identity_score"] is None else float(row["identity_score"])
            ),
        )
        for row in rows
    )


def _match_speakers(
    connection: sqlite3.Connection,
    speakers: dict[str, tuple[float, ...]],
    *,
    model_id: str,
    space_id: str,
    minimum_similarity: float,
    minimum_margin: float,
    now: datetime,
    preferred_identity: str | None = None,
) -> tuple[
    dict[str, tuple[str, float | None]],
    tuple[IdentityChange, ...],
]:
    observations = {label: (vector,) for label, vector in speakers.items()}
    claim_groups = dict.fromkeys(observations, 0)
    return _match_identities(
        connection,
        observations,
        claim_groups=claim_groups,
        modality="voice",
        model_id=model_id,
        space_id=space_id,
        minimum_similarity=minimum_similarity,
        minimum_margin=minimum_margin,
        exemplar_limit=VOICE_EXEMPLAR_LIMIT,
        now=now,
        preferred_identity=preferred_identity,
    )


def _match_identities(
    connection: sqlite3.Connection,
    observations: Mapping[str, tuple[tuple[float, ...], ...]],
    *,
    claim_groups: Mapping[str, int | None],
    modality: Literal["face", "voice"],
    model_id: str,
    space_id: str,
    minimum_similarity: float,
    minimum_margin: float,
    exemplar_limit: int,
    now: datetime,
    preferred_identity: str | None = None,
) -> tuple[dict[str, tuple[str, float | None]], tuple[IdentityChange, ...]]:
    if not observations:
        return {}, ()
    dimension = len(next(iter(observations.values()))[0])
    rows = connection.execute(
        """
        SELECT identity_id, position, vector, created_at
        FROM identity_exemplars
        WHERE modality = ? AND space_id = ? AND dimension = ?
        ORDER BY identity_id, position
        """,
        (modality, space_id, dimension),
    ).fetchall()
    existing: dict[str, list[tuple[tuple[float, ...], str]]] = {}
    for row in rows:
        identity_id = row_text(row, "identity_id")
        existing.setdefault(identity_id, []).append(
            (
                normalized_vector(
                    unpack_vector(row_blob(row, "vector"), dimension),
                    f"stored {modality} exemplar",
                ),
                row_text(row, "created_at"),
            )
        )
    preferred_identity = (
        None if preferred_identity is None else resolve_identity_id(connection, preferred_identity)
    )
    preferred_exists = preferred_identity is not None
    preferred_missing_modality = (
        preferred_identity is not None
        and connection.execute(
            """
            SELECT 1 FROM identity_exemplars
            WHERE identity_id = ? AND modality = ?
            LIMIT 1
            """,
            (preferred_identity, modality),
        ).fetchone()
        is None
    )
    known_identities = set(existing)
    if preferred_exists and preferred_identity is not None:
        known_identities.add(preferred_identity)
    # Consent restrains enrolment, not recognition. A person who withheld or withdrew it
    # still matches the exemplars already held -- destroying those is `forget_identity` --
    # but this observation adds nothing to their template, so the bank stops growing from
    # the moment they said so.
    restrained = restrained_identities(connection)
    claimed: dict[int | None, set[str]] = {}
    matches: dict[str, tuple[str, float | None]] = {}
    changes: dict[str, IdentityChange] = {}
    now_text = datetime_text(now)
    for label, vectors in observations.items():
        group_claims = claimed.setdefault(claim_groups[label], set())
        # ponytail: local identity populations use a linear scan; add a vector index only
        # after profiling shows identity matching matters beside model inference.
        accepted = accepted_identity(
            existing,
            vectors,
            claimed=group_claims,
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
        )
        if accepted is not None:
            identity_id, score = accepted
        elif (
            preferred_exists
            and preferred_missing_modality
            and preferred_identity is not None
            and len(observations) == 1
            and preferred_identity not in group_claims
        ):
            identity_id, score = preferred_identity, None
        else:
            identity_id, score = f"identity_{uuid.uuid4().hex}", None
        if identity_id in restrained:
            # Recognized, reported, and nothing written: no exemplar, no `identities` row
            # update, and so nothing for a rollback to undo either.
            matches[label] = (
                identity_id,
                None if score is None else max(0.0, min(1.0, score)),
            )
            group_claims.add(identity_id)
            continue
        if identity_id not in changes:
            previous = (
                None
                if identity_id not in known_identities
                else _identity_state(connection, identity_id)
            )
            changes[identity_id] = IdentityChange(identity_id, previous)
        identity_exists = identity_id in known_identities
        existing[identity_id] = write_identity_exemplars(
            connection,
            identity_id,
            existing.get(identity_id, ()),
            vectors,
            modality=modality,
            model_id=model_id,
            space_id=space_id,
            dimension=dimension,
            exemplar_limit=exemplar_limit,
            identity_exists=identity_exists,
            now_text=now_text,
        )
        known_identities.add(identity_id)
        matches[label] = (
            identity_id,
            None if score is None else max(0.0, min(1.0, score)),
        )
        group_claims.add(identity_id)
    return matches, tuple(changes.values())


def _identity_state(connection: sqlite3.Connection, identity_id: str) -> IdentityState:
    row = connection.execute(
        "SELECT name, created_at, updated_at FROM identities WHERE identity_id = ?",
        (identity_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("matched identity disappeared during recognition")
    exemplars = connection.execute(
        """
        SELECT modality, position, model_id, space_id, dimension, vector, created_at
        FROM identity_exemplars
        WHERE identity_id = ?
        ORDER BY modality, position
        """,
        (identity_id,),
    ).fetchall()
    return IdentityState(
        identity_id=identity_id,
        name=None if row["name"] is None else row_text(row, "name"),
        created_at=row_text(row, "created_at"),
        updated_at=row_text(row, "updated_at"),
        exemplars=tuple(
            IdentityExemplarState(
                modality=validated_identity_modality(exemplar["modality"]),
                position=int(exemplar["position"]),
                model_id=row_text(exemplar, "model_id"),
                space_id=row_text(exemplar, "space_id"),
                dimension=int(exemplar["dimension"]),
                vector=row_blob(exemplar, "vector"),
                created_at=row_text(exemplar, "created_at"),
            )
            for exemplar in exemplars
        ),
    )


class MediaAnalyses:
    """Asset descriptors and the cached analyses keyed on them."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

    def read_asset(self, asset_id: str) -> StoredAsset | None:
        """Return one persisted asset descriptor, including a cached transcript."""
        validated_sha256(asset_id)
        with self._connections.connection() as connection:
            row = connection.execute(
                """
                SELECT asset_id, modality, mime_type, size_bytes, sha256,
                       relative_path, name, transcript, created_at
                FROM media_assets
                WHERE asset_id = ?
                """,
                (asset_id,),
            ).fetchone()
        return None if row is None else asset_from_row(row)

    def read_assets(self, asset_ids: Sequence[str]) -> tuple[StoredAsset, ...]:
        """Resolve existing assets in caller order, preserving repeated IDs."""
        if not asset_ids:
            return ()
        for asset_id in asset_ids:
            validated_sha256(asset_id)
        rows: list[sqlite3.Row] = []
        unique_ids = tuple(dict.fromkeys(asset_ids))
        with self._connections.connection() as connection:
            for offset in range(0, len(unique_ids), SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT asset_id, modality, mime_type, size_bytes, sha256,
                               relative_path, name, transcript, created_at
                        FROM media_assets
                        WHERE asset_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                )
        by_id = {row_text(row, "asset_id"): asset_from_row(row) for row in rows}
        return tuple(by_id[asset_id] for asset_id in asset_ids if asset_id in by_id)

    def write_asset(self, asset: StoredAsset) -> None:
        """Persist one media descriptor without attaching it to a memory."""
        if not isinstance(asset, StoredAsset):
            raise ValueError("asset must be a StoredAsset value")
        with self._connections.transaction() as connection:
            write_asset(connection, asset)

    def read_unreferenced_assets(
        self,
        asset_ids: Sequence[str],
    ) -> tuple[StoredAsset, ...]:
        """Return supplied asset rows that no memory currently references."""
        for asset_id in asset_ids:
            validated_sha256(asset_id)
        with self._connections.connection() as connection:
            return read_unreferenced_assets(connection, tuple(dict.fromkeys(asset_ids)))

    def set_asset_transcript(self, asset_id: str, text: str) -> bool:
        """Fill or replace an asset's cached transcript.

        Empty text is meaningful: it records that transcription completed without speech.
        """
        return self.set_asset_transcripts(((asset_id, text),)) == 1

    def set_asset_transcripts(self, values: Sequence[tuple[str, str]]) -> int:
        """Cache a batch of transcripts in one durable SQLite transaction."""
        supplied = tuple(values)
        if len({asset_id for asset_id, _text in supplied}) != len(supplied):
            raise ValueError("asset transcript IDs must be unique")
        for asset_id, text in supplied:
            validated_sha256(asset_id)
            if not isinstance(text, str):
                raise ValueError("asset transcript must be text")
        if not supplied:
            return 0
        with self._connections.transaction() as connection:
            cursor = connection.executemany(
                "UPDATE media_assets SET transcript = ? WHERE asset_id = ?",
                ((text, asset_id) for asset_id, text in supplied),
            )
        return cursor.rowcount

    def read_visual_descriptions(
        self,
        asset_ids: Sequence[str],
        *,
        space_id: str,
    ) -> dict[str, str]:
        """Return the cached caption for every supplied asset described in this vision space.

        Batched because the write path describes a whole `add_many` at once, and an asset with no
        caption in this space is simply absent rather than an error: a miss is the normal state.
        """
        require_identifier(space_id, "vision space_id")
        for asset_id in asset_ids:
            validated_sha256(asset_id)
        if not asset_ids:
            return {}
        unique_ids = tuple(dict.fromkeys(asset_ids))
        found: dict[str, str] = {}
        with self._connections.connection() as connection:
            for offset in range(0, len(unique_ids), SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                found.update(
                    (row_text(row, "asset_id"), row_text(row, "description"))
                    for row in connection.execute(
                        f"""
                        SELECT asset_id, description
                        FROM visual_descriptions
                        WHERE space_id = ? AND asset_id IN ({placeholders})
                        """,
                        (space_id, *batch),
                    ).fetchall()
                )
        return found

    def write_visual_descriptions(
        self,
        descriptions: Mapping[str, str],
        *,
        model_id: str,
        space_id: str,
    ) -> int:
        """Cache derived captions for stored assets in one durable SQLite transaction.

        An asset already described in this space keeps its stored caption: the point of the row is
        that two ingests of one picture build the same document, so a concurrent writer that
        described the same bytes first wins and the later caption is dropped.
        """
        require_identifier(model_id, "vision model_id")
        require_identifier(space_id, "vision space_id")
        for asset_id, description in descriptions.items():
            validated_sha256(asset_id)
            if not isinstance(description, str) or not description.strip():
                raise ValueError("visual description must not be blank")
        if not descriptions:
            return 0
        now = datetime_text(datetime.now(timezone.utc))
        with self._connections.transaction() as connection:
            cursor = connection.executemany(
                """
                INSERT INTO visual_descriptions (
                    asset_id, space_id, model_id, description, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (asset_id, space_id) DO NOTHING
                """,
                (
                    (asset_id, space_id, model_id, description, now)
                    for asset_id, description in descriptions.items()
                ),
            )
        return cursor.rowcount

    def read_speech(
        self,
        asset_id: str,
        *,
        space_id: str,
    ) -> tuple[SpeakerSegment, ...] | None:
        """Return cached speaker turns, including an empty completed analysis."""
        validated_sha256(asset_id)
        require_identifier(space_id, "speech space_id")
        with self._connections.connection() as connection:
            analysis = connection.execute(
                "SELECT 1 FROM speech_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if analysis is None:
                return None
            return _read_speech(connection, asset_id)

    def write_speech(
        self,
        asset_id: str,
        analysis: SpeechAnalysis,
        *,
        model_id: str,
        space_id: str,
        # Provenance and calibration live on MemoryConfig.speaker_similarity/speaker_margin in
        # plugins.py; every product call site passes them, so treat these literals as a test
        # convenience and not as a settled threshold.
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[SpeakerSegment, ...]:
        """Persist one analysis and match its CAM++ exemplars to local identities."""
        segments, _rollback = self.write_speech_reversible(
            asset_id,
            analysis,
            model_id=model_id,
            space_id=space_id,
            minimum_similarity=minimum_similarity,
            minimum_margin=minimum_margin,
            preferred_identity=preferred_identity,
        )
        return segments

    def write_speech_reversible(
        self,
        asset_id: str,
        analysis: SpeechAnalysis,
        *,
        model_id: str,
        space_id: str,
        # See write_speech: MemoryConfig.speaker_similarity/speaker_margin own these values.
        minimum_similarity: float = 0.78,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[tuple[SpeakerSegment, ...], SpeechRollback | None]:
        """Persist speech and return an undo token when this call created the analysis."""
        validated_sha256(asset_id)
        require_identifier(model_id, "speech model_id")
        require_identifier(space_id, "speech space_id")
        if preferred_identity is not None:
            require_identifier(preferred_identity, "preferred_identity")
        if not isinstance(analysis, SpeechAnalysis):
            raise ValueError("analysis must be a SpeechAnalysis value")
        for value, name in (
            (minimum_similarity, "minimum_similarity"),
            (minimum_margin, "minimum_margin"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one")
        speakers = {speaker.speaker_label: speaker.values for speaker in analysis.speakers}
        if len(speakers) != len(analysis.speakers):
            raise ValueError("speaker labels must be unique within one analysis")
        labels = {turn.speaker_label for turn in analysis.turns if turn.speaker_label is not None}
        if labels - speakers.keys():
            raise ValueError("every speaker turn label must have an exemplar")
        dimensions = {len(values) for values in speakers.values()}
        if 0 in dimensions or len(dimensions) > 1:
            raise ValueError("speaker exemplars must share one non-zero dimension")
        normalized = {
            label: normalized_vector(values, "speaker exemplar")
            for label, values in speakers.items()
        }
        now = datetime.now(timezone.utc)
        transcript = "\n".join(turn.text for turn in analysis.turns)
        with self._connections.transaction() as connection:
            cached = connection.execute(
                "SELECT 1 FROM speech_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if cached is not None:
                return _read_speech(connection, asset_id), None
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("speech analysis requires a stored media asset")

            identities, identity_changes = _match_speakers(
                connection,
                normalized,
                model_id=model_id,
                space_id=space_id,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
                now=now,
                preferred_identity=preferred_identity,
            )
            connection.execute("DELETE FROM speech_analyses WHERE asset_id = ?", (asset_id,))
            connection.execute(
                """
                INSERT INTO speech_analyses (asset_id, model_id, space_id, transcript, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (asset_id, model_id, space_id, transcript, datetime_text(now)),
            )
            connection.executemany(
                """
                INSERT INTO speech_segments (
                    asset_id, position, start_ms, end_ms, transcript,
                    speaker_id, identity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        asset_id,
                        position,
                        turn.start_ms,
                        turn.end_ms,
                        turn.text,
                        None if turn.speaker_label is None else identities[turn.speaker_label][0],
                        None if turn.speaker_label is None else identities[turn.speaker_label][1],
                    )
                    for position, turn in enumerate(analysis.turns)
                ),
            )
            connection.execute(
                "UPDATE media_assets SET transcript = ? WHERE asset_id = ?",
                (transcript, asset_id),
            )
            queue_asset_identity_projection(connection, asset_id)
            return _read_speech(connection, asset_id), SpeechRollback(
                asset_id,
                identity_changes,
            )

    def rollback_speech(self, rollback: SpeechRollback) -> bool:
        """Undo one new analysis while its asset is still unreferenced by a memory."""
        if not isinstance(rollback, SpeechRollback):
            raise ValueError("rollback must be a SpeechRollback value")
        with self._connections.transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM memory_assets WHERE asset_id = ? LIMIT 1",
                    (rollback.asset_id,),
                ).fetchone()
                is not None
            ):
                return False
            deleted = connection.execute(
                "DELETE FROM speech_analyses WHERE asset_id = ?",
                (rollback.asset_id,),
            )
            if deleted.rowcount == 0:
                return False
            for change in reversed(rollback.identities):
                previous = change.previous
                if previous is None:
                    removed = connection.execute(
                        """
                        DELETE FROM identities
                        WHERE identity_id = ?
                          AND NOT EXISTS (
                              SELECT 1 FROM speech_segments WHERE speaker_id = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM face_observations WHERE identity_id = ?
                          )
                        """,
                        (change.identity_id, change.identity_id, change.identity_id),
                    )
                    if removed.rowcount != 1:
                        raise RuntimeError("new speaker identity could not be rolled back")
                    continue
                connection.execute(
                    """
                    INSERT INTO identities (identity_id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (identity_id) DO UPDATE SET
                        name = excluded.name,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        previous.identity_id,
                        previous.name,
                        previous.created_at,
                        previous.updated_at,
                    ),
                )
                connection.execute(
                    "DELETE FROM identity_exemplars WHERE identity_id = ?",
                    (previous.identity_id,),
                )
                connection.executemany(
                    """
                    INSERT INTO identity_exemplars (
                        identity_id, modality, position, model_id, space_id,
                        dimension, vector, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        (
                            previous.identity_id,
                            exemplar.modality,
                            exemplar.position,
                            exemplar.model_id,
                            exemplar.space_id,
                            exemplar.dimension,
                            exemplar.vector,
                            exemplar.created_at,
                        )
                        for exemplar in previous.exemplars
                    ),
                )
            return True

    def read_faces(
        self,
        asset_id: str,
        *,
        space_id: str,
    ) -> tuple[FaceObservation, ...] | None:
        """Return cached face observations, including an empty completed analysis."""
        validated_sha256(asset_id)
        require_identifier(space_id, "face space_id")
        with self._connections.connection() as connection:
            analysis = connection.execute(
                "SELECT 1 FROM face_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, space_id),
            ).fetchone()
            if analysis is None:
                return None
            return _read_faces(connection, asset_id)

    def write_faces(
        self,
        asset_id: str,
        analysis: FaceAnalysis,
        *,
        model_id: str,
        space_id: str,
        analysis_space_id: str | None = None,
        # See write_speech: MemoryConfig.face_similarity/face_margin own these values.
        minimum_similarity: float = 0.363,
        minimum_margin: float = 0.05,
        preferred_identity: str | None = None,
    ) -> tuple[FaceObservation, ...]:
        """Persist detected faces and match them against durable identity exemplars.

        ``preferred_identity`` lets a caller enroll this modality into an identity that lacks it,
        for a single unambiguous observation. The product write path deliberately does not use it:
        adopting an asset's lone voice for its lone face is a cross-modal claim made from one
        asset, which `Memory` instead routes through corroborated linking so that it is recorded,
        observable, and reversible. Pass it only where that claim is already established.
        """
        validated_sha256(asset_id)
        require_identifier(model_id, "face model_id")
        require_identifier(space_id, "face space_id")
        selected_analysis_space = space_id if analysis_space_id is None else analysis_space_id
        require_identifier(selected_analysis_space, "face analysis_space_id")
        if preferred_identity is not None:
            require_identifier(preferred_identity, "preferred_identity")
        if not isinstance(analysis, FaceAnalysis):
            raise ValueError("analysis must be a FaceAnalysis value")
        for value, name in (
            (minimum_similarity, "minimum_similarity"),
            (minimum_margin, "minimum_margin"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one")
        dimensions = {len(face.values) for face in analysis.faces}
        if 0 in dimensions or len(dimensions) > 1:
            raise ValueError("face exemplars must share one non-zero dimension")
        observations = {
            face.face_label: (normalized_vector(face.values, "face exemplar"),)
            for face in analysis.faces
        }
        claim_groups = {face.face_label: face.observed_at_ms for face in analysis.faces}
        now = datetime.now(timezone.utc)
        with self._connections.transaction() as connection:
            cached = connection.execute(
                "SELECT 1 FROM face_analyses WHERE asset_id = ? AND space_id = ?",
                (asset_id, selected_analysis_space),
            ).fetchone()
            if cached is not None:
                return _read_faces(connection, asset_id)
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("face analysis requires a stored media asset")
            matches, _changes = _match_identities(
                connection,
                observations,
                claim_groups=claim_groups,
                modality="face",
                model_id=model_id,
                space_id=space_id,
                minimum_similarity=float(minimum_similarity),
                minimum_margin=float(minimum_margin),
                exemplar_limit=FACE_EXEMPLAR_LIMIT,
                now=now,
                preferred_identity=preferred_identity,
            )
            connection.execute("DELETE FROM face_analyses WHERE asset_id = ?", (asset_id,))
            connection.execute(
                """
                INSERT INTO face_analyses (asset_id, model_id, space_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (asset_id, model_id, selected_analysis_space, datetime_text(now)),
            )
            connection.executemany(
                """
                INSERT INTO face_observations (
                    asset_id, position, observed_at_ms, box_x, box_y,
                    box_width, box_height, identity_id, identity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        asset_id,
                        position,
                        face.observed_at_ms,
                        *face.bounding_box,
                        matches[face.face_label][0],
                        matches[face.face_label][1],
                    )
                    for position, face in enumerate(analysis.faces)
                ),
            )
            queue_asset_identity_projection(connection, asset_id)
            return _read_faces(connection, asset_id)

    def list_unreferenced_assets(self, *, limit: int = 100) -> tuple[StoredAsset, ...]:
        """List asset rows eligible for physical garbage collection."""
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                       a.relative_path, a.name, a.transcript, a.created_at
                FROM media_assets AS a
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_assets AS ma WHERE ma.asset_id = a.asset_id
                )
                ORDER BY a.created_at, a.asset_id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(asset_from_row(row) for row in rows)

    def asset_retention_candidates(
        self,
        *,
        created_before: datetime | None = None,
        limit: int = 100,
    ) -> tuple[StoredAsset, ...]:
        """List stored assets oldest first, so a retention window can be expressed.

        Media is the overwhelming majority of storage growth, so retention is a cost and
        privacy mechanism. This is the read half only: it reports what a policy could drop,
        never drops anything, and includes assets a memory still references -- `memory_assets`
        holds those under RESTRICT, so dropping one is a separate decision with its own
        contract. Pair with `asset_storage_bytes` for a size budget and
        `list_unreferenced_assets` for the already-collectable subset.
        """
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        if created_before is not None:
            require_aware(created_before, "created_before")
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT asset_id, modality, mime_type, size_bytes, sha256,
                       relative_path, name, transcript, created_at
                FROM media_assets
                WHERE ? IS NULL OR created_at < ?
                ORDER BY created_at, asset_id
                LIMIT ?
                """,
                (
                    optional_datetime_text(created_before),
                    optional_datetime_text(created_before),
                    limit,
                ),
            ).fetchall()
        return tuple(asset_from_row(row) for row in rows)

    def asset_memory_ids(self, asset_ids: Sequence[str]) -> tuple[str, ...]:
        """Return every memory that references any of these assets, oldest memory first."""
        selected = tuple(dict.fromkeys(asset_ids))
        for asset_id in selected:
            validated_sha256(asset_id)
        if not selected:
            return ()
        found: list[str] = []
        with self._connections.connection() as connection:
            for offset in range(0, len(selected), SQLITE_PARAMETER_BATCH):
                batch = selected[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _asset_id in batch)
                found.extend(
                    row_text(row, "memory_id")
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT memory_id FROM memory_assets
                        WHERE asset_id IN ({placeholders})
                        ORDER BY memory_id
                        """,
                        batch,
                    ).fetchall()
                )
        return tuple(dict.fromkeys(found))

    def asset_storage_bytes(self) -> int:
        """Return the total size of every stored media asset descriptor."""
        with self._connections.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM media_assets"
            ).fetchone()
        return int(row["total"])

    def delete_asset_if_unreferenced(self, asset_id: str) -> bool:
        """Delete one unreferenced descriptor after its CAS file has been removed."""
        validated_sha256(asset_id)
        with self._connections.transaction() as connection:
            # Read before the delete cascades the observations away: these are the only
            # identities this asset can have orphaned.
            observed = tuple(
                row_text(row, "identity_id")
                for row in connection.execute(
                    """
                    SELECT identity_id FROM face_observations WHERE asset_id = ?
                    UNION
                    SELECT speaker_id FROM speech_segments
                    WHERE asset_id = ? AND speaker_id IS NOT NULL
                    """,
                    (asset_id, asset_id),
                ).fetchall()
            )
            cursor = connection.execute(
                """
                DELETE FROM media_assets
                WHERE asset_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_assets WHERE memory_assets.asset_id = media_assets.asset_id
                  )
                """,
                (asset_id,),
            )
            if cursor.rowcount > 0:
                delete_unobserved_identities(connection, observed)
        return cursor.rowcount > 0
