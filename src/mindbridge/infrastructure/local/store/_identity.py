"""Identity projections: naming assertions, consent, exemplars, merges, and splits."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Literal

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    canonical_json,
    datetime_text,
    normalized_vector,
    optional_row_text,
    pack_vector,
    row_text,
    unpack_vector,
)
from mindbridge.infrastructure.local.store._lineage import current_evidence_ids
from mindbridge.infrastructure.local.store.rows import validated_identity_modality
from mindbridge.types import ConsentState, EvidenceBasis, MemoryKind

# The predicate that makes one identity-bound STATE assertion a consent statement. Shared with
# `mindbridge.memory`, which writes those assertions, so the writer and the projection can never
# disagree about which records they are.
CONSENT_PREDICATE = "consent"


# The one basis a consent statement can carry, hardcoded by `_consent_proposal`. Both consent
# reads below filter on it as well as on the predicate, so a row reaching this table by any
# other route -- a future writer, a hand-edited database -- cannot be read as a person's own
# statement. Defence in depth: `Memory._bound_identity` already reserves the predicate, so no
# model-originated claim is bound to an identity in the first place.
_CONSENT_BASIS = EvidenceBasis.USER_STATEMENT.value


# Consent states under which the kernel stops enrolling new biometric exemplars for a person.
# Recognition against exemplars already held is unaffected: erasing what is held is
# `forget_identity`, and answering a question about a photo is not new processing of a person.
_RESTRAINING_CONSENT = frozenset({ConsentState.WITHHELD.value, ConsentState.WITHDRAWN.value})


FACE_EXEMPLAR_LIMIT = 10


VOICE_EXEMPLAR_LIMIT = 20


def delete_unobserved_identities(
    connection: sqlite3.Connection,
    identity_ids: Sequence[str],
) -> None:
    """Drop the anonymous identities the deleted media just left with no observation at all.

    An exemplar is a biometric template derived from stored media, so it is content: leaving one
    behind after its last face observation and speech segment are gone would make `delete()`
    incomplete. A named or merged identity is a person the caller asserted, not a by-product of
    one recording; it survives, and `forget_identity()` is what erases a person.

    Only identities this asset observed are candidates. A sweep of every unobserved identity
    would also destroy one `unlink_identity()` deliberately left anonymous with its exemplars and
    no observations, which is the state continued ingestion is supposed to corroborate again.
    """
    if not identity_ids:
        return
    placeholders = ", ".join("?" for _identity_id in identity_ids)
    connection.execute(
        f"""
        DELETE FROM identities
        WHERE identity_id IN ({placeholders})
          AND name IS NULL AND relationship IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM face_observations AS f
              WHERE f.identity_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM speech_segments AS s
              WHERE s.speaker_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM identity_aliases AS a
              WHERE a.identity_id = identities.identity_id
          )
          AND NOT EXISTS (
              SELECT 1 FROM identity_link_evidence AS e
              WHERE e.voice_id = identities.identity_id OR e.face_id = identities.identity_id
          )
        """,
        tuple(identity_ids),
    )


# The kernel derives a naming assertion's IDs in `memory.py`; importing it here would invert the
# dependency, so the two payloads are restated for the migration and pinned by a contract test
# that registers the same name through the kernel and compares the IDs it mints.
NAMING_RECIPE = "mindbridge-identity-naming-v1"


NAMING_PREDICATE = "identity"


def naming_assertion_ids(
    identity_id: str,
    name: str,
    relationship: str | None,
) -> tuple[str, str]:
    """Return the `(memory_id, lineage_id)` the kernel mints for one host naming assertion."""
    lineage = canonical_json(
        {
            "kind": MemoryKind.ENTITY.value,
            "predicate": NAMING_PREDICATE,
            "frame_id": None,
            "anchor": None,
            "subject": None,
            "identity_id": identity_id,
        }
    )
    formation = canonical_json(
        {
            "recipe": NAMING_RECIPE,
            "kind": MemoryKind.ENTITY.value,
            "identity_id": identity_id,
            "subject": canonical_subject(name),
            "predicate": NAMING_PREDICATE,
            "value": canonical_subject(relationship),
            "assertion_basis": EvidenceBasis.USER_STATEMENT.value,
            "cue_modality": None,
            "episode_source": None,
            "content": None,
            "valid_from": None,
            "valid_until": None,
            "spatial": None,
        }
    )
    return (
        hashlib.sha256(f"mindbridge-formation-v1:{formation}".encode()).hexdigest(),
        hashlib.sha256(f"mindbridge-lineage-v1:{lineage}".encode()).hexdigest(),
    )


def displaced_naming_versions(
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[str, ...]:
    """Return what this record's in-force naming assertion displaced, if it is one.

    Deleting the assertion that renamed somebody has to leave the previous name standing, the
    same reversal `rollback_operation` applies -- a bound `ENTITY` row supersedes a lineage
    rather than accumulating beside it, so nothing else brings the predecessor back. A record
    whose version is already retired displaced nothing that deleting it can restore, and a
    first assertion supersedes nothing, so a retracted name never returns on its own.
    """
    return tuple(
        row_text(row, "supersedes_id")
        for row in connection.execute(
            """
            SELECT v.supersedes_id
            FROM memory_versions AS v
            JOIN memory_semantics AS s ON s.memory_id = v.memory_id
            WHERE v.memory_id = ? AND v.retired_at IS NULL AND v.supersedes_id IS NOT NULL
              AND s.kind = ? AND s.identity_id IS NOT NULL
            ORDER BY v.version
            """,
            (memory_id, MemoryKind.ENTITY.value),
        ).fetchall()
    )


def current_naming_assertion(
    connection: sqlite3.Connection,
    identity_id: str,
    *,
    excluding: Sequence[str] = (),
) -> sqlite3.Row | None:
    """Return the naming assertion `identities.name` currently projects, newest first.

    `excluding` drops records a caller is about to delete, which is how it can rebuild indexed
    text for the projection a deletion is going to leave behind rather than the current one.
    """
    dropped = tuple(dict.fromkeys(excluding))
    placeholders = ", ".join("?" for _memory_id in dropped)
    row: sqlite3.Row | None = connection.execute(
        f"""
        SELECT s.memory_id, s.subject, s.value
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.identity_id = ? AND s.kind = ?
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
          {f"AND s.memory_id NOT IN ({placeholders})" if dropped else ""}
        ORDER BY v.recorded_at DESC, s.memory_id DESC
        LIMIT 1
        """,
        (identity_id, MemoryKind.ENTITY.value, *dropped),
    ).fetchone()
    return row


def current_consent_state(connection: sqlite3.Connection, identity_id: str) -> str | None:
    """Return the consent state one identity's standing assertion projects, or None.

    Consent is a bound STATE assertion rather than a bound ENTITY one, which is what keeps it
    out of `current_naming_assertion` above: the two live in separate lineages, so recording
    consent never displaces a name and renaming somebody never disturbs their consent.
    """
    row: sqlite3.Row | None = connection.execute(
        """
        SELECT s.value
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.identity_id = ? AND s.kind = ? AND s.predicate = ? AND s.basis = ?
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
        ORDER BY v.recorded_at DESC, s.memory_id DESC
        LIMIT 1
        """,
        (identity_id, MemoryKind.STATE.value, CONSENT_PREDICATE, _CONSENT_BASIS),
    ).fetchone()
    return None if row is None else optional_row_text(row, "value")


def restrained_identities(connection: sqlite3.Connection) -> frozenset[str]:
    """Return every identity whose standing consent assertion restrains the kernel.

    Withheld and withdrawn restrain identically; the distinction between them is audit history,
    not policy. Read fresh on each recognition rather than cached, because a person withdrawing
    consent has to take effect on the next observation, not on the next process start.
    """
    # Ordered oldest-first per identity so the newest standing assertion is the one that
    # survives the dict build, which is the same rule `current_naming_assertion` applies.
    standing: dict[str, str | None] = {
        row_text(row, "identity_id"): optional_row_text(row, "value")
        for row in connection.execute(
            """
            SELECT s.identity_id, s.value
            FROM memory_semantics AS s
            JOIN memory_versions AS v ON v.memory_id = s.memory_id
            JOIN memory_records AS r ON r.memory_id = s.memory_id
            WHERE s.kind = ? AND s.predicate = ? AND s.basis = ? AND s.identity_id IS NOT NULL
              AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
            ORDER BY s.identity_id, v.recorded_at, s.memory_id
            """,
            (MemoryKind.STATE.value, CONSENT_PREDICATE, _CONSENT_BASIS),
        ).fetchall()
    }
    return frozenset(
        identity_id for identity_id, value in standing.items() if value in _RESTRAINING_CONSENT
    )


def visible_naming_assertions(
    connection: sqlite3.Connection,
    *,
    identity_ids: Sequence[str] | None = None,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> tuple[sqlite3.Row, ...]:
    """Return the identity, subject, and memory id of every currently visible naming assertion.

    `identity_ids` narrows the scan to the people the caller already has in hand, which is what
    a read path asking "is this handful of observed people named" wants; None means everybody.
    """
    query = """
        SELECT s.identity_id, s.subject, s.memory_id
        FROM memory_semantics AS s
        JOIN memory_versions AS v ON v.memory_id = s.memory_id
        JOIN memory_records AS r ON r.memory_id = s.memory_id
        WHERE s.kind = ? AND s.identity_id IS NOT NULL
          AND v.retired_at IS NULL AND v.visible = 1 AND r.forgotten_at IS NULL
    """
    parameters: list[object] = [MemoryKind.ENTITY.value]
    if known_at is not None:
        known_text = datetime_text(known_at)
        query = query.replace(
            "AND v.retired_at IS NULL",
            "AND v.recorded_at <= ? AND (v.retired_at IS NULL OR v.retired_at > ?)",
        )
        parameters.extend((known_text, known_text))
    if valid_at is not None:
        valid_text = datetime_text(valid_at)
        query += " AND (v.valid_from IS NULL OR v.valid_from <= ?) AND (v.valid_until IS NULL OR v.valid_until > ?)"
        parameters.extend((valid_text, valid_text))
    if identity_ids is None:
        return tuple(connection.execute(query, parameters).fetchall())
    wanted = tuple(dict.fromkeys(identity_ids))
    rows: list[sqlite3.Row] = []
    for offset in range(0, len(wanted), SQLITE_PARAMETER_BATCH):
        batch = wanted[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _identity_id in batch)
        rows.extend(
            connection.execute(
                f"{query} AND s.identity_id IN ({placeholders})",
                (*parameters, *batch),
            ).fetchall()
        )
    return tuple(rows)


def visible_named_identities(
    connection: sqlite3.Connection,
    *,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> dict[str, tuple[str, str]]:
    """Return, per identity, the `(name, naming_memory_id)` its visible naming assertion gives.

    An identity with no visible assertion, or one whose subject is somehow unset, is simply
    absent -- `named_actors` treats a missing key as "not currently named".
    """
    named: dict[str, tuple[str, str]] = {}
    for row in visible_naming_assertions(connection, valid_at=valid_at, known_at=known_at):
        name = optional_row_text(row, "subject")
        if name is not None:
            named[row_text(row, "identity_id")] = (name, row_text(row, "memory_id"))
    return named


def merge_named_identity_links(
    connection: sqlite3.Connection,
    batch: Sequence[str],
    named: Mapping[str, tuple[str, str]],
    links: dict[str, list[tuple[str, str, str]]],
) -> None:
    """Resolve one batch of memories' identity edges and add the named ones to `links`.

    The identity edge is read two ways in one query: the asset-keyed speech speaker or face
    observation `provisional_identities` reads for the unnamed case, and the memory's own
    bound semantic assertion (`memory_semantics.identity_id`). A memory naming itself -- the
    naming assertion's own row -- is skipped, because it already renders as its own actors hit.
    """
    placeholders = ", ".join("?" for _memory_id in batch)
    rows = connection.execute(
        f"""
        SELECT ma.memory_id AS memory_id, sp.speaker_id AS identity_id
        FROM memory_assets AS ma
        JOIN speech_segments AS sp ON sp.asset_id = ma.asset_id
        WHERE ma.memory_id IN ({placeholders}) AND sp.speaker_id IS NOT NULL
        UNION
        SELECT ma.memory_id AS memory_id, f.identity_id AS identity_id
        FROM memory_assets AS ma
        JOIN face_observations AS f ON f.asset_id = ma.asset_id
        WHERE ma.memory_id IN ({placeholders})
        UNION
        SELECT s.memory_id AS memory_id, s.identity_id AS identity_id
        FROM memory_semantics AS s
        WHERE s.memory_id IN ({placeholders}) AND s.identity_id IS NOT NULL
        """,
        (*batch, *batch, *batch),
    ).fetchall()
    for row in rows:
        identity_id = row_text(row, "identity_id")
        entry = named.get(identity_id)
        if entry is None:
            continue
        memory_id = row_text(row, "memory_id")
        name, naming_memory_id = entry
        if memory_id == naming_memory_id:
            continue
        bucket = links.setdefault(memory_id, [])
        if not any(identity_id == existing[0] for existing in bucket):
            bucket.append((identity_id, name, naming_memory_id))


def has_naming_assertion(connection: sqlite3.Connection, identity_id: str) -> bool:
    """Return whether any naming assertion, visible or not, is bound to this identity.

    A merge only recomputes a projection when there is an assertion to project. An identity
    named through the store's own `register_identity`, with no assertion behind it, keeps the
    name the merge plan chose -- which is always the surviving identity's own name.
    """
    return (
        connection.execute(
            "SELECT 1 FROM memory_semantics WHERE identity_id = ? AND kind = ? LIMIT 1",
            (identity_id, MemoryKind.ENTITY.value),
        ).fetchone()
        is not None
    )


def canonical_subject(subject: str | None) -> str | None:
    """Normalize a semantic subject the one way the whole kernel compares them."""
    if subject is None:
        return None
    canonical = unicodedata.normalize("NFKC", subject).casefold().strip()
    return canonical or None


def naming_assertion_identities(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return the identities these records name, for the records that name one."""
    ids = tuple(dict.fromkeys(memory_ids))
    found: list[str] = []
    for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH):
        batch = ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        found.extend(
            row_text(row, "identity_id")
            for row in connection.execute(
                f"""
                SELECT DISTINCT identity_id
                FROM memory_semantics
                WHERE kind = ? AND identity_id IS NOT NULL
                  AND memory_id IN ({placeholders})
                """,
                (MemoryKind.ENTITY.value, *batch),
            ).fetchall()
        )
    return tuple(dict.fromkeys(found))


def reproject_identities(
    connection: sqlite3.Connection,
    identity_ids: Sequence[str],
) -> tuple[str, ...]:
    """Recompute `identities.name` from each identity's current visible naming assertion.

    This is the one rule behind the projection: whenever a bound naming assertion is written,
    retired, hidden, re-pointed or deleted, the registry is recomputed in the same transaction,
    so `identities.name` is never a name no visible assertion supports. Returns the identities
    whose projection actually moved, which is what a caller has to repaint indexed text for.

    `identities.updated_at` is transaction time and is taken here rather than from the caller.
    Every path that moves the projection carries a semantic time of its own -- a record's
    `recorded_at`, an operation's `applied_at`, a `forgotten_at` a host may deliberately backdate
    -- and none of them says when this row changed. Stamping one of those could put `updated_at`
    before the row's own `created_at`, which the schema refuses.
    """
    now = datetime_text(datetime.now(timezone.utc))
    changed: list[str] = []
    for identity_id in dict.fromkeys(identity_ids):
        current = connection.execute(
            "SELECT name, relationship FROM identities WHERE identity_id = ?",
            (identity_id,),
        ).fetchone()
        if current is None:
            continue
        row = current_naming_assertion(connection, identity_id)
        projected = (
            (None, None)
            if row is None
            else (optional_row_text(row, "subject"), optional_row_text(row, "value"))
        )
        if projected == (
            optional_row_text(current, "name"),
            optional_row_text(current, "relationship"),
        ):
            continue
        connection.execute(
            """
            UPDATE identities
            SET name = ?, relationship = ?, updated_at = ?
            WHERE identity_id = ?
            """,
            (*projected, now, identity_id),
        )
        changed.append(identity_id)
    return tuple(changed)


def reproject_named_identities(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> tuple[str, ...]:
    """Reproject every identity named by one of these records. The projection hook."""
    identity_ids = naming_assertion_identities(connection, memory_ids)
    if not identity_ids:
        return ()
    return reproject_identities(connection, identity_ids)


def rebind_unlinked_claims(
    connection: sqlite3.Connection,
    *,
    target: str,
    restored: str,
    modality: Literal["face", "voice"],
) -> tuple[str, ...]:
    """Re-evaluate every claim bound to a person a merge is being undone for.

    A merge made two people one, and claims derived while they were one are bound to the
    survivor. Splitting them re-examines exactly the claims whose evidence involves media that
    just moved back: a claim resting only on media that is now solely the restored person's is
    re-attributed to them, and a claim resting on media the two still share, or on both
    people's media, is unbound. Neither is left attributed to someone it was never about. A
    claim whose evidence never touched the restored person is untouched, and naming assertions
    stay with the survivor, which is the identity that keeps the profile.

    Returns the memories whose binding changed. Nothing here alters indexed text: the binding
    is a semantic column, and the projected name is repainted by the caller's index refresh.
    """
    restored_assets = _identity_asset_ids(connection, restored)
    # A clip that still shows the survivor is not evidence about the restored person alone,
    # even though their modality moved out of it.
    moved_assets = restored_assets - _identity_asset_ids(connection, target)
    if not restored_assets:
        return ()
    changed: list[str] = []
    for row in connection.execute(
        """
        SELECT memory_id FROM memory_semantics
        WHERE identity_id = ? AND kind <> ?
        ORDER BY memory_id
        """,
        (target, MemoryKind.ENTITY.value),
    ).fetchall():
        memory_id = row_text(row, "memory_id")
        assets = {
            asset_id
            for source_id in current_evidence_ids(connection, memory_id)
            for asset_id in _memory_asset_ids(connection, source_id)
        }
        if not assets & restored_assets:
            continue
        connection.execute(
            "UPDATE memory_semantics SET identity_id = ? WHERE memory_id = ?",
            (restored if assets <= moved_assets else None, memory_id),
        )
        changed.append(memory_id)
    return tuple(changed)


def _identity_asset_ids(connection: sqlite3.Connection, identity_id: str) -> set[str]:
    """Return every media asset that observed one identity, seen or heard."""
    return {
        row_text(row, "asset_id")
        for row in connection.execute(
            """
            SELECT DISTINCT asset_id FROM speech_segments WHERE speaker_id = ?
            UNION
            SELECT DISTINCT asset_id FROM face_observations WHERE identity_id = ?
            """,
            (identity_id, identity_id),
        ).fetchall()
    }


def _memory_asset_ids(connection: sqlite3.Connection, memory_id: str) -> tuple[str, ...]:
    """Return the media assets one memory references, in stored order."""
    return tuple(
        row_text(row, "asset_id")
        for row in connection.execute(
            "SELECT asset_id FROM memory_assets WHERE memory_id = ? ORDER BY position",
            (memory_id,),
        ).fetchall()
    )


def resolve_identity_id(connection: sqlite3.Connection, identity_id: str) -> str | None:
    row = connection.execute(
        """
        SELECT identity_id FROM identities WHERE identity_id = ?
        UNION ALL
        SELECT identity_id FROM identity_aliases WHERE alias_id = ?
        LIMIT 1
        """,
        (identity_id, identity_id),
    ).fetchone()
    return None if row is None else row_text(row, "identity_id")


def sole_identity_modality(
    connection: sqlite3.Connection,
    identity_id: str,
) -> Literal["face", "voice"] | None:
    """Return the only modality an identity holds, or None when it holds several."""
    rows = connection.execute(
        "SELECT DISTINCT modality FROM identity_exemplars WHERE identity_id = ?",
        (identity_id,),
    ).fetchall()
    if len(rows) != 1:
        return None
    return validated_identity_modality(rows[0]["modality"])


def merge_identity_exemplars(
    connection: sqlite3.Connection,
    target: str,
    source: str,
) -> None:
    """Move every exemplar of one identity onto another, keeping each bound intact.

    A shared-modality merge would collide on (identity_id, modality, position), so the
    source's exemplars are appended after the positions the target already holds.
    """
    for modality, limit in (("face", FACE_EXEMPLAR_LIMIT), ("voice", VOICE_EXEMPLAR_LIMIT)):
        offset = connection.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1
            FROM identity_exemplars
            WHERE identity_id = ? AND modality = ?
            """,
            (target, modality),
        ).fetchone()[0]
        connection.execute(
            """
            UPDATE identity_exemplars
            SET identity_id = ?, position = position + ?
            WHERE identity_id = ? AND modality = ?
            """,
            (target, int(offset), source, modality),
        )
        # ponytail: an overflowing merge keeps the target's exemplars by position rather
        # than re-running the diversity selection; the next observation of this identity
        # re-selects within the bound anyway.
        connection.execute(
            """
            DELETE FROM identity_exemplars
            WHERE identity_id = ? AND modality = ? AND position >= ?
            """,
            (target, modality, limit),
        )


def accepted_identity(
    exemplars_by_identity: dict[str, list[tuple[tuple[float, ...], str]]],
    vectors: Sequence[tuple[float, ...]],
    *,
    claimed: set[str],
    minimum_similarity: float,
    minimum_margin: float,
) -> tuple[str, float] | None:
    ranked = sorted(
        (
            (
                identity_id,
                max(
                    math.fsum(a * b for a, b in zip(vector, stored, strict=True))
                    for vector in vectors
                    for stored, _created_at in exemplars
                ),
            )
            for identity_id, exemplars in exemplars_by_identity.items()
            if identity_id not in claimed
        ),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked or ranked[0][1] < minimum_similarity:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < minimum_margin:
        return None
    return ranked[0]


def write_identity_exemplars(
    connection: sqlite3.Connection,
    identity_id: str,
    stored: Sequence[tuple[tuple[float, ...], str]],
    observed: Sequence[tuple[float, ...]],
    *,
    modality: Literal["face", "voice"],
    model_id: str,
    space_id: str,
    dimension: int,
    exemplar_limit: int,
    identity_exists: bool,
    now_text: str,
) -> list[tuple[tuple[float, ...], str]]:
    if not identity_exists:
        connection.execute(
            "INSERT INTO identities (identity_id, created_at, updated_at) VALUES (?, ?, ?)",
            (identity_id, now_text, now_text),
        )
    exemplars = list(stored)
    for vector in observed:
        vector = normalized_vector(
            unpack_vector(pack_vector(vector), dimension),
            f"{modality} exemplar",
        )
        if all(vector != existing for existing, _created_at in exemplars):
            exemplars.append((vector, now_text))
    selected = _diverse_exemplars(exemplars, limit=exemplar_limit)
    connection.execute(
        "DELETE FROM identity_exemplars WHERE identity_id = ? AND modality = ?",
        (identity_id, modality),
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
                identity_id,
                modality,
                position,
                model_id,
                space_id,
                dimension,
                pack_vector(vector),
                created_at,
            )
            for position, (vector, created_at) in enumerate(selected)
        ),
    )
    if identity_exists:
        connection.execute(
            "UPDATE identities SET updated_at = ? WHERE identity_id = ?",
            (now_text, identity_id),
        )
    return selected


def _diverse_exemplars(
    exemplars: Sequence[tuple[tuple[float, ...], str]],
    *,
    limit: int,
) -> list[tuple[tuple[float, ...], str]]:
    selected = list(exemplars)
    while len(selected) > limit:
        sums = tuple(
            math.fsum(vector[index] for vector, _created_at in selected)
            for index in range(len(selected[0][0]))
        )
        magnitude = math.sqrt(math.fsum(value * value for value in sums))
        if magnitude:
            centroid = tuple(value / magnitude for value in sums)
            remove = max(
                range(len(selected)),
                key=lambda index: (
                    math.fsum(
                        value * center
                        for value, center in zip(selected[index][0], centroid, strict=True)
                    ),
                    -index,
                ),
            )
        else:
            remove = max(
                range(len(selected)),
                key=lambda index: (
                    max(
                        math.fsum(
                            left * right
                            for left, right in zip(selected[index][0], candidate[0], strict=True)
                        )
                        for candidate_index, candidate in enumerate(selected)
                        if candidate_index != index
                    ),
                    -index,
                ),
            )
        selected.pop(remove)
    return selected
