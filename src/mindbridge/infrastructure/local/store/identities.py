"""People the memories are about: registration, naming, consent, linking, erasure."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Literal

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    datetime_text,
    optional_row_text,
    prepare_write_batch,
    row_text,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._identity import (
    canonical_subject,
    current_consent_state,
    current_naming_assertion,
    has_naming_assertion,
    merge_identity_exemplars,
    merge_named_identity_links,
    naming_assertion_identities,
    rebind_unlinked_claims,
    reproject_identities,
    resolve_identity_id,
    restrained_identities,
    sole_identity_modality,
    visible_named_identities,
    visible_naming_assertions,
)
from mindbridge.infrastructure.local.store._lineage import (
    SOURCE_GROUP_QUERY,
    active_evidence_dependent_closure,
    current_evidence_ids,
    deletion_cascade,
    semantic_visibility,
)
from mindbridge.infrastructure.local.store._operations import active_operation_id, insert_operation
from mindbridge.infrastructure.local.store._outbox import queue_identity_projection
from mindbridge.infrastructure.local.store._selection import require_scope_axes
from mindbridge.infrastructure.local.store.records import (
    MemoryRecords,
    delete_memory,
    replace_memory_embeddings,
)
from mindbridge.infrastructure.local.store.rows import (
    IdentityLink,
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    StoredOperation,
    require_identifier,
    require_optional_identifier,
    validated_identity_modality,
    validated_identity_name,
    validated_sha256,
)
from mindbridge.types import (
    ConsentState,
    IdentityErasure,
    IdentityProfile,
    MemoryContext,
    MemoryKind,
    SpatialContext,
)


def merge_identity_rows(
    connection: sqlite3.Connection,
    plan: IdentityLink,
) -> tuple[str, ...]:
    """Apply one validated merge inside an open transaction.

    Returns the naming assertions that moved onto the survivor, which is the one effect a
    merge has on memory records and therefore the one an operation log row can name.
    """
    target, source = plan.target_id, plan.source_id
    contributed = sole_identity_modality(connection, source)
    now = datetime_text(datetime.now(timezone.utc))
    # Read before the source's `identities` row is deleted below, so the alias row can carry
    # forward the absorbed identity's own creation time rather than the merge time. Rollback
    # (`split_identity_rows`) restores the identity from this value; storing `now` here instead
    # would make every merge-then-rollback cycle overwrite the original `created_at`.
    source_created_at = row_text(
        connection.execute(
            "SELECT created_at FROM identities WHERE identity_id = ?",
            (source,),
        ).fetchone(),
        "created_at",
    )
    # Before the re-pointing below, while the rows the index projection reads still say
    # `source`: every memory it names is about `target` afterwards, and only the outbox can
    # tell the index that.
    queue_identity_projection(connection, source)
    connection.execute(
        "UPDATE speech_segments SET speaker_id = ? WHERE speaker_id = ?",
        (target, source),
    )
    connection.execute(
        "UPDATE face_observations SET identity_id = ? WHERE identity_id = ?",
        (target, source),
    )
    merge_identity_exemplars(connection, target, source)
    connection.execute(
        """
        UPDATE identities
        SET created_at = ?, updated_at = ?
        WHERE identity_id = ?
        """,
        (plan.created_at, now, target),
    )
    moved_claims = tuple(
        row_text(row, "memory_id")
        for row in connection.execute(
            "SELECT memory_id FROM memory_semantics WHERE identity_id = ? ORDER BY memory_id",
            (source,),
        ).fetchall()
    )
    # The source's naming assertions move to the survivor before the source row goes,
    # or `ON DELETE SET NULL` would strand the name that was asserted about this person
    # and the projection would go on reporting a name nothing supports. The name is a
    # projection here too: recomputed from the assertions, never assigned.
    connection.execute(
        "UPDATE memory_semantics SET identity_id = ? WHERE identity_id = ?",
        (target, source),
    )
    connection.execute(
        "UPDATE identity_aliases SET identity_id = ? WHERE identity_id = ?",
        (target, source),
    )
    # Re-point accumulated evidence by copying it onto the target first: the
    # same asset may already carry a row for the target, and the primary key
    # would reject a plain UPDATE.
    connection.execute(
        """
        INSERT INTO identity_link_evidence (voice_id, face_id, asset_id, created_at)
        SELECT CASE WHEN voice_id = ? THEN ? ELSE voice_id END,
               CASE WHEN face_id = ? THEN ? ELSE face_id END,
               asset_id,
               created_at
        FROM identity_link_evidence
        WHERE voice_id = ? OR face_id = ?
        ON CONFLICT (voice_id, face_id, asset_id) DO NOTHING
        """,
        (source, target, source, target, source, source),
    )
    connection.execute(
        "DELETE FROM identity_link_evidence WHERE voice_id = ? OR face_id = ?",
        (source, source),
    )
    # Re-pointing turns this pair's own evidence into voice_id = face_id rows, which
    # describe an identity co-occurring with itself and can never yield a plan.
    connection.execute("DELETE FROM identity_link_evidence WHERE voice_id = face_id")
    connection.execute(
        """
        INSERT INTO identity_aliases (
            alias_id, identity_id, created_at, contributed_modality
        ) VALUES (?, ?, ?, ?)
        """,
        (source, target, source_created_at, contributed),
    )
    connection.execute("DELETE FROM identities WHERE identity_id = ?", (source,))
    if has_naming_assertion(connection, target):
        reproject_identities(connection, (target,))
    return moved_claims


def split_identity_rows(connection: sqlite3.Connection, alias_id: str) -> str | None:
    """Reverse one recorded merge inside an open transaction.

    Returns the restored identity, or None -- having changed nothing -- when the alias has
    no recorded split point or the survivor would be left with no exemplars.
    """
    now = datetime_text(datetime.now(timezone.utc))
    alias = connection.execute(
        """
        SELECT identity_id, created_at, contributed_modality
        FROM identity_aliases
        WHERE alias_id = ?
        """,
        (alias_id,),
    ).fetchone()
    if alias is None or alias["contributed_modality"] is None:
        return None
    target = row_text(alias, "identity_id")
    modality = validated_identity_modality(alias["contributed_modality"])
    modalities = {
        validated_identity_modality(row["modality"])
        for row in connection.execute(
            """
            SELECT DISTINCT modality
            FROM identity_exemplars
            WHERE identity_id = ?
            """,
            (target,),
        ).fetchall()
    }
    if modalities != {"face", "voice"}:
        return None
    # Same reason as the merge: the rows below move from `target` to `alias_id`, so the
    # index projection of every memory naming `target` has to be rebuilt.
    queue_identity_projection(connection, target)
    connection.execute(
        """
        INSERT INTO identities (identity_id, name, relationship, created_at, updated_at)
        VALUES (?, NULL, NULL, ?, ?)
        """,
        (alias_id, row_text(alias, "created_at"), now),
    )
    connection.execute(
        """
        UPDATE identity_exemplars
        SET identity_id = ?
        WHERE identity_id = ? AND modality = ?
        """,
        (alias_id, target, modality),
    )
    if modality == "face":
        connection.execute(
            "UPDATE face_observations SET identity_id = ? WHERE identity_id = ?",
            (alias_id, target),
        )
    else:
        connection.execute(
            "UPDATE speech_segments SET speaker_id = ? WHERE speaker_id = ?",
            (alias_id, target),
        )
    connection.execute("DELETE FROM identity_aliases WHERE alias_id = ?", (alias_id,))
    connection.execute(
        """
        DELETE FROM identity_link_evidence
        WHERE voice_id IN (?, ?) AND face_id IN (?, ?)
        """,
        (target, alias_id, target, alias_id),
    )
    rebind_unlinked_claims(
        connection,
        target=target,
        restored=alias_id,
        modality=modality,
    )
    if has_naming_assertion(connection, target):
        reproject_identities(connection, (target,))
    return alias_id


def _identity_equivalence_class(
    connection: sqlite3.Connection,
    identity_id: str,
) -> tuple[str, ...] | None:
    resolved_id = resolve_identity_id(connection, identity_id)
    if resolved_id is None:
        return None
    aliases = connection.execute(
        "SELECT alias_id FROM identity_aliases WHERE identity_id = ? ORDER BY alias_id",
        (resolved_id,),
    ).fetchall()
    return (resolved_id, *(row_text(row, "alias_id") for row in aliases))


def identity_link_plan(
    connection: sqlite3.Connection,
    first_id: str,
    second_id: str,
    *,
    allow_shared_modality: bool = False,
) -> IdentityLink | None:
    first = resolve_identity_id(connection, first_id)
    second = resolve_identity_id(connection, second_id)
    if first is None or second is None or first == second:
        return None
    identities = (first, second)
    rows = connection.execute(
        """
        SELECT identity_id, name, created_at
        FROM identities
        WHERE identity_id IN (?, ?)
        ORDER BY identity_id
        """,
        identities,
    ).fetchall()
    if len(rows) != 2:
        return None
    modalities = {
        identity_id: {
            validated_identity_modality(row["modality"])
            for row in connection.execute(
                """
                SELECT DISTINCT modality
                FROM identity_exemplars
                WHERE identity_id = ?
                """,
                (identity_id,),
            ).fetchall()
        }
        for identity_id in identities
    }
    if not modalities[first] or not modalities[second]:
        return None
    # A shared modality only stops blocking the plan while one side is still a
    # single-modality fragment being re-absorbed. Two identities that both hold
    # face and voice stay refused: fusing two complete people is worse than
    # leaving a fragment orphaned.
    fragment = min(len(modalities[first]), len(modalities[second])) == 1
    if modalities[first] & modalities[second] and not (allow_shared_modality and fragment):
        return None
    names = {
        row_text(row, "identity_id"): (None if row["name"] is None else row_text(row, "name"))
        for row in rows
    }
    if names[first] is not None and names[second] not in {None, names[first]}:
        return None
    created = {row_text(row, "identity_id"): row_text(row, "created_at") for row in rows}
    target, source = sorted(
        identities,
        key=lambda identity_id: (
            names[identity_id] is None,
            created[identity_id],
            identity_id,
        ),
    )
    return IdentityLink(
        target,
        source,
        names[target] or names[source],
        min(created[target], created[source]),
    )


class IdentityRegistry:
    """Identity rows, their projections, and their lifecycle."""

    def __init__(
        self,
        *,
        connections: Connections,
        records: MemoryRecords,
    ) -> None:
        self._connections = connections
        self._records = records

    def naming_projection_after_delete(
        self,
        memory_id: str,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str | None], ...]]:
        """Return what deleting this record removes, and the name each identity is left with.

        A naming assertion is an ordinary memory record, so an ordinary caller can delete it --
        and so is a record an assertion cites, which takes the assertion with it when it was
        the last evidence. Both move a projection, and the caller has to rebuild the indexed
        text that quoted the name in the same commit. This is how it finds out which names
        change, to what, and which records it must not hand back as replacements because this
        delete is about to remove them.
        """
        require_identifier(memory_id, "memory_id")
        with self._connections.connection() as connection:
            removed = deletion_cascade(connection, (memory_id,))
            affected = active_evidence_dependent_closure(connection, memory_id)
            identities = naming_assertion_identities(connection, (memory_id, *affected))
            connection.execute("SAVEPOINT deletion_projection")
            try:
                delete_memory(connection, memory_id)
                projection = tuple(
                    (
                        identity_id,
                        optional_row_text(row, "name")
                        if (
                            row := connection.execute(
                                "SELECT name FROM identities WHERE identity_id = ?",
                                (identity_id,),
                            ).fetchone()
                        )
                        is not None
                        else None,
                    )
                    for identity_id in identities
                )
            finally:
                connection.execute("ROLLBACK TO deletion_projection")
                connection.execute("RELEASE deletion_projection")
        return removed, projection

    def resolve_identity_id(self, identity_id: str) -> str | None:
        """Resolve a current or merged identity ID to its canonical identity."""
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            return resolve_identity_id(connection, identity_id)

    def speaker_memory_ids(self, speaker_id: str) -> tuple[str, ...] | None:
        """Return memories containing a speaker, or none when the identity is unknown."""
        require_identifier(speaker_id, "speaker_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, speaker_id)
            if resolved_id is None:
                return None
            if (
                connection.execute(
                    """
                    SELECT 1
                    FROM identity_exemplars
                    WHERE identity_id = ? AND modality = 'voice'
                    LIMIT 1
                    """,
                    (resolved_id,),
                ).fetchone()
                is None
            ):
                return None
            rows = connection.execute(
                """
                SELECT DISTINCT ma.memory_id
                FROM speech_segments AS s
                JOIN memory_assets AS ma ON ma.asset_id = s.asset_id
                WHERE s.speaker_id = ?
                ORDER BY ma.memory_id
                """,
                (resolved_id,),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)

    def identity_memory_ids(self, identity_id: str) -> tuple[str, ...] | None:
        """Return memories containing a face or voice occurrence for one identity."""
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            rows = connection.execute(
                """
                SELECT DISTINCT ma.memory_id
                FROM memory_assets AS ma
                WHERE EXISTS (
                    SELECT 1 FROM speech_segments AS s
                    WHERE s.asset_id = ma.asset_id AND s.speaker_id = ?
                ) OR EXISTS (
                    SELECT 1 FROM face_observations AS f
                    WHERE f.asset_id = ma.asset_id AND f.identity_id = ?
                )
                ORDER BY ma.memory_id
                """,
                (resolved_id, resolved_id),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)

    def register_speaker(
        self,
        speaker_id: str,
        name: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Assign a display name and atomically replace affected indexed memories."""
        require_identifier(speaker_id, "speaker_id")
        return self.register_identity(
            speaker_id,
            name,
            memories=memories,
            embeddings=embeddings,
        )

    def register_identity(
        self,
        identity_id: str,
        name: str,
        *,
        relationship: str | None = None,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Assign a display name and atomically replace affected indexed memories.

        A None relationship keeps whatever relationship is already recorded, so
        renaming an identity never silently drops it.
        """
        require_identifier(identity_id, "identity_id")
        normalized_name = validated_identity_name(name)
        normalized_relationship = (
            None if relationship is None else validated_identity_name(relationship, "relationship")
        )
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        now = datetime_text(datetime.now(timezone.utc))
        with self._connections.transaction() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return False
            cursor = connection.execute(
                """
                UPDATE identities
                SET name = ?,
                    relationship = COALESCE(?, relationship),
                    updated_at = ?
                WHERE identity_id = ?
                """,
                (normalized_name, normalized_relationship, now, resolved_id),
            )
            if cursor.rowcount == 0:
                return False
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        return cursor.rowcount > 0

    def projected_identity_name(
        self,
        identity_id: str,
        *,
        excluding: Sequence[str] = (),
    ) -> tuple[str | None, str | None]:
        """Return the name and relationship the current visible naming assertion projects.

        Both are `None` for an unknown or as yet unnamed identity, which is exactly what the
        projection should then write. `excluding` ignores assertions the caller is about to
        delete, so it can rebuild indexed text for the projection the deletion will leave.
        """
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            row = (
                None
                if resolved_id is None
                else current_naming_assertion(connection, resolved_id, excluding=excluding)
            )
        if row is None:
            return None, None
        return optional_row_text(row, "subject"), optional_row_text(row, "value")

    def naming_projection_after_assertion(
        self,
        identity_id: str,
        memory_id: str,
        context: MemoryContext,
    ) -> tuple[str | None, str | None]:
        """Preview the projection after applying one bound naming assertion.

        The caller uses this while holding its write lock to build speech documents before the
        assertion transaction. The store remains the source of the evidence-group visibility
        rule, so the preview and the commit cannot disagree about corroboration.
        """
        require_identifier(identity_id, "identity_id")
        require_identifier(memory_id, "memory_id")
        if context.kind is not MemoryKind.ENTITY or context.identity_id != identity_id:
            raise ValueError("context must be a naming assertion for identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None, None
            groups = {
                row_text(row, "source_group_id")
                for row in connection.execute(
                    """
                    SELECT source_group_id FROM memory_evidence
                    WHERE memory_id = ? AND retired_at IS NULL
                    """,
                    (memory_id,),
                ).fetchall()
            }
            for source_memory_id in context.evidence_ids:
                source = connection.execute(SOURCE_GROUP_QUERY, (source_memory_id,)).fetchone()
                if source is None:
                    raise sqlite3.IntegrityError("evidence source memory does not exist")
                groups.add(row_text(source, "source_group_id"))
            visible = semantic_visibility(
                connection,
                memory_id=memory_id,
                lineage_id=context.lineage_id or memory_id,
                kind=context.kind.value,
                basis=context.basis.value,
                identity_id=context.identity_id,
                evidence_count=len(groups),
                valid_from=context.valid_from,
                valid_until=context.valid_until,
            )
            if visible:
                return context.subject, context.value
            current = current_naming_assertion(connection, resolved_id)
        if current is None:
            return None, None
        return optional_row_text(current, "subject"), optional_row_text(current, "value")

    def refresh_identity_projection(
        self,
        identity_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> IdentityProfile | None:
        """Recompute `identities.name`/`relationship` from the current naming assertion.

        The assertion is the record and this is the read path, the way `memory_versions`
        carries the evidence and `refresh_evidence_projection` recomputes what reads see.
        Supplied documents replace their indexed vectors in the same transaction, so stored
        text can never disagree with the projection it was rebuilt for. Returns the projected
        profile, or None when the identity does not exist.
        """
        require_identifier(identity_id, "identity_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        now = datetime_text(datetime.now(timezone.utc))
        with self._connections.transaction() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            row = current_naming_assertion(connection, resolved_id)
            name = None if row is None else optional_row_text(row, "subject")
            relationship = None if row is None else optional_row_text(row, "value")
            cursor = connection.execute(
                """
                UPDATE identities
                SET name = ?, relationship = ?, updated_at = ?
                WHERE identity_id = ?
                """,
                (name, relationship, now, resolved_id),
            )
            if cursor.rowcount == 0:
                return None
            evidence_ids = (
                () if row is None else current_evidence_ids(connection, row_text(row, "memory_id"))
            )
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        return IdentityProfile(
            identity_id=resolved_id,
            name=name,
            relationship=relationship,
            confirmed=row is not None,
            evidence_ids=evidence_ids,
        )

    def identity_for_subject(self, subject: str) -> str | None:
        """Return the identity whose visible naming assertion canonically names this subject.

        Deterministic and never a model's decision: the comparison is the same NFKC casefold
        the semantic layer already uses, and two identities that currently project the same
        canonical name resolve to neither, because binding a claim to the wrong person is
        worse than leaving it unbound.
        """
        canonical = canonical_subject(subject)
        if canonical is None:
            return None
        with self._connections.connection() as connection:
            matches = {
                row_text(row, "identity_id")
                for row in visible_naming_assertions(connection)
                if canonical_subject(optional_row_text(row, "subject")) == canonical
            }
        return matches.pop() if len(matches) == 1 else None

    def provisional_identities(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return, per memory, the identities it observes that no visible assertion names.

        A person a memory saw or heard but nobody has named yet: present in the evidence and
        absent from every projection. Empty entries are dropped, so a caller can treat a
        missing key as "nobody unnamed here".
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            require_identifier(memory_id, "memory_id")
        if not ids:
            return {}
        provisional: dict[str, tuple[str, ...]] = {}
        observed: list[tuple[str, str]] = []
        with self._connections.connection() as connection:
            for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH // 2):
                batch = ids[offset : offset + SQLITE_PARAMETER_BATCH // 2]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows = connection.execute(
                    f"""
                    SELECT ma.memory_id AS memory_id, s.speaker_id AS identity_id
                    FROM memory_assets AS ma
                    JOIN speech_segments AS s ON s.asset_id = ma.asset_id
                    WHERE ma.memory_id IN ({placeholders}) AND s.speaker_id IS NOT NULL
                    UNION
                    SELECT ma.memory_id AS memory_id, f.identity_id AS identity_id
                    FROM memory_assets AS ma
                    JOIN face_observations AS f ON f.asset_id = ma.asset_id
                    WHERE ma.memory_id IN ({placeholders})
                    """,
                    (*batch, *batch),
                ).fetchall()
                observed.extend(
                    (row_text(row, "memory_id"), row_text(row, "identity_id")) for row in rows
                )
            # Asked only about the people actually observed here: the assertion scan is over
            # every named person otherwise, to answer about a handful.
            named = {
                row_text(row, "identity_id")
                for row in visible_naming_assertions(
                    connection,
                    identity_ids=tuple(identity_id for _memory_id, identity_id in observed),
                    valid_at=valid_at,
                    known_at=known_at,
                )
            }
        for memory_id, identity_id in observed:
            if identity_id not in named:
                provisional[memory_id] = (*provisional.get(memory_id, ()), identity_id)
        return {
            memory_id: tuple(sorted(set(identity_ids)))
            for memory_id, identity_ids in sorted(provisional.items())
        }

    def named_actors(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
    ) -> dict[str, tuple[tuple[str, str, str | None], ...]]:
        """Return, per memory, the NAMED identities its identity edge resolves to.

        A memory carries an identity edge two ways: its own semantic assertion may be bound to
        an identity (`memory_semantics.identity_id`, the field `MemoryContext.identity_id`
        projects), or its media asset may have recognized one -- the same asset-keyed join
        `provisional_identities` reads for the unnamed case. Each entry is `(identity_id,
        name, naming_memory_id)`. The naming assertion's own memory is never reported for
        itself, because it already renders as its own actors hit. Empty entries are dropped,
        so a caller can treat a missing key as "nobody named here".
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            require_identifier(memory_id, "memory_id")
        if not ids:
            return {}
        with self._connections.connection() as connection:
            named = visible_named_identities(connection, valid_at=valid_at, known_at=known_at)
            if not named:
                return {}
            links: dict[str, list[tuple[str, str, str]]] = {}
            for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH // 3):
                batch = ids[offset : offset + SQLITE_PARAMETER_BATCH // 3]
                merge_named_identity_links(connection, batch, named, links)
        return {memory_id: tuple(sorted(entries)) for memory_id, entries in sorted(links.items())}

    def co_derived_events(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Return, per memory, the active event records formed from the same observations.

        The edge is shared evidence: an event is reported for a memory exactly when both cite
        one `memory_evidence.source_memory_id`. That is co-occurrence inside one capture and
        not a cause, and the memory itself is never its own co-derived event. The candidates are
        then read through `read_memories(active_only=True)` under every scope axis a search
        hydration applies, so a retired version, a hidden assertion, a forgotten record, and
        anything outside the asked-for validity, pose, or place are all left out. That second
        read is its own transaction: a `known_at` pins what both see, and without one a write
        landing between them can only remove an event from the hop, never add one the join did
        not already hold. Empty entries are dropped, so a missing key means no event shares this
        memory's observations.
        """
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            require_identifier(memory_id, "memory_id")
        # Eagerly, because the hydration that would otherwise validate these is skipped whenever
        # the join finds nothing.
        require_scope_axes(valid_at=valid_at, known_at=known_at, near=near, radius_m=radius_m)
        require_optional_identifier(place_id, "place_id")
        require_optional_identifier(identity_id, "identity_id")
        if not ids:
            return {}
        # Both evidence sides are read as of the same transaction time the versions are, so a
        # link retired after `known_at` still joins the pair it joined then.
        current = "{alias}.retired_at IS NULL"
        as_of = (
            "{alias}.recorded_at <= ? AND ({alias}.retired_at IS NULL OR {alias}.retired_at > ?)"
        )
        clause = current if known_at is None else as_of
        known_parameters: tuple[object, ...] = (
            () if known_at is None else (datetime_text(known_at), datetime_text(known_at))
        )
        derived: dict[str, list[str]] = {}
        # Half a batch: each statement binds the cue IDs once and the time bounds twice.
        with self._connections.read_transaction() as connection:
            for offset in range(0, len(ids), SQLITE_PARAMETER_BATCH // 2):
                batch = ids[offset : offset + SQLITE_PARAMETER_BATCH // 2]
                placeholders = ", ".join("?" for _memory_id in batch)
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT cue.memory_id AS memory_id, ev.memory_id AS event_id
                    FROM memory_evidence AS cue
                    JOIN memory_evidence AS ev
                        ON ev.source_memory_id = cue.source_memory_id
                        AND ev.memory_id <> cue.memory_id
                    JOIN memory_semantics AS s ON s.memory_id = ev.memory_id
                    WHERE cue.memory_id IN ({placeholders}) AND s.kind = ?
                      AND {clause.format(alias="cue")} AND {clause.format(alias="ev")}
                    """,
                    (*batch, MemoryKind.EVENT.value, *known_parameters, *known_parameters),
                ).fetchall():
                    derived.setdefault(row_text(row, "memory_id"), []).append(
                        row_text(row, "event_id")
                    )
        if not derived:
            return {}
        candidates = tuple(
            dict.fromkeys(event_id for events in derived.values() for event_id in events)
        )
        active = {
            memory.memory_id
            for memory in self._records.read_memories(
                candidates,
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                place_id=place_id,
                identity_id=identity_id,
                active_only=True,
            )
        }
        return {
            memory_id: standing
            for memory_id, events in sorted(derived.items())
            if (standing := tuple(sorted(event for event in events if event in active)))
        }

    def identity_profile(self, identity_id: str) -> IdentityProfile | None:
        """Return one identity's profile, or None when it does not exist.

        A merged identity resolves through its alias, like every other identity read, and the
        returned profile carries the canonical ID so a caller never has to resolve it twice.

        `confirmed` and `evidence_ids` are derived, not stored: a person is confirmed exactly
        while a visible naming assertion names them, and the evidence is that assertion's.
        """
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            row = connection.execute(
                "SELECT name, relationship FROM identities WHERE identity_id = ?",
                (resolved_id,),
            ).fetchone()
            if row is None:
                return None
            assertion = current_naming_assertion(connection, resolved_id)
            evidence_ids = (
                ()
                if assertion is None
                else current_evidence_ids(connection, row_text(assertion, "memory_id"))
            )
        return IdentityProfile(
            identity_id=resolved_id,
            name=optional_row_text(row, "name"),
            relationship=optional_row_text(row, "relationship"),
            confirmed=assertion is not None,
            evidence_ids=evidence_ids,
        )

    def identity_consent(self, identity_id: str) -> ConsentState | None:
        """Return the consent state one identity's standing assertion projects, or None.

        `None` is "nobody has recorded a statement", which is not consent and not a refusal.
        A merged alias resolves to its canonical identity, like every other identity read.
        """
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return None
            value = current_consent_state(connection, resolved_id)
        return None if value is None else ConsentState(value)

    def restrained_identities(self) -> frozenset[str]:
        """Return every identity whose standing consent withholds or withdraws processing."""
        with self._connections.connection() as connection:
            return restrained_identities(connection)

    def identity_assertion_memory_ids(self, identity_id: str) -> tuple[str, ...]:
        """Return every record bound to one identity, standing or superseded.

        This is the assertion half of what is held about a person -- their names and their
        consent statements, every version of each -- which `identity_memory_ids` does not cover
        because those records observe nobody.
        """
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return ()
            rows = connection.execute(
                """
                SELECT memory_id FROM memory_semantics
                WHERE identity_id = ? ORDER BY memory_id
                """,
                (resolved_id,),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)

    def record_identity_link_evidence(self, voice_id: str, face_id: str, asset_id: str) -> int:
        """Record one voice-and-face co-occurrence and count the pair's distinct assets.

        Recording the same triple again is idempotent. Returns zero when either
        identity is unknown, or when both resolve to the same identity because the pair has
        already merged, so a caller can accumulate corroboration across assets before it
        commits an irreversible cross-modal merge.
        """
        require_identifier(voice_id, "voice identity_id")
        require_identifier(face_id, "face identity_id")
        validated_sha256(asset_id)
        now = datetime_text(datetime.now(timezone.utc))
        with self._connections.transaction() as connection:
            voice = resolve_identity_id(connection, voice_id)
            face = resolve_identity_id(connection, face_id)
            if voice is None or face is None or voice == face:
                return 0
            if (
                connection.execute(
                    "SELECT 1 FROM media_assets WHERE asset_id = ?",
                    (asset_id,),
                ).fetchone()
                is None
            ):
                raise ValueError("identity link evidence requires a stored media asset")
            connection.execute(
                """
                INSERT INTO identity_link_evidence (voice_id, face_id, asset_id, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (voice_id, face_id, asset_id) DO NOTHING
                """,
                (voice, face, asset_id, now),
            )
            row = connection.execute(
                """
                SELECT COUNT(DISTINCT asset_id) AS assets
                FROM identity_link_evidence
                WHERE voice_id = ? AND face_id = ?
                """,
                (voice, face),
            ).fetchone()
        return int(row["assets"])

    def identity_link_plan(
        self,
        first_id: str,
        second_id: str,
        *,
        allow_shared_modality: bool = False,
    ) -> IdentityLink | None:
        """Return the currently valid merge plan for two complementary identities.

        With allow_shared_modality a shared modality no longer blocks the plan, so an
        established identity can re-absorb a single-modality fragment instead of that
        fragment staying orphaned forever. Two identities that each hold face and
        voice stay refused even then: that merge fuses two complete people and is not
        recoverable in bulk.
        """
        require_identifier(first_id, "first identity_id")
        require_identifier(second_id, "second identity_id")
        with self._connections.connection() as connection:
            return identity_link_plan(
                connection,
                first_id,
                second_id,
                allow_shared_modality=allow_shared_modality,
            )

    def link_identities(
        self,
        first_id: str,
        second_id: str,
        *,
        expected: IdentityLink | None = None,
        allow_shared_modality: bool = False,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
        operation: StoredOperation | None = None,
    ) -> str | None:
        """Merge two identities under one stable ID, recording a reversible alias.

        Pass allow_shared_modality to commit a plan obtained with the same intent.

        `operation` logs the control-plane operation that committed this merge in the same
        transaction, so a cross-modal bind is visible in the operation log and reversible
        through it. The log row carries the naming assertions the merge moved onto the survivor
        as `changed_ids`. Returns None, changing nothing, when that operation key is already
        applied and not rolled back.
        """
        require_identifier(first_id, "first identity_id")
        require_identifier(second_id, "second identity_id")
        if expected is not None and not isinstance(expected, IdentityLink):
            raise ValueError("expected must be an IdentityLink value or None")
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        with self._connections.transaction() as connection:
            if (
                operation is not None
                and active_operation_id(connection, operation.operation_key) is not None
            ):
                return None
            plan = identity_link_plan(
                connection,
                first_id,
                second_id,
                allow_shared_modality=allow_shared_modality,
            )
            if plan is None:
                first = resolve_identity_id(connection, first_id)
                second = resolve_identity_id(connection, second_id)
                return first if first is not None and first == second else None
            if expected is not None and plan != expected:
                return None
            moved_claims = merge_identity_rows(connection, plan)
            if operation is not None:
                insert_operation(connection, replace(operation, changed_ids=moved_claims))
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return plan.target_id

    def identity_alias_modality(self, alias_id: str) -> Literal["face", "voice"] | None:
        """Return the modality one merged alias contributed, or None when none is recorded.

        A caller that must rebuild derived text before `unlink_identity` has to know which
        modality is about to move back, and the alias row is the only record of it.
        """
        require_identifier(alias_id, "alias_id")
        with self._connections.connection() as connection:
            row = connection.execute(
                "SELECT contributed_modality FROM identity_aliases WHERE alias_id = ?",
                (alias_id,),
            ).fetchone()
        if row is None or row["contributed_modality"] is None:
            return None
        return validated_identity_modality(row["contributed_modality"])

    def unlink_identity(
        self,
        alias_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
        operation: StoredOperation | None = None,
    ) -> str | None:
        """Reverse one recorded merge, restoring alias_id as an independent identity.

        Returns the restored identity_id, or None when alias_id is not a reversible
        alias: an unknown alias, an alias whose source held both modalities, and an
        alias merged before this schema recorded the contributed modality all have no
        recorded split point. Also returns None, changing nothing, when the target no
        longer holds the other modality, because the split would leave it with no
        exemplars at all.

        The restored identity gets no name and no relationship: a merge keeps only one
        profile and the target keeps it, so name the restored identity again if it
        needs one. Every exemplar and observation of the contributed modality moves
        back, not only the rows the source originally supplied, so unlinking a
        re-absorbed fragment also hands over what the target learned in that modality.

        Unlinking clears the pair's accumulated link evidence but does not suppress the
        pair, so continued ingestion can corroborate and merge them again. Treat this as
        resetting the evidence, not as recording that a human rejected the merge.

        Pass `memories` and `embeddings` to atomically replace indexed documents that named
        the merged person, exactly as `register_identity` does. They are applied only when the
        unlink actually commits, so a refused unlink leaves the projection untouched.

        Claims derived while the two were one identity are re-evaluated in the same
        transaction: one resting only on media that moved back is re-attributed to the
        restored identity, one resting on both people's media is unbound, and one that never
        involved the restored modality keeps its binding.

        `operation` logs the control-plane operation that split this merge, in the same
        transaction, so a split is visible in the operation log and reversible through it. It is
        written only when the split actually commits.
        """
        require_identifier(alias_id, "alias_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        with self._connections.transaction() as connection:
            if split_identity_rows(connection, alias_id) is None:
                return None
            if operation is not None:
                insert_operation(connection, operation)
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return alias_id

    def identity_equivalence_class(self, identity_id: str) -> tuple[str, ...] | None:
        """Return every ID that names one identity: the canonical ID first, then its aliases.

        Accepts a canonical ID or any merged alias, and returns None when none of them is
        known. Note what this is *not* useful for: `link_identities` re-points every speech
        segment, face observation and exemplar onto the canonical ID, so expanding a read
        across the returned class retrieves exactly what the canonical ID alone retrieves.
        The class is the erasure and audit surface -- which IDs still admit a forgotten
        person -- not a recall lever.
        """
        require_identifier(identity_id, "identity_id")
        with self._connections.connection() as connection:
            return _identity_equivalence_class(connection, identity_id)

    def forget_identity(
        self,
        identity_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
        operation: StoredOperation | None = None,
    ) -> tuple[IdentityErasure, tuple[StoredAsset, ...]] | None:
        """Erase one person's identity cluster, keeping the memories that mention them.

        Accepts a canonical ID or any merged alias and removes the whole cluster: the profile,
        every face and voice exemplar, every alias, and the accumulated cross-modal link
        evidence. Returns the erasure and the assets no remaining memory references, the way
        `delete_memory_with_assets` does, or None, changing nothing, when no such identity
        exists.

        Memories, their content and their media assets survive -- deleting a person must not
        delete the family's memory of the events. Their identity annotations do not: speech
        segments keep their transcript with `speaker_id` scrubbed to NULL, and face
        observations are removed outright, because a face row's entire payload is a box plus
        the identity claim and stripping the claim leaves only a biometric locator. The
        cached `face_analyses`/`speech_analyses` rows deliberately stay, so re-analysing
        already stored media cannot re-mint the person from the same clip.

        No tombstone is recorded, and this deliberately does not stop a future encounter from
        minting a fresh identity. Recognising someone as previously-forgotten requires keeping
        their template, which is the one thing the request asked to destroy; a deployment that
        wants "never recognise this person again" needs a retained blocklist, which is not a
        deletion and must not be spelled like one.

        Pass `memories` and `embeddings` to atomically replace indexed documents that named the
        person, exactly as `register_identity` does -- the erasure commits in SQLite before the
        outbox tells the projection.

        `operation` logs, in the same transaction, that this erasure happened. The row is audit
        history and nothing more: it names the identity and its aliases, and the naming
        assertions the erasure deleted, so `operations()` can show that a person was erased
        without holding anything a rollback could restore.

        Recoverability, stated plainly: freed cells are zero-filled (`PRAGMA secure_delete`) and
        the write-ahead log is checkpointed and truncated afterwards, so the exemplar bytes are
        no longer present in `state.sqlite3` or its `-wal`. This store runs no `VACUUM`, and
        nothing here reaches filesystem snapshots, backups, or blocks an SSD retains through
        wear levelling; full-disk encryption remains the only defence against those.
        """
        require_identifier(identity_id, "identity_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        with self._connections.transaction(secure_delete=True) as connection:
            members = _identity_equivalence_class(connection, identity_id)
            if members is None:
                return None
            resolved_id = members[0]
            # Before the erasure strips the rows the index projection reads.
            queue_identity_projection(connection, resolved_id)
            exemplars = {
                validated_identity_modality(row["modality"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT modality, COUNT(*) AS count
                    FROM identity_exemplars
                    WHERE identity_id = ?
                    GROUP BY modality
                    """,
                    (resolved_id,),
                ).fetchall()
            }
            segments = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM speech_segments WHERE speaker_id = ?",
                    (resolved_id,),
                ).fetchone()["count"]
            )
            # face_observations.identity_id is NOT NULL, so the RESTRICT that guards it cannot
            # be satisfied by anonymising in place the way speech segments are.
            observations = connection.execute(
                "DELETE FROM face_observations WHERE identity_id = ?",
                (resolved_id,),
            ).rowcount
            # The naming assertions go with the person. `ON DELETE SET NULL` below would keep
            # them as unattributed records still carrying the erased name in their content and
            # their vectors, which is exactly what erasure promises to remove. This goes through
            # the ordinary delete, so whatever cited the assertion is reconciled and reprojected
            # exactly as it is when a caller deletes the assertion itself.
            erased_claims = tuple(
                row_text(row, "memory_id")
                for row in connection.execute(
                    """
                    SELECT memory_id FROM memory_semantics
                    WHERE identity_id = ? AND kind = ?
                    ORDER BY memory_id
                    """,
                    (resolved_id, MemoryKind.ENTITY.value),
                ).fetchall()
            )
            unreferenced: list[StoredAsset] = []
            for claim_id in erased_claims:
                _deleted, orphaned = delete_memory(connection, claim_id)
                unreferenced.extend(orphaned)
            # Cascades the aliases, exemplars and link evidence; NULLs the speech segments.
            connection.execute("DELETE FROM identities WHERE identity_id = ?", (resolved_id,))
            if operation is not None:
                insert_operation(connection, replace(operation, changed_ids=erased_claims))
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
        # After the commit, and best effort: a busy checkpoint leaves the zeroed pages in the
        # log rather than losing them, and the next checkpoint still applies them.
        with self._connections.connection() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return IdentityErasure(
            identity_id=resolved_id,
            alias_ids=members[1:],
            face_exemplars=exemplars.get("face", 0),
            voice_exemplars=exemplars.get("voice", 0),
            face_observations=observations,
            speech_segments=segments,
        ), tuple({asset.asset_id: asset for asset in unreferenced}.values())

    def identity_evidence_memory_ids(
        self,
        identity_id: str,
        memory_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Return which of `memory_ids` carry a face or voice occurrence of one identity.

        The control plane asks this before it lets a proposal name a person: a name may only be
        pinned on somebody the cited evidence actually contains. Unknown identities and unknown
        memory IDs simply do not match, so the caller reads an empty result as "not involved".
        """
        require_identifier(identity_id, "identity_id")
        candidates = tuple(dict.fromkeys(memory_ids))
        if not candidates:
            return ()
        placeholders = ", ".join("?" for _memory_id in candidates)
        with self._connections.connection() as connection:
            resolved_id = resolve_identity_id(connection, identity_id)
            if resolved_id is None:
                return ()
            rows = connection.execute(
                f"""
                SELECT DISTINCT ma.memory_id
                FROM memory_assets AS ma
                WHERE ma.memory_id IN ({placeholders}) AND (
                    EXISTS (
                        SELECT 1 FROM speech_segments AS s
                        WHERE s.asset_id = ma.asset_id AND s.speaker_id = ?
                    ) OR EXISTS (
                        SELECT 1 FROM face_observations AS f
                        WHERE f.asset_id = ma.asset_id AND f.identity_id = ?
                    )
                )
                ORDER BY ma.memory_id
                """,
                (*candidates, resolved_id, resolved_id),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)
