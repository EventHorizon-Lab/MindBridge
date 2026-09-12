"""Memory records: writing, reading, listing, forgetting, and physical deletion."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

from mindbridge.infrastructure.local.store._codec import (
    SQLITE_PARAMETER_BATCH,
    asset_from_row,
    datetime_text,
    memory_from_row,
    optional_datetime_text,
    pack_vector,
    prepare_write_batch,
    require_storable_vector,
    row_text,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._identity import (
    displaced_naming_versions,
    naming_assertion_identities,
    reproject_identities,
    reproject_named_identities,
)
from mindbridge.infrastructure.local.store._lineage import (
    active_evidence_dependent_closure,
    deletion_cascade,
    grounded_affected_memory_ids,
    rebuild_reconciled_lineage,
    refresh_evidence_projection,
    restamp_dependent_evidence,
    restore_memory_versions,
    retire_evidence_clauses_for_source,
    set_forgotten,
    version_retired,
    write_memory_context,
)
from mindbridge.infrastructure.local.store._outbox import queue_memory_embeddings
from mindbridge.infrastructure.local.store._selection import (
    identity_scope,
    memory_in_scope,
    read_memory_contexts,
    select_memory_contexts,
)
from mindbridge.infrastructure.local.store.media import read_unreferenced_assets, write_asset
from mindbridge.infrastructure.local.store.rows import (
    StoredAsset,
    StoredEmbedding,
    StoredMemory,
    require_aware,
    require_identifier,
    require_optional_identifier,
)
from mindbridge.types import EvidenceBasis, MemoryKind, SpatialContext


def write_embedding(
    connection: sqlite3.Connection,
    embedding: StoredEmbedding,
) -> None:
    # The only place a caller-supplied vector reaches the authoritative table, and so the only
    # place its content is checked. See StoredEmbedding for why hydration does not repeat this.
    require_storable_vector(embedding.values, normalized=embedding.normalized)
    connection.execute(
        """
        INSERT INTO embeddings (
            embedding_id, memory_id, object_part, model_id, space_id, task,
            dimension, normalized, vector, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (embedding_id) DO UPDATE SET
            memory_id = excluded.memory_id,
            object_part = excluded.object_part,
            model_id = excluded.model_id,
            space_id = excluded.space_id,
            task = excluded.task,
            dimension = excluded.dimension,
            normalized = excluded.normalized,
            vector = excluded.vector,
            created_at = excluded.created_at
        """,
        (
            embedding.embedding_id,
            embedding.memory_id,
            embedding.object_part,
            embedding.model_id,
            embedding.space_id,
            embedding.task,
            len(embedding.values),
            int(embedding.normalized),
            pack_vector(embedding.values),
            datetime_text(embedding.created_at),
        ),
    )
    connection.execute(
        "DELETE FROM embedding_text_selectors WHERE embedding_id = ?",
        (embedding.embedding_id,),
    )
    if not embedding.text_selectors:
        return
    row = connection.execute(
        "SELECT content FROM memory_records WHERE memory_id = ?",
        (embedding.memory_id,),
    ).fetchone()
    if row is None:
        raise ValueError("text selectors require an existing parent memory")
    content = row_text(row, "content")
    parent_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    for selector_position, selector in enumerate(embedding.text_selectors):
        if selector.parent_content_sha256 != parent_digest:
            raise ValueError("text selector parent digest does not match stored content")
        source_pieces = tuple(
            content[piece.start_codepoint : piece.end_codepoint] for piece in selector.pieces
        )
        for piece, source_text in zip(selector.pieces, source_pieces, strict=True):
            if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != piece.piece_sha256:
                raise ValueError("text selector piece does not match stored content")
        embedding_input = (
            source_pieces[0]
            if len(source_pieces) == 1
            else f"{source_pieces[0]}\n\n{source_pieces[1]}"
        )
        if (
            hashlib.sha256(embedding_input.encode("utf-8")).hexdigest()
            != selector.embedding_input_sha256
        ):
            raise ValueError("text selector does not reconstruct its embedding input")
        connection.execute(
            """
            INSERT INTO embedding_text_selectors (
                embedding_id, selector_position, parent_content_sha256,
                embedding_input_sha256, recipe_version
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                embedding.embedding_id,
                selector_position,
                selector.parent_content_sha256,
                selector.embedding_input_sha256,
                selector.recipe_version,
            ),
        )
        for piece_position, piece in enumerate(selector.pieces):
            connection.execute(
                """
                INSERT INTO embedding_text_span_pieces (
                    embedding_id, selector_position, piece_position, role,
                    start_codepoint, end_codepoint, piece_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    embedding.embedding_id,
                    selector_position,
                    piece_position,
                    piece.role,
                    piece.start_codepoint,
                    piece.end_codepoint,
                    piece.piece_sha256,
                ),
            )


def replace_memory_embeddings(
    connection: sqlite3.Connection,
    memories: Sequence[StoredMemory],
    embeddings: Sequence[StoredEmbedding],
    embedding_ids: dict[str, set[str]],
) -> None:
    for memory in memories:
        if (
            connection.execute(
                "SELECT 1 FROM memory_records WHERE memory_id = ?",
                (memory.memory_id,),
            ).fetchone()
            is None
        ):
            raise ValueError("replacement embeddings require an existing memory")
        connection.execute(
            "DELETE FROM embeddings WHERE memory_id = ?",
            (memory.memory_id,),
        )
        write_memory(
            connection,
            memory,
            supplied_embedding_ids=embedding_ids[memory.memory_id],
        )
    for embedding in embeddings:
        write_embedding(connection, embedding)


def write_memory(
    connection: sqlite3.Connection,
    memory: StoredMemory,
    *,
    supplied_embedding_ids: set[str],
    transaction_memory_ids: set[str] | None = None,
    superseded: list[tuple[str, int]] | None = None,
    write_context_evidence: bool = True,
    context_recorded_at: datetime | None = None,
) -> bool:
    existing = connection.execute(
        """
        SELECT content, modality, memory_type, metadata_json, occurred_at, occurred_end,
               place_id
        FROM memory_records
        WHERE memory_id = ?
        """,
        (memory.memory_id,),
    ).fetchone()
    existing_asset_ids = tuple(
        row_text(row, "asset_id")
        for row in connection.execute(
            """
            SELECT asset_id
            FROM memory_assets
            WHERE memory_id = ?
            ORDER BY position
            """,
            (memory.memory_id,),
        ).fetchall()
    )
    supplied_asset_ids = tuple(asset.asset_id for asset in memory.assets)
    reactivating = memory.context is not None and version_retired(connection, memory.memory_id)
    index_content_changed = existing is not None and (
        row_text(existing, "content") != memory.content
        or row_text(existing, "modality") != memory.modality
        or row_text(existing, "memory_type") != memory.memory_type
        or row_text(existing, "metadata_json") != memory.metadata_json
        or existing["occurred_at"] != optional_datetime_text(memory.occurred_at)
        or existing["occurred_end"] != optional_datetime_text(memory.occurred_end)
        or existing_asset_ids != supplied_asset_ids
        # The index carries `place_id` as a filter field, so relabelling a room is a change
        # to the indexed document even though none of its text moved.
        or existing["place_id"] != memory.place_id
    )
    connection.execute(
        """
        INSERT INTO memory_records (
            memory_id, content, modality, memory_type, metadata_json,
            occurred_at, occurred_end, last_accessed_at, access_count, created_at, updated_at,
            place_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (memory_id) DO UPDATE SET
            content = excluded.content,
            modality = excluded.modality,
            memory_type = excluded.memory_type,
            metadata_json = excluded.metadata_json,
            occurred_at = excluded.occurred_at,
            occurred_end = excluded.occurred_end,
            updated_at = MAX(memory_records.updated_at, excluded.updated_at),
            place_id = excluded.place_id
        """,
        (
            memory.memory_id,
            memory.content,
            memory.modality,
            memory.memory_type,
            memory.metadata_json,
            optional_datetime_text(memory.occurred_at),
            optional_datetime_text(memory.occurred_end),
            optional_datetime_text(memory.last_accessed_at),
            memory.access_count,
            datetime_text(memory.created_at),
            datetime_text(memory.updated_at),
            memory.place_id,
        ),
    )
    for asset in memory.assets:
        write_asset(connection, asset)
    if existing_asset_ids != supplied_asset_ids:
        connection.execute(
            "DELETE FROM memory_assets WHERE memory_id = ?",
            (memory.memory_id,),
        )
        connection.executemany(
            """
            INSERT INTO memory_assets (memory_id, position, asset_id)
            VALUES (?, ?, ?)
            """,
            (
                (memory.memory_id, position, asset_id)
                for position, asset_id in enumerate(supplied_asset_ids)
            ),
        )
    if index_content_changed or reactivating:
        queue_memory_embeddings(
            connection,
            memory.memory_id,
            exclude=supplied_embedding_ids,
        )
    if memory.context is not None:
        write_memory_context(
            connection,
            memory.memory_id,
            memory.context,
            transaction_memory_ids=transaction_memory_ids,
            superseded=superseded,
            write_evidence=write_context_evidence,
            recorded_at=context_recorded_at,
        )
    return existing is None


def _read_memory_assets(
    connection: sqlite3.Connection,
    memory_ids: Sequence[str],
) -> dict[str, tuple[StoredAsset, ...]]:
    unique_ids = tuple(dict.fromkeys(memory_ids))
    if not unique_ids:
        return {}
    collected: dict[str, list[StoredAsset]] = {memory_id: [] for memory_id in unique_ids}
    for offset in range(0, len(unique_ids), SQLITE_PARAMETER_BATCH):
        batch = unique_ids[offset : offset + SQLITE_PARAMETER_BATCH]
        placeholders = ", ".join("?" for _memory_id in batch)
        rows = connection.execute(
            f"""
            SELECT ma.memory_id, ma.position,
                   a.asset_id, a.modality, a.mime_type, a.size_bytes, a.sha256,
                   a.relative_path, a.name, a.transcript, a.created_at
            FROM memory_assets AS ma
            JOIN media_assets AS a ON a.asset_id = ma.asset_id
            WHERE ma.memory_id IN ({placeholders})
            ORDER BY ma.memory_id, ma.position
            """,
            tuple(batch),
        ).fetchall()
        for row in rows:
            collected[row_text(row, "memory_id")].append(asset_from_row(row))
    return {memory_id: tuple(assets) for memory_id, assets in collected.items()}


def delete_memory(  # noqa: C901 - one atomic dependency and lineage teardown
    connection: sqlite3.Connection,
    memory_id: str,
) -> tuple[bool, tuple[StoredAsset, ...]]:
    """Delete one memory inside the caller's transaction; see `delete_memory_with_assets`."""
    if (
        connection.execute(
            "SELECT 1 FROM memory_records WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        is None
    ):
        return False, ()
    now = datetime.now(timezone.utc)
    changed_at = now
    affected = active_evidence_dependent_closure(connection, memory_id)
    clause_changes = retire_evidence_clauses_for_source(
        connection,
        memory_id,
        retired_at=changed_at,
    )
    if clause_changes:
        changed_at = max(changed_at, *clause_changes.values())
    grounded = grounded_affected_memory_ids(
        connection,
        affected,
        excluding=(memory_id,),
    )
    unsupported = tuple(dependent_id for dependent_id in affected if dependent_id not in grounded)
    for source_memory_id in unsupported:
        downstream_changes = retire_evidence_clauses_for_source(
            connection,
            source_memory_id,
            retired_at=changed_at,
        )
        for dependent_id, dependent_changed_at in downstream_changes.items():
            clause_changes[dependent_id] = max(
                clause_changes.get(dependent_id, dependent_changed_at),
                dependent_changed_at,
            )
        if downstream_changes:
            changed_at = max(changed_at, *downstream_changes.values())
    removed_ids = (memory_id, *unsupported)
    surviving_dependents = tuple(
        dependent_id for dependent_id in clause_changes if dependent_id not in removed_ids
    )
    for dependent_id in surviving_dependents:
        refresh_evidence_projection(
            connection,
            dependent_id,
            clause_changes[dependent_id],
        )

    reconciled: set[tuple[str, str]] = set()
    for source_memory_id in removed_ids:
        semantic = connection.execute(
            """
            SELECT lineage_id, kind, basis
            FROM memory_semantics WHERE memory_id = ?
            """,
            (source_memory_id,),
        ).fetchone()
        if semantic is not None and (
            row_text(semantic, "kind") == MemoryKind.STATE.value
            or (
                row_text(semantic, "kind") == MemoryKind.TRAIT.value
                and row_text(semantic, "basis") == EvidenceBasis.USER_STATEMENT.value
            )
        ):
            reconciled.add((row_text(semantic, "lineage_id"), row_text(semantic, "kind")))
    # Read before deletion: a derived naming assertion may be several dependency hops from
    # the requested source. Every unsupported node must stop feeding the identity projection.
    named_identities = naming_assertion_identities(
        connection,
        (*removed_ids, *surviving_dependents),
    )
    displaced = [
        candidate
        for removed_id in removed_ids
        for candidate in displaced_naming_versions(connection, removed_id)
    ]
    linked_ids = [
        row_text(row, "asset_id")
        for removed_id in removed_ids
        for row in connection.execute(
            """
            SELECT asset_id FROM memory_assets
            WHERE memory_id = ? ORDER BY position
            """,
            (removed_id,),
        ).fetchall()
    ]
    restamp_dependent_evidence(connection, surviving_dependents, changed_at)
    for removed_id in removed_ids:
        if removed_id != memory_id:
            connection.execute(
                "DELETE FROM memory_records WHERE memory_id = ?",
                (removed_id,),
            )
    cursor = connection.execute(
        "DELETE FROM memory_records WHERE memory_id = ?",
        (memory_id,),
    )
    for lineage_id, kind in sorted(reconciled):
        rebuild_reconciled_lineage(
            connection,
            lineage_id,
            kind,
            changed_at=changed_at,
        )
    restore_memory_versions(
        connection,
        tuple(
            candidate
            for candidate in dict.fromkeys(displaced)
            if version_retired(connection, candidate)
        ),
        recorded_at=changed_at,
    )
    reproject_identities(connection, named_identities)
    unreferenced = read_unreferenced_assets(
        connection,
        tuple(dict.fromkeys(linked_ids)),
    )
    return cursor.rowcount > 0, unreferenced


class MemoryRecords:
    """Authoritative memory rows and everything keyed on them."""

    def __init__(
        self,
        *,
        connections: Connections,
    ) -> None:
        self._connections = connections

    def write_memory(
        self,
        memory: StoredMemory,
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Create or update a memory and its supplied embeddings atomically.

        Returns true when this call created the memory. Updating memory content queues every
        existing vector again because indexed text is derived from the authoritative row.
        """
        return self.write_memories((memory,), embeddings)[0]

    def write_memories(
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding] = (),
        *,
        formation_pending_at: datetime | None = None,
    ) -> tuple[bool, ...]:
        """Create or update a batch with one commit and one durability sync.

        `formation_pending_at` enqueues each written memory for the follow-up work the caller has
        not done yet, in the same transaction that makes the record durable. The strong `add()`
        path uses it so a crash between this commit and formation leaves a queue row the next
        `settle()` finds, instead of a searchable record nothing will ever form. The row carries
        no state of its own: a queued record that already has vectors owes formation only, which
        is what `read_embedding` tells the settler.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        if not supplied_memories:
            return ()
        if formation_pending_at is not None:
            require_aware(formation_pending_at, "formation_pending_at")
        with self._connections.transaction() as connection:
            transaction_memory_ids: set[str] = set()
            created_flags = []
            for memory in supplied_memories:
                created_flags.append(
                    write_memory(
                        connection,
                        memory,
                        supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                        transaction_memory_ids=transaction_memory_ids,
                    )
                )
                transaction_memory_ids.add(memory.memory_id)
                if formation_pending_at is not None:
                    connection.execute(
                        """
                        INSERT INTO capture_queue (memory_id, enqueued_at) VALUES (?, ?)
                        ON CONFLICT (memory_id) DO NOTHING
                        """,
                        (memory.memory_id, datetime_text(formation_pending_at)),
                    )
            for embedding in supplied_embeddings:
                write_embedding(connection, embedding)
            reproject_named_identities(
                connection,
                tuple(memory.memory_id for memory in supplied_memories),
            )
        return tuple(created_flags)

    def replace_memory_embeddings(
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding],
    ) -> None:
        """Atomically replace every vector belonging to supplied existing memories."""
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        if not supplied_memories:
            return
        with self._connections.transaction() as connection:
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )

    def read_memory(self, memory_id: str) -> StoredMemory | None:
        """Return one memory, or none when it does not exist."""
        require_identifier(memory_id, "memory_id")
        with self._connections.read_transaction() as connection:
            row = connection.execute(
                """
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, occurred_end, last_accessed_at, access_count,
                       created_at, updated_at, place_id, forgotten_at
                FROM memory_records
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            assets = () if row is None else _read_memory_assets(connection, (memory_id,))[memory_id]
            contexts, _semantic_ids = read_memory_contexts(connection, (memory_id,))
        return (
            None
            if row is None
            else memory_from_row(row, assets=assets, context=contexts.get(memory_id))
        )

    def read_memories(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        active_only: bool = False,
    ) -> tuple[StoredMemory, ...]:
        """Hydrate existing memories with one query and preserve input ranking.

        `place_id` scopes the slate to one symbolic place. It is a hard filter, unlike `near`/
        `radius_m` which the caller applies to metric pose, and it is applied in SQL so a scoped
        hydration reads only the rows it returns.

        `identity_id` scopes it to one person, accepting a merged alias, and is a hard filter for
        the same reason: it is the authoritative answer the search index only approximates.
        """
        if not memory_ids:
            return ()
        for memory_id in memory_ids:
            require_identifier(memory_id, "memory_id")
        require_optional_identifier(place_id, "place_id")
        require_optional_identifier(identity_id, "identity_id")
        place_clause = "" if place_id is None else "AND place_id = ?"
        place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
        rows: list[sqlite3.Row] = []
        with self._connections.read_transaction() as connection:
            identity_clause, identity_parameters = identity_scope(connection, identity_id)
            if identity_clause is None:
                return ()
            for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT memory_id, content, modality, memory_type, metadata_json,
                               occurred_at, occurred_end, last_accessed_at, access_count,
                               created_at, updated_at, place_id, forgotten_at
                        FROM memory_records
                        WHERE memory_id IN ({placeholders})
                        {place_clause}
                        {identity_clause}
                        """,
                        (*batch, *place_parameters, *identity_parameters),
                    ).fetchall()
                )
            assets_by_memory = _read_memory_assets(connection, tuple(memory_ids))
            contexts, semantic_ids = read_memory_contexts(
                connection,
                tuple(memory_ids),
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                active_only=active_only,
            )
        by_id = {
            row_text(row, "memory_id"): memory_from_row(
                row,
                assets=assets_by_memory.get(row_text(row, "memory_id"), ()),
                context=contexts.get(row_text(row, "memory_id")),
            )
            for row in rows
            if memory_in_scope(
                row,
                active_only=active_only,
                known_at=known_at,
                near=near,
                semantic_ids=semantic_ids,
                scoped_ids=contexts.keys(),
            )
        }
        return tuple(by_id[memory_id] for memory_id in memory_ids if memory_id in by_id)

    def count_memories(
        self,
        memory_ids: Sequence[str],
        *,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
        active_only: bool = False,
    ) -> int:
        """Count what `read_memories` would return for the same arguments.

        Applies every scope predicate through the same selection pass, and reads neither
        content, media assets, nor typed context rows: a caller that only needs the number of
        surviving memories must not pay for the records.
        """
        if not memory_ids:
            return 0
        for memory_id in memory_ids:
            require_identifier(memory_id, "memory_id")
        require_optional_identifier(place_id, "place_id")
        require_optional_identifier(identity_id, "identity_id")
        place_clause = "" if place_id is None else "AND place_id = ?"
        place_parameters: tuple[object, ...] = () if place_id is None else (place_id,)
        by_id: dict[str, sqlite3.Row] = {}
        with self._connections.read_transaction() as connection:
            identity_clause, identity_parameters = identity_scope(connection, identity_id)
            if identity_clause is None:
                return 0
            for offset in range(0, len(memory_ids), SQLITE_PARAMETER_BATCH):
                batch = memory_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                for row in connection.execute(
                    f"""
                    SELECT memory_id, created_at, forgotten_at
                    FROM memory_records
                    WHERE memory_id IN ({placeholders})
                    {place_clause}
                    {identity_clause}
                    """,
                    (*batch, *place_parameters, *identity_parameters),
                ).fetchall():
                    by_id[row_text(row, "memory_id")] = row
            scoped, semantic_ids = select_memory_contexts(
                connection,
                tuple(memory_ids),
                valid_at=valid_at,
                known_at=known_at,
                near=near,
                radius_m=radius_m,
                active_only=active_only,
            )
        surviving = {
            memory_id
            for memory_id, row in by_id.items()
            if memory_in_scope(
                row,
                active_only=active_only,
                known_at=known_at,
                near=near,
                semantic_ids=semantic_ids,
                scoped_ids=scoped.keys(),
            )
        }
        # Counted per requested ID, not per distinct row, because `read_memories` repeats a
        # memory that its caller asked for twice.
        return sum(1 for memory_id in memory_ids if memory_id in surviving)

    def list_memories(
        self,
        *,
        limit: int = 100,
        after: tuple[datetime, str] | None = None,
    ) -> tuple[StoredMemory, ...]:
        """List newest first with a stable `(created_at, memory_id)` keyset cursor."""
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        parameters: tuple[object, ...]
        where = ""
        if after is None:
            parameters = (limit,)
        else:
            created_at, memory_id = after
            require_aware(created_at, "after created_at")
            require_identifier(memory_id, "after memory_id")
            created_text = datetime_text(created_at)
            where = """
                WHERE created_at < ? OR (created_at = ? AND memory_id < ?)
            """
            parameters = (created_text, created_text, memory_id, limit)
        with self._connections.read_transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT memory_id, content, modality, memory_type, metadata_json,
                       occurred_at, occurred_end, last_accessed_at, access_count,
                       created_at, updated_at, place_id, forgotten_at
                FROM memory_records
                {where}
                ORDER BY created_at DESC, memory_id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            memory_ids = tuple(row_text(row, "memory_id") for row in rows)
            assets_by_memory = _read_memory_assets(connection, memory_ids)
            contexts, _semantic_ids = read_memory_contexts(connection, memory_ids)
        return tuple(
            memory_from_row(
                row,
                assets=assets_by_memory.get(row_text(row, "memory_id"), ()),
                context=contexts.get(row_text(row, "memory_id")),
            )
            for row in rows
        )

    def reinforce_memories(self, memory_ids: Sequence[str], *, accessed_at: datetime) -> int:
        """Record one bounded retrieval reinforcement for each existing memory."""
        unique_ids = tuple(dict.fromkeys(memory_ids))
        if not unique_ids:
            return 0
        for memory_id in unique_ids:
            require_identifier(memory_id, "memory_id")
        require_aware(accessed_at, "accessed_at")
        accessed_text = datetime_text(accessed_at)
        changed = 0
        with self._connections.transaction() as connection:
            for offset in range(0, len(unique_ids), SQLITE_PARAMETER_BATCH):
                batch = unique_ids[offset : offset + SQLITE_PARAMETER_BATCH]
                placeholders = ", ".join("?" for _memory_id in batch)
                cursor = connection.execute(
                    f"""
                    UPDATE memory_records
                    SET access_count = MIN(access_count + 1, 20),
                        last_accessed_at = CASE
                            WHEN last_accessed_at IS NULL OR last_accessed_at < ? THEN ?
                            ELSE last_accessed_at
                        END
                    WHERE memory_id IN ({placeholders})
                    """,
                    (accessed_text, accessed_text, *batch),
                )
                changed += cursor.rowcount
        return changed

    def set_forgotten(
        self,
        memory_ids: Sequence[str],
        *,
        forgotten_at: datetime | None,
    ) -> tuple[str, ...]:
        """Set or clear cognitive forgetting and return the ids whose state changed."""
        ids = tuple(dict.fromkeys(memory_ids))
        for memory_id in ids:
            require_identifier(memory_id, "memory_id")
        if forgotten_at is not None:
            require_aware(forgotten_at, "forgotten_at")
        if not ids:
            return ()
        with self._connections.transaction() as connection:
            changed = set_forgotten(connection, ids, forgotten_at=forgotten_at)
            reproject_named_identities(connection, changed)
            return changed

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory; cascading embedding triggers enqueue index deletions."""
        deleted, _assets = self.delete_memory_with_assets(memory_id)
        return deleted

    def deletion_cascade(self, memory_ids: Sequence[str]) -> tuple[str, ...]:
        """Predict the exact bounded record cascade for deleting all supplied roots."""
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            require_identifier(memory_id, "memory_id")
        if not selected:
            return ()
        with self._connections.connection() as connection:
            return deletion_cascade(connection, selected)

    def assets_orphaned_by_deletion(self, memory_ids: Sequence[str]) -> tuple[str, ...]:
        """Return assets whose every current memory reference is in the supplied set."""
        selected = tuple(dict.fromkeys(memory_ids))
        for memory_id in selected:
            require_identifier(memory_id, "memory_id")
        if not selected:
            return ()
        with self._connections.read_transaction() as connection:
            # Both halves of the orphan predicate must see the complete deletion set. Binding it
            # twice in one statement exceeds SQLite's conservative parameter budget as soon as a
            # retention page crosses one batch, while checking each batch independently would
            # misclassify an asset shared by memories in different batches. A connection-local
            # table keeps that set exact without changing authoritative state.
            connection.execute(
                "CREATE TEMP TABLE retention_deletions (memory_id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.executemany(
                "INSERT INTO retention_deletions (memory_id) VALUES (?)",
                ((memory_id,) for memory_id in selected),
            )
            rows = connection.execute(
                """
                SELECT DISTINCT ma.asset_id
                FROM memory_assets AS ma
                JOIN temp.retention_deletions AS selected
                  ON selected.memory_id = ma.memory_id
                JOIN media_assets AS a ON a.asset_id = ma.asset_id
                WHERE NOT EXISTS (
                    SELECT 1 FROM memory_assets AS other
                    LEFT JOIN temp.retention_deletions AS deleting
                      ON deleting.memory_id = other.memory_id
                    WHERE other.asset_id = ma.asset_id
                      AND deleting.memory_id IS NULL
                )
                ORDER BY a.created_at, ma.asset_id
                """
            ).fetchall()
        return tuple(row_text(row, "asset_id") for row in rows)

    def delete_memory_with_assets(
        self,
        memory_id: str,
        *,
        memories: Iterable[StoredMemory] = (),
        embeddings: Iterable[StoredEmbedding] = (),
    ) -> tuple[bool, tuple[StoredAsset, ...]]:
        """Delete one memory and return its assets that no remaining memory references.

        Pass `memories` and `embeddings` to atomically replace indexed documents in the same
        commit, which deleting a naming assertion needs: the projection it fed is recomputed
        here, so the indexed text quoting the name has to be rebuilt with it.
        """
        require_identifier(memory_id, "memory_id")
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        with self._connections.transaction() as connection:
            deleted, unreferenced = delete_memory(connection, memory_id)
            replace_memory_embeddings(
                connection,
                supplied_memories,
                supplied_embeddings,
                supplied_by_memory,
            )
            return deleted, unreferenced

    def forgotten_memory_ids(
        self,
        *,
        forgotten_before: datetime,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """Return memories cognitively forgotten before a moment, oldest forgetting first.

        The read half of retention over `forget()`: cognitive forgetting is reversible by
        design, so something has to decide when it becomes final, and only a declared policy or
        a person may.
        """
        require_aware(forgotten_before, "forgotten_before")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connections.connection() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM memory_records
                WHERE forgotten_at IS NOT NULL AND forgotten_at < ?
                ORDER BY forgotten_at, memory_id
                LIMIT ?
                """,
                (datetime_text(forgotten_before), limit),
            ).fetchall()
        return tuple(row_text(row, "memory_id") for row in rows)
