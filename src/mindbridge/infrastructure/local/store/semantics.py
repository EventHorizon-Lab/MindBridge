"""Formation and evidence: typed state committed with versions, evidence, and vectors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone

from mindbridge.infrastructure.local.store._candidates import (
    FUNCTIONAL_CLAIM_PARAMETERS,
    FUNCTIONAL_CLAIM_SQL,
)
from mindbridge.infrastructure.local.store._codec import (
    datetime_text,
    prepare_write_batch,
    row_text,
)
from mindbridge.infrastructure.local.store._connections import Connections
from mindbridge.infrastructure.local.store._identity import (
    reproject_named_identities,
    resolve_identity_id,
)
from mindbridge.infrastructure.local.store._lineage import (
    add_evidence_clause,
    add_memory_evidence,
    memory_evidence_linked,
    refresh_multi_source_projections,
    require_active_memories,
    require_unretired_memories,
    set_forgotten,
    validate_formation_links,
    version_retired,
)
from mindbridge.infrastructure.local.store._operations import active_operation_id, insert_operation
from mindbridge.infrastructure.local.store.errors import StaleOperationError
from mindbridge.infrastructure.local.store.records import (
    MemoryRecords,
    replace_memory_embeddings,
    write_embedding,
    write_memory,
)
from mindbridge.infrastructure.local.store.rows import (
    StoredEmbedding,
    StoredEvidenceClauseChange,
    StoredMemory,
    StoredOperation,
    canonical_object_json,
    require_aware,
    require_identifier,
)
from mindbridge.types import SpatialContext


class Semantics:
    """Typed formation commits and the evidence they rest on."""

    def __init__(
        self,
        *,
        connections: Connections,
        records: MemoryRecords,
    ) -> None:
        self._connections = connections
        self._records = records

    def apply_formation(  # noqa: C901 - one atomic formation transaction
        self,
        memories: Iterable[StoredMemory],
        embeddings: Iterable[StoredEmbedding],
        *,
        evidence: Sequence[tuple[str, str, float]],
        evidence_clauses: Sequence[tuple[str, tuple[str, ...], float]] = (),
        source_memory_ids: Sequence[str],
        narrowed: Sequence[tuple[str, str | None, str]] = (),
        recipe: str,
        completed_at: datetime,
        operation: StoredOperation | None = None,
        forget_ids: Sequence[str] = (),
        require_active: Sequence[str] = (),
        require_unretired: Sequence[str] = (),
        projection_identity_id: str | None = None,
        projection_memories: Iterable[StoredMemory] = (),
        projection_embeddings: Iterable[StoredEmbedding] = (),
    ) -> bool:
        """Commit derived records, evidence, and source completion in one transaction.

        `operation` logs the control-plane operation that produced these records in the same
        transaction. Consolidation passes no `source_memory_ids`, so no `formation_runs` marker
        is written and the derived record stays open to further independent evidence.
        `forget_ids` is consolidation forgetting: sources the derived record replaces in ordinary
        recall, retired in this same commit and cleared again by rolling the operation back.
        `narrowed` is `(memory_id, place_id, metadata_json)` for records already written whose
        inherited columns no longer hold for all their evidence. Only those two columns change,
        and neither reaches the search index, so nothing is queued for re-indexing.
        `require_active` names memories that must still exist and still be un-forgotten inside
        this transaction; if any moved since the caller validated it, nothing is written and
        `StaleOperationError` is raised. `require_unretired` names memories whose current version
        must additionally still stand, which is what a cited derived source needs and what
        `require_active` deliberately does not check.

        The lineage rule the kernel applies to a `STATE` or user-stated `TRAIT` supersedes the
        current version of every other record in the same lineage with overlapping validity,
        including records nobody showed the backend. Those `(memory_id, version)` pairs are
        recorded on the operation row as `superseded` so `rollback_operation` can restore exactly
        them.
        """
        supplied_memories, supplied_embeddings, supplied_by_memory = prepare_write_batch(
            memories,
            embeddings,
        )
        projected_memories, projected_embeddings, projected_by_memory = prepare_write_batch(
            projection_memories,
            projection_embeddings,
        )
        if projection_identity_id is not None:
            require_identifier(projection_identity_id, "projection_identity_id")
        elif projected_memories:
            raise ValueError("projection memories require projection_identity_id")
        sources = tuple(dict.fromkeys(source_memory_ids))
        require_identifier(recipe, "recipe")
        require_aware(completed_at, "completed_at")
        validate_formation_links(sources, forget_ids, evidence)
        for memory_id, members, confidence in evidence_clauses:
            require_identifier(memory_id, "memory_id")
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("evidence clause confidence must be between zero and one")
            if not members or len(set(members)) != len(members):
                raise ValueError("evidence clause members must be non-empty and unique")
        with self._connections.transaction() as connection:
            # Same in-transaction idempotency check `apply_control_operation` makes: a duplicate
            # that arrives after the caller's pre-check must be refused, not surface as a
            # unique-index violation.
            if (
                operation is not None
                and active_operation_id(connection, operation.operation_key) is not None
            ):
                return False
            require_active_memories(connection, require_active)
            require_unretired_memories(connection, require_unretired)
            if any(
                connection.execute(
                    """
                    SELECT 1 FROM formation_runs
                    WHERE source_memory_id = ? AND recipe = ?
                    """,
                    (source_memory_id, recipe),
                ).fetchone()
                is not None
                for source_memory_id in sources
            ):
                return False
            # What this operation has to roll back is measured before anything is written: a
            # restated record the store already holds attaches its own cited evidence on the way
            # through `write_memory`, so the link's absence beforehand, not who inserted it,
            # says whether this operation created it.
            already_linked = {
                (memory_id, source_memory_id)
                for memory_id, source_memory_id, _confidence in evidence
                if memory_evidence_linked(connection, memory_id, source_memory_id)
            }
            activated = tuple(
                memory.memory_id
                for memory in supplied_memories
                if memory.context is not None and version_retired(connection, memory.memory_id)
            )
            transaction_memory_ids: set[str] = set()
            superseded: list[tuple[str, int]] = []
            clause_memory_ids = {memory_id for memory_id, _members, _confidence in evidence_clauses}
            for memory in supplied_memories:
                has_explicit_clause = memory.memory_id in clause_memory_ids
                write_memory(
                    connection,
                    memory,
                    supplied_embedding_ids=supplied_by_memory[memory.memory_id],
                    transaction_memory_ids=transaction_memory_ids,
                    superseded=superseded,
                    write_context_evidence=not has_explicit_clause,
                    context_recorded_at=completed_at if has_explicit_clause else None,
                )
                transaction_memory_ids.add(memory.memory_id)
            connection.executemany(
                """
                UPDATE memory_records
                SET place_id = ?,
                    metadata_json = ?,
                    updated_at = MAX(updated_at, ?)
                WHERE memory_id = ?
                """,
                (
                    (
                        place_id,
                        canonical_object_json(metadata_json),
                        datetime_text(completed_at),
                        memory_id,
                    )
                    for memory_id, place_id, metadata_json in narrowed
                ),
            )
            for embedding in supplied_embeddings:
                write_embedding(connection, embedding)
            linked: list[tuple[str, str]] = []
            clause_changes: list[StoredEvidenceClauseChange] = []
            for memory_id, members, confidence in evidence_clauses:
                change = add_evidence_clause(
                    connection,
                    memory_id,
                    members,
                    confidence=confidence,
                    recorded_at=completed_at,
                )
                if change is not None:
                    clause_changes.append(change)
            for memory_id, source_memory_id, confidence in evidence:
                add_memory_evidence(
                    connection,
                    memory_id,
                    source_memory_id,
                    confidence=confidence,
                    recorded_at=completed_at,
                )
                pair = (memory_id, source_memory_id)
                if pair not in already_linked:
                    already_linked.add(pair)
                    linked.append(pair)
            refresh_multi_source_projections(
                connection,
                supplied_memories,
                changed_at=completed_at,
            )
            # Forgetting comes first: the projection has to be computed against the records that
            # remain, so consolidation forgetting a naming assertion stops projecting its name.
            forgotten: tuple[str, ...] = ()
            if operation is not None:
                forgotten = set_forgotten(connection, forget_ids, forgotten_at=completed_at)
            reproject_named_identities(
                connection,
                tuple(memory.memory_id for memory in supplied_memories)
                + tuple(memory_id for memory_id, _source, _confidence in evidence)
                + tuple(forget_ids),
            )
            if projection_identity_id is not None:
                if resolve_identity_id(connection, projection_identity_id) is None:
                    raise StaleOperationError(
                        f"{projection_identity_id} is no longer an active identity"
                    )
                replace_memory_embeddings(
                    connection,
                    projected_memories,
                    projected_embeddings,
                    projected_by_memory,
                )
            connection.executemany(
                """
                INSERT INTO formation_runs (source_memory_id, recipe, completed_at)
                VALUES (?, ?, ?)
                """,
                (
                    (source_memory_id, recipe, datetime_text(completed_at))
                    for source_memory_id in sources
                ),
            )
            if operation is not None:
                insert_operation(
                    connection,
                    replace(
                        operation,
                        activated_ids=activated,
                        forgotten_ids=forgotten,
                        linked=tuple(linked),
                        clause_changes=tuple(clause_changes),
                        superseded=tuple(dict.fromkeys(superseded)),
                    ),
                )
        return True

    def formation_completed(self, source_memory_id: str, recipe: str) -> bool:
        """Return whether one source was successfully formed with this recipe."""
        require_identifier(source_memory_id, "source_memory_id")
        require_identifier(recipe, "recipe")
        with self._connections.connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM formation_runs
                WHERE source_memory_id = ? AND recipe = ?
                """,
                (source_memory_id, recipe),
            ).fetchone()
        return row is not None

    def add_memory_evidence(
        self,
        memory_id: str,
        source_memory_id: str,
        *,
        confidence: float,
        recorded_at: datetime,
    ) -> bool:
        """Attach independent evidence and version the derived confidence projection."""
        require_identifier(memory_id, "memory_id")
        require_identifier(source_memory_id, "source_memory_id")
        require_aware(recorded_at, "recorded_at")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        with self._connections.transaction() as connection:
            added = add_evidence_clause(
                connection,
                memory_id,
                (source_memory_id,),
                confidence=float(confidence),
                recorded_at=recorded_at,
            )
            # Independent evidence is what makes an inferred naming assertion visible, so it is
            # also what can move the projection.
            reproject_named_identities(connection, (memory_id,))
            return added is not None

    def read_functional_lineage_representatives(
        self,
        lineage_ids: Sequence[str],
        *,
        per_lineage_limit: int,
        valid_at: datetime | None = None,
        known_at: datetime | None = None,
        near: SpatialContext | None = None,
        radius_m: float | None = None,
        place_id: str | None = None,
        identity_id: str | None = None,
    ) -> tuple[tuple[StoredMemory, ...], frozenset[str]]:
        """Read bounded current representatives for functional claim lineages.

        This is a completion read for context compilation, not a second retrieval route.  It starts
        from the lineage index, then uses ``read_memories`` for authoritative bitemporal, place,
        identity, and metric-scope filtering before it retains the rows for each distinct value.
        The compiler chooses one deliverable representative only after confidence, support, and
        consent checks, so an ineligible copy cannot hide an eligible copy of the same assertion.
        The value bound is applied after scope, so repeated or historical copies of the same
        assertion do not by themselves make a current claim incomplete.  The read may scan all
        current visible candidate rows of a requested lineage before identity and metric scope
        can prove which values survive; it never scans an unrelated lineage or uses a relevance
        score to choose a representative.
        """
        if not lineage_ids:
            return (), frozenset()
        for lineage_id in lineage_ids:
            require_identifier(lineage_id, "lineage_id")
        if (
            isinstance(per_lineage_limit, bool)
            or not isinstance(per_lineage_limit, int)
            or per_lineage_limit <= 0
        ):
            raise ValueError("per_lineage_limit must be a positive integer")
        query_valid_at = valid_at or datetime.now(timezone.utc)
        query_known_at = known_at or datetime.now(timezone.utc)
        valid_text = datetime_text(query_valid_at)
        known_text = datetime_text(query_known_at)
        selected_ids: list[str] = []
        with self._connections.read_transaction() as connection:
            for lineage_id in dict.fromkeys(lineage_ids):
                rows = connection.execute(
                    f"""
                    SELECT s.memory_id AS memory_id
                    FROM memory_semantics AS s
                    JOIN memory_versions AS v ON v.memory_id = s.memory_id
                    JOIN memory_records AS r ON r.memory_id = s.memory_id
                    WHERE s.lineage_id = ?
                      AND {FUNCTIONAL_CLAIM_SQL}
                      AND r.forgotten_at IS NULL
                      AND v.visible = 1
                      AND v.recorded_at <= ?
                      AND (v.retired_at IS NULL OR v.retired_at > ?)
                      AND (v.valid_from IS NULL OR v.valid_from <= ?)
                      AND (v.valid_until IS NULL OR v.valid_until > ?)
                    ORDER BY s.memory_id
                    """,
                    (
                        lineage_id,
                        *FUNCTIONAL_CLAIM_PARAMETERS,
                        known_text,
                        known_text,
                        valid_text,
                        valid_text,
                    ),
                ).fetchall()
                selected_ids.extend(row_text(row, "memory_id") for row in rows)
        scoped = self._records.read_memories(
            selected_ids,
            valid_at=valid_at,
            known_at=known_at,
            near=near,
            radius_m=radius_m,
            place_id=place_id,
            identity_id=identity_id,
            active_only=True,
        )
        representatives: list[StoredMemory] = []
        values: dict[str, set[str]] = {}
        incomplete: set[str] = set()
        for memory in scoped:
            context = memory.context
            if context is None or context.lineage_id is None or context.value is None:
                continue
            lineage_values = values.setdefault(context.lineage_id, set())
            if context.value in lineage_values:
                representatives.append(memory)
                continue
            if len(lineage_values) >= per_lineage_limit:
                incomplete.add(context.lineage_id)
                continue
            lineage_values.add(context.value)
            representatives.append(memory)
        return tuple(representatives), frozenset(incomplete)
