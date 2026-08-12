"""Atomic PostgreSQL commit for verified semantic Claim decisions."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from psycopg.errors import ForeignKeyViolation

from mindbridge.application.semantic_claims import (
    ClaimConsolidationCommit,
    ClaimConsolidationWrite,
    SemanticClaimWrite,
)
from mindbridge.core import (
    ClaimId,
    DomainInvariantError,
    MemoryId,
    MemoryIntegrityError,
    Relation,
    RelationType,
    TenantId,
)
from mindbridge.infrastructure._postgres_derived_records import derived_memory_content_digest
from mindbridge.infrastructure._postgres_embeddings import write_embedding_on_connection
from mindbridge.infrastructure._postgres_graph import write_claims, write_relations
from mindbridge.infrastructure._postgres_memories import write_memory_on_connection
from mindbridge.infrastructure._postgres_types import (
    DatabaseConnection,
    DatabasePool,
    tenant_connection,
)

ClaimState: TypeAlias = tuple[datetime | None, str | None]
ClaimDecision: TypeAlias = tuple[str, str, str]


async def commit_claim_consolidation(
    pool: DatabasePool,
    tenant_id: TenantId,
    write: ClaimConsolidationWrite,
) -> ClaimConsolidationCommit:
    """Lock source Claims and commit semantic aggregates or version edges once."""
    _require_tenant(tenant_id, write)
    referenced_claim_ids = sorted(
        {
            *(claim_id for item in write.semantic_claims for claim_id in item.source_claim_ids),
            *(
                ClaimId(claim_id)
                for relation in write.relationships
                for claim_id in (relation.source_id, relation.target_id)
            ),
        }
    )
    if not referenced_claim_ids:
        return ClaimConsolidationCommit(semantic_claim_count=0, relationship_count=0)

    try:
        async with tenant_connection(pool, tenant_id) as connection:
            states = await _lock_claims(connection, tenant_id, referenced_claim_ids)
            support_targets = await _read_support_targets(
                connection,
                tenant_id,
                referenced_claim_ids,
            )
            decisions = await _read_claim_decisions(
                connection,
                tenant_id,
                referenced_claim_ids,
            )
            semantic_count = 0
            for semantic_write in write.semantic_claims:
                committed = await _commit_semantic_claim(
                    connection,
                    semantic_write,
                    states,
                    support_targets,
                )
                semantic_count += int(committed)
            relationship_count = 0
            for relationship in write.relationships:
                committed = await _commit_relationship(
                    connection,
                    relationship,
                    states,
                    decisions,
                )
                relationship_count += int(committed)
            return ClaimConsolidationCommit(
                semantic_claim_count=semantic_count,
                relationship_count=relationship_count,
            )
    except ForeignKeyViolation as error:
        raise DomainInvariantError("Claim consolidation references missing source data") from error


def _require_tenant(tenant_id: TenantId, write: ClaimConsolidationWrite) -> None:
    if any(item.claim.tenant_id != tenant_id for item in write.semantic_claims) or any(
        relation.tenant_id != tenant_id for relation in write.relationships
    ):
        raise DomainInvariantError("Claim consolidation must remain in the requested tenant")


async def _lock_claims(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    claim_ids: list[ClaimId],
) -> dict[ClaimId, ClaimState]:
    cursor = await connection.execute(
        """
        SELECT claim_id, superseded_at, supersedes_claim_id
        FROM claims
        WHERE tenant_id = %s AND claim_id = ANY(%s)
        ORDER BY claim_id
        FOR UPDATE
        """,
        (tenant_id, list(claim_ids)),
    )
    return {
        ClaimId(claim_id): (superseded_at, supersedes_claim_id)
        async for row in cursor
        for claim_id, superseded_at, supersedes_claim_id in (
            cast(tuple[str, datetime | None, str | None], row),
        )
    }


async def _read_support_targets(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    claim_ids: list[ClaimId],
) -> dict[ClaimId, ClaimId]:
    cursor = await connection.execute(
        """
        SELECT source_id, target_id
        FROM relations
        WHERE tenant_id = %s
          AND source_type = 'claim'
          AND source_id = ANY(%s)
          AND relation_type = 'supports'
          AND target_type = 'claim'
        """,
        (tenant_id, list(claim_ids)),
    )
    targets: dict[ClaimId, ClaimId] = {}
    async for row in cursor:
        source_id, target_id = cast(tuple[str, str], row)
        source = ClaimId(source_id)
        target = ClaimId(target_id)
        if source in targets and targets[source] != target:
            raise MemoryIntegrityError("one Claim supports multiple semantic successors")
        targets[source] = target
    return targets


async def _read_claim_decisions(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    claim_ids: list[ClaimId],
) -> dict[frozenset[ClaimId], ClaimDecision]:
    cursor = await connection.execute(
        """
        SELECT source_id, relation_type, target_id
        FROM relations
        WHERE tenant_id = %s
          AND source_type = 'claim'
          AND source_id = ANY(%s)
          AND target_type = 'claim'
          AND target_id = ANY(%s)
          AND relation_type IN ('contradicts', 'supersedes')
        """,
        (tenant_id, list(claim_ids), list(claim_ids)),
    )
    decisions: dict[frozenset[ClaimId], ClaimDecision] = {}
    async for row in cursor:
        source_id, relation_type, target_id = cast(tuple[str, str, str], row)
        pair = frozenset((ClaimId(source_id), ClaimId(target_id)))
        decision = (source_id, relation_type, target_id)
        if pair in decisions and decisions[pair] != decision:
            raise MemoryIntegrityError("one Claim pair stores conflicting semantic decisions")
        decisions[pair] = decision
    return decisions


async def _commit_semantic_claim(
    connection: DatabaseConnection,
    write: SemanticClaimWrite,
    states: dict[ClaimId, ClaimState],
    support_targets: dict[ClaimId, ClaimId],
) -> bool:
    source_states = tuple(states.get(claim_id) for claim_id in write.source_claim_ids)
    if any(state is None or state[0] is not None for state in source_states):
        return False
    targets = tuple(support_targets.get(claim_id) for claim_id in write.source_claim_ids)
    if all(target == write.claim.claim_id for target in targets):
        await _write_semantic_claim(connection, write)
        return False
    if any(target is not None for target in targets):
        return False
    await _write_semantic_claim(connection, write)
    for claim_id in write.source_claim_ids:
        support_targets[claim_id] = write.claim.claim_id
    return True


async def _write_semantic_claim(
    connection: DatabaseConnection,
    write: SemanticClaimWrite,
) -> None:
    await write_claims(connection, (write.claim,))
    await write_memory_on_connection(
        connection,
        write.memory,
        derived_memory_content_digest(write.memory),
    )
    await write_relations(connection, write.relations)
    await write_embedding_on_connection(connection, write.embedding)


async def _commit_relationship(
    connection: DatabaseConnection,
    relationship: Relation,
    states: dict[ClaimId, ClaimState],
    decisions: dict[frozenset[ClaimId], ClaimDecision],
) -> bool:
    source_id = ClaimId(relationship.source_id)
    target_id = ClaimId(relationship.target_id)
    pair = frozenset((source_id, target_id))
    expected = (relationship.source_id, relationship.relation_type.value, relationship.target_id)
    if pair in decisions:
        if decisions[pair] != expected:
            return False
        await write_relations(connection, (relationship,))
        return False
    source_state = states.get(source_id)
    target_state = states.get(target_id)
    if source_state is None or target_state is None:
        return False
    if source_state[0] is not None or target_state[0] is not None:
        return False
    if relationship.relation_type is RelationType.SUPERSEDES:
        if source_state[1] not in {None, target_id}:
            return False
        target_memory_id = await _lock_current_claim_memory(
            connection,
            relationship.tenant_id,
            target_id,
        )
        if target_memory_id is None:
            return False
        await _supersede_claim(connection, relationship, target_memory_id)
        states[source_id] = (None, target_id)
        states[target_id] = (relationship.created_at, target_state[1])
    await write_relations(connection, (relationship,))
    decisions[pair] = expected
    return True


async def _lock_current_claim_memory(
    connection: DatabaseConnection,
    tenant_id: TenantId,
    claim_id: ClaimId,
) -> MemoryId | None:
    cursor = await connection.execute(
        """
        SELECT memory.memory_id
        FROM relations AS representation
        JOIN memory_records AS memory
          ON memory.tenant_id = representation.tenant_id
         AND memory.memory_id = representation.target_id
        WHERE representation.tenant_id = %s
          AND representation.source_type = 'claim'
          AND representation.source_id = %s
          AND representation.relation_type = 'represented_by'
          AND representation.target_type = 'memory_record'
          AND memory.superseded_at IS NULL
        FOR UPDATE OF memory
        """,
        (tenant_id, claim_id),
    )
    rows = await cursor.fetchall()
    if len(rows) > 1:
        raise MemoryIntegrityError("one Claim has multiple current Memory representations")
    return MemoryId(cast(tuple[str], rows[0])[0]) if rows else None


async def _supersede_claim(
    connection: DatabaseConnection,
    relationship: Relation,
    target_memory_id: MemoryId,
) -> None:
    source_cursor = await connection.execute(
        """
        UPDATE claims SET supersedes_claim_id = %s
        WHERE tenant_id = %s AND claim_id = %s
          AND (supersedes_claim_id IS NULL OR supersedes_claim_id = %s)
        """,
        (
            relationship.target_id,
            relationship.tenant_id,
            relationship.source_id,
            relationship.target_id,
        ),
    )
    target_cursor = await connection.execute(
        """
        UPDATE claims SET superseded_at = %s
        WHERE tenant_id = %s AND claim_id = %s AND superseded_at IS NULL
        """,
        (relationship.created_at, relationship.tenant_id, relationship.target_id),
    )
    memory_cursor = await connection.execute(
        """
        UPDATE memory_records SET superseded_at = %s
        WHERE tenant_id = %s AND memory_id = %s AND superseded_at IS NULL
        """,
        (relationship.created_at, relationship.tenant_id, target_memory_id),
    )
    if source_cursor.rowcount != 1 or target_cursor.rowcount != 1 or memory_cursor.rowcount != 1:
        raise MemoryIntegrityError("Claim supersession did not update one complete version pair")
