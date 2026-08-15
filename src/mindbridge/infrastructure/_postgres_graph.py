"""Idempotent PostgreSQL writes for entities, claims, mentions, and relations."""

from __future__ import annotations

from mindbridge.core import Claim, Entity, EntityMention, MemoryIntegrityError, Relation
from mindbridge.infrastructure._postgres_types import DatabaseConnection


async def write_entities(
    connection: DatabaseConnection,
    entities: tuple[Entity, ...],
) -> None:
    """Insert entities once and reject collisions in their stable identity fields."""
    for entity in entities:
        identity = (
            entity.tenant_id,
            entity.entity_id,
            entity.entity_type.value,
            entity.canonical_name,
        )
        values = (
            *identity,
            entity.created_at,
        )
        cursor = await connection.execute(
            """
            INSERT INTO entities (
                tenant_id, entity_id, entity_type, canonical_name, created_at
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING entity_id
            """,
            values,
        )
        if await cursor.fetchone() is not None:
            continue
        row = await (
            await connection.execute(
                """
                SELECT tenant_id, entity_id, entity_type, canonical_name
                FROM entities WHERE tenant_id = %s AND entity_id = %s
                """,
                (entity.tenant_id, entity.entity_id),
            )
        ).fetchone()
        if row is None or tuple(row) != identity:
            raise MemoryIntegrityError("entity has conflicting deterministic identity")


async def write_entity_mentions(
    connection: DatabaseConnection,
    mentions: tuple[EntityMention, ...],
) -> None:
    """Insert exact entity occurrences after their graph nodes exist."""
    for mention in mentions:
        values = (
            mention.tenant_id,
            mention.mention_id,
            mention.entity_id,
            mention.event_id,
            mention.evidence_id,
            mention.confidence,
            mention.created_at,
        )
        cursor = await connection.execute(
            """
            INSERT INTO entity_mentions (
                tenant_id, mention_id, entity_id, event_id,
                evidence_id, confidence, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING mention_id
            """,
            values,
        )
        if await cursor.fetchone() is not None:
            continue
        row = await (
            await connection.execute(
                """
                SELECT tenant_id, mention_id, entity_id, event_id,
                       evidence_id, confidence, created_at
                FROM entity_mentions WHERE tenant_id = %s AND mention_id = %s
                """,
                (mention.tenant_id, mention.mention_id),
            )
        ).fetchone()
        if row is None or tuple(row) != values:
            raise MemoryIntegrityError("entity mention has conflicting deterministic identity")


async def write_claims(
    connection: DatabaseConnection,
    claims: tuple[Claim, ...],
) -> None:
    """Insert evidence-backed versioned assertions without overwriting history."""
    for claim in claims:
        values = (
            claim.tenant_id,
            claim.claim_id,
            claim.supersedes_claim_id,
            claim.claim_type.value,
            claim.statement,
            claim.confidence,
            claim.verification_status.value,
            claim.valid_from,
            claim.valid_to,
            claim.model_reference.model_id,
            claim.model_reference.revision,
            claim.prompt_version,
            claim.created_at,
        )
        cursor = await connection.execute(
            """
            INSERT INTO claims (
                tenant_id, claim_id, supersedes_claim_id, claim_type, statement,
                confidence, verification_status, valid_from, valid_to,
                model_id, model_revision, prompt_version, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING claim_id
            """,
            values,
        )
        created = await cursor.fetchone() is not None
        if not created:
            row = await (
                await connection.execute(
                    """
                    SELECT tenant_id, claim_id, supersedes_claim_id, claim_type, statement,
                           confidence, verification_status, valid_from, valid_to,
                           model_id, model_revision, prompt_version, created_at
                    FROM claims WHERE tenant_id = %s AND claim_id = %s
                    """,
                    (claim.tenant_id, claim.claim_id),
                )
            ).fetchone()
            if row is None or tuple(row) != values:
                raise MemoryIntegrityError("claim has conflicting deterministic identity")
            evidence_cursor = await connection.execute(
                """
                SELECT evidence_id FROM claim_evidence
                WHERE tenant_id = %s AND claim_id = %s
                """,
                (claim.tenant_id, claim.claim_id),
            )
            stored_evidence_ids = {str(row[0]) async for row in evidence_cursor}
            if stored_evidence_ids != {str(value) for value in claim.evidence_ids}:
                raise MemoryIntegrityError("claim has conflicting evidence provenance")
            continue
        async with connection.cursor() as evidence_cursor:
            await evidence_cursor.executemany(
                """
                INSERT INTO claim_evidence (tenant_id, claim_id, evidence_id)
                VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
                """,
                (
                    (claim.tenant_id, claim.claim_id, evidence_id)
                    for evidence_id in claim.evidence_ids
                ),
            )


async def write_relations(
    connection: DatabaseConnection,
    relations: tuple[Relation, ...],
) -> None:
    """Insert deterministic typed edges and reject hash collisions."""
    for relation in relations:
        values = (
            relation.tenant_id,
            relation.relation_id,
            relation.source_type.value,
            relation.source_id,
            relation.relation_type.value,
            relation.target_type.value,
            relation.target_id,
            relation.created_at,
        )
        cursor = await connection.execute(
            """
            INSERT INTO relations (
                tenant_id, relation_id, source_type, source_id, relation_type,
                target_type, target_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING relation_id
            """,
            values,
        )
        if await cursor.fetchone() is not None:
            continue
        row = await (
            await connection.execute(
                """
                SELECT tenant_id, relation_id, source_type, source_id, relation_type,
                       target_type, target_id, created_at
                FROM relations WHERE tenant_id = %s AND relation_id = %s
                """,
                (relation.tenant_id, relation.relation_id),
            )
        ).fetchone()
        if row is None or tuple(row) != values:
            raise MemoryIntegrityError("relation has conflicting deterministic identity")
