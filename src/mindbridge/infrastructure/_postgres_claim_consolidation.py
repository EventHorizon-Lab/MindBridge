"""PostgreSQL candidates for bounded semantic Claim consolidation."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias, cast

from mindbridge.application.claim_consolidation import (
    ClaimCandidate,
    ClaimCandidatePage,
    ClaimCandidateRequest,
)
from mindbridge.core import (
    Claim,
    ClaimId,
    ClaimType,
    EntityId,
    EvidenceId,
    ModelReference,
    TenantId,
    VerificationStatus,
)
from mindbridge.infrastructure._postgres_types import (
    PostgresStoreOperations,
    tenant_connection,
)

ClaimCandidateRow: TypeAlias = tuple[
    str,
    str,
    str | None,
    str,
    str,
    list[str],
    float,
    str,
    datetime,
    datetime | None,
    datetime,
    str,
    str,
    list[str],
]


def _claim_candidate_from_row(row: ClaimCandidateRow) -> ClaimCandidate:
    (
        claim_id,
        tenant_id,
        supersedes_claim_id,
        claim_type,
        statement,
        evidence_ids,
        confidence,
        verification_status,
        valid_from,
        valid_to,
        created_at,
        model_id,
        prompt_version,
        entity_ids,
    ) = row
    return ClaimCandidate(
        claim=Claim(
            claim_id=ClaimId(claim_id),
            tenant_id=TenantId(tenant_id),
            supersedes_claim_id=(
                ClaimId(supersedes_claim_id) if supersedes_claim_id is not None else None
            ),
            claim_type=ClaimType(claim_type),
            statement=statement,
            evidence_ids=tuple(EvidenceId(value) for value in evidence_ids),
            confidence=confidence,
            verification_status=VerificationStatus(verification_status),
            valid_from=valid_from,
            valid_to=valid_to,
            created_at=created_at,
            model_reference=ModelReference(model_id=model_id),
            prompt_version=prompt_version,
        ),
        entity_ids=tuple(EntityId(value) for value in entity_ids),
    )


_CURRENT_CLAIM_SQL = """
claim.superseded_at IS NULL
AND claim.created_at < %(evaluated_at)s
AND NOT EXISTS (
    SELECT 1 FROM relations AS absorbed
    WHERE absorbed.tenant_id = claim.tenant_id
      AND absorbed.source_type = 'claim'
      AND absorbed.source_id = claim.claim_id
      AND absorbed.relation_type = 'supports'
      AND absorbed.target_type = 'claim'
)
"""

_CLAIM_SEEDS_SQL = f"""
SELECT claim.claim_id
FROM claims AS claim
WHERE claim.tenant_id = %(tenant_id)s
  AND {_CURRENT_CLAIM_SQL}
  AND (%(after_claim_id)s::text IS NULL OR claim.claim_id > %(after_claim_id)s)
ORDER BY claim.claim_id
LIMIT %(limit)s
"""

_RELATED_CLAIM_IDS_SQL = f"""
WITH seed_ids AS (
    SELECT claim_id FROM unnest(%(seed_ids)s::text[]) AS seed(claim_id)
),
related_pairs AS (
    SELECT seed.claim_id AS seed_id, peer.claim_id AS peer_id
    FROM seed_ids
    JOIN claims AS seed
      ON seed.tenant_id = %(tenant_id)s AND seed.claim_id = seed_ids.claim_id
    JOIN claims AS peer
      ON peer.tenant_id = seed.tenant_id
     AND peer.claim_id > seed.claim_id
     AND peer.claim_type = seed.claim_type
     AND peer.valid_from <= seed.valid_from + make_interval(secs => %(maximum_gap_seconds)s)
     AND seed.valid_from <= peer.valid_from + make_interval(secs => %(maximum_gap_seconds)s)
    WHERE {_CURRENT_CLAIM_SQL.replace("claim.", "peer.")}
      AND NOT EXISTS (
          SELECT 1 FROM relations AS decided
          WHERE decided.tenant_id = seed.tenant_id
            AND decided.source_type = 'claim'
            AND decided.target_type = 'claim'
            AND decided.relation_type IN ('supports', 'contradicts', 'supersedes')
            AND (
                (decided.source_id = seed.claim_id AND decided.target_id = peer.claim_id)
                OR
                (decided.source_id = peer.claim_id AND decided.target_id = seed.claim_id)
            )
      )
      AND (
          EXISTS (
              SELECT 1
              FROM relations AS seed_about
              JOIN relations AS peer_about
                ON peer_about.tenant_id = seed_about.tenant_id
               AND peer_about.target_type = 'entity'
               AND peer_about.target_id = seed_about.target_id
               AND peer_about.relation_type = 'about'
              WHERE seed_about.tenant_id = seed.tenant_id
                AND seed_about.source_type = 'claim'
                AND seed_about.source_id = seed.claim_id
                AND seed_about.relation_type = 'about'
                AND peer_about.source_type = 'claim'
                AND peer_about.source_id = peer.claim_id
          )
          OR EXISTS (
              SELECT 1
              FROM embeddings AS seed_embedding
              JOIN embeddings AS peer_embedding
                ON peer_embedding.tenant_id = seed_embedding.tenant_id
               AND peer_embedding.space_id = seed_embedding.space_id
               AND peer_embedding.task = seed_embedding.task
              WHERE seed_embedding.tenant_id = seed.tenant_id
                AND seed_embedding.object_type = 'claim'
                AND seed_embedding.object_id = seed.claim_id
                AND peer_embedding.object_type = 'claim'
                AND peer_embedding.object_id = peer.claim_id
                AND 1 - (seed_embedding.embedding <=> peer_embedding.embedding)
                    >= %(minimum_similarity)s
          )
      )
),
related_ids AS (
    SELECT seed_id AS claim_id FROM related_pairs
    UNION
    SELECT peer_id AS claim_id FROM related_pairs
)
SELECT claim_id FROM related_ids ORDER BY claim_id LIMIT %(candidate_limit)s
"""

_READ_CLAIM_CANDIDATES_SQL = """
SELECT claim.claim_id,
       claim.tenant_id,
       claim.supersedes_claim_id,
       claim.claim_type,
       claim.statement,
       ARRAY(
           SELECT link.evidence_id
           FROM claim_evidence AS link
           WHERE link.tenant_id = claim.tenant_id AND link.claim_id = claim.claim_id
           ORDER BY link.evidence_id
       ) AS evidence_ids,
       claim.confidence,
       claim.verification_status,
       claim.valid_from,
       claim.valid_to,
       claim.created_at,
       claim.model_id,
       claim.prompt_version,
       ARRAY(
           SELECT relation.target_id
           FROM relations AS relation
           WHERE relation.tenant_id = claim.tenant_id
             AND relation.source_type = 'claim'
             AND relation.source_id = claim.claim_id
             AND relation.relation_type = 'about'
             AND relation.target_type = 'entity'
           ORDER BY relation.target_id
       ) AS entity_ids
FROM claims AS claim
WHERE claim.tenant_id = %s AND claim.claim_id = ANY(%s)
ORDER BY claim.valid_from, claim.claim_id
"""


class ClaimCandidateOperations(PostgresStoreOperations):
    async def list_claim_candidates(
        self,
        request: ClaimCandidateRequest,
    ) -> ClaimCandidatePage:
        """Return one stable seed page expanded by entity or aligned-vector affinity."""
        async with tenant_connection(self._pool, request.tenant_id) as connection:
            cursor = await connection.execute(
                _CLAIM_SEEDS_SQL,
                {
                    "tenant_id": request.tenant_id,
                    "evaluated_at": request.evaluated_at,
                    "after_claim_id": request.after_claim_id,
                    "limit": request.limit + 1,
                },
            )
            seed_ids = tuple([ClaimId(cast(tuple[str], row)[0]) async for row in cursor])
            page_seed_ids = seed_ids[: request.limit]
            if not page_seed_ids:
                return ClaimCandidatePage(candidates=(), scanned_count=0, next_cursor=None)

            cursor = await connection.execute(
                _RELATED_CLAIM_IDS_SQL,
                {
                    "tenant_id": request.tenant_id,
                    "evaluated_at": request.evaluated_at,
                    "seed_ids": list(page_seed_ids),
                    "maximum_gap_seconds": request.maximum_gap_seconds,
                    "minimum_similarity": request.minimum_similarity,
                    "candidate_limit": min(request.limit * 4, 64),
                },
            )
            claim_ids = tuple([ClaimId(cast(tuple[str], row)[0]) async for row in cursor])
            cursor = await connection.execute(
                _READ_CLAIM_CANDIDATES_SQL,
                (request.tenant_id, list(claim_ids)),
            )
            candidates = tuple(
                [_claim_candidate_from_row(cast(ClaimCandidateRow, row)) async for row in cursor]
            )
        return ClaimCandidatePage(
            candidates=candidates,
            scanned_count=len(page_seed_ids),
            next_cursor=(page_seed_ids[-1] if len(seed_ids) > request.limit else None),
        )
