"""Shared PostgreSQL row shape and mapping for memory records."""

from datetime import datetime
from typing import TypeAlias

from mindbridge.core import (
    EvidenceId,
    MemoryId,
    MemoryRecord,
    MemoryState,
    MemoryType,
    ModelReference,
    TenantId,
    VerificationStatus,
)

MemoryRow: TypeAlias = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    datetime,
    datetime,
    datetime,
    str | None,
    float,
    float,
    int,
    int,
    int,
    datetime | None,
    str | None,
    datetime | None,
    list[str],
]

MEMORY_SELECT_SQL = """
SELECT memory.memory_id,
       memory.tenant_id,
       memory.memory_type,
       memory.summary,
       memory.verification_status,
       memory.state,
       memory.occurred_at,
       memory.ended_at,
       memory.created_at,
       memory.model_id,
       memory.salience,
       memory.strength,
       memory.useful_access_count,
       memory.positive_feedback_count,
       memory.negative_feedback_count,
       memory.last_accessed_at,
       memory.supersedes_memory_id,
       memory.superseded_at,
       ARRAY(
           SELECT link.evidence_id
           FROM memory_evidence AS link
           WHERE link.tenant_id = memory.tenant_id AND link.memory_id = memory.memory_id
           ORDER BY link.evidence_id
       ) AS evidence_ids
FROM memory_records AS memory
"""

MEMORY_NOT_TOMBSTONED_SQL = """NOT EXISTS (
      SELECT 1
      FROM deletion_tombstones AS tombstone
      WHERE tombstone.tenant_id = memory.tenant_id
        AND (
            (tombstone.target_type = 'memory_record' AND tombstone.target_id = memory.memory_id)
            OR (
                tombstone.target_type = 'observation'
                AND EXISTS (
                    SELECT 1
                    FROM memory_evidence AS deleted_link
                    JOIN evidence_spans AS deleted_evidence
                      ON deleted_evidence.tenant_id = deleted_link.tenant_id
                     AND deleted_evidence.evidence_id = deleted_link.evidence_id
                    WHERE deleted_link.tenant_id = memory.tenant_id
                      AND deleted_link.memory_id = memory.memory_id
                      AND deleted_evidence.observation_id = tombstone.target_id
                )
            )
        )
  )"""


def memory_from_row(row: MemoryRow) -> MemoryRecord:
    """Convert one validated SQL projection into its domain record."""
    (
        memory_id,
        tenant_id,
        memory_type,
        summary,
        verification_status,
        state,
        occurred_at,
        ended_at,
        created_at,
        model_id,
        salience,
        strength,
        useful_access_count,
        positive_feedback_count,
        negative_feedback_count,
        last_accessed_at,
        supersedes_memory_id,
        superseded_at,
        evidence_ids,
    ) = row
    model_reference = ModelReference(model_id=model_id) if model_id is not None else None
    return MemoryRecord(
        memory_id=MemoryId(memory_id),
        tenant_id=TenantId(tenant_id),
        memory_type=MemoryType(memory_type),
        summary=summary,
        evidence_ids=tuple(EvidenceId(value) for value in evidence_ids),
        occurred_at=occurred_at,
        ended_at=ended_at,
        created_at=created_at,
        verification_status=VerificationStatus(verification_status),
        state=MemoryState(state),
        model_reference=model_reference,
        salience=salience,
        strength=strength,
        useful_access_count=useful_access_count,
        positive_feedback_count=positive_feedback_count,
        negative_feedback_count=negative_feedback_count,
        last_accessed_at=last_accessed_at,
        supersedes_memory_id=(
            MemoryId(supersedes_memory_id) if supersedes_memory_id is not None else None
        ),
        superseded_at=superseded_at,
    )
