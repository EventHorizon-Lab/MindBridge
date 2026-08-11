"""Tests for evidence-backed event, claim, and embedding contracts."""

from datetime import datetime, timezone

import pytest

from mindbridge.core import (
    Claim,
    ClaimId,
    DomainInvariantError,
    EmbeddedObjectType,
    EmbeddingId,
    EmbeddingRecord,
    EmbeddingSpaceReference,
    Event,
    EventId,
    EvidenceId,
    MemoryId,
    MemoryRecord,
    MemoryType,
    ModelReference,
    ObservationId,
    TenantId,
    VerificationStatus,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
TENANT_ID = TenantId("tenant_01")
EVIDENCE_ID = EvidenceId("evidence_01")
MODEL = ModelReference(model_id="Qwen/Qwen3-Omni", revision="2026-08-01")


def test_derived_records_preserve_provenance() -> None:
    """Valid derived records retain evidence and exact model metadata."""
    event = _event()
    embedding = _embedding()

    assert event.evidence_ids == (EVIDENCE_ID,)
    assert event.model_reference == MODEL
    assert embedding.model_reference.revision == "abcdef0"


def test_event_requires_traceable_evidence() -> None:
    """A semantic event cannot replace the evidence it was derived from."""
    with pytest.raises(DomainInvariantError, match="evidence_ids"):
        _event(evidence_ids=())


def test_event_rejects_duplicate_observations() -> None:
    """Repeated identifiers cannot inflate event provenance."""
    observation_id = ObservationId("observation_01")

    with pytest.raises(DomainInvariantError, match="duplicates"):
        _event(observation_ids=(observation_id, observation_id))


def test_verified_claim_requires_evidence() -> None:
    """Only evidence-backed claims may be presented as verified facts."""
    with pytest.raises(DomainInvariantError, match="verified claim"):
        _claim(evidence_ids=(), verification_status=VerificationStatus.VERIFIED)


def test_unverified_claim_may_record_unsupported_input() -> None:
    """Explicit unsupported input stays queryable without becoming a fact."""
    claim = _claim(evidence_ids=(), verification_status=VerificationStatus.UNVERIFIED)

    assert claim.evidence_ids == ()


def test_attested_memory_preserves_source_statement_without_claiming_observation() -> None:
    """Caller-provided text stays usable while distinct from verified sensor evidence."""
    memory = MemoryRecord(
        memory_id=MemoryId("memory_attested"),
        tenant_id=TENANT_ID,
        memory_type=MemoryType.SEMANTIC,
        summary="Caroline said she plans to become a counselor.",
        evidence_ids=(),
        occurred_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        verification_status=VerificationStatus.ATTESTED,
    )

    assert memory.verification_status is VerificationStatus.ATTESTED


def test_verified_memory_requires_evidence() -> None:
    """The unified memory view cannot turn an unsupported summary into fact."""
    with pytest.raises(DomainInvariantError, match="verified memory"):
        MemoryRecord(
            memory_id=MemoryId("memory_01"),
            tenant_id=TENANT_ID,
            memory_type=MemoryType.SEMANTIC,
            summary="The screwdriver is in the toolbox.",
            evidence_ids=(),
            occurred_at=NOW,
            ended_at=NOW,
            created_at=NOW,
            verification_status=VerificationStatus.VERIFIED,
        )


def test_claim_rejects_reversed_validity() -> None:
    """World-valid time cannot end before it starts."""
    with pytest.raises(DomainInvariantError, match="valid_to"):
        _claim(valid_to=datetime(2026, 8, 11, 11, 0, tzinfo=timezone.utc))


def test_embedding_dimension_must_match_values() -> None:
    """Stored vector metadata must describe the actual vector."""
    with pytest.raises(DomainInvariantError, match="dimension"):
        _embedding(dimension=3)


def test_normalized_embedding_must_have_unit_length() -> None:
    """A vector cannot be mislabeled as normalized."""
    with pytest.raises(DomainInvariantError, match="unit length"):
        _embedding(values=(1.0, 1.0))


def _event(
    *,
    observation_ids: tuple[ObservationId, ...] = (ObservationId("observation_01"),),
    evidence_ids: tuple[EvidenceId, ...] = (EVIDENCE_ID,),
) -> Event:
    return Event(
        event_id=EventId("event_01"),
        tenant_id=TENANT_ID,
        observation_ids=observation_ids,
        evidence_ids=evidence_ids,
        occurred_at=NOW,
        ended_at=NOW,
        description="The robot placed a red screwdriver beside the blue toolbox.",
        salience=0.8,
        created_at=NOW,
        model_reference=MODEL,
        prompt_version="perceive_events_v1",
    )


def _claim(
    *,
    evidence_ids: tuple[EvidenceId, ...] = (EVIDENCE_ID,),
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    valid_to: datetime | None = None,
) -> Claim:
    return Claim(
        claim_id=ClaimId("claim_01"),
        tenant_id=TENANT_ID,
        statement="The red screwdriver is beside the blue toolbox.",
        evidence_ids=evidence_ids,
        confidence=0.91,
        verification_status=verification_status,
        valid_from=NOW,
        valid_to=valid_to,
        created_at=NOW,
        model_reference=MODEL,
    )


def _embedding(
    *,
    values: tuple[float, ...] = (1.0, 0.0),
    dimension: int = 2,
) -> EmbeddingRecord:
    return EmbeddingRecord(
        embedding_id=EmbeddingId("embedding_01"),
        tenant_id=TENANT_ID,
        object_type=EmbeddedObjectType.EVENT,
        object_id="event_01",
        values=values,
        model_reference=ModelReference(
            model_id="jinaai/jina-embeddings-v5-omni-small",
            revision="abcdef0",
        ),
        space_reference=EmbeddingSpaceReference(space_id="jina-v5", revision="space-v1"),
        task="retrieval_document",
        dimension=dimension,
        normalized=True,
        created_at=NOW,
    )
