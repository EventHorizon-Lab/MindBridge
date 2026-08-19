"""Invariant tests for typed memory graph records."""

from datetime import datetime, timezone

import pytest

from mindbridge.core import (
    DomainInvariantError,
    Entity,
    EntityId,
    EntityMention,
    EntityType,
    EventId,
    EvidenceId,
    MentionId,
    Relation,
    RelationId,
    RelationNodeType,
    RelationType,
    TenantId,
)

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_entity_and_grounded_mention_accept_explicit_domain_types() -> None:
    entity = Entity(
        entity_id=EntityId("entity_tool"),
        tenant_id=TenantId("tenant_01"),
        entity_type=EntityType.OBJECT,
        canonical_name="red screwdriver",
        created_at=NOW,
    )
    mention = EntityMention(
        mention_id=MentionId("mention_tool"),
        tenant_id=entity.tenant_id,
        entity_id=entity.entity_id,
        event_id=EventId("event_01"),
        evidence_id=EvidenceId("evidence_01"),
        confidence=0.9,
        created_at=NOW,
    )

    assert mention.entity_id == entity.entity_id


def test_relation_rejects_a_self_edge() -> None:
    with pytest.raises(DomainInvariantError, match="itself"):
        Relation(
            relation_id=RelationId("relation_self"),
            tenant_id=TenantId("tenant_01"),
            source_type=RelationNodeType.EVENT,
            source_id="event_01",
            relation_type=RelationType.SAME_EPISODE,
            target_type=RelationNodeType.EVENT,
            target_id="event_01",
            created_at=NOW,
        )


def test_entity_resolution_relation_types_are_available() -> None:
    """Entity resolution needs a verdict vocabulary the store can index."""
    assert RelationType.SAME_AS.value == "same_as"
    assert RelationType.NOT_SAME_AS.value == "not_same_as"
