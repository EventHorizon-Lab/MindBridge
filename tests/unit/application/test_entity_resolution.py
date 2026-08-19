"""Pure entity-resolution derivation: what becomes an edge, and what never does."""

from datetime import datetime, timezone

import pytest

from mindbridge.application.entity_resolution import (
    EntityAdjudication,
    EntityCandidate,
    EntityCandidatePage,
    EntityCandidateRequest,
    EntityPair,
    derive_entity_resolution_write,
)
from mindbridge.core import (
    DomainInvariantError,
    Entity,
    EntityId,
    EntityType,
    EvidenceId,
    RelationType,
    TenantId,
)

_AT = datetime(2026, 8, 19, tzinfo=timezone.utc)


def _candidate(
    entity_id: str,
    *,
    name: str | None = None,
    entity_type: EntityType = EntityType.PERSON,
) -> EntityCandidate:
    return EntityCandidate(
        entity=Entity(
            entity_id=EntityId(entity_id),
            tenant_id=TenantId("tenant_01"),
            entity_type=entity_type,
            canonical_name=name if name is not None else entity_id,
            created_at=_AT,
        ),
        evidence_ids=(EvidenceId("evidence_1"),),
    )


def _pair(left: str, right: str) -> EntityPair:
    return EntityPair(left=_candidate(left), right=_candidate(right))


def test_pair_rejects_unordered_or_self_referential_input() -> None:
    """Canonical ordering is what makes one edge per unordered pair."""
    with pytest.raises(DomainInvariantError):
        EntityPair(left=_candidate("entity_b"), right=_candidate("entity_a"))
    with pytest.raises(DomainInvariantError):
        EntityPair(left=_candidate("entity_a"), right=_candidate("entity_a"))


def test_pair_rejects_two_different_entity_types() -> None:
    """A person and a place are never the same entity, so they never reach a judge."""
    with pytest.raises(DomainInvariantError):
        EntityPair(
            left=_candidate("entity_a"),
            right=_candidate("entity_b", entity_type=EntityType.PLACE),
        )


def test_identity_backed_entities_are_never_candidates() -> None:
    """A null canonical_name means an edge signal already fixed this identity."""
    with pytest.raises(DomainInvariantError):
        EntityCandidate(
            entity=Entity(
                entity_id=EntityId("entity_a"),
                tenant_id=TenantId("tenant_01"),
                entity_type=EntityType.PERSON,
                canonical_name=None,
                created_at=_AT,
            ),
            evidence_ids=(EvidenceId("evidence_1"),),
        )


def test_a_positive_verdict_writes_one_same_as_edge() -> None:
    write = derive_entity_resolution_write(
        TenantId("tenant_01"),
        ((_pair("entity_a", "entity_b"), EntityAdjudication(True, 0.9, "same scar")),),
        _AT,
    )
    assert len(write.relations) == 1
    relation = write.relations[0]
    assert relation.relation_type is RelationType.SAME_AS
    assert (relation.source_id, relation.target_id) == ("entity_a", "entity_b")


def test_a_negative_verdict_writes_not_same_as_so_the_pair_is_not_re_paid_for() -> None:
    write = derive_entity_resolution_write(
        TenantId("tenant_01"),
        ((_pair("entity_a", "entity_b"), EntityAdjudication(False, 0.95, "different height")),),
        _AT,
    )
    assert [item.relation_type for item in write.relations] == [RelationType.NOT_SAME_AS]


def test_no_edge_is_inferred_transitively() -> None:
    """A~B and B~C must not produce A~C: that is how a cluster collapses."""
    write = derive_entity_resolution_write(
        TenantId("tenant_01"),
        (
            (_pair("entity_a", "entity_b"), EntityAdjudication(True, 0.9, "cue")),
            (_pair("entity_b", "entity_c"), EntityAdjudication(True, 0.9, "cue")),
        ),
        _AT,
    )
    pairs = {(item.source_id, item.target_id) for item in write.relations}
    assert pairs == {("entity_a", "entity_b"), ("entity_b", "entity_c")}


def test_relation_ids_are_stable_across_runs() -> None:
    decided = ((_pair("entity_a", "entity_b"), EntityAdjudication(True, 0.9, "cue")),)
    first = derive_entity_resolution_write(TenantId("tenant_01"), decided, _AT)
    second = derive_entity_resolution_write(TenantId("tenant_01"), decided, _AT)
    assert first.relations[0].relation_id == second.relations[0].relation_id


def test_a_pair_has_one_verdict_slot_whichever_way_it_is_judged() -> None:
    """Keyed on the pair, not the verdict: a re-judgement replaces, it cannot contradict."""
    pair = _pair("entity_a", "entity_b")
    same = derive_entity_resolution_write(
        TenantId("tenant_01"), ((pair, EntityAdjudication(True, 0.9, "cue")),), _AT
    )
    different = derive_entity_resolution_write(
        TenantId("tenant_01"), ((pair, EntityAdjudication(False, 0.9, "cue")),), _AT
    )
    assert same.relations[0].relation_id == different.relations[0].relation_id
    assert same.relations[0].relation_type is not different.relations[0].relation_type


def test_one_pair_cannot_carry_two_verdicts_in_one_write() -> None:
    pair = _pair("entity_a", "entity_b")
    with pytest.raises(DomainInvariantError):
        derive_entity_resolution_write(
            TenantId("tenant_01"),
            (
                (pair, EntityAdjudication(True, 0.9, "cue")),
                (pair, EntityAdjudication(False, 0.9, "cue")),
            ),
            _AT,
        )


def test_an_adjudication_must_name_what_it_decided_on() -> None:
    with pytest.raises(DomainInvariantError):
        EntityAdjudication(True, 0.9, "  ")


def test_default_candidacy_is_person_only_and_bounded() -> None:
    request = EntityCandidateRequest(tenant_id=TenantId("tenant_01"), evaluated_at=_AT)
    assert request.entity_types == (EntityType.PERSON,)
    assert request.maximum_pairs == 64
    assert request.readjudicate is False


def test_a_page_reports_what_its_bound_refused_to_look_at() -> None:
    page = EntityCandidatePage(
        pairs=(_pair("entity_a", "entity_b"),),
        scanned_count=9,
        dropped_pair_count=4,
        next_cursor=EntityId("entity_b"),
    )
    assert page.dropped_pair_count == 4
    with pytest.raises(DomainInvariantError):
        EntityCandidatePage(
            pairs=(_pair("entity_a", "entity_b"), _pair("entity_a", "entity_b")),
            scanned_count=2,
            dropped_pair_count=0,
            next_cursor=None,
        )
