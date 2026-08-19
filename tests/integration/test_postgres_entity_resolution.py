"""Integration checks for the entity resolution candidate page and verdict write."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from psycopg import AsyncConnection

from mindbridge.application.consolidate_entities import EntityResolutionStore
from mindbridge.application.entity_resolution import (
    EntityAdjudication,
    EntityCandidatePage,
    EntityCandidateRequest,
    EntityResolutionWrite,
    derive_entity_resolution_write,
)
from mindbridge.core import EntityId, EntityType, RelationType, TenantId
from mindbridge.infrastructure.postgres import PostgresMemoryStore

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)


async def _seed(
    database_url: str,
    tenant_id: str,
    *,
    entities: tuple[tuple[str, str | None, str], ...],
    minutes_apart: int = 1,
) -> None:
    """Write entities, one event per entity, and a mention joining them.

    Written straight through SQL rather than through the perception pipeline: these tests are
    about the candidate query, so the interesting states are the ones perception cannot
    currently produce — a null canonical_name beside a named one, a pair already judged.
    """
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        for index, (entity_id, canonical_name, entity_type) in enumerate(entities):
            occurred_at = NOW + timedelta(minutes=index * minutes_apart)
            await connection.execute(
                "INSERT INTO entities (tenant_id, entity_id, entity_type, canonical_name,"
                " created_at) VALUES (%s, %s, %s, %s, %s)",
                (tenant_id, entity_id, entity_type, canonical_name, NOW),
            )
            await connection.execute(
                "INSERT INTO events (tenant_id, event_id, hierarchy_level, description,"
                " salience, status, occurred_at, ended_at, model_id, model_revision,"
                " prompt_version, content_digest, created_at)"
                " VALUES (%s, %s, 'event', %s, 0.5, 'active', %s, %s, 'm', 'r', 'p', %s, %s)",
                (
                    tenant_id,
                    f"event_{entity_id}",
                    f"an event mentioning {entity_id}",
                    occurred_at,
                    occurred_at + timedelta(seconds=30),
                    f"{index:064x}",
                    NOW,
                ),
            )
            await connection.execute(
                "INSERT INTO entity_mentions (tenant_id, mention_id, entity_id, event_id,"
                " evidence_id, confidence, created_at) VALUES (%s, %s, %s, %s, NULL, 0.9, %s)",
                (tenant_id, f"mention_{entity_id}", entity_id, f"event_{entity_id}", NOW),
            )


def _request(tenant_id: str, **overrides: object) -> EntityCandidateRequest:
    return EntityCandidateRequest(
        tenant_id=TenantId(tenant_id),
        evaluated_at=NOW,
        **overrides,  # type: ignore[arg-type]
    )


def _pair_ids(page: EntityCandidatePage) -> list[tuple[str, str]]:
    return [(item.left.entity.entity_id, item.right.entity.entity_id) for item in page.pairs]


async def test_candidates_exclude_identity_backed_entities(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """A null canonical_name means an edge signal already fixed this identity."""
    tenant_id = "tenant_entity_null_name"
    await _seed(
        database_url,
        tenant_id,
        entities=(
            ("entity_a", "man in denim jacket", "person"),
            ("entity_b", "man in black t-shirt", "person"),
            ("entity_c", None, "person"),
        ),
    )

    page = await store.list_entity_candidates(_request(tenant_id))

    assert _pair_ids(page) == [("entity_a", "entity_b")]


async def test_pairs_come_back_in_one_canonical_order(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """Only entity_id_a < entity_id_b, so one unordered pair is one candidate."""
    tenant_id = "tenant_entity_order"
    await _seed(
        database_url,
        tenant_id,
        entities=(
            ("entity_b", "second", "person"),
            ("entity_a", "first", "person"),
            ("entity_c", "third", "person"),
        ),
    )

    page = await store.list_entity_candidates(_request(tenant_id))

    assert _pair_ids(page) == [
        ("entity_a", "entity_b"),
        ("entity_a", "entity_c"),
        ("entity_b", "entity_c"),
    ]


async def test_a_different_entity_type_is_never_paired(
    store: PostgresMemoryStore, database_url: str
) -> None:
    tenant_id = "tenant_entity_types"
    await _seed(
        database_url,
        tenant_id,
        entities=(
            ("entity_a", "a person", "person"),
            ("entity_b", "a place", "place"),
        ),
    )

    assert _pair_ids(await store.list_entity_candidates(_request(tenant_id))) == []


async def test_an_already_judged_pair_is_not_returned_again(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """Either verdict settles the pair, so the sweep stops paying for it."""
    tenant_id = "tenant_entity_settled"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    page = await store.list_entity_candidates(_request(tenant_id))
    write = derive_entity_resolution_write(
        TenantId(tenant_id),
        ((page.pairs[0], EntityAdjudication(False, 0.9, "different height")),),
        NOW,
    )
    assert await store.commit_entity_resolution(TenantId(tenant_id), write) == 1

    assert _pair_ids(await store.list_entity_candidates(_request(tenant_id))) == []
    assert _pair_ids(
        await store.list_entity_candidates(_request(tenant_id, readjudicate=True))
    ) == [("entity_a", "entity_b")]


async def test_the_pair_bound_is_reported_not_hidden(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """A sweep that looked at less than it found has to say so."""
    tenant_id = "tenant_entity_bound"
    await _seed(
        database_url,
        tenant_id,
        entities=tuple((f"entity_{index}", f"person {index}", "person") for index in range(4)),
    )

    page = await store.list_entity_candidates(_request(tenant_id, maximum_pairs=2))

    assert len(page.pairs) == 2
    # Four named people of one type pair six ways; two were judged, four were left.
    assert page.dropped_pair_count == 4


async def test_the_time_window_excludes_a_distant_entity(
    store: PostgresMemoryStore, database_url: str
) -> None:
    tenant_id = "tenant_entity_window"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
        minutes_apart=60,
    )

    near = await store.list_entity_candidates(_request(tenant_id, maximum_gap_seconds=7_200))
    far = await store.list_entity_candidates(_request(tenant_id, maximum_gap_seconds=60))

    assert _pair_ids(near) == [("entity_a", "entity_b")]
    assert _pair_ids(far) == []


async def test_the_seed_cursor_advances_and_stops(
    store: PostgresMemoryStore, database_url: str
) -> None:
    tenant_id = "tenant_entity_cursor"
    await _seed(
        database_url,
        tenant_id,
        entities=tuple((f"entity_{index}", f"person {index}", "person") for index in range(3)),
    )

    first = await store.list_entity_candidates(_request(tenant_id, limit=2))
    assert first.next_cursor == EntityId("entity_1")
    assert first.scanned_count == 2

    second = await store.list_entity_candidates(
        _request(tenant_id, limit=2, after_entity_id=first.next_cursor)
    )
    assert second.next_cursor is None


async def test_a_re_judgement_replaces_the_verdict_instead_of_contradicting_it(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """One pair, one row. The graph must never assert same and not-same at once."""
    tenant_id = "tenant_entity_reverdict"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    page = await store.list_entity_candidates(_request(tenant_id))
    pair = page.pairs[0]

    same = derive_entity_resolution_write(
        TenantId(tenant_id), ((pair, EntityAdjudication(True, 0.9, "same scar")),), NOW
    )
    assert await store.commit_entity_resolution(TenantId(tenant_id), same) == 1
    # An unchanged re-run touches nothing, which is what makes the sweep idempotent.
    assert await store.commit_entity_resolution(TenantId(tenant_id), same) == 0

    flipped = derive_entity_resolution_write(
        TenantId(tenant_id),
        ((pair, EntityAdjudication(False, 0.95, "both on screen at once")),),
        NOW + timedelta(hours=1),
    )
    assert await store.commit_entity_resolution(TenantId(tenant_id), flipped) == 1

    assert await _verdicts(database_url, tenant_id) == [("not_same_as", "entity_a", "entity_b")]


async def test_committing_nothing_writes_nothing(store: PostgresMemoryStore) -> None:
    assert (
        await store.commit_entity_resolution(
            TenantId("tenant_entity_empty"), EntityResolutionWrite(decided=())
        )
        == 0
    )


async def test_a_committed_verdict_keeps_the_cue_it_rested_on(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The judge is required to name a cue; an operator auditing the merge must be able to
    read it back. Before this was stored, a same_as edge was a durable claim with nothing
    behind it."""
    tenant_id = "tenant_entity_cue"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    page = await store.list_entity_candidates(_request(tenant_id))
    write = derive_entity_resolution_write(
        TenantId(tenant_id),
        ((page.pairs[0], EntityAdjudication(True, 0.91, "identical scar above left brow")),),
        NOW,
    )
    assert await store.commit_entity_resolution(TenantId(tenant_id), write) == 1

    assert await _cues(database_url, tenant_id) == [
        ("entity_a", "entity_b", "same_as", 0.91, "identical scar above left brow", NOW)
    ]


async def test_a_re_judgement_that_holds_its_answer_still_replaces_the_cue(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """The edge upsert deliberately skips an unchanged verdict, so the cue write must not be
    gated on it. Otherwise --entity-readjudicate leaves yesterday's reasoning standing beside
    a verdict that was reached today on different grounds."""
    tenant_id = "tenant_entity_recue"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    pair = (await store.list_entity_candidates(_request(tenant_id))).pairs[0]

    first = derive_entity_resolution_write(
        TenantId(tenant_id), ((pair, EntityAdjudication(True, 0.80, "same red scarf")),), NOW
    )
    assert await store.commit_entity_resolution(TenantId(tenant_id), first) == 1

    later = NOW + timedelta(hours=1)
    second = derive_entity_resolution_write(
        TenantId(tenant_id), ((pair, EntityAdjudication(True, 0.97, "same scar")),), later
    )
    # Nothing an operator would call a decision changed, so the count stays honest at zero.
    assert await store.commit_entity_resolution(TenantId(tenant_id), second) == 0

    # One row still, carrying the reasoning the standing verdict actually rested on.
    assert await _cues(database_url, tenant_id) == [
        ("entity_a", "entity_b", "same_as", 0.97, "same scar", later)
    ]


async def test_a_flipped_verdict_replaces_the_cue_rather_than_appending_one(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """One pair owns one verdict row, cue included: the pair must never read as both."""
    tenant_id = "tenant_entity_flipcue"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    pair = (await store.list_entity_candidates(_request(tenant_id))).pairs[0]

    await store.commit_entity_resolution(
        TenantId(tenant_id),
        derive_entity_resolution_write(
            TenantId(tenant_id), ((pair, EntityAdjudication(True, 0.90, "same scar")),), NOW
        ),
    )
    later = NOW + timedelta(hours=1)
    await store.commit_entity_resolution(
        TenantId(tenant_id),
        derive_entity_resolution_write(
            TenantId(tenant_id),
            ((pair, EntityAdjudication(False, 0.95, "both on screen at once")),),
            later,
        ),
    )

    assert await _cues(database_url, tenant_id) == [
        ("entity_a", "entity_b", "not_same_as", 0.95, "both on screen at once", later)
    ]


async def test_deleting_an_edge_takes_its_justification_with_it(
    store: PostgresMemoryStore,
    database_url: str,
) -> None:
    """A cue that outlived its edge would justify a merge the graph no longer asserts."""
    tenant_id = "tenant_entity_cascade"
    await _seed(
        database_url,
        tenant_id,
        entities=(("entity_a", "first", "person"), ("entity_b", "second", "person")),
    )
    pair = (await store.list_entity_candidates(_request(tenant_id))).pairs[0]
    await store.commit_entity_resolution(
        TenantId(tenant_id),
        derive_entity_resolution_write(
            TenantId(tenant_id), ((pair, EntityAdjudication(True, 0.90, "same scar")),), NOW
        ),
    )
    assert await _stored_cue_count(database_url, tenant_id) == 1

    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        await connection.execute("DELETE FROM relations WHERE tenant_id = %s", (tenant_id,))

    # Counted straight off the table, never through a join to relations: the join would
    # report an orphaned cue as absent and this assertion would hold with no cascade at all.
    assert await _stored_cue_count(database_url, tenant_id) == 0


async def _stored_cue_count(database_url: str, tenant_id: str) -> int:
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        row = await (
            await connection.execute(
                "SELECT count(*) FROM entity_resolution_verdicts WHERE tenant_id = %s",
                (tenant_id,),
            )
        ).fetchone()
    assert row is not None
    return int(row[0])


async def _cues(
    database_url: str,
    tenant_id: str,
) -> list[tuple[str, str, str, float, str, datetime]]:
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        cursor = await connection.execute(
            "SELECT edge.source_id, edge.target_id, edge.relation_type, verdict.confidence,"
            " verdict.discriminating_cue, verdict.decided_at"
            " FROM entity_resolution_verdicts AS verdict"
            " JOIN relations AS edge"
            "   ON edge.tenant_id = verdict.tenant_id"
            "  AND edge.relation_id = verdict.relation_id"
            " WHERE verdict.tenant_id = %s ORDER BY edge.source_id, edge.target_id",
            (tenant_id,),
        )
        return [
            (str(row[0]), str(row[1]), str(row[2]), float(row[3]), str(row[4]), row[5])
            async for row in cursor
        ]


async def _verdicts(database_url: str, tenant_id: str) -> list[tuple[str, str, str]]:
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        cursor = await connection.execute(
            "SELECT relation_type, source_id, target_id FROM relations"
            " WHERE tenant_id = %s AND source_type = 'entity' AND target_type = 'entity'"
            " AND relation_type = ANY(%s) ORDER BY source_id, target_id",
            (tenant_id, [RelationType.SAME_AS.value, RelationType.NOT_SAME_AS.value]),
        )
        return [(str(row[0]), str(row[1]), str(row[2])) async for row in cursor]


def test_the_concrete_store_satisfies_the_use_case_port(store: PostgresMemoryStore) -> None:
    """Nothing assigns one to the other until the sweep lands, so mypy checks it here."""
    port: EntityResolutionStore = store
    assert port is store


def test_person_is_the_only_type_adjudicated_by_default() -> None:
    """Widening to objects is opt-in: identical risk, much lower value."""
    assert _request("tenant_entity_default").entity_types == (EntityType.PERSON,)
