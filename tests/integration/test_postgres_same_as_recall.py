"""Recall has to treat two records the adjudicator merged as the one entity it judged them to be."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from psycopg import AsyncConnection

from mindbridge.application.ports import EmbeddingMatch
from mindbridge.contracts import RecallQuery, RecallRequest
from mindbridge.core import EmbeddedObjectType, TenantId
from mindbridge.infrastructure.postgres import PostgresMemoryStore

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
# Ascending, because entity_resolution stores one row per pair in `left < right` order only.
PEOPLE = ("entity_a", "entity_b", "entity_c", "entity_d")


async def _seed(
    database_url: str, tenant_id: str, *, verdicts: tuple[tuple[str, str, str], ...]
) -> None:
    """One person, one event, one memory each, joined by the `relations` rows recall reads.

    Written straight through SQL for the same reason `test_postgres_entity_resolution` does: the
    interesting state is a committed `same_as` verdict, and the sweep that writes one has never
    run on any corpus, so no pipeline in this repository can produce the fixture.

    The memories are `unverified` so they need no evidence, and so this fixture needs no media
    object, observation, or evidence span. Recall never branches on verification status; every
    filter it applies is in `_STRUCTURED_RECALL_FILTER_SQL` and none of them read that column.
    """
    connection = await AsyncConnection.connect(database_url, autocommit=True)
    async with connection:
        for index, entity_id in enumerate(PEOPLE):
            occurred_at = NOW + timedelta(minutes=index)
            event_id, memory_id = f"event_{entity_id}", f"memory_{entity_id}"
            await connection.execute(
                "INSERT INTO entities (tenant_id, entity_id, entity_type, canonical_name,"
                " created_at) VALUES (%s, %s, 'person', %s, %s)",
                (tenant_id, entity_id, f"person seen as {entity_id}", NOW),
            )
            await connection.execute(
                "INSERT INTO events (tenant_id, event_id, hierarchy_level, description,"
                " salience, status, occurred_at, ended_at, model_id, prompt_version,"
                " content_digest, created_at)"
                " VALUES (%s, %s, 'event', %s, 0.5, 'active', %s, %s, 'm', 'p', %s, %s)",
                (
                    tenant_id,
                    event_id,
                    f"an event mentioning {entity_id}",
                    occurred_at,
                    occurred_at + timedelta(seconds=30),
                    f"{index:064x}",
                    NOW,
                ),
            )
            await connection.execute(
                "INSERT INTO memory_records (tenant_id, memory_id, memory_type, summary,"
                " verification_status, state, occurred_at, ended_at, content_digest,"
                " created_at, lifecycle_changed_at)"
                " VALUES (%s, %s, 'episodic', %s, 'unverified', 'active', %s, %s, %s, %s, %s)",
                (
                    tenant_id,
                    memory_id,
                    f"an event mentioning {entity_id}",
                    occurred_at,
                    occurred_at + timedelta(seconds=30),
                    f"{index:064x}",
                    NOW,
                    NOW,
                ),
            )
            await _relate(connection, tenant_id, "event", event_id, "mentions", "entity", entity_id)
            await _relate(
                connection,
                tenant_id,
                "event",
                event_id,
                "represented_by",
                "memory_record",
                memory_id,
            )
        for source_id, relation_type, target_id in verdicts:
            await _relate(
                connection, tenant_id, "entity", source_id, relation_type, "entity", target_id
            )


async def _relate(
    connection: AsyncConnection,
    tenant_id: str,
    source_type: str,
    source_id: str,
    relation_type: str,
    target_type: str,
    target_id: str,
) -> None:
    await connection.execute(
        "INSERT INTO relations (tenant_id, relation_id, source_type, source_id, relation_type,"
        " target_type, target_id, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (
            tenant_id,
            f"relation_{source_id}_{relation_type}_{target_id}",
            source_type,
            source_id,
            relation_type,
            target_type,
            target_id,
            NOW,
        ),
    )


async def _reached(
    store: PostgresMemoryStore,
    tenant_id: str,
    object_type: EmbeddedObjectType,
    object_id: str,
) -> set[str]:
    memories = await store.search_memories_by_graph_objects(
        RecallRequest(tenant_id=TenantId(tenant_id), query=RecallQuery(text="who was that")),
        (
            EmbeddingMatch(
                embedding_id="hit",
                object_type=object_type,
                object_id=object_id,
                similarity=1.0,
            ),
        ),
        limit=20,
    )
    reached = {str(memory.memory_id) for memory in memories}
    assert reached, "the hit reached nothing at all, so nothing below is being measured"
    return reached


async def test_an_entity_hit_reaches_the_entity_it_was_judged_to_be(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """Both stored directions, because the edge is only ever written with the smaller id as source.

    Following `source_id` alone would resolve the merge for `entity_a` and silently not for
    `entity_b`, which reads as an intermittent feature rather than a missing one.
    """
    tenant_id = "tenant_same_as_entity_hit"
    await _seed(database_url, tenant_id, verdicts=(("entity_a", "same_as", "entity_b"),))

    forward = await _reached(store, tenant_id, EmbeddedObjectType.ENTITY, "entity_a")
    backward = await _reached(store, tenant_id, EmbeddedObjectType.ENTITY, "entity_b")

    assert forward >= {"memory_entity_a", "memory_entity_b"}
    assert backward >= {"memory_entity_a", "memory_entity_b"}
    assert "memory_entity_c" not in forward | backward


async def test_an_event_hit_expands_through_the_merge_its_entity_carries(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """The co-mention hop is the half a name query never uses, and it has to agree with the other."""
    tenant_id = "tenant_same_as_event_hit"
    await _seed(database_url, tenant_id, verdicts=(("entity_a", "same_as", "entity_b"),))

    reached = await _reached(store, tenant_id, EmbeddedObjectType.EVENT, "event_entity_a")

    assert reached >= {"memory_entity_a", "memory_entity_b"}
    assert "memory_entity_c" not in reached


async def test_a_merge_is_never_composed_with_the_next_one(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """A~B and B~C are two verdicts, not a cluster; a join must not close what the judge refused.

    A released clustering that did compose them put 152 of 153 observations on one character. B
    reaching both A and C is not composition -- those are the two verdicts B itself carries.
    """
    tenant_id = "tenant_same_as_no_closure"
    await _seed(
        database_url,
        tenant_id,
        verdicts=(("entity_a", "same_as", "entity_b"), ("entity_b", "same_as", "entity_c")),
    )

    from_a = await _reached(store, tenant_id, EmbeddedObjectType.ENTITY, "entity_a")
    from_b = await _reached(store, tenant_id, EmbeddedObjectType.ENTITY, "entity_b")

    assert "memory_entity_c" not in from_a
    assert from_b >= {"memory_entity_a", "memory_entity_b", "memory_entity_c"}


async def test_a_pair_judged_different_stays_two_entities(
    store: PostgresMemoryStore, database_url: str
) -> None:
    """`not_same_as` records that a pair was inspected and differs; it is not a traversal."""
    tenant_id = "tenant_same_as_negative"
    await _seed(database_url, tenant_id, verdicts=(("entity_a", "not_same_as", "entity_d"),))

    reached = await _reached(store, tenant_id, EmbeddedObjectType.ENTITY, "entity_a")

    assert reached == {"memory_entity_a"}
